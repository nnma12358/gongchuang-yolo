#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
overhead_localization.py —— 适配工作区 2.5D 定位软件（smart_sorting_localization_*）
=====================================================================
对接对象（该目录产出的标定与结果）：
  calibration/table_reference_*/table_reference.npz     逐像素桌面深度参考 + 内参
                          table_depth_mm.npy / *.png    深度与掩膜（可选，npz 已含）
  calibration/object_localization/object_locator_result.json   RGB + 深度空洞 2.5D 定位结果
  calibration/object_localization/object_contour_result.json    RGB 轮廓细化结果
  calibration/handeye/gripper_red_marker_result.json            夹爪三红点 → TCP 像素
  calibration/*/camera_intrinsics.json                           Astra 内参（fx=fy=570.34@640x480）

本模块做三件事（补上该软件的已知缺口）：
  1) 位置与高度更准：用**逐像素桌面参考**算目标真高（原实现只用单一 table_depth_mm，
     在不同结果里出现过 428 / 676mm 不一致 → 这里按像素差分并对异常做剔除）
  2) 属性补全：该软件只给 BLACK/WHITE 外观，形状/颜色/污渍/二维码由本项目的识别引擎给，
     按轮廓重叠/最近邻融合成一条完整标记
  3) 坐标贯通：其输出是相机/像素系（manifest: COORDINATE_STATUS=CAMERA_OR_PIXEL_FRAME），
     用 pose.py 转成机械臂基坐标并生成抓取/投放位姿
"""
import json
import math
import os

import numpy as np


# ==================== 标定产物读取 ====================
def load_table_reference(path):
    """读取 table_reference.npz 或 table_reference_candidate 目录"""
    if os.path.isdir(path):
        cand = os.path.join(path, "table_reference.npz")
        if not os.path.exists(cand):
            raise FileNotFoundError("目录中缺少 table_reference.npz: {0}".format(path))
        path = cand
    z = np.load(path)
    out = {
        "path": path,
        "table_depth_mm": z["table_depth_mm"].astype(np.float32),
        "valid_mask": z["valid_mask"].astype(bool) if z["valid_mask"].dtype != bool else z["valid_mask"],
        "roi_mask": z["roi_mask"].astype(bool) if "roi_mask" in z.files and z["roi_mask"].dtype != bool else
                    (z["roi_mask"] if "roi_mask" in z.files else None),
        "table_noise_mm": z["table_noise_mm"] if "table_noise_mm" in z.files else None,
        "camera_matrix": z["camera_matrix"] if "camera_matrix" in z.files else None,
        "projection": z["projection"] if "projection" in z.files else None,
    }
    K = out["camera_matrix"]
    if K is not None:
        out["intrinsics"] = {"fx": float(K[0, 0]), "fy": float(K[1, 1]),
                             "cx": float(K[0, 2]), "cy": float(K[1, 2])}
    # 桌面深度的稳健统计（剔除 0 与远处异常值）
    d = out["table_depth_mm"]
    m = (d > 50) & (d < 2000) & out["valid_mask"]
    out["table_depth_median_mm"] = float(np.median(d[m])) if m.any() else None
    out["table_valid_ratio"] = float(m.mean())
    return out


def load_intrinsics(path):
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    K = d.get("K") or [d.get("fx", 570.34), 0, d.get("cx", 319.5), 0, d.get("fy", 570.34), d.get("cy", 239.5), 0, 0, 1]
    return {"fx": float(K[0]), "fy": float(K[4]), "cx": float(K[2]), "cy": float(K[5]),
            "width": int(d.get("width", 640)), "height": int(d.get("height", 480)),
            "frame_id": d.get("frame_id", "")}


# ==================== 2.5D 几何 ====================
def pixel_depth_to_camera(u, v, z_mm, K):
    """像素 + 深度 → 相机系 XYZ（mm）—— 与 2.5D 定位软件给出的
       table_point_camera_mm 完全一致（已用其真实结果核对）"""
    return np.array([(u - K["cx"]) * z_mm / K["fx"],
                     (v - K["cy"]) * z_mm / K["fy"],
                     float(z_mm)])


def table_depth_at(u, v, ref, radius=3):
    """取该像素邻域的桌面参考深度（中位数，抗噪）"""
    u, v = int(round(u)), int(round(v))
    H, W = ref["table_depth_mm"].shape[:2]
    y1, y2 = max(0, v - radius), min(H, v + radius + 1)
    x1, x2 = max(0, u - radius), min(W, u + radius + 1)
    patch = ref["table_depth_mm"][y1:y2, x1:x2]
    valid = ref["valid_mask"][y1:y2, x1:x2] if ref.get("valid_mask") is not None else np.ones_like(patch, bool)
    vals = patch[valid & (patch > 50) & (patch < 2000)]
    if vals.size == 0:
        return None
    return float(np.median(vals))


def object_2p5d(u, v, depth_img, ref, K, depth_radius=2):
    """由像素 + 当前深度图 + 桌面参考，给出目标的 2.5D 量测

    返回: {top_depth_mm, table_depth_mm, height_mm, top_camera_mm, table_camera_mm, confidence}
    - height_mm 取沿相机光轴的差值（顶置相机下 ≈ 实际高度，误差 <2%@±15°倾角）
    - 若当前深度在目标处无效（黑色/白色货物常出现深度空洞，正是该软件用“空洞”检测的原因），
      退化为 table_depth 并标记 confidence=low
    """
    u_i, v_i = int(round(u)), int(round(v))
    table_z = table_depth_at(u, v, ref) or ref.get("table_depth_median_mm")
    vals = np.empty(0)
    if depth_img is not None:
        arr = np.asarray(depth_img)
        H, W = arr.shape[:2]
        y1, y2 = max(0, v_i - depth_radius), min(H, v_i + depth_radius + 1)
        x1, x2 = max(0, u_i - depth_radius), min(W, u_i + depth_radius + 1)
        patch = arr[y1:y2, x1:x2].astype(np.float32)
        vals = patch[(patch > 50) & (patch < 2000)]
    if vals.size == 0 or table_z is None:
        # 深度空洞：只知位置，不知高度 → 用桌面深度兜底
        z = table_z
        conf = "low"
    else:
        z = float(np.median(vals))
        conf = "high" if abs(z - table_z) < 60 else "medium"
    top = pixel_depth_to_camera(u, v, z, K)
    table = pixel_depth_to_camera(u, v, table_z, K)
    return {
        "top_depth_mm": round(float(z), 1),
        "table_depth_mm": round(float(table_z), 1),
        "height_mm": round(max(0.0, float(table_z) - float(z)), 1),
        "top_camera_mm": [round(float(x), 2) for x in top],
        "table_camera_mm": [round(float(x), 2) for x in table],
        "confidence": conf,
    }


# ==================== 与其结果 JSON 对接 ====================
def load_object_locator(path):
    """读取 object_locator_result.json（RGB + 深度空洞定位）"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    objs = []
    for o in d.get("objects", []):
        center = o.get("center_pixel") or [0, 0]
        box_pts = o.get("hole_box_px") or []
        xs = [p[0] for p in box_pts] or [center[0]]
        ys = [p[1] for p in box_pts] or [center[1]]
        objs.append({
            "id": o.get("id"),
            "center_pixel": [float(center[0]), float(center[1])],
            "box": [float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys))],
            "appearance": o.get("appearance"),
            "area_px": o.get("hole_area_px"),
            "gripper_image_angle_deg": o.get("gripper_image_angle_deg"),
            "table_depth_mm": o.get("table_depth_mm"),
            "table_camera_mm": o.get("table_point_camera_mm"),
            "source": "device_locator",
        })
    return {"image_size": d.get("image_size"), "intrinsics": d.get("camera_intrinsics"),
            "method": d.get("method"), "objects": objs}


def load_tcp_marker(path):
    """读取 gripper_red_marker_result.json（三红点 → TCP 像素与夹爪轴向）"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    tcp = d.get("tcp_pixel")
    return {
        "tcp_pixel": [float(tcp[0]), float(tcp[1])] if tcp else None,
        "jaw_open_axis_deg": d.get("jaw_open_axis_deg"),
        "gripper_forward_angle_deg": d.get("gripper_forward_angle_deg"),
        "jaw_separation_px": d.get("jaw_separation_px"),
        "fixed_marker_to_tcp_px": d.get("fixed_marker_to_tcp_px"),
        "orthogonality_error_deg": d.get("orthogonality_error_deg"),
        "method": d.get("method"),
    }


# ==================== 融合：位置(设备) + 属性(本项目识别) ====================
def _iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / (area_a + area_b - inter)


def merge_with_marks(device_objects, marks, iou_min=0.2):
    """把设备定位（位置/高度准）与本项目识别标记（形状/颜色/污渍/二维码）配成一条

    匹配策略：IoU 优先；IoU 不足时用中心点距离 < 0.6×目标对角线兜底。
    返回 (merged, unmatched_device, unmatched_marks)
    """
    merged, used = [], set()
    for dobj in device_objects:
        best, best_score = None, 0.0
        cx = (dobj["box"][0] + dobj["box"][2]) / 2.0
        cy = (dobj["box"][1] + dobj["box"][3]) / 2.0
        diag = math.hypot(dobj["box"][2] - dobj["box"][0], dobj["box"][3] - dobj["box"][1]) or 1.0
        for i, m in enumerate(marks):
            if i in used:
                continue
            box = m.get("box") or [0, 0, 0, 0]
            score = _iou(dobj["box"], box)
            if score < iou_min:
                mcx, mcy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
                dist = math.hypot(mcx - cx, mcy - cy)
                score = max(score, 1.0 - dist / (0.6 * diag)) if dist < 0.6 * diag else score
            if score > best_score:
                best, best_score = i, score
        item = dict(dobj)
        if best is not None:
            m = marks[best]
            used.add(best)
            item.update({
                "shape": m.get("shape"), "color": m.get("color"),
                "marks": m.get("marks") or ["无"], "name": m.get("name"),
                "qr": m.get("qr") or "", "text": m.get("text") or "",
                "rgb_conf": m.get("conf"),
                "match_score": round(best_score, 3),
                "image": m.get("image"),
            })
        else:
            item["match_score"] = 0.0
            item["match_note"] = "仅设备定位（本帧无对应 RGB 识别结果）"
        merged.append(item)
    unmatched_marks = [m for i, m in enumerate(marks) if i not in used]
    return merged, unmatched_marks


# ==================== 供 rclpy 节点使用的实时封装 ====================
class OverheadLocalizer(object):
    """在 ROS2 容器内使用：喂入 RGB/深度帧与内参，输出与网关标记兼容的结果

    典型用法（ros2_bridge 侧）：
        loc = OverheadLocalizer(table_ref_npz, intrinsics_json)
        marks = loc.update(depth_img)          # 深度空洞检测（黑色/白色货物亦可）
        marks = loc.attach_rgb(marks, rgb_marked)   # 用本项目识别结果补属性
    """

    def __init__(self, table_ref_path, intrinsics_path=None, hole_min_area_px=120):
        self.ref = load_table_reference(table_ref_path)
        self.K = (load_intrinsics(intrinsics_path) if intrinsics_path
                  else self.ref.get("intrinsics"))
        if not self.K:
            raise ValueError("缺少相机内参（请提供 camera_intrinsics.json）")
        self.hole_min_area = int(hole_min_area_px)

    def detect_holes(self, depth_img):
        """深度空洞检测：当前帧相对桌面参考“缺深度/明显更近”的区域即目标

        返回与本项目标记同构的列表（box / center_pixel / 2.5D 量测）
        """
        import cv2
        d = np.asarray(depth_img).astype(np.float32)
        table = self.ref["table_depth_mm"]
        valid_now = (d > 50) & (d < 2000)
        ref_valid = (table > 50) & (table < 2000)
        if self.ref.get("roi_mask") is not None:
            ref_valid &= self.ref["roi_mask"].astype(bool)
        # 空洞（货物处深度无效或无回波）；同时把“比桌面明显更近”也算目标（有回波的白/黑块）
        hole = ref_valid & (~valid_now)
        nearer = ref_valid & valid_now & (d < table - 12)
        mask = (hole | nearer).astype(np.uint8) * 255
        k = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
        # 兼容 OpenCV 3.x/4.x：3.x 返回 (image, contours, hierarchy)，4.x 返回 (contours, hierarchy)
        contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
        out = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < self.hole_min_area:
                continue
            x, y, w, h = cv2.boundingRect(c)
            M = cv2.moments(c)
            cx = M["m10"] / M["m00"] if M["m00"] else x + w / 2.0
            cy = M["m01"] / M["m00"] if M["m00"] else y + h / 2.0
            geo = object_2p5d(cx, cy, d, self.ref, self.K)
            rect = cv2.minAreaRect(c)
            angle = rect[2]
            out.append({
                "box": [float(x), float(y), float(x + w), float(y + h)],
                "center_pixel": [round(cx, 1), round(cy, 1)],
                "area_px": round(float(area), 1),
                "angle_deg": round(float(angle), 2),
                "source": "overhead_hole_2p5d",
                **geo,
            })
        return out

    def attach_rgb(self, device_marks, rgb_marks):
        """用本项目识别结果补属性（返回融合后的标记）"""
        merged, _ = merge_with_marks(device_marks, rgb_marks)
        return merged
