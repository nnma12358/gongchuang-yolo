#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
抓取与定位解算（视觉 → 机械臂基坐标）
=====================================================================
对应《智能视觉分拣项目阶段性进展与后续实施方案》阶段 4~6：
  阶段4 机械臂坐标标定  → solve_homography / solve_rigid（本文件）
  阶段5 单件自动抓取    → grasp_from_detection（抓取点、偏航、夹爪目标、预抓高度）
  阶段6 分类投放        → bin_place_pose（六盒投放坐标与释放高度）

两条换算路径（现场二选一或互为校验）：
  A) 平面单应（2D，推荐先用）：托盘上取 ≥4 个标定点（像素 ←→ 机械臂 XY），
     求解 H：pixel → robot_xy。适合“货物都平放在托盘同一平面”的场景，误差 1~2mm。
  B) 2.5D 深度（更准）：由深度图得到目标表面高度 Z，结合相机内参得到相机坐标，
     再用 ≥3 个非共线点的刚体变换 (R,t) 转到机械臂基坐标：
        X_cam = (u - cx) * Z / fx,  Y_cam = (v - cy) * Z / fy,  Z_cam = Z
        P_base = R @ P_cam + t
     适合目标高度不一（不同模型高度差大）的场景。

夹爪：joint6 现场标定为 0.20(闭合) ~ 1.20(张开)，见 GRIPPER 配置。
      目标宽度 → joint6 目标值按线性映射并夹紧，实机需按实测微调标定表。
"""
import json
import math
import os

import numpy as np

# ---------------- 现场机械参数（按实机标定结果修改） ----------------
GRIPPER = {
    "joint_min": 0.20,      # 完全闭合（实机标定）
    "joint_max": 1.20,      # 完全张开（实机标定）
    "width_min_mm": 8.0,    # 对应 joint_min 的夹爪开口
    "width_max_mm": 46.0,   # 对应 joint_max 的夹爪开口（略大于货物 40mm）
    "margin_mm": 4.0,       # 抓取时开口留量（防止夹不紧/顶飞）
}

# 垂直顶抓高度（相对托盘面，mm）
PICK = {
    "approach_mm": 90.0,     # 目标上方安全高度（先到此高度再垂直下降）
    "grasp_ratio": 0.5,      # 抓取高度 = 托盘面 + 目标高度 × 该比例（夹爪夹在货物中段）
    "grasp_min_mm": 6.0,     # 抓取高度下限（防止贴托盘刮擦）
    "grasp_max_mm": 22.0,    # 抓取高度上限（受指长限制）
    "assume_height_mm": 25.0,# 无深度信息时的目标高度假设值
    "lift_mm": 80.0,         # 抓取后抬升高度（相对目标顶面）
    "release_mm": 25.0,      # 盒内释放高度（保证货物完全进入盒内）
    "home_mm": 120.0,        # 回安全位高度
}

# 机械臂可达范围（基座为中心的半径，mm；按 R550A 实机改）
WORKSPACE = {"r_min": 30.0, "r_max": 350.0}   # 按 R550A 实机可达范围标定


# ==================== A) 平面单应：像素 → 机械臂 XY ====================
def solve_homography(pixels, robots):
    """由 ≥4 组 (像素, 机械臂XY) 对应点求解单应矩阵 H（像素 → 基坐标 XY，mm）"""
    import cv2
    P = np.asarray(pixels, float).reshape(-1, 1, 2)
    R = np.asarray(robots, float).reshape(-1, 1, 2)
    if len(P) < 4:
        raise ValueError("至少需要 4 个标定点（建议 6~9 个，覆盖托盘四角与中心）")
    H, mask = cv2.findHomography(P, R, method=0)
    if H is None:
        raise ValueError("单应矩阵求解失败：请检查标定点是否共线/重复")
    return H


def pixel_to_robot(u, v, H):
    """像素 (u,v) → 托盘平面上的机械臂基坐标 (x,y) mm"""
    p = np.array([u, v, 1.0], float)
    q = np.asarray(H, float) @ p
    if abs(q[2]) < 1e-9:
        raise ValueError("单应变换奇异")
    return float(q[0] / q[2]), float(q[1] / q[2])


def homography_error(H, pixels, robots):
    """返回 (RMS 误差 mm, 每点误差列表) —— 现场验收建议 RMS < 2mm"""
    errs = []
    for (u, v), (x, y) in zip(pixels, robots):
        px, py = pixel_to_robot(u, v, H)
        errs.append(math.hypot(px - x, py - y))
    rms = math.sqrt(sum(e * e for e in errs) / max(1, len(errs)))
    return rms, errs


# ==================== B) 2.5D 深度：相机 → 机械臂基坐标 ====================
def camera_matrix_from_fov(width, height, hfov_deg):
    """由视场角得到内参 (fx, fy, cx, cy)"""
    fx = (width / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return {"fx": fx, "fy": fx, "cx": width / 2.0, "cy": height / 2.0,
            "width": width, "height": height, "hfov_deg": hfov_deg}


def pixel_depth_to_camera(u, v, z_mm, K):
    """像素 + 深度 → 相机坐标系三维点（mm，OpenCV 约定：+Z 沿光轴向前）"""
    return np.array([(u - K["cx"]) * z_mm / K["fx"],
                     (v - K["cy"]) * z_mm / K["fy"],
                     z_mm], float)


def solve_rigid(cam_pts, robot_pts):
    """≥3 个非共线点求刚体变换 (R,t)：相机系 → 机械臂基坐标系（Kabsch/SVD）

    注意：若所有点共面（例如都取在托盘平面 z=0），只能确定平面内映射 —— 此时请优先用
    平面单应（路径A），或让标定点分布在≥2 个已知高度上（例如垫 20mm 标定块）。
    """
    A = np.asarray(cam_pts, float)
    B = np.asarray(robot_pts, float)
    if A.ndim != 2 or B.ndim != 2:
        raise ValueError("点集必须是二维数组")
    if A.shape[1] == 2:                       # 允许传入 2D 相机点（补 z）
        A = np.hstack([A, np.zeros((len(A), 1))])
    if B.shape[1] == 2:                       # 机械臂点补 z=0（平面标定点）
        B = np.hstack([B, np.zeros((len(B), 1))])
    if len(A) < 3:
        raise ValueError("至少需要 3 个非共线点")
    ca, cb = A.mean(0), B.mean(0)
    H = (A - ca).T @ (B - cb)
    U, S, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    D = np.diag([1.0, 1.0, d])
    R = Vt.T @ D @ U.T
    t = cb - R @ ca
    return R, t


def coplanar_ratio(points, tol=1e-6):
    """判断点集是否共面（用于提示 2.5D 标定是否充分）"""
    A = np.asarray(points, float)
    if len(A) < 4:
        return True
    c = A.mean(0)
    _, s, _ = np.linalg.svd(A - c)
    return bool(s[2] <= max(tol, abs(s[0]) * 1e-6))


def rigid_error(R, t, cam_pts, robot_pts):
    errs = [float(np.linalg.norm(R @ np.asarray(p, float) + t - np.asarray(q, float)))
            for p, q in zip(cam_pts, robot_pts)]
    rms = math.sqrt(sum(e * e for e in errs) / max(1, len(errs)))
    return rms, errs


# ==================== 夹爪与抓取位姿 ====================
def gripper_joint_for_width(width_mm, cfg=None):
    """目标宽度 (mm) → joint6 目标值（线性映射并夹紧到实机安全范围）"""
    cfg = cfg or GRIPPER
    w = max(cfg["width_min_mm"], min(cfg["width_max_mm"], float(width_mm)))
    ratio = (w - cfg["width_min_mm"]) / max(1e-6, cfg["width_max_mm"] - cfg["width_min_mm"])
    return round(cfg["joint_min"] + ratio * (cfg["joint_max"] - cfg["joint_min"]), 3)


def grasp_from_detection(det, H=None, K=None, R=None, t=None, plane_z_mm=0.0,
                         gripper=None, pick=None):
    """
    由一条识别标记（det）生成抓取参数：
      det: {box:[x1,y1,x2,y2], box_norm:[...], shape, color, marks, z_mm?, mask?}
      H:   平面单应（路径 A）；K/R/t: 相机内参与外参（路径 B，需 det["z_mm"]）
    返回: 抓取点(基坐标 mm)、偏航角、夹爪目标、各段高度、可达性
    """
    pick_cfg = dict(PICK, **(pick or {}))
    gripper_cfg = dict(GRIPPER, **(gripper or {}))
    x1, y1, x2, y2 = det["box"]
    cu, cv_ = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w_px, h_px = max(1.0, x2 - x1), max(1.0, y2 - y1)

    # 目标宽度（mm）：优先用深度换算，否则用标定比例（像素→mm，由单应尺度得到）
    width_mm = None
    height_mm = None
    if det.get("z_mm") and K:
        z = float(det["z_mm"])
        mm_per_px = z / K["fx"]                       # 该深度处每像素对应的毫米数
        width_mm = min(w_px, h_px) * mm_per_px
        height_mm = float(det.get("height_mm") or 0.0)
        p_cam = pixel_depth_to_camera(cu, cv_, z, K)
        if R is not None and t is not None:
            grasp_xy = (np.asarray(R, float) @ p_cam + np.asarray(t, float))[:2]
        elif H is not None:
            grasp_xy = np.array(pixel_to_robot(cu, cv_, H))
        else:
            raise ValueError("需要 H（平面单应）或 (K, R, t)（2.5D 路径）之一")
    else:
        if H is None:
            raise ValueError("缺少 2.5D 深度时必须有平面单应 H")
        grasp_xy = np.array(pixel_to_robot(cu, cv_, H))
        # 用单应估计的局部尺度折算目标宽度（近似：托盘对角 160mm 对应标定范围）
        mm_per_px = estimate_mm_per_px(H, cu, cv_)
        width_mm = min(w_px, h_px) * mm_per_px

    # 夹爪开口 = 目标最小边 + 留量
    gripper_open = float(width_mm) + gripper_cfg["margin_mm"]
    joint6 = gripper_joint_for_width(gripper_open, gripper_cfg)

    # 偏航角：抓取五棱柱/长方体/立方体时夹爪需对准最小边方向
    yaw_deg = 0.0
    if det.get("angle_deg") is not None:
        yaw_deg = float(det["angle_deg"])
    elif w_px >= h_px:
        yaw_deg = 0.0
    else:
        yaw_deg = 90.0

    # 抓取高度：夹在货物中段（受指长限制），无深度时按假设高度
    eff_h = float(height_mm) if height_mm else pick_cfg["assume_height_mm"]
    grasp_z = plane_z_mm + max(pick_cfg["grasp_min_mm"],
                               min(eff_h * pick_cfg["grasp_ratio"], pick_cfg["grasp_max_mm"]))
    top_z = plane_z_mm + eff_h
    pose = {
        "grasp_xy": [round(float(grasp_xy[0]), 2), round(float(grasp_xy[1]), 2)],
        "yaw_deg": round(yaw_deg, 2),
        "target_width_mm": round(float(width_mm), 2),
        "target_height_mm": round(float(height_mm or 0.0), 2),
        "height_assumed": height_mm is None,
        "gripper_open_mm": round(gripper_open, 2),
        "joint6_target": joint6,
        "z_approach": round(top_z + pick_cfg["approach_mm"], 2),
        "z_grasp": round(grasp_z, 2),
        "z_lift": round(top_z + pick_cfg["lift_mm"], 2),
        "shape": det.get("shape"), "color": det.get("color"),
        "reachable": is_reachable(grasp_xy),
    }
    return pose


def estimate_mm_per_px(H, u, v, probe_px=10.0):
    """由单应局部差分估计该像素位置的 mm/px（无深度时用于估算目标尺寸）"""
    x0, y0 = pixel_to_robot(u, v, H)
    x1, y1 = pixel_to_robot(u + probe_px, v, H)
    x2, y2 = pixel_to_robot(u, v + probe_px, H)
    sx = math.hypot(x1 - x0, y1 - y0) / probe_px
    sy = math.hypot(x2 - x0, y2 - y0) / probe_px
    return (sx + sy) / 2.0


def is_reachable(xy, workspace=None):
    ws = workspace or WORKSPACE
    r = math.hypot(float(xy[0]), float(xy[1]))
    return bool(ws["r_min"] <= r <= ws["r_max"])


# ==================== 六储物盒投放 ====================
def bin_place_pose(bin_no, bins_calib, pick=None):
    """
    储物盒投放位姿：bins_calib = {"1": {"x":.., "y":.., "z":..}, ...}
    z 缺省时按 PICK.release_mm 相对托盘面
    """
    pick_cfg = dict(PICK, **(pick or {}))
    key = str(bin_no)
    if key not in bins_calib:
        return None
    b = bins_calib[key]
    return {
        "bin": int(bin_no),
        "place_xy": [round(float(b.get("x", 0.0)), 2), round(float(b.get("y", 0.0)), 2)],
        "z_approach": round(float(b.get("z", 0.0)) + pick_cfg["approach_mm"] * 0.5, 2),
        "z_release": round(float(b.get("z", 0.0)) + pick_cfg["release_mm"], 2),
        "reachable": is_reachable([b.get("x", 0.0), b.get("y", 0.0)]),
    }


# ==================== 标定文件读写 ====================
DEFAULT_CALIB_PATH = os.path.join(os.environ.get("DATA_DIR", "data"), "calib.json")


def load_calib(path=None):
    """读取标定文件；不存在返回 None"""
    p = path or DEFAULT_CALIB_PATH
    if not os.path.exists(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_calib(calib, path=None):
    p = path or DEFAULT_CALIB_PATH
    os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(calib, f, ensure_ascii=False, indent=2)
    return p


def calib_ready(calib):
    """标定是否可用于抓取（阶段4验收门槛）"""
    if not calib:
        return False, "缺少标定文件 calib.json"
    if calib.get("homography") is None and calib.get("rigid") is None:
        return False, "标定文件缺少 homography / rigid"
    err = calib.get("homography_rms_mm") or calib.get("rigid_rms_mm")
    if err is not None and float(err) > 3.0:
        return False, "标定误差 {0:.2f}mm 偏大（建议 <2mm，最大容忍 3mm）".format(float(err))
    if not calib.get("bins"):
        return False, "缺少六个储物盒投放坐标（bins）"
    return True, "标定可用（误差 {0}mm）".format(err)
