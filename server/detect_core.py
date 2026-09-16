#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
识别核心（视觉容器与网关共用）—— 颜色 + 形状 + 污渍/缺陷 + 二维码/文字
输出“标记（marks）”：每件货物的框、类别、属性与置信度（JSON 结构，供网关与显示屏消费）。
可选 ONNX（自训练 YOLO）引擎：DETECT_ENGINE=onnx 且模型可加载时启用，属性仍由经典算法补充。
"""
import logging
import math
import os
import threading

from catalog import match_goods, goods_key  # noqa: F401  (goods_key 供调用方使用)

logger = logging.getLogger("detect")

COLOR_TABLE = [("红色", [(0, 8), (170, 180)]), ("橙色", [(9, 22)]), ("黄色", [(23, 34)]),
               ("绿色", [(35, 85)]), ("青色", [(86, 100)]), ("蓝色", [(101, 130)]),
               ("紫色", [(131, 160)])]

_NET = None
_NET_TRIED = False
_LOCK = threading.Lock()

# ---- 托盘 ROI：只接受"框中心"落在托盘区域内的检测 ----
# 为什么需要：模型层面的负样本能把背景误检压到 0，但现场总会出现训练照片里没有的东西
# （人手、机械臂、临时放的杂物）。货物一定在托盘上 → "区域"是最便宜也最可靠的兜底。
# 规格（归一化 0~1）：
#   矩形   "0.06,0.06,0.94,0.94"
#   多边形 "0.10,0.08;0.90,0.10;0.92,0.90;0.08,0.88"（按顺序闭合）
#   空/未设置 = 不启用
_ROI = None
LAST_ROI_DROPPED = 0


def set_roi(spec):
    """配置 ROI；返回解析后的多边形（未启用返回 None）"""
    global _ROI
    _ROI = None
    spec = (spec or "").strip()
    if not spec:
        return None
    try:
        if ";" in spec:
            pts = [[float(v) for v in p.split(",")[:2]] for p in spec.split(";") if p.strip()]
        else:
            x1, y1, x2, y2 = [float(v) for v in spec.split(",")[:4]]
            pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
        if len(pts) < 3:
            raise ValueError("至少 3 个点")
        _ROI = pts
        return pts
    except Exception as e:
        logger.warning("ROI 解析失败（已忽略）: %s  ← %s", spec, e)
        return None


def roi_enabled():
    return _ROI is not None


def _in_roi(cx, cy):
    """射线法判断点是否在多边形内；未启用 ROI 时恒为 True"""
    if _ROI is None:
        return True
    inside = False
    n = len(_ROI)
    for i in range(n):
        x1, y1 = _ROI[i]
        x2, y2 = _ROI[(i + 1) % n]
        if (y1 > cy) != (y2 > cy):
            xin = x1 + (cy - y1) * (x2 - x1) / (y2 - y1 + 1e-12)
            if cx < xin:
                inside = not inside
    return inside


def load_onnx(model_path):
    """可选：加载自训练 YOLO ONNX（失败返回 None，调用方回退经典引擎）"""
    global _NET, _NET_TRIED
    with _LOCK:
        if _NET_TRIED:
            return _NET
        _NET_TRIED = True
        if not model_path or not os.path.exists(str(model_path)):
            logger.info("未提供 ONNX 模型，使用颜色+形状+标记引擎")
            return None
        try:
            import cv2
            _NET = cv2.dnn.readNetFromONNX(str(model_path))
            logger.info("已加载 ONNX 检测模型 {0}".format(os.path.basename(str(model_path))))
        except Exception as e:
            logger.warning("ONNX 模型加载失败，回退经典引擎: {0}".format(e))
            _NET = None
        return _NET


def hue_name(h, s, v):
    if v < 55:
        return "黑色"
    if s < 45:
        return "白色" if v > 190 else "灰色"
    for name, ranges in COLOR_TABLE:
        for lo, hi in ranges:
            if lo <= h <= hi:
                return name
    return "红色"


def vertices_to_shape(n, circularity):
    """按顶点数 + 圆度判定形状（俯视相机为主；斜视/球柱类建议用属性分类器复核）"""
    if circularity > 0.86:
        return "球" if n > 8 else "圆柱"
    return {3: "正四面体", 4: "正方体", 5: "五棱柱", 6: "六棱柱"}.get(n, "{0}边形".format(n))


def stable_vertices(contour, eps_list=(0.02, 0.03, 0.04, 0.05)):
    """
    多阈值扫描求稳定顶点数：立体货物的剪影在单一 eps 下顶点数会漂移，
    取各 eps 下的凸包近似顶点数众数，提升 正方体/五棱柱/六棱柱 的判定稳定性。
    """
    import cv2
    hull = cv2.convexHull(contour)
    peri = cv2.arcLength(hull, True)
    if peri <= 0:
        return 4
    counts = []
    for eps in eps_list:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        counts.append(len(approx))
    best = max(set(counts), key=counts.count)
    if best > 8:                       # 顶点过多 → 视为圆滑轮廓（球/圆柱）
        return best
    # 若相邻阈值给出更小的稳定值（3–6 之间），优先取该值（通常是顶面多边形）
    small = [c for c in counts if 3 <= c <= 6]
    if small:
        return max(set(small), key=small.count)
    return best


def decode_qr(bgr):
    """二维码/文字解码：返回 (载荷, 数量)"""
    try:
        import cv2
        ok, texts, _pts, _ = cv2.QRCodeDetector().detectAndDecodeMulti(bgr)
        if ok and texts:
            texts = [t for t in texts if t]
            if texts:
                return texts[0], len(texts)
    except Exception:
        pass
    try:
        import cv2
        data, _pts, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
        if data:
            return data, 1
    except Exception:
        pass
    return "", 0


def analyze(bgr, conf_min=0.45, min_area_ratio=0.002, stain_th=0.05, defect_th=0.14,
            engine_cfg=None):
    """
    对单帧图像做识别，返回 (detections, qr_text, image_size)
    detection = {
      box: [x1,y1,x2,y2] 像素, box_norm: [x,y,w,h] 归一化,
      class/name/shape/color/marks/text/qr/conf/area_ratio
    }
    """
    import cv2
    import numpy as np

    h0, w0 = bgr.shape[:2]
    qr_text, qr_count = decode_qr(bgr)

    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = float(min_area_ratio) * w0 * h0
    detections = []

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        circularity = 4 * math.pi * area / (peri * peri) if peri > 0 else 0
        n_vertices = stable_vertices(c) if circularity <= 0.86 else len(approx)
        shape = vertices_to_shape(n_vertices, circularity)

        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [c], -1, 255, -1)
        pixels = hsv[blob == 255]
        med_h, med_s, med_v = np.median(pixels, axis=0) if len(pixels) else (0, 0, 0)
        color = hue_name(float(med_h), float(med_s), float(med_v))

        inside = gray[blob == 255]
        dark_ratio = float(np.mean(inside < max(60.0, np.median(inside) * 0.72))) if len(inside) else 0.0
        marks = ["缺陷"] if dark_ratio > defect_th else (["污渍"] if dark_ratio > stain_th else ["无"])

        matched = match_goods(shape, color)
        x, y, w, h = cv2.boundingRect(c)
        area_ratio = area / float(w0 * h0)
        conf = round(min(0.99, 0.55 + 0.4 * min(1.0, area_ratio / 0.05)), 3)
        if conf < conf_min:
            continue
        detections.append({
            "class": (matched or {}).get("id"),
            "name": (matched or {}).get("name") or "{0}{1}".format(color, shape),
            "label": "{0}{1}".format(color, shape),
            "shape": shape,
            "color": color,
            "marks": marks,
            "dark_ratio": round(dark_ratio, 4),
            "circularity": round(circularity, 3),
            "vertices": int(n_vertices),
            "qr": qr_text if len(qr_text) else "",
            "text": qr_text or "",
            "conf": conf,
            "box": [int(x), int(y), int(x + w), int(y + h)],
            "box_norm": [round(x / w0, 4), round(y / h0, 4), round(w / w0, 4), round(h / h0, 4)],
            "area_ratio": round(area_ratio, 4),
        })

    detections.sort(key=lambda d: d["area_ratio"], reverse=True)
    detections = filter_roi(detections)
    return detections, qr_text, (w0, h0)


def filter_roi(dets):
    """按托盘 ROI 过滤检测（框中心落在区域外 → 丢弃）。

    ⚠ 所有检测来源都必须走这里：经典引擎走 analyze()，而 yolo 容器路径
    （vision_server.detect_via_services）是自己拼 dets 的，容易漏掉。
    """
    global LAST_ROI_DROPPED
    LAST_ROI_DROPPED = 0
    if _ROI is None:
        return dets
    kept = []
    for d in dets:
        bn = d.get("box_norm")
        if not bn or len(bn) < 4:
            kept.append(d)
            continue
        cx, cy = bn[0] + bn[2] / 2.0, bn[1] + bn[3] / 2.0
        if _in_roi(cx, cy):
            kept.append(d)
        else:
            LAST_ROI_DROPPED += 1
    if LAST_ROI_DROPPED:
        logger.debug("ROI 过滤掉 %d 个区域外检测", LAST_ROI_DROPPED)
    return kept


def attach_depth(detections, depth, desk_z=None, inner=0.5, table_ref=None):
    """把深度信息附加到标记上（2.5D）：
       depth: 与 RGB 对齐的深度图（uint16，单位 mm；0 表示无效）
       desk_z: 托盘平面深度；None 时用全图有效深度的中位数（托盘占画面大部分）
       为每条检测取框内中心区域深度的中位数 → z_mm（到相机距离）与 height_mm（高出托盘）
    """
    import numpy as np
    if depth is None:
        # 无当前深度帧时，若已加载逐像素桌面参考，仍可给出桌面基准（height 为 None）
        if table_ref is not None:
            for det in detections:
                x1, y1, x2, y2 = det.get("box", [0, 0, 0, 0])
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                z = _table_ref_depth(cx, cy, table_ref)
                if z:
                    det["desk_z_mm"] = round(float(z), 1)
                    det["depth_source"] = "table_reference_only"
        return detections
    d = np.asarray(depth)
    valid = d[(d > 0)]
    if valid.size == 0:
        return detections
    if desk_z is None:
        desk_z = float(np.median(valid))

    def _ref_depth(cx, cy):
        if table_ref is None:
            return None
        return _table_ref_depth(cx, cy, table_ref)
    H, W = d.shape[:2]
    for det in detections:
        x1, y1, x2, y2 = det.get("box", [0, 0, 0, 0])
        w, h = max(1, x2 - x1), max(1, y2 - y1)
        ix1 = int(max(0, x1 + w * (1 - inner) / 2)); ix2 = int(min(W, x2 - w * (1 - inner) / 2))
        iy1 = int(max(0, y1 + h * (1 - inner) / 2)); iy2 = int(min(H, y2 - h * (1 - inner) / 2))
        patch = d[iy1:max(iy1 + 1, iy2), ix1:max(ix1 + 1, ix2)]
        vals = patch[(patch > 50) & (patch < 2000)]
        ref_z0 = _ref_depth((x1 + x2) / 2.0, (y1 + y2) / 2.0) or desk_z
        det["desk_z_mm"] = round(float(ref_z0), 1)
        if vals.size == 0:
            # 深度空洞（黑色/白色货物常见）：位置可信、高度未知 → 由抓取侧按假设高度处理
            det["z_mm"] = None
            det["height_mm"] = None
            det["depth_source"] = "hole"
            continue
        z = float(np.median(vals))
        # 优先用该像素的桌面参考（比全图中位数更准，来自 2.5D 定位软件的逐像素标定）
        ref_z = _ref_depth((x1 + x2) / 2.0, (y1 + y2) / 2.0) or desk_z
        det["z_mm"] = round(z, 1)
        det["height_mm"] = round(max(0.0, ref_z - z), 1)      # 顶面比桌面更靠近相机 → 高度为正
        det["desk_z_mm"] = round(float(ref_z), 1)
        det["depth_source"] = "table_reference" if ref_z != desk_z else "frame_median"
    return detections


def _table_ref_depth(u, v, table_ref, radius=3):
    """逐像素桌面参考深度（中位数，抗噪）—— 由 table_reference.npz 载入"""
    import numpy as np
    table = np.asarray(table_ref.get("table_depth_mm"))
    if table is None or table.size == 0:
        return None
    mask = table_ref.get("valid_mask")
    H, W = table.shape[:2]
    x, y = int(round(u)), int(round(v))
    y1, y2 = max(0, y - radius), min(H, y + radius + 1)
    x1, x2 = max(0, x - radius), min(W, x + radius + 1)
    patch = table[y1:y2, x1:x2]
    valid = (patch > 50) & (patch < 2000)
    if mask is not None:
        valid &= np.asarray(mask)[y1:y2, x1:x2].astype(bool)
    vals = patch[valid]
    return float(np.median(vals)) if vals.size else None


def draw_marks(bgr, detections, hud=None, thickness=None):
    """把标记画到画面上（输出带标记的图像；hud 为右上角信息行）"""
    import cv2
    out = bgr.copy()
    h, w = out.shape[:2]
    th = thickness or max(2, int(w / 420))
    for d in detections:
        x1, y1, x2, y2 = d["box"]
        picked = (d.get("marks") or ["无"])[0]
        color = (0, 200, 255) if picked == "无" else (0, 120, 255) if picked == "污渍" else (0, 0, 255)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, th)
        label = "{0} {1}%".format(d["name"], int(d["conf"] * 100))
        if picked != "无":
            label += " [{0}]".format(picked)
        fs = max(0.5, w / 1400.0)
        (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 2)
        cv2.rectangle(out, (x1, max(0, y1 - tht - 10)), (x1 + tw + 10, y1), color, -1)
        cv2.putText(out, label, (x1 + 5, max(tht + 4, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 2)
    if hud:
        for i, line in enumerate(hud):
            cv2.putText(out, line, (10, 26 + i * 26), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 90, 200), 2)
    return out


def encode_jpeg(bgr, quality=80):
    import cv2
    ok, buf = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return buf.tobytes() if ok else b""
