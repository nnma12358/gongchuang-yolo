#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
synth_polyhedra.py —— 多面体合成训练集生成器（带阴影 / 远近 / 角度）
======================================================================
目标：生成**物理合理**的托盘俯视图，训练出的模型能直接用于现实场景。

与现实对齐的关键（sim-to-real，勿随意放宽）：
  1) 相机几何取自现场标定（config/camera.yaml：托盘上方高度、FOV、分辨率）→ 合成图的
     透视关系、目标像素尺寸与真实相机一致（40mm 货物在 1080p/400mm 高约 100px）
  2) 材质贴近现场：哑光塑料（低高光）、托盘为浅灰/白色、无花纹
  3) 光照贴近室内：环境光为主 + 柔和方向光，阴影半影（blur）随光源尺寸变化
  4) 货物平放在托盘上 → 倾斜角 ≤10°，不会出现现实中不可能的姿态
  5) 域随机化只覆盖现实存在的差异：颜色批次、曝光、白平衡、噪声、JPEG、轻微失焦、
     托盘色差、目标在托盘上的位置/朝向/远近
  6) 建议：合成数据训练后，用现场实拍 30~50 张做少量微调（见 README「sim-to-real」）

输出（与 config/sorting_competition.yaml 对齐）：
  <out>/images/{train,val}/*.jpg        训练/验证图
  <out>/labels/{train,val}/*.txt        YOLO 标注（单类 goods；bbox 来自真实投影轮廓，不含阴影）
  <out>/attributes/{color,shape,stain}/<class>/*.jpg   属性分类小图（从检测框裁剪）
  <out>/preview.jpg                     抽样拼图，便于肉眼核对
  <out>/dataset_stats.json              尺寸/遮挡/角度/颜色统计

用法：
  python3 tools/synth_polyhedra.py --out data/synth --n 800
  python3 tools/synth_polyhedra.py --out data/synth --n 40 --preview 12 --no-attributes
  python3 tools/synth_polyhedra.py --out data/synth --n 800 --per-image 2 --imgsz 1280
"""
import argparse
import json
import math
import os
import random

import cv2
import numpy as np

# ============================================================
# 现场几何（改这里即可对齐真实相机；单位：毫米）
# ============================================================
REAL = {
    "imgsz": (1280, 720),     # 顶置 Astra RGB 分辨率（现场 640x480 亦可，按需改）
    "cam_height": 320.0,      # 相机光心到托盘面高度（现场实测；320mm 时托盘约占画面 46%）
    "hfov_deg": 60.0,         # 水平视场角
    "pitch_deg": 90.0,        # 90=垂直向下
    "tray_mm": 160.0,         # 托盘 160×160mm（赛项下限）
    "goods_mm": 40.0,         # 货物尺寸 ≤40mm
}

SHAPES = ["正方体", "长方体", "圆柱", "球", "正四面体", "五棱柱", "六棱柱",
          "正十二面体", "圆锥"]
COLORS = {
    "红色": (36, 40, 205), "橙色": (40, 130, 235), "黄色": (45, 200, 240),
    "绿色": (95, 200, 110), "青色": (200, 215, 40), "蓝色": (215, 130, 60),
    "紫色": (185, 90, 150), "黑色": (52, 52, 56), "白色": (232, 232, 235),
}   # BGR，贴近常见哑光塑料件
STAIN_PROB, DEFECT_PROB = 0.22, 0.12


# ============================================================
# 1. 多面体网格（毫米，中心在 x=y=0，底面 z=0）
# ============================================================
def mesh_cube(a=40.0, b=None, c=40.0):
    b = b or a
    v = np.array([[-a / 2, -b / 2, 0], [a / 2, -b / 2, 0], [a / 2, b / 2, 0], [-a / 2, b / 2, 0],
                  [-a / 2, -b / 2, c], [a / 2, -b / 2, c], [a / 2, b / 2, c], [-a / 2, b / 2, c]], float)
    f = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
    return v, f


def mesh_prism(n=5, r=22.0, h=40.0, phase=None):
    phase = 0.0 if phase is None else phase
    v, f = [], []
    for i in range(n):
        ang = phase + i * 2 * math.pi / n
        v.append([r * math.cos(ang), r * math.sin(ang), 0])
    for i in range(n):
        ang = phase + i * 2 * math.pi / n
        v.append([r * math.cos(ang), r * math.sin(ang), h])
    v = np.array(v, float)
    for i in range(n):
        j = (i + 1) % n
        f.append([i, j, n + j, n + i])
    f.append(list(range(n - 1, -1, -1)))
    f.append(list(range(n, 2 * n)))
    return v, f


def mesh_cylinder(r=20.0, h=40.0, n=36):
    return mesh_prism(n=n, r=r, h=h)


def mesh_sphere(r=20.0, rings=14, seg=28):
    v, f = [], []
    for i in range(rings + 1):
        phi = math.pi * i / rings
        for j in range(seg):
            th = 2 * math.pi * j / seg
            v.append([r * math.sin(phi) * math.cos(th), r * math.sin(phi) * math.sin(th), r - r * math.cos(phi)])
    for i in range(rings):
        for j in range(seg):
            a = i * seg + j
            b = i * seg + (j + 1) % seg
            c = (i + 1) * seg + (j + 1) % seg
            d = (i + 1) * seg + j
            f.append([a, b, c, d])
    return np.array(v, float), f


def mesh_tetra(edge=40.0):
    """正四面体：底面等边三角形，顶点在质心上方"""
    a = edge
    R = a / math.sqrt(3)                      # 底面外接圆半径
    h = a * math.sqrt(2.0 / 3.0)
    v = np.array([[R * math.cos(math.pi / 2 + k * 2 * math.pi / 3),
                   R * math.sin(math.pi / 2 + k * 2 * math.pi / 3), 0] for k in range(3)] + [[0, 0, h]], float)
    f = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]
    return v, f


def mesh_dodecahedron(r=22.0):
    """正十二面体：20 顶点 / 12 个正五边形面（现场实拍货物之一）"""
    phi = (1 + 5 ** 0.5) / 2.0
    base = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                base.append([sx, sy, sz])
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            base.append([0, s1 / phi, s2 * phi])
            base.append([s1 / phi, s2 * phi, 0])
            base.append([s1 * phi, 0, s2 / phi])
    V = np.array(base, float)
    V = V / np.linalg.norm(V, axis=1).max() * r            # 归一化到外接半径 r
    V[:, 2] -= V[:, 2].min()                               # 底面落在 z=0
    # 面心方向 = 二十面体顶点方向；每个面取最近的 5 个顶点并按角度排序
    ico = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            ico += [[0, s1, s2 * phi], [s1, s2 * phi, 0], [s1 * phi, 0, s2]]
    faces = []
    for c in np.array(ico, float):
        c = c / np.linalg.norm(c)
        d = V @ c
        idx = np.argsort(-d)[:5]
        centre = V[idx].mean(0)
        n = np.cross(V[idx[1]] - V[idx[0]], V[idx[2]] - V[idx[0]])
        if np.dot(n, centre) < 0:                          # 统一为外法线方向
            n = -n
        u = V[idx[0]] - centre
        u = u / np.linalg.norm(u)
        w = np.cross(c, u)
        ang = [math.atan2(np.dot(V[i] - centre, w), np.dot(V[i] - centre, u)) for i in idx]
        faces.append([int(idx[k]) for k in np.argsort(ang)])
    return V, faces


def mesh_cone(r=20.0, h=42.0, n=32):
    """圆锥：底面圆 + 顶点"""
    v = [[r * math.cos(2 * math.pi * i / n), r * math.sin(2 * math.pi * i / n), 0] for i in range(n)]
    v.append([0, 0, h])
    f = [[i, (i + 1) % n, n] for i in range(n)]
    f.append(list(range(n - 1, -1, -1)))
    return np.array(v, float), f


def build_mesh(shape, scale=1.0):
    s = scale
    if shape == "正方体":
        return mesh_cube(40 * s)
    if shape == "长方体":
        return mesh_cube(44 * s, 30 * s, 28 * s)
    if shape == "圆柱":
        return mesh_cylinder(20 * s, 40 * s)
    if shape == "球":
        return mesh_sphere(20 * s)
    if shape == "正四面体":
        return mesh_tetra(46 * s)
    if shape == "五棱柱":
        return mesh_prism(5, 24 * s, 40 * s, phase=math.pi / 2)
    if shape == "六棱柱":
        return mesh_prism(6, 23 * s, 40 * s, phase=math.pi / 2)
    if shape == "正十二面体":
        return mesh_dodecahedron(21.5 * s)
    if shape == "圆锥":
        return mesh_cone(19 * s, 40 * s)
    raise ValueError(shape)


# ============================================================
# 2. 相机与光照
# ============================================================
def camera_matrix(w, h, hfov_deg):
    fx = (w / 2.0) / math.tan(math.radians(hfov_deg) / 2.0)
    return fx, fx, w / 2.0, h / 2.0


def rot_matrix(pitch_deg, yaw_deg, roll_deg):
    """世界→相机旋转矩阵（OpenCV 约定：相机沿 +Z_cam 观察，z_cam>0 在相机前方）

    pitch=90° → 光轴垂直向下（现场顶置相机）；tilt = 90 - pitch 为相对竖直的偏角，
    yaw 绕世界 Z 轴（托盘平面内旋转），roll 绕光轴。图像“上”对应世界 -Y，
    因此托盘远侧出现在画面上方，与现场俯视图一致。
    """
    tilt = math.radians(90.0 - pitch_deg)
    yaw, roll = math.radians(yaw_deg), math.radians(roll_deg)
    Rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    d = Rz @ (Rx @ np.array([0.0, 0.0, -1.0]))          # 光轴方向（指向托盘）
    up_ref = np.array([0.0, -1.0, 0.0])                 # 画面“上”的参考方向
    x_cam = np.cross(up_ref, d)
    n = np.linalg.norm(x_cam)
    x_cam = x_cam / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])
    y_cam = np.cross(d, x_cam)
    xr = math.cos(roll) * x_cam + math.sin(roll) * y_cam
    yr = -math.sin(roll) * x_cam + math.cos(roll) * y_cam
    return np.vstack([xr, yr, d])


def project(pts_world, R, t, K):
    """世界坐标(mm) → 像素；返回 (uv, cam_z)"""
    fx, fy, cx, cy = K
    pc = (R @ pts_world.T).T + t.reshape(1, 3)      # 世界→相机
    z = pc[:, 2]
    z_safe = np.where(np.abs(z) < 1e-6, 1e-6, z)
    u = fx * pc[:, 0] / z_safe + cx
    v = fy * pc[:, 1] / z_safe + cy
    return np.stack([u, v], 1), z


def shade_faces(verts_cam, faces, albedo, light_dir, spec_strength=0.18, ambient=0.30):
    """逐面 Lambert + Blinn-Phong；返回 [(多边形像素, 颜色, 深度)]"""
    out = []
    view = np.array([0.0, 0.0, -1.0])
    for fidx in faces:
        pts = verts_cam[fidx]
        n = np.cross(pts[1] - pts[0], pts[2] - pts[0])
        norm = np.linalg.norm(n)
        if norm < 1e-9:
            continue
        n = n / norm
        center = pts.mean(0)
        # 面法线朝向相机才可见（背面剔除）
        if np.dot(n, center) > 0:
            n = -n
        if np.dot(n, view) <= 0.02:
            continue
        lam = max(0.0, float(np.dot(n, light_dir)))
        spec = spec_strength * max(0.0, float(np.dot(n, (light_dir + view) / 2.0))) ** 24
        col = np.clip(np.array(albedo, float) * (ambient + (1 - ambient) * lam) + 255 * spec, 0, 255)
        out.append((fidx, col.astype(np.uint8), float(center[2])))
    out.sort(key=lambda x: -x[2])              # 画家算法：远→近
    return out


# ============================================================
# 3. 场景渲染
# ============================================================
def render(scene, W, H, rng):
    K = camera_matrix(W, H, scene["hfov"])
    R = rot_matrix(scene["pitch"], scene["yaw"], scene["roll"])
    t = -R @ scene["cam_pos"]
    img = np.full((H, W, 3), scene["bg"], np.uint8)

    # ---- 托盘（浅色平面 + 边缘） ----
    half = scene["tray"] / 2.0
    tray_pts = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]], float)
    uv, _ = project(tray_pts, R, t, K)
    tray_col = np.array(scene["tray_color"], float)
    cv2.fillPoly(img, [uv.astype(np.int32)], tuple(int(v) for v in tray_col.astype(np.uint8)))
    edge = tuple(int(v) for v in (tray_col * 0.82).astype(np.uint8))
    cv2.polylines(img, [uv.astype(np.int32)], True, edge, max(2, int(W / 640) * 2))

    light_world = np.array(scene["light_dir"], float)
    light_world = light_world / max(1e-6, np.linalg.norm(light_world))   # 指向光源的单位向量
    light_cam = R @ light_world                                          # 变换到相机坐标系做着色

    # ---- 阴影：沿光线传播方向把物体顶点投到托盘平面，取凸包后模糊成半影 ----
    shadow_layer = np.zeros((H, W), np.uint8)
    if light_world[2] > 1e-3:
        Lx, Ly, Lz = light_world
        for obj in scene["objects"]:
            v = obj["verts"] + obj["pos"]
            s = v[:, 2] / Lz                                  # 沿 -光照方向 落到 z=0
            shadow_xy = np.stack([v[:, 0] - s * Lx, v[:, 1] - s * Ly], 1)
            hull = cv2.convexHull(shadow_xy.astype(np.float32)).reshape(-1, 2)
            uv_sh, _ = project(np.hstack([hull, np.zeros((len(hull), 1))]), R, t, K)
            cv2.fillPoly(shadow_layer, [uv_sh.astype(np.int32)],
                         int(210 * scene["shadow_strength"]), lineType=cv2.LINE_AA)
    s_blur = max(3, int(scene["shadow_blur"] * W / 640) | 1)
    shadow_layer = cv2.GaussianBlur(shadow_layer, (s_blur, s_blur), 0)
    alpha = (shadow_layer.astype(np.float32) / 255.0)[..., None]
    img = (img.astype(np.float32) * (1 - alpha * 0.62) +
           np.array(scene["bg_shadow"], np.float32) * alpha * 0.62).astype(np.uint8)

    # ---- 物体（按深度排序整体绘制，面内再排序） ----
    boxes = []
    for obj in sorted(scene["objects"], key=lambda o: -np.linalg.norm(o["center"] - scene["cam_pos"])):
        verts_world = obj["verts"] + obj["pos"]
        uv_all, z_all = project(verts_world, R, t, K)
        if np.any(z_all <= 0):
            continue
        verts_cam = (R @ verts_world.T).T + t.reshape(1, 3)
        obj_mask = np.zeros((H, W), np.uint8)
        for fidx, col, _depth in shade_faces(verts_cam, obj["faces"], obj["albedo"], light_cam,
                                             spec_strength=obj["spec"], ambient=scene["ambient"]):
            poly = uv_all[fidx]
            cv2.fillPoly(img, [poly.astype(np.int32)], tuple(int(c) for c in col), lineType=cv2.LINE_AA)
            cv2.fillPoly(obj_mask, [poly.astype(np.int32)], 255)     # 轮廓掩膜：贴图不得溢出物体

        # 表面图形：污渍 / 缺陷（按相机深度换算像素半径，且限制在物体轮廓内）
        if obj["decals"]:
            layer = img.copy()
            for decal in obj["decals"]:
                p3 = (decal["p"] + obj["pos"]).reshape(1, 3)
                pc = (R @ p3.T).T + t.reshape(1, 3)
                z_cam = float(pc[0, 2])
                if z_cam <= 1.0:
                    continue
                cu = int(K[0] * pc[0, 0] / z_cam + K[2])
                cv_c = int(K[1] * pc[0, 1] / z_cam + K[3])
                r_px = max(2, int(decal["r"] * K[0] / z_cam))        # 像素半径 = r * fx / Z_cam
                if decal["kind"] == "污渍":
                    cv2.circle(layer, (cu, cv_c), r_px, decal["color"], -1, lineType=cv2.LINE_AA)
                    cv2.circle(layer, (cu + int(r_px * 0.9), cv_c + int(r_px * 0.5)),
                               max(2, r_px // 2), decal["color2"], -1, lineType=cv2.LINE_AA)
                else:
                    pts = np.array([[cu - r_px, cv_c - r_px], [cu + r_px, cv_c - int(r_px * 0.4)],
                                    [cu + int(r_px * 0.2), cv_c + r_px],
                                    [cu - int(r_px * 0.6), cv_c + int(r_px * 0.6)]], np.int32)
                    cv2.fillPoly(layer, [pts], decal["color"], lineType=cv2.LINE_AA)
            blend = (obj_mask > 0)
            img[blend] = layer[blend]

        # 二维码贴纸（可选；贴在被抓取前可见的顶面附近）
        if obj.get("qr_img") is not None:
            qr = obj["qr_img"]
            side_mm = min(12.0, obj["height"] * 0.5)
            corners = np.array([[obj["pos"][0] - side_mm, obj["pos"][1] - side_mm, obj["height"] + 0.2],
                                [obj["pos"][0] + side_mm, obj["pos"][1] - side_mm, obj["height"] + 0.2],
                                [obj["pos"][0] + side_mm, obj["pos"][1] + side_mm, obj["height"] + 0.2],
                                [obj["pos"][0] - side_mm, obj["pos"][1] + side_mm, obj["height"] + 0.2]], float)
        else:
            corners = None
        uv = uv_all
        # ---- 检测框：真实投影轮廓（不含阴影） ----
        x1, y1 = uv[:, 0].min(), uv[:, 1].min()
        x2, y2 = uv[:, 0].max(), uv[:, 1].max()
        boxes.append({"obj": obj, "uv": uv, "xyxy": (x1, y1, x2, y2), "qr_corners": corners})

    for b in boxes:
        if b["qr_corners"] is not None:
            uvc, _ = project(b["qr_corners"], R, t, K)
            dst = np.array([[0, 0], [63, 0], [63, 63], [0, 63]], np.float32)
            M = cv2.getPerspectiveTransform(dst, uvc.astype(np.float32))
            warp = cv2.warpPerspective(b["obj"]["qr_img"], M, (W, H), flags=cv2.INTER_LINEAR,
                                       borderMode=cv2.BORDER_TRANSPARENT)
            mask = warp.sum(2) > 0
            img[mask] = warp[mask]

    # ---- 后处理：只做现实存在的差异（噪声/白平衡/曝光/轻微失焦/JPEG） ----
    img = post_process(img, scene, rng)
    return img, boxes, K


def post_process(img, scene, rng):
    img = img.astype(np.float32)
    # 白平衡 / 色温偏移
    gain = np.array([1 + scene["wb"][0], 1.0, 1 + scene["wb"][1]], np.float32)
    img *= gain.reshape(1, 1, 3)
    # 曝光与对比度
    img = (img - 128) * scene["contrast"] + 128 * scene["exposure"]
    # 暗角（镜头渐晕）
    H, W = img.shape[:2]
    yy, xx = np.mgrid[0:H, 0:W]
    r2 = ((xx - W / 2) / (W / 2)) ** 2 + ((yy - H / 2) / (H / 2)) ** 2
    img *= (1 - scene["vignette"] * r2)[..., None]
    img = np.clip(img, 0, 255)
    # 轻微失焦 + 传感器噪声（ISO 抖动）
    if scene["blur"] > 0.35:
        img = cv2.GaussianBlur(img, (3, 3), scene["blur"])
    if scene["noise"] > 0:
        img += np.random.normal(0, scene["noise"], img.shape).astype(np.float32)
    img = np.clip(img, 0, 255).astype(np.uint8)
    return img


def weighted_shape(rng, shapes, prism_weight=1.0):
    """按权重采样形状（棱柱可加权，比赛货物以多面体/棱柱为主）"""
    weights = []
    for s in shapes:
        w = prism_weight if s in ("五棱柱", "六棱柱") else 1.0
        weights.append(w)
    total = sum(weights)
    r = rng.random() * total
    acc = 0.0
    for s, w in zip(shapes, weights):
        acc += w
        if r <= acc:
            return s
    return shapes[-1]


def sample_scene(rng, per_image, imgsz, tray_mm, goods_mm, shapes, colors, prism_weight=1.0):
    W, H = imgsz
    cam_h = REAL["cam_height"] * rng.uniform(0.85, 1.15)          # 远近（装夹高度误差）
    scene = {
        "hfov": REAL["hfov_deg"] * rng.uniform(0.96, 1.04),
        "pitch": REAL["pitch_deg"] + rng.uniform(-7, 7),          # 角度（俯仰）
        "yaw": rng.uniform(-6, 6), "roll": rng.uniform(-5, 5),
        "cam_pos": np.array([rng.uniform(-15, 15), rng.uniform(-15, 15), cam_h]),
        "tray": tray_mm, "tray_color": [int(v) for v in np.array([236, 236, 238]) * rng.uniform(0.93, 1.04)],
        "bg": int(np.clip(210 * rng.uniform(0.75, 1.0), 120, 245)),
        "bg_shadow": np.array([rng.uniform(60, 100)] * 3),
        "shadow_strength": rng.uniform(0.45, 1.0),
        "shadow_blur": rng.uniform(3, 12),                        # 光源尺寸 → 半影
        "ambient": rng.uniform(0.30, 0.46),
        # 光源：指向光源的单位向量（z>0 = 光源在托盘上方，室内顶灯/侧上方补光）
        "light_dir": [rng.uniform(-0.45, 0.45), rng.uniform(-0.45, 0.45), rng.uniform(0.85, 1.0)],
        "wb": (rng.uniform(-0.05, 0.05), rng.uniform(-0.05, 0.05)),
        "exposure": rng.uniform(0.92, 1.08), "contrast": rng.uniform(0.92, 1.08),
        "vignette": rng.uniform(0.0, 0.16),
        "blur": rng.uniform(0.0, 0.8), "noise": rng.uniform(0.0, 4.5),
        "objects": [], "imgsz": imgsz,
    }
    # 摆放：托盘内网格抖动，避免重叠（贴边放置，符合现场摆放）
    grid = int(math.ceil(math.sqrt(per_image)))
    cell = tray_mm / grid
    rng.shuffle_like = None
    for idx in range(per_image):
        gx, gy = idx % grid, idx // grid
        cx = -tray_mm / 2 + cell * (gx + 0.5) + rng.uniform(-cell * 0.18, cell * 0.18)
        cy = -tray_mm / 2 + cell * (gy + 0.5) + rng.uniform(-cell * 0.18, cell * 0.18)
        shape = weighted_shape(rng, shapes, prism_weight)
        color = rng.choice(colors)
        scale = rng.uniform(0.85, 1.05)                           # 尺寸公差（≤40mm）
        verts, faces = build_mesh(shape, scale)
        yaw = rng.uniform(0, 2 * math.pi)
        c, s = math.cos(yaw), math.sin(yaw)
        rot = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        # 轻微倾斜（≤10°，货物平放在托盘上的真实情况）
        tilt = math.radians(rng.uniform(0, 9))
        tax = rng.uniform(0, 2 * math.pi)
        Rt = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
        verts = (verts @ rot.T) @ Rt.T
        height = float(verts[:, 2].max())
        albedo = np.array(COLORS[color], float) * rng.uniform(0.88, 1.1)
        decals = []
        if rng.random() < STAIN_PROB:
            decals.append({"kind": "污渍", "p": np.array([rng.uniform(-6, 6), rng.uniform(-6, 6), height + 0.5]),
                           "r": rng.uniform(3, 6), "color": (40, 55, 75), "color2": (55, 72, 95)})
        if rng.random() < DEFECT_PROB:
            decals.append({"kind": "缺陷", "p": np.array([rng.uniform(-8, 8), rng.uniform(-8, 8), height + 0.5]),
                           "r": rng.uniform(2.5, 5), "color": (25, 25, 28)})
        scene["objects"].append({
            "shape": shape, "color": color, "verts": verts, "faces": faces,
            "pos": np.array([cx, cy, 0.0]), "height": height,
            "albedo": albedo, "spec": rng.uniform(0.06, 0.2),
            "decals": decals, "yaw_deg": math.degrees(yaw), "scale": scale,
            "marks": (["污渍"] if any(d["kind"] == "污渍" for d in decals) else []) +
                     (["缺陷"] if any(d["kind"] == "缺陷" for d in decals) else []),
            "center": np.array([cx, cy, height / 2]),
            "qr_img": None,
        })
    return scene


# ============================================================
# 4. 数据集写出
# ============================================================
def write_yolo_label(path, boxes, W, H, cls_id=0, min_px=6):
    lines = []
    for b in boxes:
        x1, y1, x2, y2 = b["xyxy"]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        if (x2 - x1) < min_px or (y2 - y1) < min_px:
            continue
        cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
        bw, bh = (x2 - x1) / W, (y2 - y1) / H
        lines.append("{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(cls_id, cx, cy, bw, bh))
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + ("\n" if lines else ""))


def main():
    ap = argparse.ArgumentParser(description="多面体合成数据集生成器（阴影/远近/角度 + sim-to-real）")
    ap.add_argument("--out", default="data/synth", help="输出目录")
    ap.add_argument("--n", type=int, default=400, help="图片数量")
    ap.add_argument("--per-image", type=int, default=1, help="每张图货物数量（1=属性集，2~4=检测集）")
    ap.add_argument("--prefix", default="synth", help="文件名前缀（多批次生成到同一目录时区分）")
    ap.add_argument("--start-index", type=int, default=0, help="起始序号（追加生成时避免覆盖）")
    ap.add_argument("--prism-weight", type=float, default=1.0,
                    help="棱柱（五棱柱/六棱柱）采样加权，>1 提高出现概率")
    ap.add_argument("--imgsz", type=int, nargs=2, default=list(REAL["imgsz"]), help="分辨率 W H")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--shapes", nargs="*", default=SHAPES)
    ap.add_argument("--colors", nargs="*", default=list(COLORS.keys()))
    ap.add_argument("--preview", type=int, default=8, help="抽样拼图张数")
    ap.add_argument("--no-attributes", action="store_true", help="不生成属性分类小图")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    np.random.seed(args.seed)
    W, H = int(args.imgsz[0]), int(args.imgsz[1])
    out = args.out
    os.makedirs(out, exist_ok=True)

    stats = {"images": 0, "objects": 0, "shapes": {}, "colors": {}, "marks": {},
             "bbox_px": {"min": 1e9, "max": 0, "mean": 0.0}, "tilt_deg_max": 0.0,
             "camera": REAL, "resolution": [W, H]}
    previews = []
    attr_count = 0
    px_sizes = []

    for i in range(args.n):
        split = "val" if rng.random() < args.val_ratio else "train"
        scene = sample_scene(rng, args.per_image, (W, H), REAL["tray_mm"], REAL["goods_mm"],
                             args.shapes, args.colors, args.prism_weight)
        img, boxes, K = render(scene, W, H, rng)
        name = "{0}_{1:06d}".format(args.prefix, args.start_index + i)
        img_path = os.path.join(out, "images", split, name + ".jpg")
        os.makedirs(os.path.dirname(img_path), exist_ok=True)
        quality = int(rng.uniform(85, 96))                 # 模拟相机压缩
        cv2.imwrite(img_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
        img_boxes = boxes
        write_yolo_label(os.path.join(out, "labels", split, name + ".txt"), boxes, W, H)

        for b in boxes:
            obj = b["obj"]
            x1, y1, x2, y2 = [int(v) for v in b["xyxy"]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            if x2 - x1 < 8 or y2 - y1 < 8:
                continue
            stats["objects"] += 1
            stats["shapes"][obj["shape"]] = stats["shapes"].get(obj["shape"], 0) + 1
            stats["colors"][obj["color"]] = stats["colors"].get(obj["color"], 0) + 1
            for m in (obj["marks"] or ["无"]):
                stats["marks"][m] = stats["marks"].get(m, 0) + 1
            px_sizes.append(max(x2 - x1, y2 - y1))
            stats["tilt_deg_max"] = max(stats["tilt_deg_max"], 9.0)
            if not args.no_attributes:
                crop = img[y1:y2, x1:x2]
                if crop.size:
                    crop = cv2.resize(crop, (96, 96))
                    m = 0.12
                    pad = int(max(x2 - x1, y2 - y1) * m)
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
                    pad_crop = cv2.resize(img[cy1:cy2, cx1:cx2], (96, 96)) if cy2 > cy1 and cx2 > cx1 else crop
                    for task, cls in (("color", obj["color"]), ("shape", obj["shape"]),
                                      ("stain", (obj["marks"] or ["clean"])[0] if obj["marks"] else "clean")):
                        cls_name = {"污渍": "stain", "缺陷": "defect", "无": "clean"}.get(cls, cls)
                        cls_name = {"红色": "red", "橙色": "orange", "黄色": "yellow", "绿色": "green",
                                    "青色": "cyan", "蓝色": "blue", "紫色": "purple", "黑色": "black",
                                    "白色": "white"}.get(cls_name, cls_name)
                        cls_name = {"正方体": "cube", "长方体": "cuboid", "圆柱": "cylinder", "球": "ball",
                                    "正四面体": "tetra", "五棱柱": "prism5", "六棱柱": "prism6",
                                    "正十二面体": "dodecahedron", "圆锥": "cone"}.get(cls_name, cls_name)
                        d = os.path.join(out, "attributes", task, cls_name)
                        os.makedirs(d, exist_ok=True)
                        cv2.imwrite(os.path.join(d, "{0}_{1}.jpg".format(name, len(px_sizes))),
                                    pad_crop if task != "stain" else crop)
                        attr_count += 1
        stats["images"] += 1
        if len(previews) < args.preview:
            previews.append(img)

    if px_sizes:
        stats["bbox_px"] = {"min": int(np.min(px_sizes)), "max": int(np.max(px_sizes)),
                            "mean": round(float(np.mean(px_sizes)), 1),
                            "median": int(np.median(px_sizes))}
    stats["attributes"] = attr_count

    if previews:
        cols = min(4, len(previews))
        rows = int(math.ceil(len(previews) / cols))
        tw, th = W // 2, H // 2
        grid = np.full((rows * th, cols * tw, 3), 255, np.uint8)
        for k, p in enumerate(previews):
            r, c = divmod(k, cols)
            grid[r * th:(r + 1) * th, c * tw:(c + 1) * tw] = cv2.resize(p, (tw, th))
        cv2.imwrite(os.path.join(out, "preview.jpg"), grid, [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    with open(os.path.join(out, "dataset_stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    print("=" * 56)
    print(" 合成数据集完成 → {0}".format(out))
    print("=" * 56)
    print(" 图片 {0} 张 | 目标 {1} 个 | 属性小图 {2} 张".format(stats["images"], stats["objects"], attr_count))
    print(" 目标像素尺寸: 中位 {0}px  范围 {1}~{2}px".format(
        stats["bbox_px"].get("median"), stats["bbox_px"].get("min"), stats["bbox_px"].get("max")))
    print(" 形状分布:", stats["shapes"])
    print(" 颜色分布:", stats["colors"])
    print(" 表面分布:", stats["marks"])
    print(" 预览图: {0}/preview.jpg".format(out))
    print()
    print("下一步：")
    print("  1) 合并实拍数据（推荐 30~50 张现场照片）后训练：")
    print("     python train_detection.py --config config/sorting_competition.yaml --epochs 150")
    print("  2) 属性分类器（颜色/形状/污渍）：")
    print("     python train_attributes.py --config config/sorting_competition.yaml --task shape --export-onnx")
    print("  3) 用现场照片做少量微调（sim-to-real 关键一步，见 README）")


if __name__ == "__main__":
    main()
