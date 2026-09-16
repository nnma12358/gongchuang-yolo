#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calibrate.py —— 相机/机械臂标定工具（阶段4）
=====================================================================
用法一：由标定点对求解（现场标准流程）
  1) 在托盘上放 6~9 个可辨认的标定点（例如贴纸/尖锐物），记录它们的**像素坐标**
     （可用顶置画面量取；或把标定板放托盘上拍照后用 detect 输出的 box 中心）
  2) 手动/示教把夹爪中心依次移动到这些点，记录**机械臂基坐标 XY（mm）**
  3) 写入 CSV：u,v,x_mm,y_mm[,z_mm]
  4) 运行：
     python3 scripts/calibrate.py --points calib_points.csv --out data/calib.json \
         --bins "1:120,60 2:120,0 3:120,-60 4:-120,60 5:-120,0 6:-120,-60"

用法二：用模拟数据自检（无硬件也能验证工具链）
     python3 scripts/calibrate.py --simulate --out /tmp/calib_test.json
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import pose  # noqa: E402


def read_points(path):
    pixels, robots, cam3d = [], [], []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                vals = [float(x) for x in row[:5]]
            except ValueError:
                continue
            if len(vals) < 4:
                continue
            pixels.append([vals[0], vals[1]])
            robots.append([vals[2], vals[3]])
            cam3d.append(vals[4] if len(vals) > 4 else None)
    return pixels, robots, cam3d


def parse_bins(spec):
    """'1:120,60 2:120,0' → {'1': {'x':120,'y':60,'z':0}, ...}"""
    bins = {}
    if not spec:
        return bins
    for item in spec.split():
        if ":" not in item:
            continue
        no, xy = item.split(":", 1)
        parts = [float(v) for v in xy.replace("，", ",").split(",")[:3]]
        bins[no.strip()] = {"x": parts[0], "y": parts[1], "z": parts[2] if len(parts) > 2 else 0.0}
    return bins


def main():
    ap = argparse.ArgumentParser(description="标定六储物盒与像素↔机械臂坐标映射")
    ap.add_argument("--points", help="标定点 CSV：u,v,x_mm,y_mm[,z_mm]")
    ap.add_argument("--out", default=pose.DEFAULT_CALIB_PATH, help="输出标定文件")
    ap.add_argument("--bins", default="", help='六盒坐标："1:x,y 2:x,y ..."')
    ap.add_argument("--tray-pixels", default="", help='托盘四角像素："x1,y1 x2,y2 x3,y3 x4,y4"')
    ap.add_argument("--width", type=int, default=1280, help="图像宽（内参用）")
    ap.add_argument("--height", type=int, default=720, help="图像高")
    ap.add_argument("--hfov", type=float, default=60.0, help="水平视场角（度）")
    ap.add_argument("--test-ratio", type=float, default=0.25, help="留出验证点比例")
    ap.add_argument("--board-height", type=float, default=0.0,
                    help="标定块高度 mm（2.5D 刚体标定用；0=点在托盘面上）")
    ap.add_argument("--simulate", action="store_true", help="用模拟点对自检（无需硬件）")
    args = ap.parse_args()

    if args.simulate:
        rng = np.random.RandomState(7)
        K = pose.camera_matrix_from_fov(args.width, args.height, args.hfov)
        # 真值：相机在托盘上方 320mm，托盘平面 z=0；机械臂基座与相机光心对齐
        R_true = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], float)
        t_true = np.array([0.0, 0.0, 320.0])
        robots = []
        for x in (-60, 0, 60):
            for y in (-60, 0, 60):
                robots.append([x, y])
        pix, cam3d = [], []
        for (x, y) in robots:
            p_cam = R_true.T @ (np.array([x, y, 0.0]) - t_true)   # 基坐标 → 相机系
            z = p_cam[2]
            u = K["fx"] * p_cam[0] / z + K["cx"]
            v = K["fy"] * p_cam[1] / z + K["cy"]
            pix.append([u + rng.normal(0, 0.7), v + rng.normal(0, 0.7)])   # 0.7px 取点噪声
            cam3d.append(z)
        pixels, robots, cam3d = pix, robots, cam3d
        print("[模拟] 生成 {0} 组标定点（含 0.7px 噪声）".format(len(pixels)))
        bins = parse_bins("1:120,60 2:120,0 3:120,-60 4:-120,60 5:-120,0 6:-120,-60")
    else:
        if not args.points:
            ap.error("需要 --points CSV 或 --simulate")
        pixels, robots, cam3d = read_points(args.points)
        bins = parse_bins(args.bins)
        print("[输入] 标定点 {0} 组 | 储物盒 {1} 个".format(len(pixels), len(bins)))
        if len(pixels) < 4:
            ap.error("至少 4 组标定点（建议 6~9 组）")

    # 留出验证点
    n = len(pixels)
    idx = list(range(n))
    rng = np.random.RandomState(0)
    rng.shuffle(idx)
    n_test = max(1, int(n * args.test_ratio)) if n >= 5 else 0
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    P_tr = [pixels[i] for i in train_idx]
    R_tr = [robots[i] for i in train_idx]

    calib = {"created": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
             "camera": pose.camera_matrix_from_fov(args.width, args.height, args.hfov),
             "gripper": pose.GRIPPER, "pick": pose.PICK, "workspace": pose.WORKSPACE,
             "n_points": n, "bins": bins}

    # ---- 路径 A：平面单应 ----
    H = pose.solve_homography(P_tr, R_tr)
    rms_all, errs_all = pose.homography_error(H, pixels, robots)
    calib["homography"] = [[float(v) for v in row] for row in H]
    calib["homography_rms_mm"] = round(rms_all, 3)
    calib["homography_errors_mm"] = [round(e, 2) for e in errs_all]
    print("\n[路径A 平面单应] 全部点 RMS = {0:.2f} mm  最大 = {1:.2f} mm".format(
        rms_all, max(errs_all)))
    if test_idx:
        rms_test, _ = pose.homography_error(H, [pixels[i] for i in test_idx], [robots[i] for i in test_idx])
        calib["homography_rms_test_mm"] = round(rms_test, 3)
        print("                 留出验证点 RMS = {0:.2f} mm（阶段4 目标 < 2mm）".format(rms_test))

    # ---- 路径 B：2.5D 刚体变换（标定点带深度时启用）----
    if all(z is not None for z in cam3d):
        cam_pts = [pose.pixel_depth_to_camera(pixels[i][0], pixels[i][1], cam3d[i], calib["camera"])
                   for i in range(n)]
        robot3d = [[robots[i][0], robots[i][1], args.board_height] for i in range(n)]
        if pose.coplanar_ratio(robot3d):
            print("[路径B 2.5D深度] 标定点共面 → 刚体解仅在平面内可靠；"
                  "若目标高度差异大，请用不同高度的标定块补点（或采用路径A）")
        try:
            Rb, tb = pose.solve_rigid([cam_pts[i] for i in train_idx], [robot3d[i] for i in train_idx])
            rms_r, _ = pose.rigid_error(Rb, tb, cam_pts, robot3d)
            calib["rigid"] = {"R": [[float(v) for v in row] for row in Rb],
                              "t": [float(v) for v in tb],
                              "board_height_mm": args.board_height}
            calib["rigid_rms_mm"] = round(rms_r, 3)
            print("[路径B 2.5D深度] 全部点 RMS = {0:.2f} mm".format(rms_r))
        except Exception as e:
            print("[路径B 2.5D深度] 求解失败（{0}）→ 请以路径A 为主".format(e))
    else:
        print("[路径B 2.5D深度] 跳过（CSV 未提供第 5 列深度 z_mm）")

    # ---- 六盒坐标（未提供时用单应把像素落点换算，或提示手动补）----
    if not bins:
        print("\n⚠ 未提供 --bins：请补充六个储物盒中心坐标（机械臂 XY）后才能自动放置")
    else:
        print("\n[储物盒] {0} 个: {1}".format(len(bins), ", ".join(
            "{0}号({1:.0f},{2:.0f})".format(k, v["x"], v["y"]) for k, v in sorted(bins.items()))))

    if args.tray_pixels:
        corners = [[float(v) for v in p.split(",")] for p in args.tray_pixels.split()]
        if len(corners) == 4:
            calib["tray_pixels"] = corners
            side_mm = []
            for i in range(4):
                x1, y1 = pose.pixel_to_robot(corners[i][0], corners[i][1], H)
                x2, y2 = pose.pixel_to_robot(corners[(i + 1) % 4][0], corners[(i + 1) % 4][1], H)
                side_mm.append(round(math.hypot(x1 - x2, y1 - y2), 1))
            calib["tray_side_mm"] = side_mm
            print("[托盘] 四边实测 {0} mm（赛项要求 ≥160mm）".format(side_mm))

    ok, msg = pose.calib_ready(calib)
    calib["ready"] = bool(ok)
    calib["ready_msg"] = msg
    path = pose.save_calib(calib, args.out)
    print("\n标定文件: {0}".format(path))
    print("状态: {0}".format("✅ " + msg if ok else "⚠ " + msg))
    print("\n示例（用第一个标定点反算验证）：")
    x, y = pose.pixel_to_robot(pixels[0][0], pixels[0][1], H)
    print("  像素 ({0:.0f},{1:.0f}) → 机械臂 ({2:.1f},{3:.1f}) mm   （实测 {4:.1f},{5:.1f} mm）".format(
        pixels[0][0], pixels[0][1], x, y, robots[0][0], robots[0][1]))


if __name__ == "__main__":
    main()
