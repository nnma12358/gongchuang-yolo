#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
adapt_localization.py —— 适配工作区 2.5D 定位软件的落地脚本
=====================================================================
把 smart_sorting_localization_* 的标定与结果接进本项目：

  ① 导入内参与桌面参考标定 → data/calib.json（阶段4 几何基准）
  ② 融合其定位结果与本项目识别结果 → 网关可直接消费的标记 + 抓取/投放位姿
  ③ 用其夹爪红点 TCP 检测做**自动手眼标定**：记录 (TCP 像素 ←→ 机械臂 XY) 点对
     累积到 calib_points.csv，可直接求解/更新单应

用法：
  # ① 导入桌面参考与内参
  python3 scripts/adapt_localization.py import --table <.../table_reference_candidate_*> \\
      --intrinsics <.../camera_intrinsics.json> --calib data/calib.json

  # ② 融合定位与识别（用其真实结果 JSON 验证）
  python3 scripts/adapt_localization.py merge \\
      --objects <.../object_locator_result.json> --marks /tmp/marks.json \\
      --calib data/calib.json --out /tmp/merged.json

  # ③ 自动手眼标定：每次把夹爪移到已知点后记录一对
  python3 scripts/adapt_localization.py tcp \\
      --tcp <.../gripper_red_marker_result.json> --robot-x 120 --robot-y 60 \\
      --points data/calib_points.csv            # 追加
  python3 scripts/adapt_localization.py tcp --solve --points data/calib_points.csv \\
      --calib data/calib.json --bins "1:120,60 2:120,0 ..."
"""
import argparse
import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import overhead_localization as ol  # noqa: E402
import pose  # noqa: E402


def cmd_import(args):
    ref = ol.load_table_reference(args.table)
    K = ol.load_intrinsics(args.intrinsics) if args.intrinsics else ref.get("intrinsics")
    print("[桌面参考] {0}".format(ref["path"]))
    print("  桌面深度中位 {0:.1f} mm | 有效像素占比 {1:.1%} | 内参 fx={2:.2f} cx={3:.1f} cy={4:.1f}".format(
        ref["table_depth_median_mm"] or -1, ref["table_valid_ratio"],
        (K or {}).get("fx", 0), (K or {}).get("cx", 0), (K or {}).get("cy", 0)))

    calib = pose.load_calib(args.calib) or {}
    calib["camera"] = {"fx": K["fx"], "fy": K["fy"], "cx": K["cx"], "cy": K["cy"],
                       "width": K.get("width", 640), "height": K.get("height", 480),
                       "source": "astra_intrinsics"}
    calib["table_reference"] = {
        "path": os.path.relpath(ref["path"]),
        "table_depth_median_mm": ref["table_depth_median_mm"],
        "valid_ratio": round(ref["table_valid_ratio"], 4),
        "resolution": list(ref["table_depth_mm"].shape),
    }
    path = pose.save_calib(calib, args.calib)
    print("[标定] 已写入 {0}（内参 + 桌面参考）".format(path))
    ok, msg = pose.calib_ready(calib)
    print("[状态] {0}".format(("✅ " if ok else "⚠ ") + msg))
    if not calib.get("homography"):
        print("\n下一步：用夹爪红点做自动手眼标定 ——")
        print("  python3 scripts/adapt_localization.py tcp --tcp <marker.json> --robot-x X --robot-y Y --points data/calib_points.csv")
        print("  重复采集 ≥4 个点后：--solve --points data/calib_points.csv --calib data/calib.json")


def cmd_merge(args):
    loc = ol.load_object_locator(args.objects)
    marks = json.load(open(args.marks, encoding="utf-8")) if args.marks else []
    if isinstance(marks, dict):
        marks = marks.get("detections") or marks.get("marks") or []
    print("[设备定位] 目标 {0} 个（方法 {1}，图像 {2}）".format(
        len(loc["objects"]), loc.get("method"), loc.get("image_size")))
    for o in loc["objects"]:
        print("  id={0} 像素{1} 外观{2} 面积{3}px² 夹爪角{4}°".format(
            o["id"], o["center_pixel"], o["appearance"], o["area_px"], o["gripper_image_angle_deg"]))
    if args.table:
        ref = ol.load_table_reference(args.table)
        K = loc.get("intrinsics") or ref.get("intrinsics")
        for o in loc["objects"]:
            geo = ol.object_2p5d(o["center_pixel"][0], o["center_pixel"][1], None, ref, K)
            o.update(geo)
            print("  id={0} 桌面深度 {1}mm 置信 {2}".format(o["id"], geo["table_depth_mm"], geo["confidence"]))

    merged, unmatched = ol.merge_with_marks(loc["objects"], marks)
    calib = pose.load_calib(args.calib) if args.calib else None
    ok = False
    if calib:
        ok, msg = pose.calib_ready(calib)
        print("[标定] {0}".format(msg))
    for item in merged:
        print("\n[id {0}] 像素 {1} 外观 {2} | 匹配分 {3}".format(
            item["id"], item["center_pixel"], item.get("appearance"), item["match_score"]))
        print("   识别: 形状={0} 颜色={1} 表面={2} 二维码={3}".format(
            item.get("shape"), item.get("color"), item.get("marks"), item.get("qr") or "-"))
        if item.get("height_mm") is not None:
            print("   2.5D: 顶面深度 {0}mm 高出桌面 {1}mm 顶面相机坐标 {2}".format(
                item.get("top_depth_mm"), item.get("height_mm"), item.get("top_camera_mm")))
        if ok and item.get("shape") and item.get("color"):
            calib_obj = calib or {}
            g = pose.grasp_from_detection(
                {"box": item["box"], "shape": item["shape"], "color": item["color"],
                 "marks": item.get("marks"), "height_mm": item.get("height_mm"),
                 "angle_deg": item.get("gripper_image_angle_deg")},
                H=calib_obj.get("homography"), K=calib_obj.get("camera"))
            item["grasp"] = g
            print("   抓取: {0} | joint6={1} | 可达={2}".format(g["grasp_xy"], g["joint6_target"], g["reachable"]))
            bins = calib_obj.get("bins") or {}
            if bins:
                place = pose.bin_place_pose(sorted(bins.keys(), key=lambda k: int(k))[0], bins)
                item["place"] = place
                print("   投放(示例 1 号盒): {0}".format(place))
    if unmatched:
        print("\n[未匹配的 RGB 识别] {0} 条（可能是误检或设备定位漏检，建议人工复核）".format(len(unmatched)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"merged": merged, "unmatched_marks": unmatched}, f, ensure_ascii=False, indent=2)
        print("\n输出: {0}".format(args.out))


def cmd_tcp(args):
    points = []
    if not args.solve and os.path.exists(args.points):
        with open(args.points, newline="", encoding="utf-8") as f:
            for row in csv.reader(f):
                if row and not row[0].strip().startswith("#"):
                    points.append(row)

    if not args.solve:
        m = ol.load_tcp_marker(args.tcp)
        if not m["tcp_pixel"]:
            print("✘ 未在该结果中找到 tcp_pixel"); return
        row = ["{0:.2f}".format(m["tcp_pixel"][0]), "{0:.2f}".format(m["tcp_pixel"][1]),
               "{0:.2f}".format(args.robot_x), "{0:.2f}".format(args.robot_y)]
        header_needed = not os.path.exists(args.points)
        with open(args.points, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if header_needed:
                w.writerow(["# u_px", "v_px", "robot_x_mm", "robot_y_mm"])
            w.writerow(row)
        print("[TCP 标定点] 像素 ({0:.1f},{1:.1f}) ←→ 机械臂 ({2:.1f},{3:.1f}) mm".format(
            m["tcp_pixel"][0], m["tcp_pixel"][1], args.robot_x, args.robot_y))
        print("  夹爪轴向 {0}° | 开口 {1}px | 正交误差 {2}°".format(
            m["jaw_open_axis_deg"], m["jaw_separation_px"], m["orthogonality_error_deg"]))
        print("  已追加到 {0}".format(args.points))
        n = len(points) + 1
        print("  累计 {0} 个点{1}".format(n, "（≥4 即可求解，建议 6~9 个覆盖托盘四角与中心）" if n < 6 else "（可 --solve 求解）"))
        return

    # 求解
    pixels, robots = [], []
    with open(args.points, newline="", encoding="utf-8") as f:
        for row in csv.reader(f):
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                pixels.append([float(row[0]), float(row[1])])
                robots.append([float(row[2]), float(row[3])])
            except (ValueError, IndexError):
                continue
    print("[求解] 使用 {0} 个 TCP 标定点".format(len(pixels)))
    H = pose.solve_homography(pixels, robots)
    rms, errs = pose.homography_error(H, pixels, robots)
    print("  RMS = {0:.2f} mm | 最大 = {1:.2f} mm（目标 <2mm）".format(rms, max(errs)))
    calib = pose.load_calib(args.calib) or {}
    calib["homography"] = [[float(v) for v in row] for row in H]
    calib["homography_rms_mm"] = round(rms, 3)
    calib["homography_source"] = "gripper_tcp_marker_auto"
    calib["homography_points"] = len(pixels)
    if args.bins:
        bins = {}
        for item in args.bins.split():
            if ":" in item:
                no, xy = item.split(":", 1)
                vals = [float(v) for v in xy.split(",")[:3]]
                bins[no] = {"x": vals[0], "y": vals[1], "z": vals[2] if len(vals) > 2 else 0.0}
        calib["bins"] = bins
    path = pose.save_calib(calib, args.calib)
    ok, msg = pose.calib_ready(calib)
    print("[标定] 已写入 {0}".format(path))
    print("[状态] {0}".format(("✅ " if ok else "⚠ ") + msg))


def main():
    ap = argparse.ArgumentParser(description="适配 2.5D 定位软件（导入标定 / 融合结果 / 自动手眼标定）")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p1 = sub.add_parser("import", help="导入内参与桌面参考标定")
    p1.add_argument("--table", required=True, help="table_reference.npz 或其所在目录")
    p1.add_argument("--intrinsics", help="camera_intrinsics.json")
    p1.add_argument("--calib", default="data/calib.json")
    p1.set_defaults(func=cmd_import)

    p2 = sub.add_parser("merge", help="融合设备定位与本项目识别")
    p2.add_argument("--objects", required=True, help="object_locator_result.json")
    p2.add_argument("--marks", help="本项目识别标记 JSON（数组或 {detections: [...]}）")
    p2.add_argument("--table", help="table_reference 目录（用于 2.5D 高度重算）")
    p2.add_argument("--calib", default="data/calib.json")
    p2.add_argument("--out", help="融合结果输出 JSON")
    p2.set_defaults(func=cmd_merge)

    p3 = sub.add_parser("tcp", help="夹爪红点自动手眼标定（记录点对 / 求解）")
    p3.add_argument("--tcp", help="gripper_red_marker_result.json（记录点时必填）")
    p3.add_argument("--robot-x", type=float, default=0.0)
    p3.add_argument("--robot-y", type=float, default=0.0)
    p3.add_argument("--points", default="data/calib_points.csv")
    p3.add_argument("--solve", action="store_true", help="由已记录点对求解单应并写入标定")
    p3.add_argument("--calib", default="data/calib.json")
    p3.add_argument("--bins", default="")
    p3.set_defaults(func=cmd_tcp)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
