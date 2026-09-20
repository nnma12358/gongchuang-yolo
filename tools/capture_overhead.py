# -*- coding: utf-8 -*-
"""现场顶置相机采集工具（PC 直接拉 Jetson 的 HTTP 取图接口，无需手动拍照）

用途：为「真实尺度重训」采集真实域数据。现场摆一次货、换几次位置/朝向/数量，
本工具按间隔连续抓帧并自动去重（画面几乎没变就跳过），产出可直接进标注流程的数据集。

依赖：现场已跑起 sort-ros-bridge（:8123）与/或 sort-vision（:8100）。
      PC 需能访问 Jetson 的这两个端口（host 网络模式，默认 10.60.10.24）。

产出目录结构（与 train_detection.py 的 YOLO 数据集一致）：
    <out>/<session>/images/<session>_0001.jpg      彩色帧
    <out>/<session>/depth/<session>_0001.png        深度帧（原始 16bit，单位 mm）
    <out>/<session>/depth_preview/<session>_0001.png 深度预览（8bit，归一化，仅供查看）
    <out>/<session>/labels/<session>_0001.txt      标注（初始为空，由标注/复核流程写入）
    <out>/<session>/session.json                   本次采集元信息与去重统计

用法：
    # 采集 60 张（间隔 0.8s，画面变化超过阈值才存）
    python3 tools/capture_overhead.py --session tray_a --n 60

    # 纯背景（无货物）→ 作为负样本，标注目录保持为空即可
    python3 tools/capture_overhead.py --session bg_lab --n 40

    # 只抓彩色（不抓深度）
    python3 tools/capture_overhead.py --session tray_b --n 60 --no-depth

标注建议：采集完成后用 tools/review_app.py 复核（本工具的 labels 初始为空，
        可先用 tools/propose_boxes.py 生成候选框再人工修正）。
"""
import argparse
import json
import os
import time
import urllib.request

import numpy as np


def fetch(url, timeout=6.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.read()


def main():
    ap = argparse.ArgumentParser(description="现场顶置相机采集（PC 端直接拉 HTTP 取图）")
    ap.add_argument("--host", default="10.60.10.24", help="Jetson IP")
    ap.add_argument("--bridge-port", type=int, default=8123, help="sort-ros-bridge 端口")
    ap.add_argument("--vision-port", type=int, default=8100, help="sort-vision 端口（用于 /frame.jpg 带标记图）")
    ap.add_argument("--session", required=True, help="本次采集名（同时作为文件名前缀）")
    ap.add_argument("--out", default="data/real_overhead", help="输出根目录")
    ap.add_argument("--n", type=int, default=60, help="最多保存多少张")
    ap.add_argument("--interval", type=float, default=0.8, help="抓帧间隔秒")
    ap.add_argument("--diff-thres", type=float, default=3.0,
                    help="与上一张已存帧的平均灰度差阈值，低于此值视为画面未变而跳过")
    ap.add_argument("--no-depth", action="store_true", help="不抓深度帧")
    ap.add_argument("--marked", action="store_true", help="额外保存一张带标记的 /frame.jpg（看管线当前输出）")
    args = ap.parse_args()

    import cv2

    color_url = "http://{0}:{1}/color.jpg".format(args.host, args.bridge_port)
    depth_url = "http://{0}:{1}/depth.png".format(args.host, args.bridge_port)
    marked_url = "http://{0}:{1}/frame.jpg".format(args.host, args.vision_port)

    root = os.path.join(args.out, args.session)
    img_dir = os.path.join(root, "images")
    dep_dir = os.path.join(root, "depth")            # 原始 16bit 深度（mm）
    dep_prev_dir = os.path.join(root, "depth_preview")  # 8bit 预览（仅供肉眼查看）
    lab_dir = os.path.join(root, "labels")
    for d in (img_dir, dep_dir, lab_dir):
        os.makedirs(d, exist_ok=True)

    saved, skipped, failed = 0, 0, 0
    prev_small = None
    t0 = time.time()
    print("采集目标 {0} 张 -> {1}".format(args.n, root))
    print("彩色源 {0}".format(color_url))

    while saved < args.n:
        try:
            raw = fetch(color_url)
            arr = np.frombuffer(raw, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is None:
                raise RuntimeError("解码失败")
        except Exception as e:
            failed += 1
            print("  [warn] 取图失败: {0}".format(e))
            time.sleep(1.0)
            continue

        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (80, 60)).astype(np.float32)
        diff = 999.0 if prev_small is None else float(np.abs(small - prev_small).mean())
        if prev_small is not None and diff < args.diff_thres:
            skipped += 1
            time.sleep(args.interval)
            continue

        idx = saved + 1
        name = "{0}_{1:04d}".format(args.session, idx)
        cv2.imwrite(os.path.join(img_dir, name + ".jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        open(os.path.join(lab_dir, name + ".txt"), "w").close()   # 空标注，待标注/复核

        if not args.no_depth:
            try:
                drows = fetch(depth_url)
                darr = np.frombuffer(drows, np.uint8)
                dep = cv2.imdecode(darr, cv2.IMREAD_UNCHANGED)
                if dep is not None and dep.dtype == np.uint16:
                    # 存**原始 16bit**（mm）：测距与「深度自动提议框」都依赖它，
                    # 归一化会丢失真实距离，所以另存一份 8bit 预览仅供肉眼查看。
                    cv2.imwrite(os.path.join(dep_dir, name + ".png"), dep)
                    valid = dep[dep > 0]
                    lo, hi = (float(np.percentile(valid, 5)), float(np.percentile(valid, 95))) \
                        if valid.size else (0.0, 1.0)
                    norm = np.clip((dep.astype(np.float32) - lo) / max(1e-6, hi - lo), 0, 1)
                    norm[dep == 0] = 0
                    os.makedirs(dep_prev_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(dep_prev_dir, name + ".png"),
                                (norm * 255).astype(np.uint8))
            except Exception:
                pass

        prev_small = small
        saved += 1
        print("  [{0}/{1}] {2}  画面差 {3:.1f}".format(saved, args.n, name + ".jpg", diff))
        time.sleep(args.interval)

    if args.marked:
        try:
            data = fetch(marked_url)
            with open(os.path.join(root, "marked_example.jpg"), "wb") as f:
                f.write(data)
        except Exception:
            pass

    meta = {
        "session": args.session, "host": args.host,
        "color_url": color_url, "depth_url": None if args.no_depth else depth_url,
        "saved": saved, "skipped_same_scene": skipped, "failed": failed,
        "interval_s": args.interval, "diff_thres": args.diff_thres,
        "elapsed_s": round(time.time() - t0, 1),
        "note": "labels/*.txt 初始为空；纯背景采集（无货物）请保持为空以作负样本",
    }
    with open(os.path.join(root, "session.json"), "w") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print("完成：保存 {0} 张，跳过(画面未变) {1} 次，失败 {2} 次，用时 {3:.0f}s".format(
        saved, skipped, failed, time.time() - t0))
    print("元信息: {0}".format(os.path.join(root, "session.json")))


if __name__ == "__main__":
    main()
