#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用「跨帧共识」把自动提议的草稿变成高置信标注（静止场景专用）

⚠️ **负面结论（2026-09-24 实测，先读这段）**：本思路**在当前场景不成立**，不要直接使用。
   假设是"真货物每帧都在、伪框只零星出现"，但实测：**静止场景下伪框同样是持久的**
   （桌沿、地面、深度断裂处的伪影每帧都出现），而货物框因深度抖动在帧间位置不稳、
   反而聚不成高出现率的簇。结果：33 个簇里筛出的 8 个"高出现率"框**大多落在桌面/地面**，
   真实货物基本没被保留 —— 比不做共识还差。

   → 现场俯拍标注目前**没有可靠的自动化路径**（已试三种：双通道直提议、分水岭拆分、
     跨帧共识）。请用 review_app 人工标注；propose_boxes --bands --no-rgb 的候选
     （平均 12.3 框/帧、高度 18~39mm）可作为"删框/补框"的起点，比从零画快。

背景：现场俯拍是**静止场景** —— 货物摆好后不动，12 帧里真货物位置一致；
而伪框（桌沿、地面、深度噪声）只会零星出现在少数帧。于是：

    1) 每帧跑深度高度带通提议（propose_boxes.propose_depth_bands）
    2) 把所有帧的框做跨帧聚类（中心距 + 尺寸相近）
    3) **出现率 ≥ min-ratio 的簇 = 真货物**，其余丢弃
    4) 把这些簇的框写回每一帧 → 得到一套一致的标注

这比"逐帧独立过滤"稳得多：单帧里分不清的桌沿伪框，跨帧看只出现 2/12 次，
真货物出现 12/12 次，一眼可分。

**注意：产出是"自动共识标注"，仍需人工复核确认**
（尤其：深色货物在深度图上常无回波，需要人工补框）。

用法：
    python3 tools/consensus_labels.py --session tray_full_20260924 \
        --roi 0,140,470,470 --min-ratio 0.6 --write
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from propose_boxes import propose_depth_bands  # noqa: E402


def load_depth(path):
    import cv2
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return d if (d is not None and d.dtype == np.uint16) else None


def cluster(all_boxes, dist=14.0, size_tol=0.45):
    """跨帧聚类：中心距 < dist 且 尺寸相近 → 同一簇"""
    clusters = []
    for f, boxes in all_boxes:
        for b in boxes:
            cx, cy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
            w, h = b[2] - b[0], b[3] - b[1]
            hit = None
            for c in clusters:
                ccx, ccy = (c["box"][0] + c["box"][2]) / 2.0, (c["box"][1] + c["box"][3]) / 2.0
                cw, ch = c["box"][2] - c["box"][0], c["box"][3] - c["box"][1]
                if (abs(cx - ccx) <= dist and abs(cy - ccy) <= dist and
                        abs(w - cw) <= size_tol * max(w, cw, 1) and
                        abs(h - ch) <= size_tol * max(h, ch, 1)):
                    hit = c
                    break
            if hit is None:
                clusters.append({"box": list(b), "frames": [f], "prot": [b[4]]})
            else:
                hit["frames"].append(f)
                hit["prot"].append(b[4])
                # 用出现过的中位位置代表该簇
                xs = [hit["box"][0], b[0]]
                ys = [hit["box"][1], b[1]]
                xe = [hit["box"][2], b[2]]
                ye = [hit["box"][3], b[3]]
                hit["box"] = [float(np.median(xs)), float(np.median(ys)),
                              float(np.median(xe)), float(np.median(ye))]
    return clusters


def main():
    ap = argparse.ArgumentParser(description="跨帧共识标注（静止场景）")
    ap.add_argument("--root", default="data/real_overhead")
    ap.add_argument("--session", required=True)
    ap.add_argument("--roi", default="0,140,470,470")
    ap.add_argument("--band-lo", type=float, default=12.0)
    ap.add_argument("--band-hi", type=float, default=55.0)
    ap.add_argument("--min-ratio", type=float, default=0.6,
                    help="簇至少要在多少比例的帧里出现才算真货物")
    ap.add_argument("--max-side", type=int, default=78)
    ap.add_argument("--table-band", type=float, default=120.0,
                    help="框内中值深度必须落在桌面 ±该值内（排除地面/远景）")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    import cv2
    sdir = os.path.join(args.root, args.session)
    imgs = sorted(glob.glob(os.path.join(sdir, "images", "*")))
    if not imgs:
        raise SystemExit("没有图片: {0}".format(sdir))
    roi = tuple(int(v) for v in args.roi.split(","))

    per_frame, table_ref = [], []
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        dep = load_depth(os.path.join(sdir, "depth", name + ".png"))
        img = cv2.imread(p)
        boxes = propose_depth_bands(dep, img.shape, roi=roi, lo=args.band_lo,
                                    hi=args.band_hi, max_side=args.max_side)
        # 桌面深度参考：ROI 内有效深度中位数
        if dep is not None:
            sub = dep[roi[1]:roi[3], roi[0]:roi[2]]
            v = sub[sub > 0]
            if v.size:
                table_ref.append(float(np.median(v)))
        per_frame.append((name, boxes, dep))
    table = float(np.median(table_ref)) if table_ref else None
    print("  桌面深度参考: {0}".format("None" if table is None else "{0:.0f}mm".format(table)))

    # 用桌面深度带把"地面/远景"框去掉
    def near_table(b):
        return True
        # (在下面按帧过滤，这里保留接口)

    kept = []
    for name, boxes, dep in per_frame:
        if dep is None or table is None:
            kept.append((name, boxes))
            continue
        good = []
        for b in boxes:
            xs, ys = int(b[0]), int(b[1])
            xe, ye = int(b[2]), int(b[3])
            sub = dep[max(0, ys):ye, max(0, xs):xe]
            v = sub[sub > 0]
            if v.size == 0:
                good.append(b)          # 无深度信息（深色货物）→ 保留
                continue
            med = float(np.median(v))
            if abs(med - table) <= args.table_band:
                good.append(b)
        kept.append((name, good))

    clusters = cluster([(n, b) for n, b, _ in [(n, bx, None) for n, bx in kept]])
    n_frames = len(kept)
    good_clusters = [c for c in clusters if len(set(c["frames"])) >= max(1, int(args.min_ratio * n_frames))]
    good_clusters.sort(key=lambda c: -len(set(c["frames"])))
    print("  聚类 {0} 个，出现率 ≥{1:.0%} 的 {2} 个（视为真货物）".format(
        len(clusters), args.min_ratio, len(good_clusters)))

    out_dir = os.path.join(sdir, "consensus")
    os.makedirs(out_dir, exist_ok=True)
    summary = {}
    for name, boxes, dep in per_frame:
        img = cv2.imread(os.path.join(sdir, "images", name + ".jpg"))
        H, W = img.shape[:2]
        lines = []
        for c in good_clusters:
            x1, y1, x2, y2 = c["box"]
            cx, cy = (x1 + x2) / 2.0 / W, (y1 + y2) / 2.0 / H
            bw, bh = (x2 - x1) / float(W), (y2 - y1) / float(H)
            lines.append("0 {0:.6f} {1:.6f} {2:.6f} {3:.6f}".format(cx, cy, bw, bh))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
        summary[name] = {"n": len(lines)}
        if args.write:
            with open(os.path.join(sdir, "labels", name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
            with open(os.path.join(out_dir, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(os.path.join(out_dir, name + "_vis.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    json.dump({"frames": n_frames, "clusters": len(clusters),
               "kept": len(good_clusters), "min_ratio": args.min_ratio,
               "per_frame": summary},
              open(os.path.join(out_dir, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print("  写入 {0} 张图的共识标注，每帧 {1} 个框".format(
        n_frames, len(good_clusters)))
    print("  预览: {0}/*_vis.jpg".format(out_dir))
    print("  **仍需人工复核**（深色货物在深度图上常无回波，需人工补框）" +
          ("" if args.write else "；加 --write 才写入 labels/"))


if __name__ == "__main__":
    main()
