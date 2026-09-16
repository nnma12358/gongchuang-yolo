#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_fp.py —— 误检率专项评估（mAP 之外的第二个关键指标）
=====================================================================
mAP 会把"背景误检"和"漏检"混在一起看；分拣现场更关心两件事：
  · **误检框/图**：每张画面平均多出几个"假货物"（会让机器人去抓空气）
  · **漏检率**：真实货物没被检出（会让货物分错筐）
本脚本在**人工核验**的验证集上直接量这两个数，并按置信度扫描给出推荐阈值。

参考：val 实拍图都有且只有 1 个人工框 → 与该框 IoU<0.3 的检出即误检。

用法：
  python tools/eval_fp.py --weights runs/best_after_review.pt --data data/mix_v3 \
      --data2 data/negatives            # 可选：纯背景图集，用来量"纯背景下误检"
"""
import argparse
import glob
import json
import os

import numpy as np


def load_yolo(p):
    out = []
    if os.path.exists(p):
        for line in open(p, encoding="utf-8"):
            v = line.split()
            if len(v) >= 5:
                out.append([float(x) for x in v[1:5]])
    return out


def to_xyxy(b):
    return [b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2]


def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description="误检率评估")
    ap.add_argument("--weights", default="runs/best_after_review.pt")
    ap.add_argument("--data", default="data/mix_v3", help="含人工核验 val 的数据集")
    ap.add_argument("--data2", default=None, help="纯背景图集（可选）")
    ap.add_argument("--split", default="val")
    ap.add_argument("--device", default="0")
    ap.add_argument("--confs", default="0.25,0.35,0.45,0.55,0.65")
    ap.add_argument("--fp-iou", type=float, default=0.3)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    imgs = sorted(glob.glob(os.path.join(args.data, "images", args.split, "real_*.*")))
    if not imgs:
        raise SystemExit("验证集里没有 real_* 实拍图：" + args.data)
    rows = []
    for p in imgs:
        stem = os.path.splitext(os.path.basename(p))[0]
        labels = [to_xyxy(b) for b in load_yolo(
            os.path.join(args.data, "labels", args.split, stem + ".txt"))]
        r = model.predict(p, imgsz=640, conf=0.10, device=args.device, verbose=False)[0]
        boxes = []
        if r.boxes is not None and len(r.boxes):
            H, W = r.orig_shape
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                bb = [float(b[0]) / W, float(b[1]) / H, float(b[2]) / W, float(b[3]) / H]
                boxes.append((float(c), max([iou_xyxy(bb, l) for l in labels] or [0.0])))
        rows.append({"img": stem, "n_gt": len(labels), "boxes": boxes})

    # 纯背景图（可选）
    bg = []
    if args.data2:
        for p in sorted(glob.glob(os.path.join(args.data2, "images", "*.*")))[:200]:
            r = model.predict(p, imgsz=640, conf=0.10, device=args.device, verbose=False)[0]
            cs = [] if (r.boxes is None or not len(r.boxes)) else [float(v) for v in r.boxes.conf.cpu().numpy()]
            bg.append(cs)

    print("=" * 74)
    print(" 误检专项（%d 张人工核验实拍图，每张恰好 1 个真货物）" % len(imgs))
    print("=" * 74)
    print(" %-8s %-10s %-10s %-12s %-12s" % ("conf", "命中/图", "误检/图", "漏检/图", "F1"))
    best = None
    for c in [float(x) for x in args.confs.split(",")]:
        tp = fp = fn = 0
        for r in rows:
            hit = sum(1 for conf, i in r["boxes"] if conf >= c and i >= args.fp_iou)
            extra = sum(1 for conf, i in r["boxes"] if conf >= c and i < args.fp_iou)
            tp += hit
            fp += extra
            fn += max(0, r["n_gt"] - hit)
        n = len(rows)
        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        row = {"conf": c, "tp_per_img": round(tp / n, 3), "fp_per_img": round(fp / n, 3),
               "miss_per_img": round(fn / n, 3), "P": round(prec, 4), "R": round(rec, 4),
               "F1": round(f1, 4)}
        rows_out = row
        print(" %-8.2f %-10.3f %-10.3f %-12.3f %-12.4f" % (
            c, row["tp_per_img"], row["fp_per_img"], row["miss_per_img"], row["F1"]))
        if best is None or f1 > best["F1"]:
            best = row
    print(" 推荐阈值（F1 最高）：conf=%.2f → 误检 %.2f 个/图 · 漏检 %.2f 个/图"
          % (best["conf"], best["fp_per_img"], best["miss_per_img"]))
    if bg:
        print("\n 纯背景图集（%d 张，理论上应 0 检出）：" % len(bg))
        for c in [float(x) for x in args.confs.split(",")]:
            tot = sum(1 for cs in bg for v in cs if v >= c)
            print("   conf=%.2f → %d 个检出，%.3f 个/图" % (c, tot, tot / len(bg)))
    print("=" * 74)
    if args.out:
        json.dump({"imgs": len(imgs), "best": best}, open(args.out, "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
