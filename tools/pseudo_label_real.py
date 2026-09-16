#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pseudo_label_real.py —— 用已训练检测器给实拍图做伪标注，扩充检测数据集
=====================================================================
背景（第 1 项优化）：混合集里 146 张实拍图是"几何自动预标注"（白底白物低对比，噪声大），
直接训练会发散。现在已有在干净子集上训到 mAP50 0.995 的检测器，可用它做**模型伪标注**，
并与几何预标注**交叉验证**：两者一致（IoU 高）→ 高可信；仅模型给出 → 中可信。

流程：
  1) 用检测器对每张实拍图推理（conf 0.35，imgsz 640）
  2) 候选过滤：面积比 0.4%~28%、长宽比 ≤2.2、必须落在画面内
  3) 与几何预标注框比 IoU：
        IoU ≥ 0.35 → trust=high（模型+几何一致）
        0    < IoU < 0.35 → trust=medium（仅模型，几何框不同）
     无模型检出 → 跳过（保留原引用，等待人工标注）
  4) 输出 data/pseudo_real/{images,labels} + provenance.json（每张图的来源与可信级别）

用法：
  python3 tools/pseudo_label_real.py --weights runs/detect/runs/detect/train/weights/best.pt \
      --real-dir /home/xxxffyy/工创/图 --labeled-dir /home/xxxffyy/工创/图/X-AnyLabeling \
      --out data/pseudo_real --conf 0.35
"""
import argparse
import glob
import json
import os
import shutil

import numpy as np


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-9, ua)


def yolo_to_box(line, W, H):
    p = line.split()
    if len(p) != 5:
        return None
    cx, cy, w, h = [float(v) for v in p[1:5]]
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def box_to_yolo(box, W, H):
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(W), x2), min(float(H), y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return "0 {0:.6f} {1:.6f} {2:.6f} {3:.6f}".format(
        (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def geometric_boxes(image_path, src_labels_dir):
    """读取已有的几何自动预标注框（若存在）"""
    stem = os.path.splitext(os.path.basename(image_path))[0]
    for d in (src_labels_dir,):
        p = os.path.join(d, stem + ".txt")
        if os.path.exists(p):
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return []
            H, W = img.shape[:2]
            out = []
            for line in open(p):
                b = yolo_to_box(line, W, H)
                if b:
                    out.append(b)
            return out
    return []


def main():
    ap = argparse.ArgumentParser(description="模型伪标注扩充实拍检测数据")
    ap.add_argument("--weights", required=True, help="已训练的检测器 .pt")
    ap.add_argument("--real-dir", required=True, help="实拍图目录（*.jpg）")
    ap.add_argument("--labeled-dir", help="人工标注目录（X-AnyLabeling，含 images/ labels/）")
    ap.add_argument("--geo-labels", default=None, help="几何预标注目录（用于交叉验证；默认取自混合集）")
    ap.add_argument("--out", default="data/pseudo_real")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--iou-high", type=float, default=0.35)
    ap.add_argument("--exclude-labeled", action="store_true", default=True,
                    help="跳过已有人工标注的图片（不覆盖人工标签）")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    human_stems = set()
    if args.labeled_dir:
        for p in glob.glob(os.path.join(args.labeled_dir, "images", "*.*")):
            human_stems.add(os.path.splitext(os.path.basename(p))[0])

    images = sorted(glob.glob(os.path.join(args.real_dir, "*.jpg")) +
                    glob.glob(os.path.join(args.real_dir, "*.jpeg")))
    print("[输入] 实拍图 {0} 张 | 已人工标注 {1} 张（默认跳过）".format(len(images), len(human_stems)))

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for sub in ("images", "labels"):
        os.makedirs(os.path.join(args.out, sub), exist_ok=True)

    prov, stats = {}, {"high": 0, "medium": 0, "skipped_no_det": 0, "skipped_human": 0}
    for p in images:
        stem = os.path.splitext(os.path.basename(p))[0]
        if args.exclude_labeled and stem in human_stems:
            stats["skipped_human"] += 1
            continue
        import cv2
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        r = model.predict(p, imgsz=args.imgsz, conf=args.conf, verbose=False)[0]
        cands = []
        for b in r.boxes:
            x1, y1, x2, y2 = [float(v) for v in b.xyxy[0].tolist()]
            conf = float(b.conf[0])
            area_ratio = ((x2 - x1) * (y2 - y1)) / float(W * H)
            aspect = max(x2 - x1, y2 - y1) / max(1.0, min(x2 - x1, y2 - y1))
            if not (0.004 <= area_ratio <= 0.28) or aspect > 2.2:
                continue
            cands.append({"box": [x1, y1, x2, y2], "conf": conf, "area_ratio": round(area_ratio, 4)})
        if not cands:
            stats["skipped_no_det"] += 1
            prov[os.path.basename(p)] = {"trust": "none", "reason": "检测器无输出（保留待人工标注）"}
            continue
        cands.sort(key=lambda d: -d["conf"])
        best = cands[0]
        geo = geometric_boxes(p, args.geo_labels) if args.geo_labels else []
        agree = max([iou(best["box"], g) for g in geo], default=0.0)
        trust = "high" if agree >= args.iou_high else "medium"
        line = box_to_yolo(best["box"], W, H)
        if not line:
            continue
        shutil.copy2(p, os.path.join(args.out, "images", os.path.basename(p)))
        with open(os.path.join(args.out, "labels", stem + ".txt"), "w", encoding="utf-8") as f:
            f.write(line + "\n")
        prov[os.path.basename(p)] = {"trust": trust, "conf": round(best["conf"], 3),
                                     "iou_vs_geometric": round(agree, 3),
                                     "box": [round(v, 1) for v in best["box"]],
                                     "n_candidates": len(cands)}
        stats[trust] += 1

    with open(os.path.join(args.out, "provenance.json"), "w", encoding="utf-8") as f:
        json.dump({"weights": args.weights, "conf": args.conf, "imgsz": args.imgsz,
                   "stats": stats, "images": prov}, f, ensure_ascii=False, indent=2)
    print("[伪标注] high(模型+几何一致) {high} | medium(仅模型) {medium} | "
          "无检出 {skipped_no_det} | 跳过人工 {skipped_human}".format(**stats))
    print("[输出] {0}  （provenance.json 记录每张图的可信级别）".format(os.path.abspath(args.out)))


if __name__ == "__main__":
    main()
