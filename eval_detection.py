#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_detection.py —— 验证集评估: mAP/每类AP/小目标召回/混淆

- 主指标: ultralytics val(mAP50, mAP50-95, per-class AP)
- 小目标分析: 按 GT 框面积桶统计召回(手动 IoU 匹配, 无第三方依赖)
输出: reports/eval_report.json
用法: python eval_detection.py --config config/dataset.yaml --imgsz 640
"""
import argparse
import json
import os

import yaml


def area_bucket(w, h, imgsz):
    area = (w * imgsz) * (h * imgsz)
    if area < 32 * 32:
        return "tiny(<32px)"
    if area < 96 * 96:
        return "small(<96px)"
    return "medium(>=96px)"


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (aa + ab - inter + 1e-9)


def small_object_recall(gt_boxes, pred_boxes, imgsz):
    buckets = {"tiny(<32px)": [0, 0], "small(<96px)": [0, 0],
               "medium(>=96px)": [0, 0]}
    for g in gt_boxes:
        for (cx, cy, w, h) in g:
            b = area_bucket(w, h, imgsz)
            buckets[b][1] += 1
            hit = any(iou(p, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
                      >= 0.3 for p in pred_boxes)
            if hit:
                buckets[b][0] += 1
    return {k: {"total": v[1], "hit": v[0],
                "recall": round(v[0] / v[1], 3) if v[1] else None}
            for k, v in buckets.items()}


def parse_label(path):
    out = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    out.append(tuple(float(x) for x in parts[1:]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--weights", default="runs/detect/best_final.pt")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = os.path.abspath(os.path.expanduser(cfg["root"]))
    data_yaml = os.path.join(root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit("缺少 data.yaml")

    from ultralytics import YOLO
    import torch
    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.weights)
    metrics = model.val(data=data_yaml, imgsz=args.imgsz, device=device,
                        verbose=False)
    names = metrics.names if hasattr(metrics, "names") else cfg["detection_classes"]
    ap50 = metrics.box.ap50
    report = {
        "imgsz": args.imgsz,
        "mAP50": round(float(metrics.box.map50), 4),
        "mAP50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
        "per_class_AP50": {},
    }
    try:
        for c in range(len(names)):
            report["per_class_AP50"][str(names[c])] = round(float(ap50[c]), 4)
    except Exception:
        pass

    val_txt = os.path.join(root, "splits/val.txt")
    if os.path.exists(val_txt):
        gt_all, pred_all = [], []
        for line in open(val_txt):
            img = line.strip()
            stem = os.path.splitext(os.path.basename(img))[0]
            lbl = os.path.join(root, "labels", stem + ".txt")
            gt_all.append(parse_label(lbl))
            r = model.predict(img, imgsz=args.imgsz, conf=0.25,
                              device=device, verbose=False)[0]
            pred_all.append([list(b.xyxy[0].cpu().numpy()) for b in r.boxes])
        report["small_object"] = small_object_recall(gt_all, pred_all,
                                                     args.imgsz)

    os.makedirs("reports", exist_ok=True)
    with open("reports/eval_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
