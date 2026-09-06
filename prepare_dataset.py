#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
prepare_dataset.py —— 数据集校验/统计/划分/生成 data.yaml

- 校验 images/ 与 labels/ 一一对应、标签格式(cls cx cy w h, 0<=cx<=1)
- 统计每类样本数、框面积分布(小目标提示)
- 按 train_ratio 划分, 生成 splits/train.txt val.txt
- 生成 ultralytics 直接可用的 data.yaml(root 下)
用法: python prepare_dataset.py --config config/dataset.yaml
"""
import argparse
import glob
import os
import random

import yaml

random.seed(42)


def parse_labels(label_path, n_cls):
    boxes = []
    with open(label_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 5:
                raise ValueError("%s 非法行: %s" % (label_path, line))
            cls = int(float(parts[0]))
            cx, cy, w, h = [float(x) for x in parts[1:]]
            if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1):
                print("[WARN] %s 框越界: %s" % (label_path, line))
            if cls >= n_cls:
                raise ValueError("%s 类别越界: %d nc=%d" % (label_path, cls, n_cls))
            boxes.append((cls, cx, cy, w, h))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/dataset.yaml")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = os.path.abspath(os.path.expanduser(cfg["root"]))
    if root == os.path.abspath("/path/to/my_sorting_data"):
        raise SystemExit("请先修改 config/dataset.yaml 的 root 为你的数据集路径")
    img_dir = os.path.join(root, cfg.get("images", "images"))
    lbl_dir = os.path.join(root, cfg.get("labels", "labels"))
    classes = cfg["detection_classes"]
    n_cls = len(classes)

    imgs = sorted(glob.glob(os.path.join(img_dir, "*.jpg")) +
                  glob.glob(os.path.join(img_dir, "*.png")) +
                  glob.glob(os.path.join(img_dir, "*.jpeg")))
    print("[1/4] 图像数量: %d" % len(imgs))
    if len(imgs) < 50:
        print("[WARN] 图像过少(<50), 建议 ≥800 张(多光照/色差/密排覆盖)")

    samples, cls_count = [], [0] * n_cls
    bucket = {"tiny(<32px)": 0, "small(<96px)": 0, "medium": 0}
    for img in imgs:
        stem = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(lbl_dir, stem + ".txt")
        if not os.path.exists(lbl):
            print("[WARN] 缺少标注: %s" % lbl)
            continue
        boxes = parse_labels(lbl, n_cls)
        if not boxes:
            continue
        for cls, cx, cy, w, h in boxes:
            cls_count[cls] += 1
            area = w * h
            if area < (32 / 640) ** 2:
                bucket["tiny(<32px)"] += 1
            elif area < (96 / 640) ** 2:
                bucket["small(<96px)"] += 1
            else:
                bucket["medium"] += 1
        samples.append((img, lbl, boxes))

    print("[2/4] 有效样本: %d" % len(samples))
    for i, c in enumerate(classes):
        print("      %-20s %d 个框" % (c, cls_count[i]))
    print("      框面积分布(按640基准): %s" % bucket)
    if bucket["tiny(<32px)"] > len(samples):
        print("[HINT] 小目标占比高: 建议 imgsz 用 960 或 2×2 tile")

    random.shuffle(samples)
    n_train = max(1, int(len(samples) * cfg.get("train_ratio", 0.85)))
    train_set, val_set = samples[:n_train], samples[n_train:]
    print("[3/4] train/val = %d/%d" % (len(train_set), len(val_set)))

    out_dir = os.path.join(root, "splits")
    os.makedirs(out_dir, exist_ok=True)
    for name, data in (("train.txt", train_set), ("val.txt", val_set)):
        with open(os.path.join(out_dir, name), "w") as f:
            for img, _, _ in data:
                f.write("%s\n" % img)

    with open(os.path.join(root, "data.yaml"), "w") as f:
        yaml.safe_dump({
            "path": root,
            "train": ["splits/train.txt"],
            "val": ["splits/val.txt"],
            "nc": n_cls,
            "names": classes,
        }, f, allow_unicode=True, sort_keys=False)
    print("[4/4] 生成 data.yaml: %s" % os.path.join(root, "data.yaml"))
    print("下一步: python train_detection.py --config %s" % args.config)


if __name__ == "__main__":
    main()
