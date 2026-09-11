#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
crop_attributes.py —— 从检测标注自动生成属性分类数据集（颜色 / 形状 / 污渍缺陷）

为什么需要：赛项要求“形状相同且颜色相同 → 同一储物盒”，属性判断错一次就丢一件；
属性分类器比检测直出组合类别更稳（决赛货物种类以现场为准，属性维度不变）。
本脚本用已有的检测标注（labels/*.txt）自动裁出货物小图，按文件名/类别约定生成
attributes/{color,shape,stain}/<class>/ —— 无需重复标注。

类别命名约定（detection_classes 或图片文件名）：
  red_cube.jpg          → color=red      shape=cube
  prism5_4_stain.jpg    → shape=prism5   stain=stain     （后缀 _stain / _defect）
  anyname_defect.jpg    → stain=defect

用法：
  # 方案A：类别名直接编码颜色/形状（推荐）
  python crop_attributes.py --config config/sorting_competition.yaml \
      --images data/images/train --labels data/labels/train

  # 方案B：用 <root>/attributes/<task> 已手工分好的目录补充（不覆盖已有文件）
  python crop_attributes.py --config config/sorting_competition.yaml --merge-existing
"""
import argparse
import glob
import os
import random
import shutil

import cv2
import yaml

COLOR_ALIASES = {
    "red": "red", "orange": "orange", "yellow": "yellow", "green": "green",
    "cyan": "cyan", "lightblue": "cyan", "blue": "blue", "purple": "purple",
    "black": "black", "white": "white",
}
SHAPE_ALIASES = {
    "cube": "cube", "cuboid": "cuboid", "box": "cube", "cylinder": "cylinder",
    "ball": "ball", "sphere": "ball", "tetra": "tetra", "pyramid": "tetra",
    "prism5": "prism5", "prism6": "prism6",
}
STAIN_SUFFIX = {"stain": "stain", "dirty": "stain", "defect": "defect", "broken": "defect"}


def parse_meta(stem, class_name):
    """从文件名或类别名解析 (color, shape, stain)"""
    text = "%s_%s" % (stem, class_name or "")
    text = text.lower().replace("-", "_")
    color = shape = stain = None
    for key, val in COLOR_ALIASES.items():
        if key in text:
            color = val
            break
    for key, val in SHAPE_ALIASES.items():
        if key in text:
            shape = val
            break
    for key, val in STAIN_SUFFIX.items():
        if key in text:
            stain = val
            break
    return color, shape, stain or "clean"


def read_yolo_labels(label_path, w, h):
    """YOLO txt → [(cls, x1, y1, x2, y2)] 像素坐标"""
    boxes = []
    if not os.path.exists(label_path):
        return boxes
    with open(label_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 5:
                continue
            _, cx, cy, bw, bh = parts[:5]
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append((int(cx - bw / 2), int(cy - bh / 2), int(cx + bw / 2), int(cy + bh / 2)))
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sorting_competition.yaml")
    ap.add_argument("--images", default=None, help="图片目录（默认 <root>/images/train）")
    ap.add_argument("--labels", default=None, help="标注目录（默认 <root>/labels/train）")
    ap.add_argument("--margin", type=float, default=None, help="裁剪外扩比例")
    ap.add_argument("--size", type=int, default=96, help="输出小图尺寸")
    ap.add_argument("--val-ratio", type=float, default=0.15, help="预留验证比例（写入 _val 子目录）")
    ap.add_argument("--merge-existing", action="store_true", help="保留已有属性图（不清理输出目录）")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = os.path.abspath(os.path.expanduser(cfg["root"]))
    if root == os.path.abspath("/path/to/my_sorting_data"):
        raise SystemExit("请先修改 config 的 root 为你的数据集路径")

    images_dir = args.images or os.path.join(root, "images", "train")
    labels_dir = args.labels or os.path.join(root, "labels", "train")
    margin = args.margin if args.margin is not None else float(cfg.get("attr_crop_margin", 0.12))
    size = int(cfg.get("attr_img_size", args.size))

    tasks = {"color": cfg.get("color_classes", []), "shape": cfg.get("shape_classes", []),
             "stain": cfg.get("stain_classes", ["clean", "stain", "defect"])}
    out_root = os.path.join(root, "attributes")
    for task, classes in tasks.items():
        for cls in classes:
            os.makedirs(os.path.join(out_root, task, cls), exist_ok=True)

    # 类别名映射（方案B：检测类别自带颜色/形状）
    class_names = cfg.get("detection_classes", ["goods"])
    if os.path.exists(os.path.join(root, "data.yaml")):
        with open(os.path.join(root, "data.yaml")) as f:
            dy = yaml.safe_load(f) or {}
        class_names = dy.get("names") or class_names
        if isinstance(class_names, dict):
            class_names = [class_names[k] for k in sorted(class_names)]

    files = sorted(glob.glob(os.path.join(images_dir, "*.jpg")) +
                   glob.glob(os.path.join(images_dir, "*.png")))
    print("[输入] 图片 %d 张  来源 %s" % (len(files), images_dir))
    counts = {task: {} for task in tasks}
    written = 0

    for img_path in files:
        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        stem = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        boxes = read_yolo_labels(label_path, w, h)
        if not boxes:                                   # 无标注：整图当作一件货物
            boxes = [(0, 0, w - 1, h - 1)]

        for i, (x1, y1, x2, y2) in enumerate(boxes):
            pad = int(max(x2 - x1, y2 - y1) * margin)
            x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
            x2, y2 = min(w - 1, x2 + pad), min(h - 1, y2 + pad)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (size, size))

            cls_name = class_names[0] if class_names else ""
            color, shape, stain = parse_meta(stem, cls_name)
            split = "_val" if random.random() < args.val_ratio else ""
            for task, value in (("color", color), ("shape", shape), ("stain", stain)):
                if not value or value not in tasks[task]:
                    continue
                dst_dir = os.path.join(out_root, task, value + split)
                os.makedirs(dst_dir, exist_ok=True)
                dst = os.path.join(dst_dir, "%s_%02d.jpg" % (stem, i))
                cv2.imwrite(dst, crop)
                counts[task][value] = counts[task].get(value, 0) + 1
                written += 1

    print("[输出] 属性小图 %d 张 → %s" % (written, out_root))
    for task, c in counts.items():
        print("  %-6s %s" % (task, c))
    print("\n提示：属性分类训练 → python train_attributes.py --config %s --task color|shape|stain" % args.config)
    if written == 0:
        print("⚠ 未生成任何小图：请检查文件名/检测类别是否包含颜色与形状关键字（如 red_cube、prism5_4）")


if __name__ == "__main__":
    main()
