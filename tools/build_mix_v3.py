#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mix_v3.py —— 面向真实域的检测数据集（含真实域验证集）
=====================================================================
问题（实测）：在"干净子集"上训出的检测器合成验证集 mAP50 0.995，但拿到 175 张实拍图上
推理时最高置信仅 0.266、conf≥0.35 命中 0 张 —— 典型的**域差异（sim-to-real gap）**。
用合成域验证集评估会给出虚高的结论，必须建立**真实域验证集**。

本脚本组装 data/mix_v3：
  · val（真实域优先）：实拍图（CV 几何标注 / 人工标注）占多数 + 少量生成图
  · train：人工标注 18 + 生成图 + 实拍 CV 标注（可按可信度筛选）
  · 若提供 --pseudo-dir（模型低阈值伪标注），把无 CV 标注的实拍图补进来（trust=medium）
输出 data.yaml / classes.txt / subset_summary.json（记录每张图来源与标签级别）
"""
import argparse
import glob
import json
import os
import random
import shutil

import cv2


def read_label(path):
    lines = []
    if os.path.exists(path):
        for line in open(path):
            p = line.split()
            if len(p) == 5:
                lines.append("0 {0} {1} {2} {3}".format(*p[1:]))
    return lines


def sanity(lines, W, H, min_area=0.004, max_area=0.30, max_aspect=2.4):
    """几何合理性过滤：面积比 / 长宽比 / 越界"""
    ok = []
    for line in lines:
        _, cx, cy, w, h = line.split()
        cx, cy, w, h = float(cx), float(cy), float(w), float(h)
        area = w * h
        if not (min_area <= area <= max_area):
            continue
        if w <= 0 or h <= 0:
            continue
        aspect = max(w, h) / max(1e-6, min(w, h))
        if aspect > max_aspect:
            continue
        if cx - w / 2 < -0.01 or cy - h / 2 < -0.01 or cx + w / 2 > 1.01 or cy + h / 2 > 1.01:
            continue
        ok.append(line)
    return ok


def main():
    ap = argparse.ArgumentParser(description="构建面向真实域的检测数据集")
    ap.add_argument("--mixed", default="data/real_synth_mix", help="混合集（含 dataset_summary.json）")
    ap.add_argument("--pseudo-dir", default="data/pseudo_real", help="模型伪标注目录（可选）")
    ap.add_argument("--out", default="data/mix_v3")
    ap.add_argument("--real-val", type=int, default=30, help="真实域验证集张数")
    ap.add_argument("--gen-val", type=int, default=10, help="验证集中的生成图张数")
    ap.add_argument("--min-solidity-note", action="store_true", help="仅保留 review 未标记的实拍标注")
    args = ap.parse_args()

    summary = json.load(open(os.path.join(args.mixed, "dataset_summary.json"), encoding="utf-8"))
    prov = summary["labels_source"]
    review_bad = set()
    for r in summary.get("review", []):
        review_bad.add(r["image"])

    def find_img(name):
        for split in ("train", "val", "unlabeled"):
            p = os.path.join(args.mixed, "images", split, name)
            if os.path.exists(p):
                return p
        return None

    def find_lbl(name):
        stem = os.path.splitext(name)[0]
        for split in ("train", "val"):
            p = os.path.join(args.mixed, "labels", split, stem + ".txt")
            if os.path.exists(p):
                return p
        return None

    human, gen, real_cv = [], [], []
    for name, meta in prov.items():
        if meta["source"] == "human":
            human.append(name)
        elif meta["source"] == "auto-gen":
            gen.append(name)
        elif meta["source"] == "auto-real":
            real_cv.append(name)

    # 伪标注（模型低阈值）补充无标注实拍
    pseudo = {}
    pd = os.path.join(args.pseudo_dir, "provenance.json")
    if os.path.exists(pd):
        pseudo = json.load(open(pd, encoding="utf-8")).get("images", {})

    random.seed(11)
    random.shuffle(real_cv)
    val_real = real_cv[:args.real_val]
    train_real = real_cv[args.real_val:]
    random.shuffle(gen)
    val_gen, train_gen = gen[:args.gen_val], gen[args.gen_val:]

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    stats = {"train": {"real_human": 0, "real_cv": 0, "gen": 0, "pseudo": 0, "dropped": 0},
             "val": {"real": 0, "gen": 0, "dropped": 0}}
    index = {}

    def put(name, split, kind, src_img, src_lbl=None, label_lines=None):  # noqa: C901
        img = src_img
        im = cv2.imread(img)
        if im is None:
            return False
        H, W = im.shape[:2]
        lines = label_lines if label_lines is not None else read_label(src_lbl)
        max_area = 0.80 if kind == "gen" else 0.30
        lines = sanity(lines, W, H, max_area=max_area)
        if not lines:
            stats[split]["dropped"] += 1
            return False
        os.makedirs(os.path.join(args.out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", split), exist_ok=True)
        dst_img = os.path.join(args.out, "images", split, name)
        if max(H, W) > 1280:
            sc = 1280 / float(max(H, W))
            im = cv2.resize(im, (int(W * sc), int(H * sc)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst_img, im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            shutil.copy2(img, dst_img)
        with open(os.path.join(args.out, "labels", split, os.path.splitext(name)[0] + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        stats[split][kind] = stats[split].get(kind, 0) + 1
        index[name] = {"split": split, "kind": kind, "boxes": len(lines)}
        return True

    for n in human:
        put(n, "train", "real_human", find_img(n), find_lbl(n))
    for n in train_real:
        put(n, "train", "real_cv", find_img(n), find_lbl(n))
    for n in train_gen:
        put(n, "train", "gen", find_img(n), find_lbl(n))
    for n in val_real:
        put(n, "val", "real", find_img(n), find_lbl(n))
    for n in val_gen:
        put(n, "val", "gen", find_img(n), find_lbl(n))
    # 伪标注补充（仅当该图未进入任何集合）
    done = set(index.keys())
    pi = os.path.join(args.pseudo_dir, "images")
    for p in sorted(glob.glob(os.path.join(pi, "*.*")) if os.path.isdir(pi) else []):
        name = os.path.basename(p)
        if name in done:
            continue
        lp = os.path.join(args.pseudo_dir, "labels", os.path.splitext(name)[0] + ".txt")
        put(name, "train", "pseudo", p, lp)

    with open(os.path.join(args.out, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: {0}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: goods\n".format(
            os.path.abspath(args.out)))
    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("goods\n")
    with open(os.path.join(args.out, "subset_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "index": index, "note":
                   "val 以真实域（实拍）为主，用于诚实评估跨域泛化；train 混入生成图补充多样性"},
                  f, ensure_ascii=False, indent=2)
    print("[mix_v3] train:", stats["train"])
    print("[mix_v3] val  :", stats["val"])
    print("[mix_v3] 输出 :", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
