#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
make_clean_subset.py —— 从混合数据集中抽出**标签可靠**的子集
=====================================================================
依据 dataset_summary.json 里每张图的标签来源（human=人工标注 / auto-gen=生成图
白底分割自动标注 / auto-real=实拍自动预标注），只保留 human + auto-gen：

原因：实拍自动预标注（尤其是白底白物、低对比）存在噪声，直接把 191 张混在一起训练
会让模型在震荡中退化（实测 mAP50 从 0.55 崩到 0.001）。先用干净子集把管线与配方跑通，
再逐批复核 auto-real 标注并入，是更稳的路径。
"""
import glob
import json
import os
import random
import shutil

SRC = os.environ.get("SRC", "data/real_synth_mix")
DST = os.environ.get("DST", "data/mix_clean")
KEEP = ("human", "auto-gen")
VAL_RATIO = float(os.environ.get("VAL_RATIO", "0.18"))


def find_image(src, name):
    for split in ("train", "val", "unlabeled"):
        p = os.path.join(src, "images", split, name)
        if os.path.exists(p):
            return p
    return None


def find_label(src, name):
    stem = os.path.splitext(name)[0]
    for split in ("train", "val"):
        p = os.path.join(src, "labels", split, stem + ".txt")
        if os.path.exists(p):
            return p
    return None


def main():
    summary = json.load(open(os.path.join(SRC, "dataset_summary.json"), encoding="utf-8"))
    prov = summary.get("labels_source", {})
    keep = [(n, v) for n, v in prov.items() if v.get("source") in KEEP]
    print("[子集] 总标注 {0} 张 → 干净标签 {1} 张（来源 {2}）".format(len(prov), len(keep), KEEP))

    if os.path.isdir(DST):
        shutil.rmtree(DST)
    random.seed(3)
    random.shuffle(keep)
    n_val = max(1, int(len(keep) * VAL_RATIO))
    stats = {"train": 0, "val": 0, "skipped": 0}
    for i, (name, meta) in enumerate(keep):
        split = "val" if i < n_val else "train"
        ip, lp = find_image(SRC, name), find_label(SRC, name)
        if not ip or not lp:
            stats["skipped"] += 1
            continue
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(DST, sub, split), exist_ok=True)
        shutil.copy2(ip, os.path.join(DST, "images", split, name))
        shutil.copy2(lp, os.path.join(DST, "labels", split, os.path.splitext(name)[0] + ".txt"))
        stats[split] += 1
    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("path: {0}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: goods\n".format(
            os.path.abspath(DST)))
    with open(os.path.join(DST, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("goods\n")
    json.dump({"source": os.path.abspath(SRC), "kept_sources": KEEP, "stats": stats,
               "images": len(glob.glob(os.path.join(DST, "images", "*", "*")))},
              open(os.path.join(DST, "subset_summary.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("[子集] 输出 {0} | train {1} / val {2} | 跳过 {3}".format(
        os.path.abspath(DST), stats["train"], stats["val"], stats["skipped"]))


if __name__ == "__main__":
    main()
