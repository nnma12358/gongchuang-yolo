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


def dhash(path, size=16):
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()


def color_sig(path):
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    hsv = cv2.cvtColor(cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def scene_groups(names, find_img, prov, visual_thr=20, color_thr=0.35, gap_s=None):
    """近重复图分组（**完全连接**聚类，杜绝链式合并）。

    为什么不按时间分组：本数据的文件名时间戳是导出时生成的（相邻帧只差 7ms），
    实测 164 张按 30s 间隔分只有 3 组 —— 完全不可用。
    可靠信号只有**视觉**：dHash(16×16=256bit) ≤ visual_thr 且 HSV 直方图 L1 ≤ color_thr。

    为什么必须"完全连接"：普通并查集是单连接，A≈B、B≈C 会把 A/B/C 并成一组，
    实测阈值 24 时 164 张被并成 1 组。完全连接要求合并后**组内任意两张**都满足阈值，
    实测 164 张 → 149 组（15 对真正的重复图），不再链式膨胀。

    gap_s 给定时额外按时间合并（本项目数据不建议用）。
    """
    import numpy as np
    hs, cs, ok = {}, {}, []
    for n in names:
        p = find_img(n)
        if not p:
            continue
        h, c = dhash(p), color_sig(p)
        if h is None or c is None:
            continue
        hs[n], cs[n] = h, c
        ok.append(n)
    idx = {n: i for i, n in enumerate(ok)}
    m = len(ok)
    if m == 0:
        return {n: n for n in names}
    pairs = []
    if visual_thr:
        for i in range(m):
            for j in range(i + 1, m):
                a, b = ok[i], ok[j]
                d = int(np.count_nonzero(hs[a] != hs[b]))
                if d > visual_thr:
                    continue
                cd = float(np.abs(cs[a] - cs[b]).sum())
                if cd <= color_thr:
                    pairs.append((d, a, b))
    pairs.sort()
    parent = {n: n for n in ok}
    members = {n: {n} for n in ok}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for d, a, b in pairs:
        ra, rb = find(a), find(b)
        if ra == rb:
            continue
        A, B = members[ra], members[rb]
        # 完全连接：合并后组内任意两张的距离都必须 ≤ 阈值
        if all(int(np.count_nonzero(hs[x] != hs[y])) <= visual_thr and
               float(np.abs(cs[x] - cs[y]).sum()) <= color_thr for x in A for y in B):
            parent[rb] = ra
            members[ra] = A | B
            del members[rb]
    if gap_s:                      # 可选：再按时间合并（本项目数据不用）
        import re
        def tsf(n):
            f = (prov.get(n) or {}).get("from") or ""
            mm = re.search(r"(\d{13})", str(f)) or re.search(r"(\d{13})", n)
            return int(mm.group(1)) if mm else None
        ordered = sorted([n for n in ok if tsf(n)], key=tsf)
        for a, b in zip(ordered, ordered[1:]):
            if tsf(b) - tsf(a) <= gap_s * 1000:
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[rb] = ra
                    members[ra] = members[ra] | members[rb]
                    del members[rb]
    root = {n: find(n) for n in ok}
    return {n: root.get(n, n) for n in names}


def main():
    ap = argparse.ArgumentParser(description="构建面向真实域的检测数据集")
    ap.add_argument("--mixed", default="data/real_synth_mix", help="混合集（含 dataset_summary.json）")
    ap.add_argument("--pseudo-dir", default="data/pseudo_real", help="模型伪标注目录（可选）")
    ap.add_argument("--out", default="data/mix_v3")
    ap.add_argument("--real-val", type=int, default=40, help="真实域验证集目标张数（整组进，可能超出）")
    ap.add_argument("--gen-val", type=int, default=10, help="验证集中的生成图张数")
    ap.add_argument("--group-split", dest="group_split", action="store_true", default=True,
                    help="按场景分组切分（默认开，杜绝近重复图跨 train/val 泄漏）")
    ap.add_argument("--no-group-split", dest="group_split", action="store_false")
    ap.add_argument("--visual-thr", type=int, default=20,
                    help="近重复图判定：16×16 dHash 汉明距离阈值（完全连接聚类）")
    ap.add_argument("--group-gap", type=float, default=None,
                    help="额外按时间合并（本项目文件名时间戳不可靠，默认不用）")
    ap.add_argument("--prefer-human-val", dest="prefer_human_val", action="store_true", default=True,
                    help="val 优先选人工核验过的分组（指标才可信）")
    ap.add_argument("--no-prefer-human-val", dest="prefer_human_val", action="store_false")
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
    groups_info = {}
    if args.group_split:
        # ---- 按场景分组切分：整组进 train 或 val，杜绝近重复图泄漏 ----
        all_real = human + real_cv
        gid = scene_groups(all_real, find_img, prov, visual_thr=args.visual_thr,
                            gap_s=args.group_gap)
        buckets = {}
        for n in all_real:
            buckets.setdefault(gid[n], []).append(n)
        human_set = set(human)
        keys = list(buckets)
        random.shuffle(keys)
        # 优先把"含人工核验图"的组放进 val —— val 的标注必须可信
        if args.prefer_human_val:
            keys.sort(key=lambda g: (0 if human_set & set(buckets[g]) else 1, -len(buckets[g])))
        val_real, train_real = [], []
        # 贪心装 val：不超过目标张数，且不允许"一个巨型组把 train 挤空"
        cap = max(args.real_val, int(args.real_val * 1.8))
        val_real, train_real = [], []
        for g in keys:
            if len(val_real) < args.real_val and len(val_real) + len(buckets[g]) <= cap:
                val_real.extend(buckets[g])
            else:
                train_real.extend(buckets[g])
        if not val_real:                      # 兜底：最小的组进 val
            g = min(keys, key=lambda k: len(buckets[k]))
            val_real = list(buckets[g])
            train_real = [n for k in keys if k != g for n in buckets[k]]
        groups_info = {"n_groups": len(keys),
                       "sizes": sorted((len(v) for v in buckets.values()), reverse=True),
                       "val_groups": len({gid[n] for n in val_real}),
                       "train_groups": len({gid[n] for n in train_real}),
                       "val_has_human": len([n for n in val_real if n in human_set])}
        print("[mix_v3] 分组切分：%d 组（大小 %s）→ val %d 张 / train %d 张，val 中人工核验 %d 张"
              % (len(keys), groups_info["sizes"][:8], len(val_real), len(train_real),
                 groups_info["val_has_human"]))
        random.shuffle(gen)
        val_gen, train_gen = gen[:args.gen_val], gen[args.gen_val:]
    else:
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

    human_set = set(human)
    if args.group_split:
        # 分组切分时，图片的归属完全由组决定（human 也可能被分到 val）
        for n in train_real:
            put(n, "train", "real_human" if n in human_set else "real_cv", find_img(n), find_lbl(n))
        for n in val_real:
            put(n, "val", "real_human" if n in human_set else "real_cv", find_img(n), find_lbl(n))
    else:
        for n in human:
            put(n, "train", "real_human", find_img(n), find_lbl(n))
        for n in train_real:
            put(n, "train", "real_cv", find_img(n), find_lbl(n))
        for n in val_real:
            put(n, "val", "real", find_img(n), find_lbl(n))
    for n in train_gen:
        put(n, "train", "gen", find_img(n), find_lbl(n))
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
        json.dump({"stats": stats, "index": index, "groups": groups_info,
                   "split_mode": "group" if args.group_split else "random",
                   "note": "val 以真实域（实拍）为主，用于诚实评估跨域泛化；train 混入生成图补充多样性",
                   "note_group": "分组切分：同一场景/近重复图整体进 train 或 val，避免指标虚高"
                                 if args.group_split else "随机切分（存在近重复泄漏风险）"},
                  f, ensure_ascii=False, indent=2)
    print("[mix_v3] train:", stats["train"])
    print("[mix_v3] val  :", stats["val"])
    print("[mix_v3] 输出 :", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
