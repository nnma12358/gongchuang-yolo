#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
check_labels.py —— 标签体检（复核前/后都该跑）
=====================================================================
查六类问题：
  1) 格式：字段数、类别号、数值范围、坐标越界
  2) 几何：过小/过大/超长宽比/贴边（训练不收敛的常见来源）
  3) 重复：同图内完全重复或几乎重合的框
  4) 缺失：有图无标 / 有标无图 / 空标签（空=负样本，需确认不是漏标）
  5) 近重复图片：感知哈希找"几乎同一张"，**并报告是否跨 train/val**（数据泄漏！）
  6) 统计：每类框数与面积分布、每个 split 的样本量

用法：
  python tools/check_labels.py --data data/mix_v3
  python tools/check_labels.py --data data/mix_v3 --leak-threshold 6 --csv report.csv
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np


def read_yolo(path):
    rows, bad = [], []
    for ln, line in enumerate(open(path, encoding="utf-8"), 1):
        line = line.strip()
        if not line:
            continue
        p = line.split()
        if len(p) < 5:
            bad.append((ln, "字段数 %d<5" % len(p)))
            continue
        try:
            cls = int(float(p[0]))
            cx, cy, w, h = [float(v) for v in p[1:5]]
        except ValueError:
            bad.append((ln, "非数值"))
            continue
        if cls != 0:
            bad.append((ln, "类别号 %d（单类方案应为 0）" % cls))
            continue
        if not all(0.0 <= v <= 1.0 for v in (cx, cy, w, h)):
            bad.append((ln, "坐标越界 %s" % [round(v, 3) for v in (cx, cy, w, h)]))
            continue
        if w <= 0 or h <= 0:
            bad.append((ln, "宽高非正"))
            continue
        rows.append((cls, cx, cy, w, h))
    return rows, bad


def dhash(path, size=16):
    """感知哈希：16×16 → 256 bit（8×8=64bit 对"白底+同一形状"的合成图太粗，全是假阳性）"""
    im = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if im is None:
        return None
    im = cv2.resize(im, (size + 1, size), interpolation=cv2.INTER_AREA)
    return (im[:, 1:] > im[:, :-1]).flatten()


def color_sig(path):
    """4×4×4 HSV 直方图（归一化），用于把"形状像但颜色不同"的合成图排除掉"""
    im = cv2.imread(path, cv2.IMREAD_COLOR)
    if im is None:
        return None
    hsv = cv2.cvtColor(cv2.resize(im, (64, 64), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2HSV)
    h = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
    cv2.normalize(h, h)
    return h.flatten()


def hamming(a, b):
    return int(np.count_nonzero(a != b))


def is_real(stem):
    """实拍图：只有 real_*。负样本(neg_*)是背景裁片、gen_* 是合成图，都不参与泄漏判定。"""
    return stem.startswith("real_")


def main():
    ap = argparse.ArgumentParser(description="标签体检")
    ap.add_argument("--data", default="data/mix_v3")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--leak-threshold", type=int, default=24,
                    help="16×16 dHash(256bit) 汉明距离 ≤ 该值算近重复图")
    ap.add_argument("--color-threshold", type=float, default=0.35,
                    help="HSV 直方图 L1 距离 ≤ 该值才算同一物体（排除换了颜色的合成图）")
    ap.add_argument("--max-dupes", type=int, default=400, help="最多输出多少对近重复")
    args = ap.parse_args()

    ds = args.data
    splits = [d for d in ("train", "val", "test") if os.path.isdir(os.path.join(ds, "images", d))]
    problems, stats = [], {}
    hashes = {}
    for sp in splits:
        imgs = sorted(glob.glob(os.path.join(ds, "images", sp, "*.*")))
        labs = sorted(glob.glob(os.path.join(ds, "labels", sp, "*.txt")))
        img_stems = {os.path.splitext(os.path.basename(p))[0] for p in imgs}
        lab_stems = {os.path.splitext(os.path.basename(p))[0] for p in labs}
        for s in sorted(img_stems - lab_stems):
            problems.append({"type": "缺标注", "split": sp, "image": s, "detail": "有图无 .txt"})
        for s in sorted(lab_stems - img_stems):
            problems.append({"type": "缺图片", "split": sp, "image": s, "detail": "有 .txt 无图"})
        areas, nbox, empty, geo = [], 0, 0, 0
        for p in imgs:
            stem = os.path.splitext(os.path.basename(p))[0]
            lp = os.path.join(ds, "labels", sp, stem + ".txt")
            im = cv2.imread(p)
            if im is None:
                problems.append({"type": "坏图", "split": sp, "image": stem, "detail": "无法解码"})
                continue
            h, w = im.shape[:2]
            if not os.path.exists(lp):
                continue
            rows, bad = read_yolo(lp)
            for ln, why in bad:
                problems.append({"type": "格式错误", "split": sp, "image": stem,
                                 "detail": "第%d行 %s" % (ln, why)})
            nbox += len(rows)
            if not rows:
                empty += 1
            max_area = 0.80 if not is_real(stem) else 0.30   # 合成图物体占比本来就大
            for cls, cx, cy, bw, bh in rows:
                area = bw * bh
                areas.append(area)
                if area < 0.004 or area > max_area:
                    geo += 1
                    problems.append({"type": "面积异常", "split": sp, "image": stem,
                                     "detail": "面积 %.3f（阈值 %.2f）" % (area, max_area)})
                if max(bw, bh) / max(1e-6, min(bw, bh)) > 2.4:
                    geo += 1
                    problems.append({"type": "长宽比异常", "split": sp, "image": stem,
                                     "detail": "%.2f" % (max(bw, bh) / min(bw, bh))})
                if cx - bw / 2 < -0.005 or cy - bh / 2 < -0.005 or \
                   cx + bw / 2 > 1.005 or cy + bh / 2 > 1.005:
                    problems.append({"type": "越界", "split": sp, "image": stem,
                                     "detail": "框超出画面"})
            for i in range(len(rows)):
                for j in range(i + 1, len(rows)):
                    _, ax, ay, aw, ah = rows[i]
                    _, bx, by, bw2, bh2 = rows[j]
                    if abs(ax - bx) < 0.02 and abs(ay - by) < 0.02 and \
                       abs(aw - bw2) < 0.02 and abs(ah - bh2) < 0.02:
                        problems.append({"type": "重复框", "split": sp, "image": stem,
                                         "detail": "第%d/%d行几乎重合" % (i + 1, j + 1)})
            hs = dhash(p)
            if hs is not None:
                hashes[(sp, stem)] = (hs, color_sig(p))
        stats[sp] = {"images": len(imgs), "labels": len(labs), "boxes": nbox,
                     "empty_labels": empty, "geo_suspect": geo,
                     "area_mean": round(float(np.mean(areas)), 4) if areas else 0.0,
                     "area_p50": round(float(np.median(areas)), 4) if areas else 0.0}

    # ---- 近重复图片 → 泄漏检查（实拍与合成分开看）----
    leaks, dupes, gen_dupes = [], [], []
    keys = list(hashes)
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            (ha, ca), (hb, cb) = hashes[keys[i]], hashes[keys[j]]
            d = hamming(ha, hb)
            if d > args.leak_threshold:
                continue
            cdist = float(np.abs(ca - cb).sum()) if ca is not None and cb is not None else 0.0
            if cdist > args.color_threshold:
                continue                      # 形状像但颜色差异大 → 不是同一张
            a, b = keys[i], keys[j]
            rec = {"split_a": a[0], "img_a": a[1], "split_b": b[0], "img_b": b[1],
                   "dist": d, "color_dist": round(cdist, 3), "cross_split": a[0] != b[0]}
            if not is_real(a[1]):
                gen_dupes.append(rec)
            elif a[0] != b[0]:
                leaks.append(rec)
            else:
                dupes.append(rec)

    print("=" * 74)
    print(" 标签体检：%s" % ds)
    print("=" * 74)
    for sp, s in stats.items():
        print(" [%s] 图 %d · 标 %d · 框 %d · 空标签 %d · 几何可疑 %d · 面积 中位%.3f/均%.3f"
              % (sp, s["images"], s["labels"], s["boxes"], s["empty_labels"],
                 s["geo_suspect"], s["area_p50"], s["area_mean"]))
    try:
        cnt = {}
        for p in problems:
            cnt[p["type"]] = cnt.get(p["type"], 0) + 1
        if cnt:
            print(" 问题：", " · ".join("%s %d" % (k, v) for k, v in sorted(cnt.items())))
        else:
            print(" 问题：无")
    except Exception:
        pass
    print(" 近重复图对（16×16 dHash≤%d 且颜色接近）：" % args.leak_threshold)
    print("   实拍：跨 split **%d 对（泄漏）** · 同 split %d 对" % (len(leaks), len(dupes)))
    print("   合成：跨 split %d 对（同形状不同颜色的合成图本就会接近，仅提示）" % len(gen_dupes))
    if leaks:
        print("   跨 split 泄漏示例（同一物体同一场景同时出现在 train 与 val）：")
        for r in leaks[:8]:
            print("     %s[%s] ↔ %s[%s]  哈希距离%d 颜色距离%.2f"
                  % (r["img_a"], r["split_a"], r["img_b"], r["split_b"],
                     r["dist"], r["color_dist"]))
        print("   → val 指标会被高估；请用 tools/build_mix_v3.py --group-split 重新切分")

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8-sig") as f:
            w = csv.writer(f)
            w.writerow(["类型", "split", "图片", "说明"])
            for p in problems:
                w.writerow([p["type"], p["split"], p["image"], p["detail"]])
            for r in leaks:
                w.writerow(["跨split近重复", r["split_a"], r["img_a"],
                            "与 %s[%s] 距离%d" % (r["img_b"], r["split_b"], r["dist"])])
        print(" 明细已写入:", args.csv)
    # 供 CI/脚本判断
    json.dump({"stats": stats, "problems": problems, "leaks": leaks, "dupes": dupes[:args.max_dupes]},
              open(os.path.join(ds, "label_check.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print(" 报告:", os.path.join(ds, "label_check.json"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
