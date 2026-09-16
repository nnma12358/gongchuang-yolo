#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
mine_negatives.py —— 难负样本挖掘（解决背景误检）
=====================================================================
问题：检测器只在"有货物"的图上训练过（训练日志 `0 backgrounds`），
从没学过"什么不是货物"，于是实验室设备/纸边/桌沿都被判成 goods。
实测：161 张实拍图上 36 个误检框（0.22 个/图），**位置集中在画面四角**。

做法（不需要重新拍照）：
  1) 用当前模型跑全部实拍图，挑出与人工标注 IoU<0.3 的框 = **误检**
  2) 把误检区域按上下文裁下来 → **难负样本**（空标签）
  3) 再在"没有货物、也没有误检"的区域里，按**边缘能量**挑最"花"的窗口
     → **普通负样本**（背景杂乱处最有训练价值）
  4) 出一张拼图给人过一眼（确认裁下来的确实没有货物）

输出（可直接被 build_mix_v3.py --neg-dir 使用）：
  <out>/images/*.jpg   <out>/labels/*.txt(空)   <out>/report.json   <out>/sheet.jpg

用法：
  python tools/mine_negatives.py --weights runs/best_after_review.pt \
      --data data/mix_v3 --out data/negatives --conf 0.25 --n-random 150
"""
import argparse
import glob
import json
import os

import cv2
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


def crop_window(img, cx, cy, size):
    """按像素中心与边长裁窗口，越界则平移（不补边，保证全是真实像素）"""
    H, W = img.shape[:2]
    size = int(min(size, min(H, W)))
    x1 = int(round(cx - size / 2))
    y1 = int(round(cy - size / 2))
    x1 = max(0, min(W - size, x1))
    y1 = max(0, min(H - size, y1))
    return img[y1:y1 + size, x1:x1 + size]


def inter_area(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def has_object(win, labels, pad=0.02):
    """窗口里是否沾到任何标注物体。

    ⚠ 不能用 IoU 判断：窗口 640² 而物体可能只有 150²，物体整个在窗口里时
    IoU 也只有 0.05，会被误判成"干净背景"，结果造出"含着货物的空标签"，
    教模型"这东西不算货物"——比误检更糟。这里按**物体被框住的比例**判断。
    """
    for l in labels:
        box = [l[0] - pad, l[1] - pad, l[2] + pad, l[3] + pad]
        if inter_area(win, box) > 0:   # 零容忍：连物体边缘都不许进窗口
            return True
    return False


def edge_energy(gray):
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def main():
    ap = argparse.ArgumentParser(description="难负样本挖掘")
    ap.add_argument("--weights", default="runs/best_after_review.pt")
    ap.add_argument("--data", default="data/mix_v3", help="含 images/labels 的数据集（用人工标注）")
    ap.add_argument("--out", default="data/negatives")
    ap.add_argument("--device", default="0")
    ap.add_argument("--conf", type=float, default=0.25, help="挖误检用的低阈值")
    ap.add_argument("--fp-iou", type=float, default=0.3, help="与标注 IoU<该值算误检")
    ap.add_argument("--margin", type=float, default=1.8, help="裁剪窗口 = 误检框边长 × 该系数")
    ap.add_argument("--crop", type=int, default=640, help="输出负样本边长")
    ap.add_argument("--n-random", type=int, default=200, help="另外再采多少个背景窗口")
    args = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(args.weights)

    imgs = []
    for sp in ("train", "val"):
        imgs += sorted(glob.glob(os.path.join(args.data, "images", sp, "real_*.*")))
    if not imgs:
        raise SystemExit("没找到实拍图（期望 %s/images/{train,val}/real_*）" % args.data)

    os.makedirs(os.path.join(args.out, "images"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "labels"), exist_ok=True)

    hard, rand, n_fp, n_img, dropped_hard = [], [], 0, 0, 0
    for p in imgs:
        n_img += 1
        sp = "val" if os.sep + "val" + os.sep in p else "train"
        stem = os.path.splitext(os.path.basename(p))[0]
        labels = [to_xyxy(b) for b in load_yolo(
            os.path.join(args.data, "labels", sp, stem + ".txt"))]
        img = cv2.imread(p)
        if img is None:
            continue
        H, W = img.shape[:2]
        r = model.predict(p, imgsz=640, conf=args.conf, device=args.device, verbose=False)[0]
        preds = []
        if r.boxes is not None and len(r.boxes):
            for b, c in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                bb = [float(b[0]) / W, float(b[1]) / H, float(b[2]) / W, float(b[3]) / H]
                preds.append({"xyxy": bb, "conf": float(c),
                              "iou": max([iou_xyxy(bb, l) for l in labels] or [0.0])})
        # ---- 难负样本：误检框周围（裁下来若沾到真货物则丢弃）----
        for d in preds:
            if d["iou"] >= args.fp_iou:
                continue
            n_fp += 1
            bx = d["xyxy"]
            cx, cy = (bx[0] + bx[2]) / 2 * W, (bx[1] + bx[3]) / 2 * H
            side = max(bx[2] - bx[0], bx[3] - bx[1]) * max(W, H) * args.margin
            side = max(args.crop * 0.6, side)
            crop = crop_window(img, cx, cy, side)
            ch, cw = crop.shape[:2]
            win = [(cx - cw / 2) / W, (cy - ch / 2) / H, (cx + cw / 2) / W, (cy + ch / 2) / H]
            if has_object(win, labels):
                dropped_hard += 1
                continue
            hard.append({"crop": crop, "src": stem, "conf": round(d["conf"], 3),
                         "cx": round(cx / W, 3), "cy": round(cy / H, 3)})
        # ---- 普通负样本：不含货物、不含误检的"花"区域 ----
        block = args.crop
        step = max(64, block // 2)
        cand = []
        for y in range(0, max(1, H - block + 1), step):
            for x in range(0, max(1, W - block + 1), step):
                sub = [x / W, y / H, (x + block) / W, (y + block) / H]
                if has_object(sub, labels):
                    continue
                if any(iou_xyxy(sub, d["xyxy"]) > 0.05 for d in preds):
                    continue
                g = cv2.cvtColor(img[y:y + block, x:x + block], cv2.COLOR_BGR2GRAY)
                cand.append((edge_energy(g), x, y))
        cand.sort(reverse=True)
        for e, x, y in cand[:2]:                      # 每张图最多取 2 个最花的窗口
            rand.append({"crop": img[y:y + block, x:x + block].copy(),
                         "src": stem, "edge": round(e, 1)})

    # 背景样本按"边缘能量"排序取前 N（越花越有训练价值）
    rand.sort(key=lambda d: -d["edge"])
    picked = hard + rand[:args.n_random]
    for i, d in enumerate(picked, 1):
        name = "neg_%04d" % i
        cv2.imwrite(os.path.join(args.out, "images", name + ".jpg"), d["crop"],
                    [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        open(os.path.join(args.out, "labels", name + ".txt"), "w").close()   # 空标签 = 负样本
    # 拼图速览
    tiles = []
    for d in picked[:48]:
        t = cv2.resize(d["crop"], (220, 220))
        tiles.append(t)
    rows = []
    for i in range(0, len(tiles), 8):
        chunk = tiles[i:i + 8]
        chunk += [np.full((220, 220, 3), 245, np.uint8)] * (8 - len(chunk))
        rows.append(np.hstack(chunk))
    if rows:
        cv2.imwrite(os.path.join(args.out, "sheet.jpg"), np.vstack(rows),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    report = {"images_scanned": n_img, "false_positives": n_fp,
              "fp_per_image": round(n_fp / max(1, n_img), 3),
              "hard_negatives": len(hard), "hard_dropped_object": dropped_hard,
              "random_negatives": min(args.n_random, len(rand)),
              "total": len(picked), "conf": args.conf, "fp_iou": args.fp_iou,
              "srcs": sorted({d["src"] for d in hard})}
    json.dump(report, open(os.path.join(args.out, "report.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    print("=" * 68)
    print(" 扫描 %d 张实拍图 → 误检框 %d 个（%.2f 个/图）" % (n_img, n_fp, n_fp / max(1, n_img)))
    print(" 负样本：难例 %d（另有 %d 个误检裁剪因沾到真货物被丢弃）+ 背景 %d = **%d 张**"
          % (len(hard), dropped_hard, min(args.n_random, len(rand)), len(picked)))
    print(" 输出：%s/{images,labels,report.json,sheet.jpg}" % args.out)
    print(" 下一步：python tools/build_mix_v3.py --group-split --neg-dir %s && "
          "bash tools/eval_split.sh after_neg" % args.out)
    print("=" * 68)


if __name__ == "__main__":
    main()
