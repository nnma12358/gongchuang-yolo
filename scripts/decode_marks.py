#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
decode_marks.py —— 货物表面标记解码（二维码 / 文字 / 污渍缺陷图形）

赛项：货物表面附有模拟污渍和缺陷的图形、文字、二维码等信息。
本脚本不训练模型，直接用 OpenCV 完成标记解码，并为数据集生成标记索引：

  · 二维码  cv2.QRCodeDetector().detectAndDecode（多码支持 detectAndDecodeMulti）
  · 污渍/缺陷  物体区域内暗斑占比（阈值可配）→ clean / stain / defect
  · 文字    OCR 可选接入；默认把二维码载荷作为文字（现场任务码常写在二维码内）

输出：
  · marks/marks.json   每张图片的 {qr, text, stain, dark_ratio} 索引（供训练/复核）
  · marks/marks.csv    同上（便于表格查看）
  · --annotate         生成可视化图（框 + 标记文本），便于人工核对

用法：
  python scripts/decode_marks.py --config config/sorting_competition.yaml --images data/images/train
  python scripts/decode_marks.py --config config/sorting_competition.yaml --images data/images/val --annotate
"""
import argparse
import csv
import glob
import json
import os

import cv2
import numpy as np
import yaml


def decode_qr(bgr):
    """返回 (二维码载荷, 数量)；兼容单码/多码"""
    try:
        det = cv2.QRCodeDetector()
        ok, texts, points, _ = det.detectAndDecodeMulti(bgr)
        if ok and texts:
            texts = [t for t in texts if t]
            if texts:
                return texts[0], len(texts)
    except Exception:
        pass
    try:
        data, _, _ = cv2.QRCodeDetector().detectAndDecode(bgr)
        if data:
            return data, 1
    except Exception:
        pass
    return "", 0


def stain_level(bgr, box=None, stain_th=0.05, defect_th=0.14):
    """物体区域暗斑占比 → (level, ratio)；无 box 时用整图前景"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    if box:
        x1, y1, x2, y2 = box
        region = gray[max(0, y1):y2, max(0, x1):x2]
    else:
        region = gray
    if region.size == 0:
        return "clean", 0.0
    med = float(np.median(region))
    dark = float(np.mean(region < max(60.0, med * 0.72)))
    if dark > defect_th:
        return "defect", dark
    if dark > stain_th:
        return "stain", dark
    return "clean", dark


def foreground_box(bgr):
    """Otsu 前景最大轮廓 → 外接框（用于把污渍判定限制在货物上）"""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < 0.001 * bgr.shape[0] * bgr.shape[1]:
        return None
    return cv2.boundingRect(c)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/sorting_competition.yaml")
    ap.add_argument("--images", required=True, help="图片目录，如 data/images/train")
    ap.add_argument("--out", default=None, help="输出目录（默认 config 的 marks.export_json 所在目录）")
    ap.add_argument("--annotate", action="store_true", help="输出可视化核对图到 <out>/annotated/")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    marks_cfg = cfg.get("marks", {}) or {}
    stain_th = float(marks_cfg.get("stain_dark_ratio", 0.05))
    defect_th = float(marks_cfg.get("defect_dark_ratio", 0.14))
    out_dir = args.out or os.path.dirname(marks_cfg.get("export_json", "marks/marks.json")) or "marks"
    os.makedirs(out_dir, exist_ok=True)
    if args.annotate:
        os.makedirs(os.path.join(out_dir, "annotated"), exist_ok=True)

    files = sorted(glob.glob(os.path.join(args.images, "*.jpg")) +
                   glob.glob(os.path.join(args.images, "*.png")))
    print("[标记解码] 图片 %d 张  来源 %s" % (len(files), args.images))

    index, stats = {}, {"qr_ok": 0, "clean": 0, "stain": 0, "defect": 0}
    for path in files:
        img = cv2.imread(path)
        if img is None:
            continue
        qr, n_qr = decode_qr(img)
        box = foreground_box(img)
        level, ratio = stain_level(img, box, stain_th, defect_th)
        name = os.path.basename(path)
        index[name] = {"qr": qr, "qr_count": n_qr, "text": qr, "stain": level,
                       "dark_ratio": round(ratio, 4)}
        stats["qr_ok" if qr else "clean"] = stats.get("qr_ok" if qr else "clean", 0)
        stats[level] = stats.get(level, 0) + 1

        if args.annotate:
            vis = img.copy()
            if box:
                x, y, w, h = box
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 200, 255), 2)
            label = "QR:%s" % (qr[:18] if qr else "-") + "  " + level
            cv2.putText(vis, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 90, 200), 2)
            cv2.imwrite(os.path.join(out_dir, "annotated", name), vis)

    json_path = os.path.join(out_dir, "marks.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "marks.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["image", "qr", "qr_count", "stain", "dark_ratio"])
        for name, rec in index.items():
            w.writerow([name, rec["qr"], rec["qr_count"], rec["stain"], rec["dark_ratio"]])

    total = max(1, len(index))
    print("[完成] 标记索引 %d 条 → %s" % (len(index), json_path))
    print("  二维码解码成功: %d/%d (%.1f%%)" % (stats.get("qr_ok", 0), total, 100.0 * stats.get("qr_ok", 0) / total))
    print("  表面状态: clean=%d stain=%d defect=%d" % (
        stats.get("clean", 0), stats.get("stain", 0), stats.get("defect", 0)))
    print("  验收线: 二维码解码率 ≥ %.0f%%（config.acceptance.qr_decode_rate）" % (
        100 * float(cfg.get("acceptance", {}).get("qr_decode_rate", 0.98))))


if __name__ == "__main__":
    main()
