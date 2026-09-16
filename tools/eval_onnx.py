#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_onnx.py —— ONNX 模型精度评估（FP32 vs INT8 同一管线对比）
=====================================================================
用同一套预处理 + NMS 后处理，在数据集验证集上计算：
    Precision / Recall / F1 / mAP50（单类，IoU 0.5）/ 逐图检出数
并给出 FP32 → INT8 的精度损失与延迟对比，回答“量化后还能不能用”。

用法：
  python3 tools/eval_onnx.py --data data/real_synth_mix --imgsz 640 \
      --models exports/best_fp32.onnx exports/best_int8.onnx
"""
import argparse
import glob
import json
import os
import statistics
import time

import numpy as np


def letterbox(img, size):
    import cv2
    h0, w0 = img.shape[:2]
    scale = min(size / float(w0), size / float(h0))
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas, scale, dx, dy


def load_gt(label_path, W, H):
    boxes = []
    if os.path.exists(label_path):
        for line in open(label_path):
            p = line.split()
            if len(p) == 5:
                _, cx, cy, w, h = p
                cx, cy, w, h = float(cx) * W, float(cy) * H, float(w) * W, float(h) * H
                boxes.append([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
    return boxes


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    inter = (x2 - x1) * (y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(1e-9, ua)


def infer(sess, img_path, imgsz, conf_thres=0.25, iou_thres=0.45):
    import cv2
    img = cv2.imread(img_path)
    H, W = img.shape[:2]
    blob, scale, dx, dy = letterbox(img, imgsz)
    x = np.ascontiguousarray((blob[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    name = sess.get_inputs()[0].name
    t0 = time.time()
    out = sess.run(None, {name: x})[0]
    lat = (time.time() - t0) * 1000
    preds = np.asarray(out)
    if preds.ndim == 3:
        preds = preds[0]
    if preds.shape[0] < preds.shape[1]:
        preds = preds.T
    scores_all = preds[:, 4:]
    cls = scores_all.argmax(axis=1)
    conf = scores_all.max(axis=1)
    keep = conf > conf_thres
    boxes, confs = [], []
    for (cx, cy, w, h), c in zip(preds[keep, :4], conf[keep]):
        x1 = (cx - w / 2 - dx) / scale
        y1 = (cy - h / 2 - dy) / scale
        boxes.append([x1, y1, x1 + w / scale, y1 + h / scale])
        confs.append(float(c))
    # NMS（OpenCV 实现，与检测容器一致）
    dets = []
    if boxes:
        idx = cv2.dnn.NMSBoxes([[b[0], b[1], b[2] - b[0], b[3] - b[1]] for b in boxes],
                               confs, conf_thres, iou_thres)
        for i in (idx.flatten() if idx is not None and len(idx) else []):
            dets.append({"box": boxes[i], "conf": confs[i]})
    dets.sort(key=lambda d: -d["conf"])
    return dets, lat, (W, H)


def evaluate(model_path, images, imgsz, conf_thres=0.25, iou_thres=0.45):
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    tp = fp = fn = 0
    lats = []
    matched_ious = []
    for p in images:
        lp = p.replace(os.sep + "images" + os.sep, os.sep + "labels" + os.sep)
        lp = os.path.splitext(lp)[0] + ".txt"
        dets, lat, (W, H) = infer(sess, p, imgsz, conf_thres, iou_thres)
        lats.append(lat)
        gt = load_gt(lp, W, H)
        used = set()
        for d in dets:
            best_i, best_iou = -1, 0.0
            for i, g in enumerate(gt):
                if i in used:
                    continue
                v = iou(d["box"], g)
                if v > best_iou:
                    best_i, best_iou = i, v
            if best_iou >= 0.5:
                used.add(best_i)
                tp += 1
                matched_ious.append(best_iou)
            else:
                fp += 1
        fn += len(gt) - len(used)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return {
        "model": os.path.basename(model_path),
        "size_mb": round(os.path.getsize(model_path) / 1e6, 2),
        "precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
        "map50_approx": round(precision * recall if (precision + recall) else 0.0, 4),
        "tp": tp, "fp": fp, "fn": fn,
        "mean_iou_matched": round(statistics.mean(matched_ious), 3) if matched_ious else 0.0,
        "latency_ms_mean": round(statistics.mean(lats), 2),
        "latency_ms_p95": round(sorted(lats)[int(len(lats) * 0.95) - 1], 2),
    }


def main():
    ap = argparse.ArgumentParser(description="ONNX 精度/延迟评估（单类，IoU0.5）")
    ap.add_argument("--data", default="data/real_synth_mix")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.45)
    ap.add_argument("--out", default="exports/eval_report.json")
    args = ap.parse_args()

    images = sorted(glob.glob(os.path.join(args.data, "images", "val", "*.*")))
    print("[评估] 验证集 {0} 张 · imgsz {1} · conf {2}".format(len(images), args.imgsz, args.conf))
    rows = []
    for m in args.models:
        if not os.path.exists(m):
            print("  跳过（不存在）:", m); continue
        r = evaluate(m, images, args.imgsz, args.conf, args.iou)
        rows.append(r)
        print("  {model:<22} P={precision:<7} R={recall:<7} F1={f1:<7} IoU={mean_iou_matched:<6} "
              "延迟={latency_ms_mean}ms  体积={size_mb}MB".format(**r))
    if len(rows) >= 2:
        base = rows[0]
        for r in rows[1:]:
            r["recall_drop"] = round(base["recall"] - r["recall"], 4)
            r["precision_drop"] = round(base["precision"] - r["precision"], 4)
            r["speedup"] = round(base["latency_ms_mean"] / max(1e-6, r["latency_ms_mean"]), 2)
        print("\n 相对 {0}：".format(base["model"]))
        for r in rows[1:]:
            print("   {0}: 召回变化 {1:+.4f} · 精确率变化 {2:+.4f} · 加速 {3}x".format(
                r["model"], -r["recall_drop"], -r["precision_drop"], r["speedup"]))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    def _jsonable(o):
        """numpy/float32 → 原生类型（否则 json.dump 报 float32 not serializable）。"""
        if isinstance(o, dict):
            return {k: _jsonable(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [_jsonable(v) for v in o]
        if hasattr(o, "item"):
            return o.item()
        return o

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(_jsonable({"imgsz": args.imgsz, "conf": args.conf, "iou": args.iou,
                             "val_images": len(images), "models": rows}),
                  f, ensure_ascii=False, indent=2)
    print("\n报告:", args.out)


if __name__ == "__main__":
    main()
