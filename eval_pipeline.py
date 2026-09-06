#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
eval_pipeline.py —— 端到端组合正确率评估(方案A: 检测×属性)

- 属性模型各自准确率 + 色相偏移敏感性(±3°/±8°/±15°, 模拟现场色差)
- 组合类别端到端正确率(独立假设: recall × acc_color × acc_shape)
输出: reports/pipeline_report.json
用法: python eval_pipeline.py --config config/dataset.yaml
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from train_attributes import IMG_SIZE, TinyNet, AttrDataset

HUES = [0, 3, 8, 15]      # 色相偏移(度)


def shift_hue(img_bgr, deg):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV).astype(np.int16)
    hsv[..., 0] = (hsv[..., 0] + int(deg * 2)) % 180
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def load_model(task, n_cls, device):
    ckpt = "runs/attr/%s/best.pt" % task
    if not os.path.exists(ckpt):
        return None
    m = TinyNet(n_cls).to(device)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m.eval()
    return m


def infer_shift(model, imgs_bgr, device):
    """批量推理 + 色相偏移矩阵: [n, len(HUES)] 预测类别"""
    res = np.zeros((len(imgs_bgr), len(HUES)), dtype=int)
    with torch.no_grad():
        for j, deg in enumerate(HUES):
            xs = []
            for img in imgs_bgr:
                img = cv2.resize(shift_hue(img, deg), (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
                img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
                xs.append(img.transpose(2, 0, 1))
            x = torch.from_numpy(np.stack(xs)).float().to(device)
            res[:, j] = torch.softmax(model(x), 1).argmax(1).cpu().numpy()
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/dataset.yaml")
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = os.path.abspath(os.path.expanduser(cfg["root"]))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    report = {}
    for task in ("color", "shape", "defect"):
        classes = cfg[task + "_classes"]
        model = load_model(task, len(classes), device)
        if model is None:
            print("[跳过] %s 未训练 (runs/attr/%s/best.pt)" % (task, task))
            continue
        base = os.path.join(root, cfg["attribute_datasets"][task])
        dirs = [os.path.join(base, c) for c in classes]
        ds = AttrDataset(dirs, train=False)
        if len(ds) == 0:
            continue
        lab_idx = [y for _, y in ds.samples]
        imgs = [cv2.imread(ds.samples[i][0]) for i in range(len(ds.samples))]
        pred = infer_shift(model, imgs, device)
        cell = {"accuracy_base": round(float((pred[:, 0] == lab_idx).mean()), 4)}
        for j, deg in enumerate(HUES[1:], 1):
            cell["acc_hue%d" % deg] = round(float((pred[:, j] == lab_idx).mean()), 4)
        report[task] = cell
        print(task, cell)

    det_recall = None
    if os.path.exists("reports/eval_report.json"):
        with open("reports/eval_report.json") as f:
            det_recall = json.load(f).get("recall")
    report["detector_recall"] = det_recall
    if report.get("color") and report.get("shape") and det_recall:
        combo = det_recall * report["color"]["accuracy_base"] * report["shape"]["accuracy_base"]
        report["est_combo_accuracy"] = round(float(combo), 4)
        print("组合类别端到端(估算) = %.4f (recall×color×shape)" % combo)

    os.makedirs("reports", exist_ok=True)
    with open("reports/pipeline_report.json", "w") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("报告: reports/pipeline_report.json")


if __name__ == "__main__":
    main()
