#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_attributes.py —— 颜色/形状/污渍 三个小 CNN 分类器训练(纯 PyTorch, 无 torchvision 依赖)

数据集: <root>/attributes/<task>/<class>/*.jpg (config/dataset.yaml 定义)
增强: 旋转/HSV抖动(色差)/随机擦除(遮挡)/翻转 (cv2 实现)
输出: runs/attr/<task>/best.pt (+ best.onnx 由 export_models.py 导出)
用法: python train_attributes.py --config config/dataset.yaml --task color
"""
import argparse
import glob
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, Dataset

IMG_SIZE = 96
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


class TinyNet(nn.Module):
    """80 万级小 CNN, 4 层卷积 + 2 层全连接, CPU 上 1ms 级推理"""

    def __init__(self, n_cls):
        super().__init__()
        base = 24
        self.features = nn.Sequential(
            nn.Conv2d(3, base, 3, 2, 1), nn.BatchNorm2d(base), nn.ReLU(),
            nn.Conv2d(base, base * 2, 3, 2, 1), nn.BatchNorm2d(base * 2), nn.ReLU(),
            nn.Conv2d(base * 2, base * 4, 3, 2, 1), nn.BatchNorm2d(base * 4), nn.ReLU(),
            nn.Conv2d(base * 4, base * 8, 3, 2, 1), nn.BatchNorm2d(base * 8), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.3),
            nn.Linear(base * 8 * 6 * 6, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )

    def forward(self, x):
        return self.head(self.features(x))


def aug_crop(bgr, task="color"):
    """cv2 增强: 翻转/旋转/HSV(色差)/随机擦除(遮挡)
    stain 任务额外加入噪声与模糊 —— 污渍/缺陷在真实图像中表现为暗斑与失焦"""
    img = bgr
    if random.random() < 0.5:
        img = cv2.flip(img, 1)
    if random.random() < 0.3:
        rows, cols = img.shape[:2]
        M = cv2.getRotationMatrix2D((cols / 2, rows / 2),
                                    random.uniform(-15, 15), 1.0)
        img = cv2.warpAffine(img, M, (cols, rows), borderValue=(114, 114, 114))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 0] = np.clip(hsv[..., 0] + random.uniform(-10, 10), 0, 180)
    hsv[..., 1] = np.clip(hsv[..., 1] * random.uniform(0.7, 1.3), 0, 255)
    hsv[..., 2] = np.clip(hsv[..., 2] * random.uniform(0.7, 1.3), 0, 255)
    img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    if random.random() < 0.2 and img.shape[0] > 20:
        y0 = random.randint(0, img.shape[0] - 10)
        x0 = random.randint(0, img.shape[1] - 10)
        img[y0:y0 + 8, x0:x0 + 8] = (114, 114, 114)
    if task == "stain":
        if random.random() < 0.35:                      # 模拟污渍暗斑
            h, w = img.shape[:2]
            r = random.randint(3, max(4, min(h, w) // 5))
            cv2.circle(img, (random.randint(r, w - r), random.randint(r, h - r)),
                       r, (60, 50, 40), -1)
        if random.random() < 0.25:                      # 失焦/划痕
            img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.5, 1.5))
    return img


class AttrDataset(Dataset):
    def __init__(self, class_dirs, train=True, task="color"):
        self.samples = []
        self.task = task
        for idx, d in enumerate(class_dirs):
            files = sorted(glob.glob(os.path.join(d, "*.jpg")) +
                           glob.glob(os.path.join(d, "*.png")))
            random.shuffle(files)
            split = int(len(files) * 0.85)
            files = files[:split] if train else files[split:]
            self.samples += [(f, idx) for f in files]
        self.class_names = [os.path.basename(d) for d in class_dirs]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, cls = self.samples[i]
        img = cv2.imread(path)
        if img is None:
            return torch.zeros(3, IMG_SIZE, IMG_SIZE), cls
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        if random.random() < 0.7:
            img = aug_crop(img, self.task)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        return torch.from_numpy(img.transpose(2, 0, 1)), cls


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/dataset.yaml")
    ap.add_argument("--task", required=True, choices=["color", "shape", "stain", "defect"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--export-onnx", action="store_true", help="训练后导出 ONNX（端侧部署用）")
    args = ap.parse_args()
    if args.task == "defect":
        args.task = "stain"                     # 旧命名兼容：defect → stain（clean/stain/defect）

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = os.path.abspath(os.path.expanduser(cfg["root"]))
    # 兼容 stain_classes(比赛配置) / defect_classes(旧配置)
    classes = cfg.get(args.task + "_classes") or cfg.get("defect_classes")
    if not classes:
        raise SystemExit("配置缺少 %s_classes" % args.task)
    attr_map = dict(cfg.get("attribute_datasets") or {})
    base = os.path.join(root, attr_map.get(args.task) or attr_map.get("defect") or ("attributes/" + args.task))
    class_dirs = [os.path.join(base, c) for c in classes]
    for d in class_dirs:
        if not os.path.isdir(d):
            raise SystemExit("缺少类别目录: %s（可先跑 scripts/crop_attributes.py 自动生成）" % d)

    train_ds = AttrDataset(class_dirs, train=True, task=args.task)
    val_ds = AttrDataset(class_dirs, train=False, task=args.task)
    print("[%s] train=%d val=%d classes=%s" % (
        args.task, len(train_ds), len(val_ds), classes))
    if len(val_ds) == 0:
        raise SystemExit("验证集为空, 请检查每类图像数量")

    train_ld = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2)
    val_ld = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2)

    model = TinyNet(len(classes))
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    loss_fn = nn.CrossEntropyLoss()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_acc = -1
    for ep in range(args.epochs):
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            out = model(x)
            loss = loss_fn(out, y)
            loss.backward()
            opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        sched.step()
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_ld:
                x, y = x.to(device), y.to(device)
                v_correct += (model(x).argmax(1) == y).sum().item()
                v_total += len(y)
        acc = v_correct / max(1, v_total)
        print("ep %02d  train_acc=%.4f  val_acc=%.4f" % (
            ep + 1, correct / max(1, total), acc))
        if acc > best_acc:
            best_acc = acc
            os.makedirs("runs/attr/%s" % args.task, exist_ok=True)
            torch.save(model.state_dict(), "runs/attr/%s/best.pt" % args.task)

    print("[%s] best val_acc=%.4f 模型: runs/attr/%s/best.pt" % (
        args.task, best_acc, args.task))
    target = float((cfg.get("acceptance", {}) or {}).get(args.task + "_acc", 0) or 0)
    if target:
        print("赛项验收线 %s_acc ≥ %.2f → %s (实测 %.4f)" % (
            args.task, target, "达标" if best_acc >= target else "未达标", best_acc))

    if args.export_onnx:
        model.load_state_dict(torch.load("runs/attr/%s/best.pt" % args.task, map_location="cpu"))
        model.eval()
        dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
        out_path = "runs/attr/%s/best.onnx" % args.task
        torch.onnx.export(model, dummy, out_path, input_names=["input"],
                          output_names=["logits"], opset_version=12,
                          dynamic_axes={"input": {0: "batch"}})
        print("ONNX 已导出: %s（端侧推理用）" % out_path)


if __name__ == "__main__":
    main()
