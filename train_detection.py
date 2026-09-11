#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_detection.py —— YOLOv8 目标检测自训练

- 基础权重: 默认 models/yolov8n.pt(--base 可换), 缺省时自动调用 fetch_weights.py
- 设备: 自动 GPU(0)/CPU; 也可 --device 指定
- 增强: 重点 hsv_h(色差)/erase(遮挡), 来源 config/dataset.yaml aug 段
- 断点续训: --resume
输出: runs/detect/train/weights/best.pt + 备份 runs/detect/best_final.pt
用法: python train_detection.py --config config/dataset.yaml --epochs 120 --imgsz 640
"""
import argparse
import os
import shutil

import yaml


def get_base_weights(name):
    """yolov8n.pt 等: 存在则返回; 否则调用 fetch_weights.py(本地→hf-mirror→官方)"""
    if os.path.exists(name):
        return name
    if name in ("yolov8n.pt", "yolov8s.pt", "yolov8m.pt"):
        print("[INFO] 本地无 %s, 尝试 fetch_weights.py ..." % name)
        os.system("python3 fetch_weights.py --out models/%s" % name)
        cand = os.path.join("models", name)
        if os.path.exists(cand):
            return cand
        raise SystemExit("基础权重获取失败: 手动下载后放 models/%s" % name)
    return name


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/dataset.yaml")
    ap.add_argument("--base", default=None, help="基础权重(pt 路径或 yolov8n)")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--device", default=None, help="0=GPU, cpu=CPU, 缺省自动")
    ap.add_argument("--resume", default="", help="runs/detect/train/weights/last.pt")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    root = os.path.abspath(os.path.expanduser(cfg["root"]))
    if root == os.path.abspath("/path/to/my_sorting_data"):
        raise SystemExit("请先修改 config/dataset.yaml 的 root 为你的数据集路径")
    data_yaml = os.path.join(root, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit("缺少 data.yaml, 请先运行 prepare_dataset.py")

    import torch
    from ultralytics import YOLO

    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
        print("[INFO] 自动选择设备: %s" % args.device)

    base = args.base or os.path.join("models", cfg.get("model", "yolov8n") + ".pt")
    if not args.resume:
        base = get_base_weights(base)
    model = YOLO(args.resume or base)

    epochs = args.epochs or int(cfg.get("epochs", 120))
    # 兼容 img_size(旧) / imgsz(比赛配置)：货物 ≤40mm 属小目标，默认 960
    imgsz = args.imgsz or int(cfg.get("imgsz", cfg.get("img_size", 960)))
    batch = args.batch or int(cfg.get("batch", 16))
    aug = cfg.get("aug", {}) or {}

    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=args.device,
        # --- 增强(覆盖 ultralytics 默认, 来源 config) ---
        hsv_h=float(aug.get("hsv_h", 0.05)),      # 色差模拟
        hsv_s=float(aug.get("hsv_s", 0.3)),
        hsv_v=float(aug.get("hsv_v", 0.3)),
        fliplr=float(aug.get("fliplr", 0.5)),
        flipud=float(aug.get("flipud", 0.2)),     # 俯视相机允许上下翻转
        degrees=float(aug.get("degrees", 10.0)),
        translate=float(aug.get("translate", 0.05)),
        scale=float(aug.get("scale", 0.3)),
        mosaic=float(aug.get("mosaic", 1.0)),
        close_mosaic=int(aug.get("close_mosaic", 20)),
        mixup=float(aug.get("mixup", 0.1)),
        copy_paste=float(aug.get("copy_paste", 0.0)),  # 复制粘贴：密集/遮挡场景
        erasing=float(aug.get("erase", 0.1)),     # 随机擦除模拟遮挡与污渍
        cos_lr=bool(aug.get("cos_lr", False)),
        # 小目标：关闭矩形推理，保证 letterbox 后尺度一致
        rect=False,
        cache=bool(aug.get("cache", False)),
        workers=int(aug.get("workers", 8)),
        patience=int(cfg.get("patience", 30)),
        project="runs/detect",
        name="train",
        exist_ok=True,
        verbose=True,
        plots=True,
        amp=True,
    )

    best = os.path.join("runs/detect/train", "weights", "best.pt")
    if os.path.exists(best):
        os.makedirs("runs/detect", exist_ok=True)
        shutil.copy(best, "runs/detect/best_final.pt")
        print("[OK] 检测模型备份: runs/detect/best_final.pt")

    metrics = model.val(data=data_yaml, imgsz=imgsz, device=args.device)
    print("验证集: mAP50=%.4f  mAP50-95=%.4f" % (metrics.box.map50,
                                                  metrics.box.map))
    # 赛项验收线（config.acceptance.detect_map50，默认 0.95）
    target = float((cfg.get("acceptance", {}) or {}).get("detect_map50", 0.95))
    print("赛项验收线 mAP50 ≥ %.2f → %s (实测 %.4f)" % (
        target, "达标" if metrics.box.map50 >= target else "未达标", metrics.box.map50))
    print("下一步: python export_models.py --config %s --imgsz %d" % (
        args.config, imgsz))


if __name__ == "__main__":
    main()
