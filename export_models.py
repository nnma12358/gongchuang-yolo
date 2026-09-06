#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_models.py —— 导出 ONNX(opset 11) 并生成 Jetson TensorRT FP16 引擎命令

- 检测: ultralytics export onnx(OpenCV DNN/trtexec 兼容)
- 属性: torch.onnx.export
- 输出 exports/trt_commands.sh: 在 Jetson 上执行即得 fp16 engine
用法: python export_models.py --config config/dataset.yaml --imgsz 640
"""
import argparse
import os
import shutil

import torch
import yaml

from train_attributes import TinyNet, IMG_SIZE


def export_attr(task, n_cls):
    ckpt = "runs/attr/%s/best.pt" % task
    if not os.path.exists(ckpt):
        print("[跳过] 缺少 %s" % ckpt)
        return None
    model = TinyNet(n_cls)
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.eval()
    out = "runs/attr/%s/best.onnx" % task
    torch.onnx.export(model, torch.randn(1, 3, IMG_SIZE, IMG_SIZE), out,
                      input_names=["image"], output_names=["logits"],
                      opset_version=11, do_constant_folding=True)
    print("[OK] %s -> %s (%d classes)" % (task, out, n_cls))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config/dataset.yaml")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--det_weights", default="runs/detect/best_final.pt")
    ap.add_argument("--device", default="cpu", help="导出设备: cpu / 0(GPU)")
    ap.add_argument("--model", default="yolov8n")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    os.makedirs("exports", exist_ok=True)
    cmds = ["#!/bin/bash",
            "# Jetson(TensorRT FP16) 引擎生成命令 —— 在 Jetson 上执行",
            "TRTEXEC=/usr/src/tensorrt/bin/trtexec   # JetPack4.6 路径; JetPack5+: $(which trtexec)"]

    # ---------- 检测 ----------
    if os.path.exists(args.det_weights):
        from ultralytics import YOLO
        model = YOLO(args.det_weights)
        det_onnx = "exports/%s_det_%d.onnx" % (args.model, args.imgsz)
        model.export(format="onnx", imgsz=args.imgsz, opset=11,
                     simplify=True, dynamic=False, device=args.device)
        # ultralytics 导出到 runs/detect/train/weights/best.onnx
        src = os.path.join("runs/detect/train/weights", "best.onnx")
        if os.path.exists(src):
            shutil.copy(src, det_onnx)
            print("[OK] 检测 ONNX: %s" % det_onnx)
            cmds.append("$TRTEXEC --onnx=%s --saveEngine=exports/det_fp16.engine --fp16 --memPoolSize=workspace:1024" % det_onnx)
        else:
            print("[WARN] 未找到导出文件, 请检查 runs/detect/ 下的 .onnx")
    else:
        print("[跳过] 缺少检测权重 %s" % args.det_weights)

    # ---------- 属性 ----------
    for task, key in (("color", "color_classes"), ("shape", "shape_classes"),
                      ("defect", "defect_classes")):
        p = export_attr(task, len(cfg[key]))
        if p:
            cmds.append("$TRTEXEC --onnx=%s --saveEngine=exports/%s_fp16.engine --fp16 --memPoolSize=workspace:512" % (p, task))

    with open("exports/trt_commands.sh", "w") as f:
        f.write("\n".join(cmds) + "\n")
    os.chmod("exports/trt_commands.sh", 0o755)
    print("生成: exports/trt_commands.sh (在 Jetson 执行生成 fp16 engine)")


if __name__ == "__main__":
    main()
