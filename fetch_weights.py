#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fetch_weights.py —— 官方 yolov8n 基础权重获取(本机 GitHub 直连被拦截, 三级回退)

回退顺序:
  1. 本地 models/yolov8n.pt (可自放, 或从第一步实验目录复制)
  2. hf-mirror.com/Ultralytics/YOLOv8/resolve/main/yolov8n.pt  (已验证可用)
  3. ultralytics 官方自动下载(要求 GitHub 可达)
用法: python fetch_weights.py [--out models/yolov8n.pt]
"""
import argparse
import os
import shutil
import sys

CANDIDATES = [
    ("models/yolov8n.pt", "5.智能分拣机器人项目/../YOLOv8n第一步实验/models/yolov8n.pt"),
    ("../YOLOv8n第一步实验/models/yolov8n.pt", "第一步实验产物"),
    ("/home/xxxffyy/工创/YOLOv8n第一步实验/models/yolov8n.pt", "本机第一步实验产物"),
]
HF_URL = ("https://hf-mirror.com/Ultralytics/YOLOv8/"
          "resolve/main/yolov8n.pt")


def try_copies(out):
    for src, desc in CANDIDATES:
        if os.path.exists(src) and os.path.isfile(src):
            shutil.copy(src, out)
            print("[OK] 使用本地权重: %s (%s)" % (src, desc))
            return True
    return False


def try_hf(out):
    try:
        import urllib.request
        req = urllib.request.Request(HF_URL,
                                     headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
            f.write(r.read())
        print("[OK] hf-mirror 下载成功: %s (%d bytes)" % (out,
                                                          os.path.getsize(out)))
        return os.path.getsize(out) > 1_000_000
    except Exception as e:
        print("[WARN] hf-mirror 失败: %s" % e)
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="models/yolov8n.pt")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if os.path.exists(args.out) and os.path.getsize(args.out) > 1_000_000:
        print("[OK] 已存在: %s" % args.out)
        return
    if try_copies(args.out) or try_hf(args.out):
        return
    print("[INFO] 使用 ultralytics 官方自动下载(需 GitHub 可达):")
    from ultralytics import YOLO  # noqa: F401 触发官方下载
    YOLO("yolov8n.pt")
    print("[OK] 官方下载完成; 若需固定路径, 请复制到 %s" % args.out)


if __name__ == "__main__":
    main()
