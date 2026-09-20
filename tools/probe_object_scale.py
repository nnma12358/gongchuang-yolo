# -*- coding: utf-8 -*-
"""物体像素尺度 → 检出置信度 曲线探针（定位「模型为什么在某个视角失效」）

背景：2026-09 现场实测发现，顶置 Astra 俯拍画面里真实货物只有约 30px，
模型 0 检出且把黑色工具箱误检成货物。为区分「尺度问题」还是「背景域问题」，
本探针把同一张训练图里的物体（白球）逐级缩放后贴到固定 640 画布上，分别用
训练同款灰纸背景与现场白桌面背景测置信度。

**关键：必须保持画布尺寸不变，只改变物体本身像素数。**
早期的错误做法是把整张图缩放——letterbox 又会把图拉回 640，等于没测。

结论（2026-09-20 实测，goods_yolov8n_640_fp32.onnx）：
    物体像素   灰纸背景   白桌面背景
      160px     0.469      0.147
      100px     0.708      0.721   <- 最佳
       69px     0.611      0.589   <- 最佳区间
       45px     0.382      0.299
       33px     0.108      0.117   <- 现场实际尺度，低于 0.35 阈值
       24px     0.015      0.012
⇒ 背景影响很小，尺度是主因：模型工作区间 69~100px，现场只有 ~30px。

用法：
    python3 tools/probe_object_scale.py \
        --model exports/goods_yolov8n_640_fp32.onnx \
        --src data/mix_v3/images/val/real_00082.jpg
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def letterbox_fixed(img, size=640, bg=114):
    """把 img 等比缩放到 size 画布中央（保持画布恒定，物体像素随源图变化）"""
    import cv2
    h, w = img.shape[:2]
    s = float(size) / max(h, w)
    rs = cv2.resize(img, (int(round(w * s)), int(round(h * s))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), bg, np.uint8)
    y0 = (size - rs.shape[0]) // 2
    x0 = (size - rs.shape[1]) // 2
    canvas[y0:y0 + rs.shape[0], x0:x0 + rs.shape[1]] = rs
    return canvas


def top_conf(sess, img, conf_thres=0.01):
    """返回该图最高类别置信度（不做阈值过滤，只看模型「看到了什么」）"""
    x = np.ascontiguousarray(
        (img[:, :, ::-1].astype(np.float32) / 255.0).transpose(2, 0, 1)[None])
    out = np.asarray(sess.run(None, {sess.get_inputs()[0].name: x})[0])
    if out.ndim == 3:
        out = out[0]
    if out.shape[0] < out.shape[1]:
        out = out.T
    conf = out[:, 4:].max(axis=1)
    return float(conf.max()) if len(conf) else 0.0


def main():
    ap = argparse.ArgumentParser(description="物体像素尺度 → 置信度 曲线")
    ap.add_argument("--model", required=True, help="ONNX 检测模型（固定 640 输入）")
    ap.add_argument("--src", default="data/mix_v3/images/val/real_00082.jpg",
                    help="源图：物体大且清晰的一张训练/实拍图")
    ap.add_argument("--crop", type=int, nargs=4, default=[430, 830, 400, 800],
                    metavar=("Y1", "Y2", "X1", "X2"), help="源图中物体所在裁剪框")
    ap.add_argument("--object-px", type=int, default=330,
                    help="裁剪框里物体的像素直径（用于换算缩放比例）")
    ap.add_argument("--sizes", type=int, nargs="*", default=[330, 200, 160, 100, 69, 45, 33, 24])
    ap.add_argument("--bgs", nargs="*", default=["gray:180", "white:236"],
                    help="背景名:灰度值，可多组")
    args = ap.parse_args()

    import cv2
    import onnxruntime as ort
    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    src = cv2.imread(args.src)
    if src is None:
        raise SystemExit("无法读取源图: {0}".format(args.src))
    y1, y2, x1, x2 = args.crop
    obj = src[y1:y2, x1:x2]

    bgs = [(b.split(":")[0], int(b.split(":")[1])) for b in args.bgs]
    print("{0:>10s} | {1}".format("物体像素", " | ".join("{0:>12s}".format(n) for n, _ in bgs)))
    print("-" * (13 + 15 * len(bgs)))
    for target in args.sizes:
        k = float(target) / args.object_px
        ob = cv2.resize(obj, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
        vals = []
        for _, bgv in bgs:
            canvas = letterbox_fixed(ob, 640, bg=bgv)
            vals.append(top_conf(sess, canvas))
        print("{0:>8d}px | {1}".format(
            target, " | ".join("{0:>12.3f}".format(v) for v in vals)))


if __name__ == "__main__":
    main()
