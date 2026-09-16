#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
export_quantize.py —— 模型导出 · INT8 量化 · 基准测试（一条命令走完）
=====================================================================
流程：
  1) YOLOv8 .pt → ONNX（FP32，opset12）                       [ultralytics]
  2) ONNX → INT8 动态量化（权重 INT8，无需校准集）              [onnxruntime]
  3) ONNX → INT8 静态量化（用验证集图像校准，精度更高）          [onnxruntime]
  4) 基准测试：三种模型在同一批真实图片上的**延迟**与**输出一致性**
  5) 输出对比表 + exports/quant_report.json，并把最优 INT8 复制为 best_int8.onnx

用法：
  python3 tools/export_quantize.py --weights runs/detect/train/weights/best.pt \
      --data data/real_synth_mix --imgsz 640 --calib-n 100
  # 只量化已有 ONNX（跳过导出）
  python3 tools/export_quantize.py --onnx exports/best_fp32.onnx --data data/real_synth_mix
"""
import argparse
import glob
import json
import os
import shutil
import statistics
import time

import numpy as np


# ==================== 工具 ====================
def letterbox(img, size):
    import cv2
    h0, w0 = img.shape[:2]
    scale = min(size / float(w0), size / float(h0))
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas


def preprocess(path, size):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    blob = letterbox(img, size)
    x = blob[:, :, ::-1].astype(np.float32) / 255.0          # BGR→RGB
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None, ...])


class CalibReader(object):
    """onnxruntime 静态量化校准数据读取器"""

    def __init__(self, images, size, input_name, limit=100):
        self.images = images[:limit]
        self.size = size
        self.input_name = input_name
        self.idx = 0

    def get_next(self):
        while self.idx < len(self.images):
            x = preprocess(self.images[self.idx], self.size)
            self.idx += 1
            if x is not None:
                return {self.input_name: x}
        return None

    def rewind(self):
        self.idx = 0

    def __len__(self):
        return len(self.images)


def export_fp32(weights, imgsz, out_dir):
    from ultralytics import YOLO
    model = YOLO(weights)
    # per-channel 静态量化要求 DequantizeLinear 支持 axis（opset ≥ 13）
    path = model.export(format="onnx", imgsz=imgsz, opset=13, simplify=True, dynamic=False)
    dst = os.path.join(out_dir, "best_fp32.onnx")
    shutil.copy(path, dst)
    return dst, model


def validate_onnx(path):
    """加载校验：无效模型（opset/属性不兼容）当场发现，不进入基准测试"""
    import onnxruntime as ort
    try:
        ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        return True, None
    except Exception as e:
        return False, str(e)[:200]


def sanity_onnx(path, images, imgsz, conf=0.25):
    """输出合理性校验：QDQ 静态量化常出现“能加载但检测头输出全 0”的假成功。

    返回 (ok, max_score, n_boxes)：任一真实图片上有检出才算通过。
    """
    import onnxruntime as ort
    try:
        sess = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        name = sess.get_inputs()[0].name
        best_score, best_n = 0.0, 0
        for p in images[:8]:
            x = preprocess(p, imgsz)
            if x is None:
                continue
            out = np.asarray(sess.run(None, {name: x})[0])
            if out.ndim == 3:
                out = out[0]
            if out.shape[0] < out.shape[1]:
                out = out.T
            scores = out[:, 4:]
            mx = float(scores.max()) if scores.size else 0.0
            n = int((scores.max(axis=1) > conf).sum()) if scores.size else 0
            best_score, best_n = max(best_score, mx), max(best_n, n)
        return (best_n > 0), best_score, best_n
    except Exception as e:
        return False, 0.0, 0


def quantize_dynamic(src, dst):
    from onnxruntime.quantization import QuantType, quantize_dynamic
    quantize_dynamic(src, dst, weight_type=QuantType.QInt8, per_channel=False)
    ok, err = validate_onnx(dst)
    if not ok:
        print("   动态量化产出模型无效：", err)
        return None
    return dst


def quantize_static(src, dst, images, imgsz, calibrate_n=100):
    import onnxruntime as ort
    from onnxruntime.quantization import CalibrationMethod, QuantFormat, QuantType, quantize_static
    sess = ort.InferenceSession(src, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    for per_channel in (True, False):
        try:
            reader = CalibReader(images, imgsz, input_name, limit=calibrate_n)
            quantize_static(src, dst, reader, quant_format=QuantFormat.QDQ,
                            per_channel=per_channel, weight_type=QuantType.QInt8,
                            activation_type=QuantType.QUInt8,
                            calibrate_method=CalibrationMethod.MinMax)
            ok, err = validate_onnx(dst)
            if ok:
                return dst, per_channel
            print("   per_channel={0} 模型无效，回退：{1}".format(per_channel, err))
        except Exception as e:
            print("   per_channel={0} 异常：{1}".format(per_channel, str(e)[:120]))
    return None, None


def bench(model_path, images, imgsz, warmup=3, runs=20):
    """返回 {latency_ms: {mean,p50,p95}, boxes_mean, size_mb} —— 统一用 ONNX Runtime 推理对比"""
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 4
    sess = ort.InferenceSession(model_path, sess_options=so, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    xs = [preprocess(p, imgsz) for p in images[:runs]]
    xs = [x for x in xs if x is not None]
    if not xs:
        return None
    for i in range(min(warmup, len(xs))):
        sess.run(None, {input_name: xs[i]})
    lat, boxes = [], []
    for x in xs:
        t0 = time.time()
        out = sess.run(None, {input_name: x})[0]
        lat.append((time.time() - t0) * 1000)
        p = np.asarray(out)
        if p.ndim == 3:
            p = p[0]
        if p.shape[0] < p.shape[1]:
            p = p.T
        scores = p[:, 4:]
        boxes.append(int((scores.max(axis=1) > 0.25).sum()))
    lat.sort()
    return {"latency_ms": {"mean": round(statistics.mean(lat), 2),
                           "p50": round(lat[len(lat) // 2], 2),
                           "p95": round(lat[int(len(lat) * 0.95) - 1], 2)},
            "candidate_boxes_mean": round(statistics.mean(boxes), 2),
            "size_mb": round(os.path.getsize(model_path) / 1e6, 2)}


def main():
    ap = argparse.ArgumentParser(description="导出 ONNX + INT8 量化 + 基准测试")
    ap.add_argument("--weights", help="训练得到的 .pt（与 --onnx 二选一）")
    ap.add_argument("--onnx", help="已有 FP32 ONNX（跳过导出）")
    ap.add_argument("--data", default="data/real_synth_mix", help="数据集根（用其 images/val 做校准与基准）")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--calib-n", type=int, default=100, help="静态量化校准图片数")
    ap.add_argument("--out", default="exports")
    ap.add_argument("--bench-n", type=int, default=20)
    ap.add_argument("--no-static", action="store_true", help="跳过静态量化（只做动态）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    report = {"imgsz": args.imgsz, "data": os.path.abspath(args.data)}

    # ---------- 1) 导出 FP32 ----------
    if args.onnx:
        fp32 = args.onnx
    else:
        if not args.weights:
            ap.error("需要 --weights 或 --onnx")
        print("[1/4] 导出 ONNX (FP32) …")
        fp32, _ = export_fp32(args.weights, args.imgsz, args.out)
    print("   FP32:", fp32, "%.2f MB" % (os.path.getsize(fp32) / 1e6))

    # ---------- 2) 动态量化 ----------
    print("[2/4] INT8 动态量化 …")
    dyn = os.path.join(args.out, "best_int8_dynamic.onnx")
    try:
        if quantize_dynamic(fp32, dyn) is None:
            dyn = None
        else:
            print("   动态 INT8:", dyn, "%.2f MB" % (os.path.getsize(dyn) / 1e6))
    except Exception as e:
        print("   动态量化失败:", str(e)[:160])
        dyn = None

    # ---------- 3) 静态量化（校准集=验证集） ----------
    static = None
    calib_images = sorted(glob.glob(os.path.join(args.data, "images", "val", "*.*")))
    if not args.no_static and calib_images:
        print("[3/4] INT8 静态量化（校准 {0} 张）…".format(min(args.calib_n, len(calib_images))))
        static_path = os.path.join(args.out, "best_int8_static.onnx")
        try:
            static_path, per_channel = quantize_static(fp32, static_path, calib_images, args.imgsz, args.calib_n)
            if static_path is None:
                static = None
            else:
                # 假成功过滤：QDQ 静态量化“能加载”但检测头输出全 0 是常见现象
                # （见 docs_TRAINING.md 量化章节），必须用真实图片验输出。
                ok, mx, nb = sanity_onnx(static_path, calib_images, args.imgsz)
                size_mb = os.path.getsize(static_path) / 1e6
                if ok:
                    static = static_path
                    print("   静态 INT8: {0} {1:.2f} MB（per_channel={2}，最大置信 {3:.3f}，检出 {4}）".format(
                        static_path, size_mb, per_channel, mx, nb))
                else:
                    static = None
                    print("   静态 INT8 输出无效（最大置信 {0:.3f}，检出 {1}）→ 弃用，"
                          "YOLOv8 检测头经 QDQ 静态量化后普遍塌陷；"
                          "Jetson 请用 TensorRT INT8（tools/build_trt_engine.py）".format(mx, nb))
        except Exception as e:
            print("   静态量化失败:", str(e)[:160])
            static = None
    else:
        print("[3/4] 跳过静态量化（无校准图片或 --no-static）")

    # ---------- 4) 基准测试 ----------
    print("[4/4] 基准测试（ONNX Runtime, CPU）…")
    bench_images = calib_images or sorted(glob.glob(os.path.join(args.data, "images", "train", "*.*")))
    models = [("FP32", fp32)] + ([("INT8-dynamic", dyn)] if dyn else []) + \
             ([("INT8-static", static)] if static else [])
    rows = []
    for name, path in models:
        try:
            r = bench(path, bench_images, args.imgsz, runs=args.bench_n)
        except Exception as e:
            print("   {0} 基准测试失败：{1}".format(name, str(e)[:120]))
            r = None
        if r:
            r["name"] = name
            r["path"] = path
            rows.append(r)
    report["models"] = rows
    if len(rows) >= 2:
        base = rows[0]
        for r in rows[1:]:
            r["speedup_vs_fp32"] = round(base["latency_ms"]["mean"] / max(1e-6, r["latency_ms"]["mean"]), 2)
            r["size_ratio_vs_fp32"] = round(r["size_mb"] / max(1e-6, base["size_mb"]), 3)

    # 选出最优 INT8（静态优先）
    best_int8 = static or dyn
    if best_int8:
        shutil.copy(best_int8, os.path.join(args.out, "best_int8.onnx"))
        report["best_int8"] = os.path.join(args.out, "best_int8.onnx")

    with open(os.path.join(args.out, "quant_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\n" + "=" * 74)
    print(" 模型对比（ONNX Runtime · CPU · imgsz {0}）".format(args.imgsz))
    print("=" * 74)
    print(" {0:<14}{1:>10}{2:>10}{3:>10}{4:>12}{5:>12}".format(
        "模型", "大小(MB)", "均值(ms)", "P95(ms)", "加速比", "相对体积"))
    for r in rows:
        print(" {0:<14}{1:>10.2f}{2:>10.2f}{3:>10.2f}{4:>12}{5:>12}".format(
            r["name"], r["size_mb"], r["latency_ms"]["mean"], r["latency_ms"]["p95"],
            r.get("speedup_vs_fp32", "-"), r.get("size_ratio_vs_fp32", "-")))
    print("\n报告: {0}/quant_report.json".format(args.out))
    if report.get("best_int8"):
        print("部署用 INT8: {0}（yolo 容器可用 ENGINE=opencv 加载，或 ENGINE=ort）".format(report["best_int8"]))


if __name__ == "__main__":
    main()
