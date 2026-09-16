#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_trt_engine.py —— TensorRT INT8/FP16 引擎构建（在 Jetson 上执行）
=====================================================================
为什么需要：ONNX Runtime 的静态 INT8 量化会把 YOLOv8 检测头压塌（实测分数全 0），
但 TensorRT 的 INT8（带熵校准）对检测头处理正确，能在 Nano/Orin 上拿到 2~4× 加速。

用法（Jetson，JetPack 4.6.1 / TensorRT 8.0 / Python 3.6）：
  # 自检（不需要 TensorRT，只校验 ONNX 与校准图片列表）
  python3 tools/build_trt_engine.py --check --onnx exports_clean/best_fp32.onnx \
      --calib-dir data/real_synth_mix/images/train

  # 构建（INT8 + FP16 两个引擎，校准集用现场 100~300 张）
  python3 tools/build_trt_engine.py --onnx exports_clean/best_fp32.onnx \
      --calib-dir data/real_synth_mix/images/train --calib-n 200 --imgsz 640 \
      --out engines --bench 20

产出：engines/best_int8.engine · engines/best_fp16.engine · engines/calib.cache
部署：拷到 sort-web/models/yolo/ 并在 compose 里设 ENGINE=trt、MODEL_PATH=.../*.engine
"""
import argparse
import glob
import json
import os
import statistics
import time

import numpy as np


# ==================== 校准数据（与训练同一套预处理） ====================
def letterbox(img, size):
    import cv2
    h0, w0 = img.shape[:2]
    scale = min(size / float(w0), size / float(h0))
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return canvas


def preprocess(path, size):
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    blob = letterbox(img, size)
    x = blob[:, :, ::-1].astype(np.float32) / 255.0
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def build_calibrator(images, imgsz, cache_path, batch_size=4, max_batches=None):
    """TensorRT 熵校准器（JetPack 自带 tensorrt，pycuda 需 pip3 install pycuda）"""
    import pycuda.driver as cuda
    import tensorrt as trt

    class Calibrator(trt.IInt8EntropyCalibrator2):
        def __init__(self):
            trt.IInt8EntropyCalibrator2.__init__(self)
            self.images = list(images)
            self.imgsz = imgsz
            self.batch_size = batch_size
            self.idx = 0
            self.max_batches = max_batches or int(np.ceil(len(self.images) / float(batch_size)))
            self.device_input = cuda.mem_alloc(batch_size * 3 * imgsz * imgsz * 4)
            self.cache_path = cache_path

        def get_batch_size(self):
            return self.batch_size

        def get_batch(self, names):
            if self.idx >= min(len(self.images), self.max_batches * self.batch_size):
                return None
            batch = []
            while len(batch) < self.batch_size and self.idx < len(self.images):
                x = preprocess(self.images[self.idx], self.imgsz)
                self.idx += 1
                if x is not None:
                    batch.append(x)
            if not batch:
                return None
            while len(batch) < self.batch_size:                 # 末批补齐（复制最后一帧）
                batch.append(batch[-1])
            data = np.ascontiguousarray(np.stack(batch)).astype(np.float32)
            cuda.memcpy_htod(self.device_input, data)
            return [int(self.device_input)]

        def read_calibration_cache(self):
            if os.path.exists(self.cache_path):
                with open(self.cache_path, "rb") as f:
                    return f.read()
            return None

        def write_calibration_cache(self, cache):
            with open(self.cache_path, "wb") as f:
                f.write(cache)

    return Calibrator()


# ==================== 引擎构建 ====================
def build_engine(onnx_path, out_path, imgsz, calibrator=None, fp16=True, int8=False, verbose=False):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.INFO if verbose else trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            errs = [parser.get_error(i).desc() for i in range(parser.num_errors)]
            raise RuntimeError("ONNX 解析失败: {0}".format(errs[:3]))
    config = builder.create_builder_config()
    config.max_workspace_size = 1 << 30                            # 1GB（Nano 8GB 内存充足）
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    if int8:
        if not builder.platform_has_fast_int8:
            raise RuntimeError("该平台不支持 INT8")
        config.set_flag(trt.BuilderFlag.INT8)
        config.set_flag(trt.BuilderFlag.STRICT_TYPES) if False else None
        if calibrator is not None:
            config.int8_calibrator = calibrator
    profile = builder.create_optimization_profile()
    name = network.get_input(0).name
    profile.set_shape(name, (1, 3, imgsz, imgsz), (1, 3, imgsz, imgsz), (1, 3, imgsz, imgsz))
    config.add_optimization_profile(profile)
    engine = builder.build_engine(network, config)
    if engine is None:
        raise RuntimeError("引擎构建失败（可尝试降低 imgsz 或改用 FP16）")
    with open(out_path, "wb") as f:
        f.write(engine.serialize())
    return out_path


def bench_engine(engine_path, images, imgsz, runs=20):
    """用 TensorRT runtime + pycuda 实测延迟"""
    import pycuda.autoinit  # noqa: F401
    import pycuda.driver as cuda
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.ERROR)
    with open(engine_path, "rb") as f:
        engine = trt.Runtime(logger).deserialize_cuda_engine(f.read())
    context = engine.create_execution_context()
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for i in range(engine.num_bindings):
        name = engine.get_binding_name(i)
        shape = engine.get_binding_shape(i)
        size = int(np.prod(shape))
        dtype = trt.nptype(engine.get_binding_dtype(i))
        host = cuda.pagelocked_empty(size, dtype)
        dev = cuda.mem_alloc(host.nbytes)
        bindings.append(int(dev))
        (inputs if engine.binding_is_input(i) else outputs).append({"name": name, "host": host, "dev": dev})
    lat = []
    for p in images[:runs]:
        x = preprocess(p, imgsz)
        if x is None:
            continue
        np.copyto(inputs[0]["host"], x.ravel())
        t0 = time.time()
        cuda.memcpy_htod_async(inputs[0]["dev"], inputs[0]["host"], stream)
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        for o in outputs:
            cuda.memcpy_dtoh_async(o["host"], o["dev"], stream)
        stream.synchronize()
        lat.append((time.time() - t0) * 1000)
    if not lat:
        return None
    return {"latency_ms_mean": round(statistics.mean(lat), 2),
            "latency_ms_p95": round(sorted(lat)[int(len(lat) * 0.95) - 1], 2),
            "size_mb": round(os.path.getsize(engine_path) / 1e6, 2)}


def main():
    ap = argparse.ArgumentParser(description="TensorRT INT8/FP16 引擎构建（Jetson）")
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib-dir", help="校准图片目录（现场 100~300 张最佳）")
    ap.add_argument("--calib-n", type=int, default=200)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", default="engines")
    ap.add_argument("--batch", type=int, default=4, help="校准批大小（Nano 建议 2~4）")
    ap.add_argument("--bench", type=int, default=20, help="基准测试帧数（0=跳过）")
    ap.add_argument("--fp16-only", action="store_true")
    ap.add_argument("--check", action="store_true", help="只做自检（不构建引擎，可无 TensorRT）")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    images = []
    if args.calib_dir:
        images = sorted(glob.glob(os.path.join(args.calib_dir, "*.*")))
        images = [p for p in images if os.path.splitext(p)[1].lower() in (".jpg", ".jpeg", ".png", ".bmp")]
        print("[校准集] {0} 张（取前 {1} 张）".format(len(images), min(args.calib_n, len(images))))

    if args.check:
        import onnx
        m = onnx.load(args.onnx)
        print("[自检] ONNX 可加载: {0} | 输入 {1} | opset {2} | 体积 {3:.2f} MB".format(
            os.path.basename(args.onnx), m.graph.input[0].name,
            m.opset_import[0].version, os.path.getsize(args.onnx) / 1e6))
        if images:
            x = preprocess(images[0], args.imgsz)
            print("[自检] 首张校准图预处理形状 {0} dtype {1} 值域 [{2:.3f},{3:.3f}]".format(
                x.shape, x.dtype, float(x.min()), float(x.max())))
        print("[自检] 通过；在 Jetson 上执行本脚本（去掉 --check）即可构建引擎")
        return

    try:
        import tensorrt as trt  # noqa: F401
    except ImportError:
        raise SystemExit("未找到 TensorRT。Jetson 上由 JetPack 自带（python3 -c 'import tensorrt' 应可用）；"
                         "x86 上可用 nvcr.io/nvidia/tensorrt 容器。")
    try:
        import pycuda.driver  # noqa: F401
    except ImportError:
        raise SystemExit("未找到 pycuda，请先安装：pip3 install pycuda")

    os.makedirs(args.out, exist_ok=True)
    report = {"onnx": args.onnx, "imgsz": args.imgsz, "calib_images": min(len(images), args.calib_n),
              "engines": []}

    # FP16
    fp16_path = os.path.join(args.out, "best_fp16.engine")
    print("[1/2] 构建 FP16 引擎 …")
    build_engine(args.onnx, fp16_path, args.imgsz, fp16=True, int8=False, verbose=args.verbose)
    print("   {0}  {1:.2f} MB".format(fp16_path, os.path.getsize(fp16_path) / 1e6))

    # INT8（带熵校准）
    if not args.fp16_only:
        int8_path = os.path.join(args.out, "best_int8.engine")
        print("[2/2] 构建 INT8 引擎（校准 {0} 张）…".format(min(len(images), args.calib_n)))
        cal = build_calibrator(images[:args.calib_n], args.imgsz,
                              os.path.join(args.out, "calib.cache"), batch_size=args.batch)
        build_engine(args.onnx, int8_path, args.imgsz, calibrator=cal, fp16=True, int8=True,
                     verbose=args.verbose)
        print("   {0}  {1:.2f} MB".format(int8_path, os.path.getsize(int8_path) / 1e6))

    if args.bench:
        print("[基准] TensorRT 实机延迟（{0} 帧）…".format(args.bench))
        for p in ([fp16_path] + ([int8_path] if not args.fp16_only else [])):
            try:
                r = bench_engine(p, images, args.imgsz, runs=args.bench)
            except Exception as e:
                print("   {0} 基准失败: {1}".format(os.path.basename(p), str(e)[:100])); continue
            if r:
                r["name"] = os.path.basename(p)
                report["engines"].append(r)
                print("   {name:<22} 均值 {latency_ms_mean}ms  P95 {latency_ms_p95}ms  {size_mb}MB".format(**r))
    with open(os.path.join(args.out, "trt_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print("\n报告:", os.path.join(args.out, "trt_report.json"))
    print("部署：cp {0}/*.engine ../sort-web/models/yolo/ 并在 compose 设 ENGINE=trt".format(args.out))


if __name__ == "__main__":
    main()
