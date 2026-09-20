#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
yolo_server.py —— 目标检测容器（YOLO / ONNX）
=====================================================================
职责：**只做目标检测**，输入一张图，输出货物框（不含颜色/形状判断）。
属性（颜色/形状/污渍）由 cnn 容器负责，2.5D 高度与二维码由 vision 容器负责
—— 三层分工见 README「视觉三层」。

模型：自训练 YOLOv8（工创yolo 项目 train_detection.py → export_models.py 产出 best.onnx）
      默认按 ultralytics 导出格式解析（输出 (1, 4+nc, N)）。
      · 单类 "goods"：nc=1，只检“货物”
      · 组合类别（方案B）：classes.json 给出类别名

端点（默认 :8101）：
  GET  /health        引擎/模型/类别/最近耗时
  POST /detect        上传图片 → {boxes:[{box,conf,cls}], elapsed_ms}
  GET  /model        模型信息（路径、类别、输入尺寸）

环境变量：
  MODEL_PATH   默认 /app/models/detect/goods_yolov8n_640_fp32.onnx
  CLASSES_JSON 类别名文件（可选，如 /app/models/detect/classes.json）
  IMGSZ        推理输入尺寸，默认 640（与训练一致；小目标可用 960）
  CONF_THRES   置信度阈值，默认 0.35
               （加入 223 张背景负样本重训后的实测：0.25→误检0.06/图·漏检0.03/图，
                 0.35~0.55→误检0.00/图·漏检0.03/图，0.65→漏检0.06/图。
                 取 0.35：误检已为 0，留出余量应对现场新货物）
  IOU_THRES    NMS IoU，默认 0.45
  ENGINE       opencv（默认）| ort（onnxruntime，可选）
"""
import json
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("yolo")

MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/detect/goods_yolov8n_640_fp32.onnx")
CLASSES_JSON = os.environ.get("CLASSES_JSON", "")
IMGSZ = int(os.environ.get("IMGSZ", "640"))
CONF_THRES = float(os.environ.get("CONF_THRES", "0.35"))
IOU_THRES = float(os.environ.get("IOU_THRES", "0.45"))
ENGINE = os.environ.get("ENGINE", "opencv").lower()
# OpenCV DNN 后端：auto=有 CUDA 就用 GPU FP16（JetPack 自带 OpenCV 带 CUDA），否则 CPU
#   cuda / cpu 可强制。Nano 上 CUDA FP16 通常比 CPU DNN 快 2~4 倍，且零额外依赖。
DNN_BACKEND = os.environ.get("DNN_BACKEND", "auto").lower()
PORT = int(os.environ.get("PORT", "8101"))

STATE = {"engine": None, "model": None, "classes": ["goods"], "loaded_at": None,
         "last_ms": None, "calls": 0, "name": os.path.basename(MODEL_PATH),
         "info": {}}
_ORT = {"session": None}


def load_manifest():
    """读 models/MANIFEST.json（模型登记表）：返回当前模型文件的指标/说明，便于现场核对。"""
    for cand in (os.path.join(os.path.dirname(MODEL_PATH), "..", "MANIFEST.json"),
                 "/app/models/MANIFEST.json"):
        cand = os.path.abspath(cand)
        if not os.path.exists(cand):
            continue
        try:
            man = json.load(open(cand, encoding="utf-8"))
            base = os.path.basename(MODEL_PATH)
            for v in (man.get("detector", {}) or {}).get("variants", []):
                if os.path.basename(v.get("file", "")) == base:
                    return {"manifest": cand, "md5_prefix": (v.get("md5") or "")[:12],
                            "precision": v.get("precision"), "arch": man["detector"].get("arch"),
                            "metrics": man["detector"].get("metrics"),
                            "metrics_false_positive": man["detector"].get("metrics_false_positive"),
                            "trained_on": man["detector"].get("trained_on")}
        except Exception as e:
            logger.warning("MANIFEST.json 读取失败: %s", e)
    return {}


def load_classes():
    if CLASSES_JSON and os.path.exists(CLASSES_JSON):
        with open(CLASSES_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = [data[str(i)] if str(i) in data else data[i] for i in sorted(data, key=lambda k: int(k))]
        return [str(c) for c in data]
    # 与老式 data.yaml 并存时的兜底
    for cand in (os.path.join(os.path.dirname(MODEL_PATH), "classes.txt"),):
        if os.path.exists(cand):
            return [l.strip() for l in open(cand, encoding="utf-8") if l.strip()]
    return ["goods"]



def _load_cudart():
    """加载 CUDA runtime（JetPack 上是 libcudart.so.10.2）"""
    import ctypes
    for name in ("libcudart.so.10.2", "libcudart.so.10.1", "libcudart.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    raise RuntimeError("找不到 libcudart.so（Jetson 上装 cuda-cudart-10-2）")


class TrtRunner(object):
    """TensorRT 引擎推理：tensorrt 绑定 + ctypes 调 cudart 管显存（不需要 pycuda）"""

    H2D, D2H = 1, 2      # cudaMemcpyHostToDevice / DeviceToHost

    def __init__(self, engine_path):
        import ctypes
        import tensorrt as trt
        self.ctypes = ctypes
        self.trt = trt
        self.cudart = _load_cudart()
        tlog = trt.Logger(trt.Logger.ERROR)
        with open(engine_path, "rb") as f:
            self.engine = trt.Runtime(tlog).deserialize_cuda_engine(f.read())
        if self.engine is None:
            raise RuntimeError("引擎反序列化失败（TRT 版本与构建时不一致？）")
        self.context = self.engine.create_execution_context()
        self.bindings, self.inputs, self.outputs = [], [], []
        for i in range(self.engine.num_bindings):
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            host = np.empty(int(np.prod(shape)), dtype)
            dev = ctypes.c_void_p()
            rc = self.cudart.cudaMalloc(ctypes.byref(dev), ctypes.c_size_t(host.nbytes))
            if rc != 0:
                raise RuntimeError("cudaMalloc 失败 rc=%d（显存不足？）" % rc)
            rec = {"name": self.engine.get_binding_name(i), "host": host, "dev": dev,
                   "shape": shape, "nbytes": host.nbytes, "dtype": dtype}
            self.bindings.append(int(dev.value))
            (self.inputs if self.engine.binding_is_input(i) else self.outputs).append(rec)
        self.stream = ctypes.c_void_p()
        self.cudart.cudaStreamCreate(ctypes.byref(self.stream))

    def infer(self, x):
        """x: (1,3,H,W) float32 → 输出列表（与 onnxruntime 的 run() 对齐）"""
        ct = self.ctypes
        inp = self.inputs[0]
        np.copyto(inp["host"], np.ascontiguousarray(x).ravel())
        self.cudart.cudaMemcpyAsync(inp["dev"], inp["host"].ctypes.data_as(ct.c_void_p),
                                    ct.c_size_t(inp["nbytes"]), self.H2D, self.stream)
        self.context.execute_async_v2(bindings=self.bindings,
                                      stream_handle=self.stream.value or 0)
        for out in self.outputs:
            self.cudart.cudaMemcpyAsync(out["host"].ctypes.data_as(ct.c_void_p), out["dev"],
                                        ct.c_size_t(out["nbytes"]), self.D2H, self.stream)
        self.cudart.cudaStreamSynchronize(self.stream)
        return [out["host"].reshape(out["shape"]).copy() for out in self.outputs]


def load_model():
    if ENGINE == "trt":
        # Jetson：直接加载 TensorRT 引擎（由 scripts/build-trt-on-jetson.sh 在本机生成）
        # 显存用 ctypes 直接调 CUDA runtime 管理 —— JetPack 上装不到 pycuda 轮子，
        # 而 tensorrt 的 python 绑定本身不需要 pycuda，只缺一个分配显存的工具。
        try:
            runner = TrtRunner(MODEL_PATH)
            STATE.update({"engine": "tensorrt", "model": MODEL_PATH,
                          "loaded_at": time.time(), "trt": runner,
                          "backend": "tensorrt(fp16)"})
            logger.info("已加载 TensorRT 引擎: {0}（输入 {1} 输出 {2}）".format(
                MODEL_PATH, runner.inputs[0]["shape"], runner.outputs[0]["shape"]))
            return
        except Exception as e:
            logger.warning("TensorRT 加载失败（{0}），回退 ONNX Runtime".format(e))
    if ENGINE == "ort":
        try:
            import onnxruntime as ort
            _ORT["session"] = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
            STATE.update({"engine": "onnxruntime", "model": MODEL_PATH, "loaded_at": time.time()})
            logger.info("已加载 ONNX（ONNX Runtime）: {0}".format(MODEL_PATH))
            return
        except Exception as e:
            logger.warning("ONNX Runtime 加载失败（{0}），回退 OpenCV DNN".format(e))
    import cv2
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            "未找到检测模型 {0}；请把工创yolo 导出的 ONNX 放到 "
            "models/detect/goods_yolov8n_640_fp32.onnx".format(MODEL_PATH))
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)
    backend = apply_dnn_backend(net)
    STATE.update({"engine": "opencv_dnn", "backend": backend, "model": MODEL_PATH,
                  "loaded_at": time.time(), "net": net})
    logger.info("已加载 ONNX（OpenCV DNN）: {0}（imgsz={1}, conf={2}, 后端 {3}）".format(
        MODEL_PATH, IMGSZ, CONF_THRES, backend))


def apply_dnn_backend(net):
    """选择 OpenCV DNN 后端：优先 Jetson 上的 CUDA FP16（零额外依赖），失败退回 CPU。

    Jetson（JetPack）自带的 python3-opencv 是带 CUDA 编译的，
    所以 Nano 上不用装 TensorRT/pycuda 也能先用上 GPU。
    """
    import cv2
    if not hasattr(cv2, "dnn"):        # apt 的 OpenCV 3.2 没有 dnn 模块
        logger.info("当前 OpenCV(%s) 无 dnn 模块 → 请用 ENGINE=ort", cv2.__version__)
        return "none"
    if DNN_BACKEND == "cpu":
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        return "cpu"
    try:
        if cv2.cuda.getCudaEnabledDeviceCount() > 0:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            return "cuda_fp16"
    except Exception as e:                      # 未编译 CUDA 的 OpenCV 会在这里失败
        if DNN_BACKEND == "cuda":
            logger.warning("强制 CUDA 后端但不可用: %s", e)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    return "cpu"



def blob_nchw(bgr):
    """letterbox 后的 BGR uint8 → NCHW float32 RGB。

    等价于 cv2.dnn.blobFromImage(blob, 1/255, (IMGSZ, IMGSZ), swapRB=True)，
    但**不依赖 cv2.dnn** —— Jetson 上 apt 的 OpenCV 3.2 没有 dnn 模块。
    """
    x = bgr[:, :, ::-1].astype(np.float32) * (1.0 / 255.0)
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def nms_numpy(boxes, scores, iou_thres):
    """纯 numpy NMS（OpenCV <3.3 没有 cv2.dnn.NMSBoxes）"""
    if not boxes:
        return []
    b = np.asarray(boxes, dtype=np.float32)
    sc = np.asarray(scores, dtype=np.float32)
    x1, y1 = b[:, 0], b[:, 1]
    x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
        w = np.maximum(0.0, xx2 - xx1); h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        iou = inter / np.maximum(1e-9, areas[i] + areas[rest] - inter)
        order = rest[iou <= iou_thres]
    return keep


def letterbox(bgr, size):
    import cv2
    h0, w0 = bgr.shape[:2]
    scale = min(size / float(w0), size / float(h0))
    nw, nh = int(round(w0 * scale)), int(round(h0 * scale))
    resized = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, np.uint8)
    dx, dy = (size - nw) // 2, (size - nh) // 2
    canvas[dy:dy + nh, dx:dx + nw] = resized
    return canvas, scale, dx, dy


def detect(bgr):
    """返回 [{box:[x1,y1,x2,y2], conf, cls, name}]（像素坐标，原图尺度）"""
    import cv2
    t0 = time.time()
    h0, w0 = bgr.shape[:2]
    blob, scale, dx, dy = letterbox(bgr, IMGSZ)
    boxes, confs, clss = [], [], []

    if STATE["engine"] == "tensorrt":
        preds = STATE["trt"].infer(blob_nchw(blob))[0]
    elif STATE["engine"] == "onnxruntime":
        sess = _ORT["session"]
        inp = sess.get_inputs()[0].name
        preds = sess.run(None, {inp: blob_nchw(blob)})[0]
    else:
        net = STATE["net"]
        net.setInput(blob_nchw(blob))
        preds = net.forward()

    p = np.asarray(preds)
    if p.ndim == 3:
        p = p[0]
    if p.shape[0] < p.shape[1]:          # (4+nc, N) → (N, 4+nc)
        p = p.T
    nc = p.shape[1] - 4
    for row in p:
        scores = row[4:4 + nc]
        c = int(np.argmax(scores))
        conf = float(scores[c])
        if conf < CONF_THRES:
            continue
        cx, cy, w, h = (float(v) for v in row[:4])
        boxes.append([cx - w / 2, cy - h / 2, w, h])
        confs.append(conf)
        clss.append(c)

    out = []
    if boxes:
        for i in nms_numpy(boxes, confs, IOU_THRES):
            x, y, w, h = boxes[i]
            x1 = (x - dx) / scale
            y1 = (y - dy) / scale
            x2 = x1 + w / scale
            y2 = y1 + h / scale
            name = STATE["classes"][clss[i]] if clss[i] < len(STATE["classes"]) else "cls{0}".format(clss[i])
            out.append({
                "box": [int(max(0, x1)), int(max(0, y1)), int(min(w0, x2)), int(min(h0, y2))],
                "box_norm": [round(max(0, x1) / w0, 4), round(max(0, y1) / h0, 4),
                             round(min(w, w0) / w0, 4), round(min(h, h0) / h0, 4)],
                "conf": round(confs[i], 3), "cls": clss[i], "name": name,
            })
    out.sort(key=lambda d: -(d["box"][2] - d["box"][0]) * (d["box"][3] - d["box"][1]))
    STATE["last_ms"] = round((time.time() - t0) * 1000, 1)
    STATE["calls"] += 1
    return out


def main():
    import cv2
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    import uvicorn

    STATE["classes"] = load_classes()
    STATE["info"] = load_manifest()
    try:
        load_model()
    except Exception as e:
        logger.error("模型加载失败: {0}（容器仍启动，/detect 返回 503）".format(e))

    app = FastAPI(title="sort-yolo", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        ready = STATE["engine"] is not None
        return {"ok": ready, "engine": STATE["engine"], "dnn_backend": STATE.get("backend"),
                "model": STATE["name"],
                "classes": STATE["classes"], "imgsz": IMGSZ, "conf_thres": CONF_THRES,
                "last_ms": STATE["last_ms"], "calls": STATE["calls"],
                "model_info": STATE["info"],        # 来自 models/MANIFEST.json：指标/精度/训练数据
                "error": None if ready else "模型未加载（检查 MODEL_PATH）"}

    @app.get("/model")
    def model_info():
        return {"path": MODEL_PATH, "classes": STATE["classes"], "imgsz": IMGSZ, "engine": STATE["engine"]}

    @app.post("/detect")
    async def detect_api(file: UploadFile = File(...)):
        if STATE["engine"] is None:
            return JSONResponse({"detail": "检测模型未加载", "boxes": []}, status_code=503)
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "无法解析图片")
        boxes = detect(img)
        return {"engine": STATE["engine"], "model": STATE["name"], "boxes": boxes,
                "count": len(boxes), "elapsed_ms": STATE["last_ms"],
                "size": [img.shape[1], img.shape[0]]}

    @app.post("/detect_array")
    async def detect_array(payload: dict):
        """内部接口：{"image_jpeg_base64": "..."} 或 {"boxes_only": true} 时用 /detect"""
        import base64
        raw = base64.b64decode(payload.get("image_jpeg_base64", ""))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "无法解析 base64 图片")
        return {"boxes": detect(img), "elapsed_ms": STATE["last_ms"]}

    logger.info("YOLO 检测服务: :{0} | 引擎 {1} | 类别 {2}".format(PORT, STATE["engine"], STATE["classes"]))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
