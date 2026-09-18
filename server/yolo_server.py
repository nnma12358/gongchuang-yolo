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


def load_model():
    if ENGINE == "trt":
        # Jetson：直接加载 TensorRT 引擎（由 工创yolo tools/build_trt_engine.py 生成）
        try:
            import pycuda.autoinit  # noqa: F401
            import pycuda.driver as cuda
            import tensorrt as trt
            logger_t = trt.Logger(trt.Logger.ERROR)
            with open(MODEL_PATH, "rb") as f:
                engine = trt.Runtime(logger_t).deserialize_cuda_engine(f.read())
            if engine is None:
                raise RuntimeError("引擎反序列化失败")
            context = engine.create_execution_context()
            stream = cuda.Stream()
            io, bindings = {"inputs": [], "outputs": []}, []
            for i in range(engine.num_bindings):
                shape = engine.get_binding_shape(i)
                size = int(np.prod(shape))
                dtype = trt.nptype(engine.get_binding_dtype(i))
                host = cuda.pagelocked_empty(size, dtype)
                dev = cuda.mem_alloc(host.nbytes)
                bindings.append(int(dev))
                rec = {"name": engine.get_binding_name(i), "host": host, "dev": dev, "shape": shape}
                io["inputs" if engine.binding_is_input(i) else "outputs"].append(rec)
            STATE.update({"engine": "tensorrt", "model": MODEL_PATH, "loaded_at": time.time(),
                          "trt": {"engine": engine, "context": context, "stream": stream,
                                  "bindings": bindings, "io": io}})
            logger.info("已加载 TensorRT 引擎: {0}".format(MODEL_PATH))
            return
        except Exception as e:
            logger.warning("TensorRT 加载失败（{0}），回退 OpenCV DNN".format(e))
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
    STATE.update({"engine": "opencv_dnn", "model": MODEL_PATH, "loaded_at": time.time(),
                  "net": cv2.dnn.readNetFromONNX(MODEL_PATH)})
    logger.info("已加载 ONNX（OpenCV DNN）: {0}（imgsz={1}, conf={2}）".format(MODEL_PATH, IMGSZ, CONF_THRES))


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
        import pycuda.driver as cuda
        t = STATE["trt"]
        x = cv2.dnn.blobFromImage(blob, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True)
        np.copyto(t["io"]["inputs"][0]["host"], np.ascontiguousarray(x).ravel())
        cuda.memcpy_htod_async(t["io"]["inputs"][0]["dev"], t["io"]["inputs"][0]["host"], t["stream"])
        t["context"].execute_async_v2(bindings=t["bindings"], stream_handle=t["stream"].handle)
        for o in t["io"]["outputs"]:
            cuda.memcpy_dtoh_async(o["host"], o["dev"], t["stream"])
        t["stream"].synchronize()
        preds = np.array(t["io"]["outputs"][0]["host"]).reshape(t["io"]["outputs"][0]["shape"])
    elif STATE["engine"] == "onnxruntime":
        sess = _ORT["session"]
        inp = sess.get_inputs()[0].name
        x = cv2.dnn.blobFromImage(blob, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True)
        preds = sess.run(None, {inp: x})[0]
    else:
        net = STATE["net"]
        x = cv2.dnn.blobFromImage(blob, 1 / 255.0, (IMGSZ, IMGSZ), swapRB=True)
        net.setInput(x)
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
        idx = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRES, IOU_THRES)
        for i in (idx.flatten() if idx is not None and len(idx) else []):
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
        return {"ok": ready, "engine": STATE["engine"], "model": STATE["name"],
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
