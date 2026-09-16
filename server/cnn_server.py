#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
cnn_server.py —— 属性识别容器（颜色 / 形状 / 污渍缺陷，小 CNN · ONNX）
=====================================================================
职责：**只做属性分类**。输入一张图 + 若干检测框，输出每个框的颜色/形状/污渍缺陷。
目标检测由 yolo 容器负责，2.5D 高度与二维码由 vision 容器负责（三层分工）。

模型：工创yolo 项目 train_attributes.py 训练 + --export-onnx 产出
      · /app/models/attr/color.onnx  (+ color_classes.json)
      · /app/models/attr/shape.onnx  (+ shape_classes.json)
      · /app/models/attr/stain.onnx  (+ stain_classes.json)
      输入 96×96×3（ImageNet 归一化），输出 logits。

端点（默认 :8102）：
  GET  /health                已加载任务、每任务类别数与最近耗时
  POST /classify              上传图片 + 表单字段 boxes（JSON）→ 每框属性
  POST /classify_array        {"image_jpeg_base64": "...", "boxes": [[x1,y1,x2,y2], ...]}
  GET  /model                 模型与类别信息

环境变量：
  MODEL_DIR     默认 /app/models/attr
  TASKS         默认 color,shape,stain
  IMG_SIZE      默认 96（与训练一致）
  CROP_MARGIN   裁剪外扩比例，默认 0.12
  ENGINE        opencv（默认）| ort
  PORT          默认 8102
"""
import base64
import json
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cnn")

MODEL_DIR = os.environ.get("MODEL_DIR", "/app/models/attr")
TASKS = [t.strip() for t in os.environ.get("TASKS", "color,shape,stain").split(",") if t.strip()]
IMG_SIZE = int(os.environ.get("IMG_SIZE", "96"))
CROP_MARGIN = float(os.environ.get("CROP_MARGIN", "0.12"))
ENGINE = os.environ.get("ENGINE", "opencv").lower()
PORT = int(os.environ.get("PORT", "8102"))

_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)

MODELS = {}      # task -> {"net": cv2 net | ort session, "classes": [...], "last_ms": None}
STATE = {"engine": ENGINE, "loaded": [], "calls": 0}


def load_classes(task):
    path = os.path.join(MODEL_DIR, "{0}_classes.json".format(task))
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return [data[str(i)] if str(i) in data else data[i] for i in sorted(data, key=lambda k: int(k))]
        return [str(c) for c in data]
    defaults = {
        "color": ["red", "orange", "yellow", "green", "cyan", "blue", "purple", "black", "white"],
        "shape": ["cube", "cuboid", "cylinder", "ball", "tetra", "prism5", "prism6"],
        "stain": ["clean", "stain", "defect"],
    }
    return defaults.get(task, [])


def load_models():
    import cv2
    for task in TASKS:
        path = os.path.join(MODEL_DIR, "{0}.onnx".format(task))
        if not os.path.exists(path):
            logger.warning("缺少模型 {0}（任务 {1} 将返回 unknown）".format(path, task))
            continue
        try:
            if ENGINE == "ort":
                import onnxruntime as ort
                MODELS[task] = {"session": ort.InferenceSession(path, providers=["CPUExecutionProvider"]),
                                "classes": load_classes(task), "last_ms": None, "engine": "onnxruntime"}
            else:
                MODELS[task] = {"net": cv2.dnn.readNetFromONNX(path), "classes": load_classes(task),
                                "last_ms": None, "engine": "opencv_dnn"}
            STATE["loaded"].append(task)
            logger.info("已加载 {0} 模型: {1}（{2} 类）".format(task, os.path.basename(path),
                                                              len(MODELS[task]["classes"])))
        except Exception as e:
            logger.warning("模型 {0} 加载失败: {1}".format(task, e))


def crop(bgr, box, margin=CROP_MARGIN):
    h, w = bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    pad = int(max(x2 - x1, y2 - y1) * margin)
    x1, y1 = max(0, x1 - pad), max(0, y1 - pad)
    x2, y2 = min(w, x2 + pad), min(h, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return None
    return bgr[y1:y2, x1:x2]


def preprocess(crop_bgr):
    """96×96 + ImageNet 归一化 + NCHW（与 train_attributes.py 完全一致）"""
    import cv2
    img = cv2.resize(crop_bgr, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = (img - _MEAN) / _STD
    return img.transpose(2, 0, 1)[None, ...]


def classify_task(task, crop_bgr):
    m = MODELS.get(task)
    if m is None:
        return None
    t0 = time.time()
    x = preprocess(crop_bgr)
    if m.get("engine") == "onnxruntime":
        inp = m["session"].get_inputs()[0].name
        out = m["session"].run(None, {inp: x})[0]
    else:
        m["net"].setInput(x)
        out = m["net"].forward()
    logits = np.asarray(out).reshape(-1).astype(np.float32)
    exp = np.exp(logits - logits.max())
    prob = exp / exp.sum()
    idx = int(prob.argmax())
    classes = m["classes"]
    m["last_ms"] = round((time.time() - t0) * 1000, 2)
    return {"label": classes[idx] if idx < len(classes) else "cls{0}".format(idx),
            "index": idx, "confidence": round(float(prob[idx]), 4),
            "elapsed_ms": m["last_ms"],
            "top3": [{"label": classes[i] if i < len(classes) else str(i),
                      "confidence": round(float(prob[i]), 4)}
                     for i in np.argsort(-prob)[:3]]}


def classify(bgr, boxes):
    """对每个框做 color/shape/stain 分类；返回与 boxes 等长的列表"""
    out = []
    for box in boxes:
        c = crop(bgr, box)
        if c is None or c.size == 0:
            out.append({"error": "裁剪失败（框越界）"})
            continue
        item = {"box": [int(v) for v in box]}
        for task in TASKS:
            r = classify_task(task, c)
            if r is None:
                item[task] = {"label": "unknown", "confidence": 0.0}
            else:
                item[task] = r
        out.append(item)
    STATE["calls"] += 1
    return out


def main():
    import cv2
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import JSONResponse
    import uvicorn

    load_models()
    app = FastAPI(title="sort-cnn", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return {"ok": bool(MODELS), "engine": ENGINE, "tasks": TASKS,
                "loaded": STATE["loaded"], "calls": STATE["calls"],
                "detail": {t: {"classes": len(MODELS[t]["classes"]), "last_ms": MODELS[t]["last_ms"]}
                           for t in MODELS}}

    @app.get("/model")
    def model_info():
        return {"dir": MODEL_DIR, "tasks": {t: MODELS[t]["classes"] for t in MODELS}}

    @app.post("/classify")
    async def classify_api(file: UploadFile = File(...), boxes: str = Form(...)):
        try:
            box_list = json.loads(boxes)
        except Exception:
            raise HTTPException(400, "boxes 必须是 JSON 数组，如 [[x1,y1,x2,y2], ...]")
        data = await file.read()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "无法解析图片")
        return {"results": classify(img, box_list), "size": [img.shape[1], img.shape[0]]}

    @app.post("/classify_array")
    async def classify_array(payload: dict):
        raw = base64.b64decode(payload.get("image_jpeg_base64", ""))
        img = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "无法解析 base64 图片")
        boxes = payload.get("boxes") or []
        return {"results": classify(img, boxes)}

    logger.info("CNN 属性服务: :{0} | 引擎 {1} | 任务 {2} | 已加载 {3}".format(
        PORT, ENGINE, TASKS, STATE["loaded"]))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
