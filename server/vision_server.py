#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视觉容器 —— 货物识别与标记输出（Jetson Nano · Python 3.6+）
=====================================================================
职责：独占相机 → 连续识别 → 输出“标记（marks）”，供网关容器做自动分拣。

对外接口（默认端口 8100）：
  GET  /health            相机/引擎状态、帧率、最近标记时间
  GET  /detect/frame      抓取当前帧并识别 → 标记 JSON（网关自动分拣循环调用）
  POST /detect            上传图片识别（联调/离线复核）
  GET  /frame.jpg         当前帧 JPEG（?draw=1 输出带标记的图像）
  GET  /marks/latest      最近一次标记 JSON
  GET  /stream.mjpg       可选：带标记的 MJPEG 流（现场手机/大屏旁路查看）

相机来源（环境变量）：
  CAMERA_SOURCE=synthetic  合成托盘场景（无硬件联调/演示）
  CAMERA_INDEX=0           本机 USB 相机（需 --device /dev/video0）
  CAMERA_URL=rtsp://...    网络相机 / MJPEG 流
"""
import base64
import json
import logging
import math
import os
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, Response

import detect_core
from catalog import GOODS, MARKS, SHAPES, COLORS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("vision")

# ==================== 配置 ====================
PORT = int(os.environ.get("PORT", "8100"))
CAMERA_SOURCE = os.environ.get("CAMERA_SOURCE", "auto").lower()   # auto|synthetic|index|url
CAMERA_INDEX = int(os.environ.get("CAMERA_INDEX", "0"))
CAMERA_URL = os.environ.get("CAMERA_URL", "")
CAMERA_IMAGE = os.environ.get("CAMERA_IMAGE", "")     # CAMERA_SOURCE=image 时的回放图片路径
CAMERA_WIDTH = int(os.environ.get("CAMERA_WIDTH", "1280"))
CAMERA_HEIGHT = int(os.environ.get("CAMERA_HEIGHT", "720"))
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/yolov8n.onnx")
# 深度来源（Astra 深度快照，如 ROS2 监控页 http://127.0.0.1:8088/depth.png，uint16 PNG）；
# 留空则仅用 RGB（抓取解算走平面单应路径）
DEPTH_URL = os.environ.get("DEPTH_URL", "")
# 逐像素桌面参考标定（2.5D 定位软件产物 table_reference.npz）；
# 未提供时自动在 TABLE_REF_DIR 下寻找最新的 table_reference.npz
TABLE_REF = os.environ.get("TABLE_REF", "")
TABLE_REF_DIR = os.environ.get("TABLE_REF_DIR", "/app/calibration")
_TABLE_CACHE = {"ref": None, "tried": False}
DEPTH_CACHE = {"ts": 0.0, "img": None}
DETECT_ENGINE = os.environ.get("DETECT_ENGINE", "classic").lower()   # classic | yolo
# 三层分工：yolo 容器做目标检测、cnn 容器做颜色/形状/污渍属性、本容器做编排
#   + 2.5D 高度、二维码/文字解码、标记输出（检测失败自动回退经典 CV 引擎）
YOLO_URL = os.environ.get("YOLO_URL", "").rstrip("/")
CNN_URL = os.environ.get("CNN_URL", "").rstrip("/")
YOLO_TIMEOUT = float(os.environ.get("YOLO_TIMEOUT", "3.0"))
CNN_TIMEOUT = float(os.environ.get("CNN_TIMEOUT", "3.0"))
CONF_MIN = float(os.environ.get("CONF_MIN", "0.45"))
MIN_AREA_RATIO = float(os.environ.get("MIN_AREA_RATIO", "0.002"))
STAIN_TH = float(os.environ.get("STAIN_DARK_RATIO", "0.05"))
DEFECT_TH = float(os.environ.get("DEFECT_DARK_RATIO", "0.14"))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "80"))
LOOP_FPS = float(os.environ.get("VISION_FPS", "10"))       # 采集/识别循环频率
STREAM_FPS = float(os.environ.get("STREAM_FPS", "8"))      # MJPEG 推流频率

app = FastAPI(title="sort-vision", docs_url=None, redoc_url=None)

# ==================== 相机与帧缓冲 ====================
_frame_lock = threading.Lock()
_latest = {"bgr": None, "ts": 0.0, "index": 0}
_marks_lock = threading.Lock()
_latest_marks = {"ts": 0.0, "detections": [], "qr_text": "", "frame_index": 0, "size": [0, 0]}
STATS = {"capture_fps": 0.0, "detect_fps": 0.0, "frames": 0, "detections": 0,
         "camera": "disconnected", "engine": DETECT_ENGINE, "started_at": time.time()}


class SyntheticCamera(object):
    """合成托盘场景：随机摆放 1 件货物（形状/颜色/污渍/缺陷/二维码），
    用于无硬件联调与流程演示（相机接入后自动让位给真实相机）"""

    def __init__(self, width=1280, height=720):
        self.w, self.h = width, height
        self.seed = int(time.time()) % 100000
        self._hold_until = 0
        self._current = None

    def _render(self, goods, marks, qr_text):
        """俯视视角渲染：按形状画出顶面剪影（与现场俯视相机一致）+ 污渍/缺陷 + 二维码贴纸"""
        import cv2
        import numpy as np
        img = np.full((self.h, self.w, 3), 235, dtype=np.uint8)     # 浅色托盘
        cv2.rectangle(img, (60, 50), (self.w - 60, self.h - 50), (198, 198, 198), 3)
        img[50:self.h - 50, 60:60 + 6] = (170, 170, 170)            # 托盘边缘(≥10mm)示意

        rgb = goods["rgb"]
        bgr = tuple(int(v) for v in rgb[::-1])
        dark = tuple(int(v * 0.72) for v in bgr)
        light = tuple(min(255, int(v * 1.18)) for v in bgr)
        cx, cy, r = self.w // 2, self.h // 2, 120
        geo = goods["geometry"]

        if geo in ("cube", "cuboid"):
            hw = r if geo == "cube" else int(r * 1.25)
            cv2.rectangle(img, (cx - hw, cy - r), (cx + hw, cy + r), bgr, -1)          # 顶面
            cv2.rectangle(img, (cx - hw, cy + r - 34), (cx + hw, cy + r), dark, -1)    # 侧面阴影
            cv2.rectangle(img, (cx - hw, cy - r), (cx + hw, cy - r + 26), light, -1)   # 高光
        elif geo in ("cylinder", "ball"):
            cv2.ellipse(img, (cx, cy + 16), (r, int(r * 0.62)), 0, 0, 360, dark, -1)   # 底部投影
            cv2.circle(img, (cx, cy), r, bgr, -1)
            cv2.circle(img, (cx - int(r * 0.25), cy - int(r * 0.25)), int(r * 0.55), light, -1)
            if geo == "cylinder":
                cv2.ellipse(img, (cx, cy - int(r * 0.72)), (r, int(r * 0.26)), 0, 0, 360, light, -1)
        else:
            n = {"tetra": 3, "prism5": 5, "prism6": 6}.get(geo, 4)
            pts = []
            for i in range(n):
                ang = -math.pi / 2 + i * 2 * math.pi / n
                pts.append([int(cx + r * math.cos(ang)), int(cy + r * math.sin(ang))])
            cv2.fillPoly(img, [np.array(pts, dtype=np.int32)], bgr)
            cv2.polylines(img, [np.array(pts, dtype=np.int32)], True, dark, 6)
            cv2.circle(img, (cx - int(r * 0.2), cy - int(r * 0.2)), int(r * 0.35), light, -1)

        if "污渍" in marks:
            cv2.circle(img, (cx + int(r * 0.3), cy + int(r * 0.2)), 26, (42, 55, 74), -1)
            cv2.circle(img, (cx - int(r * 0.1), cy + int(r * 0.4)), 15, (51, 70, 92), -1)
        if "缺陷" in marks:
            cv2.rectangle(img, (cx - int(r * 0.6), cy - int(r * 0.5)),
                          (cx - int(r * 0.25), cy - int(r * 0.28)), (28, 28, 28), -1)
            cv2.line(img, (cx + int(r * 0.1), cy + int(r * 0.5)),
                     (cx + int(r * 0.55), cy - int(r * 0.1)), (20, 20, 20), 5)

        if qr_text:
            try:
                import qrcode  # 可选依赖；未安装则跳过二维码贴纸
                q = qrcode.make(qr_text)
                arr = np.array(q.convert("RGB").resize((104, 104)))[:, :, ::-1]
                img[cy + r - 118:cy + r - 14, cx - r + 6:cx - r + 110] = arr
            except Exception:
                pass
        cv2.putText(img, "SYNTHETIC TRAY  {0}{1}".format(goods["color"], goods["shape"]),
                    (70, self.h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (150, 150, 150), 2)
        return img

    def read(self):
        """返回 (ok, frame)；每 ~4s 换一件新货物，模拟“放上一件 → 取走 → 再放一件”"""
        import random
        now = time.time()
        if self._current is None or now > self._hold_until:
            random.seed(int(now) + self.seed)
            goods = random.choice(GOODS)
            marks = random.choice([[], ["污渍"], ["缺陷"]])
            qr = "SORT-TASK:T-{0}".format(random.choice(["260901", "260902", "260903", "260904"]))
            self._current = (goods, marks, qr)
            self._hold_until = now + random.uniform(3.0, 5.0)
            # 中间 0.6s 的空托盘（模拟取走），便于自动分拣识别“有新货物”
            self._empty_until = now + 0.6
        goods, marks, qr = self._current
        if now < getattr(self, "_empty_until", 0):
            import cv2
            import numpy as np
            img = np.full((self.h, self.w, 3), 235, dtype=np.uint8)
            cv2.rectangle(img, (40, 40), (self.w - 40, self.h - 40), (200, 200, 200), 2)
            cv2.putText(img, "EMPTY TRAY", (60, self.h - 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (150, 150, 150), 2)
            return True, img
        return True, self._render(goods, marks, qr)

    def release(self):
        pass


class ImageCamera(object):
    """离线图片回放源：CAMERA_SOURCE=image + CAMERA_IMAGE=xxx.jpg
    用途：现场复盘、无相机联调、验证识别链路（对同一帧反复识别）"""

    def __init__(self, path):
        import cv2
        self.img = cv2.imread(path, cv2.IMREAD_COLOR)
        if self.img is None:
            raise RuntimeError("无法读取回放图片: {0}".format(path))
        self.path = path

    def read(self):
        return True, self.img.copy()

    def release(self):
        pass


def open_camera():
    """按配置打开相机：synthetic / image / 索引 / URL"""
    import cv2
    if CAMERA_SOURCE == "image":
        logger.info("相机来源：离线图片回放 {0}".format(CAMERA_IMAGE or "(未指定)"))
        return ImageCamera(CAMERA_IMAGE)
    if CAMERA_SOURCE == "synthetic":
        logger.info("相机来源：合成托盘场景（联调模式）")
        return SyntheticCamera(CAMERA_WIDTH, CAMERA_HEIGHT)
    if CAMERA_URL:
        logger.info("相机来源：{0}".format(CAMERA_URL))
        cap = cv2.VideoCapture(CAMERA_URL)
    else:
        logger.info("相机来源：/dev/video{0} ({1}x{2})".format(CAMERA_INDEX, CAMERA_WIDTH, CAMERA_HEIGHT))
        cap = cv2.VideoCapture(CAMERA_INDEX)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    if not cap.isOpened():
        if CAMERA_SOURCE == "auto":
            logger.warning("相机打开失败（{0}），自动切换到合成场景".format(CAMERA_URL or CAMERA_INDEX))
            return SyntheticCamera(CAMERA_WIDTH, CAMERA_HEIGHT)
        raise RuntimeError("无法打开相机")
    return cap


def capture_loop():
    """采集线程：持续读取最新帧，断线自动重连"""
    cap = None
    fps_t0, fps_n = time.time(), 0
    while True:
        try:
            if cap is None:
                cap = open_camera()
                STATS["camera"] = ("synthetic" if isinstance(cap, SyntheticCamera)
                                   else "image" if isinstance(cap, ImageCamera) else "connected")
            ok, frame = cap.read()
            if not ok or frame is None:
                logger.warning("读取帧失败，重连相机 …")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                STATS["camera"] = "reconnecting"
                time.sleep(1.0)
                continue
            with _frame_lock:
                _latest["bgr"] = frame
                _latest["ts"] = time.time()
                _latest["index"] += 1
                STATS["frames"] = _latest["index"]
            fps_n += 1
            now = time.time()
            if now - fps_t0 >= 2.0:
                STATS["capture_fps"] = round(fps_n / (now - fps_t0), 1)
                fps_t0, fps_n = now, 0
        except Exception as e:
            logger.warning("采集异常: {0}".format(e))
            time.sleep(1.0)
        time.sleep(max(0.0, 1.0 / max(1.0, LOOP_FPS)))


def detect_loop():
    """识别线程：对最新帧做识别，更新标记（与采集解耦，避免拖慢画面）"""
    fps_t0, fps_n = time.time(), 0
    while True:
        with _frame_lock:
            frame = None if _latest["bgr"] is None else _latest["bgr"].copy()
            idx = _latest["index"]
            ts = _latest["ts"]
        if frame is not None and ts > _latest_marks.get("ts", 0):
            try:
                dets, _engine = run_detection(frame)
                qr_text, size = "", (frame.shape[1], frame.shape[0])
                if not dets:
                    _d, qr_text, size = detect_core.analyze(
                        frame, conf_min=CONF_MIN, min_area_ratio=MIN_AREA_RATIO,
                        stain_th=STAIN_TH, defect_th=DEFECT_TH)
                with _marks_lock:
                    _latest_marks.update({"ts": time.time(), "detections": dets,
                                          "qr_text": qr_text, "frame_index": idx, "size": list(size)})
                STATS["detections"] = len(dets)
                fps_n += 1
            except Exception as e:
                logger.warning("识别异常: {0}".format(e))
        now = time.time()
        if now - fps_t0 >= 2.0:
            STATS["detect_fps"] = round(fps_n / (now - fps_t0), 1)
            fps_t0, fps_n = now, 0
        time.sleep(max(0.05, 1.0 / max(1.0, LOOP_FPS)))


@app.on_event("startup")
def _startup():
    if DETECT_ENGINE == "onnx":
        detect_core.load_onnx(MODEL_PATH)
    threading.Thread(target=capture_loop, daemon=True).start()
    threading.Thread(target=detect_loop, daemon=True).start()
    logger.info("视觉容器已启动：端口 {0} · 识别频率 {1}Hz · 引擎 {2}".format(
        PORT, LOOP_FPS, DETECT_ENGINE))


@app.get("/health")
async def health():
    with _marks_lock:
        last = _latest_marks["ts"]
    return {"ok": True, "camera": STATS["camera"], "engine": STATS["engine"],
            "capture_fps": STATS["capture_fps"], "detect_fps": STATS["detect_fps"],
            "frames": STATS["frames"], "detections": STATS["detections"],
            "last_marks_age": round(time.time() - last, 2) if last else None,
            "source": CAMERA_SOURCE if CAMERA_SOURCE != "auto" else (CAMERA_URL or "index:{0}".format(CAMERA_INDEX)),
            "depth_url": DEPTH_URL or None,
            "table_ref": (_TABLE_CACHE["ref"] or {}).get("path")}


@app.get("/detect/frame")
async def detect_frame(max_age: float = 1.5):
    """抓取当前帧并识别 → 标记 JSON（网关自动分拣循环主入口）"""
    with _frame_lock:
        frame = None if _latest["bgr"] is None else _latest["bgr"].copy()
        idx = _latest["index"]
        ts = _latest["ts"]
    if frame is None:
        raise HTTPException(503, "相机尚无可用帧（检查相机接入或 CAMERA_SOURCE）")
    if time.time() - ts > max_age:
        logger.warning("帧已过期 {0:.1f}s".format(time.time() - ts))
    dets, engine_used = run_detection(frame)
    size = (frame.shape[1], frame.shape[0])
    qr_text = ""
    if not dets:
        _d, qr_text, size = detect_core.analyze(
            frame, conf_min=CONF_MIN, min_area_ratio=MIN_AREA_RATIO,
            stain_th=STAIN_TH, defect_th=DEFECT_TH)
    depth = fetch_depth()
    if depth is not None:
        detect_core.attach_depth(dets, depth, table_ref=load_table_reference())
    elif load_table_reference() is not None:
        detect_core.attach_depth(dets, None, table_ref=load_table_reference())
    marks = {"ts": round(time.time(), 3), "frame_index": idx, "size": list(size),
             "detections": dets, "qr_text": qr_text, "engine": engine_used,
             "depth": bool(depth is not None), "depth_url": DEPTH_URL or None}
    with _marks_lock:
        _latest_marks.update(marks)
    return marks


@app.get("/marks/latest")
async def marks_latest():
    with _marks_lock:
        return json.loads(json.dumps(_latest_marks))


@app.post("/detect")
async def detect_upload(file: UploadFile = File(...)):
    """上传图片识别（离线复核 / 数据集检查）"""
    import cv2
    import numpy as np
    data = await file.read()
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "无法解析图片")
    t0 = time.time()
    dets, qr_text, size = detect_core.analyze(
        img, conf_min=CONF_MIN, min_area_ratio=MIN_AREA_RATIO,
        stain_th=STAIN_TH, defect_th=DEFECT_TH)
    return {"engine": STATS["engine"], "detections": dets, "qr_text": qr_text,
            "size": list(size), "elapsed_ms": round((time.time() - t0) * 1000, 1),
            "message": "识别到 {0} 件货物".format(len(dets)) if dets else "未识别到货物"}


def marks_for_frame(frame, idx, sync=False):
    """取与指定帧对应的标记：帧号一致直接复用，否则（sync=True）同步识别一帧，保证帧↔标记严格对应"""
    with _marks_lock:
        cached = dict(_latest_marks)
    if cached.get("frame_index") == idx and cached.get("detections") is not None and not sync:
        return cached["detections"]
    if not sync:
        return cached.get("detections") or []
    dets, _qr, _size = detect_core.analyze(
        frame, conf_min=CONF_MIN, min_area_ratio=MIN_AREA_RATIO,
        stain_th=STAIN_TH, defect_th=DEFECT_TH)
    with _marks_lock:
        _latest_marks.update({"ts": time.time(), "detections": dets,
                              "frame_index": idx, "size": [frame.shape[1], frame.shape[0]]})
    return dets


def load_table_reference():
    """载入逐像素桌面参考（惰性 + 缓存）：用于目标高度与深度空洞判定"""
    if _TABLE_CACHE["tried"]:
        return _TABLE_CACHE["ref"]
    _TABLE_CACHE["tried"] = True
    path = TABLE_REF
    if not path and os.path.isdir(TABLE_REF_DIR):
        import glob
        cands = sorted(glob.glob(os.path.join(TABLE_REF_DIR, "*", "table_reference.npz")), reverse=True)
        path = cands[0] if cands else ""
    if not path or not os.path.exists(path):
        logger.info("未提供逐像素桌面参考（TABLE_REF），高度改用当前帧中位数")
        return None
    try:
        import numpy as np
        z = np.load(path)
        ref = {"path": path,
               "table_depth_mm": z["table_depth_mm"].astype(np.float32),
               "valid_mask": z["valid_mask"] if "valid_mask" in z.files else None,
               "roi_mask": z["roi_mask"] if "roi_mask" in z.files else None}
        _TABLE_CACHE["ref"] = ref
        logger.info("已载入桌面参考: {0}（中位 {1:.0f}mm）".format(path, float(np.median(ref["table_depth_mm"][ref["table_depth_mm"] > 50]))))
    except Exception as e:
        logger.warning("桌面参考载入失败: {0}".format(e))
    return _TABLE_CACHE["ref"]


def fetch_depth():
    """拉取深度快照（带 0.3s 缓存；失败返回 None → 自动降级为纯 RGB）"""
    import numpy as np
    if not DEPTH_URL:
        return None
    now = time.time()
    if now - DEPTH_CACHE["ts"] < 0.3 and DEPTH_CACHE["img"] is not None:
        return DEPTH_CACHE["img"]
    try:
        import urllib.request
        with urllib.request.urlopen(DEPTH_URL, timeout=2) as r:
            buf = np.frombuffer(r.read(), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.dtype != np.uint16:                     # 8bit 伪彩色无法用于测距，忽略
            return None
        DEPTH_CACHE.update({"ts": now, "img": img})
        return img
    except Exception as e:
        logger.warning("深度快照获取失败: {0}".format(e))
        return None


# ==================== 三层识别编排（yolo 检测 + cnn 属性 + 本容器标记/2.5D） ====================
def _post_json(url, payload, timeout):
    import urllib.request
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def detect_via_services(frame):
    """调用 yolo 容器做检测、cnn 容器做属性；失败返回 None（由调用方回退经典引擎）"""
    import cv2
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 88])
    if not ok:
        return None
    b64 = base64.b64encode(buf.tobytes()).decode()
    yolo = _post_json(YOLO_URL + "/detect_array", {"image_jpeg_base64": b64}, YOLO_TIMEOUT)
    boxes = yolo.get("boxes") or []
    dets = []
    for b in boxes:
        dets.append({
            "class": None, "name": b.get("name") or "货物", "label": b.get("name") or "货物",
            "shape": "未知", "color": "未知", "marks": ["无"],
            "conf": b.get("conf"), "box": b.get("box"), "box_norm": b.get("box_norm"),
            "source": "yolo:{0}".format(yolo.get("model") or "onnx"),
        })
    if dets and CNN_URL:
        try:
            cnn = _post_json(CNN_URL + "/classify_array",
                             {"image_jpeg_base64": b64, "boxes": [d["box"] for d in dets]},
                             CNN_TIMEOUT)
            for det, res in zip(dets, cnn.get("results") or []):
                color = (res.get("color") or {}).get("label")
                shape = (res.get("shape") or {}).get("label")
                stain = (res.get("stain") or {}).get("label")
                det["color"] = COLOR_CN.get(color, color or "未知")
                det["shape"] = SHAPE_CN.get(shape, shape or "未知")
                if stain and stain != "clean":
                    det["marks"] = [STAIN_CN.get(stain, stain)]
                det["attr_conf"] = {"color": (res.get("color") or {}).get("confidence"),
                                    "shape": (res.get("shape") or {}).get("confidence"),
                                    "stain": (res.get("stain") or {}).get("confidence")}
                det["source"] = "yolo+cnn"
                matched = None
                for g in GOODS:
                    if g["shape"] == det["shape"] and g["color"] == det["color"]:
                        matched = g
                        break
                if matched:
                    det["class"] = matched["id"]
                    det["name"] = matched["name"]
                    det["label"] = matched["name"]
                elif det["color"] != "未知" and det["shape"] != "未知":
                    # 图库外组合：按“颜色+形状”生成货物名称（赛项要求名称必须正确）
                    det["name"] = "{0}{1}".format(det["color"], det["shape"])
                    det["label"] = det["name"]
        except Exception as e:
            logger.warning("CNN 属性服务不可用（{0}）→ 该帧属性留空".format(e))
    return dets


# 属性标签英文 → 中文（与工创yolo 训练类别一致）
COLOR_CN = {"red": "红色", "orange": "橙色", "yellow": "黄色", "green": "绿色", "cyan": "青色",
            "blue": "蓝色", "purple": "紫色", "black": "黑色", "white": "白色"}
SHAPE_CN = {"cube": "正方体", "cuboid": "长方体", "cylinder": "圆柱", "ball": "球",
            "tetra": "正四面体", "prism5": "五棱柱", "prism6": "六棱柱"}
STAIN_CN = {"stain": "污渍", "defect": "缺陷"}


def run_detection(frame):
    """统一入口：按 DETECT_ENGINE 选择 yolo 三层链路或经典 CV 引擎"""
    if DETECT_ENGINE == "yolo" and YOLO_URL:
        try:
            dets = detect_via_services(frame)
            if dets is not None:
                return dets, "yolo({0})".format("+cnn" if CNN_URL else "")
        except Exception as e:
            logger.warning("YOLO 容器不可用（{0}）→ 回退经典引擎".format(e))
    dets, _qr, _size = detect_core.analyze(
        frame, conf_min=CONF_MIN, min_area_ratio=MIN_AREA_RATIO,
        stain_th=STAIN_TH, defect_th=DEFECT_TH)
    return dets, "classic"


@app.get("/frame.jpg")
async def frame_jpg(draw: int = 1):
    """当前帧 JPEG；draw=1 时叠加标记（框 + 名称/置信度/表面状态）—— 与当前帧同步识别，避免错位"""
    with _frame_lock:
        frame = None if _latest["bgr"] is None else _latest["bgr"].copy()
        idx = _latest["index"]
    if frame is None:
        raise HTTPException(503, "尚无可用帧")
    if draw:
        dets = marks_for_frame(frame, idx, sync=True)
        hud = ["marks: {0}  ·  {1}".format(len(dets), time.strftime("%H:%M:%S")),
               "engine: {0}".format(STATS["engine"])]
        frame = detect_core.draw_marks(frame, dets, hud=hud)
    jpg = detect_core.encode_jpeg(frame, JPEG_QUALITY)
    return Response(content=jpg, media_type="image/jpeg",
                    headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/stream.mjpg")
async def stream_mjpg(draw: int = 1):
    """可选：带标记的 MJPEG 流（<img src> 直接可用）"""
    boundary = "frame"

    def gen():
        while True:
            with _frame_lock:
                frame = None if _latest["bgr"] is None else _latest["bgr"].copy()
            if frame is not None:
                if draw:
                    with _marks_lock:
                        dets = list(_latest_marks["detections"])
                    frame = detect_core.draw_marks(frame, dets)
                jpg = detect_core.encode_jpeg(frame, JPEG_QUALITY)
                yield (b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
            time.sleep(max(0.02, 1.0 / max(1.0, STREAM_FPS)))

    return Response(content=gen(), media_type="multipart/x-mixed-replace; boundary={0}".format(boundary))


@app.get("/api/meta")
async def api_meta():
    return {"shapes": SHAPES, "colors": COLORS, "marks": MARKS,
            "goods": [{"id": g["id"], "name": g["name"], "shape": g["shape"], "color": g["color"]}
                      for g in GOODS]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
