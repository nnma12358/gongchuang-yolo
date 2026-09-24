#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能分拣装置 —— 第二路相机服务（sort-cam2 / :8103）
====================================================
给分拣装置补上「第二路画面」，让操作/裁判端能同时看到两路视频。

支持的源（CAM2_SOURCE）：
    auto       未显式指定时：先试 CAM2_URL → 再试 CAM2_INDEX → 最后退回 synthetic
    index      本机 V4L2 相机，如 CAM2_INDEX=0（/dev/video0）
    url        任意可被 OpenCV 打开的视频源：MJPEG / RTSP / 视频文件
    synthetic  测试图（**明确标注**，用于未接相机时的联调，不冒充真实画面）

接口：
    GET /health          运行状态（mode / source / fps / 分辨率 / 帧数）
    GET /api/meta        同上，供前端显示来源
    GET /frame.jpg       当前帧
    GET /stream.mjpg     MJPEG 流（前端 <img src> 直接显示）

设计要点：
  · 采集在独立线程里持续跑，只保留**最新一帧**，HTTP 侧永不阻塞在相机上
  · 相机掉线自动重连（每 2s 重试），不会让服务退出
  · 与 sort-vision/sort-gateway 走同一套端口约定，前端无需特殊处理
"""
import logging
import os
import threading
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("cam2")

PORT = int(os.environ.get("CAM2_PORT", "8103"))
SOURCE = os.environ.get("CAM2_SOURCE", "auto").lower()
CAM2_INDEX = int(os.environ.get("CAM2_INDEX", "0"))
CAM2_URL = os.environ.get("CAM2_URL", "")
WIDTH = int(os.environ.get("CAM2_WIDTH", "640"))
HEIGHT = int(os.environ.get("CAM2_HEIGHT", "480"))
FPS = float(os.environ.get("CAM2_FPS", "15"))
JPEG_QUALITY = int(os.environ.get("CAM2_JPEG_QUALITY", "82"))
NAME = os.environ.get("CAM2_NAME", "第二路相机")

LOCK = threading.Lock()
FRAMES = {"jpg": None, "ts": 0.0, "seq": 0}
STATE = {"mode": SOURCE, "source": "", "fps": 0.0, "size": None,
         "frames": 0, "error": None, "started": time.time()}
# 内容冻结检测：相机被拔出/驱动卡住时，读取仍可能返回同一张图，
# 只看 fps 会误判为"正常"。记录内容最后一次变化的时间。
_CHANGE = {"sig": None, "ts": 0.0}
_SYNTH_RETRY = {"ts": 0.0}   # 测试图模式下重试真机的时间戳
_COUNTER = {"t0": time.time(), "n": 0}


def _synthetic_frame(t):
    """测试图：明确标注『测试图』，避免被误当成现场画面"""
    import cv2
    img = np.full((HEIGHT, WIDTH, 3), 26, np.uint8)
    # 移动的网格 + 计时，便于一眼看出流是活的
    step = 40
    off = int((t * 60) % step)
    for x in range(-step + off, WIDTH, step):
        cv2.line(img, (x, 0), (x, HEIGHT), (60, 60, 60), 1)
    for y in range(-step + off, HEIGHT, step):
        cv2.line(img, (0, y), (WIDTH, y), (60, 60, 60), 1)
    cx = int(WIDTH / 2 + (WIDTH / 3) * np.sin(t * 1.2))
    cy = int(HEIGHT / 2 + (HEIGHT / 4) * np.cos(t * 1.7))
    cv2.circle(img, (cx, cy), 46, (0, 200, 255), -1)
    cv2.circle(img, (cx, cy), 46, (255, 255, 255), 2)
    cv2.putText(img, "TEST PATTERN", (18, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 200, 255), 2)
    cv2.putText(img, "cam2 not connected", (18, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (170, 170, 170), 1)
    cv2.putText(img, time.strftime("%H:%M:%S"), (18, HEIGHT - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
    return img


def _open_capture():
    """按配置打开相机，返回 (cap, mode, source, error)"""
    import cv2
    order = []
    if SOURCE == "synthetic":
        order = [("synthetic", None)]
    elif SOURCE == "index":
        order = [("index", CAM2_INDEX)]
    elif SOURCE == "url":
        order = [("url", CAM2_URL)]
    else:                                   # auto
        if CAM2_URL:
            order.append(("url", CAM2_URL))
        order.append(("index", CAM2_INDEX))
        order.append(("synthetic", None))

    for mode, arg in order:
        if mode == "synthetic":
            return None, "synthetic", "测试图（未接第二路相机）", None
        try:
            cap = cv2.VideoCapture(arg)
            if mode == "index":
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
                cap.set(cv2.CAP_PROP_FPS, FPS)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if cap.isOpened():
                # 预热：设备刚上电或 UVC 未就绪时，前几帧常读不到（select timeout），
                # 多试几次再判定失败，避免"明明有相机却一直用测试图"。
                ok = False
                for _ in range(12):
                    ok, _f = cap.read()
                    if ok:
                        break
                    time.sleep(0.25)
                if ok:
                    return cap, mode, str(arg), None
            cap.release()
        except Exception as e:
            logger.warning("打开 %s(%s) 失败: %s", mode, arg, str(e)[:80])
    return None, "none", "", "无法打开任何相机源"


def capture_loop():
    import cv2
    cap, mode, source, err = _open_capture()
    with LOCK:
        STATE.update({"mode": mode, "source": source, "error": err})
    logger.info("第二路相机：mode=%s source=%s %s", mode, source, err or "")
    period = 1.0 / max(1.0, FPS)
    fail = 0
    while True:
        t0 = time.time()
        frame = None
        if mode == "synthetic":
            # 每 15s 再试一次真实相机：设备后插/后上电、或启动时没就绪的情况都能自愈
            if time.time() - _SYNTH_RETRY["ts"] >= 15.0:
                _SYNTH_RETRY["ts"] = time.time()
                cap2, mode2, src2, err2 = _open_capture()
                if cap2 is not None and mode2 != "synthetic":
                    cap, mode, source, err = cap2, mode2, src2, err2
                    with LOCK:
                        STATE.update({"mode": mode, "source": source, "error": None})
                    logger.info("第二路相机已接入真实设备：mode=%s source=%s", mode, source)
                    continue
            frame = _synthetic_frame(time.time())
        else:
            if cap is None:
                time.sleep(2.0)
                cap, mode, source, err = _open_capture()
                with LOCK:
                    STATE.update({"mode": mode, "source": source, "error": err})
                continue
            try:
                ok, f = cap.read()
                if ok and f is not None:
                    frame = f
                    fail = 0
                else:
                    fail += 1
            except Exception:
                fail += 1
            if fail >= 15:                      # 连续读失败 → 重连
                logger.warning("相机连续读失败，重连…")
                try:
                    cap.release()
                except Exception:
                    pass
                cap = None
                fail = 0
                continue
        if frame is not None:
            h, w = frame.shape[:2]
            try:                      # 记录内容是否真的变了（廉价抽样指纹）
                sig = int(frame[::16, ::16].sum())
            except Exception:
                sig = 0
            if _CHANGE["sig"] != sig:
                _CHANGE["sig"] = sig
                _CHANGE["ts"] = time.time()
            ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
            if ok:
                with LOCK:
                    FRAMES.update({"jpg": buf.tobytes(), "ts": time.time(),
                                   "seq": FRAMES["seq"] + 1})
                    STATE["size"] = [w, h]
                    STATE["frames"] += 1
                _COUNTER["n"] += 1
                now = time.time()
                if now - _COUNTER["t0"] >= 2.0:
                    with LOCK:
                        STATE["fps"] = round(_COUNTER["n"] / (now - _COUNTER["t0"]), 1)
                    _COUNTER.update({"t0": now, "n": 0})
        dt = time.time() - t0
        if dt < period:
            time.sleep(period - dt)


def _latest():
    with LOCK:
        return FRAMES["jpg"], FRAMES["ts"]


def _health():
    with LOCK:
        st = dict(STATE)
        st.update({"jpg_age_s": round(time.time() - FRAMES["ts"], 3) if FRAMES["ts"] else None,
                   "name": NAME, "port": PORT})
    st["ok"] = st["jpg_age_s"] is not None and st["jpg_age_s"] < 5.0
    st["synthetic"] = (st["mode"] == "synthetic")
    # 画面内容多久没变化（秒）—— 相机挂了但读取不报错时会持续增大
    st["frozen_s"] = round(time.time() - _CHANGE["ts"], 2) if _CHANGE["ts"] else None
    return st


def serve():
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse, Response, StreamingResponse
    import uvicorn

    app = FastAPI(title="sort-cam2", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return _health()

    @app.get("/api/meta")
    def meta():
        return _health()

    @app.get("/frame.jpg")
    def frame_jpg():
        jpg, _ts = _latest()
        if jpg is None:
            return JSONResponse({"detail": "尚无画面"}, status_code=503)
        return Response(content=jpg, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store, max-age=0"})

    @app.get("/stream.mjpg")
    def stream_mjpg():
        boundary = b"cam2frame"

        def gen():
            last = -1
            idle = 0.0
            while True:
                with LOCK:
                    jpg = FRAMES["jpg"]
                    seq = FRAMES["seq"]
                if jpg is not None and seq != last:
                    last = seq
                    yield (b"--" + boundary + b"\r\nContent-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n"
                           + jpg + b"\r\n")
                    idle = 0.0
                else:
                    idle += 0.02
                    if idle > 15.0:          # 长时间没新帧就结束，让前端重连
                        return
                time.sleep(1.0 / 25.0)

        return StreamingResponse(gen(),
                                 media_type="multipart/x-mixed-replace; boundary={0}".format(
                                     boundary.decode()))

    logger.info("%s 服务启动：:%s（mode=%s）", NAME, PORT, SOURCE)
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


def main():
    threading.Thread(target=capture_loop, daemon=True).start()
    time.sleep(0.8)
    serve()


if __name__ == "__main__":
    main()
