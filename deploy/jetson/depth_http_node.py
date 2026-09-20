#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
depth_http_node.py —— ROS2 话题 → HTTP 图像服务（跑在 ros2 容器内）
=====================================================================
为什么需要：工作区的 2.5D 定位软件（overhead_live_view.py，:8088）只暴露 RGB 相关端点，
没有深度端点；而深度是 2.5D 测高与“深度空洞”检测的关键。本节点直接订阅 Astra 话题，
把彩色/深度转成标准 HTTP 接口，**不修改他们的代码**：

  GET /health         彩色/深度帧率、帧号、帧龄（与他们的 /health 语义一致）
  GET /color.jpg      当前彩色帧（JPEG）
  GET /color.mjpg     彩色 MJPEG 流（OpenCV VideoCapture 可直接作为相机源使用）
  GET /depth.png      当前深度帧（16-bit PNG，单位 mm，0=无效）← 视觉容器 DEPTH_URL 用这个
  GET /depth_preview.jpg  深度伪彩预览（人工核对用）
  GET /points_camera.json 当前帧中“高于桌面”的候选像素 + 相机系坐标（轻量 2.5D 输出）

环境变量：
  COLOR_TOPIC  默认 /overhead_camera/color/image_raw
  DEPTH_TOPIC  默认 /overhead_camera/depth/image_raw
  PORT         默认 8123
  JPEG_QUALITY 默认 85
  DEPTH_SCALE  深度单位换算（默认 0.001，即话题为 mm；若为 16UC1 米制则设 1.0）
  RELIABLE_QOS 默认 1（现场已验证：相机发布端为 RELIABLE，订阅端必须匹配，否则画面冻结）

部署：与 ros_bridge_ros2.py 同容器启动（见 docker-compose.jetson.yml 的 ros2 服务）。
"""
import logging
import os
import threading
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("depth-http")

COLOR_TOPIC = os.environ.get("COLOR_TOPIC", "/overhead_camera/color/image_raw")
DEPTH_TOPIC = os.environ.get("DEPTH_TOPIC", "/overhead_camera/depth/image_raw")
# 注意：本容器同时跑「桥接节点(默认 PORT=8120)」与「深度 HTTP 节点」，
# 两者不能抢同一个端口 —— 这里优先读 DEPTH_HTTP_PORT，PORT 只作兜底。
PORT = int(os.environ.get("DEPTH_HTTP_PORT", os.environ.get("PORT", "8123")))
JPEG_QUALITY = int(os.environ.get("JPEG_QUALITY", "85"))
DEPTH_SCALE = float(os.environ.get("DEPTH_SCALE", "0.001"))
RELIABLE_QOS = os.environ.get("RELIABLE_QOS", "1") == "1"

FRAMES = {"color": None, "depth": None, "color_ts": 0.0, "depth_ts": 0.0,
          "color_seq": 0, "depth_seq": 0, "color_fps": 0.0, "depth_fps": 0.0}
LOCK = threading.Lock()
_COUNTERS = {"color_t0": time.time(), "color_n": 0, "depth_t0": time.time(), "depth_n": 0}


def _seq_of(msg, key):
    """取该帧的单调序号。

    ROS2 的 std_msgs/Header **没有 seq 字段**（那是 ROS1 的），直接访问会抛
    AttributeError 并让订阅回调整帧丢弃。这里优先用 header.stamp 的纳秒值，
    取不到（无 stamp / 时间为 0）时退回自增计数。
    """
    hdr = getattr(msg, "header", None)
    stamp = getattr(hdr, "stamp", None)
    if stamp is not None:
        ns = int(getattr(stamp, "sec", 0) or 0) * 1000000000 + int(getattr(stamp, "nanosec", 0) or 0)
        if ns > 0:
            return ns
    k = key + "_seq"
    _COUNTERS[k] = int(_COUNTERS.get(k, 0)) + 1
    return _COUNTERS[k]


def _qos():
    """现场结论：相机端 RELIABLE，订阅端必须匹配（否则 RGB 冻结）"""
    from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
    return QoSProfile(depth=5,
                      reliability=ReliabilityPolicy.RELIABLE if RELIABLE_QOS else ReliabilityPolicy.BEST_EFFORT,
                      history=HistoryPolicy.KEEP_LAST)


def to_bgr(msg):
    """sensor_msgs/Image → BGR ndarray（支持 rgb8/bgr8/mono8）"""
    if msg.encoding in ("rgb8", "bgr8"):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width, 3)
        return arr[:, :, ::-1].copy() if msg.encoding == "rgb8" else arr.copy()
    if msg.encoding in ("mono8", "8UC1"):
        arr = np.frombuffer(msg.data, np.uint8).reshape(msg.height, msg.width)
        return np.repeat(arr[:, :, None], 3, axis=2)
    raise ValueError("不支持的彩色编码: {0}".format(msg.encoding))


def to_depth_mm(msg):
    """sensor_msgs/Image → uint16 深度(mm)。16UC1 默认按 mm，可按 DEPTH_SCALE 缩放"""
    if msg.encoding in ("16UC1", "mono16"):
        arr = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.width).astype(np.float32)
    elif msg.encoding == "32FC1":
        arr = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.width) * 1000.0
    else:
        raise ValueError("不支持的深度编码: {0}".format(msg.encoding))
    if abs(DEPTH_SCALE - 1.0) > 1e-9:
        arr = arr * DEPTH_SCALE * 1000.0 if DEPTH_SCALE < 1 else arr * DEPTH_SCALE
    return np.clip(arr, 0, 65535).astype(np.uint16)


def start_ros():
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image

    rclpy.init(args=None)
    node = Node("depth_http_node")

    def color_cb(msg):
        try:
            bgr = to_bgr(msg)
        except Exception as e:
            logger.warning("彩色解码失败: {0}".format(e)); return
        with LOCK:
            FRAMES.update({"color": bgr, "color_ts": time.time(), "color_seq": _seq_of(msg, "color")})
        _COUNTERS["color_n"] += 1
        now = time.time()
        if now - _COUNTERS["color_t0"] >= 2.0:
            FRAMES["color_fps"] = round(_COUNTERS["color_n"] / (now - _COUNTERS["color_t0"]), 1)
            _COUNTERS.update({"color_t0": now, "color_n": 0})

    def depth_cb(msg):
        try:
            d = to_depth_mm(msg)
        except Exception as e:
            logger.warning("深度解码失败: {0}".format(e)); return
        with LOCK:
            FRAMES.update({"depth": d, "depth_ts": time.time(), "depth_seq": _seq_of(msg, "depth")})
        _COUNTERS["depth_n"] += 1
        now = time.time()
        if now - _COUNTERS["depth_t0"] >= 2.0:
            FRAMES["depth_fps"] = round(_COUNTERS["depth_n"] / (now - _COUNTERS["depth_t0"]), 1)
            _COUNTERS.update({"depth_t0": now, "depth_n": 0})

    q = _qos()
    node.create_subscription(Image, COLOR_TOPIC, color_cb, q)
    node.create_subscription(Image, DEPTH_TOPIC, depth_cb, q)
    logger.info("订阅: {0} / {1}（QoS {2}）".format(
        COLOR_TOPIC, DEPTH_TOPIC, "RELIABLE" if RELIABLE_QOS else "BEST_EFFORT"))
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()


def _render(path):
    """各接口的纯逻辑（两条后端共用）：返回 (状态码, 字节, content_type)"""
    import cv2
    now = time.time()
    if path == "/health":
        with LOCK:
            obj = {
                "color": FRAMES["color"] is not None,
                "depth": FRAMES["depth"] is not None,
                "color_seq": FRAMES["color_seq"], "depth_seq": FRAMES["depth_seq"],
                "color_age": round(now - FRAMES["color_ts"], 3) if FRAMES["color_ts"] else -1.0,
                "depth_age": round(now - FRAMES["depth_ts"], 3) if FRAMES["depth_ts"] else -1.0,
                "color_fps": FRAMES["color_fps"], "depth_fps": FRAMES["depth_fps"],
                "topics": {"color": COLOR_TOPIC, "depth": DEPTH_TOPIC},
            }
        import json as _json
        return 200, _json.dumps(obj, ensure_ascii=False).encode("utf-8"), "application/json"
    with LOCK:
        color = None if FRAMES["color"] is None else FRAMES["color"].copy()
        depth = None if FRAMES["depth"] is None else FRAMES["depth"].copy()
    if path == "/color.jpg":
        if color is None:
            return 503, '{"detail":"无彩色帧"}'.encode("utf-8"), "application/json"
        ok, buf = cv2.imencode(".jpg", color, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return 200, buf.tobytes(), "image/jpeg"
    if path == "/depth.png":
        if depth is None:
            return 503, '{"detail":"无深度帧"}'.encode("utf-8"), "application/json"
        ok, buf = cv2.imencode(".png", depth)
        return 200, buf.tobytes(), "image/png"
    if path == "/depth_preview.jpg":
        if depth is None:
            return 503, '{"detail":"无深度帧"}'.encode("utf-8"), "application/json"
        valid = depth[depth > 0]
        lo, hi = (float(np.percentile(valid, 5)), float(np.percentile(valid, 95))) if valid.size else (0, 1)
        norm = np.clip((depth.astype(np.float32) - lo) / max(1e-6, hi - lo), 0, 1)
        norm[depth == 0] = 0
        color_map = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        ok, buf = cv2.imencode(".jpg", color_map)
        return 200, buf.tobytes(), "image/jpeg"
    return 404, b'{"detail":"not found"}', "application/json"


def serve_stdlib():
    """无 fastapi 时的内置 HTTP 服务（标准库实现，含 MJPEG 推流）"""
    import cv2
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _plain(self, code, body, ctype):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/color.mjpg":
                boundary = "frame"
                self.send_response(200)
                self.send_header("Content-Type",
                                 "multipart/x-mixed-replace; boundary={0}".format(boundary))
                self.end_headers()
                try:
                    while True:
                        with LOCK:
                            f = None if FRAMES["color"] is None else FRAMES["color"].copy()
                        if f is not None:
                            ok, buf = cv2.imencode(".jpg", f,
                                                   [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                            if ok:
                                jpg = buf.tobytes()
                                self.wfile.write(b"--" + boundary.encode() +
                                                 b"\r\nContent-Type: image/jpeg\r\n"
                                                 b"Content-Length: " + str(len(jpg)).encode() +
                                                 b"\r\n\r\n" + jpg + b"\r\n")
                        time.sleep(1.0 / 15.0)
                except (BrokenPipeError, ConnectionResetError):
                    return
            code, body, ctype = _render(path)
            self._plain(code, body, ctype)

    logger.info("深度 HTTP 节点启动（内置 stdlib）: :{0} | color={1} | depth={2}".format(
        PORT, COLOR_TOPIC, DEPTH_TOPIC))
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


def main():
    import cv2

    start_ros()
    try:
        from fastapi import FastAPI
        from fastapi.responses import JSONResponse, Response, StreamingResponse
        import uvicorn
    except ImportError:
        logger.warning("未安装 fastapi → 使用内置 stdlib HTTP 服务（含 MJPEG）")
        return serve_stdlib()

    app = FastAPI(title="ros2-depth-http", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        now = time.time()
        with LOCK:
            return {
                "color": FRAMES["color"] is not None,
                "depth": FRAMES["depth"] is not None,
                "color_seq": FRAMES["color_seq"], "depth_seq": FRAMES["depth_seq"],
                "color_age": round(now - FRAMES["color_ts"], 3) if FRAMES["color_ts"] else -1.0,
                "depth_age": round(now - FRAMES["depth_ts"], 3) if FRAMES["depth_ts"] else -1.0,
                "color_fps": FRAMES["color_fps"], "depth_fps": FRAMES["depth_fps"],
                "topics": {"color": COLOR_TOPIC, "depth": DEPTH_TOPIC},
            }

    @app.get("/color.jpg")
    def color_jpg():
        with LOCK:
            f = None if FRAMES["color"] is None else FRAMES["color"].copy()
        if f is None:
            return JSONResponse({"detail": "无彩色帧"}, status_code=503)
        ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/depth.png")
    def depth_png():
        """16-bit PNG（mm）—— 视觉容器 DEPTH_URL 直接使用"""
        with LOCK:
            d = None if FRAMES["depth"] is None else FRAMES["depth"].copy()
        if d is None:
            return JSONResponse({"detail": "无深度帧"}, status_code=503)
        ok, buf = cv2.imencode(".png", d)
        return Response(content=buf.tobytes(), media_type="image/png",
                        headers={"Cache-Control": "no-store"})

    @app.get("/depth_preview.jpg")
    def depth_preview():
        with LOCK:
            d = None if FRAMES["depth"] is None else FRAMES["depth"].copy()
        if d is None:
            return JSONResponse({"detail": "无深度帧"}, status_code=503)
        valid = d[d > 0]
        lo, hi = (float(np.percentile(valid, 5)), float(np.percentile(valid, 95))) if valid.size else (0, 1)
        norm = np.clip((d.astype(np.float32) - lo) / max(1e-6, hi - lo), 0, 1)
        norm[d == 0] = 0
        color = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
        ok, buf = cv2.imencode(".jpg", color)
        return Response(content=buf.tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "no-store"})

    @app.get("/color.mjpg")
    def color_mjpg():
        """MJPEG：可作为本项目视觉容器的 CAMERA_URL（OpenCV 直接读取）"""
        boundary = "frame"

        def gen():
            while True:
                with LOCK:
                    f = None if FRAMES["color"] is None else FRAMES["color"].copy()
                if f is not None:
                    ok, buf = cv2.imencode(".jpg", f, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
                    if ok:
                        jpg = buf.tobytes()
                        yield (b"--" + boundary.encode() + b"\r\nContent-Type: image/jpeg\r\n"
                               b"Content-Length: " + str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                time.sleep(1.0 / 15.0)

        # MJPEG 是无限生成器，必须用 StreamingResponse（Response 会尝试 .encode() 而报错）
        return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary={0}".format(boundary))

    @app.get("/points_camera.json")
    def points_camera():
        """轻量 2.5D 输出：高于桌面参考的候选区域 + 相机系坐标（供快速核对/联调）"""
        with LOCK:
            d = None if FRAMES["depth"] is None else FRAMES["depth"].copy()
        if d is None:
            return JSONResponse({"detail": "无深度帧"}, status_code=503)
        ref = np.median(d[d > 0]) if (d > 0).any() else 0
        near = (d > 0) & (d < ref - 15)
        n = int(near.sum())
        ys, xs = np.nonzero(near)
        pts = []
        if n:
            # 简易聚类：按 32px 网格分组，取每组质心
            keys = {}
            for x, y in zip(xs, ys):
                keys.setdefault((x // 32, y // 32), []).append((x, y))
            for (gx, gy), members in sorted(keys.items(), key=lambda kv: -len(kv[1]))[:12]:
                mx = sum(m[0] for m in members) / len(members)
                my = sum(m[1] for m in members) / len(members)
                z = float(np.median(d[members[0][1]:members[-1][1] + 1, members[0][0]:members[-1][0] + 1][0])) \
                    if members else float(ref)
                pts.append({"center_pixel": [round(mx, 1), round(my, 1)],
                            "area_px": len(members),
                            "depth_mm": round(float(np.median([d[m[1], m[0]] for m in members])), 1),
                            "height_mm": round(float(ref - np.median([d[m[1], m[0]] for m in members])), 1)})
        return {"table_depth_median_mm": round(float(ref), 1), "near_pixel_count": n, "objects": pts}

    logger.info("HTTP 服务启动: :{0}（/color.jpg /color.mjpg /depth.png /depth_preview.jpg）".format(PORT))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="warning")


if __name__ == "__main__":
    main()
