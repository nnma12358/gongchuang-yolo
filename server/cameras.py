#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能分拣装置 —— 多路相机子系统（网关侧）
=========================================
现场可能同时存在多路画面，前端要能**同时**显示两路（或更多）：

    main   顶置 Astra 彩色（ROS 桥接 :8123 /color.jpg·/color.mjpg）—— 分拣台实时画面
    marked 顶置 · 识别标记（视觉容器 :8100 /frame.jpg·/stream.mjpg）—— 叠加框/中文标签
    depth  深度伪彩（ROS 桥接 :8123 /depth_preview.jpg）—— 2.5D 用
    cam2   第二路相机（sort-cam2 服务 :8103）—— 本机 USB 相机/腕部相机/任意 MJPEG

设计要点：
  · **统一接口**：每路都提供 /api/cameras/<id>/frame.jpg（单帧）与 /api/cameras/<id>/stream.mjpg（MJPEG）。
    没有原生 MJPEG 的源（如深度图）由本模块按帧率轮询单帧并自行封装 multipart —— 前端只用一套代码。
  · **以实际服务为准**：不可达就如实报 ready=false 与原因，绝不合成画面。
  · **不缓存、不落盘**：全部流式转发，避免 Nano 上额外的内存/磁盘压力。
  · **限制并发流**：每路流占一个线程，超过 MAX_STREAMS 主动拒绝，防止把 4GB 的 Nano 拖垮。
"""
import logging
import os
import queue
import threading
import time

import requests

logger = logging.getLogger("sort-cameras")

# ==================== 配置 ====================
BRIDGE = os.environ.get("BRIDGE_URL", "http://127.0.0.1:8123").rstrip("/")
VISION = os.environ.get("VISION_URL", "http://127.0.0.1:8100").rstrip("/")
CAM2 = os.environ.get("CAM2_URL", "http://127.0.0.1:8103").rstrip("/")

# 每路：id、显示名、单帧地址、MJPEG 地址（None 表示由本模块用单帧合成）、说明
_REGISTRY = [
    {
        "id": "main",
        "name": os.environ.get("CAM_MAIN_NAME", "顶置相机 · 实时画面"),
        "snapshot": os.environ.get("CAM_MAIN_SNAPSHOT", BRIDGE + "/color.jpg"),
        "stream": os.environ.get("CAM_MAIN_STREAM", BRIDGE + "/color.mjpg"),
        "detail": "Astra 彩色（ROS 桥接直出，无标记）",
    },
    {
        "id": "marked",
        "name": os.environ.get("CAM_MARKED_NAME", "顶置相机 · 识别标记"),
        "snapshot": os.environ.get("CAM_MARKED_SNAPSHOT", VISION + "/frame.jpg?draw=1"),
        "stream": os.environ.get("CAM_MARKED_STREAM", VISION + "/stream.mjpg"),
        "detail": "视觉容器输出，叠加检测框与中文标签",
    },
    {
        "id": "depth",
        "name": os.environ.get("CAM_DEPTH_NAME", "深度图 · 2.5D"),
        "snapshot": os.environ.get("CAM_DEPTH_SNAPSHOT", BRIDGE + "/depth_preview.jpg"),
        "stream": None,
        "detail": "深度伪彩色，用于高度/凸起判断",
    },
    {
        "id": "cam2",
        "name": os.environ.get("CAM2_NAME", "第二路相机"),
        "snapshot": os.environ.get("CAM2_SNAPSHOT", CAM2 + "/frame.jpg"),
        "stream": os.environ.get("CAM2_STREAM", CAM2 + "/stream.mjpg"),
        "detail": os.environ.get("CAM2_DETAIL", "sort-cam2 服务（USB 相机 / 外部流）"),
    },
]

PROBE_TTL = float(os.environ.get("CAM_PROBE_TTL", "2.0"))     # 状态探测缓存秒
PROBE_TIMEOUT = float(os.environ.get("CAM_PROBE_TIMEOUT", "0.8"))
STREAM_FPS = float(os.environ.get("CAM_SYNTH_FPS", "8"))       # 合成 MJPEG 的帧率
MAX_STREAMS = int(os.environ.get("CAM_MAX_STREAMS", "6"))
# 单条流最长存活秒数：即便断线检测失效，也能兜底释放槽位
MAX_STREAM_S = float(os.environ.get("CAM_STREAM_MAX_S", "1800"))

_BY_ID = {c["id"]: c for c in _REGISTRY}
_PROBE = {}                      # cid -> (ts, info)
_PROBE_LOCK = threading.Lock()
_STREAMS = {"n": 0}
_STREAM_LOCK = threading.Lock()

BOUNDARY = "sortframe"


def camera_ids():
    return [c["id"] for c in _REGISTRY]


def get_camera(cid):
    return _BY_ID.get(cid)


def _jpeg_size(data):
    """从 JPEG 字节里读出宽高（解析 SOF 段，不依赖 PIL）"""
    i = 2
    n = len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2, 0xC3, 0xC5, 0xC6, 0xC7,
                      0xC9, 0xCA, 0xCB, 0xCD, 0xCE, 0xCF):
            h = (data[i + 5] << 8) | data[i + 6]
            w = (data[i + 7] << 8) | data[i + 8]
            return int(w), int(h)
        if marker in (0xD8, 0xD9) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        seg = (data[i + 2] << 8) | data[i + 3]
        i += 2 + max(0, seg)
    return None, None


def probe(cid, force=False):
    """探测某路是否可用（带缓存）：返回 {ready, size, latency_ms, detail/error}"""
    cam = _BY_ID.get(cid)
    if cam is None:
        return {"id": cid, "ready": False, "error": "未知相机"}
    now = time.time()
    with _PROBE_LOCK:
        hit = _PROBE.get(cid)
        if hit and not force and now - hit[0] < PROBE_TTL:
            return hit[1]
    info = {"id": cid, "ready": False}
    t0 = time.time()
    try:
        r = requests.get(cam["snapshot"], timeout=PROBE_TIMEOUT)
        info["latency_ms"] = round((time.time() - t0) * 1000, 1)
        if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
            w, h = _jpeg_size(r.content)
            info.update({"ready": True, "size": [w, h], "bytes": len(r.content)})
        else:
            info["error"] = "HTTP {0}".format(r.status_code)
    except Exception as e:
        info["error"] = str(e)[:120]
    with _PROBE_LOCK:
        _PROBE[cid] = (time.time(), info)
    return info


def describe(force=False):
    """给前端的相机清单（真实状态，不假装）"""
    out = []
    for cam in _REGISTRY:
        st = probe(cam["id"], force=force)
        out.append({
            "id": cam["id"],
            "name": cam["name"],
            "detail": cam["detail"],
            "has_stream": True,
            "snapshot_url": "/api/cameras/{0}/frame.jpg".format(cam["id"]),
            "stream_url": "/api/cameras/{0}/stream.mjpg".format(cam["id"]),
            "ready": bool(st.get("ready")),
            "size": st.get("size"),
            "latency_ms": st.get("latency_ms"),
            "error": st.get("error"),
        })
    return out


# ==================== 单帧 ====================
def snapshot(cid, timeout=3.0):
    """取某路单帧 JPEG；失败抛异常由调用方转成 503"""
    cam = _BY_ID.get(cid)
    if cam is None:
        raise KeyError(cid)
    r = requests.get(cam["snapshot"], timeout=timeout)
    if r.status_code != 200:
        raise RuntimeError("上游 HTTP {0}".format(r.status_code))
    ctype = r.headers.get("Content-Type", "image/jpeg")
    return r.content, ctype


# ==================== MJPEG ====================
def _acquire_slot():
    with _STREAM_LOCK:
        if _STREAMS["n"] >= MAX_STREAMS:
            return False
        _STREAMS["n"] += 1
        return True


def _release_slot():
    with _STREAM_LOCK:
        _STREAMS["n"] = max(0, _STREAMS["n"] - 1)


def _open_remote(url, idle):
    """打开上游 MJPEG 并校验状态码；失败抛 RuntimeError（由调用方转成 503）。

    先把连接开起来再决定要不要发响应头，这样上游异常时能给出明确的 503，
    而不是先发 200 再送一个"空流"（客户端既看不到帧也看不到错误）。
    """
    r = requests.get(url, stream=True, timeout=(3.05, idle))
    if r.status_code != 200:
        code = r.status_code
        try:
            r.close()
        except Exception:
            pass
        raise RuntimeError("上游返回 HTTP {0}".format(code))
    return r


class CameraStream(object):
    """一条 MJPEG 流：后台线程读上游 → 队列 → 网关的异步生成器取走。

    **为什么不用同步生成器直接 yield**（踩过的坑）：
    Starlette 对同步生成器无法在客户端断开时中断线程（线程正阻塞在上游 read 上），
    于是 finally 迟迟不执行、并发槽位不释放。现场实测：刷新几次页面后
    6 个槽位全部被占用，之后**所有**画面都返回 503（表现为前端一片黑/破图），
    而相机清单却仍显示"在线"。
    改成"线程 + 队列 + 异步生成器轮询 is_disconnected"后，客户端一断开就释放。
    """

    def __init__(self, cid, upstream=None, snapshot=None, remote_url=None,
                 idle=8.0, fps=None):
        self.cid = cid
        self.mtype = None
        self._upstream = upstream
        self._snapshot = snapshot
        self._remote_url = remote_url or ""
        self._idle = idle
        self._fps = float(fps or STREAM_FPS)
        self._q = queue.Queue(maxsize=6)
        self._closed = False
        self._close_lock = threading.Lock()
        self._slot = True
        self.deadline = time.time() + MAX_STREAM_S
        self._thread = threading.Thread(target=self._pump, daemon=True)
        self._thread.start()

    # ---- 读取线程 ----
    def _offer(self, item):
        while not self._closed:
            try:
                self._q.put(item, timeout=0.3)
                return
            except queue.Full:
                continue

    def _pump(self):
        try:
            if self._upstream is not None:
                for data in self._upstream.iter_content(chunk_size=16384):
                    if self._closed:
                        break
                    if data:
                        self._offer(data)
            else:
                head = ("--{0}\r\nContent-Type: image/jpeg\r\n").format(BOUNDARY).encode()
                period = 1.0 / max(0.5, self._fps)
                while not self._closed:
                    t0 = time.time()
                    try:
                        r = requests.get(self._snapshot, timeout=2.0)
                        if r.status_code == 200 and r.content[:2] == b"\xff\xd8":
                            jpg = r.content
                            self._offer(head + b"Content-Length: " +
                                        str(len(jpg)).encode() + b"\r\n\r\n" + jpg + b"\r\n")
                    except Exception:
                        pass              # 上游抖动不终止整条流，下一轮重试
                    dt = time.time() - t0
                    if dt < period:
                        time.sleep(period - dt)
        except Exception as e:
            logger.info("流 %s 读取线程结束：%s", self.cid, str(e)[:70])
        finally:
            self._offer(None)          # 结束标记

    # ---- 网关侧 ----
    def read(self, timeout=1.0):
        """bytes=数据 / b''=暂时无数据 / None=流已结束"""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return b""

    def expired(self):
        return time.time() > self.deadline

    def close(self):
        """幂等：客户端断开、超时、上游结束都会走到这里，槽位一定释放"""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        if self._upstream is not None:
            try:
                self._upstream.close()
            except Exception:
                pass
        if self._slot:
            self._slot = False
            _release_slot()
        logger.info("流 %s 已关闭（当前并发 %s/%s）", self.cid,
                    _STREAMS["n"], MAX_STREAMS)


def open_stream(cid):
    """打开某路的 MJPEG 流。

    返回 (CameraStream, media_type, reason)：
      · 正常        -> (st, mtype, None)
      · 该路不可达  -> (None, None, 原因)   ← **快速失败**，不让前端干等
      · 并发超限    -> (None, None, 原因)
    """
    cam = _BY_ID.get(cid)
    if cam is None:
        raise KeyError(cid)
    # 先探一次该路是否可用：不可用就直接拒绝，避免建一条永不吐数据的连接
    # （现场实测：ROS 桥接在没有相机帧时既不关连接也不发数据）
    st = probe(cid)
    if not st.get("ready"):
        return None, None, st.get("error") or "该路当前不可达"
    remote = cam.get("stream")
    upstream = None
    mtype = None
    if remote:
        # 打开上游就先校验状态码：否则异常时已发出 200 响应头，
        # 客户端只会看到一个"空流"（既没有帧也没有错误），很难排查。
        try:
            upstream = _open_remote(remote, float(os.environ.get("CAM_STREAM_IDLE", "8")))
        except Exception as e:
            return None, None, str(e)[:100]
        # **必须原样带上上游的 boundary**（例：multipart/x-mixed-replace; boundary=frame）。
        # 丢了 boundary 时：字节其实一直在传，但浏览器无法解析 multipart，
        # 表现为"状态栏显示在线、画面却是破图"，且只有 curl 数字节的测试是发现不了的。
        mtype = upstream.headers.get("Content-Type") or \
            "multipart/x-mixed-replace; boundary=frame"
        if "boundary" not in mtype.lower():
            mtype = mtype.rstrip("; ") + "; boundary=frame"
    if not _acquire_slot():
        logger.warning("并发流已达上限 %s，拒绝 %s", MAX_STREAMS, cid)
        if upstream is not None:
            try:
                upstream.close()
            except Exception:
                pass
        return None, None, "并发画面数已达上限 {0}（如为异常残留，重启网关容器即可清零）".format(
            MAX_STREAMS)
    if upstream is not None:
        st = CameraStream(cid, upstream=upstream, remote_url=remote,
                          idle=float(os.environ.get("CAM_STREAM_IDLE", "8")))
    else:
        st = CameraStream(cid, snapshot=cam["snapshot"])
        mtype = "multipart/x-mixed-replace; boundary={0}".format(BOUNDARY)
    return st, mtype, None


def stats():
    with _STREAM_LOCK:
        return {"active_streams": _STREAMS["n"], "max_streams": MAX_STREAMS}
