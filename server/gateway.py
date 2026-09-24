#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能分拣装置 —— 分拣信息显示屏 · 网关脚本（Jetson Nano / Python 3.6+）
=====================================================================
架构（Jetson 端两容器，通过端口通信）：
    ┌─ 视觉容器 sort-vision (8100) ─┐        ┌─ 网关容器 sort-gateway (80) ─┐
    │ 独占相机 → 连续识别 → 标记输出 │ ◀────▶ │ 前端 SPA + 分拣规则 + 自动分拣 │
    └───────────────────────────────┘  HTTP  └──────────────────────────────┘
                                                   │ ROBOT_URL（可选）→ ROS 桥接

赛项约束：
  · 六个单独储物盒（1–6 号），每盒最多 4 件；形状相同且颜色相同 → 同一储物盒
  · 一次只能分拣一件；分拣信息显示完成后再分拣下一件（否则不计分）
  · 按统一指令启动装置计时；比赛结束前不得接触装置；掉落 → 本轮结束
"""
import asyncio
import json
import logging
import os
import threading
import time
from pathlib import Path

import requests
from fastapi import Body, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

import detect_core
import cameras
import pose
from catalog import (COLORS, GOODS, GOODS_BY_ID, IMAGE_FORMATS, MARKS, SHAPES,
                     formats_for, generate_goods_images, goods_svg, match_goods)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sort-gateway")

# ==================== 配置 ====================
STATIC_DIR = Path(os.environ.get("STATIC_DIR", "/app/build"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/app/data"))
GOODS_IMG_DIR = DATA_DIR / "goods"
STATE_FILE = DATA_DIR / "state.json"
MARKS_LOG = DATA_DIR / "marks.jsonl"
MODEL_PATH = os.environ.get("MODEL_PATH", "/app/models/detect/goods_yolov8n_640_fp32.onnx")
HOLD_SECONDS = float(os.environ.get("DISPLAY_HOLD_SECONDS", "3.0"))
VISION_URL = os.environ.get("VISION_URL", "http://127.0.0.1:8100").rstrip("/")
ROBOT_URL = os.environ.get("ROBOT_URL", "").rstrip("/")          # ROS 桥接 HTTP 地址
AUTO_INTERVAL = float(os.environ.get("AUTO_INTERVAL", "0.6"))
AUTO_CONFIRM_FRAMES = int(os.environ.get("AUTO_CONFIRM_FRAMES", "4"))
AUTO_CONF_MIN = float(os.environ.get("AUTO_CONF_MIN", "0.6"))
AUTO_SINGLE_ITEM = os.environ.get("AUTO_SINGLE_ITEM", "1") == "1"   # 一次一件（赛项）
AUTO_DRY_RUN = os.environ.get("AUTO_DRY_RUN", "1") == "1"           # 默认不驱动机械臂
CALIB_PATH = os.environ.get("CALIB_PATH", str(DATA_DIR / "calib.json"))  # 标定文件（阶段4产物）
GRASP_ENABLED = os.environ.get("GRASP_ENABLED", "1") == "1"         # 是否解算抓取位姿

# ---- 赛项常量 ----
BIN_COUNT = 6
BIN_CAPACITY = 4
TRAY_SIZE_MM = 160
GOODS_SIZE_MM = 40

ROUND_REASONS = {
    "complete": "本轮分拣完成", "dropped_in": "货物掉落在装置范围内 —— 本轮比赛结束",
    "dropped_out": "货物掉落在装置外 —— 本轮比赛结束", "manual": "人工结束本轮",
    "estop": "紧急停止 —— 本轮比赛结束",
}
ROBOT_ACTIONS = {
    "start": ("working", "▶ 分拣装置已启动，比赛计时开始"),
    "pause": ("paused", "⏸ 分拣已暂停"),
    "reset": ("reset", "🔄 机械臂复位中…"),
    "estop": ("estop", "⛔ 已触发紧急停止，机械臂断电保护"),
    "idle": ("idle", "机械臂回归空闲"),
}
SEED_TASKS = [
    {"id": "T-260901", "items": [{"shape": "五棱柱", "color": "青色", "count": 3},
                                 {"shape": "正四面体", "color": "橙色", "count": 2}]},
    {"id": "T-260902", "items": [{"shape": "正方体", "color": "红色", "count": 4},
                                 {"shape": "正方体", "color": "蓝色", "count": 2}]},
    {"id": "T-260903", "items": [{"shape": "正方体", "color": "绿色", "count": 3},
                                 {"shape": "圆柱", "color": "蓝色", "count": 1}]},
    {"id": "T-260904", "items": [{"shape": "圆柱", "color": "黄色", "count": 2},
                                 {"shape": "球", "color": "黑色", "count": 2}]},
]
LOCK = threading.RLock()

# ==================== 状态 ====================
def empty_display():
    return {"sequence": [], "counts": {}, "total": 0, "current": None,
            "display_locked": False, "unlock_at": 0}


def empty_bins():
    bins = {}
    for i in range(1, BIN_COUNT + 1):
        bins[str(i)] = {"no": i, "key": None, "shape": None, "color": None, "items": []}
    return bins


def empty_round():
    return {"status": "idle", "started_at": None, "ended_at": None, "elapsed_ms": 0, "reason": None}


def normalize_task(task):
    t = dict(task)
    t["total"] = sum(int(it.get("count", 1)) for it in t.get("items", []))
    t["status"] = t.get("status", "available")
    return t


def default_state():
    return {"robot_state": "idle", "tasks": [normalize_task(t) for t in SEED_TASKS],
            "current_task": None, "round": empty_round(), "bins": empty_bins(),
            "display": empty_display()}


def save_state():
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(STATE, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception as e:
        logger.warning("状态持久化失败: {0}".format(e))


def load_state():
    state = None
    if STATE_FILE.exists():
        try:
            state = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("状态文件损坏，重置: {0}".format(e))
    if not state:
        state = default_state()
    state.setdefault("display", empty_display())
    state.setdefault("round", empty_round())
    if not state.get("bins"):
        state["bins"] = empty_bins()
    state.setdefault("tasks", [normalize_task(t) for t in SEED_TASKS])
    return state


STATE = load_state()
CALIB = pose.load_calib(CALIB_PATH)
CALIB_OK, CALIB_MSG = pose.calib_ready(CALIB) if GRASP_ENABLED else (False, "GRASP_ENABLED=0")
logger.info("标定: {0}（{1}）".format("可用" if CALIB_OK else "不可用", CALIB_MSG))

# 自动分拣循环状态
AUTO = {
    "status": "idle", "error": None, "started_at": None, "count": 0,
    "interval": AUTO_INTERVAL, "confirm_frames": AUTO_CONFIRM_FRAMES,
    "conf_min": AUTO_CONF_MIN, "single_item_only": AUTO_SINGLE_ITEM,
    "dry_run": AUTO_DRY_RUN, "require_ack": False, "max_items": 0,
    "waiting_clear": False, "track": {"key": None, "hits": 0},
    "last_marks": None, "last_event": None, "last_robot": None, "skipped": 0,
}
AUTO_STOP = threading.Event()


def refresh_lock():
    disp = STATE["display"]
    if disp.get("display_locked") and disp.get("unlock_at") and time.time() >= disp["unlock_at"]:
        disp["display_locked"] = False
        disp["unlock_at"] = 0


def round_elapsed():
    r = STATE["round"]
    if r["status"] == "running" and r.get("started_at"):
        return time.time() - r["started_at"]
    return (r.get("elapsed_ms") or 0) / 1000.0


def bins_view():
    out = []
    for key in sorted(STATE["bins"].keys(), key=lambda k: int(k)):
        b = STATE["bins"][key]
        out.append({"no": b["no"], "key": b["key"], "shape": b["shape"], "color": b["color"],
                    "count": len(b["items"]), "capacity": BIN_CAPACITY,
                    "full": len(b["items"]) >= BIN_CAPACITY, "empty": len(b["items"]) == 0,
                    "items": b["items"]})
    return out


def append_marks_log(record):
    """标记输出落盘（JSONL）：供机器人端/复盘读取"""
    try:
        MARKS_LOG.parent.mkdir(parents=True, exist_ok=True)
        with MARKS_LOG.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.warning("标记落盘失败: {0}".format(e))


# ==================== 赛项规则 ====================
def assign_bin(shape, color, requested=None):
    key = "{0}|{1}".format(shape, color)
    bins = STATE["bins"]
    for b in bins.values():
        if b["key"] == key:
            if len(b["items"]) >= BIN_CAPACITY:
                raise HTTPException(409, "{0}{1} 对应的 {2} 号储物盒已满（{3} 件），"
                                         "按赛项规定每个储物盒最多存放 {3} 件货物".format(
                                             color, shape, b["no"], BIN_CAPACITY))
            return b["no"]
    if requested:
        b = bins.get(str(requested))
        if b is None:
            raise HTTPException(400, "储物盒编号须为 1–{0}".format(BIN_COUNT))
        if b["key"]:
            raise HTTPException(409, "{0} 号储物盒已存放 {1}·{2}，同盒只能放同形同色货物".format(
                requested, b["shape"], b["color"]))
        b["key"], b["shape"], b["color"] = key, shape, color
        return b["no"]
    for b in bins.values():
        if not b["key"]:
            b["key"], b["shape"], b["color"] = key, shape, color
            return b["no"]
    raise HTTPException(409, "六个储物盒均已分配（最多 {0} 种形状+颜色组合），本轮无法继续分拣".format(BIN_COUNT))


def assert_intervention_allowed(debug):
    if STATE["round"]["status"] == "running" and not debug:
        raise HTTPException(409, "比赛进行中，按赛项规定参赛队员不得再次接触装置；"
                                 "如需联调请先结束本轮，或使用 ?debug=1")


def start_round():
    STATE["display"] = empty_display()
    STATE["bins"] = empty_bins()
    STATE["round"] = {"status": "running", "started_at": time.time(),
                      "ended_at": None, "elapsed_ms": 0, "reason": None}
    STATE["robot_state"] = "working"
    save_state()
    logger.info("[round] 比赛开始，计时启动，六个储物盒已清空")
    return STATE["round"]


def finish_round(reason):
    r = STATE["round"]
    now = time.time()
    r["status"] = "finished"
    r["ended_at"] = now
    r["elapsed_ms"] = int((now - r["started_at"]) * 1000) if r.get("started_at") else 0
    r["reason"] = ROUND_REASONS.get(reason, reason)
    STATE["robot_state"] = "estop" if reason == "estop" else "idle"
    save_state()
    logger.info("[round] 本轮结束（{0}），用时 {1:.1f}s".format(r["reason"], r["elapsed_ms"] / 1000.0))
    return r


def record_sort_event(payload):
    refresh_lock()
    if STATE["round"]["status"] == "finished":
        raise HTTPException(409, "本轮比赛已结束（{0}），请复位后开始新一轮".format(
            STATE["round"].get("reason") or "已停止"))
    disp = STATE["display"]
    if disp["display_locked"]:
        raise HTTPException(409, "上一件分拣信息尚未显示完成（剩余 {0:.1f}s），"
                                 "按赛项规定不得分拣下一件货物".format(max(0, disp["unlock_at"] - time.time())))
    class_id = str(payload.get("class") or "").strip()
    catalog = GOODS_BY_ID.get(class_id)
    shape = str(payload.get("shape") or (catalog or {}).get("shape") or "").strip()
    color = str(payload.get("color") or (catalog or {}).get("color") or "").strip()
    if not shape or not color:
        raise HTTPException(400, "无法确定货物属性：请给出图库货物编号 class，或同时提供 shape 与 color")
    matched = catalog or match_goods(shape, color)
    goods_id = (matched or {}).get("id") or class_id or "{0}_{1}".format(shape, color)

    bin_no = assign_bin(shape, color, payload.get("bin"))
    bin_obj = STATE["bins"][str(bin_no)]
    marks = payload.get("marks") or ([payload["mark"]] if payload.get("mark") else [])

    seq = len(disp["sequence"]) + 1
    cumulative = disp["total"] + 1
    record = {
        "seq": seq, "class": goods_id,
        "name": payload.get("name") or (matched or {}).get("name") or "{0}{1}".format(color, shape),
        "image": "/api/images/{0}.{1}".format(goods_id, payload.get("image_format") or "svg"),
        "image_format": payload.get("image_format") or "svg",
        "cumulative": cumulative, "bin": bin_no, "bin_no": bin_no,
        "shape": shape, "color": color, "marks": marks or ["无"],
        "text": payload.get("text") or "", "qr": payload.get("qr") or "",
        "confidence": payload.get("confidence"), "source": payload.get("source", "vision"),
        "box": payload.get("box"),
        "elapsed_ms": int(round_elapsed() * 1000), "ts": time.strftime("%H:%M:%S"),
    }
    # ---- 抓取位姿解算（阶段5）：像素 → 机械臂基坐标 / 夹爪目标 / 各段高度 ----
    if GRASP_ENABLED and CALIB_OK:
        try:
            det = {"box": payload.get("box") or _box_from_record(record, payload),
                   "shape": shape, "color": color,
                   "angle_deg": payload.get("angle_deg"),
                   "z_mm": payload.get("z_mm"), "height_mm": payload.get("height_mm")}
            record["grasp"] = pose.grasp_from_detection(
                det, H=CALIB.get("homography"), K=CALIB.get("camera"),
                R=(CALIB.get("rigid") or {}).get("R"), t=(CALIB.get("rigid") or {}).get("t"))
        except Exception as e:
            logger.warning("抓取位姿解算失败: {0}".format(e))
        try:
            record["place"] = pose.bin_place_pose(bin_no, CALIB.get("bins") or {})
        except Exception as e:
            logger.warning("投放位姿解算失败: {0}".format(e))
    elif not CALIB_OK:
        record["grasp_error"] = CALIB_MSG

    bin_obj["items"].append({"seq": seq, "name": record["name"],
                             "image": record["image"], "marks": record["marks"]})
    disp["sequence"].append(record)
    counts = dict(disp["counts"])
    counts[goods_id] = counts.get(goods_id, 0) + 1
    disp["counts"] = counts
    disp["total"] = cumulative
    disp["current"] = record
    disp["display_locked"] = True
    disp["unlock_at"] = time.time() + HOLD_SECONDS
    save_state()
    append_marks_log({"type": "sort_event", "ts": record["ts"], "record": record})
    logger.info("[投放顺序 {0}] {1} → {2} 号储物盒 ({3}/{4}) 分拣成功总数量 {5}；显示屏闩锁 {6}s".format(
        seq, record["name"], bin_no, len(bin_obj["items"]), BIN_CAPACITY, cumulative, HOLD_SECONDS))
    return record


# ==================== 视觉容器客户端 ====================
def vision_get(path, timeout=2.0):
    return requests.get("{0}{1}".format(VISION_URL, path), timeout=timeout)


def vision_detect_frame(timeout=3.0):
    r = vision_get("/detect/frame", timeout=timeout)
    r.raise_for_status()
    return r.json()


def _box_from_record(record, payload):
    """无视觉框时给出兜底抓取点（图像中心）——仅用于联调，现场应由视觉提供 box"""
    w = int(os.environ.get("FRAME_W", "1280"))
    h = int(os.environ.get("FRAME_H", "720"))
    cx, cy = w // 2, h // 2
    return [cx - 40, cy - 40, cx + 40, cy + 40]


def dispatch_robot(record):
    """把抓取/放置指令交给机器人执行器（ROS 桥接 HTTP）；dry_run 时仅记录"""
    payload = {
        "task": "pick_and_place", "seq": record["seq"], "class": record["class"],
        "name": record["name"], "shape": record["shape"], "color": record["color"],
        "bin": record["bin"], "marks": record["marks"], "qr": record["qr"],
        "confidence": record.get("confidence"),
        # 抓取与投放位姿（阶段5/6；由 calib.json 解算）
        "grasp": record.get("grasp"), "place": record.get("place"),
    }
    if AUTO["dry_run"] or not ROBOT_URL:
        AUTO["last_robot"] = {"mode": "dry_run", "payload": payload}
        logger.info("[robot] dry_run：{0} → {1} 号盒".format(record["name"], record["bin"]))
        return AUTO["last_robot"]
    try:
        resp = requests.post("{0}/execute".format(ROBOT_URL), json=payload, timeout=10)
        AUTO["last_robot"] = {"mode": "dispatched", "status": resp.status_code,
                              "reply": resp.text[:200]}
        logger.info("[robot] 已下发 {0} → {1} 号盒 (HTTP {2})".format(
            record["name"], record["bin"], resp.status_code))
    except Exception as e:
        AUTO["last_robot"] = {"mode": "error", "error": str(e)}
        logger.warning("[robot] 下发失败: {0}".format(e))
    return AUTO["last_robot"]


# ==================== 自动分拣循环 ====================
def auto_loop():
    logger.info("[auto] 自动分拣循环启动：视觉 {0} · 间隔 {1}s · 确认 {2} 帧 · dry_run={3}".format(
        VISION_URL, AUTO["interval"], AUTO["confirm_frames"], AUTO["dry_run"]))
    while not AUTO_STOP.is_set():
        started = time.time()
        try:
            marks = vision_detect_frame()
            dets = marks.get("detections") or []
            AUTO["last_marks"] = {"ts": marks.get("ts"), "count": len(dets),
                                  "top": dets[0] if dets else None,
                                  "qr_text": marks.get("qr_text", "")}

            if not dets:
                # 托盘已空 → 允许登记下一件（避免同一件重复入库）
                if AUTO["waiting_clear"]:
                    logger.info("[auto] 托盘已空，等待下一件货物")
                AUTO["waiting_clear"] = False
                AUTO["track"] = {"key": None, "hits": 0}
            elif AUTO["waiting_clear"]:
                pass                                        # 等机械臂取走，暂不登记
            elif AUTO["single_item_only"] and len(dets) > 1:
                AUTO["skipped"] += 1
                AUTO["track"] = {"key": None, "hits": 0}    # 赛项：一次只能分拣一件
            else:
                top = dets[0]
                key = "{0}|{1}|{2}".format(top.get("shape"), top.get("color"),
                                           "/".join(top.get("marks") or ["无"]))
                track = AUTO["track"]
                track = {"key": key, "hits": track["hits"] + 1} if track.get("key") == key else {"key": key, "hits": 1}
                AUTO["track"] = track

                if track["hits"] >= AUTO["confirm_frames"] and (top.get("conf") or 0) >= AUTO["conf_min"]:
                    refresh_lock()
                    if STATE["round"]["status"] != "running":
                        AUTO["skipped"] += 1                # 未开始本轮：只输出标记，不入库
                    elif STATE["display"]["display_locked"] and AUTO["require_ack"]:
                        pass                                # 等待人工确认显示完成
                    else:
                        with LOCK:
                            record = record_sort_event({
                                "class": top.get("class"), "shape": top.get("shape"),
                                "color": top.get("color"), "marks": top.get("marks"),
                                "text": top.get("text"), "qr": top.get("qr"),
                                "confidence": top.get("conf"), "box": top.get("box"),
                                "source": "auto",
                            })
                        AUTO["last_event"] = record
                        AUTO["count"] += 1
                        AUTO["waiting_clear"] = True
                        AUTO["track"] = {"key": None, "hits": 0}
                        dispatch_robot(record)
                        if AUTO["max_items"] and AUTO["count"] >= AUTO["max_items"]:
                            AUTO["status"] = "finished"
                            logger.info("[auto] 已达设定件数 {0}，循环结束".format(AUTO["count"]))
                            break
        except requests.exceptions.RequestException as e:
            AUTO["error"] = "视觉容器不可达: {0}".format(e)
            AUTO["status"] = "error"
            logger.warning("[auto] {0}".format(AUTO["error"]))
        except HTTPException as e:
            AUTO["error"] = e.detail
            logger.warning("[auto] 规则拦截: {0}".format(e.detail))
        except Exception as e:
            AUTO["error"] = str(e)
            logger.warning("[auto] 循环异常: {0}".format(e))

        sleep_left = AUTO["interval"] - (time.time() - started)
        AUTO_STOP.wait(max(0.05, sleep_left))
    logger.info("[auto] 自动分拣循环停止（已处理 {0} 件）".format(AUTO["count"]))


# ==================== FastAPI ====================
app = FastAPI(title="sort-gateway", docs_url=None, redoc_url=None)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
try:
    generate_goods_images(GOODS_IMG_DIR)
except Exception as _e:
    logger.warning("货物图片生成跳过: {0}".format(_e))


@app.get("/health")
async def health():
    vision = None
    try:
        vision = vision_get("/health", timeout=1.5).json()
    except Exception as e:
        vision = {"ok": False, "error": str(e)[:120]}
    # service/name 两个字段供 PC 端「自动发现 Jetson」识别身份：
    # Nano 的 IP 随所连 WiFi 热点变化，PC 端需要能确认扫描到的就是本网关，而不是别的 HTTP 服务。
    return {"ok": True, "service": "sort-gateway", "name": "智能分拣网关",
            "port": int(os.environ.get("PORT", "80")),
            # 下面这几个字段是给前端 SPA 判断"是否连上实际服务"用的：
            # 页面由网关自己托管时，网关当然可达 —— 之前缺这几个字段，
            # 导致在 Nano 上打开页面时顶部误报"未连接实际服务"、右侧面板全显示"无数据"。
            "deployed": True, "gateway_reachable": True, "pc_role": "gateway",
            "role": "Jetson 网关容器（页面由本容器托管）", "vision_url": VISION_URL,
            "cameras": cameras.camera_ids(), "vision": vision, "auto": AUTO["status"]}


@app.get("/api/status")
async def status():
    with LOCK:
        refresh_lock()
        return {
            "gateway": "ok", "robot_state": STATE["robot_state"],
            "round": dict(STATE["round"], elapsed=round_elapsed()),
            "display_locked": STATE["display"]["display_locked"],
            "sorted_total": STATE["display"]["total"], "bins": bins_view(),
            "auto": {"status": AUTO["status"], "count": AUTO["count"], "error": AUTO["error"],
                     "dry_run": AUTO["dry_run"], "waiting_clear": AUTO["waiting_clear"],
                     "last_event": AUTO["last_event"], "last_robot": AUTO["last_robot"]},
            "vision_url": VISION_URL, "robot_url": ROBOT_URL or None,
            "intervention_allowed": STATE["round"]["status"] != "running",
            "available_tasks": len([t for t in STATE["tasks"] if t["status"] == "available"]),
            "current_task": STATE.get("current_task"), "version": "0.6.0",
        }


@app.get("/api/display")
async def display():
    with LOCK:
        refresh_lock()
        return dict(STATE["display"], hold_seconds=HOLD_SECONDS,
                    round=dict(STATE["round"], elapsed=round_elapsed()),
                    bins=bins_view(),
                    constants={"bin_count": BIN_COUNT, "bin_capacity": BIN_CAPACITY,
                               "tray_size_mm": TRAY_SIZE_MM, "goods_size_mm": GOODS_SIZE_MM},
                    goods=[{k: g[k] for k in ("id", "name", "class", "shape", "color")} for g in GOODS],
                    shapes=SHAPES, colors=COLORS, marks=MARKS, formats=IMAGE_FORMATS)


# ---------- 轮次 ----------
@app.post("/api/round/start")
async def round_start():
    with LOCK:
        return {"ok": True, "round": start_round()}


@app.post("/api/round/stop")
async def round_stop(payload: dict = Body(default=None)):
    payload = payload or {}
    reason = payload.get("reason", "manual")
    if reason not in ROUND_REASONS:
        raise HTTPException(400, "未知结束原因 {0}".format(reason))
    with LOCK:
        return {"ok": True, "round": finish_round(reason)}


@app.post("/api/round/fault")
async def round_fault(payload: dict = Body(default=None)):
    payload = payload or {}
    with LOCK:
        return {"ok": True, "round": finish_round("dropped_out" if payload.get("where") == "out" else "dropped_in")}


# ---------- 分拣记录 ----------
@app.post("/api/sort/event")
async def sort_event(payload: dict = Body(...)):
    with LOCK:
        return {"ok": True, "record": record_sort_event(payload), "display": STATE["display"]}


@app.post("/api/sort/ack")
async def sort_ack():
    with LOCK:
        STATE["display"]["display_locked"] = False
        STATE["display"]["unlock_at"] = 0
        save_state()
    return {"ok": True, "display": STATE["display"]}


@app.post("/api/sort/reset")
async def sort_reset(debug: int = 0):
    with LOCK:
        assert_intervention_allowed(bool(debug))
        STATE["display"] = empty_display()
        STATE["bins"] = empty_bins()
        STATE["round"] = empty_round()
        STATE["robot_state"] = "idle"
        save_state()
    return {"ok": True}


@app.post("/api/sort/demo")
async def sort_demo(payload: dict = Body(default=None), debug: int = 0):
    payload = payload or {}
    with LOCK:
        assert_intervention_allowed(bool(debug))
        class_id = str(payload.get("class") or "").strip()
        if not class_id:
            pool = [it.get("class") for it in (STATE.get("current_task") or {}).get("items", [])
                    if it.get("class")] or [g["id"] for g in GOODS]
            counts = STATE["display"]["counts"]
            class_id = sorted(pool, key=lambda c: counts.get(c, 0))[0]
        record = record_sort_event({"class": class_id, "source": "demo"})
        return {"ok": True, "record": record, "display": STATE["display"]}


# ---------- 自动分拣 ----------
@app.post("/api/auto/start")
async def auto_start(payload: dict = Body(default=None)):
    payload = payload or {}
    if AUTO["status"] == "running":
        return {"ok": True, "auto": auto_status_payload(), "message": "自动分拣已在运行"}
    AUTO.update({
        "status": "running", "error": None, "started_at": time.time(),
        "interval": float(payload.get("interval", AUTO_INTERVAL)),
        "confirm_frames": int(payload.get("confirm_frames", AUTO_CONFIRM_FRAMES)),
        "conf_min": float(payload.get("conf_min", AUTO_CONF_MIN)),
        "single_item_only": bool(payload.get("single_item_only", AUTO_SINGLE_ITEM)),
        "dry_run": bool(payload.get("dry_run", AUTO_DRY_RUN)),
        "require_ack": bool(payload.get("require_ack", False)),
        "max_items": int(payload.get("max_items", 0)),
        "auto_start_round": bool(payload.get("auto_start_round", False)),
    })
    if payload.get("auto_start_round") and STATE["round"]["status"] != "running":
        with LOCK:
            start_round()
    AUTO_STOP.clear()
    threading.Thread(target=auto_loop, daemon=True).start()
    return {"ok": True, "auto": auto_status_payload()}


@app.post("/api/auto/stop")
async def auto_stop():
    AUTO_STOP.set()
    AUTO["status"] = "stopped"
    return {"ok": True, "auto": auto_status_payload()}


def auto_status_payload():
    return {"status": AUTO["status"], "count": AUTO["count"], "error": AUTO["error"],
            "dry_run": AUTO["dry_run"], "interval": AUTO["interval"],
            "confirm_frames": AUTO["confirm_frames"], "conf_min": AUTO["conf_min"],
            "single_item_only": AUTO["single_item_only"], "require_ack": AUTO["require_ack"],
            "max_items": AUTO["max_items"], "waiting_clear": AUTO["waiting_clear"],
            "skipped": AUTO["skipped"], "track": AUTO["track"],
            "last_marks": AUTO["last_marks"], "last_event": AUTO["last_event"],
            "last_robot": AUTO["last_robot"], "vision_url": VISION_URL,
            "robot_url": ROBOT_URL or None}


@app.get("/api/auto/status")
async def auto_status():
    return auto_status_payload()


# ---------- 标记输出 ----------
@app.get("/api/marks/latest")
async def marks_latest():
    """最近一帧标记（透传视觉容器）+ 最近一次入库记录"""
    try:
        marks = vision_get("/marks/latest", timeout=2.0).json()
    except Exception as e:
        marks = {"error": str(e)[:120], "detections": []}
    marks["last_event"] = AUTO["last_event"]
    return marks


@app.get("/api/marks/log")
async def marks_log(limit: int = 50):
    """标记日志（JSONL 尾部 N 条）"""
    if not MARKS_LOG.exists():
        return {"items": []}
    lines = MARKS_LOG.read_text(encoding="utf-8").strip().splitlines()
    items = []
    for line in lines[-max(1, min(limit, 500)):]:
        try:
            items.append(json.loads(line))
        except Exception:
            continue
    return {"items": items, "total": len(lines)}


@app.get("/api/frame.jpg")
async def frame_jpg(draw: int = 1):
    """当前帧（含标记）—— 同一容器内前端直接显示，避免跨域"""
    try:
        r = vision_get("/frame.jpg?draw={0}".format(1 if draw else 0), timeout=3.0)
        return Response(content=r.content, media_type="image/jpeg",
                        headers={"Cache-Control": "no-store, max-age=0"})
    except Exception as e:
        raise HTTPException(503, "视觉容器不可达: {0}".format(str(e)[:100]))


@app.get("/api/vision/health")
async def vision_health():
    try:
        return vision_get("/health", timeout=2.0).json()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)[:120]}, status_code=503)


# ---------- 多路相机（前端可同时显示多路画面） ----------
@app.get("/api/cameras")
async def cameras_list(refresh: int = 0):
    """相机清单 + 真实可用状态（不可达就如实报，不合成画面）"""
    cams = cameras.describe(force=bool(refresh))
    ready = [c["id"] for c in cams if c["ready"]]
    return {
        "cameras": cams,
        "ready": ready,
        "count": len(cams),
        "stats": cameras.stats(),
        "default_pair": (ready + [c["id"] for c in cams])[:2],
        "hint": "每路都可用 /api/cameras/<id>/stream.mjpg 直接作为 <img src> 显示",
    }


@app.get("/api/cameras/{cid}/frame.jpg")
async def camera_frame(cid: str):
    """单路单帧（抓拍/轮询用）"""
    try:
        data, ctype = cameras.snapshot(cid)
    except KeyError:
        raise HTTPException(404, "未知相机: {0}".format(cid))
    except Exception as e:
        raise HTTPException(503, "相机不可达（{0}）: {1}".format(cid, str(e)[:100]))
    return Response(content=data, media_type=ctype,
                    headers={"Cache-Control": "no-store, max-age=0"})


@app.get("/api/cameras/{cid}/stream.mjpg")
async def camera_stream(cid: str, request: Request):
    """单路 MJPEG 流 —— 长连接，前端 <img src> 直接显示。

    没有原生 MJPEG 的源（深度图等）由 cameras 模块按帧率轮询单帧并封装 multipart，
    前端因此对每一路都用同一套代码。该路不可达时**立即返回 503**。

    这里用**异步生成器轮询 request.is_disconnected()**，一旦客户端断开就关流、
    释放并发槽位。此前用同步生成器时，Starlette 无法中断阻塞在上游 read 上的线程，
    槽位会泄漏——刷新几次页面后 6 个槽位全被占满，之后所有画面都 503
    （前端表现为一片黑/破图，但相机清单仍显示"在线"）。
    """
    try:
        st, mtype, reason = cameras.open_stream(cid)
    except KeyError:
        raise HTTPException(404, "未知相机: {0}".format(cid))
    if st is None:
        raise HTTPException(503, "画面不可用（{0}）：{1}".format(cid, reason or "未知原因"))

    loop = asyncio.get_event_loop()

    async def gen():
        try:
            while True:
                if await request.is_disconnected():
                    break
                if st.expired():
                    logger.info("流 %s 超过最长存活时间，主动结束", cid)
                    break
                chunk = await loop.run_in_executor(None, st.read, 1.0)
                if chunk is None:          # 上游结束
                    break
                if chunk:
                    yield chunk
        finally:
            st.close()                     # 幂等释放槽位

    return StreamingResponse(gen(), media_type=mtype,
                             headers={"Cache-Control": "no-store, max-age=0",
                                      "X-Accel-Buffering": "no"})


# ---------- 识别（手动，转发视觉容器） ----------
@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    data = await file.read()
    try:
        r = requests.post("{0}/detect".format(VISION_URL),
                          files={"file": ("snapshot.jpg", data, "image/jpeg")}, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        # 视觉容器不可达 → 网关本地兜底识别（便于单容器联调）
        import cv2
        import numpy as np
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise HTTPException(400, "无法解析图片")
        dets, qr_text, size = detect_core.analyze(img)
        return {"engine": "gateway-local", "detections": dets, "qr_text": qr_text,
                "size": list(size), "message": "视觉容器不可达（{0}），已用网关本地引擎识别".format(str(e)[:60])}


# ---------- 货物图库 / 图片 ----------
@app.get("/api/goods")
async def goods_list():
    items = []
    for g in GOODS:
        item = dict((k, g[k]) for k in ("id", "name", "class", "shape", "color"))
        item["images"] = formats_for(GOODS_IMG_DIR, g["id"])
        items.append(item)
    return {"goods": items, "shapes": SHAPES, "colors": COLORS, "marks": MARKS,
            "formats": IMAGE_FORMATS, "bin_count": BIN_COUNT, "bin_capacity": BIN_CAPACITY}


@app.get("/api/images/{filename}")
async def goods_image(filename: str):
    stem, _, ext = filename.rpartition(".")
    stem, ext = stem or filename, (ext or "svg").lower()
    if ext == "jpeg":
        ext = "jpg"
    if ext not in IMAGE_FORMATS:
        raise HTTPException(400, "不支持的图片格式 {0}".format(ext))
    path = GOODS_IMG_DIR / "{0}.{1}".format(stem, ext)
    if path.exists():
        mime = {"png": "image/png", "jpg": "image/jpeg", "webp": "image/webp",
                "gif": "image/gif", "bmp": "image/bmp", "svg": "image/svg+xml"}[ext]
        return FileResponse(str(path), media_type=mime, headers={"Cache-Control": "no-cache"})
    if stem in GOODS_BY_ID:
        return Response(content=goods_svg(GOODS_BY_ID[stem]), media_type="image/svg+xml",
                        headers={"Cache-Control": "no-cache"})
    raise HTTPException(404, "未找到图片 {0}".format(filename))


@app.post("/api/goods/{goods_id}/image")
async def upload_goods_image(goods_id: str, file: UploadFile = File(...)):
    if goods_id not in GOODS_BY_ID:
        raise HTTPException(404, "未知货物 {0}".format(goods_id))
    ext = (file.filename or "upload.png").rpartition(".")[2].lower() or "png"
    if ext == "jpeg":
        ext = "jpg"
    if ext not in IMAGE_FORMATS:
        raise HTTPException(400, "不支持的格式 {0}".format(ext))
    GOODS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    (GOODS_IMG_DIR / "{0}.{1}".format(goods_id, ext)).write_bytes(await file.read())
    return {"ok": True, "image": "/api/images/{0}.{1}".format(goods_id, ext)}


# ---------- 任务 ----------
@app.get("/api/tasks")
async def tasks(available: int = 0):
    with LOCK:
        items = [normalize_task(t) for t in STATE["tasks"]]
    if available:
        items = [t for t in items if t["status"] == "available"]
    return {"robot_state": STATE["robot_state"], "tasks": items, "current": STATE.get("current_task")}


@app.post("/api/tasks/claim")
async def claim_task(payload: dict = Body(...)):
    task_id = str(payload.get("task_id", "")).strip()
    worker = str(payload.get("worker", "操作员")).strip() or "操作员"
    with LOCK:
        for t in STATE["tasks"]:
            if t["id"].upper() == task_id.upper():
                if t["status"] != "available":
                    raise HTTPException(409, "任务 {0} 已被 {1} 领取".format(t["id"], t.get("claimed_by", "他人")))
                t.update({"status": "claimed", "claimed_by": worker,
                          "claimed_at": time.strftime("%Y-%m-%d %H:%M:%S")})
                STATE["current_task"] = normalize_task(t)
                STATE["display"] = empty_display()
                STATE["bins"] = empty_bins()
                STATE["round"] = empty_round()
                save_state()
                return {"ok": True, "task": STATE["current_task"]}
        raise HTTPException(404, "未找到任务 {0}".format(task_id))


@app.post("/api/tasks/{task_id}/{flag}")
async def finish_task(task_id: str, flag: str):
    if flag not in ("complete", "cancel"):
        raise HTTPException(400, "flag 须为 complete | cancel")
    with LOCK:
        for t in STATE["tasks"]:
            if t["id"].upper() == task_id.upper():
                t["status"] = flag
                STATE["current_task"] = None
                save_state()
                return {"ok": True}
        raise HTTPException(404, "未找到任务 {0}".format(task_id))


@app.post("/api/robot/action")
async def robot_action(payload: dict = Body(...)):
    action = str(payload.get("action", "")).lower()
    if action not in ROBOT_ACTIONS:
        raise HTTPException(400, "未知指令 {0}".format(action))
    robot_state, message = ROBOT_ACTIONS[action]
    with LOCK:
        STATE["robot_state"] = robot_state
        save_state()
    if action == "estop" and STATE["round"]["status"] == "running":
        with LOCK:
            finish_round("estop")
    return {"ok": True, "action": action, "state": robot_state, "message": message}


# ---------- 标定（阶段4）与机器人结果回报 ----------
@app.get("/api/calib")
async def calib_get():
    ok, msg = pose.calib_ready(CALIB) if CALIB else (False, "未加载标定")
    return {"path": CALIB_PATH, "ready": bool(ok), "message": msg,
            "calib": CALIB, "gripper": pose.GRIPPER, "pick": pose.PICK,
            "workspace": pose.WORKSPACE}


@app.post("/api/calib")
async def calib_put(payload: dict = Body(...)):
    """上传/更新标定（由 scripts/calibrate.py 生成）"""
    global CALIB, CALIB_OK, CALIB_MSG
    pose.save_calib(payload, CALIB_PATH)
    CALIB = pose.load_calib(CALIB_PATH)
    CALIB_OK, CALIB_MSG = pose.calib_ready(CALIB)
    logger.info("标定已更新: {0}".format(CALIB_MSG))
    return {"ok": True, "ready": bool(CALIB_OK), "message": CALIB_MSG}


@app.post("/api/robot/result")
async def robot_result(payload: dict = Body(...)):
    """机械臂执行结果回报：ok | fail | dropped_in | dropped_out
       —— 抓取失败可触发重试；掉落按赛项结束本轮"""
    status = str(payload.get("status", "")).lower()
    seq = payload.get("seq")
    detail = payload.get("detail", "")
    logger.info("[robot] seq={0} status={1} {2}".format(seq, status, detail))
    if status in ("dropped_in", "dropped_out"):
        with LOCK:
            finish_round(status)
    state = {"seq": seq, "status": status, "detail": detail,
             "ts": time.strftime("%H:%M:%S")}
    AUTO["last_result"] = state
    append_marks_log({"type": "robot_result", "ts": state["ts"], **state})
    return {"ok": True, "result": state, "round": STATE["round"]["status"]}


@app.get("/api/rules")
async def rules():
    return {"bins": {"count": BIN_COUNT, "capacity": BIN_CAPACITY,
                     "assignment": "same_shape_color_same_bin"},
            "tray": {"size_mm": TRAY_SIZE_MM, "edge_height_mm": 10, "goods_size_mm": GOODS_SIZE_MM},
            "shapes": SHAPES, "colors": COLORS, "marks": MARKS,
            "cycle": {"one_item_at_a_time": True, "display_hold_seconds": HOLD_SECONDS,
                      "drop_fault_ends_round": True}}


if STATIC_DIR.exists():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR), html=True), name="spa")
else:
    @app.get("/")
    async def missing_build():
        return JSONResponse({"error": "build/ 未找到，请先在 PC 上执行 npm run build:static"}, status_code=500)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "80")))
