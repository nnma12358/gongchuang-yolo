#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 2 桥接节点（运行在 ros2 容器内，与机械臂驱动共用 DDS）
=====================================================================
职责：把网关的抓取指令（HTTP）翻译成 ROS 2 调用，并把执行结果回报给网关。

数据流：
    网关容器 ──HTTP POST /execute──► 本节点 ──ROS2 话题/服务──► 机械臂驱动
        ▲                                                        │
        └────────── HTTP POST /api/robot/result ◄────────────────┘（可选回报）

接收的指令（来自网关 /api/robot/execute 的 payload）：
    {"task":"pick_and_place","seq":3,"class":"prism5_4","name":"青色五棱柱",
     "bin":1,"marks":["污渍"],
     "grasp":{"grasp_xy":[-54.9,3.0],"yaw_deg":15.0,"joint6_target":0.854,
              "z_approach":130.0,"z_grasp":20.0,"z_lift":120.0,"reachable":true},
     "place":{"bin":1,"place_xy":[120,60],"z_approach":45,"z_release":25}}

环境变量：
    PORT              本节点 HTTP 端口（默认 8120）
    ARM_TOPIC         ROS2 话题（默认 /arm/pick_place，std_msgs/String 承载 JSON）
    ARM_SERVICE       若设置则改调服务（std_srvs/Trigger，按现场 srv 类型可改）
    ARM_ACTION        可选：action 名称（现场用 action 时填，见 send_via_action）
    GATEWAY_URL       网关地址（回报执行结果；留空则不回报）
    ROS_DRY_RUN       1=只打印不调用 ROS（无硬件联调）
    JOINT6_CLOSED/OPEN 夹爪关节范围（现场标定：0.20 / 1.20）
"""
import json
import logging
import os
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ros2-bridge")

PORT = int(os.environ.get("PORT", "8120"))
ARM_TOPIC = os.environ.get("ARM_TOPIC", "/arm/pick_place")
ARM_SERVICE = os.environ.get("ARM_SERVICE", "")
GATEWAY_URL = os.environ.get("GATEWAY_URL", "").rstrip("/")
DRY_RUN = os.environ.get("ROS_DRY_RUN", "0") == "1"
JOINT6_CLOSED = float(os.environ.get("JOINT6_CLOSED", "0.20"))
JOINT6_OPEN = float(os.environ.get("JOINT6_OPEN", "1.20"))
DEFAULT_PICK_Z = float(os.environ.get("DEFAULT_PICK_Z", "20.0"))

STATE = {"last": None, "done": [], "results": [], "ros_ready": False}

# ---------------- ROS 2 ----------------
_pub = None
_node = None


def ros_init():
    """初始化 rclpy 节点与发布者（失败不阻塞：仍可用 HTTP 接口联调）"""
    global _pub, _node, STATE
    try:
        import rclpy
        from std_msgs.msg import String
        rclpy.init(args=None)
        _node = rclpy.create_node("sort_gateway_bridge")
        _pub = _node.create_publisher(String, ARM_TOPIC, 10)
        STATE["ros_ready"] = True
        logger.info("ROS2 就绪：发布话题 {0}".format(ARM_TOPIC))
        threading.Thread(target=rclpy.spin, args=(_node,), daemon=True).start()
    except Exception as e:
        logger.warning("ROS2 初始化失败（{0}）：将以 dry-run 方式运行".format(e))
        STATE["ros_ready"] = False


def build_command(payload):
    """把网关指令整理成机械臂执行器可直接消费的 JSON（含抓取/投放位姿与夹爪目标）"""
    grasp = payload.get("grasp") or {}
    place = payload.get("place") or {}
    joint6 = grasp.get("joint6_target")
    if joint6 is None:                     # 无标定时按目标宽度兜底
        joint6 = JOINT6_OPEN
    return {
        "seq": payload.get("seq"),
        "name": payload.get("name"),
        "shape": payload.get("shape"),
        "color": payload.get("color"),
        "bin": payload.get("bin"),
        "marks": payload.get("marks"),
        "stage": ["home", "approach", "descend", "close_gripper", "lift", "move_to_bin", "release", "home"],
        "pick": {
            "x": (grasp.get("grasp_xy") or [0, 0])[0],
            "y": (grasp.get("grasp_xy") or [0, 0])[1],
            "yaw_deg": grasp.get("yaw_deg", 0.0),
            "z_approach": grasp.get("z_approach"),
            "z_grasp": grasp.get("z_grasp", DEFAULT_PICK_Z),
            "z_lift": grasp.get("z_lift"),
            "joint6_close": joint6,
            "reachable": grasp.get("reachable", True),
        },
        "place": {
            "bin": place.get("bin") or payload.get("bin"),
            "x": (place.get("place_xy") or [0, 0])[0],
            "y": (place.get("place_xy") or [0, 0])[1],
            "z_approach": place.get("z_approach"),
            "z_release": place.get("z_release"),
            "joint6_open": JOINT6_OPEN,
        },
        "source": "gateway",
        "ts": time.strftime("%H:%M:%S"),
    }


def send_to_arm(cmd):
    """下发指令：话题 / 服务 / dry-run"""
    if DRY_RUN or not STATE["ros_ready"]:
        logger.info("[dry-run] 抓取 {0} → 放置 {1} 号盒 | pick=({0})".format(
            cmd["seq"], cmd["place"]["bin"]))
        return {"ok": True, "mode": "dry-run"}
    try:
        if ARM_SERVICE:
            import rclpy
            from std_srvs.srv import Trigger
            cli = _node.create_client(Trigger, ARM_SERVICE)
            if not cli.wait_for_service(timeout_sec=3.0):
                return {"ok": False, "error": "服务不可用: {0}".format(ARM_SERVICE)}
            fut = cli.call_async(Trigger.Request())
            t0 = time.time()
            while not fut.done() and time.time() - t0 < 30:
                time.sleep(0.05)
            return {"ok": True, "mode": "service"}
        from std_msgs.msg import String
        _pub.publish(String(data=json.dumps(cmd, ensure_ascii=False)))
        logger.info("已发布 {0}: #{1} {2} → {3} 号盒".format(
            ARM_TOPIC, cmd["seq"], cmd["name"], cmd["place"]["bin"]))
        return {"ok": True, "mode": "topic"}
    except Exception as e:
        logger.warning("下发失败: {0}".format(e))
        return {"ok": False, "error": str(e)}


def report_result(seq, status, detail=""):
    """把执行结果回报给网关（抓取失败/掉落会驱动状态机与比赛轮次）"""
    STATE["results"].append({"seq": seq, "status": status, "detail": detail, "ts": time.time()})
    if not GATEWAY_URL:
        return
    try:
        import urllib.request
        req = urllib.request.Request(
            GATEWAY_URL + "/api/robot/result",
            data=json.dumps({"seq": seq, "status": status, "detail": detail}).encode(),
            headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(req, timeout=5).read()
        logger.info("已回报网关: seq={0} status={1}".format(seq, status))
    except Exception as e:
        logger.warning("回报网关失败: {0}".format(e))


# ---------------- HTTP 接口（供网关调用） ----------------
def _handle(path, payload):
    """路由的**纯逻辑**部分（FastAPI 与 stdlib 两条后端共用，保证行为一致）"""
    if path == "/health":
        return 200, {"ok": True, "ros_ready": STATE["ros_ready"], "dry_run": DRY_RUN,
                     "topic": ARM_TOPIC, "service": ARM_SERVICE or None,
                     "gateway": GATEWAY_URL or None, "done": len(STATE["done"]),
                     "last": STATE["last"]}
    if path == "/execute":
        STATE["last"] = payload
        cmd = build_command(payload or {})
        res = send_to_arm(cmd)
        if res.get("ok"):
            STATE["done"].append(cmd)
        return 200, {"ok": res.get("ok", False), "mode": res.get("mode"),
                     "command": cmd, "error": res.get("error")}
    if path == "/result":
        report_result((payload or {}).get("seq"), (payload or {}).get("status", "unknown"),
                      (payload or {}).get("detail", ""))
        return 200, {"ok": True}
    if path == "/queue":
        return 200, STATE
    return 404, {"detail": "not found"}


def serve_stdlib():
    """无 fastapi 时的内置 HTTP 服务（只用标准库）——这样现场镜像里什么都不用装也能跑。"""
    import json as _json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, obj):
            body = _json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            code, obj = _handle(self.path.split("?")[0], None)
            self._send(code, obj)

        def do_POST(self):
            n = int(self.headers.get("Content-Length") or 0)
            try:
                payload = _json.loads(self.rfile.read(n) or b"{}")
            except Exception:
                payload = {}
            code, obj = _handle(self.path.split("?")[0], payload)
            self._send(code, obj)

    logger.info("ROS2 桥接启动（内置 stdlib HTTP）: :{0} | topic={1} | dry_run={2}".format(
        PORT, ARM_TOPIC, DRY_RUN))
    ThreadingHTTPServer(("0.0.0.0", PORT), H).serve_forever()


def main():
    ros_init()
    try:
        from fastapi import Body, FastAPI
        import uvicorn
    except ImportError:
        logger.warning("未安装 fastapi → 使用内置 stdlib HTTP 服务（功能相同）")
        return serve_stdlib()

    app = FastAPI(title="sort-ros2-bridge", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return {"ok": True, "ros_ready": STATE["ros_ready"], "dry_run": DRY_RUN,
                "topic": ARM_TOPIC, "service": ARM_SERVICE or None,
                "gateway": GATEWAY_URL or None, "done": len(STATE["done"]),
                "last": STATE["last"]}

    @app.post("/execute")
    def execute(payload: dict = Body(...)):
        STATE["last"] = payload
        cmd = build_command(payload)
        res = send_to_arm(cmd)
        if res.get("ok"):
            STATE["done"].append(cmd)
        return {"ok": res.get("ok", False), "mode": res.get("mode"),
                "command": cmd, "error": res.get("error")}

    @app.post("/result")
    def result(payload: dict = Body(...)):
        """机械臂执行器可主动调用：{"seq":3,"status":"ok|fail|dropped_in|dropped_out"}"""
        report_result(payload.get("seq"), payload.get("status", "unknown"), payload.get("detail", ""))
        return {"ok": True}

    @app.get("/queue")
    def queue():
        return STATE

    logger.info("ROS2 桥接启动: HTTP :{0} | topic={1} | service={2} | dry_run={3}".format(
        PORT, ARM_TOPIC, ARM_SERVICE or "-", DRY_RUN))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
