#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ROS 桥接（参考实现）—— 网关自动分拣 → R550A 机械臂动作
=====================================================================
网关在登记一件分拣后，会向 ROBOT_URL（默认 http://127.0.0.1:8120）POST /execute：
    {"task":"pick_and_place","seq":3,"class":"prism5_4","name":"青色五棱柱",
     "shape":"五棱柱","color":"青色","bin":1,"marks":["污渍"],"qr":"SORT-TASK:T-260901"}

本脚本把它翻译成 ROS 调用（按你的实际接口二选一）：
  · 话题：发布自定义 msg 到 /arm_pick_place（字段：bin、seq、name）
  · 服务：调用 /arm_pick_place_srv（std_srvs/Trigger 或自定义 srv）

运行（宿主机 ROS Melodic 或 ros:melodic-ros-base 容器，host 网络）：
  ROS_MASTER_URI=http://127.0.0.1:11311 PORT=8120 python3 ros_bridge.py
"""
import json
import logging
import os
import threading
import time

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("ros-bridge")

PORT = int(os.environ.get("PORT", "8120"))
TOPIC = os.environ.get("ARM_TOPIC", "/arm_pick_place")
SERVICE = os.environ.get("ARM_ACTION", "")
DRY_RUN = os.environ.get("ROS_DRY_RUN", "0") == "1"

STATE = {"queue": [], "done": [], "last": None}


def ros_publish(payload):
    """发布抓取/放置指令到 ROS 话题（需要 rospy 与自定义 msg；此处用 std_msgs/String 传 JSON）"""
    import rospy
    from std_msgs.msg import String
    pub = rospy.Publisher(TOPIC, String, queue_size=10)
    rospy.sleep(0.2)                       # 等待订阅者连接
    pub.publish(String(data=json.dumps(payload, ensure_ascii=False)))
    logger.info("已发布到 {0}: {1} → {2} 号盒".format(TOPIC, payload.get("name"), payload.get("bin")))


def ros_call_service(payload):
    """调用抓取服务（按实际 srv 类型替换）"""
    import rospy
    from std_srvs.srv import Trigger
    rospy.wait_for_service(SERVICE, timeout=5)
    call = rospy.ServiceProxy(SERVICE, Trigger)
    resp = call()
    logger.info("服务 {0} 返回: {1}".format(SERVICE, getattr(resp, "message", resp)))
    return resp


def handle(payload):
    STATE["last"] = payload
    STATE["queue"].append(payload)
    try:
        if DRY_RUN:
            logger.info("[dry-run] 收到抓取指令: {0}".format(payload))
        elif SERVICE:
            ros_call_service(payload)
        else:
            ros_publish(payload)
        STATE["done"].append(payload)
        return {"ok": True, "mode": "dry-run" if DRY_RUN else ("service" if SERVICE else "topic")}
    except Exception as e:
        logger.warning("ROS 调用失败: {0}".format(e))
        return {"ok": False, "error": str(e)}


def main():
    from fastapi import Body, FastAPI
    import uvicorn

    app = FastAPI(title="sort-ros-bridge", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health():
        return {"ok": True, "topic": TOPIC, "service": SERVICE or None,
                "dry_run": DRY_RUN, "done": len(STATE["done"]), "last": STATE["last"]}

    @app.post("/execute")
    def execute(payload: dict = Body(...)):
        return handle(payload)

    @app.get("/queue")
    def queue():
        return STATE

    logger.info("ROS 桥接启动: :{0} → {1}".format(PORT, SERVICE or TOPIC))
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")


if __name__ == "__main__":
    main()
