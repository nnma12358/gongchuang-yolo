#!/usr/bin/env python3
"""
分拣装置侧客户端 —— 视觉/机械臂节点 → 显示屏网关
=================================================================
赛项流程（现场初赛）：
  1) 视觉节点识别货物 → POST /api/detect（可选）
  2) 机械臂完成一件分拣 → POST /api/sort/event（写入投放顺序并闩锁显示屏）
  3) 若返回 409，说明上一件分拣信息尚未显示完成 —— 按赛项规定不得分拣下一件，
     本脚本会自动等待并重试
  4) 显示屏信息展示完成后（自动超时或人工确认）即可分拣下一件

用法：
  python3 sort_client.py --gateway http://localhost:8090 --demo 5
  python3 sort_client.py --class pentagon_prism
  python3 sort_client.py --class red_cube --bin A --wait
"""
import argparse
import json
import time
import urllib.error
import urllib.request


def request(method, url, payload=None, timeout=10.0):
    """HTTP 请求（兼容 Jetson Nano 上的 Python 3.6）"""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, method=method,
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, json.loads(resp.read() or b"{}")
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="ignore")
        try:
            return e.code, json.loads(body)
        except Exception:
            return e.code, {"detail": body}


def post_event(gateway, class_id, **extra):
    payload = {"class": class_id, **extra}
    while True:
        code, body = request("POST", f"{gateway}/api/sort/event", payload)
        if code == 200:
            rec = body["record"]
            print("  ✔ 投放顺序 #{0:<3} {1:<10} → {2} 号储物盒   "
                  "分拣成功总数量 {3}  表面 {4}".format(
                      rec["seq"], rec["name"], rec["bin"], rec["cumulative"],
                      "/".join(rec.get("marks") or ["无"])))
            return rec
        if code == 409:
            detail = body.get("detail", "")
            # 本轮已结束 / 储物盒已满 → 等待无意义，直接报告
            if "本轮比赛已结束" in detail or "已满" in detail or "均已分配" in detail:
                print("  ✘ 无法分拣：{0}".format(detail))
                return None
            print("  ⏳ 显示屏显示中（{0}）等待 0.4s".format(detail[:28]))
            time.sleep(0.4)
            continue
        print("  ✘ 上报失败 [{0}] {1}".format(code, body.get("detail")))
        return None


def main():
    ap = argparse.ArgumentParser(description="分拣显示屏网关客户端（机器人侧）")
    ap.add_argument("--gateway", default="http://localhost", help="网关地址")
    ap.add_argument("--class", dest="cls", help="货物编号（如 prism5_4 / cube_0），未知货物可用 --shape+--color")
    ap.add_argument("--shape", help="形状（图库无此货物时使用，如 五棱柱）")
    ap.add_argument("--color", help="颜色（如 青色）")
    ap.add_argument("--marks", help="表面标记，逗号分隔：污渍,缺陷")
    ap.add_argument("--text", help="表面文字")
    ap.add_argument("--qr", help="表面二维码内容（如 SORT-TASK:T-260901）")
    ap.add_argument("--name", help="货物名称（未知类别时使用）")
    ap.add_argument("--bin", help="指定储物盒编号（默认按“同形同色同盒”自动分配）")
    ap.add_argument("--conf", type=float, help="识别置信度（记录用）")
    ap.add_argument("--demo", type=int, help="模拟分拣 N 件（走网关 /api/sort/demo）")
    ap.add_argument("--wait", action="store_true", help="等待显示屏放行后再上报")
    ap.add_argument("--ack", action="store_true", help="显示完成后由脚本确认放行")
    args = ap.parse_args()

    code, health = request("GET", f"{args.gateway}/api/health".replace("/api/health", "/health"))
    print(f"网关 {args.gateway} 健康检查: {code} {health}")

    if args.demo:
        for i in range(args.demo):
            print("[{0}/{1}]".format(i + 1, args.demo))
            code, body = request("POST", "{0}/api/sort/demo?debug=1".format(args.gateway), {})
            guard = 0
            while code == 409 and guard < 60:
                detail = body.get("detail", "")
                if "本轮比赛已结束" in detail:
                    print("  ✘ {0}".format(detail))
                    return
                print("  ⏳ {0}".format(detail[:36]))
                time.sleep(0.4)
                guard += 1
                code, body = request("POST", "{0}/api/sort/demo?debug=1".format(args.gateway), {})
            if code == 200:
                rec = body["record"]
                print("  ✔ #{0} {1} → {2} 号储物盒  分拣成功总数量 {3}".format(
                    rec["seq"], rec["name"], rec["bin"], rec["cumulative"]))
                if args.ack:
                    time.sleep(0.2)
                    request("POST", "{0}/api/sort/ack".format(args.gateway), {})
        return

    if not args.cls and not (args.shape and args.color):
        ap.error("需要 --class，或同时给出 --shape 与 --color（--demo 可无）")

    if args.wait:
        while True:
            code, body = request("GET", "{0}/api/display".format(args.gateway))
            if code == 200 and not body.get("display_locked"):
                break
            time.sleep(0.3)

    marks = [m.strip() for m in (args.marks or "").split(",") if m.strip()]
    post_event(args.gateway, args.cls, shape=args.shape, color=args.color,
               marks=marks, text=args.text, qr=args.qr,
               name=args.name, bin=args.bin, confidence=args.conf, source="robot")


if __name__ == "__main__":
    main()
