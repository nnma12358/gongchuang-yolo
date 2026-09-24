#!/usr/bin/env bash
# 一键起「离线演示栈」——没有 Jetson、没有相机也能看前端双画面
# ------------------------------------------------------------------
# 起 5 个进程：
#   8192  模拟 ROS 桥接上游（彩色 + 深度预览）
#   8194  模拟视觉容器上游（带标记画面）
#   8103  真实的第二路相机服务（合成测试图，即 deploy/jetson/cam2_node.py）
#   8099  真实网关（server/gateway.py，指向上面三个）
#   3100  前端 PC 壳（adapter-node 产物），代理到 :8099
#
# 用法：  bash scripts/dev_demo_stack.sh          # 起（脚本随即退出，服务在后台）
#         bash scripts/dev_demo_stack.sh --stop   # 停掉全部
# 浏览器打开：  http://localhost:3100
set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY=${PY:-$ROOT/.venv-gw/bin/python}
[ -x "$PY" ] || PY=$(command -v python3)
PIDS=/tmp/sort_demo_stack.pids

stop_all() {
  if [ -f "$PIDS" ]; then
    while read -r p; do [ -n "$p" ] && kill "$p" 2>/dev/null && echo "停止 pid=$p"; done < "$PIDS"
    rm -f "$PIDS"
  fi
  for port in 3100 8099 8192 8194 8103; do
    pid=$(ss -ltnp 2>/dev/null | grep ":$port" | grep -o 'pid=[0-9]*' | head -1 | cut -d= -f2)
    [ -n "${pid:-}" ] && kill "$pid" 2>/dev/null && echo "停止占用 :$port 的 pid=$pid"
  done
  return 0
}

if [ "${1:-}" = "--stop" ]; then stop_all; exit 0; fi
stop_all >/dev/null 2>&1
: > "$PIDS"

# setsid：让服务脱离当前会话，脚本退出后继续运行
spawn() { setsid nohup "$@" >/dev/null 2>&1 & echo $! >> "$PIDS"; }

echo "① 模拟上游（桥接 :8192 / 视觉 :8194）…"
spawn "$PY" "$ROOT/scripts/dev_mock_cameras.py" --port 8192 --mode bridge
spawn "$PY" "$ROOT/scripts/dev_mock_cameras.py" --port 8194 --mode vision

echo "② 第二路相机服务（:8103，合成测试图）…"
CAM2_SOURCE=synthetic CAM2_PORT=8103 CAM2_NAME="第二路相机(测试图)" \
  setsid nohup "$PY" "$ROOT/deploy/jetson/cam2_node.py" >/tmp/demo_cam2.log 2>&1 &
echo $! >> "$PIDS"

sleep 4
echo "③ 网关（:8099）…"
PORT=8099 BRIDGE_URL=http://127.0.0.1:8192 VISION_URL=http://127.0.0.1:8194 \
  CAM2_URL=http://127.0.0.1:8103 DATA_DIR=/tmp/demo_gw_data STATIC_DIR=/tmp/demo_no_static \
  setsid nohup "$PY" "$ROOT/server/gateway.py" >/tmp/demo_gateway.log 2>&1 &
echo $! >> "$PIDS"

if [ ! -f "$ROOT/build/index.js" ]; then
  echo "④ 构建前端…"; npm run build >/tmp/demo_build.log 2>&1 || { echo "构建失败：见 /tmp/demo_build.log"; exit 1; }
fi
sleep 3
echo "⑤ 前端 PC 壳（:3100）…"
PORT=3100 GATEWAY_URL=http://127.0.0.1:8099 \
  setsid nohup node "$ROOT/build/index.js" >/tmp/demo_pcshell.log 2>&1 &
echo $! >> "$PIDS"

sleep 6
echo
echo "================ 就绪 ================"
curl -s -m 6 -o /dev/null -w "  前端页面    http://localhost:3100              HTTP %{http_code}\n" http://127.0.0.1:3100/
curl -s -m 6 -o /dev/null -w "  相机清单    http://localhost:3100/api/cameras    HTTP %{http_code}\n" http://127.0.0.1:3100/api/cameras
curl -s -m 6 -o /dev/null -w "  第一路流    .../api/cameras/marked/stream.mjpg  HTTP %{http_code}\n" http://127.0.0.1:3100/api/cameras/marked/stream.mjpg
echo "  浏览器打开： http://localhost:3100"
echo "  停止：       bash scripts/dev_demo_stack.sh --stop"
echo "======================================"
