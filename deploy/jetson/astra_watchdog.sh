#!/usr/bin/env bash
# ============================================================
# Astra 相机看护（在 Jetson 宿主机上跑，systemd 常驻）
# ------------------------------------------------------------
# 现场实际情况：相机经常被拔插、设备偶发重启，而"重启后要手动起容器 + 起相机节点"
# 几乎每次都要人来一遍。这个看护把这些动作自动化：
#
#   1) ROS 容器没在跑 → 自动 docker start（**只起容器，不起机械臂驱动**）
#   2) USB 上出现 Astra（2bc5:0402）但相机节点没跑 → 自动拉起 /start_astra.sh
#   3) 相机节点在跑但画面**冻结**（桥接的 color_frozen_s 持续增大）→ 自动重启节点
#   4) 相机被拔掉（USB 消失）→ 什么都不做，只等它回来（不会刷日志、不会误重启）
#
# 采用"看护"而不是"常驻重启"的理由：相机被拔是**正常状态**，不该被当成故障；
# 只有"设备在但不出图"才是真故障。
#
# 用法（宿主机）：
#   bash astra_watchdog.sh                 # 前台看日志
#   bash astra_watchdog.sh --once          # 只检查一轮（便于脚本调用）
# 环境变量：
#   ROS_CONTAINER   默认 ros2_arm_container
#   ASTRA_USB_ID    默认 2bc5:0402
#   BRIDGE_META     默认 http://127.0.0.1:8123/api/meta
#   FROZEN_LIMIT_S  画面冻结多少秒后重启节点，默认 90
#   CHECK_INTERVAL  检查间隔秒，默认 20
#   LOG             默认 /tmp/astra_watchdog.log
# ============================================================
set -uo pipefail

ROS_CONTAINER="${ROS_CONTAINER:-ros2_arm_container}"
ASTRA_USB_ID="${ASTRA_USB_ID:-2bc5:0402}"
BRIDGE_META="${BRIDGE_META:-http://127.0.0.1:8123/api/meta}"
FROZEN_LIMIT_S="${FROZEN_LIMIT_S:-90}"
CHECK_INTERVAL="${CHECK_INTERVAL:-20}"
LOG="${LOG:-/tmp/astra_watchdog.log}"
ONCE=0
[ "${1:-}" = "--once" ] && ONCE=1

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

astra_on_usb()   { lsusb 2>/dev/null | grep -q "$ASTRA_USB_ID"; }
container_up()   { docker ps --format '{{.Names}}' 2>/dev/null | grep -qx "$ROS_CONTAINER"; }
node_running()   { docker exec "$ROS_CONTAINER" bash -lc 'pgrep -f "[a]stra_camera_node" >/dev/null' 2>/dev/null; }
start_node()     {
  # 脚本可能不在容器里（容器重建后）→ 先从宿主机拷进去
  docker exec "$ROS_CONTAINER" bash -lc '[ -f /start_astra.sh ]' 2>/dev/null || {
    local src
    for src in "$HOME/sort-jetson-deploy-"*/deploy/jetson/start_astra.sh; do
      [ -f "$src" ] && { docker cp "$src" "$ROS_CONTAINER:/start_astra.sh" 2>/dev/null && break; }
    done
  }
  docker exec -d "$ROS_CONTAINER" bash -lc 'nohup bash /start_astra.sh > /tmp/astra.log 2>&1 &'
  log "已拉起 Astra 相机节点（日志：容器内 /tmp/astra.log）"
}

# 画面冻结秒数（桥接容器按"内容指纹"统计）；取不到返回空
frozen_s() {
  curl -fsS --max-time 4 "$BRIDGE_META" 2>/dev/null \
    | sed -n 's/.*"color_frozen_s"[[:space:]]*:[[:space:]]*\([0-9.]*\).*/\1/p'
}

check_once() {
  if ! astra_on_usb; then
    [ "$node_running" ] && log "  USB 上无 Astra（已拔）—— 相机节点仍在跑？不动它"
    return 0
  fi
  if ! container_up; then
    log "ROS 容器未运行 → 启动 $ROS_CONTAINER"
    docker start "$ROS_CONTAINER" >/dev/null 2>&1 || { log "  ✗ 启动失败"; return 1; }
    sleep 6
  fi
  if ! node_running; then
    log "Astra 在 USB 上但相机节点未运行 → 启动节点"
    start_node
    return 0
  fi
  local fz; fz="$(frozen_s)"
  if [ -n "$fz" ]; then
    # 用整数比较（脚本里避免浮点依赖 bc）
    local fzi="${fz%%.*}"
    if [ "${fzi:-0}" -ge "$FROZEN_LIMIT_S" ]; then
      log "⚠ 画面已冻结 ${fzi}s（≥${FROZEN_LIMIT_S}s）→ 重启相机节点"
      docker exec "$ROS_CONTAINER" bash -lc 'pkill -f "[a]stra_camera_node"' 2>/dev/null
      sleep 3
      start_node
    fi
  fi
  return 0
}

log "Astra 看护启动：容器=$ROS_CONTAINER 设备=$ASTRA_USB_ID 冻结阈值=${FROZEN_LIMIT_S}s 间隔=${CHECK_INTERVAL}s"
while true; do
  check_once || true
  [ "$ONCE" = "1" ] && break
  sleep "$CHECK_INTERVAL"
done
