#!/usr/bin/env bash
# ============================================================
# 分拣显示屏 kiosk —— "网关就绪后自动全屏打开前端"
# ------------------------------------------------------------
# 设计要点（针对现场实际踩过的问题）：
#   1) **先等网关就绪再开浏览器**：网关是容器，开机后要几十秒才起来。
#      旧实现只等 60s 且超时也照开 → 浏览器停在"无法访问"，之后一直不会自己恢复。
#      这里改成：等 /health 返回 200（默认最长 300s，可调），期间每 2s 重试。
#   2) **崩了/被关了要自己回来**：浏览器退出后自动重启（可配间隔），
#      现场没人盯着屏幕，显示不能一崩就黑。
#   3) **网关重启后页面自动恢复**：持续探测 /health，一旦"断过又恢复"就强制刷新
#      （有 xdotool 就发 F5，没有就重启浏览器）。
#   4) **等 X 会话就绪**：从 systemd 拉起时 X 可能还没起，先等 DISPLAY 可用。
#   5) 关息屏/DPMS、屏蔽首启弹窗、关闭翻译条 —— 比赛期间屏幕必须干净常亮。
#
# 用法：
#   bash deploy/jetson/kiosk.sh                 # 前台运行（给桌面自启/systemd 用）
#   bash deploy/jetson/kiosk.sh --once          # 只开一次，退出后不重启
#   KIOSK_URL=http://localhost bash deploy/jetson/kiosk.sh
#   bash deploy/jetson/kiosk.sh --wait-only     # 只等网关就绪（供脚本串联）
#   bash deploy/jetson/kiosk.sh --stop          # 停掉显示（并暂停自动重启）
#
# 显示模式（KIOSK_MODE）：
#   max        默认：**最大化窗口**，GNOME 顶栏保留 → 随时点右上角查 WiFi/音量/设置
#   kiosk      真全屏无边框（比赛展示用），需要 F11 或 --stop 才能退出
#   两种模式都可用 F11 切换全屏/窗口，方便"平时能操作、比赛时干净"
# 环境变量：
#   KIOSK_URL          默认 http://localhost
#   KIOSK_WAIT_S       等网关就绪最长秒数，默认 300
#   KIOSK_RESTART_S    浏览器退出后重启间隔，默认 5
#   KIOSK_LOG          日志文件，默认 /tmp/kiosk.log
# ============================================================
set -uo pipefail

URL="${KIOSK_URL:-http://localhost}"
MODE="${KIOSK_MODE:-max}"          # max(默认,保留顶栏) | kiosk(真全屏)
PAUSE_FLAG="${KIOSK_PAUSE:-/tmp/kiosk.pause}"
WAIT_S="${KIOSK_WAIT_S:-300}"
RESTART_S="${KIOSK_RESTART_S:-5}"
LOG="${KIOSK_LOG:-/tmp/kiosk.log}"
ONCE=0
WAIT_ONLY=0
STOP=0
for a in "$@"; do
  case "$a" in
    --once) ONCE=1 ;;
    --wait-only) WAIT_ONLY=1 ;;
    --stop) STOP=1 ;;
    -h|--help) sed -n '2,30p' "$0"; exit 0 ;;
    http*) URL="$a" ;;
    *) echo "未知参数: $a"; exit 1 ;;
  esac
done

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

# ---------- 0. 等 X 会话（从 systemd 拉起时 X 可能还没起来）----------
wait_x() {
  local i=0
  while [ $i -lt 60 ]; do
    [ -n "${DISPLAY:-}" ] && [ -S "/tmp/.X11-unix/X${DISPLAY#:}" ] && return 0
    # 未设置 DISPLAY 时猜 :0
    if [ -z "${DISPLAY:-}" ] && [ -S /tmp/.X11-unix/X0 ]; then
      export DISPLAY=:0
      return 0
    fi
    i=$((i + 1)); sleep 1
  done
  log "⚠ 等 X 会话超时（DISPLAY=${DISPLAY:-未设置}）"
  return 1
}

# ---------- 1. 等网关就绪 ----------
gateway_ready() { curl -fsS -o /dev/null --max-time 3 "${URL}/health" 2>/dev/null; }

wait_gateway() {
  log "等待网关就绪：${URL}/health（最长 ${WAIT_S}s）"
  local t=0
  while [ $t -lt "$WAIT_S" ]; do
    if gateway_ready; then
      log "✓ 网关已就绪（等待 ${t}s）"
      return 0
    fi
    sleep 2; t=$((t + 2))
  done
  log "✗ 等网关超时（${WAIT_S}s）—— 仍会打开浏览器，并在网关恢复后自动刷新"
  return 1
}

if [ "$STOP" = "1" ]; then
  touch "$PAUSE_FLAG"                       # 暂停自动重启，否则马上又被拉起来
  pkill -f chromium-browser 2>/dev/null || true
  pkill -f "kiosk.sh" 2>/dev/null || true
  DISPLAY="${DISPLAY:-:0}" xset dpms force off 2>/dev/null || true
  echo "已停止显示，并写入暂停标记 $PAUSE_FLAG（想恢复：rm $PAUSE_FLAG 后重新运行本脚本）"
  exit 0
fi

if [ "$WAIT_ONLY" = "1" ]; then wait_gateway; exit $?; fi

# ---------- 2. 找浏览器 ----------
find_browser() {
  for b in chromium-browser chromium google-chrome google-chrome-stable; do
    command -v "$b" >/dev/null 2>&1 && { echo "$b"; return 0; }
  done
  return 1
}
BROWSER="$(find_browser)" || { log "✗ 未找到浏览器（chromium-browser/chromium/chrome）"; exit 1; }
log "浏览器: $BROWSER"

XTOOL=""
command -v xdotool >/dev/null 2>&1 && XTOOL="xdotool"

# ---------- 3. 屏幕常亮 ----------
screen_on() {
  xset s off 2>/dev/null || true
  xset -dpms 2>/dev/null || true
  xset s noblank 2>/dev/null || true
}

# ---------- 4. 主循环 ----------
rm -f "$PAUSE_FLAG"          # 手动/自启唤起时清掉暂停标记
wait_x && screen_on
wait_gateway

log "启动 kiosk：$URL"
while true; do
  screen_on
  # --disable-gpu：Jetson Nano 上 chromium 反复报 "Error: Can't initialize nvrm channel"
  # （GPU 通道不可用）→ 让它直接走软件渲染，日志干净、kiosk 更稳
  MODE_ARGS=(--start-maximized)          # max：最大化窗口，GNOME 顶栏保留（可查 WiFi）
  [ "$MODE" = "kiosk" ] && MODE_ARGS=(--kiosk)
  "$BROWSER" \
      "${MODE_ARGS[@]}" \
      --disable-gpu \
      --noerrdialogs \
      --disable-infobars \
      --disable-session-crashed-bubble \
      --disable-features=TranslateUI,Translate \
      --password-store=basic \
      --incognito \
      --check-for-update-interval=31536000 \
      --overscroll-history-navigation=0 \
      "$URL" >>"$LOG" 2>&1 &
  BPID=$!
  log "浏览器 pid=$BPID"

  # 浏览器存活期间：持续探测网关；"断过又恢复" → 强制刷新页面
  was_down=0
  while kill -0 "$BPID" 2>/dev/null; do
    sleep 5
    if gateway_ready; then
      if [ "$was_down" = "1" ]; then
        log "网关已恢复 → 刷新页面"
        if [ -n "$XTOOL" ]; then
          DISPLAY="${DISPLAY:-:0}" "$XTOOL" key --clearmodifiers F5 2>/dev/null || kill "$BPID" 2>/dev/null
        else
          kill "$BPID" 2>/dev/null     # 没有 xdotool 就重启浏览器达到等效刷新
        fi
        was_down=0
      fi
    else
      [ "$was_down" = "0" ] && log "⚠ 网关暂时不可达（等待恢复）"
      was_down=1
    fi
  done
  wait "$BPID" 2>/dev/null
  log "浏览器已退出（code=$?）"
  [ "$ONCE" = "1" ] && break
  if [ -e "$PAUSE_FLAG" ]; then
    log "检测到暂停标记 $PAUSE_FLAG → 不再自动重启（删除该文件即可恢复）"
    break
  fi
  log "  ${RESTART_S}s 后重新拉起…"
  sleep "$RESTART_S"
  wait_gateway
done
