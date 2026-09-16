#!/usr/bin/env bash
# ============================================================
# Jetson Nano 开机自启（在 Jetson 上执行一次）
# ------------------------------------------------------------
# 做两件事：
#   1) systemd 单元 sort-web.service —— 开机自动 docker compose up -d（并等待服务就绪）
#   2) 高亮显示屏 kiosk 自启 —— 登录后自动全屏打开 http://localhost，并关闭息屏
#
# 用法：
#   bash deploy/jetson/setup-autostart.sh              # 安装
#   bash deploy/jetson/setup-autostart.sh --uninstall  # 卸载
#   bash deploy/jetson/setup-autostart.sh --no-kiosk   # 只装服务自启，不装浏览器全屏
#
# 说明：脚本会自动探测 docker / docker compose(v2) / docker-compose(v1)；
#       兼容 Jetson 上较旧的 Docker 版本。
# ============================================================
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.jetson.yml"
SERVICE_NAME="sort-web"
UNIT_PATH="/etc/systemd/system/${SERVICE_NAME}.service"
KIOSK_DIR="$HOME/.config/autostart"
KIOSK_FILE="$KIOSK_DIR/sort-display.desktop"
KIOSK_SH="$PROJECT_DIR/deploy/jetson/kiosk.sh"
KIOSK_URL="${KIOSK_URL:-http://localhost}"

INSTALL=1
WITH_KIOSK=1
for arg in "$@"; do
  case "$arg" in
    --uninstall) INSTALL=0 ;;
    --no-kiosk)  WITH_KIOSK=0 ;;
    -h|--help)   sed -n '2,18p' "$0"; exit 0 ;;
    *) echo "未知参数: $arg"; exit 1 ;;
  esac
done

# ---------- 探测 docker / compose ----------
if ! command -v docker >/dev/null 2>&1; then
  echo "❌ 未找到 docker，请先安装 Docker（JetPack 自带 docker 服务）"; exit 1
fi
if docker compose version >/dev/null 2>&1; then
  COMPOSE_CMD="docker compose"
elif command -v docker-compose >/dev/null 2>&1; then
  COMPOSE_CMD="docker-compose"
else
  echo "❌ 未找到 docker compose / docker-compose"; exit 1
fi
echo "docker: $(command -v docker) | compose: $COMPOSE_CMD"

# ---------- 卸载 ----------
if [ "$INSTALL" = "0" ]; then
  echo "== 卸载 =="
  sudo systemctl disable --now "${SERVICE_NAME}.service" 2>/dev/null || true
  sudo rm -f "$UNIT_PATH"
  sudo systemctl daemon-reload
  rm -f "$KIOSK_FILE"
  echo "✓ 已卸载（容器仍在运行；如需停止：$COMPOSE_CMD -f $COMPOSE_FILE down）"
  exit 0
fi

[ -f "$COMPOSE_FILE" ] || { echo "❌ 找不到 $COMPOSE_FILE（请确认项目路径）"; exit 1; }

# ---------- [1/2] systemd 服务 ----------
echo "== [1/2] 安装 systemd 单元 =="
sudo tee "$UNIT_PATH" >/dev/null <<EOF
[Unit]
Description=智能分拣装置（视觉容器 + 网关容器）
Requires=docker.service
After=docker.service network-online.target
Wants=network-online.target

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$PROJECT_DIR
# 开机拉起两容器；若镜像尚未构建，会先 build
ExecStartPre=-$COMPOSE_CMD -f $COMPOSE_FILE pull --ignore-pull-failures
ExecStart=$COMPOSE_CMD -f $COMPOSE_FILE up -d --build
ExecStop=$COMPOSE_CMD -f $COMPOSE_FILE down
TimeoutStartSec=0
Restart=on-failure
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable "${SERVICE_NAME}.service"
echo "  ✓ 已启用（首启日志：journalctl -u ${SERVICE_NAME} -f）"

# ---------- [2/2] 显示屏 kiosk ----------
if [ "$WITH_KIOSK" = "1" ]; then
  echo "== [2/2] 安装显示屏 kiosk 自启 =="
  cat > "$KIOSK_SH" <<'EOF'
#!/usr/bin/env bash
# 高亮显示屏 kiosk：全屏打开显示屏页面，并关闭屏幕保护/息屏
URL="${1:-http://localhost}"
# 关闭息屏与 DPMS（比赛期间屏幕必须常亮）
xset s off 2>/dev/null || true
xset -dpms 2>/dev/null || true
xset s noblank 2>/dev/null || true
# 等待网关就绪（最多 60s）
for i in $(seq 1 60); do
  curl -fsS -o /dev/null "${URL}/health" && break
  sleep 1
done
BROWSER=""
for b in chromium-browser chromium google-chrome; do
  command -v "$b" >/dev/null 2>&1 && { BROWSER="$b"; break; }
done
[ -z "$BROWSER" ] && { echo "未找到浏览器（chromium-browser/chromium）"; exit 1; }
exec "$BROWSER" --kiosk --noerrdialogs --disable-infobars --disable-session-crashed-bubble \
     --incognito --check-for-update-interval=31536000 "$URL"
EOF
  chmod +x "$KIOSK_SH"

  mkdir -p "$KIOSK_DIR"
  cat > "$KIOSK_FILE" <<EOF
[Desktop Entry]
Type=Application
Name=分拣显示屏 (kiosk)
Comment=开机全屏显示 http://localhost
Exec=$KIOSK_SH $KIOSK_URL
X-GNOME-Autostart-enabled=true
Terminal=false
EOF
  echo "  ✓ 已写入 $KIOSK_FILE"
  echo "  ✓ kiosk 脚本: $KIOSK_SH（URL 可用 KIOSK_URL 覆盖）"
fi

# ---------- 立即启动一次 ----------
echo "== 立即启动 =="
sudo systemctl start "${SERVICE_NAME}.service" || true
sleep 6
"$COMPOSE_CMD" -f "$COMPOSE_FILE" ps || docker ps --format '{{.Names}} | {{.Status}}'
echo
echo "自检："
curl -fsS http://localhost/health && echo
curl -fsS http://localhost:8100/health && echo
cat <<EOF

=========================================
 开机自启已安装
=========================================
 服务:   systemctl status ${SERVICE_NAME}
 日志:   journalctl -u ${SERVICE_NAME} -f
 显示屏: 登录桌面后自动全屏打开 $KIOSK_URL
 卸载:   bash deploy/jetson/setup-autostart.sh --uninstall
EOF
