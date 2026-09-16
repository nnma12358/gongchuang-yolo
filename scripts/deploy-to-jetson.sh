#!/usr/bin/env bash
# ============================================================
# 一键部署到 Jetson Nano（PC 上执行）
# ------------------------------------------------------------
# 用法：
#   bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50
#   bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50 --use-tar     # 用离线镜像包
#   bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50 --no-build    # 不重新构建前端
#   bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50 --dir ~/sort-web
#
# 做什么：
#   1) 本地构建前端静态产物（除非 --no-build）
#   2) rsync 同步必要文件到 Jetson（build/ server/ deploy/ scripts/ models/ 与编排文件）
#   3) 远端 docker compose up -d --build（或 docker load 离线包后启动）
#   4) 远端健康自检：网关 :80 / 视觉 :8100 / 自动分拣状态
#
# 注意：远端 .env 不会被覆盖（保留现场配置）；data/ 卷数据保留。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

TARGET="${1:-}"
[ -z "$TARGET" ] && { sed -n '2,16p' "$0"; exit 1; }
shift || true

REMOTE_DIR="~/sort-web"
DO_BUILD=1
USE_TAR=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-build) DO_BUILD=0 ;;
    --use-tar)  USE_TAR=1 ;;
    --dir)      REMOTE_DIR="${2:?--dir 需要参数}"; shift ;;
    -h|--help)  sed -n '2,16p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
  shift
done

command -v rsync >/dev/null || { echo "❌ 未找到 rsync"; exit 1; }
command -v ssh  >/dev/null || { echo "❌ 未找到 ssh"; exit 1; }

echo "========================================="
echo "  部署到 $TARGET:$REMOTE_DIR"
echo "  构建前端: $DO_BUILD | 使用离线镜像: $USE_TAR"
echo "========================================="

# ---------- [1/4] 前端静态产物 ----------
if [ "$DO_BUILD" = "1" ]; then
  echo "[1/4] 构建前端静态产物…"
  [ -d node_modules ] || npm install --no-audit --no-fund
  npm run build:static
  [ -f build/index.html ] || { echo "❌ 前端构建失败"; exit 1; }
else
  [ -f build/index.html ] || { echo "❌ build/index.html 不存在，去掉 --no-build 或先构建"; exit 1; }
  echo "[1/4] 跳过构建（沿用现有 build/）"
fi

# ---------- [2/4] 同步文件 ----------
echo "[2/4] 同步文件到 Jetson…"
ssh "$TARGET" "mkdir -p $REMOTE_DIR"
rsync -az --delete \
  --exclude '.env' --exclude 'data/' --exclude 'dist/' --exclude 'node_modules/' \
  build/ "$TARGET:$REMOTE_DIR/build/"
rsync -az --delete server/ "$TARGET:$REMOTE_DIR/server/"
rsync -az --delete scripts/ "$TARGET:$REMOTE_DIR/scripts/"
rsync -az --delete deploy/ "$TARGET:$REMOTE_DIR/deploy/"
rsync -az models/ "$TARGET:$REMOTE_DIR/models/"
rsync -az docker-compose.jetson.yml docker-compose.jetson-camera.yml README.md "$TARGET:$REMOTE_DIR/"
echo "  ✓ 同步完成（远端 .env 与 data/ 未改动）"

# ---------- [3/4] 启动容器 ----------
echo "[3/4] 启动容器…"
if [ "$USE_TAR" = "1" ]; then
  TAR="dist/sort-jetson-images.tar"
  [ -f "$TAR" ] || { echo "❌ 缺少 $TAR，请先执行 bash scripts/build-for-jetson.sh"; exit 1; }
  echo "  上传镜像包（$(du -sh "$TAR" | cut -f1)）…"
  rsync -az --progress "$TAR" "$TARGET:$REMOTE_DIR/"
  ssh "$TARGET" "cd $REMOTE_DIR && docker load -i sort-jetson-images.tar && \
                 docker compose -f docker-compose.jetson.yml up -d"
else
  ssh "$TARGET" "cd $REMOTE_DIR && docker compose -f docker-compose.jetson.yml up -d --build"
fi

# ---------- [4/4] 健康自检 ----------
echo "[4/4] 健康自检…"
sleep 8
ssh "$TARGET" "REMOTE_DIR='$REMOTE_DIR' bash -s" <<'REMOTE'
set -e
DIR="${REMOTE_DIR/#\~/$HOME}"
echo "  --- 容器状态 ---"
docker compose -f "$DIR/docker-compose.jetson.yml" ps 2>/dev/null || docker ps --format '  {{.Names}} | {{.Status}}'
echo "  --- 网关 ---"
curl -fsS http://localhost/health || echo "  ✘ 网关 /health 未响应"
echo
echo "  --- 视觉容器 ---"
curl -fsS http://localhost:8100/health || echo "  ✘ 视觉容器 /health 未响应"
echo
echo "  --- 自动分拣状态 ---"
curl -fsS http://localhost/api/auto/status || true
echo
REMOTE

cat <<EOF

=========================================
 部署完成
=========================================
显示屏（Jetson HDMI）:  http://localhost
PC 浏览器查看（可选）:  在 sort-web/.env 填
    GATEWAY_URL=http://${TARGET#*@}
    VISION_URL=http://${TARGET#*@}:8100
  然后 `docker compose up -d` 重启 PC 调试容器

开机自启（Jetson 上执行一次）:
    bash $REMOTE_DIR/deploy/jetson/setup-autostart.sh
EOF
