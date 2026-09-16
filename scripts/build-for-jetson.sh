#!/usr/bin/env bash
# ============================================================
# PC 端打包 Jetson 镜像/产物（两个镜像：网关 + 视觉）
# ------------------------------------------------------------
# 用法：
#   bash scripts/build-for-jetson.sh                 # 前端静态产物 + 两个 arm64 镜像 + tar
#   bash scripts/build-for-jetson.sh --no-image      # 只构建前端静态产物 build/
#   bash scripts/build-for-jetson.sh --gateway-only  # 只构建网关镜像
#   bash scripts/build-for-jetson.sh --platform linux/arm64
#
# 产出：
#   build/                            前端静态 SPA（Jetson 网关容器托管）
#   dist/sort-jetson-images.tar       两个镜像的离线包（docker load 即可用）
#
# 若 PC 无法访问 nvcr.io（NVIDIA 镜像源），请在 Jetson 上执行：
#   docker compose -f docker-compose.jetson.yml up -d --build
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

PLATFORM="linux/arm64"
WITH_IMAGE=1
TARGET="both"          # both | gateway | vision

while [ $# -gt 0 ]; do
  case "$1" in
    --no-image)     WITH_IMAGE=0 ;;
    --gateway-only) TARGET="gateway" ;;
    --vision-only)  TARGET="vision" ;;
    --platform)     PLATFORM="${2:?--platform 需要参数}"; shift ;;
    -h|--help)      sed -n '2,20p' "$0"; exit 0 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
  shift
done

echo "========================================="
echo "  智能分拣装置 · Jetson 打包"
echo "  平台: $PLATFORM | 目标: $TARGET | 构建镜像: $WITH_IMAGE"
echo "========================================="

# ---------- [0/3] 预检 ----------
echo "[0/3] 预检…"
[ -f package.json ] || { echo "❌ 请在 sort-web 项目根目录执行"; exit 1; }
[ -f models/yolov8n.onnx ] || echo "  ⚠ models/yolov8n.onnx 不存在（DETECT_ENGINE=onnx 时必需，classic 引擎可忽略）"
if [ "$WITH_IMAGE" = "1" ]; then
  command -v docker >/dev/null || { echo "❌ 未找到 docker"; exit 1; }
  docker buildx version >/dev/null 2>&1 || { echo "❌ 未启用 docker buildx（Docker Desktop 默认启用）"; exit 1; }
fi
echo "  ✓ 预检通过"

# ---------- [1/3] 前端静态产物 ----------
echo "[1/3] 构建前端静态产物 (ADAPTER=static)…"
if [ -d node_modules ]; then
  npm run build:static
else
  echo "  node_modules 缺失，先 npm install…"
  npm install --no-audit --no-fund && npm run build:static
fi
[ -f build/index.html ] || { echo "❌ 前端构建失败：缺少 build/index.html"; exit 1; }
echo "  ✓ build/ 就绪（$(du -sh build | cut -f1)）"

if [ "$WITH_IMAGE" = "0" ]; then
  echo
  echo "下一步（在 Jetson 上）："
  echo "  bash scripts/deploy-to-jetson.sh <user>@<jetson-ip>   # 一键同步并启动"
  echo "  或手动：docker compose -f docker-compose.jetson.yml up -d --build"
  exit 0
fi

# ---------- [2/3] 构建 arm64 镜像 ----------
mkdir -p dist
IMAGES=()
if [ "$TARGET" != "vision" ]; then
  echo "[2/3] 构建网关镜像 sort-gateway:jetson ($PLATFORM)…"
  docker buildx build --platform "$PLATFORM" \
    -f deploy/jetson/Dockerfile -t sort-gateway:jetson --load .
  IMAGES+=("sort-gateway:jetson")
fi
if [ "$TARGET" != "gateway" ]; then
  echo "[2/3] 构建视觉镜像 sort-vision:jetson ($PLATFORM)…"
  docker buildx build --platform "$PLATFORM" \
    -f deploy/jetson/Dockerfile.vision -t sort-vision:jetson --load .
  IMAGES+=("sort-vision:jetson")
fi

# ---------- [3/3] 导出离线包 ----------
TAR="dist/sort-jetson-images.tar"
echo "[3/3] 导出镜像包 → $TAR"
docker save "${IMAGES[@]}" -o "$TAR"
echo "  ✓ $(du -sh "$TAR" | cut -f1)"

cat <<EOF

=========================================
 完成
=========================================
镜像: ${IMAGES[*]}
离线包: $TAR

在 Jetson 上部署（二选一）：
  A) 一键脚本（推荐，自动 rsync + 启动 + 自检）
     bash scripts/deploy-to-jetson.sh nvidia@<jetson-ip> --use-tar
  B) 手动
     scp $TAR nvidia@<jetson-ip>:~/
     # Jetson 上：
     docker load -i ~/sort-jetson-images.tar
     docker compose -f docker-compose.jetson.yml up -d

显示屏开机自启（Jetson 上执行一次）：
     bash deploy/jetson/setup-autostart.sh
EOF
