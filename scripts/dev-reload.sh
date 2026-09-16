#!/usr/bin/env bash
# ============================================================
# PC 前端调试容器：改完代码后秒级重建并重启
# ------------------------------------------------------------
# 用法：
#   bash scripts/dev-reload.sh            # 重建镜像 + 重启容器
#   bash scripts/dev-reload.sh --logs     # 重建后跟踪日志
#
# 为什么需要它：PC 容器的源码随镜像构建（不依赖 Docker Desktop 绑定挂载，
# 该挂载在 Desktop 重启后常失效）。npm 依赖层有缓存，重建通常 5~10 秒。
# ============================================================
set -euo pipefail
cd "$(dirname "$0")/.."

echo "==> 重建并重启 PC 前端容器（依赖层走缓存）…"
docker compose up -d --build

sleep 5
echo "==> 自检"
docker compose ps
curl -s -o /dev/null -w "  页面 http://localhost:5173  HTTP %{http_code}\n" http://localhost:5173/ || true
curl -s http://localhost:5173/health | head -c 200 || true
echo

if [ "${1:-}" = "--logs" ]; then
  echo "==> 跟踪日志（Ctrl+C 退出，容器继续运行）"
  docker compose logs -f
fi
