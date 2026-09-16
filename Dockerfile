# ============================================================
# PC 端 · Node 22 + Svelte 5 前端调试容器
# ------------------------------------------------------------
# 用途：仅用于在 PC 上调试前端 UI（Vite 开发服务器 + HMR 热更新）
#       SvelteKit 的服务端路由（/api/*）在 dev 模式下同样由 Node 提供，
#       因此界面上的分拣流程、任务、显示闩锁都能直接点通。
#
#   docker compose up -d --build      → http://localhost:5173
#   docker compose logs -f            → 实时日志（含 Vite HMR）
#
# 说明：源码通过 compose 绑定挂载进容器，宿主机改代码 → 容器内即时热更新，
#       不需要重新构建镜像。
# ============================================================
FROM node:22-alpine

WORKDIR /app

# ---- 依赖层（单独缓存，改源码不会重装依赖） ----
COPY package.json package-lock.json ./
RUN npm ci --no-audit --no-fund

# ---- 源码（调试时由 volume 覆盖挂载） ----
COPY . .

ENV NODE_ENV=development
ENV HOST=0.0.0.0
ENV PORT=5173
EXPOSE 5173

CMD ["npm", "run", "dev"]
