# 智能分拣装置 · 分拣信息显示屏（sort-web）

> 赛项：**智能分拣赛项 / 现场初赛**
> 技术栈：Svelte 5 (SvelteKit) · Node 22 · Docker

## 1. 赛项要求 → 实现对照

| 赛项要求 | 实现 |
|---|---|
| 采用 AI 技术识别货物（颜色、形状、文字、图像、二维码） | 机器人端识别后上报 `POST /api/sort/event`（字段：class/shape/color/marks/text/qr）；网关内置 OpenCV 引擎（颜色 HSV + 形状轮廓 + 污渍缺陷暗斑 + 二维码 `QRCodeDetector`）见 `server/gateway.py` |
| 高亮显示屏显示投放顺序、货物名称、图片、数量 | 「当前分拣信息」大卡（投放顺序 / 货物名称 / 图片 / 分拣成功总数量）＋「投放顺序表」；图片支持 `svg/png/jpg/webp/gif/bmp` 多格式播放 |
| **六个单独储物盒（标注 1–6 序号）** | 界面「储物盒（1–6）」面板：每盒显示所指派的“颜色+形状”、已放件数 / 4、缩略图与满盒状态；网关 `assignBin()` 按 1→6 顺序占用空盒 |
| **每盒最多 4 件货物** | 网关硬校验：满 4 件后再上报该组合返回 **409**（界面提示“已满，按赛项最多 4 件”） |
| **形状相同且颜色相同的货物 → 同一储物盒** | 储物盒以 `形状\|颜色` 为键分配；界面显示每盒归属的组合，评估脚本 `eval_pipeline.py` 校验一致性 = 1.00 |
| **一次只能分拣一件；分拣信息显示后再分拣下一件，否则不计分** | 显示闩锁：`POST /api/sort/event` 写入后锁定 `DISPLAY_HOLD_SECONDS` 秒，期间再上报返回 **409**；界面顶部倒计时 + 「确认显示完成」放行 |
| **按统一指令启动，计时开始；比赛结束前不得接触装置** | 「开始比赛」→ 计时启动（顶栏与状态条实时显示 mm:ss.d）；比赛进行中人工干预被拒（模拟/登记/复位需 `?debug=1` 调试模式） |
| **货物掉落在装置范围内/外 → 本轮比赛结束** | 「掉落（装置内/外）」按钮 → `POST /api/round/fault` → 停止计时、记录结束原因、冻结本轮 |
| 货物 ≤40mm、表面附有污渍/缺陷图形、文字、二维码 | 当前分拣信息卡展示 `颜色/形状/储物盒/表面(污渍·缺陷)` 以及文字、二维码内容；表格「表面信息」列汇总 |

## 2. 容器架构（Jetson 端两容器 · 通过端口通信）

```
                    Jetson Nano（现场全部功能）
   ┌───────────────────────────────┐        ┌──────────────────────────────────────┐
   │ 视觉容器 sort-vision  :8100    │  HTTP  │ 网关容器 sort-gateway  :80            │
   │ · 独占相机（USB/RTSP/合成）    │ ◀────▶ │ · 前端 SPA（本容器内托管）             │
   │ · 连续识别：颜色/形状/污渍/二维码│        │ · 分拣规则：六盒 1–6、每盒 4 件、同形同色同盒 │
   │ · 标记输出：marks JSON / 带框帧 │        │ · 自动分拣循环 → 显示屏 → 下一位        │
   └───────────────────────────────┘        └───────────────┬──────────────────────┘
                                                            │ ROBOT_URL（可选）
                                                   ┌────────▼─────────┐
                                                   │ ROS 桥接 :8120    │ → R550A 机械臂
                                                   └──────────────────┘

   PC（仅调试前端 UI）: node22 容器 → .env 里填 VISION_URL / GATEWAY_URL 指向 Jetson
```

| 位置 | 容器 | 镜像 | 用途 |
|---|---|---|---|
| **Jetson** | `sort-vision` | `nvcr.io/nvidia/l4t-base:r32.7.1` | 相机 + 连续识别 + **标记输出**（:8100） |
| **Jetson** | `sort-gateway` | 同上 | 前端 SPA + 分拣规则 + **自动分拣循环**（:80） |
| **Jetson（可选）** | `ros-bridge` | `ros:melodic-ros-base` | 把抓取指令转成 ROS 调用（:8120，`--profile robot`） |
| **PC** | `sort-svelte-dev` | `node:22-alpine` | **纯前端壳**：只托管 UI，并把 `/api/**` 代理到 Jetson；**不含任何分拣逻辑与模拟数据** |

> ⚠ **以实际服务为准**：PC 容器不提供任何服务，页面数据全部来自 Jetson。
> 未配置/不可达时，页面顶部显示「未连接实际服务」，各面板显示「无数据」，自动分拣按钮禁用 ——
> 不会出现与真实部署不符的状态。配置：在 `sort-web/.env` 填
> `GATEWAY_URL=http://<jetson-ip>`（可选 `VISION_URL=http://<jetson-ip>:8100`）后 `docker compose up -d`。


### 视觉三层（识别拆成独立容器）

```
Astra/相机 ──► sort-vision(:8100)  编排层：抓帧、2.5D 高度、二维码/文字、标记输出
                 │  POST /detect_array（base64 帧）
                 ├──► sort-yolo(:8101)  检测层：YOLOv8 ONNX → 货物框
                 └──► sort-cnn (:8102)  属性层：小 CNN ONNX → 颜色/形状/污渍缺陷
                 ▼
              marks（框 + 中文货物名 + 属性 + 相机系坐标/高度）──► sort-gateway(:80)
```

- `DETECT_ENGINE=yolo` 时走三层链路；yolo 容器不可达则**自动回退**经典 CV 引擎（颜色+轮廓+二维码），保证现场不致停摆
- `DETECT_ENGINE=classic` 时只用经典引擎（无模型依赖，便于快速联调）
- 模型通过 `./models:/app/models:ro` 挂载，换模型只需替换文件重启容器；模型说明见 `models/README.md`

### 自动分拣闭环（对应赛项流程）

```
视觉容器识别 ──► 网关连续 N 帧确认 ──► 同形同色同盒入库(每盒≤4) ──► 显示屏保持 3s
      ▲                                                                    │
      └──────── 托盘清空(机械臂取走) ←── 下发 ROBOT_URL /execute ←──────────┘
```

- 循环运行在网关容器内（与页面无关，刷新/关页面不影响），`POST /api/auto/start` 启动
- 赛项闸门全部生效：一次一件（多件时跳过）、显示未完成不登记、满盒/六盒用尽拒绝
- `dry_run`（默认开）只记录不驱动机械臂，便于无硬件联调；现场置 `AUTO_DRY_RUN=0`
- **标记输出**：`GET /api/marks/latest`（最近一帧标记）、`GET /api/marks/log`（JSONL 历史）、
  `GET /api/frame.jpg?draw=1`（带框画面，PC 页面「视觉画面(Jetson)」即显示这一路）

## 3. PC 端：调试前端 UI（Node 22 容器）

```bash
docker compose up -d --build      # 构建并启动
# 浏览器打开  http://localhost:5173
docker compose logs -f            # Vite 日志 / HMR 输出
docker compose down               # 停止
```

- `src/` 目录已挂载进容器：**改 `.svelte` 文件浏览器即时热更新**
- 改了 `vite.config.js` / `svelte.config.js` / `package.json` 需 `up -d --build`
- 生产产物自检（可选）：`docker compose -f docker-compose.prod.yml up -d --build` → http://localhost:8090

## 4. Jetson Nano：两容器部署（现场）

```bash
# ① 一键部署（推荐）：本地构建前端 → rsync 到 Jetson → 构建并启动 → 自检
bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50
#    离线镜像方式（Jetson 无外网时）：
#    bash scripts/build-for-jetson.sh              # 出 dist/sort-jetson-images.tar
#    bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50 --use-tar

# ② 或手动：PC 构建前端静态产物 → 同步 → Jetson 上启动
bash scripts/build-for-jetson.sh --no-image
docker compose -f docker-compose.jetson.yml up -d --build
curl http://localhost/health                     # 网关
curl http://localhost:8100/health                # 视觉容器（相机/引擎/帧率）

# ③ 显示放开机自启 + kiosk 全屏（Jetson 上执行一次）
bash deploy/jetson/setup-autostart.sh
```

相机接入：

```bash
# 默认 privileged: true 已可访问 /dev/video*；宿主机禁用 privileged 时显式映射设备：
docker compose -f docker-compose.jetson.yml -f docker-compose.jetson-camera.yml up -d
# 无相机：保持默认（CAMERA_SOURCE=auto 自动回落合成场景，便于联调）
```

现场开关（`docker-compose.jetson.yml` 环境变量或 Jetson 上的 `.env`）：

| 变量 | 默认 | 说明 |
|---|---|---|
| `CAMERA_SOURCE` | `auto` | `auto` 无相机自动转合成场景；`index`/`url`/`synthetic` 可指定 |
| `CAMERA_INDEX` / `CAMERA_URL` | `0` / 空 | USB 相机序号 / RTSP·MJPEG 流地址 |
| `DETECT_ENGINE` | `classic` | `classic`=颜色+形状+污渍+二维码；`onnx`=自训练 YOLO（属性仍由经典算法补） |
| `AUTO_DRY_RUN` | `1` | `1` 只记录不驱动机械臂（联调）；现场置 `0` |
| `ROBOT_URL` | 空 | ROS 桥接地址（`http://127.0.0.1:8120`），为空则仅显示与记录 |
| `DISPLAY_HOLD_SECONDS` | `3.0` | 分拣信息在屏最短时间，期间拒绝下一件（赛项） |
| `AUTO_CONFIRM_FRAMES` | `4` | 连续 N 帧同类才入库（防误检空抓） |

详见 `deploy/jetson/README.md`。

## 5. 目录结构

```
src/routes/+page.svelte              显示屏界面（Svelte 5 runes）
src/lib/server/proxy.js              PC 容器代理（指向 Jetson 实际服务；无本地逻辑）
src/routes/api/**/+server.js         PC 端 /api/** → Jetson 网关透传（含自动分拣/标记/画面）
src/routes/health/+server.js         部署状态（是否已连接实际服务）

server/catalog.py                    货物图库（形状×颜色、SVG/位图生成）
server/detect_core.py                识别核心：颜色+形状+污渍缺陷+二维码 → 标记
server/vision_server.py              视觉容器服务（相机采集 / 连续识别 / 标记输出）
server/gateway.py                    网关服务（SPA + 六盒规则 + 自动分拣循环 + 标记转发）

deploy/jetson/Dockerfile             网关镜像（L4T r32.7.1 / py3.6）
deploy/jetson/Dockerfile.vision      视觉镜像（相机设备）
deploy/jetson/requirements*.txt      py3.6 兼容依赖（pydantic 固定 1.9.2 等）
deploy/jetson/ros_bridge.py          抓取指令 → ROS 桥接（参考实现）
deploy/jetson/setup-autostart.sh     开机自启（systemd）+ 显示屏 kiosk 全屏
deploy/jetson/kiosk.sh               kiosk 启动脚本（关息屏、等网关就绪）

docker-compose.jetson.yml            Jetson 两容器编排（视觉 + 网关）
docker-compose.jetson-camera.yml     相机设备映射覆盖（禁用 privileged 时用）
docker-compose.yml / Dockerfile      PC Node 22 调试容器（:5173）
docker-compose.prod.yml              PC 生产预览（adapter-node，:8090）

scripts/build-for-jetson.sh          前端静态产物 + 两个 arm64 镜像 + 离线 tar
scripts/deploy-to-jetson.sh          一键 rsync + ssh 部署 + 健康自检
scripts/sort_client.py               机器人侧客户端（上报分拣、等待显示放行）
.dockerignore / Dockerfile*.dockerignore  构建上下文精简（含 .env 排除）
```

## 6. 主要接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/display` | GET | 投放顺序 / 累计数量 / 储物盒(1–6) / 轮次计时 / 显示闩锁（显示屏轮询） |
| `/api/round/start` | POST | 按统一指令启动装置，**计时开始**，清空显示屏与储物盒 |
| `/api/round/stop` | POST | 结束本轮 `{reason: complete\|dropped_in\|dropped_out\|manual\|estop}` |
| `/api/round/fault` | POST | 掉落告警 `{where: in\|out}` → 本轮比赛结束 |
| `/api/sort/event` | POST | 上报一件已分拣（同形同色同盒；显示闩锁冲突 **409**） |
| `/api/sort/ack` | POST | 确认显示完成 → 放行下一件 |
| `/api/sort/reset` | POST | 复位新一轮（比赛进行中需 `?debug=1`） |
| `/api/sort/demo` | POST | 模拟一件（联调；比赛进行中需 `?debug=1`） |
| `/api/detect` | POST | 识别入口：配置 `VISION_URL` 时转发视觉服务，否则返回 501 说明 |
| `/api/goods`、`/api/images/{货物}.{格式}` | GET | 货物图库（形状×颜色）/ 多格式图片（svg·png·jpg·webp·gif·bmp） |
| `/api/goods/{货物}/image` | POST | 上传现场实物图替换显示图片 |
| `/api/tasks`、`/api/tasks/claim` | GET/POST | 任务队列 / 二维码领取（领取后显示屏与储物盒复位） |
| `/api/robot/action` | POST | 快捷指令 start / pause / reset / estop / idle |
| `/api/rules` | GET | 赛项规则（六盒、每盒 4 件、同形同色同盒、一次一件） |

## 7. 现场自检

```bash
curl http://localhost/api/status                        # Jetson
curl http://localhost:5173/api/status                    # PC 调试
python3 scripts/sort_client.py --gateway http://localhost --demo 5   # 走通显示流程
```

## 8. 视觉容器接口（sort-vision :8100）

| 接口 | 说明 |
|---|---|
| `GET /health` | 相机状态、引擎、采集/识别帧率、最近标记时延 |
| `GET /detect/frame` | 抓取当前帧并识别 → **标记 JSON**（网关自动分拣主入口） |
| `POST /detect` | 上传图片识别（离线复核 / 数据集检查） |
| `GET /frame.jpg?draw=1` | 当前帧 JPEG，`draw=1` 叠加标记框与属性文字 |
| `GET /marks/latest` | 最近一次标记 JSON |
| `GET /stream.mjpg?draw=1` | 可选：带标记 MJPEG 流（手机/旁路监看，非必需） |

标记（mark）结构示例：

```json
{"ts": 1789132703.0, "frame_index": 2372, "size": [1280, 720], "engine": "classic",
 "qr_text": "SORT-TASK:T-260902",
 "detections": [{"class": "cube_3", "name": "绿色正方体", "shape": "正方体", "color": "绿色",
                 "marks": ["缺陷"], "dark_ratio": 0.031, "circularity": 0.392, "vertices": 4,
                 "box": [521, 241, 761, 480], "box_norm": [0.407, 0.335, 0.188, 0.332],
                 "conf": 0.94, "qr": "SORT-TASK:T-260902"}]}
```

## 9. 本地联调（无 Jetson / 无相机）

```bash
# 终端 1：视觉容器（合成托盘场景，自动轮换货物/污渍/二维码）
CAMERA_SOURCE=synthetic PORT=8100 python3 server/vision_server.py

# 终端 2：网关容器
VISION_URL=http://127.0.0.1:8100 PORT=80 python3 server/gateway.py
curl -X POST localhost/api/auto/start -H 'Content-Type: application/json' \
     -d '{"auto_start_round":true,"max_items":5,"dry_run":true}'
curl localhost/api/auto/status        # 观察自动分拣进度
curl localhost/api/marks/log          # 标记输出历史
```
