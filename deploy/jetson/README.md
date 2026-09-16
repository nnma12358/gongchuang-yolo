# Jetson Nano 部署说明 · 自动分拣系统（视觉容器 + 网关容器）

> 适用：Jetson Nano（JetPack 4.6.1 · aarch64 · Python 3.6）
> 现场全部功能都在 Jetson 端：相机采集、识别、标记输出、分拣规则、自动分拣循环、显示屏页面。

## 1. 容器与职责（视觉 + 网关 + ROS 2）

| 容器 | 端口 | 职责 | 关键接口 |
|---|---|---|---|
| `sort-vision`（`Dockerfile.vision`） | 8100 | 独占相机 → 连续识别（颜色/形状/污渍缺陷/二维码）→ **输出标记** | `/detect/frame`、`/marks/latest`、`/frame.jpg?draw=1`、`/health` |
| `sort-gateway`（`Dockerfile`） | 80 | 前端 SPA + 六盒规则 + **自动分拣循环** + **抓取/投放解算** | `/api/auto/start`、`/api/display`、`/api/calib`、`/api/robot/result` |
| `ros2`（`ROS2_IMAGE=ros:foxy-ros-base`） | 8120 | 机械臂驱动 / Astra / MoveIt / **ROS2 桥接**（与驱动共用 DDS） | `POST /execute`、`GET /health` |
| 兼容 `ros-bridge`（`--profile ros1`） | 8121 | 机械臂仍是 ROS 1 时的 HTTP→ROS1 桥接 | `POST /execute` |

通信：host 网络下网关通过 `VISION_URL=http://127.0.0.1:8100` 调视觉容器；视觉容器不需要知道网关存在。

```
相机 ─► 视觉容器(:8100) ──标记JSON──► 网关容器(:80) ──► 显示屏页面(http://localhost)
                                          │
                                          └──抓取指令──► ROS 桥接(:8120) ──► 机械臂
```

## 1.5 部署脚本（PC 上一键完成）

```bash
# 一键：本地构建前端 → rsync 到 Jetson → 构建并启动 → 健康自检
bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50
# 离线镜像（Jetson 无外网）：先出 tar，再部署
bash scripts/build-for-jetson.sh
bash scripts/deploy-to-jetson.sh nvidia@192.168.1.50 --use-tar

# Jetson 上装开机自启（systemd + 显示屏 kiosk 全屏、关闭息屏）
bash deploy/jetson/setup-autostart.sh
bash deploy/jetson/setup-autostart.sh --uninstall     # 卸载
```

## 2. 部署

```bash
# ① PC 上生成前端静态产物（本仓库根目录）
bash scripts/build-for-jetson.sh --no-image       # 产出 build/

# ② 把仓库同步到 Jetson 后启动两容器
docker compose -f docker-compose.jetson.yml up -d --build

# ③ 自检
curl http://localhost/health                      # 网关 + 视觉容器状态
curl http://localhost:8100/health                 # 相机 / 引擎 / 帧率
curl http://localhost/api/auto/status             # 自动分拣循环状态

# ④ 显示屏 kiosk（HDMI 高亮屏）
chromium-browser --kiosk --noerrdialogs http://localhost
```

无相机时可用合成场景先跑通流程：`CAMERA_SOURCE=synthetic docker compose -f docker-compose.jetson.yml up -d`
（视觉容器会生成随机货物/污渍/二维码的托盘画面，**不需要 /dev/video0 存在**）。

宿主机禁用 `privileged` 时，用设备映射覆盖文件（注意 /dev/videoN 必须真实存在）：

```bash
docker compose -f docker-compose.jetson.yml -f docker-compose.jetson-camera.yml up -d
```

> **Python 3.6 依赖约束**（已在 `requirements*.txt` 固定，勿升级）：
> `pydantic==1.9.2`（1.10+ 要求 py≥3.7）、`requests==2.27.1`、`typing-extensions==4.1.1`、
> `uvicorn==0.15.0`；opencv/numpy/Pillow 走 apt（`python3-opencv`/`python3-numpy`/`python3-pil`），
> 不用 pip 装 Pillow（aarch64+py3.6 无 wheel，会触发源码编译）。
> 两个 Dockerfile 会先用阿里云镜像重写 apt 源，并把 pip 升到 `<22`（L4T 自带 pip 9 装不了 manylinux2014 轮子）。

## 3. 自动分拣（赛项流程）

```bash
# 启动自动分拣（自动开始本轮计时；先 dry_run 联调，不驱动机械臂）
curl -X POST http://localhost/api/auto/start -H 'Content-Type: application/json' \
     -d '{"auto_start_round":true,"dry_run":true,"max_items":5}'

# 观察
curl http://localhost/api/auto/status     # count / waiting_clear / last_event / last_robot
curl http://localhost/api/display         # 投放顺序表 + 六个储物盒 + 计时
curl http://localhost/api/marks/log       # 标记输出历史（JSONL）

# 现场正式运行（驱动机械臂）
ROBOT_URL=http://127.0.0.1:8120 AUTO_DRY_RUN=0 docker compose -f docker-compose.jetson.yml up -d
```

循环内部严格按赛项执行：**连续 N 帧确认 → 同形同色同盒（每盒≤4）→ 写入投放顺序并闩锁显示屏 3s →
下发抓取 → 等托盘清空 → 下一件**；一旦掉落（装置内/外）本轮结束并停止计时。

## 4. 机器人侧对接（ROS 桥接）

```bash
# 方式 A：容器内运行（推荐）
docker compose -f docker-compose.jetson.yml --profile robot up -d

# 方式 B：宿主机 ROS 直接运行
ROS_MASTER_URI=http://127.0.0.1:11311 PORT=8120 ARM_ACTION=/arm_pick_place \
  python3 deploy/jetson/ros_bridge.py
```

网关登记一件后会 POST 到 `ROBOT_URL`：

```json
{"task":"pick_and_place","seq":3,"class":"prism5_4","name":"青色五棱柱",
 "shape":"五棱柱","color":"青色","bin":1,"marks":["污渍"],"qr":"SORT-TASK:T-260901"}
```

`ros_bridge.py` 把它转成 ROS 话题/服务调用（按现场接口改 10 行即可）。

## 5. 关键接口

| 接口 | 方法 | 说明 |
|---|---|---|
| `/api/display` | GET | 投放顺序 / 累计 / 储物盒(1–6) / 轮次计时 / 显示闩锁 |
| `/api/auto/start`、`/api/auto/stop`、`/api/auto/status` | POST/GET | 自动分拣循环控制与状态 |
| `/api/marks/latest`、`/api/marks/log` | GET | 标记输出（最近一帧 / JSONL 历史） |
| `/api/frame.jpg?draw=1` | GET | 当前帧（含标记框），前端「视觉画面」用 |
| `/api/sort/event`、`/api/sort/ack` | POST | 手动上报一件 / 确认显示完成 |
| `/api/round/start|stop|fault` | POST | 计时开始 / 结束 / 掉落结束 |
| `/api/goods`、`/api/images/{货物}.{格式}` | GET | 货物图库 / 多格式图片 |
| `/api/tasks`、`/api/tasks/claim` | GET/POST | 任务队列 / 二维码领取 |


## 7. ROS 2 容器与抓取定位（阶段4~6）

### 7.1 三个容器

| 容器 | 镜像 | 职责 |
|---|---|---|
| `sort-vision` | L4T r32.7.1 | 相机 + 连续识别 + 标记输出（:8100） |
| `sort-gateway` | L4T r32.7.1 | 前端 SPA + 六盒规则 + 自动分拣 + 抓取解算（:80） |
| `ros2` | `ros:foxy-ros-base`（可换 `ROS2_IMAGE`） | 机械臂驱动 / Astra / MoveIt / **ROS2 桥接**（:8120） |

三者用 host 网络 + CycloneDDS 互通（`deploy/jetson/cyclonedds.xml`）。
数据流：`网关 --HTTP--> 桥接 --ROS2 话题--> 机械臂驱动`，驱动结果再回报网关。

```bash
# 启动（含 ROS2 容器）
docker compose -f docker-compose.jetson.yml up -d --build
# 宿主机限制 privileged 时叠加设备映射（相机 + /dev/wheeltec_controller）
docker compose -f docker-compose.jetson.yml -f docker-compose.jetson-devices.yml up -d
# 若机械臂仍是 ROS 1（melodic）：改用 ros1 profile
docker compose -f docker-compose.jetson.yml --profile ros1 up -d
```

用自己的驱动镜像（保留你现有的工作空间与 launch）：

```bash
ROS2_IMAGE=my-robot:foxy \
ROS2_CMD="bash -lc 'source /root/ws/install/setup.bash && ros2 launch arm_bringup arm.launch.py'" \
docker compose -f docker-compose.jetson.yml up -d ros2
```

### 7.2 标定（阶段4，**必须先做**）

标定文件 `data/calib.json` 决定“像素 → 机械臂基坐标”的精度，也是自动抓取的前提。

```bash
# ① 托盘上放 6~9 个标定点（贴纸/尖锐物），记录像素坐标（顶置画面上量取或识别框中心）
# ② 示教把夹爪中心依次移到这些点，记录机械臂基坐标 XY（mm），写入 CSV：u,v,x_mm,y_mm[,z_mm]
# ③ 求解并写入标定（含六储物盒中心坐标）
python3 scripts/calibrate.py --points calib_points.csv \
    --out data/calib.json \
    --bins "1:120,60 2:120,0 3:120,-60 4:-120,60 5:-120,0 6:-120,-60" \
    --tray-pixels "300,200 620,210 610,520 290,510"
# 无硬件先自检工具链
python3 scripts/calibrate.py --simulate --out /tmp/calib_test.json
```

判据：**留出验证点 RMS < 2mm**（现场实测 6~9 点、0.7px 取点噪声下约 0.3mm）。
标定结果直接生效（网关启动即读取），也可在线更新：

```bash
curl http://localhost/api/calib          # 查看当前标定与是否可用
curl -X POST http://localhost/api/calib -H 'Content-Type: application/json' -d @data/calib.json
```

### 7.3 抓取与投放解算（阶段5~6）

网关对每件分拣记录自动附加抓取/投放位姿并随指令下发：

```json
{"seq":1,"name":"青色五棱柱","bin":1,
 "grasp":{"grasp_xy":[-54.9,2.9],"yaw_deg":15.0,"joint6_target":0.854,
          "z_approach":130.0,"z_grasp":20.0,"z_lift":120.0,"reachable":true},
 "place":{"bin":1,"place_xy":[120,60],"z_approach":45,"z_release":25}}
```

- **抓取点**：目标框中心经单应（或 2.5D 深度）换算到基坐标
- **抓取高度**：托盘面 + 目标高度 × 0.5（夹爪夹中段，受指长限制 6~22mm；无深度时按 25mm 假设）
- **偏航角**：长方体/棱柱按最小边方向对齐夹爪（`angle_deg` 由视觉给出，缺省按框长边）
- **夹爪目标**：开口 = 目标最小边 + 4mm 留量，线性映射到 joint6 的 0.20~1.20（实机标定值）
- **可达性**：按 `WORKSPACE`（默认 30~350mm 半径）判定，超界标记 `reachable=false` 便于跳过
- **投放**：六盒中心 + 释放高度（`PICK.release_mm`），保证货物完全入盒

机械臂执行完回报网关（掉落会按赛项结束本轮）：

```bash
curl -X POST http://localhost/api/robot/result -H 'Content-Type: application/json' \
     -d '{"seq":1,"status":"ok"}'          # ok | fail | dropped_in | dropped_out
```

### 7.4 现场参数（按实机改这几处）

| 参数 | 位置 | 当前值 |
|---|---|---|
| 夹爪 joint6 范围 | `server/pose.py: GRIPPER` | 0.20(闭) ~ 1.20(开) |
| 抓取各段高度 | `server/pose.py: PICK` | 接近 90 / 抬升 80 / 释放 25mm |
| 可达半径 | `server/pose.py: WORKSPACE` | 30~350mm |
| 桥接话题/服务 | compose `ARM_TOPIC` / `ARM_SERVICE` | `/arm/pick_place` |
| 深度来源 | compose `DEPTH_URL` | 空（可选：Astra 深度快照 URL，用于 2.5D） |


## 9. 适配工作区 2.5D 定位软件（smart_sorting_localization_*）

该导出包含：8088 三画面实时服务、逐像素桌面参考标定、RGB+深度空洞 2.5D 定位、
RGB 轮廓细化、夹爪三红点 TCP 检测。目录状态标注 `HAND_EYE_CALIBRATION=NOT_COMPLETED`、
`COORDINATE_STATUS=CAMERA_OR_PIXEL_FRAME` —— 即：**定位到像素/相机系，缺“相机→机械臂”这一步**，
正是本项目 `pose.py` + 自动手眼标定补齐的部分。

### 9.1 已完成的对接

| 他们的产物 | 本项目用法 |
|---|---|
| `table_reference.npz`（逐像素桌面深度 480×640 + valid/roi 掩膜） | 视觉容器 `TABLE_REF_DIR` 挂载 `/app/calibration`，`detect_core.attach_depth` 用它算目标**真高**（实测 40mm 物体测到 39.5mm） |
| `camera_intrinsics.json`（fx=fy=570.34, cx=319.5, cy=239.5 @640×480） | `scripts/adapt_localization.py import` 写入 `data/calib.json`，抓取解算按此换算相机系坐标 |
| `object_locator_result.json`（RGB+深度空洞定位） | `scripts/adapt_localization.py merge` 与本项目 RGB 识别融合（位置/高度取自他们，形状/颜色/污渍/二维码取本项目），实测匹配分 1.0 / 0.95 |
| `gripper_red_marker_result.json`（TCP 像素 + 夹爪轴向） | `scripts/adapt_localization.py tcp` **自动手眼标定**：每次移到已知点记录一对 (TCP 像素 ←→ 机械臂 XY)，≥4 点即可求解单应 |
| 8088 实时服务 | 无深度端点；改用本项目的 `depth_http_node.py`（见 9.2） |

### 9.2 深度/彩色接入（不修改他们的代码）

`deploy/jetson/depth_http_node.py` 跑在 ros2 容器内，直接订阅 Astra 话题并转成 HTTP：

```
/health              彩色/深度帧率、帧号、帧龄（与他们的 /health 语义一致）
/color.jpg | /color.mjpg   彩色帧 / MJPEG —— 可作视觉容器的 CAMERA_URL
/depth.png           16-bit PNG（mm）—— 视觉容器的 DEPTH_URL
/depth_preview.jpg   深度伪彩预览（人工核对）
/points_camera.json  轻量 2.5D 候选（相机系坐标），快速联调用
```

现场已验证的坑：相机端 QoS 为 **RELIABLE**，订阅端必须匹配，否则 RGB 冻结 ——
节点默认 `RELIABLE_QOS=1`。

### 9.3 三步接上自动抓取

```bash
# ① 导入他们的标定（内参 + 桌面参考）
python3 scripts/adapt_localization.py import \
    --table ~/smart_sorting_localization_*/calibration/table_reference_candidate_* \
    --intrinsics ~/smart_sorting_localization_*/calibration/*/camera_intrinsics.json \
    --calib data/calib.json

# ② 自动手眼标定（夹爪红点）：把夹爪移到托盘上的已知点，各记一对
python3 scripts/adapt_localization.py tcp --tcp <gripper_red_marker_result.json> \
    --robot-x 120 --robot-y 60 --points data/calib_points.csv      # 重复 ≥6 个点
python3 scripts/adapt_localization.py tcp --solve --points data/calib_points.csv \
    --calib data/calib.json --bins "1:120,60 2:120,0 3:120,-60 4:-120,60 5:-120,0 6:-120,-60"

# ③ 融合定位与识别（离线核对；在线由网关自动完成）
python3 scripts/adapt_localization.py merge \
    --objects <object_locator_result.json> --marks <本项目识别.json> \
    --table <table_reference 目录> --calib data/calib.json --out /tmp/merged.json
```

### 9.4 对他们软件的三点优化（已体现在适配层）

1. **高度更准**：原实现用单一 `table_depth_mm`（结果里出现过 428 / 676mm 不一致）；
   本适配层按**逐像素桌面参考**差分，并对深度空洞（黑/白货物无回波）显式标记 `depth_source=hole`，
   交给抓取侧按假设高度处理，而不是给出错误的 0mm 高度。
2. **属性补全**：他们的定位只给 BLACK/WHITE 外观 + 位置；形状/颜色/污渍/二维码由本项目识别引擎提供，
   融合时用 IoU + 中心距离双判据，匹配分与未匹配项一并输出，便于复核。
3. **坐标系贯通**：相机/像素系 → 机械臂基坐标（单应或 2.5D 刚体），并直接给出抓取/投放位姿与夹爪关节值。

## 10. 环境变量（各容器）

| 容器 | 变量 | 默认 | 说明 |
|---|---|---|---|
| vision | `CAMERA_SOURCE` | `auto` | `auto`/`index`/`url`/`synthetic` |
| vision | `CAMERA_INDEX`、`CAMERA_URL` | `0`、空 | USB 相机序号 / RTSP·MJPEG 地址 |
| vision | `DETECT_ENGINE`、`MODEL_PATH` | `classic` | `onnx` 时加载自训练 YOLO（属性仍走经典算法） |
| vision | `VISION_FPS`、`CONF_MIN` | `10`、`0.45` | 识别频率 / 置信度下限 |
| vision | `STAIN_DARK_RATIO`、`DEFECT_DARK_RATIO` | `0.05`、`0.14` | 污渍 / 缺陷判定阈值 |
| gateway | `VISION_URL` | `http://127.0.0.1:8100` | 视觉容器地址 |
| gateway | `ROBOT_URL` | 空 | ROS 桥接地址；为空则只显示与记录 |
| gateway | `AUTO_DRY_RUN` | `1` | `1` 不驱动机械臂（联调）；现场 `0` |
| gateway | `AUTO_CONFIRM_FRAMES`、`AUTO_CONF_MIN` | `4`、`0.6` | 确认帧数 / 置信度下限 |
| gateway | `DISPLAY_HOLD_SECONDS` | `3.0` | 显示保持时间（期间拒绝下一件） |
| gateway | `CALIB_PATH` / `GRASP_ENABLED` | `data/calib.json` / `1` | 标定文件路径 / 是否解算抓取位姿 |
| gateway | `ROBOT_URL` | `http://127.0.0.1:8120` | ROS2 桥接地址（置空=不驱动机械臂） |
| ros2 | `ROS2_IMAGE` / `ROS2_CMD` | `ros:foxy-ros-base` / 桥接 | 驱动镜像与启动命令 |
| ros2 | `ROS_DOMAIN_ID` / `RMW_IMPLEMENTATION` | `0` / `rmw_cyclonedds_cpp` | DDS 域与实现 |
| ros2 | `ARM_TOPIC` / `ARM_SERVICE` | `/arm/pick_place` / 空 | 下发方式（话题或服务） |
| ros2 | `JOINT6_CLOSED` / `JOINT6_OPEN` | `0.20` / `1.20` | 夹爪闭合/张开关节值 |
