<!-- 本文件由 开发交接记录-20260921.md 脱敏生成：已移除设备密码等敏感信息。
     含凭据的完整版仅在本地交接包内，未入库。 -->

# 开发交接记录 · 分拣视觉项目（截至 2026-09-21）

> 这份文档的目标：**换一台设备、用任意 agent 工具，仅凭这份记录就能继续开发**。
> 所有路径、账号、命令、实测数字、踩坑记录都在里面。文中标注 ⚠️ 的是"需现场复核"（记录时设备已断开）。

---

## 0. 给下一个 Agent 的阅读顺序

1. 本文 **§1 项目是什么** + **§2 系统架构** —— 建立全局认知（3 分钟）
2. **§7 当前状态** + **§8 下一步待办** —— 知道现在该干什么
3. **§10 踩坑清单** —— **动手前必读**，能省掉今天踩过的所有坑
4. 仓库内的补充文档：`工创yolo/docs_OVERHEAD.md`（重训交接）、`工创yolo/docs_TRAINING.md`（训练全流程）、`sort-web/README.md`（部署）

**当前唯一阻塞**：等用户提供**现场真实照片**（同一相机视角/尺度），才能跑正式的重训与导出（§8）。
在此之前不要重新训练、不要部署新模型（用户明确要求"等我真实照片再训练导出"）。

---

## 1. 项目是什么

一台**移动机械臂 + 顶置相机**的自动分拣系统。视觉负责：识别台面上的多面体货物（形状/颜色/表面缺陷）→ 输出 2.5D 位置 → 由机械臂抓取分拣。

- **硬件**：Jetson Nano（视觉/网关）+ 轮式机械臂（Wheeltec mini_mec 六轴）+ 顶置 Orbbec Astra 相机
- **AI 链路**：YOLOv8n 检测（单类 `goods`）→ CNN 属性分类（颜色 9 / 形状 9 / 污渍 3）→ 2.5D 深度 → 标记输出
- **前端**：SvelteKit（摄像头画面 / 快捷任务按钮 / 扫码领任务），由网关容器托管

### 三个代码位置

| 位置 | 内容 | 远程 |
|---|---|---|
| `/home/xxxffyy/工创/sort-web` | 前端 + 网关 + Jetson 部署编排（compose/Dockerfile/桥接脚本） | `git@github.com:nnma12358/gongchuang-yolo.git` 分支 `sort-web`（本地 `master` 跟踪它） |
| `/home/xxxffyy/工创/工创yolo` | 训练/标注/数据集工具链 + 权重 + 合成数据生成器 | 同一远程，分支 `main` |
| `/home/xxxffyy/工创/视觉测试结果` | 实测结果与图片（已打包见 §9） | 未入版本库 |

**今日结尾的提交**（两侧都已 push）：

```
sort-web : 11204a8  fix(bridge/vision): 修 ROS2 Header 无 seq、MJPEG 用 StreamingResponse、
                    fetch_depth 缺 import cv2；QoS 默认改 BEST_EFFORT
工创yolo : d598253  feat(overhead): 现场真实尺度重训全套工具（采集/导入/组装/自动提议/微调导出/交接文档）
           9176d51  feat(tools): 加 probe_object_scale.py —— 正确测尺度-置信度曲线
```

---

## 2. 系统架构

```
                    ┌──────────────── Jetson Nano（宿主机网络 host）─────────────────┐
 顶置 Astra 相机 ──► │ ros2_arm_container （用户自带，未改动）                        │
 (USB 2bc5:0402)    │   ├─ astra_camera_node  → /camera/color/image_raw             │
                    │   │                       /camera/depth/image_raw             │
                    │   └─ wheeltec_arm_driver ← arm_joint_command (JointState)      │
                    │                                                                │
                    │ sort-ros-bridge  :8120 HTTP↔ROS2 指令 | :8123 话题→HTTP 图像   │
                    │        ▲                                                       │
                    │ sort-yolo :8101 (TensorRT FP16)  ← sort-vision :8100 (编排)    │
                    │ sort-cnn  :8102 (ONNX 属性)          │                         │
                    │                                      ▼                         │
                    │ sort-gateway :80  ── SvelteKit 前端 + 自动分拣状态机            │
                    └────────────────────────────────────────────────────────────────┘
```

| 服务 | 端口 | 作用 | 备注 |
|---|---|---|---|
| `sort-yolo` | 8101 | 检测（TRT FP16，218~331ms/帧） | 模型 `goods_yolov8n_640_fp16.engine` |
| `sort-cnn` | 8102 | 颜色/形状/污渍属性 | ONNX Runtime |
| `sort-vision` | 8100 | 三层编排 + 2.5D + 二维码 + 标记绘制 | 相机源 = 桥接 MJPEG |
| `sort-gateway` | 80 | 前端页面 + 自动分拣流程 | 前端所显示的状态必须来自真实服务 |
| `sort-ros-bridge` | 8120 / 8123 | HTTP→ROS2 指令 / ROS2→HTTP 图像 | 用户机械臂的唯一接口 |
| `ros2_arm_container` | — | 用户自带的 ROS2 机械臂环境 | **绝对不要重建/删除**，只允许 `docker exec` |

源码挂载差异（改代码后的生效方式不同，**很容易踩**）：
- `sort-ros-bridge`：`deploy/jetson/*.py` 是**只读挂载** → 改完 `docker restart sort-ros-bridge` 即生效
- `sort-vision` / `sort-yolo` / `sort-cnn` / `sort-gateway`：源码是 **COPY 进镜像**的 → 必须 `docker-compose build <svc>` + `up -d --force-recreate <svc>`

---

## 3. 现场设备访问信息

| 项 | 值 |
|---|---|
| Jetson | `wheeltec@10.60.10.24` |
| 密码 / sudo 密码 | **不在版本库中记录**，向项目负责人索取（用法见 §11 命令示例） |
| SSH 公钥 | 已装在设备上（换设备需重新分发公钥） |
| 部署目录（设备上） | `~/sort-jetson-deploy-20260917`（已在原地打补丁，含今日全部修复） |
| ROS2 容器 | `ros2_arm_container`（镜像 `wheeltec_ros2_astra:foxy`，`--runtime nvidia`，手动启动，`restart=no`） |
| ROS2 环境 | `ROS_DOMAIN_ID=95`、`ROS_LOCALHOST_ONLY=1`、`RMW_IMPLEMENTATION=rmw_cyclonedds_cpp` |
| 设备版本 | Jetson Nano / Tegra X1 Maxwell SM5.3 / L4T r32.5.2 = **JetPack 4.5.1** / Python 3.6.9 / Docker 20.10.21 / docker-compose **v1.29.2** / TensorRT **7.1.3**（`trtexec` 在 `/usr/src/tensorrt/bin/trtexec`） |
| 机械臂接口 | 订阅 `sensor_msgs/JointState` 话题 **`arm_joint_command`**；消息类型见 `wheeltec_arm_interfaces/*`；MoveIt 配置 `mini_mec_six_arm_moveit_config`；手眼标定 `~/ros2_ws/passive_handeye_calibration` |

### Astra 相机怎么启动（容器里**没有 `ros2` CLI**，别用 `ros2 launch`）

```bash
# 1) 在设备上生成启动脚本（容器内 /tmp）
cat > /tmp/start_astra.sh <<'EOS'
#!/bin/bash
PREFIX=/ros2_ws/wheeltec_arm_ros2_foxy_ready/install/astra_camera
RT=$PREFIX/lib/astra_camera/openni2
source /ros2_ws/wheeltec_arm_ros2_foxy_ready/install/setup.bash
export OPENNI2_REDIST=$RT OPENNI2_DRIVERS_PATH=$RT/OpenNI2/Drivers
export LD_LIBRARY_PATH=$RT:${LD_LIBRARY_PATH}
export ROS_DOMAIN_ID=95 ROS_LOCALHOST_ONLY=1 RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
exec $PREFIX/lib/astra_camera/astra_camera_node --ros-args \
  -r /camera/rgb/image_raw:=/camera/color/image_raw \
  -r /camera/rgb/camera_info:=/camera/color/camera_info \
  -p device_uri:=ANY_DEVICE -p enable_color:=true -p enable_depth:=true -p enable_ir:=false \
  -p depth_registration:=true \
  -p color_width:=640 -p color_height:=480 -p color_fps:=30 \
  -p depth_width:=640 -p depth_height:=480 -p depth_fps:=30
EOS
docker cp /tmp/start_astra.sh ros2_arm_container:/tmp/start_astra.sh
docker exec -d ros2_arm_container bash -lc 'chmod +x /tmp/start_astra.sh && nohup /tmp/start_astra.sh > /tmp/astra.log 2>&1 &'
docker exec ros2_arm_container bash -lc 'tail -20 /tmp/astra.log'   # 期望看到 "Astra camera opened: Astra"
```

要点：
- 只有 `wheeltec_arm_ros2_foxy_ready/install/astra_camera` 这一份带 **OpenNI2 运行时**（`astra_ws` 那份没有 openni2 目录，跑不起来）
- 相机原生话题是 `/camera/rgb/image_raw`、`/camera/depth/image_raw`；用户的 launch 文件把它们 remap 到 `/overhead_camera/*`，而他们的 `dry_run_config.json`/`config.py` 用的是 `/camera/*` —— **`/camera/*` 才是与部署一致的那套**，所以上面直接 remap 到 `/camera/*`（`/camera/depth/image_raw` 本来就同名）
- 容器重启后 `/tmp/start_astra.sh` 会丢，需要重新生成
- 相机发布端是 **BEST_EFFORT（SensorDataQoS）**，订阅端必须用 BEST_EFFORT（见 §10 坑 4）

---

## 4. 今日完成的工作

### 4.1 打通现场相机（此前前端根本没接上真机）

现场 Astra 由用户自己的 `astra_camera_node` 驱动，但容器里没有 `ros2` CLI。改为直接跑节点可执行文件，彩色 640×480 + 深度都通了；桥接容器的彩色 MJPEG / 深度 PNG 也通了（`/color.jpg`、`/depth.png`、`/color.mjpg`、`/depth_preview.jpg`）。

### 4.2 修掉 4 个真实 bug（详见 §5）

`ROS2 Header 无 seq`、`MJPEG 生成器交给 Response`、`fetch_depth 缺 cv2`、`QoS 订阅端不兼容`。

### 4.3 用真实货物做了完整实测，并定位漏检根因

在 660mm 安装高度下，把货物摆在台面上，用**部署后的完整链路**做了 60 帧采样：
- 真实货物 **0/5~6 检出**
- 60 帧中 **48 帧（80%）**存在同一个假检：**黑色工具箱**，conf 0.365~0.487，属性被判"黑色正四面体 + 缺陷"
- 经典 CV 引擎更差（40/40 帧误检，出现半屏大框）

根因（用排除法实验定位，不是猜测）：
**物体像素尺度**。模型工作区间 69~100px，现场货物只有约 **30px**。
不是阈值（0.01 也仅 0.122）、不是背景（灰纸 vs 白桌面几乎同分）、不是引擎、不是 ROI/数字放大（网格裁剪最高 0.152）。

几何测算：相机离台面约 **660mm**（深度中位数 663mm），而合成生成器里 `REAL["cam_height"]=320.0`（注释写"现场实测"）——**设计值的一半**，货物像素正好少一半。

### 4.4 铺好「按真实尺度重训」的工具链（等照片就能跑）

`工创yolo/tools/` 下新增 6 个工具 + 1 个训练配方 + 1 份交接文档，全部自测过（§8、§9）。

### 4.5 交付成果打包

实测结果与图片打包（§9），两个仓库的修复都已 push。

---

## 5. 今日修掉的 Bug 清单（症状 → 根因 → 修法）

| # | 文件 | 症状 | 根因 | 修法 |
|---|---|---|---|---|
| 1 | `deploy/jetson/depth_http_node.py` | 桥接订阅回调每帧抛异常、收不到图 | ROS2 的 `std_msgs/Header` **没有 `seq` 字段**（那是 ROS1 的），`msg.header.seq` 抛 `AttributeError` | 新增 `_seq_of()`：优先取 `header.stamp` 纳秒值，取不到则自增计数 |
| 2 | `deploy/jetson/depth_http_node.py` | `/color.mjpg` 一直 HTTP 500，报 `'generator' object has no attribute 'encode'` | MJPEG 是无限生成器，却被塞进 `fastapi.responses.Response`（会 `.encode()`） | 改用 `StreamingResponse` |
| 3 | `server/vision_server.py` | 日志 `深度快照获取失败: name 'cv2' is not defined`，2.5D 一直不可用 | 该文件**没有模块级** `import cv2`（只在别的函数里局部导入过），`fetch_depth()` 里直接用 `cv2.imdecode` | 在 `fetch_depth()` 内补 `import cv2` |
| 4 | `docker-compose.jetson.yml` | 桥接一帧图都收不到，日志反复报 `requesting incompatible QoS` | Astra 发布端为 **BEST_EFFORT**，而订阅端是 **RELIABLE**，二者不兼容 | `RELIABLE_QOS=0`；**默认值也从 1 改成 0**（BEST_EFFORT 订阅端对 RELIABLE/BEST_EFFORT 两种发布端都兼容，更稳） |
| 5 | `工创yolo/tools/assemble_overhead_dataset.py` | 组装后"含货物 0"（原有 197 个正样本全变空标注） | `label_for()` 拼标签路径时把 `images/<split>` 当成了 `<sess>/images`，路径算错，`copy_pair` 只好写空文件 | 按两种布局分别处理（标准 YOLO / 采集会话），修复后 train 含货物 210、val 53 |

---

## 6. 关键实测数据（可直接引用的数字）

### 6.1 模型标定（24 张人工复核过的实拍特写）

| 指标 | 值 |
|---|---|
| 命中 | **24/24（IoU≥0.5）**，0 漏检 0 误检 |
| IoU | 均值 0.869（0.78~0.96） |
| 检测置信度 | 均值 0.890 |
| 属性置信度 | 颜色 0.995 / 形状 0.887 / 污渍 0.852 |
| 耗时 | 503~863ms/帧（Jetson TRT FP16） |
| 识别结果举例 | 白色 dodecahedron / 白色正方体 / 白色球 / 黑色正方体 |

### 6.2 现场真实货物（60 帧采样，660mm 俯拍）

| 指标 | 值 |
|---|---|
| 真实货物检出 | **0 / 5~6** |
| 有检出帧 | 48/60 = 80%，全部是同一个假检 |
| 假检 | 黑色工具箱，框≈(0,0,170,220)，conf 0.365~0.487，判为「黑色正四面体 + 缺陷」 |
| 深度 2.5D | 正常：桌面 **663mm**、工具箱凸起 **73mm** |
| 采集/检测帧率 | 采集 9.2fps / 检测 1.7~2.1fps（TRT 245.9ms 实测） |

### 6.3 尺度-置信度曲线（根因证据，固定 640 画布只改物体像素）

| 物体像素 | 灰纸背景（训练同款） | 白桌面背景（现场同款） |
|---|---|---|
| 330px | 0.022 | 0.035 |
| 200px | 0.243 | 0.208 |
| 160px | 0.469 | 0.147 |
| **100px** | **0.708** | **0.721** |
| **69px** | **0.611** | **0.589** |
| 45px | 0.382 | 0.299 |
| **33px（现场实际）** | **0.108** | **0.117** |
| 24px | 0.015 | 0.012 |

结论：最佳区间 **69~100px**；现场 30px 落在崩溃区（0.11 < 0.35 阈值）。
复现脚本 `工创yolo/tools/probe_object_scale.py`（**注意它必须"固定画布、只缩放物体"**，见 §10 坑 7）。

### 6.4 模型的其它已知短板

- 形状 CNN 跨域仍偏弱（真实货物形状识别约 4/8），补齐需要每种形状 30~50 张真实裁剪图
- 数据集：真实域验证集 = 40 张真实（全部人工复核）+ 10 张生成 + 15 张背景，按组切分无泄漏；mAP50 0.9950 / mAP50-95 0.7617；conf 0.35 时误检 0.00/图

---

## 7. 当前部署状态与配置

### 7.1 设备上 `.env` 的关键项（记录时值）

```ini
CAMERA_SOURCE=bridge
CAMERA_URL=http://127.0.0.1:8123/color.mjpg     # 现场相机走桥接 MJPEG
DEPTH_URL=http://127.0.0.1:8123/depth.png
DETECT_ENGINE=yolo                              # yolo 三层链路（失败回退 classic）
YOLO_CONF=0.35
RELIABLE_QOS=0
ROI=                                            # 空 = 全画面（未启用托盘 ROI）
```

### 7.2 容器状态（记录时）

`sort-yolo` / `sort-cnn` / `sort-vision` / `sort-gateway` 全部 healthy；`sort-ros-bridge` up；
`ros2_arm_container` up（今日全程未改动）。内存约 2.4GB 已用 / 1.25GB 空闲；限额合计 2.4GB RAM + 4.5 CPU。

### 7.3 模型文件

```
sort-web/models/detect/goods_yolov8n_640_fp32.onnx     # 部署基准
sort-web/models/detect/goods_yolov8n_640_int8.onnx     # 量化版
设备上 models/detect/goods_yolov8n_640_fp16.engine      # 11MB，在设备上用 trtexec 生成（TRT 版本必须与设备一致）
sort-web/models/MANIFEST.json                          # 供 /health 返回 model_info
```

⚠️ 记录时 `detections: 0` 是**预期状态**——就是待重训解决的那个问题，不是故障。

---

## 8. 未完成目标与下一步（等用户提供真实照片）

**目标**：保持 660mm 安装高度，用真实成像尺度重训检测模型并部署验证。

**唯一前提**：用户提供**同一相机视角/尺度**的真实照片（俯拍，货物约占画面宽 5%，即 640 宽时约 30px）。
手机近景特写**无效**——那正是现有 24 张实拍集已覆盖的域。

### 流程（照片到位后按序执行）

```bash
cd /home/xxxffyy/工创/工创yolo

# ① 采集（也可由 agent 远程直接采，PC 能直连 10.60.10.24:8123，约 77ms/帧）
python3 tools/capture_overhead.py --session tray_a --n 60      # 有货：现场换位置/朝向/数量
python3 tools/capture_overhead.py --session bg_lab --n 40      # 纯背景（负样本，抑制工具箱误检）
# 或用用户拍好的照片：
python3 tools/import_real_photos.py --src ~/照片/现场俯拍 --session tray_phone

# ② 自动提议框（深度管浅色货、RGB 暗块管深色货）→ 人工复核
python3 tools/propose_boxes.py --session tray_a --roi 180,150,470,360 --write
python3 tools/review_app.py --help                              # 已有的人工复核流程

# ③④⑤ 一键：合成(660mm几何) → 组装 → 微调 → 评估 → 导出 ONNX → 打印设备端重建引擎命令
bash tools/finetune_overhead.sh
bash tools/finetune_overhead.sh --skip-synth                    # 复用已生成合成数据

# 回归验收（修复前基线：真实货物 0/5~6 检出；48/60 帧工具箱假检 0.365~0.487）
python3 tools/capture_overhead.py --session regression --n 60 --marked
```

**设备端重建引擎**（TRT 必须在设备上编，`finetune_overhead.sh` 第 8 步会打印）：

```bash
scp ../sort-web/models/detect/goods_yolov8n_640_fp32.onnx \
    wheeltec@10.60.10.24:~/sort-jetson-deploy-20260917/models/detect/
scp scripts/build-trt-on-jetson.sh wheeltec@10.60.10.24:~/sort-jetson-deploy-20260917/scripts/
ssh wheeltec@10.60.10.24 'cd ~/sort-jetson-deploy-20260917 && printf '<密码>
' | sudo -S bash scripts/build-trt-on-jetson.sh'
ssh wheeltec@10.60.10.24 'cd ~/sort-jetson-deploy-20260917 && printf '<密码>
' | sudo -S docker-compose -f docker-compose.jetson.yml up -d --no-build yolo'
```

### 备选路线（若允许改硬件）

把相机降到离台面 **200~290mm**，货物像素回到 69~100px（置信度 0.59~0.72），**不需重训**。
代价：视场缩小到约 230~335mm 宽，需确认能覆盖托盘与机械臂作业范围。用户已选择"保持 660mm 重训"。

---

## 9. 工具与产物清单

### 9.1 今日新增的训练工具（`工创yolo/tools/`，全部自测过）

| 文件 | 作用 | 自测结果 |
|---|---|---|
| `probe_object_scale.py` | 尺度-置信度曲线探针（定位漏检根因） | ✅ 曲线复现 §6.3 |
| `capture_overhead.py` | PC 直连现场相机抓帧，自动去重，存**原始 16bit 深度**+预览 | ✅ 77ms/帧 |
| `import_real_photos.py` | 导入用户照片：去重/校验/拼图 + 尺度提醒 | ✅ |
| `propose_boxes.py` | 深度+RGB 双通道自动提议框 | ✅ 现场帧 5~6 个货物全覆盖 |
| `assemble_overhead_dataset.py` | 组装 mix_v3 + 真实俯拍 + 现场几何合成 | ✅（修了 1 个路径 bug） |
| `finetune_overhead.sh` | 一键微调→评估→导出→部署指引 | 已写，待数据 |
| `config/train_overhead.yaml` | 现场尺度微调配方（imgsz 640 与现有引擎形状一致） | — |
| `docs_OVERHEAD.md` | 重训交接文档（与本文互补） | — |

组装器的**安全规则**（很重要）：`bg_*` 会话当负样本；其它会话**只收标注非空的图**——有货无标注会把货物教成背景，脚本会告警并整批排除。

### 9.2 交付压缩包（都在 `/home/xxxffyy/工创/`）

| 文件 | 大小 | md5 | 内容 |
|---|---|---|---|
| `视觉识别结果-20260921.zip` | 5.9M | `3da96ca1bd3c0fd709da537cf62ec704` | 视觉识别结果 + 图片（含 README、汇总 CSV） |
| `视觉识别结果-20260921.tar.gz` | 5.9M | `943c5e2cd3f56f71c171694569e263e2` | 同上 |
| `sort-jetson-deploy-20260921.tar.gz` | 25M | `7a642c32f5b9605204b74c259054f779` | 设备端一键解压构建用的完整部署包 |
| `sort-jetson-upgrade-20260921-r2.tar.gz` | 13M | — | 在旧部署上的增量升级包 |
| `sort-toolchain-20260921.tar.gz` | 103K | — | 训练/标注工具链 |

`视觉识别结果` 包结构：`01_模型标定_24张实拍/`、`02_真实货物实测_现场Astra相机/`、`03_诊断证据_尺度实验/`。

### 9.3 PC 上两个 Python 环境（别搞混）

| 环境 | 用途 | 版本 |
|---|---|---|
| `/home/xxxffyy/工创/sort-web/.venv-gw/bin/python` | 图像/ONNX 分析、打包、合成数据生成 | py3.12 · cv2 5.0.0 · onnxruntime 1.30.0 · PIL 12.3.0 · numpy 2.5.3 |
| `/home/xxxffyy/gpu_env/bin/python` | 训练/导出（GPU） | py3.12 · torch 2.6.0+cu124 · ultralytics 8.4.153 · GPU RTX 4050 Laptop 6GB |

---

## 10. 踩坑清单（动手前必读）

1. **顺序千万别反**：`docker-compose up -d --no-build` 在配置没变时会输出 `up-to-date` 而**不重启容器**，只读挂载的脚本改动不会生效。改脚本后要 `docker restart <svc>` 或 `up -d --force-recreate <svc>`。
2. **改了 `sort-vision` 的源码必须重建镜像**（源码是 COPY 进去的，只有 `calibration/` 是挂载）；`sort-ros-bridge` 的 `deploy/jetson/*.py` 才是只读挂载。
3. **不要动 `ros2_arm_container`**：用户手动启动、`restart=no`、`--runtime nvidia`，重建会毁掉他们的环境。只允许 `docker exec`。
4. **ROS2 QoS 单向兼容**：RELIABLE 发布 + BEST_EFFORT 订阅 = 兼容；BEST_EFFORT 发布 + RELIABLE 订阅 = **一帧都收不到**（日志 `requesting incompatible QoS`）。现场 Astra 是 BEST_EFFORT，所以订阅端一律用 BEST_EFFORT。
5. **ROS2 的 `Header` 没有 `seq`**（ROS1 才有），别照抄 ROS1 代码。
6. **`fastapi.responses.Response` 不能包生成器**（MJPEG 必须 `StreamingResponse`）。
7. **做尺度实验时必须"固定画布、只缩放物体"**：如果直接把整图缩小，YOLO 的 letterbox 又会把它拉回 640，等于没测。今天第一版探针就栽在这。
8. **`synth_polyhedra.py` 的默认几何是设计值（cam_height 320mm），不是现场实测值（约 660mm）**。要按现场尺度生成必须显式传 `--cam-height 660`。
9. **Jetson Nano 是 ARMv8.0**：pip 的 aarch64 wheel 多为 ARMv8.2 编译 → `Illegal instruction`/段错误。已知可用组合：`numpy==1.19.2` + `onnxruntime==1.9.0 --no-deps`（1.19.5/1.18.5 会崩）。用 apt 的 `python3-numpy/python3-opencv/python3-pil`，不要 pip 装 opencv。
10. **apt 版 OpenCV 3.2 没有 `cv2.dnn`** → 检测走 ONNX Runtime + 自写 numpy blob/NMS。
11. **`cv2.findContours` 返回值个数在 3.x/4.x 不同** → 用 `[-2]` 取轮廓。
12. **`daemon.json` 里必须有 `runtimes.nvidia`**，否则 `--runtime nvidia` 容器报 `Unknown runtime specified nvidia`（今天差点因为覆盖 daemon.json 而踩到）。
13. **Dockerfile 用经典解析器**：`ENV` 行尾不能写注释（会 `Syntax error - can't find = in "#"`），`apt` 前必须 `ARG DEBIAN_FRONTEND=noninteractive`（否则 tzdata/debconf 挂住），`pip` 要 `ENV LANG=C.UTF-8`（否则 `UnicodeDecodeError: 'ascii' codec`）。
14. **compose v1 的 `command: >` 折行会被保留换行**，导致 `bash: line 2: -i: command not found` → 用单行数组写法。
15. **`.env` 空值 + 行内注释会污染变量**（例如 `RMW_IMPLEMENTATION=` 后面跟注释）→ 注释单独一行。
16. **compose 里不要出现重复的 `environment` 键**（`non-unique items`）→ 仓库里有 `scripts/check-compose.py` 做体检。
17. **`pgrep -f`/`pkill -f` 会匹配到自己的命令行**，导致杀掉自己的 shell → 用方括号技巧 `[p]attern` 或分开调用。
18. **Pillow 5.x 没有 `ImageDraw.textbbox`** → 用 `textsize` 兜底；Jetson 镜像里没有 CJK 字体，标记中文要靠挂载宿主机的 Noto CJK。
19. **TRT 引擎必须在设备上用 `trtexec` 重建**（PC 上编的 TRT 版本不兼容设备）。
20. **桥接脚本 `.env` 里的 `RELIABLE_QOS` 只作用于图像订阅**，不影响机械臂指令通道。

---

## 11. 常用命令速查

```bash
# ---- 设备访问 ----
ssh wheeltec@10.60.10.24
printf '<密码>
' | sudo -S docker ps

# ---- 部署/重启 ----
cd ~/sort-jetson-deploy-20260917
printf '<密码>
' | sudo -S docker-compose -f docker-compose.jetson.yml up -d --no-build <svc>
printf '<密码>
' | sudo -S docker-compose -f docker-compose.jetson.yml up -d --no-build --force-recreate <svc>
docker logs --tail 30 <container>

# ---- 自检接口 ----
curl -s http://127.0.0.1:8100/health      # 视觉容器（camera/engine/fps/table_ref）
curl -s http://127.0.0.1:8100/detect/frame
curl -s http://127.0.0.1:8101/health      # yolo（engine/conf_thres/model_info）
curl -s http://127.0.0.1:8123/color.jpg -o /tmp/c.jpg
curl -s http://127.0.0.1:8123/points_camera.json   # 2.5D 候选点
curl -s -X POST http://127.0.0.1:8120/execute -H 'Content-Type: application/json' \
     -d '{"joints":[0.1,-0.45,0.6,0.2,0,1.2],"joint_names":["joint1","joint2","joint3","joint4","joint5","joint6"]}'

# ---- PC 端直连现场相机（不需要设备上操作）----
curl -s http://10.60.10.24:8123/color.jpg -o /tmp/live.jpg

# ---- 训练/导出（PC）----
bash tools/finetune_overhead.sh
/home/xxxffyy/gpu_env/bin/python tools/probe_object_scale.py \
    --model /home/xxxffyy/工创/sort-web/models/detect/goods_yolov8n_640_fp32.onnx
```

---

## 12. 已验证的集成事实（不用再怀疑）

- 6/6 DDS 收发在 `sort-ros-bridge` ↔ `ros2_arm_container` 之间验证通过
- `POST http://localhost:8120/execute` 的关节指令能真实送达机械臂控制器
- 纯笛卡尔请求会被拒（提示需要 IK），需带关节解
- 用户的 dry-run 排序节点当时在跑：`calibration_status: MISSING_FINAL_CALIBRATION`、`motion_authorized: false`（**尚未完成最终标定 → 不会有真实动作**）
- 相机/深度/桥接/视觉/检测四段链路今日全部实测可用（彩色 640×480 JPEG、深度 16bit PNG 可用）

---

## 13. 一句话总结今天

> 把前端接上了现场真机相机，修掉 4 个真实 bug，用真实货物做了完整实测 —— **发现模型在现场俯拍视角下 0 检出、还误检工具箱**；
> 用实验证明根因是**成像尺度（现场约 30px vs 模型最佳 69~100px）**，并把"按真实尺度重训"的全套工具链铺好（采集/自动提议框/组装/微调导出）。
> **下一步只剩：等用户真实照片到位，跑 `tools/finetune_overhead.sh`。**
