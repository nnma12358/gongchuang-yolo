#!/usr/bin/env bash
# ============================================================
# pack-jetson-bundle.sh —— 打一个"拷到 Jetson 解压即可构建"的离线包
# ------------------------------------------------------------
# 产出（默认放工作区上级目录）：
#   sort-jetson-deploy-<版本>.tar.gz    部署包：compose + Dockerfile + server + models + 前端产物
#   sort-toolchain-<版本>.tar.gz        工具包：训练/评估/复核脚本（PC 端用，不含数据集）
#
# 用法：
#   bash scripts/pack-jetson-bundle.sh                 # 默认版本=日期
#   bash scripts/pack-jetson-bundle.sh --version v1.0  # 指定版本
#   bash scripts/pack-jetson-bundle.sh --out /media/usb
#
# 设计说明：
#   · 只装"构建容器必需"的东西：models/ 与 build/ 绝不能漏（Dockerfile 里 COPY）
#   · 自动校验：compose 引用的文件是否齐全、模型是否就位、前端产物是否是静态 SPA
#   · 生成 MD5SUMS 与 MANIFEST，方便对拷传输后核对
# ============================================================
set -u
VERSION=""
OUTDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --out) OUTDIR="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done
VERSION=${VERSION:-$(date +%Y%m%d)}
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
WORK="${ROOT}/.."                        # 工作区根（工创/）
[ -n "$OUTDIR" ] || OUTDIR="$WORK"
cd "$ROOT"

echo "=== [1/5] 检查构建必需内容 ==="
[ -d build ] && [ -f build/index.html ] || { echo "❌ build/ 不存在：先跑 ADAPTER=static npm run build:static"; exit 1; }
if ! grep -q 'adapter-static\|_app' build/index.html && [ ! -d build/_app ]; then
  echo "❌ build/ 看起来不是静态 SPA 产物（缺 _app/）"; exit 1
fi
for f in models/detect/goods_yolov8n_640_fp32.onnx models/detect/classes.json \
         models/attr/color.onnx models/attr/color_classes.json \
         models/attr/shape.onnx models/attr/shape_classes.json \
         models/attr/stain.onnx models/attr/stain_classes.json; do
  [ -f "$f" ] || { echo "❌ 缺模型文件: $f"; exit 1; }
done
echo "  ✅ 前端静态产物 + 8 个模型文件就位"

STAGE="$OUTDIR/sort-jetson-deploy-$VERSION"
rm -rf "$STAGE"; mkdir -p "$STAGE"/{deploy/jetson,server,scripts,models/detect,models/attr,calibration,.ros2_ws,src,static}

echo "=== [2/5] 拷贝部署包 ==="
cp -a docker-compose.jetson.yml docker-compose.jetson-devices.yml "$STAGE"/
cp -a .dockerignore "$STAGE"/ 2>/dev/null || true
cp -a deploy/jetson/. "$STAGE"/deploy/jetson/
cp -a server/. "$STAGE"/server/
cp -a scripts/. "$STAGE"/scripts/
cp -a models/. "$STAGE"/models/
cp -a build "$STAGE"/build
# 前端源码（现场要改页面时用得上；不参与镜像构建）
cp -a src/. "$STAGE"/src/
cp -a static/. "$STAGE"/static/ 2>/dev/null || true
for f in package.json package-lock.json svelte.config.js vite.config.js; do
  [ -e "$f" ] && cp -a "$f" "$STAGE"/ || true
done
# 清掉不该进包的东西
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true
find "$STAGE" -name "*.pyc" -delete 2>/dev/null || true
touch "$STAGE/calibration/.gitkeep" "$STAGE/.ros2_ws/.gitkeep"

echo "=== [3/5] 写 .env 模板与说明 ==="
cat > "$STAGE/.env.jetson" <<'ENVEOF'
# ============================================================
# Jetson 现场配置：复制为 .env 后按现场改（docker compose 自动读取）
#   cp .env.jetson .env
# ============================================================

# ---- 基础镜像（国内 nvcr.io 常不可达，换成代理前缀即可，不用改代码）----
# 先跑 bash scripts/preflight-jetson.sh 体检，它会告诉你哪个源通
BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.7.1
# BASE_IMAGE=docker.m.daocloud.io/nvcr.io/nvidia/l4t-base:r32.7.1
# ROS 镜像（只有用 --profile ros / ros1 时才需要）
ROS2_IMAGE=ros:foxy-ros-base
ROS1_IMAGE=ros:melodic-ros-base

# ---- 相机 ----
CAMERA_SOURCE=auto            # auto | synthetic | index | url
CAMERA_INDEX=0
CAMERA_WIDTH=1280
CAMERA_HEIGHT=720

# ---- 视觉三层：yolo 检测 + cnn 属性 ----
DETECT_ENGINE=yolo            # yolo=三层链路（失败自动回退经典引擎）；classic=纯经典 CV
YOLO_IMGSZ=640                # 必须与训练一致（本模型 640）
YOLO_CONF=0.35                # 实测：0.35~0.55 误检均为 0/图
YOLO_IOU=0.45

# ---- Jetson 推理加速（三条路径，自动降级，不会崩）----
# ① TensorRT FP16（最快）：先在设备上 bash scripts/build-trt-on-jetson.sh 构建引擎，
#    再 bash scripts/gen-trt-override.sh 生成叠加文件，然后：
#      docker-compose -f docker-compose.jetson.yml -f docker-compose.jetson-trt.yml up -d sort-yolo
#    ⚠ Jetson Nano 的 Maxwell GPU 没有 INT8 硬件 → 用 FP16，不要用 INT8
# ② OpenCV CUDA FP16（零额外依赖，JetPack 自带 OpenCV 带 CUDA）：DNN_BACKEND=cuda
# ③ OpenCV CPU（兜底）：DNN_BACKEND=cpu 或不设且无 CUDA
DNN_BACKEND=auto              # auto | cuda | cpu

# ---- 托盘 ROI（防人手/机械臂/杂物被当成货物）----
# 留空=不启用。按标定好的托盘范围填，例如四周留 6% 边距：
# ROI=0.06,0.06,0.94,0.94
ROI=

# ---- 自动分拣循环 ----
AUTO_INTERVAL=0.6             # 循环间隔（秒）
AUTO_CONFIRM_FRAMES=4         # 连续 N 帧识别一致才动作（防抖）
AUTO_CONF_MIN=0.6
AUTO_SINGLE_ITEM=1            # 赛项：一次只分拣一件
AUTO_DRY_RUN=1                # ⚠ 1=只算不动机器人，现场确认无误后再改 0

# ---- 机械臂（ROS 2 桥）----
ROBOT_URL=http://127.0.0.1:8120
ARM_SERVICE=/arm_pick_place
BRIDGE_PORT=8120
ROS_DOMAIN_ID=0
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# ---- 深度 / 2.5D 定位（可选）----
DEPTH_URL=http://127.0.0.1:8123/depth.png
DEPTH_HTTP_PORT=8123
CALIBRATION_DIR=./calibration  # 放 table_reference.npz 等标定产物
GRASP_ENABLED=1
DEFAULT_PICK_Z=20.0

# ---- 屏显 ----
DISPLAY_HOLD_SECONDS=3.0       # 分拣信息在屏最短保持时间（赛项"显示后再下一件"）
ENVEOF

cat > "$STAGE/README-JETSON.md" <<'DOCEOF'
# Jetson 离线部署包 —— 解压即构建

> 这个包**自带全部模型和前端产物**，不需要联网 npm/pip 之外的任何东西，
> 也不需要 PC 上的训练工程。

## 0. 前提

- Jetson 已装 **JetPack 4.6.x**（L4T r32.7.x，aarch64）与 Docker + docker compose
- 磁盘剩余 ≥ 8 GB（4 个镜像 + 构建缓存）
- 首次构建需要能访问 aliyun 镜像源（Dockerfile 里已把 apt/pip 源换成阿里云）

```bash
docker --version && docker compose version     # 确认 compose v2 可用
ls /dev/video*                                  # 确认相机设备存在（没有也能跑合成画面）
```

## 1. 解压

```bash
tar -xzf sort-jetson-deploy-<版本>.tar.gz
cd sort-jetson-deploy-<版本>
cp .env.jetson .env          # 按现场改 .env（至少确认 CAMERA_SOURCE 与 AUTO_DRY_RUN）

# 起服务前先体检（Jetson 上装的是 docker-compose v1，对重复项是硬报错）
docker-compose -f docker-compose.jetson.yml config > /dev/null && echo "compose OK"
python3 scripts/check-compose.py docker-compose.jetson.yml   # 同类问题提前查
```

> 命令用 `docker-compose`（v1，JetPack 自带）；若你装了 v2 插件，`docker compose` 等价。
> `docker-compose config` 里出现 "deploy key will be ignored" 的**警告是正常的**（CPU/内存限制只在 swarm 生效），不影响运行。

## 1.5 构建前体检（强烈建议，能省掉 90% 的失败）

```bash
bash scripts/preflight-jetson.sh
```

它会查：DNS 是否可用、nvcr.io / docker.io / 阿里云源是否可达、Docker 是否配了镜像加速与 DNS、
磁盘是否够 8G，并在最后**直接打印修法**。两类最常见的失败：

| 报错 | 原因 | 修法 |
|---|---|---|
| `lookup nvcr.io on 127.0.1.1:53: connection refused` | 宿主 DNS 解析器（systemd-resolved）没工作 | `sudo systemctl restart systemd-resolved`，或把 `/etc/resolv.conf` 写成 `nameserver 223.5.5.5` |
| 拉基镜像超时 / `Get https://nvcr.io/v2/: ...` | nvcr.io 国内不可达 | `.env` 里 `BASE_IMAGE=docker.m.daocloud.io/nvcr.io/nvidia/l4t-base:r32.7.1` |

给 Docker 配镜像加速与 DNS（一次配好，之后所有镜像都受益）：

```bash
sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "dns": ["223.5.5.5", "114.114.114.114"],
  "registry-mirrors": [
    "https://docker.m.daocloud.io",
    "https://docker.1ms.run",
    "https://docker.xuanyuan.me"
  ],
  "log-driver": "json-file",
  "log-opts": { "max-size": "20m", "max-file": "3" }
}
JSON
sudo systemctl restart docker
docker info | grep -A3 "Registry Mirrors"
```

## 2. 构建 + 启动

```bash
# 只装 4 个核心容器：网关(Python+前端) / 视觉 / YOLO 检测 / CNN 属性
docker-compose -f docker-compose.jetson.yml up -d --build \
    sort-gateway sort-vision sort-yolo sort-cnn

# 需要 ROS 2 桥 + 深度节点（Astra 相机）时再加：
docker-compose -f docker-compose.jetson.yml --profile ros up -d

# 有 USB 相机时用这个叠加文件（直接挂 /dev/video0）：
docker-compose -f docker-compose.jetson.yml -f docker-compose.jetson-devices.yml up -d --build
```

首次构建约 15~30 分钟（Nano 上编译 numpy/opencv 绑定较慢），之后重建只走缓存。

## 3. 自检（必须全绿再上场）

```bash
curl -s http://localhost:8101/health    # YOLO 检测：ok:true, model:best.onnx, conf_thres:0.35
curl -s http://localhost:8102/health    # CNN 属性：已加载 color/shape/stain
curl -s http://localhost:8100/health    # 视觉：engine=yolo(+cnn)，相机 connected
curl -s http://localhost:8100/detect/frame | head -c 600   # 看 detections / roi_dropped
curl -s http://localhost/health         # 网关 + 前端
```

浏览器打开 `http://<jetson-ip>/` 应看到分拣显示界面；HDMI 屏可用
`chromium-browser --kiosk http://localhost`。

**判定线**：`/detect/frame` 的 `engine` 必须是 `yolo+cnn`；随便放一件货物，
`detections` 里应出现 1 条、`conf` ≥ 0.8。若 `engine` 退化成 `classic`，
说明 yolo/cnn 容器没起来，先看 `docker logs sort-yolo`。

## 3.5 把推理跑到最优（Jetson Nano）

Nano 的瓶颈几乎全在检测前向。三条路径**自动降级**，`/health` 的 `engine` 与 `dnn_backend` 会告诉你实际走哪条：

| 路径 | 怎么做 | 预期 |
|---|---|---|
| ① **TensorRT FP16**（推荐） | `bash scripts/build-trt-on-jetson.sh` → `bash scripts/gen-trt-override.sh` → 用叠加文件启动 | 最快；比 OpenCV CPU 快约一个数量级 |
| ② OpenCV CUDA FP16 | `.env` 里 `DNN_BACKEND=cuda` | 零额外依赖，JetPack 自带 OpenCV 带 CUDA |
| ③ OpenCV CPU | 默认兜底 | 一定能跑，但最慢 |

**为什么是 FP16 而不是 INT8**：Jetson Nano 是 Tegra X1（Maxwell，SM 5.3），**没有 INT8 张量核、
也没有 DP4A 指令** —— TensorRT 的 INT8 在这块芯片上只能用 FP16 模拟，基本不提速还掉精度。
只有 Xavier(SM 7.2)/Orin(SM 8.7) 才值得做 INT8。构建脚本会自动识别芯片并给建议。

引擎与 GPU 架构 + TensorRT 版本绑定，**必须在 Jetson 本机构建**（PC 上构建的拷过去加载不了）。

## 4. 现场调优

| 现象 | 改哪里 |
|---|---|
| 误检（背景/人手被当货物） | 先填 `ROI=0.06,0.06,0.94,0.94`；仍多则补拍照片走负样本重训 |
| 漏检小目标 | `YOLO_IMGSZ=960` 试（未验证配置，需重新评估） |
| 抖动/重复触发 | 调大 `AUTO_CONFIRM_FRAMES`（默认 4 帧） |
| 机器人不动 | `AUTO_DRY_RUN=1` 时不会动；确认后改 0 |

## 5. 换模型（不用重建镜像）

`models/` 是只读挂载进容器的，替换文件后重启对应容器即可：

```bash
cp 新的 ONNX models/detect/goods_yolov8n_640_fp32.onnx
docker-compose -f docker-compose.jetson.yml restart sort-yolo sort-vision
```

## 6. 目录说明

```
docker-compose.jetson.yml         核心编排（网关/视觉/YOLO/CNN + ros profile）
docker-compose.jetson-devices.yml 叠加：挂 /dev/video* 相机设备
deploy/jetson/                    4 个 Dockerfile + requirements + ROS 桥/深度节点脚本
scripts/build-trt-on-jetson.sh    ★ 在设备上构建 TensorRT FP16 引擎 + 实测 + 部署
scripts/gen-trt-override.sh       ★ 按设备真实路径生成 TRT 运行时 compose 叠加文件
scripts/preflight-jetson.sh       ★ 构建前体检：DNS / 镜像源可达性 / Docker 配置 / 磁盘
server/                           gateway.py vision_server.py yolo_server.py cnn_server.py
                                  detect_core.py pose.py catalog.py …
models/detect/goods_yolov8n_640_fp32.onnx   ✅ 检测模型（YOLOv8n 单类 goods）
models/attr/{color,shape,stain}.onnx        ✅ 属性 CNN（颜色/形状/表面）
build/                            ✅ 前端静态产物（网关容器 COPY 进镜像）
src/ static/ package*.json        前端源码（现场要改页面时：ADAPTER=static npm run build:static）
calibration/                      2.5D 定位标定产物放这里（table_reference.npz）
.env.jetson                       配置模板（cp 成 .env）
MD5SUMS                           传输后核对：md5sum -c MD5SUMS
```

## 7. 传输

```bash
# PC 上
scp sort-jetson-deploy-<版本>.tar.gz nvidia@<jetson-ip>:~/
# Jetson 上核对完整性（可选，但建议）
tar -xzf sort-jetson-deploy-<版本>.tar.gz && cd sort-jetson-deploy-<版本> && md5sum -c MD5SUMS
```
DOCEOF

echo "=== [4/5] 生成清单与校验和 ==="
( cd "$STAGE" && find . -type f ! -name MD5SUMS ! -name MANIFEST.txt -print0 \
    | sort -z | xargs -0 md5sum > MD5SUMS )
{
  echo "打包时间: $(date '+%F %T')"
  echo "版本: $VERSION"
  echo "文件数: $(find "$STAGE" -type f | wc -l)"
  echo "大小:   $(du -sh "$STAGE" | cut -f1)"
  echo
  echo "模型文件:"
  ( cd "$STAGE" && md5sum models/detect/*.onnx models/attr/*.onnx )
  echo
  echo "目录树(2 层):"
  ( cd "$STAGE" && find . -maxdepth 2 -type d | sort )
} > "$STAGE/MANIFEST.txt"

echo "=== [4.3/5] 校验：Dockerfile 对 Jetson 老解析器是否合法 ==="
"${PY:-python3}" - "$STAGE" <<'PYEOF'
import glob, os, re, sys
root = sys.argv[1]
bad = []
for f in glob.glob(os.path.join(root, "deploy/jetson/Dockerfile*")) + [os.path.join(root, "Dockerfile")]:
    if not os.path.exists(f):
        continue
    txt = open(f, encoding="utf-8").read().splitlines()
    for i, line in enumerate(txt, 1):
        # ① ENV/LABEL 行尾注释：老解析器按 name=value 逐词解析 → "can't find = in \"#\""
        if re.match(r'^\s*(ENV|LABEL)\s+.*\s+#', line):
            bad.append("%s:%d ENV/LABEL 行尾注释（Jetson 老解析器会报 can't find = in \"#\"）" % (os.path.basename(f), i))
        # ② BuildKit 专属语法
        if re.match(r'^\s*RUN\s+--', line):
            bad.append("%s:%d RUN --mount/--network 需 BuildKit" % (os.path.basename(f), i))
        if re.match(r'^\s*#\s*syntax=', line):
            bad.append("%s:%d # syntax= 指令需 BuildKit" % (os.path.basename(f), i))
        if re.search(r'<<-?[A-Za-z_]+\s*$', line):
            bad.append("%s:%d heredoc 需 BuildKit" % (os.path.basename(f), i))
        if re.search(r'COPY\s+--(chmod|link|parents)', line):
            bad.append("%s:%d COPY --chmod/--link 需 BuildKit" % (os.path.basename(f), i))
if bad:
    print("  ❌ Dockerfile 与老解析器不兼容：")
    for b in bad:
        print("     -", b)
    sys.exit(1)
missing = [os.path.basename(f) for f in glob.glob(os.path.join(root, "deploy/jetson/Dockerfile*"))
           if "ARG BASE_IMAGE" not in open(f, encoding="utf-8").read()]
if missing:
    print("  ❌ 这些 Dockerfile 没做成可换基镜像（缺 ARG BASE_IMAGE）：%s" % missing)
    sys.exit(1)
print("  ✅ Dockerfile 兼容经典解析器，且基础镜像可用 BASE_IMAGE 覆盖（国内可换代理源）")
PYEOF
[ $? -eq 0 ] || { echo "包不完整，已中止"; exit 1; }

echo "=== [4.4/5] 校验：compose 对 docker-compose v1 是否合法 ==="
"${PY:-python3}" scripts/check-compose.py \
    docker-compose.jetson.yml docker-compose.jetson-devices.yml || {
  echo "  ⚠ compose 有问题（Jetson 上装的是 v1，会直接拒绝启动）"; exit 1; }

echo "=== [4.5/5] 校验：构建所需文件是否齐全 ==="
"${PY:-python3}" - "$STAGE" <<'PYEOF'
import os, re, sys, yaml
root = os.path.abspath(sys.argv[1])
errs, n = [], 0
for cf in ("docker-compose.jetson.yml", "docker-compose.jetson-devices.yml"):
    p = os.path.join(root, cf)
    if not os.path.exists(p):
        errs.append("缺 %s" % cf); continue
    for name, svc in (yaml.safe_load(open(p, encoding="utf-8")) or {}).get("services", {}).items():
        b = svc.get("build")
        if b:
            ctx = b.get("context", ".") if isinstance(b, dict) else b
            df = b.get("dockerfile", "Dockerfile") if isinstance(b, dict) else "Dockerfile"
            dfp = os.path.join(root, ctx, df)
            if not os.path.exists(dfp):
                errs.append("[%s] 缺 Dockerfile %s" % (name, df)); continue
            n += 1
            for line in open(dfp, encoding="utf-8"):
                m = re.match(r"^\s*(COPY|ADD)\s+(.+)$", line)
                if not m or "--from=" in line:
                    continue
                parts = [x for x in m.group(2).split() if not x.startswith("--")]
                for s2 in parts[:-1]:
                    t = os.path.join(root, ctx, s2.rstrip("/"))
                    if not os.path.exists(t):
                        errs.append("[%s] %s COPY 的 %s 缺失" % (name, df, s2))
                    else:
                        n += 1
        for v in svc.get("volumes", []) or []:
            if isinstance(v, str) and v.startswith("./"):
                src = v.split(":")[0]
                if not os.path.exists(os.path.join(root, src)):
                    errs.append("[%s] 挂载源缺失 %s" % (name, src))
                else:
                    n += 1
if errs:
    print("  ❌ 校验失败："); [print("    -", e) for e in errs]; sys.exit(1)
print("  ✅ 校验通过：%d 项构建依赖全部就位" % n)
PYEOF
[ $? -eq 0 ] || { echo "包不完整，已中止"; exit 1; }

echo "=== [5/5] 打 tar.gz ==="
TARBALL="$OUTDIR/sort-jetson-deploy-$VERSION.tar.gz"
tar -czf "$TARBALL" -C "$OUTDIR" "sort-jetson-deploy-$VERSION"
echo "✅ $TARBALL  ($(du -h "$TARBALL" | cut -f1))"
echo "   解压目录: $(du -sh "$STAGE" | cut -f1)  ·  文件 $(find "$STAGE" -type f | wc -l) 个"
echo
echo "下一步（可选）工具包：bash scripts/pack-toolchain.sh --version $VERSION"
