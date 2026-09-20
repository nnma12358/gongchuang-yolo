# ============================================================
# Jetson Nano · 分拣信息显示屏 —— 网关容器
# ------------------------------------------------------------
# 基镜像：NVIDIA L4T r32.7.1（JetPack 4.6.1，aarch64，Python 3.6）
# 容器内容：Python 网关（server/gateway.py）+ 前端静态产物（build/）
#          + 可选自训练 ONNX 检测模型（models/）
#
# 构建（在项目根目录执行，上下文必须是仓库根）：
#   docker build -f deploy/jetson/Dockerfile -t sort-gateway:jetson .
#
# 运行：
#   docker compose -f docker-compose.jetson.yml up -d
#   显示屏浏览器（Jetson 本机 HDMI 屏）：chromium-browser --kiosk http://localhost
# ============================================================
# 基础镜像可用 --build-arg / .env 里的 BASE_IMAGE 换成国内代理，例如：
#   docker.m.daocloud.io/nvcr.io/nvidia/l4t-base:r32.7.1
ARG BASE_IMAGE=nvcr.io/nvidia/l4t-base:r32.7.1
FROM ${BASE_IMAGE}

# 非交互安装：不加这行，apt 会在 tzdata 等处弹 debconf 交互界面，构建就卡住了
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Shanghai

# ---- 系统依赖：Python3 / OpenCV(JetPack 自带 4.1.1) / Pillow / numpy ----
# 国内网络：先切阿里云 apt 镜像（L4T 基础镜像默认走 archive.ubuntu.com，国内常超时）
RUN set -e; \
    if [ -f /etc/apt/sources.list ]; then \
        cp /etc/apt/sources.list /etc/apt/sources.list.bak; \
        sed -i -e 's@http://ports.ubuntu.com/@http://mirrors.aliyun.com/@g' \
               -e 's@http://archive.ubuntu.com/@http://mirrors.aliyun.com/@g' \
               -e 's@http://security.ubuntu.com/@http://mirrors.aliyun.com/@g' \
               /etc/apt/sources.list || true; \
    fi; \
    apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-dev build-essential \
        python3-numpy python3-opencv python3-pil \
        curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- 升级 pip/setuptools（L4T 镜像自带 pip 9，装不了 manylinux2014 轮子） ----
RUN python3 -m pip install --no-cache-dir --upgrade \
        -i https://mirrors.aliyun.com/pypi/simple/ "pip<22" "setuptools<60" wheel \
    && rm -rf /root/.cache/pip/*

# ---- Python 依赖（固定为兼容 Python 3.6 的版本，见 requirements.txt 注释） ----
COPY deploy/jetson/requirements.txt /app/requirements.txt
RUN pip3 install --no-cache-dir \
        -i https://mirrors.aliyun.com/pypi/simple/ \
        -r /app/requirements.txt \
    && rm -rf /root/.cache/pip/*

# ---- 网关脚本 + 公共模块 + 前端 SPA 产物 + 机器人侧客户端脚本 + 检测模型 ----
COPY server/catalog.py server/detect_core.py server/gateway.py /app/
COPY scripts/ /app/scripts/
COPY build/ /app/build/
COPY models/ /app/models/

ENV STATIC_DIR=/app/build
ENV DATA_DIR=/app/data
ENV MODEL_PATH=/app/models/detect/goods_yolov8n_640_fp32.onnx
# classic = 颜色+形状+污渍+二维码（纯 OpenCV，无需模型）
# onnx    = 使用自训练检测模型（models/detect/ 下放 ONNX，见 models/README.md）
ENV DETECT_ENGINE=classic
# 一件货物分拣完成后，分拣信息在屏显示的最短保持时间（秒）
# 期间网关拒绝下一件分拣上报 —— 对应赛项“显示后再分拣下一件，否则不计分”
ENV DISPLAY_HOLD_SECONDS=3.0
# 视觉容器地址（与 sort-vision 通过 host 网络互通）
ENV VISION_URL=http://127.0.0.1:8100
# 机器人执行器（ROS 桥接）地址；留空则不驱动机械臂，仅显示与记录
ENV ROBOT_URL=
# ---- 自动分拣循环 ----
# 轮询视觉容器间隔(秒)
ENV AUTO_INTERVAL=0.6
# 连续 N 帧同类才登记（防误检）
ENV AUTO_CONFIRM_FRAMES=4
# 置信度下限
ENV AUTO_CONF_MIN=0.6
# 一次一件（赛项）
ENV AUTO_SINGLE_ITEM=1
# 1=不实际驱动机械臂（联调）；现场置 0
ENV AUTO_DRY_RUN=1
ENV PORT=80

EXPOSE 80

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -fsS http://localhost:80/health || exit 1

CMD ["python3", "gateway.py"]
