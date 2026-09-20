#!/usr/bin/env bash
# ============================================================
# preflight-jetson.sh —— 构建前的联网体检（DNS / 镜像源 / 磁盘）
# ------------------------------------------------------------
# 为什么需要：Jetson 上第一次 build 最常见的两类失败都不是代码问题：
#   ① DNS 挂了 → "lookup nvcr.io on 127.0.1.1:53: connection refused"
#   ② nvcr.io / docker.io 在国内不可达 → 拉基镜像超时
# 这个脚本把两件事都查出来，并直接打印该怎么修。
#
# 用法：bash scripts/preflight-jetson.sh
# ============================================================
set -u
OK="✅"; NO="❌"; WARN="⚠ "
FAIL=0

echo "=============================================================="
echo " Jetson 构建前体检  $(date '+%F %T')"
echo "=============================================================="

# ---------------- 1) DNS ----------------
echo "[1] DNS 解析"
RESOLV="$(cat /etc/resolv.conf 2>/dev/null | grep -vE '^\s*#|^\s*$' | tr '\n' ' ')"
echo "    /etc/resolv.conf: ${RESOLV:-（空）}"
for h in nvcr.io docker.io registry-1.docker.io; do
  ip="$(python3 -c "import socket;print(socket.gethostbyname('$h'))" 2>/dev/null || true)"
  if [ -n "$ip" ]; then echo "    $OK $h → $ip"; else echo "    $NO $h 解析失败"; FAIL=1; fi
done
if grep -q "127.0.0.1\|127.0.1.1" /etc/resolv.conf 2>/dev/null; then
  echo "    $WARN 本机解析器（127.0.0.1/127.0.1.1）—— systemd-resolved 没起来时 Docker 就解析不了任何域名"
fi
systemctl is-active systemd-resolved >/dev/null 2>&1 \
  && echo "    systemd-resolved: active" \
  || echo "    $WARN systemd-resolved 未运行"

# ---------------- 2) 各 registry 可达性 ----------------
echo
echo "[2] 镜像仓库可达性（HTTP 401/200 都算通，超时/解析失败算不通）"
check_url() {
  local name="$1" url="$2"
  local code
  code="$(curl -s -o /dev/null -m 8 -w '%{http_code}' "$url" 2>/dev/null)"
  [ -n "$code" ] || code=000
  if [ "$code" = "000" ]; then echo "    $NO $name（不可达）"; return 1
  else echo "    $OK $name（HTTP $code）"; return 0; fi
}
check_url "nvcr.io            " "https://nvcr.io/v2/"                              || true
check_url "DaoCloud 多仓库代理 " "https://docker.m.daocloud.io/v2/"                  || true
check_url "docker.io          " "https://registry-1.docker.io/v2/"                   || true
check_url "aliyun pypi        " "https://mirrors.aliyun.com/pypi/simple/"            || true
check_url "aliyun apt         " "https://mirrors.aliyun.com/ubuntu/dists/bionic/"    || true

# ---------------- 3) Docker 守护进程配置 ----------------
echo
echo "[3] Docker 守护进程"
docker version --format '{{.Server.Version}}' >/dev/null 2>&1 \
  && echo "    $OK 守护进程可访问（$(docker version --format '{{.Server.Version}}' 2>/dev/null)）" \
  || { echo "    $NO 访问不到 docker 守护进程"; FAIL=1; }
if [ -f /etc/docker/daemon.json ]; then
  echo "    /etc/docker/daemon.json:"; sed 's/^/      /' /etc/docker/daemon.json
  grep -q registry-mirrors /etc/docker/daemon.json \
    && echo "    $OK 已配置 registry-mirrors" \
    || echo "    $WARN 未配置 registry-mirrors（拉 docker.io 镜像会慢/失败）"
  grep -q '"dns"' /etc/docker/daemon.json \
    && echo "    $OK 已给容器配置 dns" \
    || echo "    $WARN 未给容器配置 dns（宿主 DNS 不稳时容器内解析会失败）"
else
  echo "    $WARN 没有 /etc/docker/daemon.json（未配镜像加速与 DNS）"
fi
echo "    当前生效的 registry-mirrors: $(docker info --format '{{json .RegistryConfig.Mirrors}}' 2>/dev/null || echo '（读不到）')"
# ⚠ 关键：nvidia runtime 是否已注册 —— 现场用 --runtime nvidia 创建的容器（如 ROS2 容器）
#    在缺少注册时会直接报 "Unknown runtime specified nvidia" 起不来。
HAS_NVR="$(command -v nvidia-container-runtime || true)"
REG_NVR="$(docker info --format '{{range .Runtimes}}{{.}} {{end}}' 2>/dev/null | grep -c nvidia || true)"
if [ -n "$HAS_NVR" ] && [ "${REG_NVR:-0}" = "0" ]; then
  echo "    $NO 宿主机装了 nvidia-container-runtime，但 daemon.json 里**没注册 nvidia runtime**"
  echo "       → 用 --runtime nvidia 创建的容器（现场 ROS2 容器）会报 Unknown runtime specified nvidia"
  FAIL=1
elif [ "${REG_NVR:-0}" != "0" ]; then
  echo "    $OK 已注册 nvidia runtime"
fi
# 现场有哪些容器依赖 nvidia runtime
for c in $(docker ps -a --format '{{.Names}}' 2>/dev/null); do
  rt="$(docker inspect "$c" --format '{{.HostConfig.Runtime}}' 2>/dev/null)"
  [ "$rt" = "nvidia" ] && echo "    ⚠ 容器 $c 依赖 nvidia runtime（$([ "${REG_NVR:-0}" != "0" ] && echo 可启动 || echo 现在起不来)）"
done

# ---------------- 4) 磁盘 ----------------
echo
echo "[4] 磁盘"
avail="$(df -BG --output=avail / | tail -1 | tr -dc '0-9')"
if [ "${avail:-0}" -ge 8 ]; then echo "    $OK 根分区可用 ${avail}G（构建需 ≥8G）"
else echo "    $NO 根分区仅剩 ${avail}G，构建会失败（docker system prune -a 或换盘）"; FAIL=1; fi

# ---------------- 5) 结论与修法 ----------------
echo
echo "=============================================================="
if [ "$FAIL" = "0" ]; then
  echo " 结论：体检通过，可以直接 build"
else
  echo " 结论：有问题，按下面修"
fi
echo "=============================================================="
cat <<'FIX'

【DNS 修法】（先看 /etc/resolv.conf；只修一次，重启后可能回退）
  # 方式 A：重启解析器
  sudo systemctl restart systemd-resolved && sudo systemctl enable systemd-resolved
  # 方式 B：直接写公共 DNS（阿里 + 114，国内更快）
  sudo rm -f /etc/resolv.conf
  printf 'nameserver 223.5.5.5\nnameserver 114.114.114.114\n' | sudo tee /etc/resolv.conf
  # 验证
  python3 -c "import socket;print(socket.gethostbyname('nvcr.io'))"

【Docker 守护进程修法】（镜像加速 + DNS + **保留 nvidia runtime**）
  ⚠ 千万不要直接覆盖 daemon.json：现场 ROS2 容器是用 --runtime nvidia 创建的，
    一旦把 runtimes 段删掉，`docker start ros2_arm_container` 会报
    "Unknown runtime specified nvidia" 而起不来（这个坑已经踩过一次）。

  # 先备份
  sudo cp /etc/docker/daemon.json /etc/docker/daemon.json.bak.$(date +%s) 2>/dev/null
  # 再写入完整配置（含 runtimes —— 宿主机有 nvidia-container-runtime 就必须保留）
  sudo tee /etc/docker/daemon.json >/dev/null <<'JSON'
{
  "runtimes": {
    "nvidia": { "path": "/usr/bin/nvidia-container-runtime", "runtimeArgs": [] }
  },
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
  docker info | grep -E "Runtimes|Registry Mirrors" -A3
  # 验证现场容器能起来
  docker start ros2_arm_container && docker ps | grep ros2_arm

【基镜像 nvcr.io 不可达的修法】（不用改代码，改 .env 一行）
  # 在 .env 里换成代理前缀（DaoCloud 支持 nvcr.io 代理）
  BASE_IMAGE=docker.m.daocloud.io/nvcr.io/nvidia/l4t-base:r32.7.1
  # 然后重新构建
  docker-compose -f docker-compose.jetson.yml build sort-yolo

【ROS 镜像同理】（可选，只有用 ros profile 时才需要）
  ROS2_IMAGE=docker.m.daocloud.io/library/ros:foxy-ros-base
  ROS1_IMAGE=docker.m.daocloud.io/library/ros:melodic-ros-base
FIX
