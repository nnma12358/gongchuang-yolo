#!/usr/bin/env bash
# ============================================================
# adopt-existing-ros2.sh —— 把现场已跑通的 ROS2 容器"编排进来"（自动探测 + 生成配置）
# ------------------------------------------------------------
# 现场往往已经有一个跑通的 ROS2 容器（如 ros2_arm_container / wheeltec_ros2_astra:foxy），
# 里面跑着机械臂驱动、Astra 相机、MoveIt 等。我们**不重新拉 ros:foxy-ros-base**，
# 只在那个镜像上加一层薄层做桥接 —— 但前提是 ROS 2 的"域号 + RMW 实现 + 网络模式"三者对齐，
# 否则两个容器互相发现不了，表现为"话题列表是空的"。
#
# 本脚本读 docker inspect 把这三样自动抠出来，写进 .env，并给出验证命令。
#
# 用法：
#   bash scripts/adopt-existing-ros2.sh                      # 默认容器名 ros2_arm_container
#   bash scripts/adopt-existing-ros2.sh my_ros_container
#   bash scripts/adopt-existing-ros2.sh --dry-run            # 只看探测结果，不改 .env
# ============================================================
set -u
DRY=0
NAME="ros2_arm_container"
for a in "$@"; do
  case "$a" in
    --dry-run) DRY=1 ;;
    -*) echo "未知参数: $a"; exit 1 ;;
    *) NAME="$a" ;;
  esac
done

echo "=============================================================="
echo " 探测现场 ROS2 容器：$NAME"
echo "=============================================================="
if ! docker inspect "$NAME" >/dev/null 2>&1; then
  echo "❌ 找不到容器 $NAME。列出正在运行的容器："
  docker ps --format '  {{.Names}}\t{{.Image}}\t{{.Status}}'
  echo
  echo "把正确的名字作为参数再跑一次： bash scripts/adopt-existing-ros2.sh <容器名>"
  exit 1
fi

# ---- 抓取关键信息（不依赖 jq，用 python3 解析）----
python3 - "$NAME" > /tmp/.ros_probe.sh <<'PYEOF'
import json, subprocess, sys
name = sys.argv[1]
c = json.loads(subprocess.check_output(["docker", "inspect", name]))[0]
cfg = c["Config"]
env = dict(e.split("=", 1) for e in (cfg.get("Env") or []) if "=" in e)
net = (c.get("HostConfig", {}).get("NetworkMode") or "")
print("# 探测结果")
print("IMAGE=%s" % cfg.get("Image", ""))
print("NETWORK_MODE=%s" % net)
for k in ("ROS_DOMAIN_ID", "RMW_IMPLEMENTATION", "CYCLONEDDS_URI", "ROS_LOCALHOST_ONLY",
          "ROS_DISTRO", "ROS_MASTER_URI", "FASTRTPS_DEFAULT_PROFILES_FILE"):
    v = env.get(k, "")
    print('%s="%s"' % (k.replace("-", "_"), v))
PYEOF
. /tmp/.ros_probe.sh

echo "[镜像]        ${IMAGE}"
echo "[网络模式]    ${NETWORK_MODE:-（默认 bridge）}"
echo "[ROS_DOMAIN_ID] ${ROS_DOMAIN_ID:-（未设置 → 默认 0）}"
echo "[RMW]          ${RMW_IMPLEMENTATION:-（未设置 → 用发行版默认，Foxy 是 rmw_fastrtps_cpp）}"
echo "[CYCLONEDDS_URI] ${CYCLONEDDS_URI:-（未设置）}"
echo "[ROS_LOCALHOST_ONLY] ${ROS_LOCALHOST_ONLY:-（未设置）}"

# ---- 判断网络模式该怎么写 ----
if [ "$NETWORK_MODE" = "host" ]; then
  NET_OURS="host"
  NET_NOTE="现场容器用 host 网络 → 桥接容器也用 host，DDS 在同一网络里互相可见 ✅"
else
  NET_OURS="container:$NAME"
  NET_NOTE="现场容器用的是 ${NETWORK_MODE}（非 host）→ 桥接容器**共享它的网络命名空间**最稳"
fi

# ---- CycloneDDS 配置文件：现场如果真的用了，把它导出到我们的目录 ----
CYCL_FILE="./deploy/jetson/cyclonedds.xml"
URI_SRC="${CYCLONEDDS_URI:-}"
URI_OURS=""
if [ -n "$URI_SRC" ]; then
  SRC="${URI_SRC#file://}"
  if [ "$DRY" = "0" ] && docker cp "$NAME:$SRC" "$CYCL_FILE.field" >/dev/null 2>&1; then
    CYCL_FILE="$CYCL_FILE.field"
    echo "[配置] 已从现场容器导出 CycloneDDS 配置（$SRC）→ $CYCL_FILE"
  else
    echo "[配置] ⚠ 现场使用 CycloneDDS 配置 $SRC（导出失败请手动 docker cp）"
  fi
  # 关键：URI 要指到我们**挂载进去的目标路径**，不能照抄现场的 /root/xxx
  URI_OURS="file:///app/cyclonedds.xml"
  echo "[配置] CYCLONEDDS_URI 将改写成 $URI_OURS（挂载目标路径）"
fi

echo
echo "=============================================================="
echo " 将要写入 .env 的内容"
echo "=============================================================="
cat <<EOF
ROS_IMAGE=${IMAGE}
ROS_DOMAIN_ID=${ROS_DOMAIN_ID:-0}
RMW_IMPLEMENTATION=${RMW_IMPLEMENTATION}
CYCLONEDDS_URI=${URI_OURS}
CYCLONEDDS_FILE=${CYCL_FILE}
ROS_NETWORK_MODE=${NET_OURS}
ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-0}
EOF
echo
echo "说明：$NET_NOTE"

if [ "$DRY" = "1" ]; then
  echo
  echo "（--dry-run：没有改 .env）"
  exit 0
fi

# ---- 写进 .env（存在则替换，不存在则追加）----
[ -f .env ] || cp .env.jetson .env 2>/dev/null || touch .env
python3 - "$IMAGE" "${ROS_DOMAIN_ID:-0}" "${RMW_IMPLEMENTATION:-}" "$URI_OURS" \
         "$CYCL_FILE" "$NET_OURS" "${ROS_LOCALHOST_ONLY:-0}" <<'PYEOF'
import sys
img, dom, rmw, uri, cycl, net, lh = sys.argv[1:8]
vals = {"ROS_IMAGE": img, "ROS_DOMAIN_ID": dom, "RMW_IMPLEMENTATION": rmw,
        "CYCLONEDDS_URI": uri, "CYCLONEDDS_FILE": cycl,
        "ROS_NETWORK_MODE": net, "ROS_LOCALHOST_ONLY": lh}
import re as _re
lines = open(".env", encoding="utf-8").read().splitlines()
# 把「空值 + 行尾注释」拆成两行：docker-compose 会把注释当成值
_fixed = []
for _l in lines:
    _m = _re.match(r'^([A-Za-z_][A-Za-z0-9_]*)=(\s*)#\s*(.*)$', _l)
    if _m:
        _fixed.append('# ' + _m.group(3))
        _fixed.append('%s=' % _m.group(1))
    else:
        _fixed.append(_l)
lines = _fixed
out, done = [], set()
for line in lines:
    k = line.split("=", 1)[0].strip()
    if k in vals:
        out.append("%s=%s" % (k, vals[k])); done.add(k)
    else:
        out.append(line)
if done:
    out.append("")
    out.append("# 以下由 scripts/adopt-existing-ros2.sh 自动追加")
for k, v in vals.items():
    if k not in done:
        out.append("%s=%s" % (k, v))
open(".env", "w", encoding="utf-8").write("\n".join(out) + "\n")
print("✅ 已更新 .env（%d 项直接替换，%d 项追加）" % (len(done), len(vals) - len(done)))
PYEOF

cat <<'NEXT'

==============================================================
 下一步
==============================================================
 # 1) 构建桥接容器（**基镜像是现场镜像，不联网拉取**）
 docker-compose -f docker-compose.jetson.yml build ros2

 # 2) 启动（只需桥接容器；现场那套 ROS2 保持不动）
 docker-compose -f docker-compose.jetson.yml up -d ros2

 # 3) 验证两个容器在同一 DDS 域里互相可见（在自己容器里看到现场的话题才算通）
 docker exec sort-ros-bridge bash -lc 'source /opt/ros/$ROS_DISTRO/setup.bash 2>/dev/null; ros2 topic list'
 #    应能看到现场机械臂/相机的话题（如 /arm/... 、/overhead_camera/...）

 # 4) 验证 HTTP 桥接
 curl -s http://localhost:8120/health
 curl -s -o /dev/null -w '%{http_code}\n' http://localhost:8123/depth.png

 # 若话题列表是空的 → 域号或 RMW 不一致：重跑本脚本（它会重新探测），
 # 或手动确认 `docker exec <现场容器> env | grep -E 'ROS_DOMAIN_ID|RMW'`
==============================================================
NEXT
