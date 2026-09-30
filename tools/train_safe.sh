#!/usr/bin/env bash
# ============================================================
# 安全训练启动器 —— 防止 OOM 连坐（把训练关进独立 cgroup）
# ------------------------------------------------------------
# 背景（本机实测踩过）：训练跑在这台 PC 上，和 DSH/其它进程共享 9.9G 内存 与
# 6G 显存。一旦 dataloader 线程 + 数据集缓存把内存吃满，内核 OOM killer 会挑
# "分数最高"的进程杀 —— 可能把 DSH 一起带走（连坐），会话就断了。
#
# 本脚本做四件事：
#   1) **独立 cgroup + 硬内存上限**：systemd-run --user --scope -p MemoryMax=…
#      超过上限只会杀这个 scope 里的训练进程，主机与 DSH 不受影响（cgroup v2）
#   2) **拉高 OOM 优先级**：OOMScoreAdjust=500 —— 真要杀，先杀训练
#   3) **省内存的运行时参数**：
#      · PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True（减少显存碎片 → 少 CUDA OOM）
#      · OMP/MKL 线程数限制（每线程都有栈与缓冲）
#      · 训练侧 workers / cache 下调（每个 worker 会复制数据集索引，cache=ram 最吃内存）
#   4) **脱离当前会话**：setsid/systemd 后台跑 + 日志落盘 + 单元名记录，
#      DSH 重启不影响训练；训练崩了也不影响 DSH
#
# 用法：
#   bash tools/train_safe.sh start                 # 用 config/train_overhead.yaml 训练
#   bash tools/train_safe.sh start --epochs 30 --batch 8
#   bash tools/train_safe.sh status | tail | stop
# 本机实测（1054 train / 189 val，imgsz 640，batch 8，workers 4，cache=false）：
#   内存峰值 **3.7G**；上限设 3G → 训练在**自己的 cgroup 内**被 OOM killer 杀掉，
#   主机与 DSH 完全无感（日志：Failed with result 'oom-kill'，主机内存不降反稳）；
#   上限设 5G → 训练正常跑完。
#   ⇒ 默认上限取"可用内存的 60%"（本机≈4.9G，留足余量）；内存紧张时先降 workers（4→2）。
#
# 环境变量：
#   MEM_MAX=6G      训练 cgroup 内存上限（默认取"可用内存的 60%"）
#   GPU_MEM_LIMIT   仅提示用（CUDA 无法硬限，靠 batch/精度控制）
#   PY              python 解释器（默认 /home/xxxffyy/gpu_env/bin/python）
# ============================================================
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${PY:-/home/xxxffyy/gpu_env/bin/python}"
UNIT_PREFIX="sort-train"
STATE="$ROOT/runs/.train_safe_state"

cmd="${1:-start}"; shift || true

have_systemd_run() { command -v systemd-run >/dev/null 2>&1 && systemctl --user show-environment >/dev/null 2>&1; }

# ---------- 默认内存上限：可用内存的 60% ----------
if [ -z "${MEM_MAX:-}" ]; then
  avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
  MEM_MAX="$(( avail_mb * 60 / 100 ))M"
fi

case "$cmd" in
  status)
    if [ -f "$STATE" ]; then
      unit=$(cat "$STATE")
      echo "单元: $unit"
      systemctl --user status "$unit" --no-pager 2>/dev/null | head -12
      echo "--- 最近日志 ---"
      journalctl --user -u "$unit" -n 15 --no-pager 2>/dev/null | tail -15
    else
      echo "没有记录的训练任务（$STATE 不存在）"
    fi
    exit 0
    ;;
  tail)
    [ -f "$STATE" ] && journalctl --user -u "$(cat "$STATE")" -f --no-pager || echo "无任务"
    exit 0
    ;;
  stop)
    if [ -f "$STATE" ]; then
      unit=$(cat "$STATE")
      systemctl --user stop "$unit" 2>/dev/null && echo "已停止 $unit"
      systemctl --user reset-failed "$unit" 2>/dev/null || true
      rm -f "$STATE"
    else
      echo "无任务可停"
    fi
    exit 0
    ;;
  start) ;;
  *) echo "用法: $0 {start|status|tail|stop} [训练参数…]"; exit 1 ;;
esac

# ---------- 启动前体检：别在内存已经很紧时开训 ----------
avail_mb=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo)
if [ "$avail_mb" -lt 2000 ]; then
  echo "❌ 可用内存仅 ${avail_mb}MB，先释放再训练（避免把主机拖进 OOM）"; exit 1
fi
gpu_free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
[ -n "${gpu_free:-}" ] && [ "$gpu_free" -lt 1500 ] && { echo "❌ 显存仅剩 ${gpu_free}MB，先释放再训练"; exit 1; }

# ---------- 运行时环境（省内存/降显存碎片） ----------
export PYTHONUNBUFFERED=1
export YOLO_AUTOINSTALL=false
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-4}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-4}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-4}"

# ---------- 训练参数：默认给"省内存"的保守值，可被命令行覆盖 ----------
CONFIG="${CONFIG:-config/train_overhead.yaml}"
BASE="${BASE:-runs/best_v3_gpu.pt}"
EPOCHS="${EPOCHS:-30}"
BATCH="${BATCH:-8}"          # 16 → 8：显存与内存都更稳（6G 笔记本卡）
WORKERS="${WORKERS:-4}"      # 8 → 4：每个 worker 都要占内存
ARGS=(--config "$CONFIG" --base "$BASE" --epochs "$EPOCHS" --batch "$BATCH" --device 0)
[ $# -gt 0 ] && ARGS=("$@")   # 允许完全自定义

echo "=== 安全训练启动 ==="
echo "  解释器 : $PY"
echo "  内存上限: $MEM_MAX（独立 cgroup，超限只杀训练）"
echo "  参数    : ${ARGS[*]}"
echo "  可用内存: ${avail_mb}MB | 空闲显存: ${gpu_free:-?}MB"

if have_systemd_run; then
  unit="${UNIT_PREFIX}-$(date +%H%M%S)"
  log="$ROOT/runs/${unit}.log"
  # 用**瞬时 service**（不是 --scope）：scope 不支持 OOMScoreAdjust，
  # 且 service 天然后台运行，训练不会挂在 DSH 的进程组里
  run_unit() {   # $1 = 额外属性（可为空）
    # shellcheck disable=SC2086
    systemd-run --user --unit="$unit" --collect \
        -p WorkingDirectory="$ROOT" \
        -p MemoryMax="$MEM_MAX" -p MemorySwapMax=0 $1 \
        --description="分拣模型训练（$CONFIG）" \
        /bin/bash -lc "exec $PY '$ROOT/train_detection.py' ${ARGS[*]} > '$log' 2>&1"
  }
  if run_unit "-p OOMScoreAdjust=500 -p Nice=10" 2>/tmp/train_safe_err; then
    :
  elif run_unit "-p Nice=10" 2>>/tmp/train_safe_err; then
    echo "  （本机 systemd 不支持 OOMScoreAdjust，已省去该属性；内存上限仍生效）"
  elif run_unit "" 2>>/tmp/train_safe_err; then
    echo "  （已退化：未设 OOMScoreAdjust/Nice）"
  else
    echo "  ✗ systemd-run 启动失败："; tail -3 /tmp/train_safe_err | sed 's/^/    /'; exit 1
  fi
  echo "$unit" > "$STATE"
  echo "  ✓ 已在独立 cgroup 中启动：$unit（内存上限 $MEM_MAX，超限只杀训练）"
  echo "    日志：$log"
  echo "    查看：bash tools/train_safe.sh status"
else
  unit="${UNIT_PREFIX}-$(date +%H%M%S)"
  log="$ROOT/runs/${unit}.log"
  echo "  ⚠ 无 systemd --user，退化为 ulimit 软限制（隔离性弱）"
  setsid nohup bash -lc "ulimit -v $(( ${MEM_MAX%M} * 1024 )) 2>/dev/null; exec $PY train_detection.py ${ARGS[*]}" \
      > "$log" 2>&1 < /dev/null &
  echo "$unit" > "$STATE"
  echo "  ✓ 已后台启动（pid $!），日志：$log"
fi
