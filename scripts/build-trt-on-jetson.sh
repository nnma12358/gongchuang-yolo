#!/usr/bin/env bash
# ============================================================
# build-trt-on-jetson.sh —— 在 Jetson 上构建 TensorRT 引擎（并在设备上实测对比）
# ------------------------------------------------------------
# 关键结论（先说）：
#   · **Jetson Nano（Tegra X1 / Maxwell / SM 5.3）没有 INT8 硬件**（无 DP4A、无张量核），
#     TensorRT 的 INT8 在这块芯片上只能靠 FP16 模拟 → 基本不比 FP16 快，还掉精度。
#     **Nano 上的最优目标是 FP16。**
#   · Xavier / Orin（SM ≥7.2）才有真正的 INT8 张量核，那时 INT8 才值得做。
#   · 引擎**必须在设备上构建**（TRT 引擎与 GPU 架构 + TRT 版本绑定，PC 上构建的拷过去用不了）。
#
# 用法（在 Jetson 上、项目根目录执行）：
#   bash scripts/build-trt-on-jetson.sh --check            # 只体检：平台/TRT/依赖/现有模型
#   bash scripts/build-trt-on-jetson.sh                    # 构建 FP16 引擎 + 实测 + 部署
#   bash scripts/build-trt-on-jetson.sh --int8             # 额外构建 INT8（校准图，仅 SM≥7.2 建议）
#   bash scripts/build-trt-on-jetson.sh --imgsz 640 --calib 200
#
# 产出：models/detect/goods_yolov8n_640_fp16.engine（+ 可选 _int8.engine）
#       并打印需要写进 .env 的内容
# ============================================================
set -u
ONNX=""
OUT_DIR="models/detect"
IMGSZ=640
CALIB_DIR=""
CALIB_N=200
WORKSPACE="512"          # MiB；Nano 只有 4GB 内存，构建期别开太大
BENCH_ITERS=200
DO_INT8=0
CHECK_ONLY=0
while [ $# -gt 0 ]; do
  case "$1" in
    --onnx) ONNX="$2"; shift 2 ;;
    --imgsz) IMGSZ="$2"; shift 2 ;;
    --calib-dir) CALIB_DIR="$2"; shift 2 ;;
    --calib) CALIB_N="$2"; shift 2 ;;
    --workspace) WORKSPACE="$2"; shift 2 ;;
    --int8) DO_INT8=1; shift ;;
    --check) CHECK_ONLY=1; shift ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 1
[ -n "$ONNX" ] || ONNX="$OUT_DIR/goods_yolov8n_640_fp32.onnx"

echo "=============================================================="
echo " Jetson 端 TensorRT 引擎构建"
echo "=============================================================="

# ---------------- 1) 平台体检 ----------------
MODEL_TXT="$(cat /proc/device-tree/model 2>/dev/null | tr -d '\0')"
SOC="$(cat /sys/devices/soc0/machine 2>/dev/null || echo 未知)"
L4T="$(head -1 /etc/nv_tegra_release 2>/dev/null || echo 未知)"
TRT_VER="$(dpkg -l 2>/dev/null | awk '/libnvinfer[0-9]/{print $3}' | head -1)"
[ -n "$TRT_VER" ] || TRT_VER="$(python3 -c 'import tensorrt;print(tensorrt.__version__)' 2>/dev/null || echo 未知)"
TRTEXEC="$(command -v trtexec || true)"
[ -n "$TRTEXEC" ] || [ -x /usr/src/tensorrt/bin/trtexec ] && TRTEXEC="${TRTEXEC:-/usr/src/tensorrt/bin/trtexec}"
CUDA_VER="$(nvcc --version 2>/dev/null | awk '/release/{print $5}' | tr -d ,)"
[ -n "$CUDA_VER" ] || CUDA_VER="$(dpkg -l 2>/dev/null | awk '/cuda-cudart/{print $3}' | head -1)"

echo "[平台] 型号      : ${MODEL_TXT:-$SOC}"
echo "[平台] L4T/JetPack: ${L4T}"
echo "[平台] CUDA      : ${CUDA_VER:-未知}"
echo "[平台] TensorRT  : ${TRT_VER:-未安装}"
echo "[平台] trtexec   : ${TRTEXEC:-未找到（JetPack 一般在 /usr/src/tensorrt/bin/trtexec）}"

# 精度建议
REC="FP16"
case "${MODEL_TXT}${SOC}" in
  *Nano*)   REC="FP16";   WHY="Tegra X1(Maxwell/SM5.3) 无 INT8 硬件：INT8 只能 FP16 模拟，不快且掉精度" ;;
  *TX2*)    REC="FP16";   WHY="TX2(SM6.2) 有 DP4A，INT8 有小幅收益，但 FP16 更稳更好" ;;
  *Xavier*) REC="INT8";   WHY="Xavier(SM7.2) 有 INT8 张量核，INT8 通常比 FP16 再快 1.5~2×" ;;
  *Orin*)   REC="INT8";   WHY="Orin(SM8.7) 有 INT8 张量核，INT8 收益最大" ;;
  *)        REC="FP16";   WHY="识别不到型号，先按 FP16（最稳）" ;;
esac
echo "[建议] 精度      : **$REC** —— $WHY"

# 依赖体检
echo "[依赖] python3 tensorrt : $(python3 -c 'import tensorrt;print("OK",tensorrt.__version__)' 2>/dev/null || echo '缺失')"
echo "[依赖] python3 pycuda   : $(python3 -c 'import pycuda;print("OK",pycuda.VERSION_TEXT)' 2>/dev/null || echo '缺失（容器内跑引擎需要：sudo pip3 install pycuda）')"
echo "[依赖] python3 onnx     : $(python3 -c 'import onnx;print("OK",onnx.__version__)' 2>/dev/null || echo '缺失（仅 --check 与 INT8 校准需要）')"
if [ -f "$ONNX" ]; then
  echo "[模型] ONNX      : $ONNX（$(du -h "$ONNX" | cut -f1)）"
else
  echo "[模型] ❌ 找不到 $ONNX"
fi
[ "$CHECK_ONLY" = "0" ] || { echo; echo "（--check 模式，结束）"; exit 0; }

[ -f "$ONNX" ] || { echo "❌ 缺少 ONNX 模型：$ONNX"; exit 1; }
[ -n "$TRTEXEC" ] || { echo "❌ 找不到 trtexec，无法构建引擎（JetPack 自带：/usr/src/tensorrt/bin/trtexec）"; exit 1; }
mkdir -p "$OUT_DIR" engines

# 不同 TRT 版本的 workspace 参数不同
TRT_MAJOR="$(echo "${TRT_VER}" | grep -oE '^[0-9]+' || echo 8)"
if [ "${TRT_MAJOR:-8}" -ge 8 ]; then
  WS_FLAG="--memPoolSize=workspace:${WORKSPACE}MiB"
else
  WS_FLAG="--workspace=${WORKSPACE}"
fi
echo "[构建] workspace 参数: $WS_FLAG"

# ---------------- 2) 构建 FP16 ----------------
ENG_FP16="$OUT_DIR/goods_yolov8n_640_fp16.engine"
echo
echo "=== [1/3] 构建 FP16 引擎 ==="
rm -f "$ENG_FP16"
"$TRTEXEC" --onnx="$ONNX" --saveEngine="$ENG_FP16" --fp16 $WS_FLAG \
    --shapes="images:1x3x${IMGSZ}x${IMGSZ}" 2>&1 | tail -6
if [ ! -s "$ENG_FP16" ]; then
  echo "❌ FP16 引擎构建失败（内存不足可试 --workspace 256）"
  exit 1
fi
echo "✅ $(basename "$ENG_FP16")  $(du -h "$ENG_FP16" | cut -f1)"

# ---------------- 3) 可选 INT8 ----------------
ENG_INT8=""
if [ "$DO_INT8" = "1" ]; then
  echo
  echo "=== [2/3] 构建 INT8 引擎（需要校准图）==="
  if [ -z "$CALIB_DIR" ] || [ ! -d "$CALIB_DIR" ]; then
    echo "⚠ 未给 --calib-dir（现场实拍 100~300 张最佳），跳过 INT8"
  else
    echo "   校准集: $CALIB_DIR（取前 $CALIB_N 张）"
    # INT8 用 Python 构建器（熵校准），trtexec 的 INT8 需要预生成 calib.cache 不方便
    python3 tools/build_trt_engine.py --onnx "$ONNX" --calib-dir "$CALIB_DIR" \
        --calib-n "$CALIB_N" --imgsz "$IMGSZ" --out engines --bench 0 2>&1 | tail -8
    if [ -s engines/best_int8.engine ]; then
      cp -f engines/best_int8.engine "$OUT_DIR/goods_yolov8n_640_int8.engine"
      ENG_INT8="$OUT_DIR/goods_yolov8n_640_int8.engine"
      echo "✅ $(basename "$ENG_INT8")  $(du -h "$ENG_INT8" | cut -f1)"
    else
      echo "⚠ INT8 构建未产出（$REC 平台上本就不推荐，继续用 FP16 即可）"
    fi
  fi
fi

# ---------------- 4) 设备上实测 ----------------
echo
echo "=== [3/3] 设备实测（trtexec benchmark）==="
bench() {
  local f="$1" name="$2"
  [ -s "$f" ] || return
  local out
  out="$("$TRTEXEC" --loadEngine="$f" --iterations="$BENCH_ITERS" --warmUp=500 --duration=5 2>&1)"
  local gpu qps
  gpu="$(echo "$out" | grep -oE 'GPU Compute Time:.*mean = [0-9.]+ ms' | grep -oE 'mean = [0-9.]+' | head -1 | grep -oE '[0-9.]+')"
  qps="$(echo "$out" | grep -oE 'Throughput: [0-9.]+ qps' | grep -oE '[0-9.]+' | head -1)"
  printf "  %-34s GPU 前向 %s ms   吞吐 %s qps\n" "$name" "${gpu:-?}" "${qps:-?}"
}
bench "$ENG_FP16" "FP16 引擎"
[ -n "$ENG_INT8" ] && bench "$ENG_INT8" "INT8 引擎"

echo
echo "=============================================================="
echo " 部署：把下面几行写进 .env（然后重启 sort-yolo）"
echo "=============================================================="
echo "ENGINE=trt"
echo "MODEL_PATH=/app/models/detect/$(basename "$ENG_FP16")"
[ -n "$ENG_INT8" ] && echo "# 若实测 INT8 更快且精度可接受，可换成：MODEL_PATH=/app/models/detect/$(basename "$ENG_INT8")"
echo
echo "  ⚠ 容器里跑 TensorRT 需要宿主机的 TRT 运行时与 Python 绑定，用叠加文件启动："
echo "     docker-compose -f docker-compose.jetson.yml -f docker-compose.jetson-trt.yml up -d sort-yolo"
echo "     缺 pycuda 时：sudo pip3 install pycuda   （宿主机装一次即可，容器直接挂进去）"
echo
echo "  对比基线（不进 TRT 时的两条路，都不用装任何东西）："
echo "     DNN_BACKEND=cuda  → JetPack 自带 OpenCV 的 CUDA FP16 后端（GPU，零依赖）"
echo "     不设则自动：有 CUDA 就用，没有就 CPU"
echo "=============================================================="
