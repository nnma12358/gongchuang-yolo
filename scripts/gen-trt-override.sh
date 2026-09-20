#!/usr/bin/env bash
# ============================================================
# gen-trt-override.sh —— 生成 TensorRT 运行时的 compose 叠加文件（在 Jetson 上执行）
# ------------------------------------------------------------
# 为什么需要：sort-yolo 容器里跑 `.engine` 要三样东西，而它们的路径随 JetPack 版本变化：
#   ① TensorRT 运行时库（libnvinfer / libnvinfer_plugin / libnvonnxparser）
#   ② TensorRT 的 Python 绑定（tensorrt 包）
#   ③ pycuda（分配显存用）
# 与其在 compose 里写死版本号（换个 JetPack 就失效），不如在设备上**探测真实路径**再生成。
#
# 用法：
#   bash scripts/gen-trt-override.sh                       # 探测 + 写 docker-compose.jetson-trt.yml
#   bash scripts/gen-trt-override.sh --engine models/detect/goods_yolov8n_640_fp16.engine
#
# 之后启动：
#   docker-compose -f docker-compose.jetson.yml -f docker-compose.jetson-trt.yml up -d sort-yolo
# ============================================================
set -u
ENGINE="/app/models/detect/goods_yolov8n_640_fp16.engine"
OUT="docker-compose.jetson-trt.yml"
while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done

LIBDIR="/usr/lib/aarch64-linux-gnu"
echo "[探测] TensorRT 运行时库 …"
LIBS=()
for pat in libnvinfer.so libnvinfer_plugin.so libnvonnxparser.so; do
  f="$(ls -1 ${LIBDIR}/${pat}* 2>/dev/null | grep -vE '\.so$' | head -1)"
  [ -n "$f" ] || f="$(ls -1 ${LIBDIR}/${pat}* 2>/dev/null | head -1)"
  if [ -n "$f" ]; then LIBS+=("$f"); echo "   ✓ $(basename "$f")"; else echo "   ✗ 缺 $pat"; fi
done

echo "[探测] Python 绑定 …"
TRT_PY="$(python3 -c 'import tensorrt,os;print(os.path.dirname(tensorrt.__file__))' 2>/dev/null || true)"
PYCUDA_PY="$(python3 -c 'import pycuda,os;print(os.path.dirname(pycuda.__file__))' 2>/dev/null || true)"
[ -n "$TRT_PY" ]    && echo "   ✓ tensorrt → $TRT_PY"    || echo "   ✗ 缺 tensorrt python 绑定（JetPack: sudo apt install python3-libnvinfer）"
[ -n "$PYCUDA_PY" ] && echo "   ✓ pycuda   → $PYCUDA_PY" || echo "   ✗ 缺 pycuda（宿主机执行: sudo pip3 install pycuda）"

# 探测结果写进叠加文件；缺的项生成注释，Docker 不会因注释报错
{
  echo "# ============================================================"
  echo "# 由 scripts/gen-trt-override.sh 自动生成（$(date '+%F %T')）"
  echo "# 作用：让 sort-yolo 容器用宿主机的 TensorRT 跑 .engine"
  echo "# 启动：docker-compose -f docker-compose.jetson.yml -f docker-compose.jetson-trt.yml up -d sort-yolo"
  echo "# ============================================================"
  echo "services:"
  echo "  sort-yolo:"
  echo "    environment:"
  echo "      - ENGINE=trt"
  echo "      - MODEL_PATH=${ENGINE}"
  VOLS=()
  for f in "${LIBS[@]:-}"; do [ -n "$f" ] && VOLS+=("      - ${f}:${f}:ro"); done
  [ -n "$TRT_PY" ]    && VOLS+=("      - ${TRT_PY}:${TRT_PY}:ro")
  [ -n "$PYCUDA_PY" ] && VOLS+=("      - ${PYCUDA_PY}:${PYCUDA_PY}:ro")
  if [ ${#VOLS[@]} -gt 0 ]; then
    echo "    volumes:"
    for v in "${VOLS[@]}"; do echo "$v"; done
  else
    echo "    # ⚠ 宿主机没探测到 TensorRT/pycuda，无挂载项；容器会自动回退 OpenCV DNN"
  fi
  echo
  echo "# 说明：TensorRT 引擎与 GPU 架构 + TRT 版本绑定，"
  echo "#       引擎文件必须由本机 scripts/build-trt-on-jetson.sh 构建。"
} > "$OUT"

echo
echo "✅ 已生成 $OUT"
echo "   启动：docker-compose -f docker-compose.jetson.yml -f $OUT up -d sort-yolo"
echo "   自检：curl -s http://localhost:8101/health   # engine 应为 tensorrt，dnn_backend 为 cuda_fp16/cpu"
if [ -z "$TRT_PY" ] || [ -z "$PYCUDA_PY" ]; then
  echo
  echo "⚠ 上面有缺失项：容器会**自动回退**到 OpenCV DNN（不会崩），"
  echo "   但要跑满速请先补齐：sudo apt install python3-libnvinfer && sudo pip3 install pycuda"
fi
