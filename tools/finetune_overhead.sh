#!/usr/bin/env bash
# finetune_overhead.sh —— 现场真实尺度微调 + 导出（等真实照片到位后一键跑）
#
# 背景：顶置 Astra 离台面约 660mm，40mm 货物在 640 画幅里只有约 30px，
#       而现有模型工作区间是 69~100px（实测置信度 0.11 vs 0.59~0.72），
#       所以在现场 0 检出、且把黑色工具箱误检成货物。
#       本流水线用「现场几何合成数据 + 真实俯拍照片」把模型拉回现场尺度。
#
# 前置：
#   1) 真实照片已采集并标注：
#        python3 tools/capture_overhead.py --session tray_a --n 60     # 现场摆货时抓帧
#        python3 tools/capture_overhead.py --session bg_lab --n 40      # 纯背景（负样本）
#        python3 tools/review_app.py ...                               # 人工标注/复核
#   2) GPU 环境：/home/xxxffyy/gpu_env/bin/python（CUDA torch）
#
# 用法：
#   bash tools/finetune_overhead.sh                       # 全流程
#   bash tools/finetune_overhead.sh --skip-synth          # 复用已生成的合成数据
#   BASE=runs/best_v3_gpu.pt EPOCHS=60 bash tools/finetune_overhead.sh
set -u
cd /home/xxxffyy/工创/工创yolo

PY=${PY:-/home/xxxffyy/gpu_env/bin/python}
BASE=${BASE:-runs/best_v3_gpu.pt}
EPOCHS=${EPOCHS:-}
SYNTH_DIR=${SYNTH_DIR:-data/synth_realgeo}
REAL_DIR=${REAL_DIR:-data/real_overhead}
OUT_DIR=${OUT_DIR:-data/mix_overhead}
EXPORT_DIR=${EXPORT_DIR:-exports_overhead}
LOG=runs/finetune_overhead.log
SKIP_SYNTH=0
[ "${1:-}" = "--skip-synth" ] && SKIP_SYNTH=1

export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1
exec > >(tee -a "$LOG") 2>&1

echo "=== 0. 环境自检 ==="
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader || true
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda", torch.cuda.is_available())
PY
[ -f "$BASE" ] || { echo "[FAIL] 基线权重不存在: $BASE"; exit 1; }
echo "基线权重: $BASE"

echo "=== 1. 现场几何合成数据（cam_height≈660mm）==="
if [ "$SKIP_SYNTH" = "1" ]; then
  echo "跳过（--skip-synth），复用 $SYNTH_DIR"
else
  rm -rf "$SYNTH_DIR"
  # 每图 3~6 个货物：现场台面就是多货同框；imgsz 1280 与现场采样一致
  "$PY" tools/synth_polyhedra.py --out "$SYNTH_DIR" --n 500 --per-image 4 \
      --prefix syngeo --cam-height 660 --tray-mm 160 --seed 20260921 --no-attributes \
      || { echo "[FAIL] 合成数据生成失败"; exit 1; }
fi

echo "=== 2. 组装数据集（原 mix_v3 + 真实俯拍 + 现场几何合成）==="
"$PY" tools/assemble_overhead_dataset.py --real "$REAL_DIR" --synth "$SYNTH_DIR" \
    --base data/mix_v3 --out "$OUT_DIR" || exit 1

echo "=== 3. 微调训练 ==="
TRAIN_ARGS="--config config/train_overhead.yaml --base $BASE --device 0"
[ -n "$EPOCHS" ] && TRAIN_ARGS="$TRAIN_ARGS --epochs $EPOCHS"
rm -rf runs/detect/train
"$PY" train_detection.py $TRAIN_ARGS || { echo "[FAIL] 训练失败"; exit 1; }
BEST=runs/detect/train/weights/best.pt
[ -f "$BEST" ] || { echo "[FAIL] 未生成 $BEST"; exit 1; }
cp -f "$BEST" runs/best_overhead.pt
echo "=== 权重: runs/best_overhead.pt ==="

echo "=== 4. 验证集评估（含真实俯拍 val）==="
"$PY" - <<PY
from ultralytics import YOLO
r = YOLO('runs/best_overhead.pt').val(data='$OUT_DIR/data.yaml', imgsz=640, split='val', verbose=False)
print('VAL mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f' % (r.box.map50, r.box.map, r.box.mp, r.box.mr))
PY

echo "=== 5. 导出 ONNX(FP32) + INT8 量化 ==="
rm -rf "$EXPORT_DIR"
"$PY" tools/export_quantize.py --weights runs/best_overhead.pt --data "$OUT_DIR" \
    --imgsz 640 --calib-n 60 --bench-n 20 --out "$EXPORT_DIR" || exit 1

echo "=== 6. 精度/延迟对比 ==="
"$PY" tools/eval_onnx.py --data "$OUT_DIR" --imgsz 640 \
    --models "$EXPORT_DIR"/best_fp32.onnx "$EXPORT_DIR"/best_int8_dynamic.onnx \
    --out "$EXPORT_DIR/eval_report.json" || true

echo "=== 7. 部署到 sort-web（供容器加载）==="
cp -f "$EXPORT_DIR"/best_fp32.onnx ../sort-web/models/detect/goods_yolov8n_640_fp32.onnx
cp -f "$EXPORT_DIR"/best_int8_dynamic.onnx ../sort-web/models/detect/goods_yolov8n_640_int8.onnx
ls -l ../sort-web/models/detect/

cat <<'TIP'

=== 8. 部署到 Jetson（在 PC 上执行；引擎必须在设备上用 trtexec 重建）===
  # 8.1 传模型与脚本
  scp ../sort-web/models/detect/goods_yolov8n_640_fp32.onnx wheeltec@10.60.10.24:~/sort-jetson-deploy-20260917/models/detect/
  scp scripts/build-trt-on-jetson.sh wheeltec@10.60.10.24:~/sort-jetson-deploy-20260917/scripts/

  # 8.2 在 Jetson 上重建 FP16 引擎（TRT 版本必须与设备一致，不能在 PC 上编）
  ssh wheeltec@10.60.10.24 'cd ~/sort-jetson-deploy-20260917 && \
      printf "<设备密码>\n" | sudo -S bash scripts/build-trt-on-jetson.sh'

  # 8.3 重启检测容器加载新引擎
  ssh wheeltec@10.60.10.24 'cd ~/sort-jetson-deploy-20260917 && \
      printf "<设备密码>\n" | sudo -S docker-compose -f docker-compose.jetson.yml up -d --no-build yolo'

  # 8.4 回归：现场相机 60 帧采样（PC 直接跑，看真实货物检出与工具箱假检）
  python3 tools/capture_overhead.py --session regression --n 60 --marked
  #   对照基线：修复前 = 真实货物 0 检出 + 工具箱假检 0.365~0.487（80% 帧）
TIP
echo "DONE"
