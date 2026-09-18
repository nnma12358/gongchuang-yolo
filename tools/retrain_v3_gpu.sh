#!/usr/bin/env bash
# retrain_v3_gpu.sh —— GPU 重训流水线（WSL 直跑，RTX 4050 / CUDA）
#   1) 真实域微调（AdamW lr0 5e-4 / 40ep / patience 12）
#   2) 真实域验证集评估（28 实拍 + 10 生成，唯一可信指标）
#   3) 导出 ONNX(FP32) + 动态 INT8 量化 + 延迟基准
#   4) ONNX 精度对比 → 部署到 sort-web/models/detect/
#
# 用法: bash tools/retrain_v3_gpu.sh
# 环境: PY=/home/xxxffyy/gpu_env/bin/python（CUDA 版 torch）
set -u
cd /home/xxxffyy/工创/工创yolo
PY=${PY:-/home/xxxffyy/gpu_env/bin/python}
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1
LOG=runs/v3_gpu.log
exec > >(tee -a "$LOG") 2>&1

echo "=== GPU 自检 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
"$PY" - <<'PY'
import torch
print("torch", torch.__version__, "| cuda_available", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0), "| cap",
          torch.cuda.get_device_capability(0))
PY

BASE=runs/base_v3.pt
[ -f "$BASE" ] || BASE=models/yolov8n.pt
echo "=== 训练（基线 $BASE，device 0）==="
rm -rf runs/detect/train
"$PY" train_detection.py --config config/train_v3.yaml --base "$BASE" --device 0
BEST=runs/detect/train/weights/best.pt
if [ ! -f "$BEST" ]; then echo "[FAIL] 未生成 $BEST"; exit 1; fi
cp -f "$BEST" runs/best_v3_gpu.pt
echo "=== 权重: runs/best_v3_gpu.pt ==="

echo "=== 真实域评估（28 实拍 + 10 生成）==="
"$PY" - <<'PY'
from ultralytics import YOLO
r = YOLO('runs/best_v3_gpu.pt').val(data='data/mix_v3/data.yaml', imgsz=640,
                                    split='val', verbose=False)
print('REAL_DOMAIN mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f' % (
    r.box.map50, r.box.map, r.box.mp, r.box.mr))
PY

echo "=== 导出 ONNX(FP32) + INT8 量化 ==="
rm -rf exports_v3gpu
"$PY" tools/export_quantize.py --weights runs/best_v3_gpu.pt --data data/mix_v3 \
    --imgsz 640 --calib-n 60 --bench-n 20 --out exports_v3gpu

echo "=== ONNX 精度/延迟对比 ==="
"$PY" tools/eval_onnx.py --data data/mix_v3 --imgsz 640 \
    --models exports_v3gpu/best_fp32.onnx exports_v3gpu/best_int8_dynamic.onnx \
    --out exports_v3gpu/eval_report.json

echo "=== 部署到 sort-web（供网关/容器加载）==="
cp -f exports_v3gpu/best_fp32.onnx ../sort-web/models/detect/goods_yolov8n_640_fp32.onnx
cp -f exports_v3gpu/best_int8_dynamic.onnx ../sort-web/models/detect/goods_yolov8n_640_int8.onnx
ls -l ../sort-web/models/detect/
echo "DONE"
