#!/usr/bin/env bash
# finish_v3_gpu.sh —— 训练已完成，接着做：真实域评估 → 导出量化 → ONNX 评估 → 部署
# 用法: bash tools/finish_v3_gpu.sh [权重路径]
set -u
cd /home/xxxffyy/工创/工创yolo
PY=${PY:-/home/xxxffyy/gpu_env/bin/python}
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1

SRC=${1:-runs/detect/runs/detect/train/weights/best.pt}
if [ ! -f "$SRC" ]; then SRC=$(find runs -name best.pt -newermt "-6 hours" | head -1); fi
echo "=== 权重: $SRC ==="
cp -f "$SRC" runs/best_v3_gpu.pt
ls -l runs/best_v3_gpu.pt

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
