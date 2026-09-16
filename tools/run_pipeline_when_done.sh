#!/usr/bin/env bash
# 等训练结束后自动：导出 ONNX → INT8 量化 → FP32/INT8 精度与延迟对比
set -u
cd "$(dirname "$0")/.."
PY=/home/xxxffyy/miao_llm_env/bin/python
export YOLO_AUTOINSTALL=false
echo "[pipeline] 等待训练结束 …"
while pgrep -f "train_detection.py --config config/train_mix.yaml" >/dev/null; do sleep 20; done
echo "[pipeline] 训练已结束，开始导出与量化"
BEST=$(ls -t runs/detect/runs/detect/train*/weights/best.pt 2>/dev/null | head -1)
[ -z "$BEST" ] && BEST=$(find runs -name best.pt | head -1)
echo "[pipeline] 权重: $BEST"
rm -rf exports_final && $PY tools/export_quantize.py --weights "$BEST" \
    --data data/real_synth_mix --imgsz 640 --calib-n 60 --bench-n 20 --out exports_final
echo "[pipeline] 精度对比 …"
$PY tools/eval_onnx.py --data data/real_synth_mix --imgsz 640 \
    --models exports_final/best_fp32.onnx exports_final/best_int8.onnx \
    --out exports_final/eval_report.json
echo "[pipeline] 完成"
