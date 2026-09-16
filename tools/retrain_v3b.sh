#!/usr/bin/env bash
# 真实域重训（修正 LR 配方：AdamW lr0 5e-4）→ 实时评估 → 导出量化 → 部署
set -u
cd /home/xxxffyy/工创/工创yolo
PY=/home/xxxffyy/miao_llm_env/bin/python
export YOLO_AUTOINSTALL=false
LOG=runs/v3b.log
exec > >(tee -a "$LOG") 2>&1

sed -i 's/^epochs:.*/epochs: 40/' config/train_v3.yaml
sed -i 's/^patience:.*/patience: 12/' config/train_v3.yaml
echo "=== 训练（AdamW lr0 5e-4，40 epochs，真实域验证集）==="
rm -rf runs/detect/runs
$PY train_detection.py --config config/train_v3.yaml --base /tmp/base_v3.pt --device cpu 2>&1 | tail -80
BEST=$(ls -t runs/detect/runs/detect/train*/weights/best.pt 2>/dev/null | head -1)
echo "=== 权重: $BEST ==="
echo "=== 真实域评估（best 检查点）==="
$PY - <<PY
from ultralytics import YOLO
r = YOLO('$BEST').val(data='data/mix_v3/data.yaml', imgsz=640, split='val', verbose=False)
print('REAL_DOMAIN mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f' % (r.box.map50, r.box.map, r.box.mp, r.box.mr))
PY
echo "=== 导出 ONNX + INT8 ==="
rm -rf exports_v3b && $PY tools/export_quantize.py --weights "$BEST" --data data/mix_v3 \
    --imgsz 640 --calib-n 60 --bench-n 20 --out exports_v3b
echo "=== 精度对比 ==="
$PY tools/eval_onnx.py --data data/mix_v3 --imgsz 640 \
    --models exports_v3b/best_fp32.onnx exports_v3b/best_int8_dynamic.onnx --out exports_v3b/eval_report.json
echo "=== 部署到 sort-web ==="
cp exports_v3b/best_fp32.onnx ../sort-web/models/yolo/best.onnx
cp exports_v3b/best_int8_dynamic.onnx ../sort-web/models/yolo/best_int8.onnx
echo "DONE"
