#!/usr/bin/env bash
set -u
cd /home/xxxffyy/工创/工创yolo
PY=/home/xxxffyy/miao_llm_env/bin/python
export YOLO_AUTOINSTALL=false
exec > >(tee -a runs/finalize_v3.log) 2>&1
echo "[1/4] 等待真实域微调结束 …"
while pgrep -f "[t]rain_detection.py --config config/train_v3.yaml" >/dev/null; do sleep 20; done
BEST=$(ls -t runs/detect/runs/detect/train*/weights/best.pt 2>/dev/null | head -1)
[ -z "$BEST" ] && BEST=$(find runs -name best.pt | head -1)
echo "[2/4] 权重 $BEST → 真实域验证集评估"
$PY - <<PY
from ultralytics import YOLO
r = YOLO('$BEST').val(data='data/mix_v3/data.yaml', imgsz=640, split='val', verbose=False)
print('  真实域验证集: mAP50=%.4f mAP50-95=%.4f P=%.4f R=%.4f' % (
    r.box.map50, r.box.map, r.box.mp, r.box.mr))
PY
echo "[3/4] 导出 ONNX + INT8 量化"
rm -rf exports_v3 && $PY tools/export_quantize.py --weights "$BEST" --data data/mix_v3 \
    --imgsz 640 --calib-n 60 --bench-n 20 --out exports_v3
echo "[4/4] 精度对比 + 部署到 sort-web"
$PY tools/eval_onnx.py --data data/mix_v3 --imgsz 640 \
    --models exports_v3/best_fp32.onnx exports_v3/best_int8_dynamic.onnx --out exports_v3/eval_report.json
cp exports_v3/best_fp32.onnx ../sort-web/models/detect/goods_yolov8n_640_fp32.onnx
cp exports_v3/best_int8_dynamic.onnx ../sort-web/models/detect/goods_yolov8n_640_int8.onnx
echo "  已部署到 sort-web/models/detect/"
echo "[完成]"
