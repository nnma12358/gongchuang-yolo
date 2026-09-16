#!/usr/bin/env bash
# 干净标签子集 → 训练 → 导出量化 → 精度评估（全自动，后台运行）
# 背景：混合集里 146 张自动预标注的实拍图标签有噪声，导致训练震荡；
#      先用“人工标注 18 张 + 生成图 60 张”的干净子集验证管线与配方。
set -u
cd "$(dirname "$0")/.."
PY=/home/xxxffyy/miao_llm_env/bin/python
export YOLO_AUTOINSTALL=false
LOG=runs/clean_pipeline.log
exec > >(tee -a "$LOG") 2>&1

echo "=== [1/4] 构建干净标签子集 ==="
$PY tools/make_clean_subset.py
[ -f data/mix_clean/data.yaml ] || { echo "子集构建失败"; exit 1; }

echo "=== [2/4] 训练（40 epochs, imgsz 640, AdamW lr0 1e-3）==="
sed 's#^root:.*#root: "'"$PWD"'/data/mix_clean"#' config/train_mix.yaml > config/train_clean.yaml
sed -i 's/^freeze:.*/freeze: 0/' config/train_clean.yaml     # 域差异大，不冻结骨干
$PY train_detection.py --config config/train_clean.yaml --device cpu

echo "=== [3/4] 导出 ONNX + INT8 量化 ==="
BEST=$(find runs/detect/runs -name best.pt -newermt '-3 hours' 2>/dev/null | head -1)
[ -z "$BEST" ] && BEST=$(find runs -name best.pt | head -1)
echo "权重: $BEST"
rm -rf exports_clean && $PY tools/export_quantize.py --weights "$BEST" \
    --data data/mix_clean --imgsz 640 --calib-n 30 --bench-n 20 --out exports_clean

echo "=== [4/4] FP32 vs INT8 精度/延迟对比 ==="
$PY tools/eval_onnx.py --data data/mix_clean --imgsz 640 \
    --models exports_clean/best_fp32.onnx exports_clean/best_int8.onnx \
    --out exports_clean/eval_report.json
echo "=== 全部完成 ==="
