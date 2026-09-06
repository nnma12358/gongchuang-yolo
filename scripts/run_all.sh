#!/bin/bash
# 自训练全流程: 权重获取 -> 数据准备 -> 检测训练 -> 属性训练 -> 导出 -> 评估
# 用法: bash scripts/run_all.sh [config/dataset.yaml]
set -e
cd "$(dirname "$0")/.."
CFG=${1:-config/dataset.yaml}

echo "== [0/6] 基础权重 =="
python fetch_weights.py || echo "(跳过: 已存在或自动获取失败, 可手动放置)"

echo "== [1/6] 数据准备 =="
python prepare_dataset.py --config "$CFG"

echo "== [2/6] 目标检测训练 =="
EPOCHS=${EPOCHS:-120}; IMGSZ=${IMGSZ:-640}
python train_detection.py --config "$CFG" --epochs "$EPOCHS" --imgsz "$IMGSZ"

echo "== [3/6] 属性分类器训练(方案A) =="
for t in color shape defect; do
  python train_attributes.py --config "$CFG" --task "$t" || echo "(跳过 $t: 无属性数据集)"
done

echo "== [4/6] 导出 =="
python export_models.py --config "$CFG" --imgsz "$IMGSZ"

echo "== [5/6] 评估 =="
python eval_detection.py --config "$CFG" --imgsz "$IMGSZ"
python eval_pipeline.py --config "$CFG" || true
echo "全部完成, 结果见 runs/ exports/ reports/"
