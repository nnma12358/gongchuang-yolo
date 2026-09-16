#!/usr/bin/env bash
# 属性数据集生成 → CNN（颜色/形状/污渍）训练 → ONNX 导出（全自动后台）
# 目标：每类 ≥200 张，训出可用的属性模型供 sort-cnn 容器使用
set -u
cd "$(dirname "$0")/.."
PY=/home/xxxffyy/miao_llm_env/bin/python
export YOLO_AUTOINSTALL=false
LOG=runs/attr_pipeline.log
exec > >(tee -a "$LOG") 2>&1

N=${N:-2400}
echo "=== [1/5] 生成属性数据集（$N 张单目标图，含阴影/远近/角度）==="
rm -rf data/attr_ds
$PY tools/synth_polyhedra.py --out data/attr_ds --n "$N" --per-image 1 \
    --imgsz 1024 576 --prefix attr --preview 0 --val-ratio 0.15

echo "=== [2/5] 统计每类样本数 ==="
for task in color shape stain; do
  echo -n "  $task: "
  for d in data/attr_ds/attributes/$task/*/; do
    printf "%s=%s " "$(basename $d)" "$(ls "$d" | wc -l)"
  done
  echo
done

echo "=== [3/5] 训练三个属性分类器（TinyNet, 96x96）==="
cat > config/train_attr.yaml <<EOF
root: "$PWD/data/attr_ds"
attribute_datasets:
  color: "attributes/color"
  shape: "attributes/shape"
  stain: "attributes/stain"
color_classes: ["red","orange","yellow","green","cyan","blue","purple","black","white"]
shape_classes: ["cube","cuboid","cylinder","ball","tetra","prism5","prism6"]
stain_classes: ["clean","stain","defect"]
acceptance:
  color_acc: 0.95
  shape_acc: 0.95
  stain_acc: 0.90
EOF
for task in color shape stain; do
  echo "--- $task ---"
  $PY train_attributes.py --config config/train_attr.yaml --task "$task" --epochs 30 --batch 64 --export-onnx
done

echo "=== [4/5] 汇总指标 ==="
for task in color shape stain; do
  ls -la runs/attr/$task/best.onnx 2>/dev/null | awk -v t=$task '{printf "  %-6s best.onnx %.2f MB\n", t, $5/1048576}'
done

echo "=== [5/5] 部署到 sort-web 容器模型目录 ==="
DST=../sort-web/models/attr
mkdir -p "$DST"
for task in color shape stain; do
  [ -f runs/attr/$task/best.onnx ] && cp runs/attr/$task/best.onnx "$DST/$task.onnx"
done
$PY - <<'PY'
import json, yaml, os
cfg = yaml.safe_load(open('config/train_attr.yaml'))
os.makedirs('../sort-web/models/attr', exist_ok=True)
for task in ('color', 'shape', 'stain'):
    json.dump({str(i): n for i, n in enumerate(cfg[task + '_classes'])},
              open('../sort-web/models/attr/%s_classes.json' % task, 'w'), ensure_ascii=False, indent=1)
print("classes.json 已写入 sort-web/models/attr/")
PY
echo "=== 属性模型流水线完成 ==="
