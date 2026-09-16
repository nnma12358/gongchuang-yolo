#!/usr/bin/env bash
# eval_split.sh —— 在当前 data/mix_v3 切分上训练+评估（不部署），给"复核前/后"做同口径对比
# 用法: bash tools/eval_split.sh [标签]     例如 bash tools/eval_split.sh before_review
set -u
cd /home/xxxffyy/工创/工创yolo
PY=${PY:-/home/xxxffyy/gpu_env/bin/python}
TAG=${1:-run}
export YOLO_AUTOINSTALL=false PYTHONUNBUFFERED=1
LOG=runs/eval_split_$TAG.log
exec > >(tee -a "$LOG") 2>&1

echo "=== [$TAG] 切分概况 ==="
"$PY" - <<'PY'
import json
s = json.load(open('data/mix_v3/subset_summary.json', encoding='utf-8'))
print(" split:", s.get('split_mode'), "| groups:", s.get('groups'))
print(" train:", s['stats']['train'])
print(" val  :", s['stats']['val'])
PY

echo "=== [$TAG] 训练（AdamW lr0 5e-4 / 40ep / patience 12 / GPU）==="
rm -rf runs/detect/train
"$PY" train_detection.py --config config/train_v3.yaml --base runs/base_v3.pt --device 0
BEST=runs/detect/train/weights/best.pt
[ -f "$BEST" ] || { echo "[FAIL] 没有产出 $BEST"; exit 1; }
cp -f "$BEST" "runs/best_$TAG.pt"

echo "=== [$TAG] 真实域验证集评估（分组切分，无泄漏）==="
"$PY" - <<PY
import json
from ultralytics import YOLO
r = YOLO('runs/best_$TAG.pt').val(data='data/mix_v3/data.yaml', imgsz=640, split='val', verbose=False)
out = {"tag": "$TAG", "split_mode": json.load(open('data/mix_v3/subset_summary.json',encoding='utf-8')).get('split_mode'),
       "mAP50": round(float(r.box.map50), 5), "mAP50_95": round(float(r.box.map), 5),
       "P": round(float(r.box.mp), 5), "R": round(float(r.box.mr), 5)}
print("RESULT %s" % json.dumps(out, ensure_ascii=False))
json.dump(out, open('runs/eval_$TAG.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
PY
echo "=== [$TAG] 误检专项（误检/图 · 漏检/图 · 纯背景图检出）==="
"$PY" tools/eval_fp.py --weights "runs/best_$TAG.pt" --data data/mix_v3 \
    --data2 data/negatives --device 0 --out "runs/fp_$TAG.json" 2>&1 | tail -16

echo "=== [$TAG] 完成，权重 runs/best_$TAG.pt ==="
