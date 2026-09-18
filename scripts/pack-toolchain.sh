#!/usr/bin/env bash
# ============================================================
# pack-toolchain.sh —— 打包训练/评估/复核工具链（PC 端用，不含数据集）
# ------------------------------------------------------------
# 产出：sort-toolchain-<版本>.tar.gz
#   工创yolo/ 的脚本与配置（训练、导出量化、误检评估、标签复核、负样本挖掘）
#   ❌ 不含 data/ runs/ exports*/ models/（体积大，且在 PC 上可重新生成）
#
# 用法：bash sort-web/scripts/pack-toolchain.sh --version 20260917
# ============================================================
set -u
VERSION=""
OUTDIR=""
while [ $# -gt 0 ]; do
  case "$1" in
    --version) VERSION="$2"; shift 2 ;;
    --out) OUTDIR="$2"; shift 2 ;;
    *) echo "未知参数: $1"; exit 1 ;;
  esac
done
VERSION=${VERSION:-$(date +%Y%m%d)}
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WS="$(cd "$HERE/../.." && pwd)"          # 工作区根
SRC="$WS/工创yolo"
[ -d "$SRC" ] || { echo "❌ 找不到 $SRC"; exit 1; }
[ -n "$OUTDIR" ] || OUTDIR="$WS"

STAGE="$OUTDIR/sort-toolchain-$VERSION"
rm -rf "$STAGE"
mkdir -p "$STAGE"/{tools,config}

echo "=== 拷贝工具链（不含 data/runs/模型）==="
cp -a "$SRC"/*.py "$SRC"/*.md "$SRC"/requirements.txt "$STAGE"/ 2>/dev/null || true
cp -a "$SRC"/tools/*.py "$SRC"/tools/*.sh "$STAGE"/tools/
cp -a "$SRC"/config/*.yaml "$STAGE"/config/
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} + 2>/dev/null || true

cat > "$STAGE/README-TOOLCHAIN.md" <<'DOCEOF'
# 训练 / 评估 / 复核工具链（PC 端）

不含数据集与模型（体积大，见下"数据从哪来"）。在 PC 上跑，产出 ONNX 后拷给 Jetson。

## 环境

```bash
python3 -m venv gpu_env && source gpu_env/bin/activate
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.6.0 torchvision==0.21.0 ultralytics onnx onnxruntime onnxslim \
    opencv-python-headless pillow pyyaml matplotlib pandas
```

> Linux 版 PyPI 的 torch wheel 自带 CUDA 运行时（含 cudnn/cublas），不需要额外加
> `--index-url download.pytorch.org`；Windows 版才是 CPU build。

## 常用命令

```bash
# 1) 合成多面体图片（扩样本用）
python tools/synth_polyhedra.py --out data/synth --n 200 --per-image 2

# 2) 组装数据集（真实域优先 + 分组切分防泄漏 + 可选负样本）
python tools/build_mix_v3.py --group-split --neg-dir data/negatives

# 3) 训练 + 真实域评估 + 误检专项 + 导出量化 + 部署
bash tools/retrain_v3_gpu.sh
bash tools/finish_v3_gpu.sh runs/best_after_neg2.pt

# 4) 只看某套切分的效果（不部署）
bash tools/eval_split.sh <标签>

# 5) 误检专项：误检/图 · 漏检/图 · 纯背景图检出
python tools/eval_fp.py --weights runs/best.pt --data data/mix_v3 --data2 data/negatives

# 6) 标签复核流水线（分诊 → 浏览器复核 → 回写 → 体检）
python tools/label_audit.py --weights runs/best.pt --out data/review
python tools/review_app.py --port 8770          # 浏览器打开 localhost:8770
python tools/apply_review.py --dry-run          # 确认后加 --write
python tools/check_labels.py --data data/mix_v3

# 7) 负样本挖掘（治背景误检）
python tools/mine_negatives.py --weights runs/best.pt --data data/mix_v3 --out data/negatives
```

## 数据从哪来

本包**不含** `data/`（实拍原图 + 标注）与 `runs/`（训练产物）。迁移时请一并拷贝：

```
data/real_synth_mix/    实拍图 + 标注 + dataset_summary.json（溯源）
data/mix_v3/            组装后的训练/验证集
data/negatives/         背景负样本（空标签）
runs/                   训练产物（best_*.pt / 日志 / 评估报告）
```

## 关键文档

- `docs_TRAINING.md` —— 训练/量化/误检治理全记录（含每次实验的真实数字）
- `docs_REVIEW.md` —— 标签人工复核 SOP（为什么复核、怎么复核、验收标准）
DOCEOF

( cd "$STAGE" && find . -type f -print0 | sort -z | xargs -0 md5sum > MD5SUMS )
TARBALL="$OUTDIR/sort-toolchain-$VERSION.tar.gz"
tar -czf "$TARBALL" -C "$OUTDIR" "sort-toolchain-$VERSION"
echo "✅ $TARBALL  ($(du -h "$TARBALL" | cut -f1)，$(find "$STAGE" -type f | wc -l) 个文件)"
