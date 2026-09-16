# 训练 · 量化 · 测试（本次实测记录）

## 1. 一条命令跑完

```bash
# 干净子集全流程（建子集 → 训练 → 导出 → 量化 → 精度对比）
nohup bash tools/run_clean_pipeline.sh > runs/clean_pipeline_boot.log 2>&1 &
tail -f runs/clean_pipeline.log

# 手动分步
python3 tools/make_clean_subset.py                       # 抽干净标签子集
python3 train_detection.py --config config/train_clean.yaml --device cpu
python3 tools/export_quantize.py --weights runs/.../best.pt --data data/mix_clean --imgsz 640 --out exports_clean
python3 tools/eval_onnx.py --data data/mix_clean --imgsz 640 \
        --models exports_clean/best_fp32.onnx exports_clean/best_int8.onnx
```

## 2. 两次训练的关键发现

| 运行 | 数据 | 配方 | 结果 |
|---|---|---|---|
| 第 1 次 | 混合集 191 张（含 146 张**自动预标注**实拍图） | 默认 SGD lr0=0.01 + copy_paste/mixup | **发散**：mAP50 0.55 → 0.001 |
| 第 2 次 | 混合集同上 | AdamW lr0=1e-3 + freeze 10 + 温和增强 | 仍震荡：0.44 → 0.17 → 0.07 |
| 第 3 次 | **干净子集 78 张**（人工标注 18 + 生成图 60） | AdamW lr0=1e-3，不冻结 | **epoch 1 即 mAP50 0.942 / mAP50-95 0.736** |

结论：**标签质量 > 数据数量**。实拍自动预标注（白底白物、低对比）噪声是主因；
先用干净子集跑通配方与管线，再逐批复核 `images/unlabeled` 与 `.review_needed.txt` 中的图片并入。

## 3. 量化结果（ONNX Runtime · CPU · imgsz 640）

| 模型 | 体积 | 均值延迟 | P95 | 相对 FP32 |
|---|---|---|---|---|
| FP32 ONNX | 12.27 MB | 58.1 ms | 75.7 ms | 1.00× |
| INT8 动态量化 | 3.36 MB | 172.96 ms | 198.2 ms | **0.34×（更慢，不采用）** |
| **INT8 静态量化（QDQ + 校准）** | **3.46 MB** | **19.7 ms** | 22.9 ms | **2.95× 加速，体积 28%** |

要点：
- **动态量化对卷积网络反而更慢**（权重反量化开销），必须用**静态量化**；
- 静态量化需 opset ≥ 13（per-channel 的 `DequantizeLinear.axis`），导出时已固定 opset=13；
- 若 per-channel 产出无效模型，脚本会自动回退 per-channel=False 并做加载校验。

## 4. 与容器集成（已实测）

```bash
# 把导出的模型拷进部署目录
cp exports_clean/best_fp32.onnx ../sort-web/models/yolo/best.onnx      # 容器默认（OpenCV DNN）
cp exports_clean/best_int8.onnx ../sort-web/models/yolo/best_int8.onnx # 可选（ONNX Runtime）
# yolo 容器：ENGINE=opencv 用 FP32；ENGINE=ort 用 INT8
```

实测：`sort-web/server/yolo_server.py` 两种引擎均正常加载（`/health` ok，
FP32/OpenCV 71ms · INT8/ONNXRuntime 66ms 端到端，含预处理与 NMS）。

## 5. 文件

```
tools/make_clean_subset.py     抽干净标签子集（按 dataset_summary.json 的来源字段）
tools/export_quantize.py       .pt → ONNX → INT8（动态/静态）→ 基准测试 → quant_report.json
tools/eval_onnx.py             FP32 vs INT8 的 P/R/F1/mAP50/IoU/延迟对比 → eval_report.json
tools/run_clean_pipeline.sh    以上全流程串行（后台跑）
config/train_mix.yaml          小数据集稳定配方（AdamW + lr0 1e-3 + 温和增强）
config/train_clean.yaml        干净子集配方（自动生成，freeze=0）
```
