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

## 6. 属性 CNN（颜色 / 形状 / 污渍）—— 本次已训练并达标

数据集：`tools/run_attr_pipeline.sh` 生成 **2400 张单目标图**（1024×576，含阴影/远近/角度），
按类别导出属性小图；实测每类样本数：

| 任务 | 类别数 | 每类样本 | 最佳验证准确率 | 验收线 |
|---|---|---|---|---|
| color | 9 | 232–292 | **95.59%** | ≥95% ✅ |
| shape | 7 | 319–372 | **99.45%** | ≥95% ✅ |
| stain | 3 | clean 1654 / stain 532 / defect 214 | **96.41%** | ≥90% ✅ |

训练中修掉的两个真实问题：
1. `BatchNorm` 在最后一批 batch=1 时崩溃 → DataLoader 加 `drop_last=True`；
2. 归一化常量用 numpy 默认 float64，把输入提升成 double → 常量显式 `dtype=np.float32`。

导出与部署：
```bash
python3 tools/run_attr_pipeline.sh                    # 生成→训练→导出→部署到 sort-web/models/attr/
# 产物：runs/attr/{color,shape,stain}/best.onnx → ../sort-web/models/attr/{task}.onnx
```

容器实测（sort-cnn :8102，训练集外图片）：
```
真值 颜色=red    → 预测 red(1.00)
真值 形状=cube   → 预测 cube(0.99)   prism5 → prism5(1.00)   prism6 → prism6(1.00)
真值 表面=stain  → 预测 stain(0.98)
```

三层全链路实测（yolo 检测 → cnn 属性 → 视觉编排，输入一张完整场景图）：
```
引擎 yolo(+cnn) | 检出 1 件 | 框 [54,63,339,370]（真值 中心(0.385,0.424) 尺寸(0.523,0.590)）
  · 形状=正四面体(0.78) 颜色=紫色(1.00) 表面=无
```

## 7. TensorRT（Jetson 上的 INT8）

`tools/build_trt_engine.py`：在 Jetson 上构建 **INT8（熵校准）+ FP16** 引擎，输出 `*.engine` 与 `calib.cache`。

```bash
# PC 上先自检（无需 TensorRT）
python3 tools/build_trt_engine.py --check --onnx exports_clean/best_fp32.onnx --calib-dir data/mix_clean/images/train
# Jetson 上构建（校准集建议用现场实拍 100~300 张）
python3 tools/build_trt_engine.py --onnx exports_clean/best_fp32.onnx \
    --calib-dir data/mix_clean/images/train --calib-n 200 --imgsz 640 --out engines --bench 20
```
部署：`cp engines/*.engine ../sort-web/models/yolo/`，compose 里设 `ENGINE=trt`
（`server/yolo_server.py` 已支持 `ENGINE=trt`，用 pycuda + TensorRT runtime 执行）。
