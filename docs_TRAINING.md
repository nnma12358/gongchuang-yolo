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

## 8. 域差异（sim-to-real gap）实测与修复 —— 第 1 项优化

**问题**：干净子集模型在合成验证集上 mAP50 0.995，但拿到 175 张实拍图上推理时：
```
实拍(60 张)：最高置信 0.266 | conf≥0.5 命中 0/60 | ≥0.35 0 | ≥0.15 19 | ≥0.05 53
合成(14 张)：最高置信 0.737 | conf≥0.5 命中 2/14 | ≥0.35 8 | ≥0.15 13 | ≥0.05 14
```
→ **合成域验证集给出的高分是虚高的**；实拍图上模型"看得见但不敢确定"（低置信），
直接用于现场会在 conf 0.35 下零检出。

**修复（两步）**：
1. `tools/pseudo_label_real.py`：用当前检测器对实拍图做**低阈值伪标注**，
   并与几何预标注交叉验证 IoU → trust=high/medium，输出 `data/pseudo_real`（含 provenance.json）；
2. `tools/build_mix_v3.py`：组装面向真实域的数据集 —— **验证集以实拍为主**（28 实拍 + 10 生成），
   训练集 = 人工标注 18 + 实拍几何标注 111（经面积/长宽比/越界过滤，剔除 5 张不合理）
   + 生成图 50 + 伪标注 2 = 181 张；生成图面积上限放宽到 0.80（合成图物体天生占画面大）。
   然后从干净子集模型**微调**（AdamW lr0 5e-4，30 epochs，温和增强）。

```bash
python3 tools/pseudo_label_real.py --weights runs/.../best.pt --real-dir 工创/图 \
        --labeled-dir 工创/图/X-AnyLabeling --geo-labels data/real_synth_mix/labels/train --out data/pseudo_real
python3 tools/build_mix_v3.py --mixed data/real_synth_mix --pseudo-dir data/pseudo_real --out data/mix_v3
python3 train_detection.py --config config/train_v3.yaml --base runs/.../best.pt --device cpu
```

**教训**：任何宣称的 mAP 都必须标注**在哪个域上评估**；合成域评估只能证明"学到了合成分布"。
后续现场采集的实拍图应直接进 `mix_v3` 的验证集，形成"实拍闭环评估"。

## 9. 颜色/形状模型的跨域复验与修复（本轮）

**颜色模型**：把色相抖动按任务区分后重训（color ±3°，shape/stain ±8°）——
原先 ±20° 的抖动会把红↔橙标签互相污染，是颜色精度上不去的根因。

| 指标 | 修复前 | 修复后 |
|---|---|---|
| color 合成验证集 | 94.81% | **100.00%**（60 epochs） |
| color 跨域（8 张实拍正十二面体，人工标注框裁剪） | — | **8/8 正确**（white 0.99~1.00） |
| shape 合成验证集 | 99.29% | **99.77%**（注入 18 张实拍裁剪后） |
| shape 跨域（同上 8 张） | 0/8 | **4/8**（仍不足，需更多实拍形状样本） |

**结论**：颜色可跨域（合成训练即够用）；**形状跨域必须用实拍数据训练**——
18 张少样本注入只能到 4/8，建议每个形状补 30~50 张现场照片（保持与推理一致的裁剪方式）。
这也是"实拍标注"这件事优先级最高的原因。

---

## 10. GPU 重训（RTX 4050 · CUDA）—— 本轮实测

前面几轮训练都跑在 CPU（torch 2.6.0+cpu），40 epochs 要数小时，且中途被打断过几次。
本轮把训练切到 GPU，配方（AdamW lr0 5e-4 / 40ep / patience 12 / 真实域验证集）不变，
**只换算力**，结果直接拉高一大截。

### 10.1 环境（PC 端，不依赖 Docker）

Docker Hub 在本机约 0.45 MB/s，拉 `pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime`
（约 4.5 GB）要 2 小时以上；改用清华 PyPI 镜像（实测 5.14 MB/s）在 WSL 里直接建环境：

```bash
python3 -m venv /home/xxxffyy/gpu_env
/home/xxxffyy/gpu_env/bin/pip install -i https://pypi.tuna.tsinghua.edu.cn/simple \
    torch==2.6.0 torchvision==0.21.0 ultralytics onnx onnxruntime onnxslim \
    opencv-python-headless pillow pyyaml matplotlib pandas
# 校验：torch 2.6.0+cu124 · cuda_available True · NVIDIA GeForce RTX 4050 Laptop GPU
```

> Linux 版 PyPI 的 `torch` wheel 自带 CUDA 运行时（含 cudnn/cublas 等 nvidia-* 依赖），
> **不需要** 额外指定 `--index-url download.pytorch.org`；Windows 版才是 CPU build。

### 10.2 两个"卡住"陷阱（都已修）

| 现象 | 根因 | 处理 |
|---|---|---|
| 训练开头空转数分钟 | ultralytics 8.4 的 **AMP 自检**会下载 GitHub 上的 `yolo26n.pt`，本机 GitHub 不可达 → 反复 30s 超时重试 | `config/train_v3.yaml: amp: false`（`train_detection.py` 已支持从 config 读 amp）；yolov8n@640/batch16 在 6 GB 显存上不需要 AMP |
| 权重找不到 | `project="runs/detect"` 是**相对路径**时，ultralytics 会拼到 `SETTINGS.runs_dir` 后 → 实际落在 `runs/detect/runs/detect/train` | `train_detection.py` 改用 `os.path.abspath("runs/detect")`，输出固定在 `runs/detect/train/` |

另外把 DejaVuSans 拷成 `~/.config/Ultralytics/Arial.ttf`，避免画图时再去 GitHub 下字体。

### 10.3 结果（GPU vs CPU）

```bash
bash tools/retrain_v3_gpu.sh     # GPU 训练（约 4 分钟跑完 33 epochs，早停）
bash tools/finish_v3_gpu.sh      # 真实域评估 → 导出量化 → ONNX 评估 → 部署
```

| 指标（真实域验证集 28 实拍 + 10 生成） | CPU（v3a） | **GPU（v3gpu）** |
|---|---|---|
| YOLO mAP50 | 0.8502 | **0.9427**（最佳 epoch 20） |
| YOLO mAP50-95 | — | **0.6048** |
| ONNX FP32 精确率 P | 0.9615 | 0.8378 |
| ONNX FP32 召回率 R | 0.6579 | **0.8158** |
| ONNX FP32 F1 | 0.7812 | **0.8267** |
| 训练耗时 | 数小时（未跑完） | **≈4 分钟**（3.4 GB 显存，4.5 it/s） |
| INT8 动态 P/R/F1 | — | 0.8649 / **0.8421** / **0.8533**（3.36 MB） |

赛项验收线 `detect_map50 ≥ 0.80` → **达标（0.9427）**。
精确率略降、召回率大涨，对分拣任务更有利（漏检比误检代价高）。

### 10.4 静态 INT8 的"假成功"已堵住

`tools/export_quantize.py` 里 QDQ 静态量化只做"能否加载"校验，结果**输出全 0 也能通过**，
并按"静态优先"把废模型复制成 `best_int8.onnx` —— 本次实测最大置信 0.000、检出 0。
现已加 `sanity_onnx()`：用真实图跑一遍，**有检出才认**，否则弃用并提示走 TensorRT：

```
静态 INT8 输出无效（最大置信 0.000，检出 0）→ 弃用
```

所以 PC/容器侧用 **FP32**（35 ms/帧，1280²）或 **动态 INT8**（3.36 MB，精度还略高但慢 3 倍）；
**Jetson 上用 TensorRT INT8**（`tools/build_trt_engine.py`）才是真正的加速路径。

### 10.5 部署与在线验证

```bash
# 已自动部署到 sort-web（供网关/容器加载）
sort-web/models/yolo/best.onnx        12.27 MB  FP32
sort-web/models/yolo/best_int8.onnx    3.36 MB  动态 INT8
```

用 sort-web 自己的服务代码（`server/yolo_server.py`, ENGINE=opencv）加载新模型实测：

| 验证 | 结果 |
|---|---|
| `/health` | `ok:true, engine:opencv_dnn, model:best.onnx` |
| 12 张实拍 val 图逐个 `/detect` | 11/12 检出，最高置信 0.527~0.965，单帧 50~71 ms |
