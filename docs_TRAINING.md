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
cp exports_clean/best_fp32.onnx ../sort-web/models/detect/goods_yolov8n_640_fp32.onnx   # 容器默认（OpenCV DNN）
cp exports_clean/best_int8.onnx ../sort-web/models/detect/goods_yolov8n_640_int8.onnx  # 可选（ONNX Runtime）
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

## 7. TensorRT（Jetson 上的加速）

`tools/build_trt_engine.py`：在 Jetson 上构建 **FP16 / INT8（熵校准）** 引擎，输出 `*.engine` 与 `calib.cache`。

⚠ **精度选择看芯片，不是越高压缩越好**：

| 平台 | GPU | 建议 | 原因 |
|---|---|---|---|
| **Jetson Nano**（Tegra X1） | Maxwell **SM 5.3** | **FP16** | Maxwell 没有 INT8 张量核、也没有 DP4A 指令，TRT 的 INT8 只能用 FP16 模拟 → 基本不提速，还掉精度 |
| Jetson TX2 | Pascal SM 6.2 | FP16（INT8 有小幅收益） | 有 DP4A |
| Xavier NX / AGX | Volta SM 7.2 | INT8 | 有 INT8 张量核，比 FP16 再快 1.5~2× |
| Orin | Ampere SM 8.7 | INT8 / FP16 | INT8 收益最大 |

引擎与 **GPU 架构 + TensorRT 版本绑定**，必须在设备上构建（PC 上构建的拷过去加载会失败）。


```bash
# PC 上先自检（无需 TensorRT）
python3 tools/build_trt_engine.py --check --onnx exports_clean/best_fp32.onnx --calib-dir data/mix_clean/images/train
# Jetson 上构建（校准集建议用现场实拍 100~300 张）
python3 tools/build_trt_engine.py --onnx exports_clean/best_fp32.onnx \
    --calib-dir data/mix_clean/images/train --calib-n 200 --imgsz 640 --out engines --bench 20
```
部署：`cp engines/*.engine ../sort-web/models/detect/`，compose 里设 `ENGINE=trt`
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
**Jetson Nano 上用 TensorRT FP16**（`sort-web/scripts/build-trt-on-jetson.sh`）才是真正的加速路径（Nano 无 INT8 硬件）。

### 10.5 部署与在线验证

```bash
# 已自动部署到 sort-web（供网关/容器加载）
sort-web/models/detect/goods_yolov8n_640_fp32.onnx   12.27 MB  FP32
sort-web/models/detect/goods_yolov8n_640_int8.onnx   3.36 MB  动态 INT8
```

用 sort-web 自己的服务代码（`server/yolo_server.py`, ENGINE=opencv）加载新模型实测：

| 验证 | 结果 |
|---|---|
| `/health` | `ok:true, engine:opencv_dnn, model:best.onnx` |
| 12 张实拍 val 图逐个 `/detect` | 11/12 检出，最高置信 0.527~0.965，单帧 50~71 ms |

---

## 11. 人工复核后的真实域结果（本轮，最关键的一次提升）

复核 193 张（`docs_REVIEW.md` 的流程）→ 写回源数据集 → 分组切分重训，**同一套协议**只换了标签：

| 指标（真实域验证集 40 实拍 + 10 生成，无泄漏） | 复核前 | **复核后** |
|---|---|---|
| **mAP50** | 0.7738 | **0.9927** |
| **mAP50-95** | 0.3665 | **0.7158** |
| 精确率 P | 0.7149 | **0.9929** |
| 召回率 R | 0.7676 | **0.9600** |
| 训练曲线（mAP50，逐 epoch） | 0.82→0.61→0.75 反复跳 | **epoch3 起稳定 0.99+，不再塌陷** |
| 模型↔标注 IoU（val，留出集） | 0.758（<0.5 占 10.3%） | **0.853（<0.5 占 0.0%）** |

**复核工作量与收益**：164 张有标注的图里 **123 张被人工修正**（75%），中心位移中位数 8%、
最大 69%（`real_00150` 从画面左下角改到上方中央）；另有 29 张自动预标注失败的图从零标注。
也就是说：**之前那 0.77 里，绝大部分损失不是模型不行，而是标签在教它错的东西。**

### 11.1 训练震荡的真正原因（结论）

复核前 mAP 反复跳（0.85→0.2→回升）有两个原因，**标签噪声是主因**：

1. **主因**：每轮拟合的是不同的"错目标"，验证集又是同样带噪的标注 → 指标随噪声起舞
2. **次因**：val 只有 49 张，单张错标就能让 mAP 跳 2~3 个点

复核后同一份代码、同一套超参，训练曲线从 epoch3 起就贴在 0.99，**早停触发时仍在高位**。

### 11.2 部署模型的置信度阈值（实测扫描）

用**人工核验**的 val（50 张）扫描部署用 ONNX（`sort-web/models/detect/goods_yolov8n_640_fp32.onnx`）：

| conf | 精确率 P | 召回率 R | F1 |
|---|---|---|---|
| 0.25 | 0.7353 | **1.000** | 0.8475 |
| 0.35 | 0.8197 | **1.000** | 0.9009 |
| **0.45** | 0.9091 | **1.000** | **0.9524** |
| 0.55 | 0.9231 | 0.960 | 0.9412 |
| 0.65 | **1.000** | 0.960 | 0.9796 |

→ `sort-web/server/yolo_server.py` 的 `CONF_THRES` 默认值已从 0.35 提到 **0.45**：
在不丢目标（R=1.00）的前提下把背景误检压掉一半。现场若发现漏检可回落到 0.35。

### 11.3 本轮产物

```bash
runs/best_after_review.pt                      # 复核后权重（mAP50 0.9927）
exports_v3gpu/best_fp32.onnx                   # 部署用 FP32（12.27 MB）
exports_v3gpu/best_int8_dynamic.onnx           # 动态 INT8（3.36 MB）
sort-web/models/detect/goods_yolov8n_640_{fp32,int8}.onnx  # 已部署
runs/eval_before_review.json / eval_after_review.json   # 两份同口径评估报告
```

---

## 12. 背景误检的治理（负样本挖掘 + ROI 双保险）

### 12.1 根因：训练集里一张负样本都没有

训练日志里一直写着 **`0 backgrounds`** —— 每张训练图都恰好有一个货物框，
模型**从没学过"什么不是货物"**。于是实验室设备、纸边、桌沿这些高对比区域被判成货物：

| 诊断（161 张实拍图，conf≥0.25） | 数值 |
|---|---|
| 误检框总数 / 每图 | 36 个 / **0.22 个/图** |
| 位置分布 | **下右 10 · 上左 8 · 下左 7 · 上右 6**（全在画面四角）· 中中 3 |
| 能过部署阈值 0.45 的 | 15 个 |

### 12.2 解决一：难负样本挖掘（`tools/mine_negatives.py`）

不需要重新拍照：用当前模型跑全部实拍图 → 挑出与人工标注 IoU<0.3 的框 = 误检 →
把误检区域连上下文裁下来 (**难负样本**) + 在无货物区域按边缘能量挑"最花"的窗口 (**背景负样本**)，
统一以**空标签**加入训练集。

⚠ 两个必须注意的坑（都已在脚本里处理）：

1. **不能用 IoU 判断"窗口里有没有货物"**：窗口 640²、货物 150² 时，货物整个在窗口里 IoU 也只有 0.05，
   会被误判成干净背景 → 造出"含着货物的空标签"，教模型"这东西不算货物"，比误检更糟。
   脚本按**物体被框住的比例**判断，且零容忍（物体边缘都不许进窗口）。
2. 上一版就因为这条规则松，149 张"负样本"里有近一半其实含着货物（拼图 `sheet.jpg` 一眼看出）。

| 轮次 | 挖到 | 累计负样本 |
|---|---|---|
| 第 1 轮（复核后模型） | 难例 19（另 17 个因沾到真货物被丢弃）+ 背景 95 | 114 |
| 第 2 轮（负样本重训后模型） | 难例 3 + 背景 106 | **223** |

### 12.3 效果（人工核验 val：50 张实拍图，每张恰好 1 个真货物）

| 指标 | 复核后（无负样本） | **+223 张负样本** |
|---|---|---|
| mAP50 | 0.9927 | **0.9950** |
| mAP50-95 | 0.7158 | **0.7617** |
| 精确率 P / 召回率 R | 0.993 / 0.960 | **1.000 / 0.995** |
| **误检/图 @conf0.35** | 0.333 | **0.000** |
| **误检/图 @conf0.45** | 0.139 | **0.000** |
| 纯背景图上的检出（200 张，理论应 0）@0.25 | 0.211 个/图 | **0.035 个/图**（−83%） |
| 模型自身再挖误检（161 张） | 36 个（0.22/图） | **13 个（0.08/图）** |

**结论：加负样本后误检清零，mAP 不降反升**（模型终于学会了"背景长什么样"）。
阈值也随之回到 **0.35**（0.35~0.55 区间误检均为 0，取低值给现场新货物留余量）。

### 12.4 解决二：推理端托盘 ROI（`ROI` 环境变量）

模型负样本治的是"训练集里见过的背景"；现场还会出现**训练照片里没有的东西**
（人手、机械臂、临时放的杂物）。货物一定在托盘上 → 用区域兜底最便宜可靠：

```bash
# docker-compose.jetson.yml（sort-vision）
ROI=0.06,0.06,0.94,0.94                 # 矩形，归一化
ROI=0.10,0.08;0.90,0.10;0.92,0.90;0.08,0.88   # 或按托盘标定填多边形
```

实现要点：`detect_core.filter_roi()` 按**框中心**判定（射线法，支持多边形），
**两条检测路径都挂了**——经典引擎走 `analyze()`，yolo 容器路径走 `detect_via_services()`
（这条当初差点漏掉，只挂一条会导致 ROI 在生产路径上完全不生效）。
丢了多少个框会在 `/detect/frame` 的 `roi_dropped` 字段里报出来，便于现场确认过滤在工作。

实测（80 张实拍图，conf≥0.10 的全部候选 89 个）：启用 0.06~0.94 矩形后丢弃 4 个（4.5%），
`real_00011`/`real_00023` 从 3 框收敛到 1 框。

### 12.5 误检率做成可回归的指标

`tools/eval_fp.py` 直接量 **误检/图 · 漏检/图 · 纯背景图检出数**，并按置信度扫描推荐阈值；
已并入 `tools/eval_split.sh`，每次训练都会打印，**误检变多会立刻暴露**。

```bash
python tools/eval_fp.py --weights runs/best_after_neg2.pt --data data/mix_v3 \
    --data2 data/negatives --device 0
```

---

## 13. 实机跑通记录（Jetson Nano 4GB / L4T r32.5.2 = JetPack 4.5.1）

> ⚠ 现场这台 Nano 是 **JetPack 4.5.1**（不是 4.6.x），GPU 是 **Tegra X1 = Maxwell / ARMv8.0**。
> 这决定了后面所有依赖取舍 —— **pip 上稍新的 aarch64 轮子都是按 ARMv8.2 编的，在这颗 CPU 上直接 SIGILL**。

### 13.1 依赖取舍（实测踩坑表）

| 组件 | 试过的方案 | 结果 | 最终采用 |
|---|---|---|---|
| numpy | pip 1.19.5 / 1.18.5 | ❌ Illegal instruction | **pip 1.19.2**（实测可用的最后一个版本） |
| OpenCV | pip opencv-python-headless 4.5.4 | ❌ Illegal instruction | **apt 的 python3-opencv 3.2**（只做编解码） |
| OpenCV | 宿主自编 3.4.5（挂载进容器） | ⚠ 能导入但解析不了 YOLOv8 的 ONNX | 放弃 |
| ONNX 推理 | cv2.dnn | ❌ apt 的 3.2 **没有 dnn 模块** | — |
| ONNX 推理 | onnxruntime 1.9.0 + numpy 1.13.3 | ❌ 段错误 | — |
| ONNX 推理 | **onnxruntime 1.9.0 + numpy 1.19.2** | ✅ 可用（~750ms/帧 CPU） | 备用路径 `ENGINE=ort` |
| ONNX 推理 | **TensorRT FP16 引擎** | ✅ **46.6ms/帧**，精度与 FP32 完全一致 | **主路径 `ENGINE=trt`** |

**教训**：在被"旧 ARMv8.0 + 老 JetPack"锁死的板子上，**能用 apt/官方为设备编译的二进制就别用 pip 轮子**。

### 13.2 TensorRT FP16（Nano 上的最优解）

```bash
# 在 Jetson 上、项目根目录执行：构建 FP16 引擎（约 15 分钟，TRT 7.1 的 tactics 搜索较慢）
bash scripts/build-trt-on-jetson.sh --check        # 先体检
bash scripts/build-trt-on-jetson.sh                # 构建 models/detect/goods_yolov8n_640_fp16.engine
```

**为什么是 FP16 不是 INT8**：Tegra X1（Maxwell, SM 5.3）**没有 INT8 张量核、也没有 DP4A 指令**，
TRT 的 INT8 只能靠 FP16 模拟 —— 基本不提速还掉精度。INT8 留给 Xavier/Orin。

实测对比（同一张 1280² 实拍图，含 JPEG 解码 + 预处理 + NMS 的**完整 detect()**）：

| 路径 | 单帧耗时 | 检测结果 |
|---|---|---|
| ONNX Runtime（CPU，numpy 1.19.2） | 1035 ~ 1443 ms | box [464,176,**733**,432] conf 0.947 |
| **TensorRT FP16（GPU）** | **218 ~ 244 ms** | box [464,176,**732**,432] conf **0.947** |

`trtexec` 纯推理基准：**GPU 前向 46.6ms · 吞吐 21.2 qps**（对比 CPU ORT ~750ms → **约 16 倍**）。
识别循环帧率 **0.6 → 1.2~1.3 fps**；配合 `AUTO_CONFIRM_FRAMES` 从 4 降到 2，确认一件货物的时间
从约 6.7 秒降到约 1.6 秒。

**容器里怎么跑 TRT（关键点）**：

1. 镜像内 apt 装 `python3-libnvinfer`（TRT 7.1.3 运行时+Python 绑定）+ `cuda-cudart/nvrtc/cublas/cudnn`；
2. 显存管理用 **ctypes 直接调 libcudart**（`cudaMalloc/cudaMemcpyAsync/cudaStreamCreate`）——
   JetPack 上装不到 pycuda 轮子，而 `tensorrt` 绑定本身不需要它；
3. compose 里给 sort-yolo 开 `privileged: true`（拿到 `/dev/nvhost-*` GPU 设备节点）
   并挂载宿主 `/usr/lib/aarch64-linux-gnu/tegra`（`libcuda.so.1` 驱动只能来自宿主）+ `LD_LIBRARY_PATH`；
4. 引擎与 GPU 架构 + TRT 版本绑定，**必须在设备上用本机 trtexec 构建**。

### 13.3 与现场 ROS2 容器共存（4GB 内存的分配）

现场 `ros2_arm_container`（`wheeltec_ros2_astra:foxy`）曾被 **OOM 杀掉（Exited 137）**。
因此给本项目的 4 个容器都加了硬限额（compose v1 里实际生效的是 `deploy.resources.limits`）：

| 容器 | CPU 上限 | 内存上限 | 实测占用 |
|---|---|---|---|
| sort-yolo（TRT） | 2.0 核 | **1200M**（TRT/CUDA 上下文本身 500~700MB） | 876 MB |
| sort-vision | 1.5 核 | 512M | 110 MB |
| sort-cnn | 0.5 核 | 384M | 92 MB |
| sort-gateway | 0.5 核 | 320M | 37 MB |
| **合计上限** | 4.5 核 | **2.4 GB** | ~1.1 GB |

即：**给现场 ROS2/机械臂容器始终留出 ≥1.5GB 内存与 CPU 余量**，且不改动 docker 守护进程
（不重启 docker，不碰现场容器）。

---

## 14. 与现场 ROS 2 栈集成（实测打通）

### 14.1 现场实际运行方式（从 `.bash_history` 与源码确认）

现场机器人的 ROS 2 栈跑在**他们自己的容器** `ros2_arm_container`（镜像 `wheeltec_ros2_astra:foxy`）里，
启动命令是手工 `docker exec` 进去拉起，关键环境：

```bash
ROS_DOMAIN_ID=95            # 他们导出在 shell 里，不在容器 env
ROS_LOCALHOST_ONLY=1        # 只走 loopback
RMW_IMPLEMENTATION=rmw_cyclonedds_cpp
python3 -u .../smart_sorting_dry_run --config .../dry_run_config.json   # 他们自己的感知（DRY_RUN）
```

### 14.2 机械臂的真实接口（从源码确认，不是猜的）

`wheeltec_arm_driver/src/driver_node.cpp:78`：

```cpp
create_subscription<sensor_msgs::msg::JointState>("arm_joint_command", 100, &WheeltecArmDriver::joint_state_callback, this);
```

| 项 | 值 |
|---|---|
| 命令话题 | **`arm_joint_command`** |
| 消息类型 | **`sensor_msgs/msg/JointState`**（`name` = joint1..joint6，`position` = 弧度） |
| 其它订阅 | `arm_teleop`、`cmd_vel`、`joint_states`、`color_result`、`face_ik_result`、`gesture_arm`、`voice_joint_states` |
| 发布 | `joint_states`、`odom`、`pose`、`imu`、`PowerVoltage` |
| 自定义消息 | `wheeltec_arm_interfaces`：`ArmTarget`、`ArmTargetPosition`、`PickAndPut`、`ColorIkResult` … |
| 相机话题 | `/camera/color/image_raw`、`/camera/depth/image_raw`、`/camera/color/camera_info` |

桥接容器因此改成：**默认发布 `arm_joint_command` 的 `JointState`**（`ARM_TOPIC`/`ARM_MSG` 可配），
请求里带 `joints=[j1..j6]` 才下发；只给笛卡尔目标时会**明确拒绝并提示需要逆解**（避免"以为动了其实没动"）。

### 14.3 三个容器共存的对齐要点（都踩过）

| 对齐项 | 值 | 不对齐的后果 |
|---|---|---|
| `ROS_DOMAIN_ID` | **95** | `ros2 topic list` 空空如也，互相发现不了 |
| `RMW_IMPLEMENTATION` | `rmw_cyclonedds_cpp` | 不同 DDS 实现之间完全不可见 |
| `ROS_LOCALHOST_ONLY` | `1` | 与现场不一致时发现行为不同（都用 host 网络时靠 lo 单播） |
| 网络模式 | `host` | 非 host 时 loopback 单播发现失败 |
| **`runtime: nvidia`** | 桥接容器需要 | 现场镜像里的 OpenCV 是 CUDA 版，缺 libcublas 直接 import 失败 |
| daemon.json | 必须注册 `runtimes.nvidia` | 现场容器报 `Unknown runtime specified nvidia` 起不来 |

### 14.4 联调实测

```
# 1) 我们的桥接容器与他们的容器在同一 DDS 域：发布 6 条，收到 6 条 ✓
[收到] hello-from-sort-gateway-0 … -5            共收到 6 条

# 2) 网关 → 桥接 → 机械臂话题（HTTP POST /execute 带 joints）
POST /execute {"joints":[0.10,-0.45,0.60,0.20,0.00,1.20], "joint_names":["joint1".."joint6"]}
→ 他们的容器收到：
  [收到关节指令] names=['joint1','joint2','joint3','joint4','joint5','joint6']
                positions=[0.1, -0.45, 0.6, 0.2, 0.0, 1.2]  ✓
```

### 14.5 还差的一步（不在我们这侧）

现场 `dry_run_config.json` 的 `calibration_status = MISSING_FINAL_CALIBRATION`、
`motion_authorized = false` —— **手眼标定与逆解还没完成**。要让整条链闭环还缺：

1. **逆解**：我们有笛卡尔抓取点（`pose.py` 的 `grasp_from_detection`），现场用 MoveIt 2
   (`mini_mec_six_arm_moveit_config`) 或 `ColorIkResult` 做 IK → 得到 6 个关节角；
2. **手眼标定**：`~/ros2_ws/passive_handeye_calibration` 是他们现成的标定工程；
3. 拿到关节角后 POST 给桥接（或让网关自动调用 IK）：

```bash
curl -X POST http://127.0.0.1:8120/execute -H 'Content-Type: application/json' \
  -d '{"seq":1,"name":"白色正十二面体","bin":3,"joints":[j1,j2,j3,j4,j5,j6]}'
```
