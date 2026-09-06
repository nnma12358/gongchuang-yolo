# YOLOv8n 自训练（第二版）—— 智能分拣物料检测/属性模型训练代码
>
> 依据根目录 `yolo实现思路.txt` 第二版：第一步实验(官方预训练)已证明 COCO 模型
> 对分拣色块零检测，本仓库实现**自训练全流程**：数据准备 → 检测训练 → 属性分类器训练
> → ONNX/TensorRT 导出 → 评估。数据由**你自己采集标注**（本仓库只产出训练代码）。

## 1. 环境（本机已验证）

```bash
# 使用 miao_llm_env（已装 torch 2.6.0+cpu / torchvision 0.21.0+cpu / ultralytics 8.4.60）
/home/xxxffyy/miao_llm_env/bin/pip install -r requirements.txt
# 全新环境: python3 -m venv venv && . venv/bin/activate && pip install -r requirements.txt
```

- 训练可在 CPU（20 核机，yolov8n@320 约 1~3 分钟/epoch）或 GPU（RTX 4050，装 CUDA torch 后 `--device 0`）进行；
- 权重获取：本机 GitHub 直连被拦截 → 见 **`python fetch_weights.py`**（本地 → hf-mirror → ultralytics 官方三级回退）。

## 2. 数据集规范（你负责采集标注）

```
data/                          # 你的数据集根目录(在 config/dataset.yaml 的 root 配置)
├── images/                    # 原图(俯视 1080p 为佳, JPG)
│   ├── train_0001.jpg ...
├── labels/                    # LabelImg 导出 txt: "cls cx cy w h"(0-1 归一化)
├── attributes/                # (可选, 方案A) 属性分类器数据
│   ├── color/{red,yellow,blue,green,black,lightblue}/...
│   ├── shape/{cube,cylinder,ball,cone,triangle}/...
│   └── defect/{clean,defect}/...
```

标注规范：
- 检测方案二选一：
  - **方案B（检测直出组合类别）**：`detection_classes: [red_cube, blue_cylinder, ...]`，每类 ≥60 件实例；
  - **方案A（通用检测+属性）**：`detection_classes: [goods]`，另采属性数据（推荐，适应抽签/决赛变化）。
- 数量建议：≥800 张 / 每种光照×摆放覆盖（密排、遮挡、色差批次、污渍等级）；
- 现场色差应对：训练增强已含 hsv_h 抖动，另建议采集 2~3 个色差批次。

## 3. 训练流程

```bash
# 0) 获取官方 yolov8n 基础权重(本机网络可用)
python fetch_weights.py

# 1) 校验标注/统计/划分/生成 data.yaml
python prepare_dataset.py --config config/dataset.yaml

# 2) 目标检测训练(方案B或A通用检测)
python train_detection.py --config config/dataset.yaml --epochs 120 --imgsz 640
#    GPU: python train_detection.py --device 0 --batch 16

# 3) (方案A) 属性分类器
python train_attributes.py --config config/dataset.yaml --task color
python train_attributes.py --config config/dataset.yaml --task shape
python train_attributes.py --config config/dataset.yaml --task defect

# 4) 导出 ONNX + 生成 Jetson TensorRT 命令
python export_models.py --config config/dataset.yaml --imgsz 640

# 5) 评估
python eval_detection.py --config config/dataset.yaml --imgsz 640
python eval_pipeline.py  --config config/dataset.yaml
```

或一键：`bash scripts/run_all.sh config/dataset.yaml`

## 4. 输出与验收

```
runs/detect/train/weights/best.pt   # 检测最优权重(→ 复制为 runs/detect/best_final.pt)
runs/attr/<task>/best.pt / best.onnx
exports/trt_commands.sh             # 在 Jetson 上执行生成 FP16 engine
reports/eval_report.json            # mAP/每类AP/小目标召回
reports/pipeline_report.json        # 属性准确率/色差敏感性
```

| 指标 | 目标 |
|------|------|
| 检测 mAP@0.5 | ≥0.95 |
| 颜色/形状分类 | ≥98% |
| 组合端到端 | ≥95%（±5% 色相信移） |

## 5. 部署（Jetson）

1. 把 ONNX/engine 拷到 Jetson，`bash exports/trt_commands.sh` 生成 FP16 引擎；
2. `vision_node.py` 配置 `config/yolo.yaml`: `model_path` 指向引擎/onnx；
3. JetPack 4.6.1 = Python 3.6：Jetson 上**不装 ultralytics**，推理走 OpenCV DNN/TensorRT（见 5.智能分拣机器人项目）。

## 6. Git 管理

```bash
git log --oneline          # 本仓库已按 脚手架/脚本/文档 分步提交
git add . && git commit    # 你的模型/数据不入库(data/、runs/、exports/、models/*.pt 已忽略)
```
