# YOLOv8n 自训练（第三版）—— 智能分拣赛项视觉方案

> 依据赛项说明（2027 中国大学生工程实践与创新能力大赛 · 智能+工程创新赛道 · 智能分拣赛项）：
> **货物 ≤40mm、常用形状+常用颜色、表面附有模拟污渍/缺陷图形、文字、二维码；六个储物盒（1–6），每盒 ≤4 件；
> 形状相同且颜色相同必须分拣到同一个储物盒；一次只能分拣一件货物；分拣信息显示完成后再分拣下一件。**

第一步实验（官方预训练模型）已证明 COCO 模型对分拣货物零检测，因此本仓库提供**自训练全流程**；
第三版针对比赛约束做了三项关键优化：**小目标（≤40mm）**、**属性分层（颜色/形状/污渍缺陷）**、**标记解码（二维码/文字）**。

## 1. 视觉方案（三层分工）

```
托盘(≥160×160mm)  ──►  ① 检测层  YOLOv8n @960  →  只检 "goods"（托盘 ROI 内，小目标）
                                      │
                                      ├─►  ② 属性层  小 CNN 96×96   →  颜色 / 形状 / 污渍缺陷
                                      │
                                      └─►  ③ 标记层  OpenCV        →  二维码 / 文字（QRCodeDetector）
                                                    │
                        (形状, 颜色) ───────────────┴──►  储物盒编号 1–6（同形同色同盒，每盒 ≤4）
```

为什么不让检测器直出“红正方体”这类组合类别？
- 决赛货物种类、颜色、信息**以现场为准**；组合类别一旦变化就要重训检测器；
- 属性拆开后：新增颜色/形状只需补属性样本，检测器不动（赛项兜底能力更强）；
- 属性分类器 96×96 输入、80 万参数，CPU 单张 1ms 级，适合 Jetson Nano 上跑。

## 2. 快速开始

```bash
# 0) 环境（本机已验证）：torch 2.6.0+cpu / ultralytics 8.4.60
/home/xxxffyy/miao_llm_env/bin/pip install -r requirements.txt
python fetch_weights.py                      # 获取 yolov8n 基础权重（本地→hf-mirror→官方）

# 1) 采集标注：images/ + labels/（LabelImg 导出 YOLO txt），建议 ≥800 张，
#    覆盖：密排、遮挡、2–3 个色差批次、污渍/缺陷样本、贴有二维码/文字的货物正反面
#    文件名或类别名带颜色/形状关键字（red_cube.jpg、prism5_4.jpg）→ 可自动生成属性集

# 2) 校验标注 + 划分 + 生成 data.yaml
python prepare_dataset.py --config config/dataset.yaml

# 3) 属性数据集自动生成（无需重复标注）
python scripts/crop_attributes.py --config config/sorting_competition.yaml

# 4) 表面标记解码（二维码/文字/污渍占比 → marks/marks.json，可视化核对加 --annotate）
python scripts/decode_marks.py --config config/sorting_competition.yaml \
    --images data/images/train --annotate

# 5) 检测训练（小目标：imgsz 960，copy_paste + 提前关闭 mosaic）
python train_detection.py --config config/sorting_competition.yaml --epochs 150

# 6) 属性分类器（颜色 / 形状 / 污渍缺陷）
python train_attributes.py --config config/sorting_competition.yaml --task color --export-onnx
python train_attributes.py --config config/sorting_competition.yaml --task shape --export-onnx
python train_attributes.py --config config/sorting_competition.yaml --task stain --export-onnx

# 7) 导出（ONNX + Jetson TensorRT 命令）
python export_models.py --config config/sorting_competition.yaml --imgsz 960

# 8) 验收评估（自动对照赛项验收线，含“同形同色同盒”规则校验）
python eval_detection.py --config config/sorting_competition.yaml --imgsz 960
python eval_pipeline.py  --config config/sorting_competition.yaml
```

一键：`bash scripts/run_all.sh config/sorting_competition.yaml`

## 3. 配置文件

| 文件 | 用途 |
|---|---|
| `config/sorting_competition.yaml` | **比赛版主配置**：ROI、imgsz 960、属性类别、标记解码阈值、六盒规则、验收线 |
| `config/dataset.yaml` | 数据集/训练通用配置（与本版对齐） |
| `config/sorting_rules.yaml` | 储物盒与分拣规则（六个盒、每盒 4 件、同形同色同盒、一次一件、掉落结束） |

## 4. 本版新增/优化点

| 优化 | 说明 |
|---|---|
| **小目标适配** | 货物 ≤40mm：`imgsz 960`、`scale 0.5`、`copy_paste 0.3`、`close_mosaic 30`、`rect=False`、托盘 ROI 裁剪 |
| **属性分层** | 检测只出 `goods`；颜色 9 类 / 形状 7 类 / 污渍缺陷 3 类，各自小 CNN + ONNX 导出 |
| **污渍缺陷识别** | 两种手段互为校验：属性分类器（`stain` 任务，训练增强模拟暗斑/失焦）+ 暗斑占比阈值（`decode_marks.py`） |
| **二维码/文字** | `scripts/decode_marks.py`：`QRCodeDetector` 多码解码 → 任务码/文字；导出 `marks/marks.json` 供复核 |
| **属性集自动生成** | `scripts/crop_attributes.py`：从检测标注直接裁出 96×96 属性样本，省一遍标注 |
| **节拍约束** | `decision.confirm_frames=5`、`single_item=true`（一次一件）、托盘有货/空盘校验，避免空抓与掉落 |
| **验收自动化** | `eval_pipeline.py` 输出赛项对照表（mAP50 / 颜色 / 形状 / 污渍 / 组合端到端 / 同盒一致性） |

## 5. 输出与验收

```
runs/detect/train/weights/best.pt     检测最优权重（→ runs/detect/best_final.pt）
runs/attr/{color,shape,stain}/best.pt 属性分类器（best.onnx 亦在此）
marks/marks.json                      每张图的二维码/文字/污渍标记索引
reports/eval_report.json              检测 mAP / 每类 AP / 小目标召回
reports/pipeline_report.json          属性准确率 + 色差敏感性 + 赛项验收对照 + 六盒规则一致性
```

| 指标 | 赛项验收线 | 对应配置项 |
|------|-----------|-----------|
| 检测 mAP@0.5 | ≥0.95 | `acceptance.detect_map50` |
| 颜色 / 形状准确率 | ≥0.98 | `acceptance.color_acc` / `shape_acc` |
| 污渍缺陷准确率 | ≥0.95 | `acceptance.stain_acc` |
| 二维码解码率 | ≥0.98 | `acceptance.qr_decode_rate` |
| 组合端到端 | ≥0.95 | `acceptance.combo_end_to_end` |
| 同形同色同盒一致性 | =1.00 | `acceptance.bin_consistency` |

## 6. 部署（Jetson Nano / 网关联动）

1. `export_models.py` 产出 ONNX，Jetson 上按 `exports/trt_commands.sh` 生成 FP16 engine；
2. 检测 + 属性 ONNX 交给机器人端推理节点（JetPack 4.6.1 = Python 3.6，**不装 ultralytics**，走 OpenCV DNN / TensorRT）；
3. 一件货物分拣完成后，机器人端上报网关，显示屏按赛项要求展示“投放顺序 / 货物名称 / 图片 / 分拣成功总数量”：

```bash
python3 scripts/sort_client.py --gateway http://localhost \
    --class prism5_4                       # 或直接传 shape/color/marks/text/qr
```

网关侧规则（`sort-web` 项目 `docker-compose.jetson.yml` 部署）：
- 同形同色 → 同一储物盒；新组合依次占用 1→6 号空盒；满盒拒绝继续放置；
- 显示屏闩锁 `DISPLAY_HOLD_SECONDS`（默认 3s）：期间拒绝下一件上报（否则不计分）；
- 掉落（装置内/外）→ `/api/round/fault` → 本轮结束、停止计时。

## 7. Git 管理

```
git log --oneline          # 脚手架 / 训练脚本 / 使用说明 / 赛项视觉方案（本版）
git add . && git commit    # 数据与权重不入库（data/、runs/、exports/、models/*.pt 已忽略）
```
