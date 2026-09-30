# 现场尺度微调 v2（60 轮，2026-09-30）

## 训练

| 项 | 值 |
|---|---|
| 数据 | mix_v3 470 + 现场几何合成 600 + 真实标注 41 + 真实硬负样本 132 → **train 1054 / val 189** |
| 配置 | 60 轮，imgsz 640，**batch 8 / workers 4 / cache false / amp true**（防 OOM 调优）|
| 基线 | `runs/best_v3_gpu.pt` |
| 结果 | **val mAP50 = 0.9610，mAP50-95 = 0.8403** |
| 内存 | 峰值 3.7G；用独立 cgroup（上限 5G）跑，超限只杀训练、不影响主机与 DSH |
| 权重 | `runs/best_overhead_v2.pt` |

## 导出

`exports_overhead_v2_fp32.onnx`（12.3MB）· 输入 `[1,3,640,640]` · 输出 `[1,5,8400]`
→ **与设备端 `yolo_server` 期望一致，trtexec 重建无需改代码**

## 真实数据实测（阈值 0.35，对照机器草稿标注）

| 模型 | TP | FP | FN | 精确率 | 召回率 | 平均置信 |
|---|---|---|---|---|---|---|
| 现役（scale 不匹配） | 0 | 0 | 184 | 0% | **0%** | — |
| **新模型 v2** | **28** | **4** | 156 | **88%** | **15%** | **0.524** |

24 张人工复核实拍集：现役 22/24 → **新模型 24/24**（无退化）

**结论**：召回 0% → 15%，精确率 88%，假检 0.1/帧（正对"工具箱假检消除"验收标准）。
召回仍偏低，根因是**真实标注只有 41 张且同一姿势、且是机器草稿** —— 靠 (c) 扩充姿势 + 人工复核提升。

## 部署（待设备在线）

```bash
# 设备端
cd ~/sort-jetson-deploy-20260917
cp -a models/detect models_backup_$(date +%m%d_%H%M)          # 先备份
# 从 PC 传新 ONNX 后：
bash scripts/build-trt-on-jetson.sh                            # 用 trtexec 重建 FP16 引擎（必须在设备上）
docker-compose -f docker-compose.jetson.yml up -d --no-build yolo
# 回归：同一套脚本对比改前后
python3 scripts/regression_live_sample.py --base http://<jetson-ip> --n 60
```
