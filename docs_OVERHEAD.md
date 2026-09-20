# 现场俯拍「真实尺度」重训交接文档

> 起因：2026-09-20 实测发现，部署在 Jetson 上的视觉链路在现场顶置 Astra 相机下
> **真实货物 0 检出**，并把黑色工具箱误检成货物（60 帧里 48 帧有此假检）。
> 根因不是阈值、不是背景、不是引擎，而是**成像尺度**：货物在画面里只有约 30px，
> 而模型工作区间是 69~100px。详细数据见 `视觉测试结果/实拍/真实货物实测报告.md`。

## 一、现场几何（实测）

| 项 | 值 | 说明 |
|---|---|---|
| 相机到台面 | **≈660mm** | 深度中位数 663mm |
| 货物尺寸 | ≤40mm | 赛项规定 |
| 货物在画面里的像素 | **≈30px** | 640×480 画面 |
| 模型最佳区间 | 69~100px | 置信度 0.59~0.72（69/100px）|
| 生成器原设计 | `REAL["cam_height"]=320.0` | 注释写"现场实测"，实际差一倍 |

因此：**训练数据必须覆盖 30px 量级的货物**。合成数据已支持按现场几何生成
（`tools/synth_polyhedra.py --cam-height 660`，1280 图里 40mm 货物约 67px，
letterbox 到 640 后 ≈33px，与现场一致）。

## 二、数据采集（PC 直接拉现场相机，无需手动拍照）

现场已跑 `sort-ros-bridge`(:8123)；PC 能直连 `10.60.10.24:8123`，每帧约 77ms。

```bash
# 有货场景：现场把货物换位置/朝向/数量，边换边采（画面变化不大自动跳过）
python3 tools/capture_overhead.py --session tray_a --n 60

# 纯背景（无货物）→ 作为负样本抑制工具箱等误检
python3 tools/capture_overhead.py --session bg_lab --n 40
```

产出：`data/real_overhead/<session>/`
- `images/*.jpg` 彩色帧（640×480）
- `depth/*.png` **原始 16bit 深度（mm）** —— 测距与自动提议框都依赖它
- `depth_preview/*.png` 8bit 预览，仅供肉眼查看
- `labels/*.txt` 标注（初始为空，待标注）

也可以在别处拍好照片再导入：

```bash
python3 tools/import_real_photos.py --src ~/照片/现场俯拍 --session tray_phone
```

> ⚠️ 导入的照片**必须是现场相机同一视角/尺度**（货物约占画面宽 5%，即 640 宽时约 30px）。
> 手机近景特写解决不了现场漏检——那正是现有 24 张实拍集已经覆盖的域。

## 三、标注（深度 + RGB 双通道自动提议，人工只做删框）

白货白桌在 RGB 上分不出，但深度上比桌面高 17~23mm；黑货在深度图上没有回波
（红外被吸收），RGB 上却是明显暗块。两条通道合并后现场帧里 5~6 个货物**全部有候选框**，
另有 2~3 个杂物框需人工删除。

```bash
# 先预览（只出 _vis.jpg，不写标注）
python3 tools/propose_boxes.py --session tray_a

# 用 --roi 限定工作区可大幅减少杂物候选（x1,y1,x2,y2）
python3 tools/propose_boxes.py --session tray_a --roi 180,150,470,360 --write

# 人工复核（已有流程）
python3 tools/review_app.py --help
```

## 四、训练与导出（等你照片到位后一键跑）

```bash
bash tools/finetune_overhead.sh          # 合成(cam_height 660) → 组装 → 微调 → 评估 → 导出 ONNX
bash tools/finetune_overhead.sh --skip-synth
```

- 配方：`config/train_overhead.yaml`（imgsz 640 与现有引擎形状一致，无需改引擎；加强
  尺度/旋转/mosaic 增强以适配小目标）
- 组装规则（`tools/assemble_overhead_dataset.py`）：`bg_*` 会话作负样本；
  其它会话**只收标注非空的图**——有货无标注会把货物教成背景，脚本会明确告警并整批排除。
- 导出后需在 **Jetson 上用 trtexec 重建 FP16 引擎**（TRT 版本必须与设备一致），
  命令见 `tools/finetune_overhead.sh` 第 8 步打印的内容。

## 五、回归验收

```bash
python3 tools/capture_overhead.py --session regression --n 60 --marked
```

对照基线（修复前，2026-09-20 实测）：**真实货物 0/5~6 检出；48/60 帧存在黑色工具箱假检
（conf 0.365~0.487）**。验收要求：真实货物检出显著提升、工具箱假检消失、
`sort-yolo` 的 TRT 推理耗时仍在可接受范围（原 218~331ms/帧）。

## 六、本次顺带修掉的部署 bug（已推送）

- `depth_http_node.py`：ROS2 `Header` 没有 `seq` → 订阅回调每帧抛异常被丢弃
- `depth_http_node.py`：`/color.mjpg` 用 `Response` 包生成器 → 一直 500，改 `StreamingResponse`
- `vision_server.py`：`fetch_depth()` 缺 `import cv2` → 深度 2.5D 一直不可用
- compose：`RELIABLE_QOS` 默认改 0（现场 Astra 是 BEST_EFFORT 发布；BEST_EFFORT 订阅端
  对两种发布端都兼容，更稳）
