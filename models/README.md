# models/ —— 视觉模型目录（yolo 检测 + cnn 属性）

> ✅ **本目录已换成实训练模型**（不再是随机权重的管线验证模型）。
> 指标全部来自**真实域验证集**（28 张现场实拍 + 10 张生成图），不是合成域自评。

## 1. 目录约定

```
models/
├── yolo/
│   ├── best.onnx           # YOLO 目标检测 FP32（工创yolo 自训练导出，12.27 MB）
│   ├── best_int8.onnx      # 动态 INT8（3.36 MB，精度不降，CPU 上更慢；小体积/低带宽场景用）
│   └── classes.json        # 类别名（单类方案：{"0": "goods"}）
└── attr/
    ├── color.onnx          # 颜色分类（9 类）
    ├── color_classes.json
    ├── shape.onnx          # 形状分类（9 类，含现场货物 dodecahedron/cone）
    ├── shape_classes.json
    ├── stain.onnx          # 污渍/缺陷分类（clean/stain/defect）
    └── stain_classes.json
```

两个容器都用 `./models:/app/models:ro` 只读挂载 —— **换模型只需替换文件并重启对应容器**，不必重建镜像。

## 2. 当前模型与实测指标

| 文件 | 模型 | 实测指标 |
|---|---|---|
| `yolo/best.onnx` | YOLOv8n 单类 `goods`，真实域微调 | **mAP50 0.9427 / mAP50-95 0.6048**（真实域验证集：28 实拍 + 10 生成，最佳 epoch 20）；ONNX 侧 P 0.838 / R 0.816 / F1 0.827 |
| `yolo/best_int8.onnx` | 动态 INT8 量化 | P 0.865 / R 0.842 / F1 **0.853**、3.36 MB（88 ms/帧 CPU，比 FP32 慢 3 倍，仅图体积小） |
| `attr/color.onnx` | 颜色 CNN | 合成验证集 **100.00%**；跨域 8/8 实拍正确（white 0.99~1.00） |
| `attr/shape.onnx` | 形状 CNN（9 类） | 合成验证集 **99.77%**；跨域 4/8 —— **仍不足**，需每形状补 30~50 张实拍裁剪 |
| `attr/stain.onnx` | 表面 CNN | clean/stain/defect 三分类 |

在线验证（用本项目服务代码加载，非离线脚本）：

```bash
curl http://localhost:8101/health     # ok:true, engine:opencv_dnn, model:best.onnx
# 12 张实拍 val 图逐个 /detect → 11/12 检出，最高置信 0.527~0.965，单帧 50~71 ms（1280², CPU）
```

## 3. 各引擎的取舍（实测结论）

| 场景 | 用哪个 | 原因 |
|---|---|---|
| PC / x86 容器 | **FP32**（`best.onnx`） | 35 ms/帧；动态 INT8 反而慢到 120 ms |
| Jetson | **TensorRT INT8** | `工创yolo tools/build_trt_engine.py`，真正提速路径 |
| 带宽/存储受限 | 动态 INT8 | 3.36 MB，精度不降 |
| ~~ONNX Runtime 静态 INT8 (QDQ)~~ | **不要用** | 能加载但检测头输出全 0（最大置信 0.000），已在导出脚本中拦截 |

## 4. 重新训练 + 替换（PC 训练，Jetson 部署）

```bash
cd 工创yolo
# GPU 路线（推荐，≈4 分钟跑完）：见 docs_TRAINING.md §10
bash tools/retrain_v3_gpu.sh     # 训练 → 真实域评估 → 导出量化 → 部署到 sort-web
bash tools/finish_v3_gpu.sh      # 已有权重时只跑后三步
# 属性 CNN
python train_attributes.py --config config/train_attr2.yaml --task color --export-onnx
python train_attributes.py --config config/train_attr2.yaml --task shape --export-onnx
python train_attributes.py --config config/train_attr.yaml  --task stain --export-onnx

# ---- Jetson：同步并重启三个识别容器 ----
bash scripts/deploy-to-jetson.sh nvidia@<jetson-ip>          # 会同步 models/
docker compose -f docker-compose.jetson.yml restart sort-yolo sort-cnn sort-vision
```

## 5. 自检

```bash
curl http://localhost:8101/health   # 检测容器：引擎/类别/imgsz/最近耗时
curl http://localhost:8102/health   # 属性容器：已加载任务与类别数
curl http://localhost:8100/health   # 视觉容器：相机、DETECT_ENGINE=yolo、检测帧率
curl http://localhost:8100/detect/frame | python3 -m json.tool | head -40   # 标记输出
```

判据：`/detect/frame` 的 `engine` 为 `yolo(+cnn)`，每条标记都有 `shape/color/marks` 与
`attr_conf`（属性置信度应 >0.9；若仍看到 ~0.1，说明属性模型没换成实训练版本）。

## 6. 版本控制

`models/**/*.onnx` 不入库（体积大、可由 工创yolo 重新导出）：

```bash
cp 工创yolo/exports_v3gpu/best_fp32.onnx         models/yolo/best.onnx
cp 工创yolo/exports_v3gpu/best_int8_dynamic.onnx models/yolo/best_int8.onnx
cp 工创yolo/runs/attr/<task>/best.onnx           models/attr/<task>.onnx
```
