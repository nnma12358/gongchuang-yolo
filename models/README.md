# models/ —— 视觉模型目录（yolo 检测 + cnn 属性）

> ⚠ **本目录当前是"管线验证模型"，不是训练好的现场模型**，必须替换后再上场：
> 用它们能跑通链路，但识别结果没有意义。

## 1. 目录约定

```
models/
├── yolo/
│   ├── best.onnx           # YOLO 目标检测（工创yolo 自训练导出）
│   └── classes.json        # 类别名（单类方案：{"0": "goods"}）
└── attr/
    ├── color.onnx          # 颜色分类（9 类）
    ├── color_classes.json
    ├── shape.onnx          # 形状分类（7 类）
    ├── shape_classes.json
    ├── stain.onnx          # 污渍/缺陷分类（clean/stain/defect）
    └── stain_classes.json
```

两个容器都用 `./models:/app/models:ro` 只读挂载 —— **换模型只需替换文件并重启对应容器**，不必重建镜像。

## 2. 当前文件状态（务必替换）

| 文件 | 现状 | 来源 / 替换方式 |
|---|---|---|
| `yolo/best.onnx` | YOLOv8n **COCO 预训练**导出（opset12, imgsz640, 12.8MB） | 仅用于验证 ONNX 管线；现场用 工创yolo：`train_detection.py` → `export_models.py` 产出替换 |
| `yolo/classes.json` | `{"0": "goods"}` | 与自训练单类一致；若临时用 COCO 模型测试，请换成 80 类名（`bus`/`person`…） |
| `attr/*.onnx` | **随机权重** TinyNet（3 个任务，各约 4.4MB） | 仅验证分类管线；现场用 `train_attributes.py --config config/sorting_competition.yaml --task <color\|shape\|stain> --export-onnx` 产出替换 |
| `attr/*_classes.json` | 与训练类别一致（9 颜色 / 7 形状 / 3 表面） | 由训练脚本或手工生成（见下） |

## 3. 现场替换步骤（在 PC 训练，Jetson 部署）

```bash
# ---- PC：训练 + 导出 ----
cd 工创yolo
python tools/synth_polyhedra.py --out data/synth --n 2000 --per-image 2 --imgsz 960 720   # 合成预训练
#   再混入现场实拍 30~50 张做微调（sim-to-real 关键一步）
python prepare_dataset.py   --config config/sorting_competition.yaml
python train_detection.py   --config config/sorting_competition.yaml --epochs 150
python train_attributes.py  --config config/sorting_competition.yaml --task color --export-onnx
python train_attributes.py  --config config/sorting_competition.yaml --task shape --export-onnx
python train_attributes.py  --config config/sorting_competition.yaml --task stain --export-onnx

# ---- 拷到本项目 ----
cp runs/detect/train/weights/best.onnx        sort-web/models/yolo/best.onnx
cp runs/attr/color/best.onnx                  sort-web/models/attr/color.onnx
cp runs/attr/shape/best.onnx                  sort-web/models/attr/shape.onnx
cp runs/attr/stain/best.onnx                  sort-web/models/attr/stain.onnx
# classes.json（未自动生成时手工写）
python - <<'PY'
import json, yaml
cfg = yaml.safe_load(open('config/sorting_competition.yaml'))
json.dump({str(i): n for i, n in enumerate(cfg['detection_classes'])}, open('sort-web/models/yolo/classes.json','w'))
for task in ('color', 'shape', 'stain'):
    json.dump({str(i): n for i, n in enumerate(cfg[task + '_classes'])},
              open('sort-web/models/attr/%s_classes.json' % task, 'w'))
PY

# ---- Jetson：同步并重启两个识别容器 ----
bash scripts/deploy-to-jetson.sh nvidia@<jetson-ip>          # 会同步 models/
docker compose -f docker-compose.jetson.yml restart sort-yolo sort-cnn sort-vision
```

## 4. 自检

```bash
curl http://localhost:8101/health   # 检测容器：引擎/类别/imgsz/最近耗时
curl http://localhost:8102/health   # 属性容器：已加载任务与类别数
curl http://localhost:8100/health   # 视觉容器：相机、DETECT_ENGINE=yolo、检测帧率
curl http://localhost:8100/detect/frame | python3 -m json.tool | head -40   # 标记输出
```

判据：`/detect/frame` 的 `engine` 为 `yolo(+cnn)`，每条标记都有 `shape/color/marks` 与
`attr_conf`（属性置信度应 >0.9；当前随机权重模型约 0.1，属预期）。

## 5. 版本控制

`models/**/*.onnx` 不入库（体积大、可由 工创yolo 重新导出）：

```bash
cp 工创yolo/exports_clean/best_fp32.onnx models/yolo/best.onnx
cp 工创yolo/runs/attr/<task>/best.onnx   models/attr/<task>.onnx
```
