# models/ —— 视觉模型目录（两层模型：检测 + 属性）

> ✅ 已换成**实训练模型**。指标全部来自**真实域验证集**（40 张实拍全部人工核验 + 10 张生成 + 15 张背景图，
> 分组切分、无跨集泄漏）。机器可读的登记表见 **`MANIFEST.json`**（md5 / 输入形状 / 精度 / 指标）。

## 1. 目录约定

```
models/
├── MANIFEST.json                              # 模型登记表（换模型后重新生成，见 §7）
├── detect/                                    # ① 检测模型：货物在哪里
│   ├── goods_yolov8n_640_fp32.onnx            #   YOLOv8n 单类 goods，imgsz 640，FP32（12.27 MB）★推荐
│   ├── goods_yolov8n_640_int8.onnx            #   同一模型的动态 INT8（3.36 MB，精度不降但 CPU 上更慢）
│   └── classes.json                           #   {"0": "goods"}
└── attr/                                      # ② 属性模型：这是什么货物（CNN）
    ├── color.onnx  + color_classes.json       #   颜色 9 类
    ├── shape.onnx  + shape_classes.json       #   形状 9 类（含 dodecahedron / cone）
    └── stain.onnx  + stain_classes.json       #   表面 3 类（clean / stain / defect）
```

## 2. 两个模型分别干什么（三层识别链路）

```
相机帧
  │
  ├─① sort-yolo :8101      「有没有货物？在哪里？」
  │     goods_yolov8n_640_fp32.onnx → 若干框 [x1,y1,x2,y2] + conf
  │     ⚠ 单类检测：它只知道"这是货物"，不知道颜色/形状
  │
  └─② sort-cnn :8102       「这个框里是什么货物？」（对①的每个框裁剪 96×96）
        color.onnx → 红/橙/黄/绿/青/蓝/紫/黑/白
        shape.onnx → 正方体/长方体/圆柱/球/四面体/五棱柱/六棱柱/正十二面体/圆锥
        stain.onnx → 干净/污渍/缺陷
        └─ 三者合成"颜色+形状" = 货物身份，喂给分拣规则（同色同形 → 同筐）与屏显（货物名称）
```

**为什么非要 CNN**：检测模型是单类的（现场货物形态太多，逐类检测样本不够、泛化差），
"在哪里"用检测最稳，"是什么"用分类最省样本（裁剪图 96×96，几百张就能训）。
两者拆开还有个好处：**换货物种类只需重训 CNN，检测模型不用动**。

## 3. 在哪运行

| 模型 | 容器 | 端口 | 引擎 | 被谁调用 |
|---|---|---|---|---|
| `detect/*.onnx` | **sort-yolo** | 8101 | OpenCV DNN（默认）/ ONNX Runtime / TensorRT | `sort-vision` :8100 内部 `POST /detect_array` |
| `attr/*.onnx` | **sort-cnn** | 8102 | OpenCV DNN（默认）/ ONNX Runtime | `sort-vision` :8100 内部 `POST /classify_array` |
| 编排 + 屏显 + 分拣规则 | **sort-gateway** | 80 | Python | 浏览器 / HDMI 屏 |

- 三个容器都在 **Jetson 本机**（`docker-compose.jetson.yml`，host 网络），互相走 `127.0.0.1`
- 相机帧只抓一次：`sort-vision` 抓帧 → 编码 JPEG 同时发给 yolo 与 cnn → 合并成一条标记 JSON
- `models/` 是**只读挂载**（`./models:/app/models:ro`）→ 换模型只需替换文件 + 重启容器，不用重建镜像
- PC 端调试同样跑这两个容器（或直接 `python server/yolo_server.py`），模型文件完全一致

## 4. 实测指标

| 文件 | 作用 | 指标 |
|---|---|---|
| `detect/goods_yolov8n_640_fp32.onnx` | 检测 | **mAP50 0.9950 / mAP50-95 0.7617**，P 1.000 / R 0.995；**误检 0.00 个/图 @conf0.35**、漏检 0.03 个/图 |
| `detect/goods_yolov8n_640_int8.onnx` | 检测（小体积） | 精度基本持平，体积 1/4；CPU 上比 FP32 慢 3~5 倍，只用于省空间 |
| `attr/color.onnx` | 颜色 | 合成验证集 **100.00%**；跨域实拍 8/8 正确（white 0.99~1.00） |
| `attr/shape.onnx` | 形状 | 合成验证集 **99.77%**；跨域实拍 4/8 —— **仍是短板**，需每形状补 30~50 张实拍裁剪 |
| `attr/stain.onnx` | 表面 | clean/stain/defect 三分类 |

在线抽查（用本项目服务代码，非离线脚本）：`real_00001.jpg` → 检测 1 框 conf 0.948 →
CNN 判 **white 0.998 / dodecahedron 0.896 / clean 0.995**（与人工标注一致）。

> 形状 CNN 的跨域置信度会偏低（实拍球体 shape 置信仅 0.436），做阈值判断时别拿它当硬门限。

## 5. 推理引擎取舍

| 场景 | 用哪个 | 原因 |
|---|---|---|
| PC / x86 容器 | **FP32** + `CONF_THRES=0.35` | 30 ms/帧；动态 INT8 反而慢到 120 ms |
| Jetson | **TensorRT INT8**（`工创yolo tools/build_trt_engine.py` 生成 `.engine`） | 真正提速路径 |
| 存储受限 | 动态 INT8 | 3.36 MB，精度不降 |
| ~~ONNX Runtime 静态 INT8 (QDQ)~~ | **不要用** | 能加载但检测头输出全 0，已在导出脚本里拦截 |

## 6. 现场误检的兜底：托盘 ROI

模型负样本治的是"训练集里见过的背景"；现场的人手/机械臂/临时杂物要靠区域兜底：

```bash
# docker-compose.jetson.yml → sort-vision
ROI=0.06,0.06,0.94,0.94        # 归一化矩形；也可填多边形 "x1,y1;x2,y2;..."
```

启用后**框中心**在区域外的检测一律丢弃，丢弃数量在 `/detect/frame` 的 `roi_dropped` 里可见。

## 7. 换模型 / 重新训练

```bash
# ---- PC：训练 + 导出（工创yolo 工程）----
bash tools/retrain_v3_gpu.sh                           # 训练 → 真实域评估 → 导出量化 → 自动部署到本目录
bash tools/finish_v3_gpu.sh runs/best_after_neg2.pt    # 已有权重时只跑后三步

# ---- 刷新模型登记表（换了模型一定要跑）----
python scripts/gen_model_manifest.py \
    --metrics ../工创yolo/runs/eval_after_neg2.json --fp ../工创yolo/runs/fp_after_neg2.json

# ---- Jetson：同步并重启 ----
bash scripts/deploy-to-jetson.sh nvidia@<jetson-ip>   # 会同步 models/
docker compose -f docker-compose.jetson.yml restart sort-yolo sort-cnn sort-vision
```

## 8. 自检

```bash
curl -s http://localhost:8101/health | python3 -m json.tool
#   → model: goods_yolov8n_640_fp32.onnx
#     model_info.md5_prefix / metrics.mAP50 / metrics_false_positive ← 来自 MANIFEST.json
curl -s http://localhost:8102/health | python3 -m json.tool     # 三个任务是否都 loaded
curl -s http://localhost:8100/health                            # engine=yolo(+cnn)、相机状态
curl -s http://localhost:8100/detect/frame | python3 -m json.tool | head -40
```

判据：`/detect/frame` 的 `engine` 为 `yolo+cnn`，每条标记都有 `shape/color/marks` 与
`attr_conf`（属性置信度应 >0.9；若看到 ~0.1，说明属性模型没换成实训练版本）。

## 9. 版本控制

`models/**/*.onnx` 不入库（体积大、可由工创yolo 重新导出）；`MANIFEST.json` 与类别 json 入库，
用于核对现场实际部署的是哪一版：

```bash
cp 工创yolo/exports_v3gpu/best_fp32.onnx         models/detect/goods_yolov8n_640_fp32.onnx
cp 工创yolo/exports_v3gpu/best_int8_dynamic.onnx models/detect/goods_yolov8n_640_int8.onnx
cp 工创yolo/runs/attr/<task>/best.onnx           models/attr/<task>.onnx
```
