#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gen_model_manifest.py —— 生成 models/MANIFEST.json（模型登记表）
=====================================================================
算 md5/体积/输入输出形状/类别，并把训练侧指标并进来。
现场排查"到底加载的是哪个模型"时，这份表 + `/health` 就够了。

用法：
  # 只更新指纹与形状
  python scripts/gen-model-manifest.py
  # 顺带写入最新指标（训练脚本产出的评估 json）
  python scripts/gen_model_manifest.py \
      --metrics runs/eval_after_neg2.json --fp runs/fp_after_neg2.json \
      --note "193 张实拍人工复核 + 223 张背景负样本 + 分组切分"
"""
import argparse
import glob
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
MODELS = os.path.join(ROOT, "models")

TASK_CN = {
    "color": "颜色分类（红/橙/黄/绿/青/蓝/紫/黑/白）",
    "shape": "形状分类（正方体/长方体/圆柱/球/四面体/五棱柱/六棱柱/正十二面体/圆锥）",
    "stain": "表面状态（干净/污渍/缺陷）",
}


def md5(path, chunk=1 << 20):
    h = hashlib.md5()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def onnx_io(path):
    try:
        import onnx
    except ImportError:
        return None, None
    m = onnx.load(path, load_external_data=False)

    def shape(t):
        return [(d.dim_value or d.dim_param) for d in t.type.tensor_type.shape.dim]
    return ([shape(i) for i in m.graph.input], [shape(o) for o in m.graph.output])


def load_classes(path):
    if os.path.exists(path):
        return json.load(open(path, encoding="utf-8"))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", default=None, help="检测评估 json（eval_*.json）")
    ap.add_argument("--fp", default=None, help="误检专项 json（fp_*.json）")
    ap.add_argument("--note", default="", help="训练数据说明")
    ap.add_argument("--out", default=os.path.join(MODELS, "MANIFEST.json"))
    args = ap.parse_args()

    def rd(p):
        if not p:
            return {}
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else {}

    met, fp = rd(args.metrics), rd(args.fp)
    old = rd(args.out)

    det = {"task": "目标检测（找出货物在哪里，只判是不是货物）",
           "arch": "YOLOv8n", "classes": load_classes(os.path.join(MODELS, "detect", "classes.json")),
           "variants": []}
    for f in sorted(glob.glob(os.path.join(MODELS, "detect", "*.onnx"))):
        ins, outs = onnx_io(f)
        det["variants"].append({
            "file": "detect/" + os.path.basename(f),
            "precision": "int8(动态量化)" if "int8" in f else "fp32",
            "size_mb": round(os.path.getsize(f) / 1e6, 2),
            "md5": md5(f), "input": ins, "output": outs,
            "recommended": "int8" not in f,
        })
    if met:
        det["metrics"] = {k: met.get(k) for k in ("mAP50", "mAP50_95", "P", "R") if k in met}
        det["metrics"]["val"] = "40 实拍（全部人工核验）+ 10 生成 + 15 背景，分组切分、无跨集泄漏"
    if fp:
        b = fp.get("best") or {}
        det["metrics_false_positive"] = {
            "conf": b.get("conf"), "fp_per_image": b.get("fp_per_img"),
            "miss_per_image": b.get("miss_per_img"),
            "note": "conf≥0.35 时误检 0 个/图（详见 工创yolo/docs_TRAINING.md §12）"}
    if args.note:
        det["trained_on"] = args.note
    elif old.get("detector", {}).get("trained_on"):
        det["trained_on"] = old["detector"]["trained_on"]

    attrs = []
    for task in ("color", "shape", "stain"):
        f = os.path.join(MODELS, "attr", task + ".onnx")
        if not os.path.exists(f):
            continue
        ins, outs = onnx_io(f)
        cls = load_classes(os.path.join(MODELS, "attr", task + "_classes.json")) or {}
        attrs.append({"task": task, "task_cn": TASK_CN.get(task, task),
                      "file": "attr/%s.onnx" % task, "arch": "TinyNet(4 conv)",
                      "size_mb": round(os.path.getsize(f) / 1e6, 2), "md5": md5(f),
                      "input": ins, "output": outs, "n_classes": len(cls),
                      "classes": cls,
                      "runs_in": "sort-cnn 容器 :8102"})

    man = {
        "schema": 1,
        "note": "本文件由 scripts/gen_model_manifest.py 生成；换模型后请重新生成",
        "detector": det,
        "attributes": attrs,
        "where": {
            "detector": "sort-yolo 容器 :8101（被 sort-vision :8100 调用）",
            "attributes": "sort-cnn 容器 :8102（被 sort-vision :8100 调用）",
        },
    }
    json.dump(man, open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print("✅ 已写入 %s" % args.out)
    for v in det["variants"]:
        print("   检测 %-42s %6.2fMB  %s" % (v["file"], v["size_mb"], v["md5"][:12]))
    for a in attrs:
        print("   属性 %-42s %6.2fMB  %d 类  %s" % (a["file"], a["size_mb"], a["n_classes"], a["md5"][:12]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
