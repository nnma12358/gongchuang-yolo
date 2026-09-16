#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_mixed_dataset.py —— 混合数据集构建（实拍全选 + 生成图小部分）
=====================================================================
输入目录（工作区 `图/`）：
  · 实拍图   *.jpg（3072×3072，手机拍摄，物体为深色多面体，白纸背景）
            其中 X-AnyLabeling/ 下 18 张已**人工标注**（X-AnyLabeling JSON → YOLO txt）
  · 生成图   *.png（512×512，白底渲染多面体，无标注）→ 本工具自动标注并按类别筛选
  · classes: X-AnyLabeling 标注给出 dodecahedron(0) / sphere(1)

做法：
  实拍：**全部保留**；已有人工标注的用人工标注，其余用几何自动预标注
        （暗物体 + 白纸背景 + 形状/长宽比/实心度过滤，避免把遥控器/桌垫当目标）
  生成：全部自动标注（白底分割），随后**只保留与实拍类别一致**的少量图片
        （圆度/顶点数判定 dodecahedron-like 或 sphere-like），数量由 --gen-keep 控制

输出（YOLO 结构，可直接训练）：
  <out>/images/{train,val}/*.jpg|png
  <out>/labels/{train,val}/*.txt
  <out>/classes.txt
  <out>/dataset_summary.json     每张图的标签来源（human / auto-real / auto-gen）
  <out>/.review_needed.txt       需要人工复核的自动标注清单（建议用 X-AnyLabeling 打开）

用法：
  python3 tools/build_mixed_dataset.py \
      --real-dir /home/xxxffyy/工创/图 \
      --gen-dir  /home/xxxffyy/工创/图 \
      --out data/real_synth_mix --gen-keep 60 --val-ratio 0.15
"""
import argparse
import glob
import json
import os
import random
import shutil

import cv2
import numpy as np

# 检测类别：单类 goods —— 现场货物形状多样（十二面体/球/锥/四面体/立方体…），
# 检测器只负责“找到货物”，形状/颜色由 cnn 属性容器判定（见 sort-web 视觉三层）。
CLASSES = ["goods"]
SHAPE_ATTR_CLASSES = ["dodecahedron", "sphere", "cube", "cuboid", "cylinder", "cone", "tetra", "prism"]


# ==================== 工具 ====================
def yolo_line(cls, box, W, H):
    x1, y1, x2, y2 = box
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(W, x2), min(H, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    return "{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}".format(
        cls, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, (x2 - x1) / W, (y2 - y1) / H)


def shape_features(contour):
    area = cv2.contourArea(contour)
    peri = cv2.arcLength(contour, True)
    circ = 4 * np.pi * area / (peri * peri) if peri > 0 else 0
    hull = cv2.convexHull(contour)
    solidity = area / max(1e-6, cv2.contourArea(hull))
    verts = len(cv2.approxPolyDP(hull, 0.02 * peri, True)) if peri > 0 else 0
    x, y, w, h = cv2.boundingRect(contour)
    aspect = max(w, h) / max(1.0, min(w, h))
    return {"area": area, "circularity": round(circ, 3), "solidity": round(solidity, 3),
            "vertices": verts, "aspect": round(aspect, 2), "box": [x, y, x + w, y + h],
            "area_ratio": area / float(1)}


def classify_shape(feat):
    """由剪影特征判定类别：sphere（近圆）/ dodecahedron-like（多面体，7~12 顶点）/ 其他"""
    if feat["circularity"] >= 0.92:
        return "sphere"
    if 0.72 <= feat["circularity"] < 0.92 and feat["vertices"] >= 7 and feat["solidity"] >= 0.9:
        return "dodecahedron"
    return "other"


# ==================== 生成图：白底分割自动标注 ====================
def auto_label_white_bg(path, min_area_ratio=0.02, max_area_ratio=0.9):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # 白底：低饱和 + 高亮度；目标 = 非白区域
    mask = ((hsv[:, :, 1] > 40) | (hsv[:, :, 2] < 235)).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    feat = shape_features(c)
    ratio = feat["area"] / float(W * H)
    if not (min_area_ratio <= ratio <= max_area_ratio):
        return None
    feat["cls"] = classify_shape(feat)
    feat["shape_ratio"] = round(ratio, 3)
    feat["size"] = [W, H]
    return feat


# ==================== 实拍图：白纸背景上的多面体自动预标注 ====================
# 现场实拍有两种：黑色多面体（强对比）与白/浅色多面体（与白纸几乎同亮度，只能靠棱面明暗
# 与接触阴影的局部对比）。因此这里跑两条路径并取最优候选：
#   路径1 黑帽(151) + Otsu      → 适合黑物体
#   路径2 黑帽(151) + 自适应阈值 → 适合白物体的弱对比棱面/阴影
def _candidates(mask, bh, W):
    """从掩膜里挑选像货物的候选：面积/实心度/长宽比过滤 + 居中与对比度打分"""
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8), iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8), iterations=2)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for c in contours:
        area = cv2.contourArea(c)
        ratio = area / float(W * W)
        if not (0.002 <= ratio <= 0.30):
            continue
        sol = area / max(1e-6, cv2.contourArea(cv2.convexHull(c)))
        x, y, w, h = cv2.boundingRect(c)
        aspect = max(w, h) / max(1.0, min(w, h))
        if sol < 0.65 or aspect > 2.4:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        if abs(cx - W / 2) > W * 0.36 or abs(cy - W / 2) > W * 0.36:
            continue                                   # 居中先验：货物在画面中部
        sub = bh[y:y + h, x:x + w]
        inside = mask[y:y + h, x:x + w] > 0
        contrast = float(np.mean(sub[inside])) if inside.any() else 0.0
        size_prior = 1.0 if 0.008 <= ratio <= 0.18 else 0.6
        center_bias = 1.0 - (abs(cx - W / 2) + abs(cy - W / 2)) / (W * 1.2)
        score = sol * (area ** 0.5) * (1 + contrast / 30.0) * size_prior * center_bias
        out.append({"score": score, "box": [x, y, x + w, y + h], "ratio": round(ratio, 4),
                    "solidity": round(sol, 3), "aspect": round(aspect, 2),
                    "contrast": round(contrast, 1), "path": None})
    return out


def auto_label_real(path, min_area_ratio=0.002, max_area_ratio=0.30):
    """返回最佳候选（含诊断字段）；失败返回 None"""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        return None
    H, W = img.shape[:2]
    S = 768
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.GaussianBlur(cv2.resize(gray, (S, S), interpolation=cv2.INTER_AREA), (3, 3), 0)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (151, 151))
    bh = cv2.morphologyEx(small, cv2.MORPH_BLACKHAT, k)
    _, m1 = cv2.threshold(bh, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, dark = cv2.threshold(small, 110, 255, cv2.THRESH_BINARY_INV)
    m1 = cv2.bitwise_or(m1, dark)                       # 黑物体直接进掩膜
    m2 = cv2.adaptiveThreshold(bh, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 101, -4)

    cands = []
    for tag, mask in (("blackhat+otsu", m1), ("blackhat+adaptive", m2)):
        for c in _candidates(mask, bh, S):
            c["path"] = tag
            cands.append(c)
    if not cands:
        return None
    best = max(cands, key=lambda c: c["score"])
    scale = W / float(S)
    feat = {k: v for k, v in best.items() if k != "box"}
    return {"box": [int(v * scale) for v in best["box"]], "feat": feat,
            "size": [W, H], "cls": "dodecahedron"}


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser(description="混合数据集构建（实拍全选 + 生成图小部分）")
    ap.add_argument("--real-dir", required=True, help="含实拍 jpg 的目录（如 工创/图）")
    ap.add_argument("--gen-dir", default=None, help="含生成 png 的目录（默认同 real-dir）")
    ap.add_argument("--labeled-dir", default=None, help="人工标注目录（X-AnyLabeling，含 images/ labels/ json/）")
    ap.add_argument("--out", default="data/real_synth_mix")
    ap.add_argument("--gen-keep", type=int, default=60, help="保留多少张生成图（类别一致的优先）")
    ap.add_argument("--gen-shapes", default="dodecahedron,sphere",
                    help="允许保留的生成图类别（逗号分隔）")
    ap.add_argument("--val-ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--copy", action="store_true", help="复制原图（默认是复制；--no-copy 则只写标注）")
    ap.add_argument("--attributes", default="1", help="是否导出 CNN 属性小图（1/0）")
    ap.add_argument("--attr-size", type=int, default=96)
    ap.add_argument("--resize-real", type=int, default=1280,
                    help="实拍图缩放到该边长再入库（0=保持原图；标注为归一化坐标不受影响）")
    args = ap.parse_args()

    gen_dir = args.gen_dir or args.real_dir
    rng = random.Random(args.seed)
    os.makedirs(args.out, exist_ok=True)
    allow = set(s.strip() for s in args.gen_shapes.split(",") if s.strip())

    summary = {"classes": CLASSES, "real": {}, "generated": {}, "labels_source": {},
               "params": vars(args), "review": []}
    items = []        # (src_path, label_lines, source, group)
    attr_jobs = []    # (src_path, cls_idx, task, class_name, source)

    # ---------- 1) 人工标注的实拍图（X-AnyLabeling） ----------
    human_n = 0
    if args.labeled_dir:
        limg = os.path.join(args.labeled_dir, "images")
        llab = os.path.join(args.labeled_dir, "labels")
        ljson = os.path.join(args.labeled_dir, "json")
        for ip in sorted(glob.glob(os.path.join(limg, "*.*"))):
            base = os.path.splitext(os.path.basename(ip))[0]
            tp = os.path.join(llab, base + ".txt")
            jp = os.path.join(ljson, base + ".json")
            # 类别索引以 json 的 label 为准（避免 classes.txt 为空导致错类）
            idx_map = {}
            json_labels = []
            if os.path.exists(jp):
                try:
                    shapes = json.load(open(jp, encoding="utf-8")).get("shapes", [])
                    labels = []
                    for s in shapes:
                        if s.get("label") not in labels:
                            labels.append(s.get("label"))
                    json_labels = labels
                    idx_map = {i: 0 for i in range(len(labels))}   # 单类：全部映射到 goods(0)
                except Exception:
                    idx_map = {}
            if not os.path.exists(tp):
                continue
            lines = []
            for line in open(tp):
                p = line.split()
                if len(p) == 5:
                    # 单类数据集：坐标沿用人工标注，类别索引统一写 0
                    # （X-AnyLabeling 导出的索引按它自己的类别表，不能直接沿用）
                    lines.append("0 {0} {1} {2} {3}".format(*p[1:]))
            if lines:
                items.append((ip, lines, "human", "real"))
                human_n += 1
                # 人工标注里的形状标签（dodecahedron/sphere…）→ CNN 形状属性样本
                if json_labels:
                    for name in json_labels:
                        attr_jobs.append((ip, 0, "shape", name, "human"))
    summary["real"]["human_labeled"] = human_n

    # ---------- 2) 其余实拍图：自动预标注 ----------
    real_all = sorted([p for p in glob.glob(os.path.join(args.real_dir, "*.jpg"))
                       + glob.glob(os.path.join(args.real_dir, "*.jpeg"))
                       + glob.glob(os.path.join(args.real_dir, "*.JPG"))])
    auto_ok, auto_fail, unlabeled = 0, 0, []
    for p in real_all:
        feat = auto_label_real(p)
        if feat is None:
            auto_fail += 1
            unlabeled.append(p)
            summary["review"].append({"image": os.path.basename(p),
                                      "reason": "自动预标注失败 → 已放入 images/unlabeled 待人工标注"})
            continue
        line = yolo_line(0, feat["box"], feat["size"][0], feat["size"][1])
        if not line:
            auto_fail += 1
            continue
        items.append((p, [line], "auto-real", "real"))
        auto_ok += 1
        f = feat["feat"]
        if f["solidity"] < 0.88 or f["ratio"] < 0.01 or f["contrast"] < 8:
            summary["review"].append({"image": os.path.basename(p), "reason": "预标注置信一般，建议复核",
                                      "feat": f})
    # 未标注的实拍图同样全部保留（尊重“全选实拍”），但不进训练集，避免错标签
    for p in unlabeled:
        d = os.path.join(args.out, "images", "unlabeled")
        os.makedirs(d, exist_ok=True)
        dst = os.path.join(d, os.path.basename(p))
        im = cv2.imread(p) if args.resize_real else None
        if im is not None and max(im.shape[:2]) > args.resize_real:
            sc = args.resize_real / float(max(im.shape[:2]))
            im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)), interpolation=cv2.INTER_AREA)
            cv2.imwrite(dst, im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
        else:
            shutil.copy2(p, dst)
    summary["real"].update({"auto_prelabeled": auto_ok, "auto_failed": auto_fail,
                            "unlabeled_kept": len(unlabeled),
                            "labeled_total": human_n + auto_ok,
                            "files_found": len(real_all)})

    # ---------- 3) 生成图：自动标注 + 按类别筛少量 ----------
    gen_files = sorted(glob.glob(os.path.join(gen_dir, "*.png")))
    gen_eval = []
    for p in gen_files:
        feat = auto_label_white_bg(p)
        if feat is None:
            continue
        gen_eval.append((p, feat))
    by_cls = {}
    for p, feat in gen_eval:
        by_cls.setdefault(feat["cls"], []).append((p, feat))
    picked, pool = [], []
    for cls in sorted(allow):
        pool.extend(by_cls.get(cls, []))
    # 若允许类别不足，从 other 里补（明确记录来源类别）
    if len(pool) < args.gen_keep:
        pool.extend(by_cls.get("other", []))
    rng.shuffle(pool)
    for p, feat in pool[:args.gen_keep]:
        line = yolo_line(0, feat["box"], feat["size"][0], feat["size"][1])   # 单类 goods
        if line:
            picked.append((p, [line], "auto-gen", "generated"))
            shape_name = feat["cls"] if feat["cls"] != "other" else "polyhedron"
            attr_jobs.append((p, 0, "shape", shape_name, "auto-gen"))
    summary["generated"] = {"files_found": len(gen_files), "auto_labeled": len(gen_eval),
                            "by_shape": {k: len(v) for k, v in sorted(by_cls.items())},
                            "kept": len(picked), "kept_shapes": {
                                k: sum(1 for _, f in pool[:args.gen_keep] if f["cls"] == k)
                                for k in sorted(set(f["cls"] for _, f in pool[:args.gen_keep]))}}

    # ---------- 4) 写出 train/val ----------
    rng.shuffle(items)
    all_items = items + picked
    n_val = max(1, int(len(all_items) * args.val_ratio))
    for split in ("train", "val"):
        for sub in ("images", "labels"):
            os.makedirs(os.path.join(args.out, sub, split), exist_ok=True)
    written = {"train": 0, "val": 0}
    src_stat = {}
    for i, (src, lines, source, group) in enumerate(all_items):
        split = "val" if i < n_val else "train"
        ext = os.path.splitext(src)[1].lower()
        name = "{0}_{1:05d}{2}".format(group[:4], i, ext)
        dst = os.path.join(args.out, "images", split, name)
        ext_l = os.path.splitext(name)[1].lower()
        resized = False
        if group == "real" and args.resize_real and ext_l in (".jpg", ".jpeg", ".png"):
            im = cv2.imread(src)
            if im is not None and max(im.shape[:2]) > args.resize_real:
                sc = args.resize_real / float(max(im.shape[:2]))
                im = cv2.resize(im, (int(im.shape[1] * sc), int(im.shape[0] * sc)),
                                interpolation=cv2.INTER_AREA)
                cv2.imwrite(dst, im, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                resized = True
        if not resized:
            shutil.copy2(src, dst)
        with open(os.path.join(args.out, "labels", split, os.path.splitext(name)[0] + ".txt"),
                  "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
        written[split] += 1
        src_stat[source] = src_stat.get(source, 0) + 1
        summary["labels_source"][name] = {"source": source, "group": group,
                                          "from": os.path.basename(src)}

    # ---------- 5) CNN 属性小图（形状：人工标注 + 生成图渲染形状） ----------
    attr_count = {}
    if str(args.attributes) == "1":
        for src, ci, task, cls_name, source in attr_jobs:
            if not cls_name or cls_name in ("other", "polyhedron"):
                continue
            img = cv2.imread(src)
            if img is None:
                continue
            # 复用该图已求得的框（人工/自动标注）
            box = None
            for it_src, it_lines, it_src_kind, _g in items + picked:
                if it_src == src and it_lines:
                    p0 = it_lines[0].split()
                    H, W = img.shape[:2]
                    cx, cy, w, h = [float(v) for v in p0[1:5]]
                    box = [int((cx - w / 2) * W), int((cy - h / 2) * H),
                           int((cx + w / 2) * W), int((cy + h / 2) * H)]
                    break
            if box is None:
                continue
            x1, y1, x2, y2 = box
            pad = int(max(x2 - x1, y2 - y1) * 0.12)
            H, W = img.shape[:2]
            crop = img[max(0, y1 - pad):min(H, y2 + pad), max(0, x1 - pad):min(W, x2 + pad)]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (args.attr_size, args.attr_size))
            # 颜色：由裁剪均值判定黑/白（现场货物就这两类），不确定则不写
            gray_mean = float(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).mean())
            color_name = "black" if gray_mean < 95 else ("white" if gray_mean > 175 else None)
            for t, name in (("shape", cls_name), ("color", color_name)):
                if not name:
                    continue
                d = os.path.join(args.out, "attributes", t, name)
                os.makedirs(d, exist_ok=True)
                cv2.imwrite(os.path.join(d, "{0}_{1}.jpg".format(source, len(attr_count))), crop)
                attr_count[(t, name)] = attr_count.get((t, name), 0) + 1
    summary["attributes"] = {"{0}/{1}".format(k[0], k[1]): v for k, v in sorted(attr_count.items())}

    with open(os.path.join(args.out, "classes.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(CLASSES) + "\n")
    summary["written"] = written
    summary["labels_source_counts"] = src_stat
    summary["total_images"] = sum(written.values())
    summary["objects"] = sum(len(l) for _, l, _, _ in all_items)
    with open(os.path.join(args.out, "dataset_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    review = summary["review"]
    if review:
        with open(os.path.join(args.out, ".review_needed.txt"), "w", encoding="utf-8") as f:
            f.write("# 建议用 X-AnyLabeling 打开以下图片复核自动预标注\n")
            for r in review:
                f.write("{0}\t{1}\n".format(r["image"], r["reason"]))

    print("=" * 60)
    print(" 混合数据集 → {0}".format(os.path.abspath(args.out)))
    print("=" * 60)
    print(" 类别:", CLASSES)
    print(" 实拍图: 共 {0} 张 → 人工标注 {1} | 自动预标注 {2} | 待人工标注 {3}（已全部保留）".format(
        summary["real"]["files_found"], summary["real"]["human_labeled"],
        summary["real"]["auto_prelabeled"], summary["real"]["unlabeled_kept"]))
    print(" 生成图: 找到 {0} 张 → 自动标注 {1} 张 → 保留 {2} 张（{3}）".format(
        summary["generated"]["files_found"], summary["generated"]["auto_labeled"],
        summary["generated"]["kept"], summary["generated"]["kept_shapes"]))
    print(" 生成图形态分布:", summary["generated"]["by_shape"])
    print(" 输出: train {0} / val {1}（共 {2} 张，{3} 个标注框）".format(
        written["train"], written["val"], summary["total_images"], summary["objects"]))
    print(" 标注来源:", src_stat)
    if review:
        print(" ⚠ 需人工复核: {0} 张（见 .review_needed.txt）".format(len(review)))
    print("\n下一步：")
    print("  python prepare_dataset.py --config config/sorting_competition.yaml   # 若需重划分")
    print("  python train_detection.py --config config/sorting_competition.yaml --epochs 150")


if __name__ == "__main__":
    main()
