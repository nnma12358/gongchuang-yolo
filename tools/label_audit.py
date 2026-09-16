#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
label_audit.py —— 标签质量审计 + 人工复核分诊包
=====================================================================
目的：把"人工复核 150 张"这件事变成**只看该看的、按优先级看**，而不是逐张盲看。

做三件事：
  1) 自动体检：几何规则 + **模型预测与现有标注的一致性（IoU）**，给每张图算"可疑度"
  2) 出可视化：overlays/（绿=现有标注，红=模型预测）+ sheets/（拼图速览，一眼扫完）
  3) 出复核包：review_set/{images,labels} + priority.csv（按可疑度排序，含理由）
     可直接用 X-AnyLabeling 或本项目自带的 review_app.py 打开

用法：
  python tools/label_audit.py --mixed data/real_synth_mix --weights runs/best_v3_gpu.pt \
      --out data/review --device 0
  python tools/label_audit.py ... --no-model          # 只跑几何规则（无 GPU/无权重时）
"""
import argparse
import csv
import glob
import json
import os
import re
import shutil

import cv2
import numpy as np


# ==================== 基础 ====================
def read_yolo(path):
    out = []
    if path and os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            p = line.split()
            if len(p) >= 5:
                out.append([float(v) for v in p[1:5]])
    return out


def xyxy(box, W, H):
    cx, cy, w, h = box
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def ts_of(name, prov):
    """从 provenance 的 from 字段取拍摄时间戳（毫秒）"""
    f = (prov.get(name) or {}).get("from") or ""
    m = re.search(r"(\d{13})", str(f))
    if m:
        return int(m.group(1))
    m = re.search(r"(\d{13})", name)
    return int(m.group(1)) if m else None


def assign_groups(items, gap_ms):
    """按拍摄时间聚类：相邻间隔 > gap 视为新场景（同一场景连拍的多张必须同组，防泄漏）"""
    items = sorted([i for i in items if i[1]], key=lambda x: x[1])
    groups, cur, last = {}, 0, None
    for name, t in items:
        if last is None or t - last > gap_ms:
            cur += 1
        groups[name] = cur
        last = t
    return groups


# ==================== 几何体检 ====================
def geometry_reasons(lines, W, H):
    reasons, feats = [], []
    for box in lines:
        cx, cy, w, h = box
        area = w * h
        aspect = max(w, h) / max(1e-6, min(w, h))
        off = max(abs(cx - 0.5), abs(cy - 0.5))
        x1, y1, x2, y2 = xyxy(box, W, H)
        border = min(x1, y1, W - x2, H - y2)
        r = []
        if area < 0.004:
            r.append("框过小(<0.4%)")
        if area > 0.25:
            r.append("框过大(>25%)")
        if aspect > 2.0:
            r.append("长宽比>2.0")
        if off > 0.30:
            r.append("严重偏心")
        if border < 0.01 * max(W, H):
            r.append("贴边")
        feats.append({"area": round(area, 4), "aspect": round(aspect, 2),
                      "center_off": round(off, 3), "border_px": int(border)})
        reasons += r
    return reasons, feats


# ==================== 打分 ====================
def prio(r):
    """复核顺序（与 docs_REVIEW.md 一致）：
       0 = val 实拍（唯一的"尺子"，必须 100% 人工核验）
       1 = 未标注图（自动预标注失败，要从零画框，直接增加真实数据）
       2 = train 实拍（按可疑度挑着修）
    """
    if r.get("v3_split") == "val":
        return 0
    if r.get("kind") == "unlabeled":
        return 1
    return 2


def suspicion(n_label, n_model_std, n_model_lo, best_iou, geo_reasons, model_conf):
    score, why = 0, []
    if n_label == 0:
        score += 60
        why.append("标注为空（可能漏标）" if n_model_lo else "标注为空")
    if n_model_lo and best_iou < 0.3:
        score += 50
        why.append("与模型完全不重合(IoU<0.3)")
    elif n_model_lo and best_iou < 0.5:
        score += 30
        why.append("与模型偏差大(IoU<0.5)")
    elif n_model_lo and best_iou < 0.7:
        score += 15
        why.append("与模型有偏差(IoU<0.7)")
    extra = n_model_lo - n_label
    if extra > 0:
        score += min(30, 15 * extra)
        why.append("模型多检出 %d 个（可能漏标）" % extra)
    if n_model_std > n_label:
        score += 10
        why.append("高置信多检出")
    if geo_reasons:
        score += min(24, 8 * len(geo_reasons))
        why += geo_reasons
    if n_model_lo == 0 and n_label > 0 and model_conf < 0.2:
        score += 12
        why.append("模型完全找不到（标注可疑或货物异常）")
    return min(100, score), why


# ==================== 绘图 ====================
def draw_overlay(img, label_boxes, model_boxes, header, tile=760):
    """绿=现有标注；红=模型预测(conf≥0.25)；橙=模型低置信(0.1~0.25)

    注：线宽必须 ≥3 且用饱和色 —— JPEG 4:2:0 色度下采样会把 1~2px 细线的颜色糊掉
    （细绿框在 JPEG 里会“消失”，PNG 无损才看得见）。
    """
    h, w = img.shape[:2]
    sc = tile / float(max(h, w))
    vis = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
    H, W = vis.shape[:2]
    for b in model_boxes:
        if b["conf"] < 0.25:
            continue
        x1, y1, x2, y2 = [int(v * sc) for v in b["xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 30, 230), 3)
        cv2.putText(vis, "M%.2f" % b["conf"], (x1, max(16, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 230), 2, cv2.LINE_AA)
    for b in model_boxes:
        if b["conf"] >= 0.25:
            continue
        x1, y1, x2, y2 = [int(v * sc) for v in b["xyxy"]]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 140, 255), 2)
    for i, box in enumerate(label_boxes):
        x1, y1, x2, y2 = [int(v) for v in xyxy(box, W, H)]
        cv2.rectangle(vis, (x1, y1), (x2, y2), (30, 190, 30), 3)
        cv2.putText(vis, "L%d" % (i + 1), (x1, min(H - 6, y2 + 18)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 190, 30), 2, cv2.LINE_AA)
    bar = np.full((34, W, 3), 28, np.uint8)
    cv2.putText(bar, header, (6, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (235, 235, 235), 1, cv2.LINE_AA)
    return np.vstack([bar, vis])


def contact_sheet(paths, cols=4, tile=420):
    ims = []
    for p in paths:
        im = cv2.imread(p)
        if im is None:
            continue
        h, w = im.shape[:2]
        sc = tile / float(max(h, w))
        ims.append(cv2.resize(im, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA))
    if not ims:
        return None
    th = max(i.shape[0] for i in ims)
    tw = max(i.shape[1] for i in ims)
    rows = []
    for i in range(0, len(ims), cols):
        chunk = ims[i:i + cols]
        chunk += [np.full((th, tw, 3), 245, np.uint8)] * (cols - len(chunk))
        rows.append(np.hstack(chunk))
    return np.vstack(rows)


# ==================== 主流程 ====================
def main():
    ap = argparse.ArgumentParser(description="标签审计 + 复核分诊")
    ap.add_argument("--mixed", default="data/real_synth_mix")
    ap.add_argument("--mix-v3", default="data/mix_v3", help="用于标注每张图在 v3 里的 split")
    ap.add_argument("--weights", default="runs/best_v3_gpu.pt")
    ap.add_argument("--out", default="data/review")
    ap.add_argument("--device", default="0")
    ap.add_argument("--no-model", action="store_true", help="跳过模型推理（仅几何规则）")
    ap.add_argument("--conf-lo", type=float, default=0.10)
    ap.add_argument("--group-gap", type=float, default=30.0, help="场景分组的时间间隔(秒)")
    ap.add_argument("--copy-unlabeled", action="store_true", default=True)
    args = ap.parse_args()

    summ = json.load(open(os.path.join(args.mixed, "dataset_summary.json"), encoding="utf-8"))
    prov = summ.get("labels_source", {})
    flagged = {r["image"]: r.get("reason", "") for r in summ.get("review", [])}
    v3 = {}
    p3 = os.path.join(args.mix_v3, "subset_summary.json")
    if os.path.exists(p3):
        v3 = json.load(open(p3, encoding="utf-8")).get("index", {})

    # 收集待审图片：real_* 实拍（已标注）+ unlabeled 未标注
    items = []          # (name, path, label_path, kind)
    for split in ("train", "val"):
        for p in sorted(glob.glob(os.path.join(args.mixed, "images", split, "real_*.*"))):
            n = os.path.basename(p)
            lp = os.path.join(args.mixed, "labels", split, os.path.splitext(n)[0] + ".txt")
            items.append((n, p, lp if os.path.exists(lp) else None, "labeled"))
    for p in sorted(glob.glob(os.path.join(args.mixed, "images", "unlabeled", "*.*"))):
        items.append((os.path.basename(p), p, None, "unlabeled"))

    # 场景分组：与 build_mix_v3 用同一套"完全连接"视觉聚类
    # （文件名时间戳是导出时生成的，相邻帧只差 7ms，按时间分组不可用）
    ts = {n: ts_of(n, prov) for n, _, _, _ in items}
    path_of = {n: p for n, p, _, _ in items}
    try:
        from build_mix_v3 import scene_groups
        raw = scene_groups(list(path_of), lambda n: path_of.get(n), prov, visual_thr=20)
    except Exception as e:                                   # 退路：时间聚类
        print("[audit] 视觉分组不可用（改用时间聚类）：", str(e)[:80])
        raw = assign_groups([(n, ts[n]) for n in ts], args.group_gap * 1000)
    roots, groups = {}, {}
    for n in path_of:
        r = raw.get(n, n)
        roots.setdefault(r, len(roots) + 1)
        groups[n] = roots[r]
    n_groups = len(roots)

    # 模型
    model = None
    if not args.no_model and os.path.exists(args.weights):
        from ultralytics import YOLO
        model = YOLO(args.weights)
        print("[audit] 模型:", args.weights)

    # 每次重建前清空旧产物：否则上一轮文件会残留（曾出现 193 张变 386 张），
    # 旧序号还会与新序号撞名，复核台里就会出现"空标注 / 对不上图"的项。
    for sub in ("overlays", "sheets"):
        p = os.path.join(args.out, sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    for sub in ("images", "labels"):
        p = os.path.join(args.out, "review_set", sub)
        if os.path.isdir(p):
            shutil.rmtree(p)
    os.makedirs(os.path.join(args.out, "overlays"), exist_ok=True)
    os.makedirs(os.path.join(args.out, "sheets"), exist_ok=True)
    rows = []
    hints = {}      # name → 模型建议框（归一化 xyxy），供复核界面一键采纳
    for i, (name, path, lp, kind) in enumerate(items, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        H, W = img.shape[:2]
        lines = read_yolo(lp)
        mboxes = []
        if model is not None:
            r = model.predict(path, imgsz=640, conf=args.conf_lo, device=args.device, verbose=False)[0]
            if r.boxes is not None and len(r.boxes):
                xyxy_arr = r.boxes.xyxy.cpu().numpy()
                conf_arr = r.boxes.conf.cpu().numpy()
                for b, c in zip(xyxy_arr, conf_arr):
                    mboxes.append({"xyxy": [float(v) for v in b], "conf": float(c)})
        lb = [xyxy(b, W, H) for b in lines]
        mb = [m["xyxy"] for m in mboxes]
        hints[name] = [{"xyxy": [round(m["xyxy"][0] / W, 4), round(m["xyxy"][1] / H, 4),
                                 round(m["xyxy"][2] / W, 4), round(m["xyxy"][3] / H, 4)],
                        "conf": round(m["conf"], 3)} for m in mboxes]
        # 贪心匹配求最优 IoU
        best_iou, used = 0.0, set()
        for a in lb:
            bi, bj = 0.0, None
            for j, b in enumerate(mb):
                if j in used:
                    continue
                v = iou(a, b)
                if v > bi:
                    bi, bj = v, j
            if bj is not None:
                used.add(bj)
            best_iou = max(best_iou, bi)
        n_std = sum(1 for m in mboxes if m["conf"] >= 0.25)
        n_lo = len(mboxes)
        top_conf = max([m["conf"] for m in mboxes], default=0.0)
        geo, feats = geometry_reasons(lines, W, H)
        score, why = suspicion(len(lines), n_std, n_lo, best_iou, geo, top_conf)
        if kind == "unlabeled":
            score, why = 100, ["自动预标注失败 → 需从零标注"]
        src = (prov.get(name) or {}).get("source", "unlabeled" if kind == "unlabeled" else "?")
        split3 = (v3.get(name) or {}).get("split", "-")
        if flagged.get(name) and kind == "labeled":
            why.append("原始预标注：" + flagged[name][:28])
            score = min(100, score + 10)
        rows.append({
            "image": name, "kind": kind, "source": src, "v3_split": split3,
            "group": groups.get(name, -1), "ts": ts.get(name),
            "n_label": len(lines), "n_model": n_lo, "n_model_std": n_std,
            "best_iou": round(best_iou, 3), "top_conf": round(top_conf, 3),
            "area": feats[0]["area"] if feats else 0.0,
            "aspect": feats[0]["aspect"] if feats else 0.0,
            "center_off": feats[0]["center_off"] if feats else 0.0,
            "score": score, "why": "；".join(dict.fromkeys(why)),
            "path": path, "label": lp,
        })

    rows.sort(key=lambda r: (prio(r), -r["score"], r["image"]))

    # ---- 复核包（按可疑度排序复制，文件名加序号前缀，编辑器里就是优先级顺序）----
    rv_img = os.path.join(args.out, "review_set", "images")
    rv_lbl = os.path.join(args.out, "review_set", "labels")
    os.makedirs(rv_img, exist_ok=True)
    os.makedirs(rv_lbl, exist_ok=True)
    mapping = []
    for k, r in enumerate(rows, 1):
        stem = "%03d_%s" % (k, os.path.splitext(r["image"])[0])
        ext = os.path.splitext(r["path"])[1]
        shutil.copy2(r["path"], os.path.join(rv_img, stem + ext))
        if r["label"]:
            shutil.copy2(r["label"], os.path.join(rv_lbl, stem + ".txt"))
        else:
            open(os.path.join(rv_lbl, stem + ".txt"), "w").close()   # 空文件 = 待画框
        mapping.append({"order": k, "review_name": stem + ext, "original": r["image"],
                        "score": r["score"], "why": r["why"], "v3_split": r["v3_split"]})
        ov = draw_overlay(img, lines, mboxes,
                          "%03d  %s  [%s/%s/组%d]  标注%d 模型%d  IoU%.2f  可疑%d %s"
                          % (k, r["image"], r["source"], r["kind"], r["group"],
                             r["n_label"], r["n_model"], r["best_iou"], r["score"], r["why"][:46]))
        cv2.imwrite(os.path.join(args.out, "overlays", stem + ".jpg"), ov,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 88])

    # ---- 拼图速览 ----
    ovs = [os.path.join(args.out, "overlays", m["review_name"].rsplit(".", 1)[0] + ".jpg")
           for m in mapping]
    per = 20
    for i in range(0, len(ovs), per):
        sheet = contact_sheet(ovs[i:i + per], cols=4)
        if sheet is not None:
            cv2.imwrite(os.path.join(args.out, "sheets", "sheet_%02d.jpg" % (i // per + 1)),
                        sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

    # ---- 报表 ----
    with open(os.path.join(args.out, "audit.csv"), "w", newline="", encoding="utf-8-sig") as f:
        wr = csv.DictWriter(f, fieldnames=[k for k in rows[0] if k not in ("path", "label")])
        wr.writeheader()
        for r in rows:
            wr.writerow({k: v for k, v in r.items() if k not in ("path", "label")})
    json.dump({"items": rows, "mapping": mapping, "groups": n_groups,
               "group_gap_s": args.group_gap},
              open(os.path.join(args.out, "audit.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(hints, open(os.path.join(args.out, "hints.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    # ---- 控制台摘要 ----
    tot = len(rows)
    need = sum(1 for r in rows if r["score"] >= 40)
    mild = sum(1 for r in rows if 15 <= r["score"] < 40)
    ok = sum(1 for r in rows if r["score"] < 15)
    print("=" * 68)
    print(" 审计完成：%d 张（近重复图分组 %d 组，完全连接视觉聚类）" % (tot, n_groups))
    print("   可疑度 ≥40（优先修）: %d 张" % need)
    print("   15~39（顺手看一眼） : %d 张" % mild)
    print("   <15（基本可信）      : %d 张" % ok)
    if model is not None:
        ious = [r["best_iou"] for r in rows if r["n_label"] > 0 and r["n_model"] > 0]
        if ious:
            a = np.array(ious)
            print(" 模型↔标注 IoU：均值 %.3f · 中位 %.3f · <0.5 占比 %.1f%%"
                  % (a.mean(), np.median(a), 100.0 * (a < 0.5).mean()))
    print(" 输出：%s/{audit.csv, overlays/, sheets/, review_set/}" % args.out)
    print("=" * 68)


if __name__ == "__main__":
    main()
