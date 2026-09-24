#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""现场俯拍帧的自动标注（针对「白桌面 + 密集多面体 + 机械臂杂物」场景）

⚠️ **负面结论（2026-09-24 实测，先读这段再用）**：
   本工具在当前现场场景下**未达到可用质量**，不建议直接用于训练标注：
     · 全局比例阈值取种子 → 掩膜里机械臂大块把 dt.max() 拉高，小货物一个种子都取不到（0 框）
     · 改用距离变换局部极大取种子 → 反过来过分割，实测 **平均 52 框/帧**（实际货物约 10~15），
       框大量重叠且落在线缆/阴影/桌沿上
     · 根因是掩膜本身噪声大：暗色 RGB 通道会把线缆、阴影、机械臂黑件一起收进来；
       深度通道会把机械臂/控制板/底座（凸起 100mm+，连成 58k px 大块）当成货物
   结论：**这个场景优先用 propose_boxes.py（双通道直提议，实测 5~8 框/帧、多数落在货物上）
   + review_app 人工复核**，把工作量从"画 10 个框"降到"改 2~3 个框"。
   本工具保留作为记录：若将来场景变成"稀疏摆放 + 桌面干净"，可再评估。

与 propose_boxes.py 的区别：本工具是**面向标注**的，目标是把「画框」变成
「确认/微调」，所以做了三件 propose_boxes 没做的事：

  1. **ROI 内自适应桌面平面**：整幅图的深度中位数会被地面/机械臂带偏
     （实测 75 分位已到 1267mm），必须只在托盘区域内估桌面。
  2. **双通道互补 + 分水岭拆分**：
     浅色货物 → 深度凸起；深色货物 → 深度常无回波、改用 RGB 暗块；
     密集摆放时连通域会粘成一片 → 用距离变换 + watershed 拆开。
  3. **尺寸/形状过滤**：现场货物 15~60px；机械臂、控制板、底座、线缆
     会产生 100~280px 大块，直接滤掉。

产出：<session>/auto/*.txt（YOLO 标注草稿）+ <session>/auto/*_vis.jpg（叠加图）
**仍然需要人工复核**——本工具只负责把工作量从"画 10 个框"降到"改 2~3 个框"。

用法：
    python3 tools/label_overhead_auto.py --session tray_full_20260924 \
        --roi 0,140,470,470 --write
"""
import argparse
import glob
import json
import os

import numpy as np


def load_depth(path):
    import cv2
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    return d if (d is not None and d.dtype == np.uint16) else None


def build_mask(depth, img, roi, height_mm=14.0, dark_ratio=0.72, white_ratio=1.18):
    """返回 (mask, 说明)；mask 只在 ROI 内有效"""
    import cv2
    x1, y1, x2, y2 = roi
    sub_d = depth[y1:y2, x1:x2].astype(np.float32) if depth is not None else None
    sub_i = img[y1:y2, x1:x2]
    gray = cv2.cvtColor(sub_i, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray.shape

    masks = []
    info = {}
    if sub_d is not None:
        valid = sub_d > 0
        if valid.sum() > 200:
            table = float(np.median(sub_d[valid]))
            # 只保留「桌面附近」的深度：太近的多半是机械臂/人手/支架
            near = valid & (sub_d < table - height_mm) & (sub_d > table - 120)
            masks.append(near)
            info["table_mm"] = round(table, 1)
            info["depth_px"] = int(near.sum())
    # 深色货物：白桌面上的暗块（黑三棱柱/黑球/黑正方体在深度图上常无回波）
    bright = gray[gray > np.percentile(gray, 60)]
    tv = float(np.median(bright)) if bright.size else 200.0
    dark = gray < tv * dark_ratio
    masks.append(dark)
    info["table_gray"] = round(tv, 1)
    info["dark_px"] = int(dark.sum())

    mask = np.zeros((h, w), np.uint8)
    for m in masks:
        mask[m] = 255
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    return mask, info


def split_components(mask, min_area=70, max_area=3200, max_side=78, max_aspect=2.6):
    """距离变换 + watershed 把粘在一起的货物拆开，再按尺寸/形状过滤"""
    import cv2
    dt = cv2.distanceTransform(mask, cv2.DIST_L2, 3)
    if dt.max() <= 1.0:
        return []
    # 前景种子：距离变换的**局部极大**（每颗货物一个种子）。
    # 注意：不能用"全局比例阈值"取种子 —— 掩膜里若存在机械臂那样的大块，
    # dt.max() 会被它拉高，导致小货物一个种子都取不到（实测直接 0 个候选）。
    local_max = (cv2.dilate(dt, np.ones((11, 11), np.uint8)) == dt) & (dt > 2.0)
    sure_fg = np.uint8(local_max) * 255
    n_seed, markers = cv2.connectedComponents(sure_fg)
    if n_seed <= 1:
        return []
    markers = markers + 1
    unknown = cv2.subtract(mask, sure_fg)
    markers[unknown == 255] = 0
    img3 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    markers = cv2.watershed(img3, markers)

    out = []
    for lab in range(2, n_seed + 1):
        ys, xs = np.nonzero(markers == lab)
        if ys.size < min_area:
            continue
        x1, x2 = int(xs.min()), int(xs.max())
        y1, y2 = int(ys.min()), int(ys.max())
        bw, bh = x2 - x1 + 1, y2 - y1 + 1
        area = int(ys.size)
        if area > max_area or max(bw, bh) > max_side:
            continue
        if max(bw, bh) / max(1.0, min(bw, bh)) > max_aspect:
            continue
        fill = area / float(bw * bh)
        if fill < 0.45:            # 太稀疏的轮廓多半是线缆阴影
            continue
        out.append([x1, y1, x2, y2, area, round(fill, 2)])
    out.sort(key=lambda b: -b[4])
    return out


def main():
    ap = argparse.ArgumentParser(description="现场俯拍帧自动标注（草稿）")
    ap.add_argument("--root", default="data/real_overhead")
    ap.add_argument("--session", required=True)
    ap.add_argument("--roi", default="0,140,470,470", help="托盘区域 x1,y1,x2,y2")
    ap.add_argument("--height-mm", type=float, default=14.0)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()

    import cv2
    roi = tuple(int(v) for v in args.roi.split(","))
    sdir = os.path.join(args.root, args.session)
    imgs = sorted(glob.glob(os.path.join(sdir, "images", "*")))
    if not imgs:
        raise SystemExit("没有图片: {0}".format(sdir))
    out_dir = os.path.join(sdir, "auto")
    os.makedirs(out_dir, exist_ok=True)

    total, summary = 0, {}
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        dep = load_depth(os.path.join(sdir, "depth", name + ".png"))
        mask, info = build_mask(dep, img, roi, args.height_mm)
        boxes = split_components(mask)
        H, W = img.shape[:2]
        lines = []
        for x1, y1, x2, y2, area, fill in boxes:
            X1, Y1 = x1 + roi[0], y1 + roi[1]
            X2, Y2 = x2 + roi[0], y2 + roi[1]
            cx, cy = (X1 + X2) / 2.0 / W, (Y1 + Y2) / 2.0 / H
            bw, bh = (X2 - X1) / float(W), (Y2 - Y1) / float(H)
            lines.append("0 {0:.6f} {1:.6f} {2:.6f} {3:.6f}".format(cx, cy, bw, bh))
            cv2.rectangle(img, (X1, Y1), (X2, Y2), (0, 0, 255), 2)
        if args.write:
            with open(os.path.join(out_dir, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(os.path.join(out_dir, name + "_vis.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        summary[name] = {"n": len(boxes), "table_mm": info.get("table_mm")}
        total += len(boxes)
        print("  {0}: {1} 个候选  桌面 {2}mm".format(name, len(boxes), info.get("table_mm")))

    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"),
              ensure_ascii=False, indent=2)
    print("共 {0} 个框 / {1} 张图（平均 {2:.1f} 个/图）".format(total, len(imgs), total / max(1, len(imgs))))
    print("预览: {0}/*_vis.jpg".format(out_dir))
    print("**草稿仍需人工复核**" + ("（已写入 auto/*.txt）" if args.write else "（加 --write 写入）"))


if __name__ == "__main__":
    main()
