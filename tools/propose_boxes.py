# -*- coding: utf-8 -*-
"""用深度图自动提议货物框（供人工复核，大幅减少标注工作量）

原理：货物摆在桌面上，深度图上比桌面平面高 15~80mm；而白货白桌在 RGB 上几乎
分不出来，深度却能稳定分开。于是：
    1) 取有效深度中位数作为桌面平面
    2) 掩膜 = 比桌面近 threshold_mm 以上的像素（即凸起物）
    3) 形态学去噪 + 连通域 → 外接框
    4) 按面积/长宽比/高度过滤，缩放到 RGB 尺寸后写成 YOLO 标注

输出：
    <session>/proposals/<name>.txt    候选 YOLO 标注（class 0 goods）
    <session>/proposals/<name>_vis.jpg 叠加框的预览图，供肉眼确认
    <session>/proposals/summary.json  每张图的候选数与高度统计

**这些只是候选，必须人工复核后再当作训练标注使用**（复核可直接用 tools/review_app.py）。

用法：
    python3 tools/propose_boxes.py --session tray_a
    python3 tools/propose_boxes.py --session tray_a --height-mm 12 --min-area 60
"""
import argparse
import glob
import json
import os

import numpy as np


def load_depth_png(path):
    import cv2
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def propose_rgb_dark(img_bgr, roi=None, min_side=14, max_side=130, max_aspect=2.0,
                     dark_ratio=0.72):
    """白桌面上的深色货物（黑三棱柱/黑球/黑正方体等）—— 这些在深度图上常没有回波"""
    """RGB 通道：白桌面上的深色货物（黑三棱柱/黑球/黑正方体）在深度图上常常没有回波
    （红外被黑色吸收），但 RGB 上与桌面反差很大。用「桌面亮度 × dark_ratio」阈值找暗块。"""
    import cv2
    H, W = img_bgr.shape[:2]
    x1, y1, x2, y2 = roi if roi else (0, 0, W, H)
    sub = img_bgr[y1:y2, x1:x2]
    gray = cv2.cvtColor(sub, cv2.COLOR_BGR2GRAY)
    # 桌面亮度：取亮部中位数（现场桌面为白色，占画面主部）
    bright = gray[gray > np.percentile(gray, 60)]
    table_v = float(np.median(bright)) if bright.size else 200.0
    mask = (gray < table_v * dark_ratio).astype(np.uint8) * 255
    k = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[-2] if len(cnts) == 3 else cnts[0]
    out = []
    for c in cnts:
        bx, by, bw, bh = cv2.boundingRect(c)
        # 贴边的大块多半是画面外的杂物/暗地面，直接丢
        if bx <= 1 or by <= 1 or bx + bw >= sub.shape[1] - 1 or by + bh >= sub.shape[0] - 1:
            continue
        if not (min_side <= bw <= max_side and min_side <= bh <= max_side):
            continue
        if max(bw, bh) / max(1.0, min(bw, bh)) > max_aspect:
            continue
        if (bw * bh) < 0.45 * (bw * bh + 1):   # 占位（保序，便于将来按填充率过滤）
            continue
        out.append([bx + x1, by + y1, bx + x1 + bw, by + y1 + bh, 0.0, float(bw * bh)])
    return out


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    iw, ih = max(0.0, x2 - x1), max(0.0, y2 - y1)
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def merge_boxes(depth_boxes, rgb_boxes, iou_thres=0.25):
    """深度块优先（带真实高度），RGB 块只补深度看不到的深色货物"""
    merged = list(depth_boxes)
    for b in rgb_boxes:
        if all(iou(b, m) < iou_thres for m in merged):
            merged.append(b)
    merged.sort(key=lambda x: -x[4])          # 有高度的排前面
    return merged



def propose_depth_bands(depth_u16, rgb_shape, roi=None, lo=12.0, hi=55.0,
                        min_area=150, max_side=78, max_aspect=2.6,
                        med_prot=14.0, close_px=3):
    """深度"高度带通"提议框 —— 针对「白桌面 + 机械臂杂物」场景的主力通道。

    与朴素的"比桌面中位数近就当选"相比，这里做两件关键的事：
      1. **背景建模**：大核形态学闭运算得到背景面，凸起 = 背景 - 当前深度。
         这样桌面本身的倾斜/起伏不会被误判（现场桌面中位 683mm，但远处地面 1267mm，
         全局中位数根本不能当平面）。
      2. **高度带通**：只取凸起在 [lo, hi]（默认 12~55mm）的像素。
         货物凸起 20~40mm 落在带内；而机械臂/控制板/底座凸起 100mm+ 被上限排除 ——
         这正是纯下限阈值会把机械臂当货物的原因。

    再叠加形态学去斑 + 面积/长宽比/最长边过滤 + **组件中值凸起门槛**（去掉深度噪声
    在物体边缘产生的细碎亮斑）。
    """
    import cv2
    if depth_u16 is None:
        return []
    d = depth_u16.astype(np.float32)
    valid = d > 0
    if valid.sum() < 500:
        return []
    x1, y1, x2, y2 = roi if roi else (0, 0, d.shape[1], d.shape[0])
    sub = d[y1:y2, x1:x2]
    sub_valid = valid[y1:y2, x1:x2]
    fill = sub.copy()
    fill[~sub_valid] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (41, 41))
    bg = cv2.morphologyEx(fill, cv2.MORPH_CLOSE, k)
    prot = bg - sub
    mask = (sub_valid & (prot > lo) & (prot < hi)).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []
    kk = np.ones((close_px, close_px), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kk)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kk)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[-2] if len(cnts) == 3 else cnts[0]
    H, W = rgb_shape[:2]
    # **缩放要按"整幅深度图 vs RGB"算**，不能按裁剪后的 mask 尺寸算：
    # mask 是 ROI 裁剪过的，用它当分母会把坐标放大 (整幅/ROI) 倍
    # （实测 ROI 470x330 时放大 1.36~1.45 倍，框全部偏离货物 —— 一度让我误判提议质量差）。
    dh, dw = d.shape[:2]
    sx, sy = float(W) / dw, float(H) / dh
    if x1 >= dw or y1 >= dh:            # ROI 用 RGB 坐标给出且分辨率一致时无需换算
        pass
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = int((mask[y:y + h, x:x + w] > 0).sum())
        if area < min_area:
            continue
        if max(w, h) > max_side or max(w, h) / max(1.0, min(w, h)) > max_aspect:
            continue
        vals = prot[y:y + h, x:x + w][mask[y:y + h, x:x + w] > 0]
        if vals.size == 0 or float(np.median(vals)) < med_prot:
            continue                       # 中值凸起不够 → 多半是噪声边缘
        fill_ratio = area / float(w * h)
        if fill_ratio < 0.5:
            continue
        out.append([(x + x1) * sx, (y + y1) * sy, (x + x1 + w) * sx, (y + y1 + h) * sy,
                    float(np.median(vals)), float(area)])
    out.sort(key=lambda b: -b[4])
    return out


def propose(depth_u16, rgb_shape, height_mm=15.0, min_area=50, max_area=40000,
            max_aspect=3.2, min_height=12.0, close_px=3, max_side=0):
    """返回 [(x1,y1,x2,y2,height_mm,area), ...]（RGB 像素坐标）"""
    import cv2
    if depth_u16 is None:
        return []
    d = depth_u16.astype(np.float32)
    valid = d > 0
    if valid.sum() < 100:
        return []
    table = float(np.median(d[valid]))
    mask = ((d > 0) & (d < table - height_mm)).astype(np.uint8) * 255
    if mask.sum() == 0:
        return []
    k = np.ones((close_px, close_px), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[-2] if len(cnts) == 3 else cnts[0]
    H, W = rgb_shape[:2]
    # **缩放要按"整幅深度图 vs RGB"算**，不能按裁剪后的 mask 尺寸算：
    # mask 是 ROI 裁剪过的，用它当分母会把坐标放大 (整幅/ROI) 倍
    # （实测 ROI 470x330 时放大 1.36~1.45 倍，框全部偏离货物 —— 一度让我误判提议质量差）。
    dh, dw = d.shape[:2]
    sx, sy = float(W) / dw, float(H) / dh
    if x1 >= dw or y1 >= dh:            # ROI 用 RGB 坐标给出且分辨率一致时无需换算
        pass
    out = []
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = w * h
        if area < min_area or area > max_area:
            continue
        if max(w, h) / max(1.0, min(w, h)) > max_aspect:
            continue
        # 尺寸上限：现场货物只有 15~60px，而机械臂/控制板/底座/线缆会产生
        # 100~280px 的大块。不过滤的话候选框一半是杂物，人工复核反而更累。
        if max_side and max(w * sx, h * sy) > max_side:
            continue
        sub = d[y:y + h, x:x + w]
        sv = sub[(sub > 0) & (sub < table - min_height)]
        if sv.size == 0:
            continue
        hgt = float(table - np.median(sv))
        if hgt < min_height:
            continue
        out.append([x * sx, y * sy, (x + w) * sx, (y + h) * sy, hgt, float(area)])
    out.sort(key=lambda b: -b[4])
    return out


def main():
    ap = argparse.ArgumentParser(description="深度自动提议货物框（供人工复核）")
    ap.add_argument("--root", default="data/real_overhead")
    ap.add_argument("--session", required=True)
    ap.add_argument("--height-mm", type=float, default=15.0, help="高于桌面多少 mm 算货物")
    ap.add_argument("--max-side", type=int, default=0,
                    help="候选框最长边上限（像素，0=不限）。现场货物 15~60px，"
                         "建议 70~90：可滤掉机械臂/控制板/线缆等大块杂物")
    ap.add_argument("--min-area", type=int, default=50, help="最小框面积（RGB 像素）")
    ap.add_argument("--max-area", type=int, default=40000)
    ap.add_argument("--max-aspect", type=float, default=3.2)
    ap.add_argument("--min-height", type=float, default=12.0, help="最小凸起高度 mm")
    ap.add_argument("--roi", default=None, help="工作区 ROI: x1,y1,x2,y2（限定 RGB 通道搜索范围）")
    ap.add_argument("--no-rgb", action="store_true", help="只用深度通道")
    ap.add_argument("--bands", action="store_true",
                    help="用深度高度带通通道（现场白桌面+机械臂场景首选）")
    ap.add_argument("--band-lo", type=float, default=12.0, help="高度带通下限 mm")
    ap.add_argument("--band-hi", type=float, default=55.0, help="高度带通上限 mm（排除机械臂等大凸起）")
    ap.add_argument("--write", action="store_true", help="写入 proposals/*.txt（默认只预览）")
    args = ap.parse_args()

    import cv2
    sdir = os.path.join(args.root, args.session)
    imgs = sorted(glob.glob(os.path.join(sdir, "images", "*")))
    if not imgs:
        raise SystemExit("会话里没有图片: {0}".format(sdir))
    out_dir = os.path.join(sdir, "proposals")
    os.makedirs(out_dir, exist_ok=True)

    summary, n_box = {}, 0
    for p in imgs:
        name = os.path.splitext(os.path.basename(p))[0]
        img = cv2.imread(p)
        dep = load_depth_png(os.path.join(sdir, "depth", name + ".png"))
        # 采集时深度已归一化为 8bit 伪彩色，无法直接测距 → 提示改用原始 16bit
        if dep is not None and dep.dtype != np.uint16 and dep.ndim == 3:
            dep = None
        if args.bands:
            roi0 = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
            boxes = propose_depth_bands(dep, img.shape, roi=roi0,
                                        lo=args.band_lo, hi=args.band_hi,
                                        min_area=max(120, args.min_area),
                                        max_side=args.max_side or 78)
        else:
            boxes = propose(dep, img.shape, args.height_mm, args.min_area, args.max_area,
                            args.max_aspect, args.min_height, max_side=args.max_side)
        if not args.no_rgb:
            roi = None
            if args.roi:
                roi = tuple(int(v) for v in args.roi.split(","))
            boxes = merge_boxes(boxes, propose_rgb_dark(img, roi=roi,
                                                        max_side=args.max_side or 130))
        summary[name] = {"n": len(boxes),
                         "heights_mm": [round(b[4], 1) for b in boxes]}
        n_box += len(boxes)
        H, W = img.shape[:2]
        lines = []
        for x1, y1, x2, y2, hgt, area in boxes:
            cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
            bw, bh = (x2 - x1) / W, (y2 - y1) / H
            lines.append("0 {0:.6f} {1:.6f} {2:.6f} {3:.6f}".format(cx, cy, bw, bh))
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)
            cv2.putText(img, "{0:.0f}mm".format(hgt), (int(x1), max(10, int(y1) - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        if args.write:
            with open(os.path.join(out_dir, name + ".txt"), "w") as f:
                f.write("\n".join(lines) + ("\n" if lines else ""))
        cv2.imwrite(os.path.join(out_dir, name + "_vis.jpg"), img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("  {0}: {1} 个候选 {2}".format(
            name, len(boxes), [round(b[4]) for b in boxes[:8]]))

    json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"),
              ensure_ascii=False, indent=2)
    print("共 {0} 个候选框，覆盖 {1} 张图".format(n_box, len(imgs)))
    print("预览: {0}/*_vis.jpg".format(out_dir))
    if args.write:
        print("已写候选标注: {0}/*.txt —— 请务必人工复核后再用于训练".format(out_dir))
    else:
        print("（当前只预览；确认参数合适后加 --write 写入候选标注）")


if __name__ == "__main__":
    main()
