#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从现场真实帧里挖"硬负样本"（非货物区域），用于压制误检

为什么需要：赛项验收里"不要把箱子/机械臂/杂物认成货物"和"检出货物"同等重要。
现场排查时最典型的假检就是**黑色工具箱/机械臂被当成货物**（实测 conf 0.365~0.487、
60 帧里 48 帧都有）。合成数据里没有现场那些具体杂物，所以必须从**真实画面**里挖。

做法：按分区裁出**确定不含货物**的区域，存成 `bg_*` 会话（空标注）——
assemble_overhead_dataset.py 会把 `bg_*` 会话整批当负样本放进训练集。

分区（按现场画面布局，可用 --regions 覆盖）：
    clutter  上方杂物区（机械臂/控制板/线缆）—— 假检主要来源
    floor    右侧地面
    table    空桌面（教模型"白桌面≠货物"）

用法：
    python3 tools/mine_scene_negatives.py --session tray_full_20260924 \
        --out-session bg_tray_20260924 --patch 256 --stride 128
"""
import argparse
import glob
import os

DEFAULT_REGIONS = {
    # 注意：分区必须与"货物簇"错开，否则块会被 --exclude 全部挡掉（实测踩过：默认 256 裁块 → 0 个负样本）
    "clutter": (0, 0, 340, 180),      # 上方机械臂/控制板/线缆
    "floor": (470, 0, 640, 480),      # 右侧地面
    "table": (330, 150, 470, 260),    # 右侧空桌面（教"白桌面≠货物"）
}


def main():
    ap = argparse.ArgumentParser(description="从真实帧挖硬负样本（非货物区域）")
    ap.add_argument("--root", default="data/real_overhead")
    ap.add_argument("--session", required=True, help="真实帧会话（只取里面的图像）")
    ap.add_argument("--out-session", default="bg_scene", help="负样本会话名（建议 bg_ 前缀）")
    ap.add_argument("--patch", type=int, default=128, help="裁块边长（要小于分区尺寸，否则出不了块）")
    ap.add_argument("--stride", type=int, default=64, help="滑窗步长")
    ap.add_argument("--exclude", default="60,250,320,480",
                    help="货物簇区域 x1,y1,x2,y2 —— 与其重叠的块一律丢弃，避免把货物当背景")
    args = ap.parse_args()

    import cv2
    src = os.path.join(args.root, args.session, "images")
    imgs = sorted(glob.glob(os.path.join(src, "*")))
    if not imgs:
        raise SystemExit("没有图片: {0}".format(src))
    ex = tuple(int(v) for v in args.exclude.split(","))
    out_img = os.path.join(args.root, args.out_session, "images")
    out_lab = os.path.join(args.root, args.out_session, "labels")
    os.makedirs(out_img, exist_ok=True)
    os.makedirs(out_lab, exist_ok=True)

    def overlaps(a):
        return not (a[2] <= ex[0] or a[0] >= ex[2] or a[3] <= ex[1] or a[1] >= ex[3])

    n = 0
    for p in imgs:
        img = cv2.imread(p)
        H, W = img.shape[:2]
        base = os.path.splitext(os.path.basename(p))[0]
        for rname, (rx1, ry1, rx2, ry2) in DEFAULT_REGIONS.items():
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(W, rx2), min(H, ry2)
            for y in range(ry1, max(ry1 + 1, ry2 - args.patch + 1), args.stride):
                for x in range(rx1, max(rx1 + 1, rx2 - args.patch + 1), args.stride):
                    box = (x, y, x + args.patch, y + args.patch)
                    if box[2] > W or box[3] > H:
                        continue
                    if overlaps(box):
                        continue                      # 与货物区重叠 → 不能当背景
                    crop = img[box[1]:box[3], box[0]:box[2]]
                    if crop.size == 0:
                        continue
                    name = "{0}_{1}_{2:03d}_{3:03d}".format(base, rname, x, y)
                    cv2.imwrite(os.path.join(out_img, name + ".jpg"), crop,
                                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
                    open(os.path.join(out_lab, name + ".txt"), "w").close()   # 空标注=负样本
                    n += 1
        # 整幅缩略图也作为负样本？不 —— 整幅含货物，交给人工标注
    print("已生成 {0} 个硬负样本 -> {1}".format(n, os.path.join(args.root, args.out_session)))
    print("  分区: " + "; ".join("{0}={1}".format(k, v) for k, v in DEFAULT_REGIONS.items()))
    print("  排除货物区: {0}".format(ex))
    print("  下一步: assemble_overhead_dataset.py 会把 bg_* 会话当负样本并入训练集")


if __name__ == "__main__":
    main()
