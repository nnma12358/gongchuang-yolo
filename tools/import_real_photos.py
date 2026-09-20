# -*- coding: utf-8 -*-
"""把「用户提供的真实照片」导入为训练数据集（校验 + 去重 + 报告 + 拼图）

现场照片最要紧的不是数量而是**尺度对得上**：这次要解决的正是「货物在画面里只有
约 30px 导致漏检」，所以导入时会打印每张图的尺寸与一张拼图，方便肉眼确认。

用法：
    # 只导入图片（后续人工标注）
    python3 tools/import_real_photos.py --src ~/照片/现场俯拍 --session tray_a

    # 图片与同名 label 一起导入（label 可以是 YOLO txt，或后接 --from-labels）
    python3 tools/import_real_photos.py --src ./photos --labels ./labels --session tray_b

产出：data/real_overhead/<session>/{images,labels}/ ，并打印统计 + 生成 _contact.jpg
"""
import argparse
import hashlib
import json
import os
import shutil

IMG_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def dhash(img_gray_small):
    """极简感知哈希：8x8 均值二值化，用于剔除连拍重复图"""
    import numpy as np
    a = img_gray_small.astype("float32")
    bits = (a > a.mean()).flatten()
    return hashlib.md5(bits.tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="导入用户真实照片为训练数据集")
    ap.add_argument("--src", required=True, help="照片目录（递归扫描）")
    ap.add_argument("--labels", default=None, help="标注目录（同名 .txt；不给则留空待标注）")
    ap.add_argument("--session", required=True, help="会话名（作为输出子目录与文件名前缀）")
    ap.add_argument("--out", default="data/real_overhead")
    ap.add_argument("--min-side", type=int, default=200, help="短边小于此值的图跳过（太小无法用）")
    ap.add_argument("--contact", type=int, default=24, help="拼图抽样张数（0=不生成）")
    args = ap.parse_args()

    import cv2
    import numpy as np

    files = []
    for root, _, names in os.walk(args.src):
        for n in sorted(names):
            if n.lower().endswith(IMG_EXT):
                files.append(os.path.join(root, n))
    if not files:
        raise SystemExit("在 {0} 下没有找到图片".format(args.src))
    print("扫描到 {0} 张候选图片".format(len(files)))

    img_dir = os.path.join(args.out, args.session, "images")
    lab_dir = os.path.join(args.out, args.session, "labels")
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(lab_dir, exist_ok=True)

    seen, kept, skipped = set(), [], []
    for p in files:
        img = cv2.imread(p)
        if img is None:
            skipped.append((p, "无法读取")); continue
        h, w = img.shape[:2]
        if min(h, w) < args.min_side:
            skipped.append((p, "太小 {0}x{1}".format(w, h))); continue
        g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (9, 8))
        key = dhash(g)
        if key in seen:
            skipped.append((p, "重复")) ; continue
        seen.add(key)
        kept.append((p, img, w, h))

    print("去重/过滤后保留 {0} 张，跳过 {1} 张".format(len(kept), len(skipped)))
    for p, why in skipped[:8]:
        print("   跳过 {0}（{1}）".format(os.path.basename(p), why))

    n_lab, n_emp = 0, 0
    sizes = []
    for i, (p, img, w, h) in enumerate(kept, 1):
        name = "{0}_{1:04d}".format(args.session, i)
        shutil.copyfile(p, os.path.join(img_dir, name + ".jpg"))
        sizes.append((w, h))
        if args.labels:
            cand = [os.path.join(args.labels, os.path.splitext(os.path.basename(p))[0] + ext)
                    for ext in (".txt",)]
            src_lab = next((c for c in cand if os.path.isfile(c)), None)
            dst = os.path.join(lab_dir, name + ".txt")
            if src_lab:
                shutil.copyfile(src_lab, dst)
                if os.path.getsize(src_lab) > 0:
                    n_lab += 1
                else:
                    n_emp += 1
            else:
                open(dst, "w").close(); n_emp += 1
        else:
            open(os.path.join(lab_dir, name + ".txt"), "w").close(); n_emp += 1

    sizes = np.array(sizes)
    print("图片尺寸：宽 {0:.0f}~{1:.0f}，高 {2:.0f}~{3:.0f}".format(
        sizes[:, 0].min(), sizes[:, 0].max(), sizes[:, 1].min(), sizes[:, 1].max()))
    if args.labels:
        print("带非空标注 {0} 张 / 空标注 {1} 张".format(n_lab, n_emp))
    else:
        print("未给 --labels：全部为待标注（空 txt）。")
    print("【提醒】这些图必须与现场相机同一视角/尺度（货物约占画面宽 5%，即 640 宽时约 30px），")
    print("      否则重训解决不了现场漏检——可对照本目录生成的 _contact.jpg 与现场帧比对。")

    if args.contact and kept:
        import math
        k = min(args.contact, len(kept))
        step = max(1, len(kept) // k)
        picks = kept[::step][:k]
        cols = 6
        rows = int(math.ceil(len(picks) / cols))
        cw, ch = 220, 165
        sheet = np.full((rows * ch, cols * cw, 3), 255, np.uint8)
        for i, (p, img, w, h) in enumerate(picks):
            r, c = divmod(i, cols)
            t = cv2.resize(img, (cw - 4, ch - 4))
            sheet[r * ch + 2:r * ch + ch - 2, c * cw + 2:c * cw + cw - 2] = t
        out = os.path.join(args.out, args.session, "_contact.jpg")
        cv2.imwrite(out, sheet, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print("拼图: {0}".format(out))

    json.dump({"session": args.session, "src": args.src, "kept": len(kept),
               "skipped": [{"file": p, "why": w} for p, w in skipped],
               "labeled": n_lab, "empty_label": n_emp},
              open(os.path.join(args.out, args.session, "import.json"), "w"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
