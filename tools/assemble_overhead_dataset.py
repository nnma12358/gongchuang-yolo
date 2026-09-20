# -*- coding: utf-8 -*-
"""组装「现场真实尺度」训练集：原有 mix_v3 + 真实俯拍照片 + 现场几何合成数据

关键规则（避免把「有货物但没标注」的图当成背景喂进去）：
  * 真实会话名以 `bg_` 开头 → 视为纯背景负样本，空标注合法，全部进 train；
  * 其它真实会话 → **只收标注非空的图**；没有标注的整批不收（会在报告里明确提示）；
  * 真实会话按 8:2 切 train/val（按序号切，避免同一批连拍跨集泄漏）；
  * 原 mix_v3（含人工复核过的实拍 + 旧合成 + 负样本）整体并入，防止遗忘。

用法：
    python3 tools/assemble_overhead_dataset.py \
        --real data/real_overhead --synth data/synth_realgeo \
        --base data/mix_v3 --out data/mix_overhead
"""
import argparse
import glob
import json
import os
import shutil


def nonempty(p):
    try:
        return os.path.getsize(p) > 0
    except OSError:
        return False


def copy_pair(src_img, src_lab, dst_root, split, new_name):
    di = os.path.join(dst_root, "images", split)
    dl = os.path.join(dst_root, "labels", split)
    os.makedirs(di, exist_ok=True)
    os.makedirs(dl, exist_ok=True)
    shutil.copyfile(src_img, os.path.join(di, new_name + os.path.splitext(src_img)[1]))
    if src_lab and os.path.isfile(src_lab):
        shutil.copyfile(src_lab, os.path.join(dl, new_name + ".txt"))
    else:
        open(os.path.join(dl, new_name + ".txt"), "w").close()


def label_for(img_path):
    """图片路径 → 对应标注 txt。两种布局都要支持：
       标准 YOLO： <root>/images/<split>/x.jpg  -> <root>/labels/<split>/x.txt
       采集会话： <sess>/images/x.jpg           -> <sess>/labels/x.txt
    """
    d, n = os.path.split(img_path)
    base = os.path.splitext(n)[0] + ".txt"
    parent = os.path.basename(d)          # train | val | images
    root = os.path.dirname(d)
    if parent == "images":                # 采集会话布局
        return os.path.join(root, "labels", base)
    return os.path.join(os.path.dirname(root), "labels", parent, base)


def main():
    ap = argparse.ArgumentParser(description="组装现场真实尺度训练集")
    ap.add_argument("--base", default=None, help="原有数据集（如 data/mix_v3），整包并入")
    ap.add_argument("--real", default="data/real_overhead", help="真实照片根目录（每个子目录一个会话）")
    ap.add_argument("--synth", default=None, help="现场几何合成数据目录")
    ap.add_argument("--out", default="data/mix_overhead")
    ap.add_argument("--val-ratio", type=float, default=0.2)
    args = ap.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for s in ("train", "val"):
        os.makedirs(os.path.join(args.out, "images", s), exist_ok=True)
        os.makedirs(os.path.join(args.out, "labels", s), exist_ok=True)

    report = {"base": 0, "synth": 0, "real_pos": 0, "real_neg": 0, "skipped_real": {}}

    # 1) 原有数据集
    if args.base:
        for s in ("train", "val"):
            for p in sorted(glob.glob(os.path.join(args.base, "images", s, "*"))):
                name = "base_" + os.path.splitext(os.path.basename(p))[0]
                copy_pair(p, label_for(p), args.out, s, name)
                report["base"] += 1

    # 2) 现场几何合成数据
    if args.synth and os.path.isdir(args.synth):
        for s in ("train", "val"):
            for p in sorted(glob.glob(os.path.join(args.synth, "images", s, "*"))):
                name = "syngeo_" + os.path.splitext(os.path.basename(p))[0]
                copy_pair(p, label_for(p), args.out, s, name)
                report["synth"] += 1

    # 3) 真实俯拍会话
    if os.path.isdir(args.real):
        for sess in sorted(os.listdir(args.real)):
            sdir = os.path.join(args.real, sess)
            imgs = sorted(glob.glob(os.path.join(sdir, "images", "*")))
            if not imgs:
                continue
            is_bg = sess.startswith("bg_")
            kept, dropped = [], 0
            for p in imgs:
                lab = label_for(p)
                if is_bg or nonempty(lab):
                    kept.append((p, lab))
                else:
                    dropped += 1
            if dropped:
                report["skipped_real"][sess] = dropped
            if not kept:
                continue
            n_val = int(round(len(kept) * args.val_ratio))
            for i, (p, lab) in enumerate(kept):
                split = "val" if (n_val > 0 and i >= len(kept) - n_val) else "train"
                name = "real_{0}_{1:04d}".format(sess, i + 1)
                copy_pair(p, lab, args.out, split, name)
                if is_bg:
                    report["real_neg"] += 1
                else:
                    report["real_pos"] += 1

    yaml_path = os.path.join(args.out, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write("path: {0}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: goods\n".format(
            os.path.abspath(args.out)))

    def count(split):
        n = len(glob.glob(os.path.join(args.out, "images", split, "*")))
        pos = sum(1 for p in glob.glob(os.path.join(args.out, "labels", split, "*.txt")) if nonempty(p))
        return n, pos

    tr, trp = count("train")
    va, vap = count("val")
    print("=== 组装完成 -> {0}".format(args.out))
    print("  原有数据 {0} 张 | 现场几何合成 {1} 张 | 真实有货 {2} 张 | 真实背景 {3} 张".format(
        report["base"], report["synth"], report["real_pos"], report["real_neg"]))
    print("  train: {0} 张（含货物 {1}）   val: {2} 张（含货物 {3}）".format(tr, trp, va, vap))
    if report["skipped_real"]:
        print("  [注意] 以下会话有图但缺标注，已整批排除（有货无标注会把货物教成背景）：")
        for k, v in report["skipped_real"].items():
            print("      {0}: {1} 张".format(k, v))
    if report["real_pos"] == 0:
        print("  [警告] 没有任何「带标注的真实俯拍图」进入训练集——")
        print("         只靠合成数据难以消除现场漏检，建议先完成标注再训练。")
    json.dump(report, open(os.path.join(args.out, "assemble_report.json"), "w"),
              ensure_ascii=False, indent=2)
    print("  data.yaml: {0}".format(yaml_path))


if __name__ == "__main__":
    main()
