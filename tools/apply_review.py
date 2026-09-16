#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
apply_review.py —— 把人工复核结果并回源数据集（并标记为 human）
=====================================================================
复核包 data/review/review_set/{images,labels} 里的文件带了顺序前缀（001_xxx.jpg），
本脚本按 audit.json 的 mapping 还原成原名，写回 data/real_synth_mix：
  · 已标注图 → 覆盖 labels/<split>/<name>.txt，provenance.source 改为 "human"
  · 未标注图（原来在 images/unlabeled）→ 移到 images/train 并生成标注
  · 复核后为空的标签 → **保留为空文件**（负样本）；源数据集里没有空标签惯例时直接写入即可

安全措施：
  · 默认 --dry-run，只看会发生什么；确认后加 --write
  · 覆盖前把原文件备份到 <mixed>/.backup_<时间戳>/
  · 只有 review_state.json 里标为已复核的图才会被写入（--all 可强制全部）

用法：
  python tools/apply_review.py --dry-run
  python tools/apply_review.py --write
"""
import argparse
import json
import os
import shutil
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser(description="复核结果并回源数据集")
    ap.add_argument("--review", default="data/review/review_set")
    ap.add_argument("--audit", default="data/review/audit.json")
    ap.add_argument("--mixed", default="data/real_synth_mix")
    ap.add_argument("--write", action="store_true", help="真正写回（默认只预览）")
    ap.add_argument("--all", action="store_true", help="不要求 reviewed 标记，全部写回")
    args = ap.parse_args()

    def rp(p):
        return p if os.path.isabs(p) else os.path.join(ROOT, p)

    rv, mixed = rp(args.review), rp(args.mixed)
    a = json.load(open(rp(args.audit), encoding="utf-8"))
    st_path = os.path.join(rv, "review_state.json")
    state = json.load(open(st_path, encoding="utf-8")) if os.path.exists(st_path) else {}
    summ_path = os.path.join(mixed, "dataset_summary.json")
    summ = json.load(open(summ_path, encoding="utf-8"))

    backup = os.path.join(mixed, ".backup_" + time.strftime("%Y%m%d_%H%M%S"))
    todo = []
    for m in a["mapping"]:
        rv_name, orig = m["review_name"], m["original"]
        if not args.all and not state.get(rv_name):
            continue
        lp = os.path.join(rv, "labels", os.path.splitext(rv_name)[0] + ".txt")
        if not os.path.exists(lp):
            continue
        lines = [ln for ln in open(lp, encoding="utf-8").read().splitlines() if ln.strip()]
        # 原图在哪
        src_split, src_img = None, None
        for sp in ("train", "val", "unlabeled"):
            p = os.path.join(mixed, "images", sp, orig)
            if os.path.exists(p):
                src_split, src_img = sp, p
                break
        todo.append({"review_name": rv_name, "orig": orig, "boxes": len(lines),
                     "src_split": src_split, "label_lines": lines})

    n_done = sum(1 for v in state.values() if v)
    print("=" * 70)
    print(" 复核记录：%d 张已标记完成（review_state.json）" % n_done)
    print(" 本次将写回：%d 张%s" % (len(todo), "（--all）" if args.all else ""))
    print("=" * 70)
    kinds = {}
    for t in todo:
        k = "未标注→新增标注" if t["src_split"] == "unlabeled" else "覆盖已有标注"
        kinds[k] = kinds.get(k, 0) + 1
    for k, v in kinds.items():
        print("   %s：%d 张" % (k, v))
    empt = [t for t in todo if t["boxes"] == 0]
    if empt:
        print("   其中复核为空标签（按负样本处理）：%d 张 —— %s"
              % (len(empt), ", ".join(t["orig"] for t in empt[:6])))
    print("   框数变化：%d → %d" % (0, sum(t["boxes"] for t in todo)))

    if not args.write:
        print("\n[预览模式] 加 --write 才会真正写入。备份目录将是：", backup)
        for t in todo[:10]:
            print("   %-22s %-10s → %s（%d 框）"
                  % (t["orig"], t["src_split"], "新增" if t["src_split"] == "unlabeled"
                     else "覆盖", t["boxes"]))
        return

    os.makedirs(backup, exist_ok=True)
    changed = 0
    for t in todo:
        orig, sp = t["orig"], t["src_split"]
        if sp is None:
            print("  跳过（源图不存在）:", orig)
            continue
        stem = os.path.splitext(orig)[0]
        # 备份原标签
        old_lp = os.path.join(mixed, "labels", sp if sp != "unlabeled" else "train", stem + ".txt")
        if os.path.exists(old_lp):
            os.makedirs(os.path.join(backup, "labels"), exist_ok=True)
            shutil.copy2(old_lp, os.path.join(backup, "labels", stem + ".txt"))
        # 未标注图 → 移到 train
        if sp == "unlabeled":
            dst_img = os.path.join(mixed, "images", "train", orig)
            shutil.move(os.path.join(mixed, "images", "unlabeled", orig), dst_img)
            sp = "train"
        os.makedirs(os.path.join(mixed, "labels", sp), exist_ok=True)
        new_lp = os.path.join(mixed, "labels", sp, stem + ".txt")
        with open(new_lp, "w", encoding="utf-8") as f:
            f.write("\n".join(t["label_lines"]) + ("\n" if t["label_lines"] else ""))
        # 溯源改为 human
        summ.setdefault("labels_source", {})[orig] = {
            "source": "human", "group": "real", "reviewed": True,
            "from": (summ["labels_source"].get(orig) or {}).get("from", orig)}
        changed += 1
    json.dump(summ, open(summ_path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    n_human = sum(1 for v in summ["labels_source"].values() if v.get("source") == "human")
    print("\n✅ 已写回 %d 张；源数据集人工标注总数：%d" % (changed, n_human))
    print("   备份：", backup)
    print("   下一步：python tools/build_mix_v3.py --group-split && "
          "python tools/check_labels.py --data data/mix_v3")


if __name__ == "__main__":
    main()
