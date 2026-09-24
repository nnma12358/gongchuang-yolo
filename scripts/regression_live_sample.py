#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""现场实机回归：连续采样视觉链路，输出检出统计与标记图

用途（重训目标的第 ⑤ 步）：模型换版后，用**同一套脚本**在同一现场条件下
对比"改前 / 改后"，而不是凭感觉说"好像好了"。

采集内容（每轮）：
    · /api/cameras/<id>/frame.jpg           原始帧（不含标记）
    · 视觉容器的 /detect/frame JSON         检出框 / 置信度 / 属性 / 2.5D
    · 视觉容器的 /frame.jpg                 带标记的画面（供人工核对）

输出：<out>/det_*.json、raw_*.jpg、marked_*.jpg、summary.json

基线（2026-09-20 实测，供对比）：
    真实货物 0/5~6 检出；60 帧中 48 帧存在同一个假检（黑色工具箱，conf 0.365~0.487）

用法：
    python3 scripts/regression_live_sample.py --base http://10.60.10.24 --n 60
    python3 scripts/regression_live_sample.py --base http://127.0.0.1 --n 30 --save-every 5
"""
import argparse
import json
import os
import time

import requests


def get(url, timeout=10, binary=False):
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.content if binary else r.json()


def main():
    ap = argparse.ArgumentParser(description="现场实机回归采样")
    ap.add_argument("--base", default="http://127.0.0.1", help="Jetson 网关地址（:80）")
    ap.add_argument("--cam", default="main", help="取原图用哪一路相机")
    ap.add_argument("--n", type=int, default=60, help="采样轮数")
    ap.add_argument("--interval", type=float, default=0.0, help="每轮额外间隔秒")
    ap.add_argument("--out", default="regression_out")
    ap.add_argument("--save-every", type=int, default=5, help="每 N 轮存一张原图/标记图（0=全存）")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    recs, t0 = [], time.time()
    hit_frames, errors, all_boxes = 0, 0, {}

    for i in range(args.n):
        row = {"i": i, "t": round(time.time() - t0, 2)}
        try:
            j = get("{0}/api/vision/detect/frame".format(args.base), timeout=25)
            dets = j.get("detections") or []
            row["det"] = j
            row["n"] = len(dets)
            if dets:
                hit_frames += 1
            for d in dets:
                key = d.get("label") or d.get("name")
                all_boxes.setdefault(key, []).append(round(float(d.get("conf") or 0), 3))
            with open(os.path.join(args.out, "det_{0:04d}.json".format(i)), "w") as f:
                json.dump(j, f, ensure_ascii=False)
        except Exception as e:
            row["err"] = str(e)[:120]
            errors += 1
        if args.save_every and i % args.save_every == 0:
            for name, url in (("raw", "{0}/api/cameras/{1}/frame.jpg".format(args.base, args.cam)),
                              ("marked", "{0}/frame.jpg?draw=1".format(args.base))):
                try:
                    data = get(url, timeout=10, binary=True)
                    with open(os.path.join(args.out, "{0}_{1:04d}.jpg".format(name, i)), "wb") as f:
                        f.write(data)
                except Exception:
                    pass
        recs.append(row)
        if args.interval:
            time.sleep(args.interval)

    el = time.time() - t0
    summary = {
        "rounds": len(recs), "elapsed_s": round(el, 1),
        "frames_with_detection": hit_frames,
        "rounds_failed": errors,
        "hit_rate": round(hit_frames / max(1, len(recs)), 3),
        "boxes_by_label": {k: {"count": len(v), "conf_mean": round(sum(v) / len(v), 3),
                               "conf_min": min(v), "conf_max": max(v)} for k, v in all_boxes.items()},
        "note": "基线：真实货物 0 检出 + 黑色工具箱假检 48/60 帧（conf 0.365~0.487）",
    }
    json.dump(summary, open(os.path.join(args.out, "summary.json"), "w"), ensure_ascii=False, indent=2)
    print("采样 {0} 轮，用时 {1:.1f}s，有检出 {2} 帧（{3:.0%}），失败 {4} 轮".format(
        len(recs), el, hit_frames, summary["hit_rate"], errors))
    if errors == len(recs):
        print("  [FAIL] 全部轮次都失败 —— 请检查网关/视觉容器是否在跑、地址是否正确：")
        print("         {0}".format(recs[0].get("err")))
        print("         （注意：不要把「请求失败」当成「没有检出」）")
        return 1
    for k, v in summary["boxes_by_label"].items():
        print("  {0}: {1} 次，conf 均值 {2}（{3}~{4}）".format(k, v["count"], v["conf_mean"],
                                                              v["conf_min"], v["conf_max"]))
    print("输出目录: {0}（summary.json / det_*.json / 抽样图）".format(args.out))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main() or 0)
