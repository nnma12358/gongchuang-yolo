#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ui_smoke_test.py —— 复核台 UI 冒烟测试（真浏览器，防"看不到框"这类回归）
=====================================================================
为什么需要它：复核台是"画布 + 键盘"的应用，用 curl 只能测到 API；
真正的坑（坐标归一化 vs 像素、按键冲突、反复按 N 堆叠）只有真浏览器能发现。
本脚本用 Playwright + Chromium 打开复核台，检查：

  1) 页面无 JS 报错
  2) 图片加载成功、画布尺寸正常
  3) **标注框真的画在图片上**（在框的边框像素上采样颜色）
  4) 拖拽能新建框
  5) 按 N 能并入模型建议框，且**重复按不会越加越多**（去重）
  6) 按 Z 撤销、按 A 标记完成并跳下一张、按 ',' 返回上一张
  7) 方向键能微调选中框

准备（只需一次）：
    pip install playwright
    PLAYWRIGHT_DOWNLOAD_HOST=https://cdn.npmmirror.com/binaries/playwright \
        python -m playwright install chromium
  WSL 里若缺 libnspr4/libnss3（无 root），可免安装解包：
    mkdir -p /tmp/pwlibs && cd /tmp/pwlibs && apt-get download libnspr4 libnss3 \
      && for d in *.deb; do dpkg -x "$d" .; done
    export LD_LIBRARY_PATH=/tmp/pwlibs/usr/lib/x86_64-linux-gnu

用法：
    python tools/review_app.py --port 8770 &        # 另开一个终端
    python tools/ui_smoke_test.py --url http://127.0.0.1:8770
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser(description="复核台 UI 冒烟测试")
    ap.add_argument("--url", default="http://127.0.0.1:8770")
    ap.add_argument("--shot", default="/tmp/review_ui.png")
    args = ap.parse_args()

    from playwright.sync_api import sync_playwright
    fails, notes = [], []

    def check(name, ok, info=""):
        print(("  ✅ " if ok else "  ❌ ") + name + ("  " + str(info) if info else ""))
        if not ok:
            fails.append(name)

    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page(viewport={"width": 1360, "height": 860})
        errs, dialogs = [], []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("dialog", lambda d: (dialogs.append(d.message), d.accept()))
        pg.goto(args.url, wait_until="networkidle")
        pg.wait_for_timeout(1800)

        check("页面无 JS 报错", not errs, errs[:2])
        st = pg.evaluate("()=>({n:S.items.length, i:S.i, boxes:S.boxes.length,"
                         " W:cv.width, H:cv.height, iw:S.img?S.img.width:0})")
        check("图片已加载", st["iw"] > 0, st)
        check("画布尺寸正常", st["W"] > 64 and st["H"] > 64)

        # 关键回归：框必须画在图片上（采样边框像素）
        box = pg.evaluate("""()=>{
            const b=S.boxes[0]; if(!b) return null;
            const W=cv.width,H=cv.height;
            const at=(x,y)=>Array.from(ctx.getImageData(Math.min(W-1,Math.max(0,Math.round(x))),
                                                        Math.min(H-1,Math.max(0,Math.round(y))),1,1).data);
            return {box:b, top:at((b[0]+b[2]/2)*W, b[1]*H), left:at(b[0]*W,(b[1]+b[3]/2)*H)};
        }""")
        if box is None:
            check("第 1 张有标注框", False, "该图无框，换一张带框的再测")
        else:
            def greenish(px):      # #22d46e
                return abs(px[0] - 34) < 60 and abs(px[1] - 212) < 60 and abs(px[2] - 110) < 60
            check("标注框画在图片上（非左上角 0 像素）", greenish(box["top"]) and greenish(box["left"]),
                  {"top": box["top"], "left": box["left"], "box": [round(v, 4) for v in box["box"]]})

        # 拖拽新建框
        r = pg.evaluate("()=>{const q=cv.getBoundingClientRect();return [q.x,q.y,q.width,q.height];}")
        n0 = pg.evaluate("()=>S.boxes.length")
        pg.mouse.move(r[0] + r[2] * 0.55, r[1] + r[3] * 0.55)
        pg.mouse.down()
        pg.mouse.move(r[0] + r[2] * 0.78, r[1] + r[3] * 0.80, steps=8)
        pg.mouse.up()
        pg.wait_for_timeout(500)
        n1 = pg.evaluate("()=>S.boxes.length")
        check("拖拽能新建框", n1 == n0 + 1, "%d → %d" % (n0, n1))

        # N 去重
        pg.keyboard.press("z"); pg.wait_for_timeout(300)
        nh = pg.evaluate("()=>S.hints.length")
        if nh:
            pg.keyboard.press("n"); pg.wait_for_timeout(500)
            a = pg.evaluate("()=>S.boxes.length")
            pg.keyboard.press("n"); pg.wait_for_timeout(500)
            c = pg.evaluate("()=>S.boxes.length")
            check("按 N 并入模型建议框", a > n0, "%d → %d" % (n0, a))
            check("重复按 N 不堆叠（去重生效）", c == a, "%d → %d" % (a, c))
        else:
            notes.append("本张无模型建议框，跳过 N 测试")

        pg.keyboard.press("z"); pg.wait_for_timeout(300)
        check("按 Z 撤销回原始框数", pg.evaluate("()=>S.boxes.length") == n0)

        # 方向键微调需要先选中
        pg.keyboard.press("1"); pg.wait_for_timeout(200)
        b0 = pg.evaluate("()=>S.boxes[0].slice()")
        pg.keyboard.press("ArrowRight"); pg.wait_for_timeout(400)
        b1 = pg.evaluate("()=>S.boxes[0].slice()")
        check("方向键微调选中框", abs(b1[0] - b0[0]) > 1e-4, "%s → %s" % (round(b0[0], 4), round(b1[0], 4)))

        # A 标记 + 翻页
        i0 = pg.evaluate("()=>S.i")
        pg.keyboard.press("a"); pg.wait_for_timeout(900)
        i1 = pg.evaluate("()=>S.i")
        check("按 A 标记完成并跳下一张", i1 == i0 + 1, "%d → %d" % (i0, i1))
        pg.keyboard.press(","); pg.wait_for_timeout(700)
        check("按 , 返回上一张", pg.evaluate("()=>S.i") == i0)

        pg.screenshot(path=args.shot)
        print("  截图:", args.shot)
        b.close()

    for n in notes:
        print("  ⚠ ", n)
    print("\n结果：%s" % ("全部通过 ✅" if not fails else "失败 %d 项 ❌ %s" % (len(fails), fails)))
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
