#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
货物图库（形状 × 颜色）—— 网关容器与视觉容器共用
赛项：常用形状 + 常用颜色；货物 ≤40mm；表面附有污渍/缺陷图形、文字、二维码。
储物盒不在此处预设：由网关按“形状相同且颜色相同 → 同一储物盒”动态分配。
"""
import logging
import math
import os

logger = logging.getLogger("catalog")

SHAPES = ["正方体", "长方体", "圆柱", "球", "正四面体", "五棱柱", "六棱柱"]
COLORS = ["红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "黑色", "白色"]
MARKS = ["无", "污渍", "缺陷"]
IMAGE_FORMATS = ["svg", "png", "jpg", "webp", "gif", "bmp"]

RGB = {
    "红色": [214, 48, 49], "橙色": [230, 126, 34], "黄色": [241, 196, 15],
    "绿色": [46, 204, 113], "青色": [26, 188, 196], "蓝色": [52, 120, 246],
    "紫色": [155, 89, 182], "黑色": [45, 52, 54], "白色": [236, 240, 241],
}
GEOMETRY = {"正方体": "cube", "长方体": "cuboid", "圆柱": "cylinder", "球": "ball",
            "正四面体": "tetra", "五棱柱": "prism5", "六棱柱": "prism6"}
COLOR_ORDER = list(RGB.keys())

_COMBOS = [
    ("五棱柱", "青色"), ("正四面体", "橙色"), ("正方体", "红色"), ("正方体", "绿色"),
    ("正方体", "蓝色"), ("正方体", "黄色"), ("圆柱", "蓝色"), ("球", "黑色"),
    ("六棱柱", "紫色"), ("长方体", "橙色"), ("圆柱", "白色"), ("球", "红色"),
    ("五棱柱", "蓝色"), ("六棱柱", "绿色"), ("正四面体", "黄色"), ("长方体", "红色"),
]


def make_id(shape, color):
    return "{0}_{1}".format(GEOMETRY[shape], COLOR_ORDER.index(color))


GOODS = [{
    "id": make_id(s, c), "name": "{0}{1}".format(c, s), "class": make_id(s, c),
    "shape": s, "color": c, "rgb": RGB[c], "geometry": GEOMETRY[s],
} for s, c in _COMBOS]
GOODS_BY_ID = dict((g["id"], g) for g in GOODS)


def goods_key(shape, color):
    """货物身份键：形状 + 颜色（赛项：同形同色必须同盒）"""
    return "{0}|{1}".format(shape, color)


def match_goods(shape, color):
    for g in GOODS:
        if g["shape"] == shape and g["color"] == color:
            return g
    return None


# ==================== 图形生成（SVG / 位图，用于显示屏多格式播放） ====================
def _shade(rgb, factor):
    return tuple(max(0, min(255, int(v * factor))) for v in rgb)


def _hex(rgb):
    return "#%02x%02x%02x" % tuple(_shade(rgb, 1.0))


def polygons(geometry, size=320):
    cx = cy = size / 2.0
    s = float(size)
    if geometry in ("cube", "cuboid"):
        wx = 0.3 if geometry == "cuboid" else 0.22
        top = [(cx - wx * s, cy - .10 * s), (cx + .04 * s, cy - .26 * s),
               (cx + (wx + .08) * s, cy - .10 * s), (cx + .04 * s, cy + .06 * s)]
        left = [(cx - wx * s, cy - .10 * s), (cx + .04 * s, cy + .06 * s),
                (cx + .04 * s, cy + .34 * s), (cx - wx * s, cy + .18 * s)]
        right = [(cx + .04 * s, cy + .06 * s), (cx + (wx + .08) * s, cy - .10 * s),
                 (cx + (wx + .08) * s, cy + .18 * s), (cx + .04 * s, cy + .34 * s)]
        return [(left, .85), (right, .62), (top, 1.12)]
    if geometry in ("prism5", "prism6"):
        n = 5 if geometry == "prism5" else 6
        r, h, cy_top = .26 * s, .30 * s, cy - .16 * s
        top, bottom = [], []
        for i in range(n):
            ang = -math.pi / 2 + i * 2 * math.pi / n
            top.append((cx + r * math.cos(ang), cy_top + .16 * s * math.sin(ang)))
            bottom.append((cx + r * math.cos(ang), cy_top + h + .16 * s * math.sin(ang)))
        faces = []
        for i in range(n):
            j = (i + 1) % n
            if (top[i][1] + top[j][1]) / 2 > cy_top:
                faces.append(([top[i], top[j], bottom[j], bottom[i]], .72 if i % 2 else .90))
        return faces + [(top, 1.14)]
    if geometry == "tetra":
        apex = (cx, cy - .30 * s)
        b0 = (cx - .30 * s, cy + .26 * s)
        b1 = (cx + .30 * s, cy + .26 * s)
        b2 = (cx + .04 * s, cy + .12 * s)
        return [([b0, apex, b2], .95), ([b2, apex, b1], .68), ([b0, b1, b2], 1.10)]
    if geometry == "cylinder":
        w, top_y, bot_y = .26 * s, cy - .26 * s, cy + .26 * s
        return [([(cx - w, top_y), (cx + w, top_y), (cx + w, bot_y), (cx - w, bot_y)], .78),
                ([(cx - w, top_y - .06 * s), (cx + w, top_y - .06 * s),
                  (cx + w, top_y + .06 * s), (cx - w, top_y + .06 * s)], 1.15)]
    r = .28 * s
    return [([(cx - r, cy - r), (cx + r, cy - r), (cx + r, cy + r), (cx - r, cy + r)], .85),
            ([(cx - .18 * s, cy - .18 * s), (cx + .06 * s, cy - .18 * s),
              (cx + .06 * s, cy + .06 * s), (cx - .18 * s, cy + .06 * s)], 1.25)]


def goods_svg(goods, size=320, marks=None):
    parts = []
    for pts, tone in polygons(goods["geometry"], size):
        pts_str = " ".join("{0:.1f},{1:.1f}".format(x, y) for x, y in pts)
        parts.append('<polygon points="{0}" fill="{1}" stroke="#ffffff" stroke-width="1"/>'.format(
            pts_str, _hex(_shade(goods["rgb"], tone))))
    cx = cy = size / 2.0
    overlay = ""
    if marks and "污渍" in marks:
        overlay += ('<circle cx="{0:.0f}" cy="{1:.0f}" r="{2:.0f}" fill="#4a3b2a" opacity="0.55"/>'
                    '<circle cx="{3:.0f}" cy="{4:.0f}" r="{5:.0f}" fill="#5b4a33" opacity="0.45"/>').format(
            cx + size * .06, cy + size * .02, size * .07, cx - size * .05, cy - size * .04, size * .035)
    if marks and "缺陷" in marks:
        overlay += ('<path d="M {0:.0f} {1:.0f} l {2:.0f} {3:.0f} l {4:.0f} {5:.0f} z" fill="#1f2937" opacity="0.7"/>'
                    '<rect x="{6:.0f}" y="{7:.0f}" width="{8:.0f}" height="{9:.0f}" fill="#111827" opacity="0.6"/>').format(
            cx - size * .10, cy + size * .10, size * .06, -size * .07, size * .05, size * .05,
            cx - size * .12, cy - size * .13, size * .09, size * .05)
    return ('<svg xmlns="http://www.w3.org/2000/svg" width="{0}" height="{1}" viewBox="0 0 {0} {1}">'
            '<rect width="{0}" height="{1}" fill="#ffffff"/>{2}{3}</svg>').format(
                size, size, "".join(parts), overlay)


def generate_goods_images(img_dir, size=320):
    """生成货物图片：SVG 必生成；位图由 Pillow 生成（个别格式不支持时自动跳过）"""
    img_dir = str(img_dir)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir, 0o755)
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        Image = None
    for g in GOODS:
        svg_file = os.path.join(img_dir, "{0}.svg".format(g["id"]))
        if not os.path.exists(svg_file):
            with open(svg_file, "w", encoding="utf-8") as f:
                f.write(goods_svg(g, size))
        png_file = os.path.join(img_dir, "{0}.png".format(g["id"]))
        if Image is None or os.path.exists(png_file):
            continue
        img = Image.new("RGB", (size, size), (255, 255, 255))
        d = ImageDraw.Draw(img)
        for pts, tone in polygons(g["geometry"], size):
            d.polygon([(int(x), int(y)) for x, y in pts], fill=_shade(g["rgb"], tone))
        for ext, kw in (("png", {}), ("jpg", {"quality": 92}), ("webp", {"quality": 92}), ("bmp", {})):
            try:
                img.save(os.path.join(img_dir, "{0}.{1}".format(g["id"], ext)), **kw)
            except Exception as e:
                logger.warning("图片格式 {0} 生成失败: {1}".format(ext, e))
        try:
            img.convert("P", palette=Image.ADAPTIVE, colors=64).save(
                os.path.join(img_dir, "{0}.gif".format(g["id"])))
        except Exception as e:
            logger.warning("gif 生成失败: {0}".format(e))
    logger.info("货物图片就绪: {0} 类 · 格式 {1}".format(len(GOODS), "/".join(IMAGE_FORMATS)))


def formats_for(img_dir, goods_id):
    out = {}
    for fmt in IMAGE_FORMATS:
        if os.path.exists(os.path.join(str(img_dir), "{0}.{1}".format(goods_id, fmt))):
            out[fmt] = "/api/images/{0}.{1}".format(goods_id, fmt)
    return out
