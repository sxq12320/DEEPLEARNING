# -*- coding: utf-8 -*-
"""Compose 4:3 and 3:4 covers from title-card stills out/title_NNN.png."""
import os, re, sys
from PIL import Image, ImageDraw, ImageFont

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(BASE, 'out')
FONT = r'C:\Windows\Fonts\simyou.ttf'

def title_of(n):
    p = os.path.join(BASE, 'episodes', str(n), 'script.md')
    with open(p, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line.startswith('#'):
                return line.lstrip('#').strip()
    return '第%d期' % n

def bg_of(img):
    w, h = img.size
    px = img.convert('RGB')
    pts = [px.getpixel((x, y)) for x in (4, w - 5) for y in (4, h - 5)]
    r = sum(p[0] for p in pts) // len(pts)
    g = sum(p[1] for p in pts) // len(pts)
    b = sum(p[2] for p in pts) // len(pts)
    return (r, g, b)

def text_w(draw, s, font):
    return draw.textbbox((0, 0), s, font=font)[2]

def cover_43(still, bg, n, title):
    W, H = 1440, 1080
    c = Image.new('RGB', (W, H), bg)
    s = still.resize((W, int(W * still.size[1] / still.size[0])), Image.LANCZOS)
    c.paste(s, (0, (H - s.size[1]) // 2))
    return c

def cover_34(still, bg, n, title):
    W, H = 1080, 1440
    c = Image.new('RGB', (W, H), bg)
    d = ImageDraw.Draw(c)
    f_badge = ImageFont.truetype(FONT, 42)
    f_title = ImageFont.truetype(FONT, 96)
    f_small = ImageFont.truetype(FONT, 38)
    # badge pill top-center
    badge = '冷知识 · 第%d期' % n
    bw = text_w(d, badge, f_badge)
    d.rounded_rectangle([(W // 2 - bw // 2 - 30, 90), (W // 2 + bw // 2 + 30, 160)],
                        radius=35, outline=(255, 255, 255), width=3)
    d.text((W // 2 - bw // 2, 105), badge, font=f_badge, fill=(255, 255, 255))
    # title, wrap to <=2 lines
    if text_w(d, title, f_title) <= W - 120:
        lines = [title]
    else:
        mid = len(title) // 2
        best = min(range(len(title)),
                   key=lambda i: abs(text_w(d, title[:i], f_title) - text_w(d, title[i:], f_title)))
        lines = [title[:best], title[best:]]
    y = 200
    for ln in lines:
        lw = text_w(d, ln, f_title)
        d.text(((W - lw) // 2, y), ln, font=f_title, fill=(255, 255, 255))
        y += 118
    # still centered in remaining space
    s = still.resize((W, int(W * still.size[1] / still.size[0])), Image.LANCZOS)
    y_still = 380 + (1440 - 380 - s.size[1]) // 2
    c.paste(s, (0, y_still))
    tag = '每期一个冷知识'
    tw = text_w(d, tag, f_small)
    d.text(((W - tw) // 2, H - 110), tag, font=f_small, fill=(200, 205, 220))
    return c

def main():
    for n in range(201, 251):
        p = os.path.join(OUT, 'title_%d.png' % n)
        if not os.path.exists(p):
            print('skip %d (no still)' % n); continue
        still = Image.open(p)
        bg = bg_of(still)
        t = title_of(n)
        cover_43(still, bg, n, t).save(os.path.join(OUT, 'cover_%d_4x3.png' % n))
        cover_34(still, bg, n, t).save(os.path.join(OUT, 'cover_%d_3x4.png' % n))
        print('done %d' % n)

if __name__ == '__main__':
    main()
