# -*- coding: utf-8 -*-
"""Generate 4:3 landscape + 3:4 portrait covers per lesson.

Layout: cream paper bg -> badge + course tag -> title -> 16:9 showcase
frame card -> keyword bullets. Frames are pulled from the 1080p renders.
"""
import json
import os
import subprocess
import sys

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = os.path.dirname(os.path.abspath(__file__))
VDIR = os.path.join("media", "videos", "scenes", "1080p30")
OUTDIR = os.path.join(ROOT, "课程成片")
FRAMES = os.path.join(ROOT, "_build", "cover_frames")

BG = "#F7F1E3"
INK = "#1E2A45"
ACCENT = "#E8A000"
MUTED = "#5A6478"
NOTE = "#0B9BD8"

FONT_BOLD = r"C:\Windows\Fonts\msyhbd.ttc"
FONT_REG = r"C:\Windows\Fonts\msyh.ttc"
if not os.path.exists(FONT_BOLD):
    FONT_BOLD = FONT_REG = r"C:\Windows\Fonts\simhei.ttf"

META = {
    "L01": ("什么是机械", "绪论 · 构件 · 运动副 · 运动简图",
            "S02_EngineDissect", 0.65, ["机器三特征", "构件≠零件", "运动简图"]),
    "L02": ("机构能不能动", "自由度 · 三大陷阱 · 组成原理",
            "S07_BigExample", 0.6, ["F=3n−2P_L−P_H", "三大陷阱", "杆组原理"]),
    "L03": ("让速度看得见", "平面机构运动分析",
            "S04_ICVelocity", 0.6, ["瞬心法", "三心定理", "哥氏加速度"]),
    "L04": ("机器中的暗力量", "力分析 · 摩擦 · 效率 · 自锁",
            "S05_SelfLock", 0.6, ["摩擦圆", "机械效率", "自锁条件"]),
    "L05": ("让机器不抖", "机械的平衡",
            "S03_DynamicBalance", 0.55, ["静平衡", "动平衡", "平衡精度"]),
    "L06": ("机器的心跳", "运转 · 速度波动 · 飞轮",
            "S04_Flywheel", 0.55, ["运转三阶段", "盈亏功", "飞轮设计"]),
    "L07": ("四根杆的智慧 · 上", "连杆机构类型与特性",
            "S02_ThreeTypes", 0.6, ["格拉肖夫条件", "急回特性", "压力角·死点"]),
    "L08": ("四根杆的智慧 · 下", "四杆机构设计",
            "S05_CouplerAtlas", 0.7, ["位置设计", "按K设计", "连杆曲线图谱"]),
    "L09": ("按剧本运动的机器", "凸轮机构",
            "S04_InversionDraw", 0.75, ["推程四规律", "反转法作图", "压力角"]),
    "L10": ("最完美的曲线 · 上", "渐开线与标准齿轮",
            "S02_InvoluteBirth", 0.7, ["啮合定律", "渐开线", "重合度"]),
    "L11": ("最完美的曲线 · 下", "范成 · 根切 · 变位 · 空间齿轮",
            "S02_Generating", 0.7, ["范成法", "根切与变位", "斜齿·蜗杆"]),
    "L12": ("齿轮的组合艺术", "轮系",
            "S05_Differential", 0.6, ["定轴轮系", "转化机构法", "差速器"]),
    "L13": ("精巧的小机构", "棘轮 · 槽轮 · 万向节",
            "S02_Geneva", 0.55, ["棘轮", "槽轮", "万向节"]),
    "L14": ("让机器成为系统", "方案设计 + 收官",
            "S05_EngineReunion", 0.8, ["设计流程", "方案比选", "机构创新"]),
}


def dur(p):
    out = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_format", p],
        capture_output=True, text=True, encoding="utf-8", errors="replace")
    return float(json.loads(out.stdout)["format"]["duration"])


def grab_frame(lesson, scene, frac):
    v = os.path.join(ROOT, lesson, VDIR, scene + ".mp4")
    if not os.path.exists(v):
        v = os.path.join(ROOT, lesson, "media", "videos", "scenes",
                         "480p15", scene + ".mp4")
    png = os.path.join(FRAMES, f"{lesson}_{scene}.png")
    if not os.path.exists(png):
        subprocess.run(["ffmpeg", "-y", "-v", "error", "-ss",
                        f"{dur(v)*frac:.2f}", "-i", v, "-frames:v", "1", png])
    return Image.open(png).convert("RGB")


def rounded_card(img, w, radius=28, border=5):
    """Resize to width w (16:9), round corners, INK border + soft shadow."""
    h = int(w * img.height / img.width)
    img = img.resize((w, h), Image.LANCZOS)
    mask = Image.new("L", (w, h), 0)
    ImageDraw.Draw(mask).rounded_rectangle([0, 0, w, h], radius, fill=255)
    card = Image.new("RGB", (w, h))
    card.paste(img, (0, 0), mask)
    d = ImageDraw.Draw(card)
    d.rounded_rectangle([border // 2, border // 2, w - border // 2 - 1,
                         h - border // 2 - 1], radius, outline=INK,
                        width=border)
    return card


def draw_cover(lesson, W, H):
    title, sub, scene, frac, keys = META[lesson]
    cv = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(cv)
    m = int(W * 0.055)

    n = int(lesson[1:])
    f_badge = ImageFont.truetype(FONT_BOLD, int(H * 0.028))
    badge = f"第 {n} 讲"
    bw = d.textlength(badge, font=f_badge)
    pill_h = int(H * 0.052)
    d.rounded_rectangle([m, m, m + bw + pill_h * 1.1, m + pill_h],
                        pill_h // 2, fill=ACCENT)
    d.text((m + pill_h * 0.55, m + pill_h * 0.5), badge, font=f_badge,
           fill=INK, anchor="lm")
    f_tag = ImageFont.truetype(FONT_REG, int(H * 0.024))
    d.text((W - m, m + pill_h * 0.5), "机械原理 · 动画公益课",
           font=f_tag, fill=MUTED, anchor="rm")

    f_title = ImageFont.truetype(FONT_BOLD, int(H * 0.075))
    ty = m + pill_h + int(H * 0.035)
    d.text((m, ty), title, font=f_title, fill=INK)
    f_sub = ImageFont.truetype(FONT_REG, int(H * 0.030))
    ty += int(H * 0.075) + int(H * 0.022)
    d.text((m, ty), sub, font=f_sub, fill=NOTE)

    frame = rounded_card(grab_frame(lesson, scene, frac), W - 2 * m)
    fy = ty + int(H * 0.030) + int(H * 0.035)
    cv.paste(frame, (m, fy))

    f_key = ImageFont.truetype(FONT_REG, int(H * 0.028))
    ky = fy + frame.height + int(H * 0.045)
    x = m
    for k in keys:
        tw = d.textlength(k, font=f_key)
        box_w = tw + int(H * 0.05)
        d.rounded_rectangle([x, ky, x + box_w, ky + int(H * 0.052)],
                            int(H * 0.026), outline=INK, width=3)
        d.text((x + box_w / 2, ky + int(H * 0.026)), k, font=f_key,
               fill=INK, anchor="mm")
        x += box_w + int(H * 0.02)

    return cv


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    os.makedirs(FRAMES, exist_ok=True)
    for lesson in sorted(META):
        land = draw_cover(lesson, 1600, 1200)   # 4:3 横屏
        port = draw_cover(lesson, 1200, 1600)   # 3:4 竖屏
        p1 = os.path.join(OUTDIR, f"{lesson}_封面_横版4x3.png")
        p2 = os.path.join(OUTDIR, f"{lesson}_封面_竖版3x4.png")
        land.save(p1)
        port.save(p2)
        print(lesson, "covers saved", flush=True)


if __name__ == "__main__":
    main()
