# -*- coding: utf-8 -*-
"""mechlib.style — 版式、配色、中文字体、LessonScene 基类."""
from __future__ import annotations

import inspect
import os

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, Circumscribe, FadeIn, Indicate,
    Line, MathTex, MovingCameraScene, Restore, TexTemplate, Text, VGroup,
)

CJK = "Microsoft YaHei"          # Windows 自带中文字体（Pango 渲染）

# 中英混排公式模板：xelatex + ctex（pdflatex 无法渲染 \text{} 内的中文）
CJK_TEX_TEMPLATE = TexTemplate(
    tex_compiler="xelatex",
    output_format=".xdv",
    preamble=r"\usepackage[fontset=windows]{ctex}" "\n"
             r"\usepackage{amsmath,amssymb,bm}",
)


def mtex(*args, **kw) -> MathTex:
    """支持中文的 MathTex：所有含中文的公式请用 mtex() 而非 MathTex()。"""
    kw.setdefault("tex_template", CJK_TEX_TEMPLATE)
    return MathTex(*args, **kw)

# ---------------------------------------------------------------- 配色规范
# Kurzgesagt 风·浅色版：米白纸质底 + 降明度的高饱和色（浅底对比度优先）
BG = "#F7F1E3"                   # 米白纸质背景（场景底色）
INK = "#1E2A45"                  # 墨色：正文/主描边/销钉（浅底上的"白"）
ACCENT = "#E8A000"               # 强调：当前正在讲的元素/结论（琥珀黄）
GOOD = "#00A381"                 # 正确 / 成立 / 优点（深薄荷）
BAD = "#E0306B"                  # 错误 / 陷阱 / 冲击（品红）
NOTE = "#0B9BD8"                 # 注记 / 页码 / 辅助说明（深青）
MUTED = "#67769A"                # 次要信息（灰蓝）
LINK_A = "#E8A000"               # 原动件（曲柄）
LINK_B = "#0B9BD8"               # 连杆（青）
LINK_C = "#00A381"               # 摇杆/从动件（薄荷）
LINK_D = "#E0673F"               # 第二从动件/滑块（珊瑚橙）
FRAME_C = "#5A6478"              # 机架/固定件（深冷灰）
GEAR_1 = "#0B9BD8"
GEAR_2 = "#8B5CF6"               # 啮合对二轮（紫）
RULE_C = "#3A4670"               # 页首分隔线（深靛）


def ctext(s: str, size: int = 34, color=INK, weight: str = "NORMAL") -> Text:
    """中文文本（统一微软雅黑）。"""
    return Text(s, font=CJK, font_size=size, color=color, weight=weight)


def ctexts(*parts) -> VGroup:
    """多段异色中文一行排开：ctexts(("速度 ", INK), ("v", ACCENT), (" 指向瞬心", INK))"""
    row = VGroup(*[ctext(t, size=sz, color=c) for (t, c, sz) in
                   [(p[0], p[1] if len(p) > 1 else INK, p[2] if len(p) > 2 else 30)
                    for p in parts]])
    row.arrange(RIGHT, buff=0.08)
    return row


def title_bar(title: str, sub: str = "") -> VGroup:
    """页首标题条：主标题 + 可选小标题 + 通栏细线，置顶。"""
    t = ctext(title, size=40, weight="BOLD")
    g = VGroup(t)
    if sub:
        g.add(ctext(sub, size=23, color=MUTED))
    g.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UP + LEFT, buff=0.4)
    rule = Line(g.get_corner(DOWN + LEFT), g.get_corner(DOWN + LEFT) + RIGHT * 12.4,
                stroke_width=2, color=RULE_C).shift(DOWN * 0.14)
    return VGroup(g, rule)


def bullets(items, size: int = 30, buff: float = 0.3, marker: str = "▸ ",
            color=INK) -> VGroup:
    """竖排要点列表。"""
    rows = VGroup(*[ctext(marker + s, size=size, color=color) for s in items])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return rows


def page_ref(p: str) -> Text:
    """教材页码水印（正确性锚点），右下角。"""
    return ctext(p, size=19, color=NOTE).to_corner(DOWN + RIGHT, buff=0.28)


def chip(text: str, color=NOTE, size: int = 22) -> VGroup:
    """小节标签胶囊（如 §8-3）。"""
    from manim import RoundedRectangle
    t = ctext(text, size=size, color=color)
    box = RoundedRectangle(corner_radius=0.12, width=t.width + 0.4,
                           height=t.height + 0.22, color=color, stroke_width=1.5)
    return VGroup(box, t.move_to(box))


def glow(vmob, color=None, width: float | None = None) -> VGroup:
    """霓虹发光曲线：宽-中-窄三层描边叠出光晕。对生成曲线/轨迹很出片。"""
    c = color or vmob.get_color()
    w = width or vmob.get_stroke_width() or 3.0
    halo = vmob.copy().set_stroke(color=c, width=w * 4.5, opacity=0.16)
    mid = vmob.copy().set_stroke(color=c, width=w * 2.4, opacity=0.4)
    core = vmob.copy().set_stroke(color=c, width=w, opacity=1.0)
    return VGroup(halo, mid, core)


class LessonScene(MovingCameraScene):
    """课程场景基类：统一 header/hold/takeaway 节奏；docstring 即讲稿要点。

    3b1b 化增强：镜头推拉 zoom_to/zoom_reset、结论圈注脉冲、
    右上角常驻缓转齿轮 + 集数徽章（自动推导，场景零改动）。
    """

    lesson = ""     # 如 "L01"
    seg = ""        # 分镜名

    # ----------------------------------------------------------- 片头/徽章
    def setup(self):
        super().setup()
        self.camera.background_color = BG

    def header(self, title: str, sub: str = ""):
        self.camera.background_color = BG
        h = title_bar(title, sub)
        badge = self._episode_badge()
        self.play(FadeIn(h, shift=DOWN * 0.15), FadeIn(badge, shift=DOWN * 0.1),
                  run_time=0.6)
        if getattr(self.camera.frame, "saved_state", None) is None:
            self.camera.frame.save_state()
        return h

    def _episode_badge(self) -> VGroup:
        """右上角：缓转小齿轮 + 'L07 · S07' 集数水印（ambient 微动元素）。"""
        tag = type(self).__name__.split("_")[0]
        d = os.path.basename(os.path.dirname(inspect.getfile(type(self))))
        les = d.upper() if d.upper().startswith("L") else tag
        txt = ctext(f"{les} · {tag}", size=17, color=MUTED)
        txt.to_corner(UP + RIGHT, buff=0.42)
        try:
            from .curves import gear_profile
            icon = gear_profile(0.045, 8, color=MUTED, stroke_width=1.4)
            icon.scale_to_width(0.3)
        except Exception:
            from manim import Circle
            icon = Circle(radius=0.15, color=MUTED, stroke_width=1.5)
        icon.next_to(txt, LEFT, buff=0.18)
        icon.add_updater(lambda m, dt: m.rotate(-0.22 * dt,
                                                about_point=m.get_center()))
        return VGroup(txt, icon)

    # ----------------------------------------------------------- 镜头语言
    def zoom_to(self, target, scale: float = 0.45, run_time: float = 1.4):
        """推近镜头到 target（mobject 或点），scale=画面高度比例。"""
        c = target.get_center() if hasattr(target, "get_center") \
            else np.array(target, dtype=float)
        if getattr(self.camera.frame, "saved_state", None) is None:
            self.camera.frame.save_state()
        self._zoomed = True
        self.play(self.camera.frame.animate.scale(scale).move_to(c),
                  run_time=run_time)

    def zoom_reset(self, run_time: float = 1.3):
        """镜头复位到全景。"""
        self._zoomed = False
        self.play(Restore(self.camera.frame), run_time=run_time)

    # ----------------------------------------------------------- 强调三件套
    def emphasize(self, mob, color=ACCENT, run_time: float = 0.9):
        """脉冲指示：元素闪色+微放大，引导视线。"""
        self.play(Indicate(mob, color=color, scale_factor=1.15),
                  run_time=run_time)

    def focus(self, mob, color=ACCENT, run_time: float = 1.1):
        """圈注一闪：在元素周围画一个消隐的框（3b1b 式高亮）。"""
        self.play(Circumscribe(mob, color=color, fade_out=True,
                               stroke_width=3), run_time=run_time)

    # ----------------------------------------------------------- 节奏
    def hold(self, t: float = 1.5):
        """讲解留白（配音空间）。"""
        self.wait(t)

    def takeaway(self, text: str, p: str = "", size: int = 30):
        """底部结论条：入场 + 圈注脉冲；若在放大镜头中先自动复位。"""
        if getattr(self, "_zoomed", False):
            self.zoom_reset()
        t = ctext(text, size=size, color=INK).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(t, shift=UP * 0.2))
        self.focus(t, color=ACCENT)
        if p:
            self.add(page_ref(p))
        return t
