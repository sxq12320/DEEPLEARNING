# -*- coding: utf-8 -*-
"""L12 番外：那些精巧的小机构——间歇运动机构（孙桓八版 第12章, p261-287）

目标成片 40-50 min（轻松完结篇，演示多推导少；可拆条发短视频引流）。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（2min）：连续转动世界里的'走走停停'——间歇运动机构合集。"""

    def construct(self):
        big = ctext("番外   精巧的小机构", size=60, weight="BOLD")
        sub = ctext("棘轮 · 槽轮 · 不完全齿轮 —— 走走停停的艺术", size=30, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(3)


class S02_Ratchet(LessonScene):
    """棘轮机构（10min，p261-264）。
    讲稿要点：摆杆带棘爪推棘轮单向转，止回爪防倒转；转角可调（遮板）；
    应用：手动葫芦、卷扬制动、扳手；噪声与冲击是短板 → 摩擦式棘轮。"""

    def construct(self):
        self.header("棘轮机构", "只进不退的单行道 · p261-264")
        ctr = np.array([-2.5, -0.4, 0])
        n_teeth = 12
        pts = []
        for k in range(n_teeth):
            a0 = k * TAU / n_teeth
            a1 = a0 + TAU / n_teeth * 0.72
            pts.append(ctr + 1.3 * np.array([np.cos(a0), np.sin(a0), 0]))
            pts.append(ctr + 1.62 * np.array([np.cos(a1), np.sin(a1), 0]))
        wheel = VMobject(color=TEAL, stroke_width=4)
        wheel.set_points_as_corners([*pts, pts[0]])
        pawl_pivot = ctr + np.array([0.75, 2.05, 0])
        self.play(Create(wheel), FadeIn(pin_joint(ctr, 0.1)))
        thv = ValueTracker(0.0)

        def pawl():
            # 摆动棘爪：往复摆动，推程贴轮
            sw = 0.35 * np.sin(thv.get_value())
            tip = ctr + 1.62 * np.array([np.cos(PI / 2 + 0.25 - max(sw, 0)), np.sin(PI / 2 + 0.25 - max(sw, 0)), 0])
            return VGroup(link_line(pawl_pivot, tip, ORANGE, w=5), pin_joint(pawl_pivot, 0.07),
                          Dot(tip, color=ORANGE, radius=0.06))

        self.play(FadeIn(always_redraw(pawl)))
        t1 = ctext("摆杆往复：推程咬齿带轮走一格，回程滑过齿背——单向间歇", size=27).to_edge(DOWN, buff=1.2)
        self.play(Write(t1))
        for k in range(3):
            self.play(thv.animate.set_value((k + 0.5) * PI), Rotate(wheel, -TAU / n_teeth, about_point=ctr),
                      run_time=1.2, rate_func=smooth)
            self.play(thv.animate.set_value((k + 1) * PI), run_time=0.9)
        apps = ctext("应用：手拉葫芦 · 卷扬止逆 · 棘轮扳手 · 自行车飞轮（摩擦式）", size=27,
                     color=ACCENT).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(t1, apps))
        self.add(page_ref("孙桓八版 p261-264"))
        self.hold(3)


class S03_Geneva(LessonScene):
    """★槽轮机构与运动系数（12min，p264-268，番外唯一推导）。
    讲稿要点：拨销进槽带槽轮转位、锁止弧定位；进出槽切向条件 → 2φ1 = π − 2π/z；
    运动系数 τ = 运动时间/周期 = (z−2)/(2z) < 1/2 → 单销槽轮永远'动少停多'；
    z≥3；多销可增 τ。应用：电影放映机(z=4, 1/4 周期走片)、转位工作台。"""

    def construct(self):
        self.header("槽轮机构（马耳他机构）", "运动系数 τ · p264-268")
        z = 4
        ctr_g = np.array([1.8, -0.3, 0]); ctr_d = ctr_g + np.array([-2.35, 0, 0])
        # 槽轮外形（示意：圆盘+径向槽）
        disk = Circle(radius=1.5, color=TEAL, stroke_width=4).move_to(ctr_g)
        slots = VGroup(*[Line(ctr_g + 0.45 * np.array([np.cos(a), np.sin(a), 0]),
                              ctr_g + 1.5 * np.array([np.cos(a), np.sin(a), 0]),
                              color=TEAL, stroke_width=6)
                         for a in np.linspace(0, TAU, z, endpoint=False)])
        driver = Circle(radius=0.9, color=GOLD, stroke_width=4).move_to(ctr_d)
        pin = Dot(ctr_d + np.array([0.9, 0, 0]), color=RED, radius=0.1)
        self.play(Create(disk), Create(slots), Create(driver), FadeIn(pin))
        t1 = ctext("拨盘连续转：销进槽 → 槽轮转 90° → 销出槽 → 锁止弧锁住不动", size=26).to_edge(DOWN, buff=1.25)
        self.play(Write(t1))
        grp_d = VGroup(driver, pin)
        for k in range(2):
            self.play(Rotate(grp_d, PI / 2, about_point=ctr_d),
                      Rotate(VGroup(disk, slots), -PI / 2, about_point=ctr_g),
                      run_time=1.4, rate_func=smooth)
            self.play(Rotate(grp_d, PI * 3 / 2, about_point=ctr_d), run_time=2.0, rate_func=linear)
        f = [
            MathTex(r"\text{进/出槽无冲击: 销速度沿槽向} \Rightarrow 2\varphi_1 = \pi - \frac{2\pi}{z}", font_size=40),
            MathTex(r"\tau = \frac{t_\text{动}}{t_\text{周}} = \frac{2\varphi_1}{2\pi} = \frac{z-2}{2z}",
                    font_size=46, color=YELLOW),
            MathTex(r"z=4:\ \tau=\tfrac14;\quad \tau<\tfrac12\ \text{恒成立（动少停多）};\ z\ge 3", font_size=38, color=GREEN),
        ]
        self.play(FadeOut(t1))
        formula_reveal(self, f, anchor=DOWN * 2.3, buff=0.3, wait=1.9)
        self.add(page_ref("孙桓八版 p264-268"))
        self.hold(2.5)


class S04_OtherMechs(LessonScene):
    """不完全齿轮 · 凸轮式间歇 · 螺旋机构（10min，p268-274）。
    讲稿要点：不完全齿轮——想停多久停多久（缺齿段），进入冲击需瞬心线附加杆；
    凸轮式间歇（分度凸轮）——高速分度王者；螺旋机构——转动变移动，微调与自锁两开花。"""

    def construct(self):
        self.header("间歇家族其他成员", "p268-274")
        pts = bullets([
            "不完全齿轮：齿留一段删一段——停歇比例任意定制（计数器、转位）",
            "凸轮式间歇（弧面分度凸轮）：预紧无隙、高速高精度——机床刀塔",
            "螺旋机构：Δs = L·Δφ/2π——千分尺的微调、台钳的自锁",
            "组合机构：串并联混出新特性（齿轮-连杆、凸轮-连杆…创新孵化器）",
        ], size=29)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=3)
        self.add(page_ref("孙桓八版 p268-274, 282"))
        self.hold(4)


class S05_SeriesEnd(LessonScene):
    """全系列彩蛋结语（3min）：课程知识地图全家福 + 开源信息。"""

    def construct(self):
        t = ctext("机械原理 · 全系列完", size=64, weight="BOLD")
        s = ctext("课程动画代码全部开源 · 欢迎二创与纠错", size=30, color=GREY_B)
        VGroup(t, s).arrange(DOWN, buff=0.6)
        self.play(Write(t), FadeIn(s, shift=UP * 0.3))
        self.hold(4)
