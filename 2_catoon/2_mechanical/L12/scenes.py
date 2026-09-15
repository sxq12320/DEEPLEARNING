# -*- coding: utf-8 -*-
"""L12 齿轮的组合艺术——轮系及其传动比（第11章, p237-260）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_TrainTypes(LessonScene):
    """轮系分类（~6min, p237-239）：定轴轮系/周转轮系（行星+差动）/复合——
    三示意并排，转臂 H 是否旋转是分水岭。"""

    def construct(self):
        self.header("齿轮成排：轮系", "分类 · p237-239")
        cards = VGroup()
        for t, d in [("定轴轮系", "所有轴线固定不动"),
                     ("周转轮系", "有轴线绕别的轴转（转臂H）"),
                     ("复合轮系", "定轴 + 周转 的混合")]:
            box = RoundedRectangle(corner_radius=0.15, width=3.8, height=1.5,
                                   color=LINK_B, fill_opacity=0.12,
                                   fill_color=LINK_B)
            cards.add(VGroup(box, ctext(t, size=27, weight="BOLD",
                                        color=ACCENT).move_to(
                                            box.get_top() + DOWN * 0.35),
                             ctext(d, size=21).move_to(
                                 box.get_center() + DOWN * 0.25)))
        cards.arrange(RIGHT, buff=0.55).shift(UP * 0.5)
        self.play(FadeIn(cards, lag_ratio=0.4), run_time=2.5)
        q = ctext("分水岭：转臂 H 转不转？——转 → 不能直接套定轴公式",
                  size=27, color=ACCENT).to_edge(DOWN, buff=1.0)
        self.play(Write(q))
        self.add(page_ref("孙桓八版 p237-239"))
        self.hold(3)


class S02_FixedAxis(LessonScene):
    """定轴轮系（~10min, p239-241）：i = ±(从动轮齿数积)/(主动轮齿数积)；
    外啮合次数定正负；惰轮只改方向不改大小——三对齿轮链动画。"""

    def construct(self):
        self.header("定轴轮系传动比", "p239-241")
        z = [14, 22, 30]
        m = 0.11
        c1 = P(-4.9, 0.2)
        a12 = m * (z[0] + z[1]) / 2
        a23 = m * (z[1] + z[2]) / 2
        c2 = c1 + P(a12 * np.cos(np.deg2rad(25)), a12 * np.sin(np.deg2rad(25)))
        c3 = c2 + P(a23 * np.cos(np.deg2rad(-18)),
                    a23 * np.sin(np.deg2rad(-18)))
        g1 = gear_profile(m, z[0], color=GEAR_1).move_to(c1)
        g2 = gear_profile(m, z[1], color=GEAR_2).move_to(c2)
        g3 = gear_profile(m, z[2], color=LINK_D).move_to(c3)
        th = ValueTracker(0)
        g1a = always_redraw(lambda: gear_profile(
            m, z[0], color=GEAR_1).rotate(
                th.get_value(), about_point=ORIGIN).move_to(c1))
        g2a = always_redraw(lambda: gear_profile(
            m, z[1], color=GEAR_2).rotate(
                -th.get_value() * z[0] / z[1], about_point=ORIGIN).move_to(c2))
        g3a = always_redraw(lambda: gear_profile(
            m, z[2], color=LINK_D).rotate(
                th.get_value() * z[0] / z[2], about_point=ORIGIN).move_to(c3))
        self.play(FadeIn(VGroup(g1a, g2a, g3a)),
                  FadeIn(VGroup(fixed_pin(c1), fixed_pin(c2), fixed_pin(c3))))
        self.play(th.animate.set_value(2 * TAU), run_time=5,
                  rate_func=linear)
        self.hold(1)
        steps = [
            mtex(r"i_{13}=\frac{\omega_1}{\omega_3}=(-1)^k"
                    r"\frac{z_3}{z_1}\cdot(\text{惰轮只换向})",
                    font_size=44, color=ACCENT),
            mtex(r"\text{箭头法画转向；外啮合 }k\text{ 次定 }(-1)^k",
                    font_size=36),
        ]
        formula_reveal(self, steps, anchor=DOWN * 2.4, wait=1.8)
        self.add(page_ref("孙桓八版 p239-241"))
        self.hold(3)


class S03_InversionMethod(LessonScene):
    """转化机构法（~12min, p240-243 核心）：给整个周转轮系加 −ω_H——
    '摄像机随转臂转'动画：转臂定格、行星变定轴、套定轴公式。"""

    def construct(self):
        self.header("转化机构法：给系统'踩刹车'", "−ωH 反转 · p240-243")
        pl = Planetary(z1=20, z2=10)
        m = AnimatedPlanetary(pl, m=0.075, origin=np.array([-3.2, -0.5, 0]))
        self.play(FadeIn(m.group))
        t1 = ctext("原系统：转臂 H 转 + 太阳轮转 + 齿圈固定", size=25,
                   color=INK).to_edge(UP, buff=1.7)
        self.play(Write(t1))
        # 状态1：n1=1, n3=0 → nH
        sp = pl.speeds(n1=1.0, n3=0.0)
        self.play(m.t1.animate.set_value(TAU),
                  m.tH.animate.set_value(TAU * sp["nH"]),
                  run_time=5, rate_func=linear)
        self.hold(1)
        # 反转视角：全体减 ωH → H 定住
        t2 = ctext("转化视角：全体 −ωH → 转臂'定住'，变定轴轮系！",
                   size=25, color=ACCENT).to_edge(UP, buff=1.7)
        self.play(Transform(t1, t2))
        self.emphasize(m.group)
        self.play(m.t1.animate.set_value(m.t1.get_value() + TAU *
                                         (sp["n1"] - sp["nH"])),
                  m.tH.animate.set_value(m.tH.get_value()),  # H 停
                  run_time=5, rate_func=linear)
        steps = [
            mtex(r"\frac{n_1-n_H}{n_3-n_H}=-\frac{z_3}{z_1}",
                    font_size=50, color=ACCENT),
            mtex(r"\text{注意：转化机构中首末轮转向看正负号}",
                    font_size=36),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 2.9 + DOWN * 0.6,
                       wait=1.8)
        self.add(page_ref("孙桓八版 p240-243"))
        self.hold(3)


class S04_BigRatio(LessonScene):
    """大传动比算例（~8min, p243-246）：行星轮系小机构大减速——
    数字现场代入公式。"""

    def construct(self):
        self.header("小小行星，大大传动比", "算例 · p243-246")
        pl = Planetary(z1=60, z2=20)     # z3=100
        sp = pl.speeds(n1=1.0, n3=0.0)
        eq1 = mtex(r"\frac{n_1-n_H}{0-n_H}=-\frac{100}{60}",
                      font_size=44)
        eq2 = mtex(r"1-\frac{n_1}{n_H}=-\frac{5}{3}\ \Rightarrow\ "
                      r"i_{H1}=\frac{n_H}{n_1}=\frac{3}{8}",
                      font_size=46, color=ACCENT)
        VGroup(eq1, eq2).arrange(DOWN, buff=0.6).shift(UP * 1.15 + LEFT * 1.4)
        self.play(Write(eq1))
        self.hold(1.5)
        self.play(Write(eq2))
        self.hold(2)
        m = AnimatedPlanetary(pl, m=0.036,
                              origin=np.array([2.9, -1.7, 0]))
        self.play(FadeIn(m.group))
        self.play(m.t1.animate.set_value(TAU * 2),
                  m.tH.animate.set_value(TAU * 2 * sp["nH"]),
                  run_time=5, rate_func=linear)
        lab = ctext("太阳轮转 8 圈，转臂才转 3 圈", size=25,
                    color=ACCENT).to_edge(DOWN, buff=0.4).shift(LEFT * 1.8)
        self.play(Write(lab))
        self.add(page_ref("孙桓八版 p243-246"))
        self.hold(3)


class S05_Differential(LessonScene):
    """汽车差速器（~10min, p246-248）：n_L+n_R=2n_H——直行两轮同速、
    转弯内慢外快；两工况同屏动画。"""

    def construct(self):
        self.header("差速器：转弯不磨胎的秘密", "p246-248")
        # 简化示意：转臂输入 + 左右半轴
        box = RoundedRectangle(corner_radius=0.15, width=4.6, height=2.2,
                               color=LINK_B, fill_opacity=0.08,
                               fill_color=LINK_B)
        box.shift(LEFT * 3.4 + UP * 0.2)
        lab = ctext("差速器", size=24).next_to(box, UP, buff=0.2)
        wl = Dot(box.get_left() + LEFT * 0.8, radius=0.2, color=LINK_C)
        wr = Dot(box.get_right() + RIGHT * 0.8, radius=0.2, color=LINK_C)
        lbar = link_line(box.get_left(), wl.get_center(), FRAME_C, 5)
        rbar = link_line(box.get_right(), wr.get_center(), FRAME_C, 5)
        self.play(FadeIn(VGroup(box, lab, wl, wr, lbar, rbar)))
        t1 = ctext("直行：nL = nR = nH（行星轮不转）", size=25,
                   color=GOOD).to_edge(UP, buff=1.7)
        self.play(Write(t1))
        self.hold(2)
        t2 = ctext("转弯：nL + nR = 2nH —— 内轮慢、外轮快，差多少分多少",
                   size=25, color=ACCENT).to_edge(UP, buff=1.7)
        self.play(Transform(t1, t2))
        eq = mtex(r"n_L + n_R = 2\,n_H", font_size=52,
                     color=ACCENT).shift(RIGHT * 3.2 + UP * 0.2)
        self.play(Write(eq))
        # 转弯时轮速对比：内轮小弧、外轮大弧
        al = Arrow(wl.get_center() + UP * 0.38, wl.get_center() + RIGHT * 0.38,
                   path_arc=-PI / 2, buff=0, color=BAD, stroke_width=5)
        ar = Arrow(wr.get_center() + UP * 0.62, wr.get_center() + RIGHT * 0.62,
                   path_arc=-PI / 2, buff=0, color=GOOD, stroke_width=5)
        self.play(Create(al), Create(ar))
        self.play(Write(ctext("内轮小弧慢 / 外轮大弧快", size=22,
                              color=NOTE).next_to(box, DOWN, buff=0.35)))
        self.add(page_ref("孙桓八版 p246-248"))
        self.hold(3)


class S06_CompoundTrain(LessonScene):
    """复合轮系（~8min, p248-252）：先拆（找周转：轴线动的齿轮+其转臂）→
    各列方程 → 联立求解——拆分流程动画。"""

    def construct(self):
        self.header("复合轮系：先拆再算", "p248-252")
        steps = bullets([
            "① 找'轴线会动'的齿轮 → 圈出周转轮系",
            "② 剩下的全是定轴轮系",
            "③ 周转部分列转化机构方程、定轴部分列传动比方程",
            "④ 公共转速联立求解",
        ], size=28).shift(LEFT * 2.6 + UP * 0.5)
        self.play(FadeIn(steps, lag_ratio=0.5), run_time=2.8)
        warn = ctext("最常见的错：把周转部分也当定轴算", size=26,
                     color=BAD).to_edge(DOWN, buff=0.9)
        self.play(Write(warn))
        self.add(page_ref("孙桓八版 p248-252"))
        self.hold(3)


class S07_TrainFunctions(LessonScene):
    """轮系六大功用（~5min, p252-255）：变速/换向/分路/合成/大传动比/远距
    传动——图标矩阵。"""

    def construct(self):
        self.header("为什么要用轮系？", "六大功用 · p252-255")
        items = ["变速", "换向", "分路传动", "运动合成/分解", "大传动比",
                 "远距离传动"]
        cards = VGroup(*[
            VGroup(RoundedRectangle(corner_radius=0.14, width=3.4, height=1.1,
                                    color=LINK_B, fill_opacity=0.12,
                                    fill_color=LINK_B),
                   ctext(s, size=25).move_to(
                       RoundedRectangle(width=3.4, height=1.1).get_center()))
            for s in items])
        for i, c in enumerate(cards):
            c[1].move_to(c[0])
        cards.arrange_in_grid(rows=2, cols=3, buff=0.55).shift(UP * 0.3)
        self.play(FadeIn(cards, lag_ratio=0.3), run_time=2.5)
        self.hold(2.5)
        self.add(page_ref("孙桓八版 p252-255"))
        self.hold(2)


class S08_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 12 讲小结")
        pts = bullets([
            "定轴轮系：i=±∏从动/∏主动，外啮合定号      (p239-241)",
            "周转轮系：全体 −ωH → 转化机构按定轴算    (p240-243)",
            "行星轮系轻松实现大传动比                 (p243-246)",
            "差速器 nL+nR=2nH：直行同速/转弯差速    (p246-248)",
            "复合轮系：先拆周转再列方程联立            (p248-252)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：不做匀速运动的机构们——间歇机构与其他",
                  size=25, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
