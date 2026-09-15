# -*- coding: utf-8 -*-
"""L05 让机器不抖——机械的平衡（第6章, p85-94）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Unbalance(LessonScene):
    """不平衡的破坏力（~6min, p85-86）：偏心转子离心力 F=m·e·ω² 随转角扫掠
    → 支座振动演示（弹簧底座上下抖）。"""

    def construct(self):
        self.header("为什么会抖？", "不平衡力 · p85-86")
        O = P(-2.2, 0.6)
        th = ValueTracker(0.0)

        def rotor():
            a = th.get_value()
            m_pos = O + np.array([np.cos(a), np.sin(a), 0]) * 0.55
            base_y = -1.0 + 0.08 * np.cos(a)          # 支座微振
            return VGroup(
                Circle(radius=0.85, color=LINK_B, stroke_width=5).move_to(O),
                Dot(O, radius=0.06),
                Dot(m_pos, radius=0.16, color=LINK_D),     # 偏心质量
                dashed(O, m_pos, color=LINK_D),
                # 弹簧支座
                spring(P(-3.0, -0.25), P(-3.0, base_y), coils=6),
                spring(P(-1.4, -0.25), P(-1.4, base_y), coils=6),
                Line(P(-3.4, -0.25), P(-1.0, -0.25), color=FRAME_C,
                     stroke_width=4),
            )
        mech = always_redraw(rotor)
        f_arrow = always_redraw(lambda: vec(
            O, (np.array([np.cos(th.get_value()), np.sin(th.get_value()), 0])
                * 1.6), BAD))
        self.play(FadeIn(mech), FadeIn(f_arrow))
        self.play(th.animate.set_value(4 * TAU), run_time=6, rate_func=linear)
        self.hold(1)
        self.play(Write(mtex(r"F = m\,e\,\omega^2", font_size=52,
                                color=BAD).to_edge(DOWN, buff=1.2)))
        self.takeaway("不平衡质量 → 离心力旋转扫掠 → 整机振动",
                        p="孙桓八版 p85-86")
        self.hold(3)


class S02_StaticBalance(LessonScene):
    """静平衡（~8min, p87-88）：同平面偏心质量——加配重使质径积矢量和为零。
    矢量作图：m_i·r_i 多边形闭合。"""

    def construct(self):
        self.header("静平衡：一个平面内配平", "质径积法 · p87-88")
        O = P(-3.2, 0.3)
        disk = Circle(radius=1.5, color=LINK_B, stroke_width=5).move_to(O)
        ms = [np.array([0.9, 0.5, 0]), np.array([-0.8, 0.7, 0]),
              np.array([-0.2, -1.0, 0])]
        dots = VGroup(*[Dot(O + m, radius=0.13, color=LINK_D) for m in ms])
        self.play(Create(disk), FadeIn(dots), FadeIn(Dot(O, radius=0.06)))
        self.hold(1)
        # 质径积矢量图（右侧）
        ori = P(2.4, -0.6)
        vecs = [np.array([m[0], m[1], 0]) * 0.85 for m in ms]
        acc = ori.copy()
        polys = VGroup()
        for v in vecs:
            a = Arrow(acc, acc + v, buff=0, color=LINK_D, tip_length=0.14,
                      stroke_width=4)
            polys.add(a)
            acc = acc + v
        cb = acc - ori                      # 配重矢量 = 闭合边
        close = Arrow(acc, ori, buff=0, color=GOOD, tip_length=0.16,
                      stroke_width=5)
        self.play(FadeIn(polys, lag_ratio=0.5), run_time=2.5)
        self.play(FadeIn(close))
        self.play(Write(ctext("补上闭合边 mb·rb → 质径积和为零", size=25,
                              color=GOOD).next_to(polys, UP, buff=0.4)))
        self.hold(2)
        # 圆盘上加配重（方向 = −Σm·r，即闭合边箭头指向）
        cb2 = np.array([cb[0], cb[1], 0])
        bal = Dot(O - cb2 / np.linalg.norm(cb2) * 1.05, radius=0.15,
                  color=GOOD)
        self.play(FadeIn(bal))
        self.emphasize(bal, color=GOOD)
        self.play(Write(mtex(r"\sum m_i\vec{r}_i = 0", font_size=46,
                                color=ACCENT).to_edge(DOWN, buff=0.6)))
        self.add(page_ref("孙桓八版 p87-88"))
        self.hold(3)


class S03_DynamicBalance(LessonScene):
    """动平衡（~10min, p88-91）：轴向两偏心质量——力平衡了但力偶还在（翻转
    演示）；须在两个校正平面配重：力矢量和=0 且力偶矢量和=0。"""

    def construct(self):
        self.header("动平衡：两个平面才够", "力 + 力偶 · p88-91")
        # 转子轴 + 两平面偏心质量（侧视示意）
        shaft = Line(P(-4.5, 0), P(1.5, 0), color=FRAME_C, stroke_width=6)
        p1 = Circle(radius=0.55, color=LINK_B).move_to(P(-3.4, 0))
        p2 = Circle(radius=0.55, color=LINK_B).move_to(P(0.4, 0))
        m1 = Dot(P(-3.4, 0.7), radius=0.13, color=BAD)
        m2 = Dot(P(0.4, -0.7), radius=0.13, color=BAD)
        self.play(FadeIn(VGroup(shaft, p1, p2, m1, m2)))
        f1 = vec(P(-3.4, 0.7), UP * 0.8, BAD)
        f2 = vec(P(0.4, -0.7), DOWN * 0.8, BAD)
        self.play(FadeIn(f1), FadeIn(f2))
        self.hold(1.5)
        t1 = ctext("F1 与 F2 等大反向 → 力平衡了…但形成力偶，轴仍要扭！",
                   size=26, color=BAD).to_edge(UP, buff=1.7)
        self.play(Write(t1))
        self.hold(2)
        steps = [
            mtex(r"\text{动平衡条件：}\ \sum \vec{F}_i = 0"
                    r"\ \ \text{且}\ \ \sum \vec{M}_i = 0", font_size=44,
                    color=ACCENT),
            mtex(r"\text{须在 } \ge 2\text{ 个校正平面内配重}",
                    font_size=40),
        ]
        formula_reveal(self, steps, anchor=DOWN * 1.9, wait=1.8)
        self.add(page_ref("孙桓八版 p88-91"))
        self.hold(3)


class S04_BalanceQuality(LessonScene):
    """平衡精度（~5min, p91-93）：许用不平衡量 [e]·ω；平衡机概念；
    静平衡实验（导轨滚动找重点）小动画。"""

    def construct(self):
        self.header("平衡到什么程度才够？", "许用不平衡量 · p91-93")
        pts = bullets([
            "判据：质心偏心距 e 与角速度 ω 的乘积 [eω]（平衡精度等级）",
            "静平衡实验：转子放平行导轨，重点滚到最低——反向加配重",
            "动平衡机：测两支承振动反推两平面所需质径积",
        ], size=28).shift(UP * 0.4)
        self.play(FadeIn(pts, lag_ratio=0.5), run_time=2.5)
        self.hold(2.5)
        # 导轨静平衡小动画
        rail = VGroup(Line(P(-2.5, -2.4), P(0.5, -2.4), color=FRAME_C,
                           stroke_width=5),
                      Line(P(-2.5, -2.1), P(-2.5, -2.4), color=FRAME_C),
                      Line(P(0.5, -2.1), P(0.5, -2.4), color=FRAME_C))
        disk = Circle(radius=0.7, color=LINK_B, stroke_width=5).move_to(
            P(-1.0, -1.68))
        heavy = Dot(P(-1.0, -2.15), radius=0.12, color=LINK_D)
        self.play(FadeIn(rail), FadeIn(disk), FadeIn(heavy))
        self.play(Rotate(VGroup(disk, heavy), PI / 3,
                         about_point=disk.get_center()),
                  Rotate(VGroup(disk, heavy), -PI / 3,
                         about_point=disk.get_center()),
                  run_time=2, rate_func=there_and_back)
        note = ctext("重点总滚到最低 → 对侧加配重", size=24,
                     color=NOTE).next_to(rail, DOWN, buff=0.5)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p91-93"))
        self.hold(3)


class S05_MechBalance(LessonScene):
    """平面机构平衡一瞥（~5min, p93-94）：连杆机构质心随运动漂移——
    质量代换到铰链点+对称布置/配重平衡。"""

    def construct(self):
        self.header("连杆机构的平衡", "质量代换 · p93-94")
        fb = FourBar(3.6, 1.0, 2.8, 2.4, origin=np.array([-3.4, -1.2, 0]))
        m = AnimatedFourBar(fb)
        cm_dot = always_redraw(lambda: Dot(
            fb.coupler_point(m.theta.get_value(), 0.5), radius=0.1,
            color=BAD))
        self.play(FadeIn(m.group), FadeIn(cm_dot))
        tr = TracedPath(cm_dot.get_center, stroke_color=BAD, stroke_width=2)
        self.add(tr)
        self.play(m.theta.animate.set_value(0.6 + TAU), run_time=5,
                  rate_func=linear)
        self.hold(1)
        pts = bullets([
            "机构总质心随机架振动 → 基座受摇摆力偶",
            "对策：质量代换到铰链点 / 对称机构 / 加平衡配重",
        ], size=26).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p93-94"))
        self.hold(3)


class S06_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 5 讲小结")
        pts = bullets([
            "不平衡力 F=m·e·ω² 随转速平方增长        (p85-86)",
            "静平衡：单平面，Σm·r=0                  (p87-88)",
            "动平衡：两平面，ΣF=0 且 ΣM=0            (p88-91)",
            "精度等级 [eω]；静平衡导轨/动平衡机        (p91-93)",
            "机构平衡：质量代换+对称布置              (p93-94)",
        ], size=27)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：转速忽快忽慢？——飞轮与速度波动调节",
                  size=25, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
