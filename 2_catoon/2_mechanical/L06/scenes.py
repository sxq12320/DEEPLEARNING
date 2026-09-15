# -*- coding: utf-8 -*-
"""L06 机器的心跳——运转·速度波动·飞轮（第7章, p95-118）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_ThreeStages(LessonScene):
    """机器运转三阶段（~6min, p95-97）：ω-t 曲线三段——启动/稳定/停车，
    能量视角：输入功>输出功则加速。"""

    def construct(self):
        self.header("机器的一生：三条速度段", "运转阶段 · p95-97")
        ax = Axes(x_range=[0, 10, 1], y_range=[0, 1.3, 0.5],
                  x_length=10.5, y_length=4.2,
                  axis_config={"color": FRAME_C, "stroke_width": 3})
        ax.shift(DOWN * 0.3)
        labx = ctext("t", size=26).next_to(ax.x_axis, RIGHT, buff=0.15)
        laby = ctext("ω", size=26).next_to(ax.y_axis, UP, buff=0.15)
        # ω 曲线：升 → 波动稳态 → 降
        seg1 = ax.plot(lambda t: 1 - np.exp(-t * 2), x_range=[0, 2],
                       color=ACCENT, stroke_width=4)
        seg2 = ax.plot(lambda t: 0.86 + 0.05 * np.sin((t - 2) * 4),
                       x_range=[2, 7], color=GOOD, stroke_width=4)
        seg3 = ax.plot(lambda t: (0.86 + 0.05 * np.sin(20)) *
                       np.exp(-(t - 7) * 1.5), x_range=[7, 10],
                       color=BAD, stroke_width=4)
        self.play(Create(ax), FadeIn(labx), FadeIn(laby))
        self.play(Create(seg1), run_time=1.6)
        t1 = ctext("启动：驱动功 > 阻抗功 → 加速", size=25,
                   color=ACCENT).next_to(seg1, UP, buff=0.3)
        self.play(Write(t1))
        self.play(Create(seg2), run_time=2)
        t2 = ctext("稳定运转：周期内驱动功 = 阻抗功（仍小波动）", size=25,
                   color=GOOD).move_to(ax.c2p(5.6, 1.22))
        self.play(Write(t2))
        self.play(Create(seg3), run_time=1.6)
        t3 = ctext("停车：撤驱动力，耗能减速", size=25,
                   color=BAD).next_to(seg3, DOWN, buff=0.4)
        self.play(Write(t3))
        self.add(page_ref("孙桓八版 p95-97"))
        self.hold(3)


class S02_Equivalent(LessonScene):
    """等效动力学模型（~10min, p99-104）：整机 → 单根等效构件；
    等效转动惯量 J_e（动能等效）、等效力矩 M_e（功率等效）。"""

    def construct(self):
        self.header("把整机'折算'成一根杆", "等效模型 · p99-104")
        # 左：机构 右：等效杆
        cs = AnimatedCrankSlider(CrankSlider(0.6, 1.8, 0.0,
                                             origin=np.array([-4.9, -0.3, 0])),
                                 cylinder=False)
        self.play(FadeIn(cs.group))
        arr = vec(P(-1.6, 0.4), RIGHT * 1.2, ACCENT)
        self.play(FadeIn(arr))
        eq_bar = VGroup(fixed_pin(P(1.8, 0.0)),
                        link_line(P(1.8, 0.0), P(3.2, 1.1), LINK_A),
                        Dot(P(3.2, 1.1), radius=0.12, color=LINK_D))
        self.play(FadeIn(eq_bar))
        lab = ctext("等效构件\nJe + Me", size=24,
                    color=ACCENT).next_to(eq_bar, UP, buff=0.5)
        self.play(Write(lab))
        self.hold(1.5)
        steps = [
            mtex(r"J_e = \sum_i m_i\Big(\frac{v_{Si}}{\omega}\Big)^2"
                    r" + \sum_i J_{Si}\Big(\frac{\omega_i}{\omega}\Big)^2",
                    font_size=40),
            mtex(r"M_e = \sum_i F_i\frac{v_i\cos\alpha_i}{\omega}"
                    r" + \sum_i M_i\frac{\omega_i}{\omega}", font_size=40),
            mtex(r"\text{原则：动能等效 }E_e=E,\ \text{功率等效 }P_e=P",
                    font_size=36, color=ACCENT),
        ]
        formula_reveal(self, steps, anchor=DOWN * 2.9, buff=0.35, wait=1.8)
        self.add(page_ref("孙桓八版 p99-104"))
        self.hold(3)


class S03_EnergyDiagram(LessonScene):
    """盈亏功与能量指示图（~10min, p109-111）：M_d−M_r 曲线逐段面积正/负
    累积 → ωmax/ωmin 出现在累积峰谷处。"""

    def construct(self):
        self.header("能量指示图", "盈亏功 → 最大/最小角速度 · p109-111")
        ax = Axes(x_range=[0, 6.4, 1], y_range=[-1.2, 1.6, 1],
                  x_length=9.6, y_length=3.4,
                  axis_config={"color": FRAME_C, "stroke_width": 3})
        ax.shift(LEFT * 0.9 + UP * 0.7)
        curve = ax.plot(lambda t: 0.9 * np.sin(t) + 0.15, x_range=[0, TAU],
                        color=LINK_B, stroke_width=4)
        self.play(Create(ax), Create(curve))
        zero = ax.plot(lambda t: 0, x_range=[0, TAU], color=MUTED,
                       stroke_width=2)
        self.play(Create(zero))
        lt = ctext("Md − Mr（驱动力矩−阻抗力矩）", size=24).move_to(
            ax.c2p(2.1, 1.32))
        self.play(Write(lt))
        # 盈亏面积分段着色
        pos = ax.get_area(curve, x_range=[0.28, 3.42], color=GOOD, opacity=0.4)
        neg = ax.get_area(curve, x_range=[3.42, 6.0], color=BAD, opacity=0.4)
        self.play(FadeIn(pos))
        self.play(Write(ctext("盈功：加速", size=23, color=GOOD)
                        .next_to(pos, DOWN, buff=0.15)))
        self.play(FadeIn(neg))
        self.play(Write(ctext("亏功：减速", size=23, color=BAD)
                        .move_to(neg.get_center())))
        self.hold(2)
        note = ctext("累积能量最高点→ωmax；最低点→ωmin；差=最大盈亏功 ΔW",
                     size=25, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p109-111"))
        self.hold(3)


class S04_Flywheel(LessonScene):
    """飞轮设计（~10min, p111-115）：δ=(ωmax−ωmin)/ωm 不均匀系数；
    J_F=ΔW_max/(ω_m²δ) 推导——飞轮'削峰填谷'动画。"""

    def construct(self):
        self.header("飞轮：机器的能量银行", "δ 与 JF · p111-115")
        # 左：装飞轮的曲柄滑块
        cs = AnimatedCrankSlider(CrankSlider(0.6, 1.9, 0.0,
                                             origin=np.array([-4.6, -1.2, 0])),
                                 cylinder=False)
        fw = flywheel(P(-4.6, -1.2), r=1.05)
        self.play(FadeIn(cs.group), FadeIn(fw))
        self.play(cs.theta.animate.set_value(2 * TAU),
                  Rotate(fw, 2 * TAU, about_point=P(-4.6, -1.2)),
                  run_time=5, rate_func=linear)
        self.hold(1)
        steps = [
            mtex(r"\delta = \frac{\omega_{max}-\omega_{min}}{\omega_m}",
                    font_size=46),
            mtex(r"\Delta W_{max} = J_F\,\omega_m^2\,\delta"
                    r"\ \Rightarrow\ J_F=\frac{\Delta W_{max}}"
                    r"{\omega_m^2\,\delta}", font_size=46, color=ACCENT),
            mtex(r"\text{飞轮装在高速轴：同样 }J\text{ 储能翻倍}",
                    font_size=36),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 2.7 + UP * 0.4, wait=1.8)
        self.add(page_ref("孙桓八版 p111-115"))
        self.hold(3)


class S05_Governor(LessonScene):
    """周期性 vs 非周期性波动（~6min, p115-117）：飞轮管周期性；非周期性靠
    调速器——离心调速器原理动画（转速升→飞球张开→油门关小）。"""

    def construct(self):
        self.header("非周期性波动：调速器", "p115-117")
        # 离心调速器示意：立轴+两飞球
        O = P(-3.4, -0.6)
        th = ValueTracker(0.5)

        def gov():
            sp = 0.5 + 0.5 * np.sin(th.get_value())  # 张角随“转速”变
            b1 = O + np.array([np.cos(PI / 2 + sp), np.sin(PI / 2 + sp), 0]) * 1.4
            b2 = O + np.array([np.cos(PI / 2 - sp), np.sin(PI / 2 - sp), 0]) * 1.4
            collar = O + UP * (1.4 * np.cos(sp)) * 0.8
            return VGroup(
                Line(O + DOWN * 0.4, O + UP * 2.2, color=FRAME_C,
                     stroke_width=5),
                link_line(O + UP * 0.6, b1, LINK_B, 5),
                link_line(O + UP * 0.6, b2, LINK_B, 5),
                Dot(b1, radius=0.14, color=LINK_D),
                Dot(b2, radius=0.14, color=LINK_D),
                Rectangle(width=0.5, height=0.3, color=ACCENT,
                          fill_opacity=0.4).move_to(collar + UP * 0.1),
            )
        g = always_redraw(gov)
        self.play(FadeIn(g))
        self.play(th.animate.set_value(0.5 + TAU * 1.5), run_time=5)
        self.hold(1)
        pts = bullets([
            "转速↑ → 飞球张开 → 套筒上移 → 关小油门 → 回落",
            "转速↓ → 飞球收拢 → 开大油门——负反馈调速",
            "飞轮 vs 调速器：一个管'周期内'，一个管'趋势性'",
        ], size=26).shift(RIGHT * 2.4 + DOWN * 0.5)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p115-117"))
        self.hold(3)


class S06_FlywheelInLife(LessonScene):
    """飞轮实物感（~4min）：冲床单发冲裁靠飞轮储能；内燃机飞轮平抑做功冲程
    脉动——功率曲线脉动 vs 平滑输出对照。"""

    def construct(self):
        self.header("飞轮就在身边", "冲床 · 发动机 · p95")
        ax = Axes(x_range=[0, 8, 1], y_range=[0, 1.5, 0.5], x_length=10,
                  y_length=3.6, axis_config={"color": FRAME_C})
        ax.shift(UP * 0.3)
        load = ax.plot(lambda t: 1.35 if (t % 2) > 1.8 else 0.12,
                       x_range=[0, 8], color=BAD, stroke_width=4)
        smooth = ax.plot(lambda t: 0.35, x_range=[0, 8], color=GOOD,
                         stroke_width=4)
        self.play(Create(ax))
        self.play(Create(load))
        self.play(Write(ctext("阻抗功率（冲床冲裁瞬间峰值）", size=23,
                              color=BAD).next_to(ax, UP, buff=0.3)))
        self.play(Create(smooth))
        self.play(Write(ctext("电机只出平稳小功率，差额由飞轮吐纳", size=23,
                              color=GOOD).move_to(P(1.4, -1.85))))
        self.hold(2.5)
        self.takeaway("飞轮=储能器：以小电机带动脉动大负载", p="孙桓八版 p95-115")
        self.hold(3)


class S07_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 6 讲小结")
        pts = bullets([
            "运转三阶段：启动/稳定/停车；能量守恒判向     (p95-97)",
            "等效模型：Je 动能等效、Me 功率等效        (p99-104)",
            "能量指示图找 ΔWmax → ωmax/ωmin          (p109-111)",
            "δ=(ωmax−ωmin)/ωm；JF=ΔW/(ωm²δ)          (p111-115)",
            "周期性波动→飞轮；非周期性→调速器           (p115-117)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：机构设计主线——四根杆的智慧",
                  size=25, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
