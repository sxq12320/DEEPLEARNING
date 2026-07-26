# -*- coding: utf-8 -*-
"""L11 让机器跑得稳——平衡与速度波动调节（孙桓八版 第6章+第7章, p85-118）+ 全课收官

目标成片 90-100 min。核心：静/动平衡、等效力学模型、能量指示图、飞轮 J_F 推导、知识地图收官。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：洗衣机脱水'跳舞'=不平衡；冲床忽快忽慢=速度波动。
    两种'心律不齐'，两副药：配重与飞轮。(p85, p99)"""

    def construct(self):
        big = ctext("第 11 讲   让机器跑得稳", size=58, weight="BOLD")
        sub = ctext("平衡 · 速度波动 · 飞轮   —— 收官之战", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        pts = bullets([
            "病症一：转子不平衡 → 离心力抖动、轴承折寿（洗衣机跳舞）",
            "病症二：驱动力≠阻力 → 转速周期波动（冲床喘气）",
            "药方：配重平衡术 + 飞轮'蓄水池'",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(pts, lag_ratio=0.4))
        self.hold(3)


class S02_StaticBalance(LessonScene):
    """★刚性转子静平衡（12min，p86-87）。
    讲稿要点：不平衡质量的离心力 F=mrω²；静平衡条件 ∑F=0 ⇔ ∑m_i r_i = 0（质径积矢量和为零）；
    单平衡面配重的矢量多边形解法。动画：不平衡轮抖动 → 加配重 → 平稳。"""

    def construct(self):
        self.header("静平衡", "质径积的矢量游戏 · p86-87")
        ctr0 = np.array([-3.4, 0.3, 0])
        rotor = Circle(radius=1.3, color=GREY_B, stroke_width=5).move_to(ctr0)
        m1 = Dot(ctr0 + np.array([0.8, 0.5, 0]), color=RED, radius=0.14)
        m2 = Dot(ctr0 + np.array([-0.5, 0.9, 0]), color=ORANGE, radius=0.11)
        g = VGroup(rotor, m1, m2)
        self.play(Create(rotor), FadeIn(m1), FadeIn(m2))
        t1 = ctext("偏心质量随转动甩出离心力 F = m r ω² —— 方向一直在转！", size=27).to_edge(DOWN, buff=1.25)
        self.play(Write(t1))
        # 抖动演示
        for k in range(3):
            self.play(g.animate.shift(UP * 0.08 + RIGHT * 0.05), run_time=0.12)
            self.play(g.animate.shift(DOWN * 0.16 + LEFT * 0.1), run_time=0.12)
            self.play(g.animate.shift(UP * 0.08 + RIGHT * 0.05), run_time=0.12)
        f = [
            MathTex(r"\text{静平衡条件: } \sum \vec F_i = 0\ \iff\ \sum m_i \vec r_i = 0", font_size=44, color=YELLOW),
            MathTex(r"\text{配重: } m_b \vec r_b = -\sum m_i \vec r_i\ \text{（质径积矢量多边形闭合）}",
                    font_size=42, color=GREEN),
        ]
        self.play(FadeOut(t1))
        formula_reveal(self, f, anchor=RIGHT * 2.6 + UP * 0.9, buff=0.42, wait=1.9)
        mb = Dot(ctr0 + np.array([-0.35, -1.05, 0]), color=GREEN, radius=0.13)
        t2 = ctext("加上配重（绿）→ 质径积闭合 → 支承不再受周期力", size=27, color=GOOD).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(mb), Write(t2))
        self.play(Rotate(VGroup(g, mb), TAU, about_point=ctr0), run_time=3, rate_func=linear)
        self.add(page_ref("孙桓八版 p86-87"))
        self.hold(2.5)


class S03_DynamicBalance(LessonScene):
    """★动平衡（12min，p87-88）。
    讲稿要点：'静平衡但动不平衡'经典案例——同轴两侧对称反向的偏心质量：
    ∑F=0 但形成力偶 → 高速摆头；动平衡条件 ∑F=0 且 ∑M=0；
    任一不平衡量可分解到任选两个平衡基面 → 两面各配一次即可；'动平衡必含静平衡'。"""

    def construct(self):
        self.header("动平衡", "看不见的力偶 · p87-88")
        axis = Line(LEFT * 4.5, RIGHT * 1.5, color=GREY_B, stroke_width=4).shift(UP * 0.5)
        rot1 = Circle(radius=0.9, color=GREY_C, stroke_width=4).move_to(LEFT * 3.2 + UP * 0.5)
        rot2 = Circle(radius=0.9, color=GREY_C, stroke_width=4).move_to(RIGHT * 0.2 + UP * 0.5)
        mA = Dot(rot1.get_center() + UP * 0.65, color=RED, radius=0.13)
        mB = Dot(rot2.get_center() + DOWN * 0.65, color=RED, radius=0.13)
        self.play(Create(axis), Create(rot1), Create(rot2), FadeIn(mA), FadeIn(mB))
        t1 = ctext("两块偏心质量：大小相等、方向相反 → ∑F = 0，静平衡 ✓", size=27).to_edge(DOWN, buff=1.3)
        self.play(Write(t1))
        self.hold(2)
        fA = Arrow(mA.get_center(), mA.get_center() + UP * 1.0, buff=0, color=RED)
        fB = Arrow(mB.get_center(), mB.get_center() + DOWN * 1.0, buff=0, color=RED)
        t2 = ctext("但转起来它们隔着距离反向甩 → 力偶！转子高速'摆头'——动不平衡", size=27,
                   color=BAD).to_edge(DOWN, buff=1.3)
        self.play(GrowArrow(fA), GrowArrow(fB), ReplacementTransform(t1, t2))
        self.play(Rotate(VGroup(axis, rot1, rot2, mA, mB, fA, fB), 0.09, about_point=UP * 0.5 + LEFT * 1.5),
                  run_time=0.4)
        self.play(Rotate(VGroup(axis, rot1, rot2, mA, mB, fA, fB), -0.18, about_point=UP * 0.5 + LEFT * 1.5),
                  run_time=0.6)
        self.play(Rotate(VGroup(axis, rot1, rot2, mA, mB, fA, fB), 0.09, about_point=UP * 0.5 + LEFT * 1.5),
                  run_time=0.4)
        f = [
            MathTex(r"\text{动平衡条件: } \sum\vec F = 0\ \ \textbf{且}\ \ \sum\vec M = 0", font_size=46, color=YELLOW),
            MathTex(r"\text{任一偏心量都可分解到任选的两个平衡基面 T'/T''}", font_size=38),
            MathTex(r"\Rightarrow\ \text{两个面各加一个配重即可平衡任意刚性转子}", font_size=40, color=GREEN),
            MathTex(r"\text{动平衡} \supset \text{静平衡（反之不然）}", font_size=38),
        ]
        self.play(FadeOut(t2))
        formula_reveal(self, f, anchor=DOWN * 2.2, buff=0.3, wait=1.9)
        self.add(page_ref("孙桓八版 p87-88"))
        self.hold(2.5)


class S04_BalancingPractice(LessonScene):
    """平衡实验与许用不平衡量（8min，p88-92）。
    讲稿要点：静平衡架(滚下最低点=重侧)；动平衡机(测两基面振动相位与幅值)；
    许用不平衡量按转速分级(G 等级概念)；宽径比 b/d 小的盘状件只需静平衡。"""

    def construct(self):
        self.header("工厂里怎么做平衡", "p88-92")
        pts = bullets([
            "静平衡架：让转子自由滚动，重的一侧总停在最下——磨掉或对侧配重",
            "动平衡机：转起来测两端轴承振动的幅值+相位 → 算出两基面配重",
            "宽径比小（薄盘, b/d<0.2）：静平衡即可；长转子必须动平衡",
            "许用不平衡量：按转速与用途分 G 等级（精密磨头 G1，普通风机 G6.3）",
        ], size=29)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=3)
        self.add(page_ref("孙桓八版 p88-92"))
        self.hold(4)


class S05_MotionStages(LessonScene):
    """机械运转三阶段（8min，p99-101）。
    讲稿要点：起动(Wd>Wr, E↑)、稳定运转(周期内 Wd=Wr)、停车(Wd=0, E↓)；
    周期性速度波动的根源：一个周期内任意瞬时 Md≠Mr。动画：ω(t) 三段曲线。"""

    def construct(self):
        self.header("机械的一生：三个阶段", "p99-101")
        ax = Axes(x_range=[0, 10, 1], y_range=[0, 1.5, 0.5], x_length=9.5, y_length=3.6,
                  axis_config={"include_tip": False, "include_ticks": False}).shift(UP * 0.4)
        def om(t):
            if t < 2.5:
                return 1.0 * (1 - np.exp(-1.6 * t))
            if t < 7.5:
                return 1.0 + 0.08 * np.sin(4 * TAU * (t - 2.5) / 5)
            return max(0, (1.0 + 0.0) * np.exp(-1.2 * (t - 7.5)))
        cur = ax.plot(om, x_range=[0, 10, 0.02], color=TEAL)
        xl = ctext("t", size=26).next_to(ax.x_axis, RIGHT); yl = MathTex(r"\omega", font_size=34).next_to(ax.y_axis, UP)
        zones = VGroup(
            ctext("起动\nWd>Wr", size=24, color=GREEN).move_to(ax.c2p(1.2, 1.32)),
            ctext("稳定运转（周期波动）\n每周期 Wd=Wr", size=24, color=YELLOW).move_to(ax.c2p(5.0, 1.32)),
            ctext("停车\nWd=0", size=24, color=RED).move_to(ax.c2p(8.7, 1.32)),
        )
        self.play(Create(ax), FadeIn(xl), FadeIn(yl))
        self.play(Create(cur), run_time=3)
        self.play(FadeIn(zones, lag_ratio=0.3))
        note = ctext("稳定段的'锯齿'就是周期性速度波动——本讲要驯服的对象", size=28,
                     color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p99-101"))
        self.hold(3)


class S06_EquivalentModel(LessonScene):
    """★等效力学模型（12min，p101-105）。
    讲稿要点：把整台机器'压缩'成一个等效构件——动能等效定 Je、功率等效定 Me；
    Je = Σ[mi(vi/ω)² + Ji(ωi/ω)²]（速比只与位置有关 → Je 是位置的函数）；
    运动方程 d(Jeω²/2) = (Md−Mr)dδ。"""

    def construct(self):
        self.header("等效力学模型", "整台机器 = 一个转动构件 · p101-105")
        f = [
            MathTex(r"\text{动能等效: } \tfrac{1}{2}J_e\omega^2 = \sum\left[\tfrac{1}{2}m_i v_{Si}^2 + \tfrac{1}{2}J_{Si}\omega_i^2\right]",
                    font_size=42),
            MathTex(r"\Rightarrow\ J_e = \sum\left[m_i\Big(\tfrac{v_{Si}}{\omega}\Big)^2 + J_{Si}\Big(\tfrac{\omega_i}{\omega}\Big)^2\right]",
                    font_size=44, color=YELLOW),
            MathTex(r"\text{速比只随位置变} \Rightarrow J_e = J_e(\varphi)\ \text{与转速无关}", font_size=38, color=GREEN),
            MathTex(r"\text{功率等效: } M_e\omega = \sum(F_i v_i \cos\theta_i \pm M_i\omega_i)", font_size=40),
            MathTex(r"\text{能量形式运动方程: } d\!\left(\tfrac{1}{2}J_e\omega^2\right) = (M_{ed}-M_{er})\,d\varphi",
                    font_size=42, color=YELLOW),
        ]
        formula_reveal(self, f, anchor=UP * 0.2, buff=0.36, wait=2.0)
        self.add(page_ref("孙桓八版 p101-105"))
        self.hold(3)


class S07_Flywheel(LessonScene):
    """★飞轮设计推导（15min，p109-114，全课最后一个大推导）。
    讲稿要点：δ=(ωmax−ωmin)/ωm；能量指示图逐段累加 Wd−Wr 找最大盈亏功 Wmax
    （最高/最低动能位置之间）；ωmax²−ωmin²=2Wmax/J → J_F = Wmax/(δωm²) − Jе0；
    δ↓ 或 ωm↑ 都省飞轮 → 飞轮装高速轴。"""

    def construct(self):
        self.header("飞轮：机器的蓄水池", "J_F = W_max / (δ ω_m²) · p109-114")
        # 能量指示图
        ax = Axes(x_range=[0, TAU, PI / 2], y_range=[-1.4, 1.4, 1], x_length=6.4, y_length=2.9,
                  axis_config={"include_tip": False, "include_ticks": False}).shift(LEFT * 2.7 + UP * 1.3)
        dM = lambda x: 0.9 * np.sin(2 * x) + 0.35 * np.sin(x)
        cur = ax.plot(dM, x_range=[0, TAU, 0.02], color=TEAL)
        lab = MathTex(r"M_d - M_r", font_size=32, color=TEAL).next_to(ax, UP, buff=0.1)
        self.play(Create(ax), Create(cur), Write(lab))
        t1 = ctext("盈亏功：曲线与横轴围出的面积，正盈负亏——逐段累加找能量最高/最低点",
                   size=26).to_edge(DOWN, buff=1.25)
        self.play(Write(t1))
        area1 = ax.get_area(cur, x_range=[0, PI / 2 + 0.35], color=GREEN, opacity=0.4)
        area2 = ax.get_area(cur, x_range=[PI / 2 + 0.35, PI + 0.6], color=RED, opacity=0.4)
        self.play(FadeIn(area1), FadeIn(area2))
        self.hold(2.2)
        f = [
            MathTex(r"\delta = \frac{\omega_{\max}-\omega_{\min}}{\omega_m},\qquad \omega_m=\frac{\omega_{\max}+\omega_{\min}}{2}",
                    font_size=40),
            MathTex(r"W_{\max} = E_{\max}-E_{\min} = \tfrac{1}{2}J(\omega_{\max}^2-\omega_{\min}^2)", font_size=40),
            MathTex(r"\omega_{\max}^2-\omega_{\min}^2 = (\omega_{\max}+\omega_{\min})(\omega_{\max}-\omega_{\min}) = 2\omega_m^2\delta",
                    font_size=40),
            MathTex(r"\boxed{J_F = \frac{W_{\max}}{\delta\,\omega_m^2} - J_{e0}}", font_size=52, color=YELLOW),
        ]
        self.play(FadeOut(t1))
        formula_reveal(self, f, anchor=RIGHT * 2.9 + DOWN * 0.4, buff=0.32, wait=2.0)
        notes = bullets([
            "δ 定死（发电机 1/300，冲床 1/10）→ 反解 J_F",
            "J_F ∝ 1/ωm²：飞轮装在高速轴上最省料",
            "飞轮调'周期性'波动；非周期波动要靠调速器",
        ], size=26).to_edge(DOWN, buff=0.2)
        self.play(FadeIn(notes, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p109-114"))
        self.hold(3)


class S08_FlywheelDemo(LessonScene):
    """加/不加飞轮对比（8min）。动画：同一 Md−Mr 激励下两条 ω(t) 曲线——
    小 J 大起大落 vs 大 J 平稳；'蓄水池'类比：忙时放水闲时蓄水。"""

    def construct(self):
        self.header("飞轮前 vs 飞轮后", "同样的激励，不同的心跳")
        ax = Axes(x_range=[0, 4 * TAU, TAU], y_range=[0.5, 1.5, 0.25], x_length=10.5, y_length=4.2,
                  axis_config={"include_tip": False, "include_ticks": False}).shift(DOWN * 0.2)
        w_small = ax.plot(lambda t: 1 + 0.22 * np.sin(2 * t) + 0.07 * np.sin(t), x_range=[0, 4 * TAU, 0.02],
                          color=RED)
        w_big = ax.plot(lambda t: 1 + 0.045 * np.sin(2 * t) + 0.014 * np.sin(t), x_range=[0, 4 * TAU, 0.02],
                        color=GREEN)
        leg = VGroup(ctext("无飞轮：δ 大，机器'喘气'", size=26, color=RED),
                     ctext("加飞轮：δ 小，运转平稳", size=26, color=GREEN)
                     ).arrange(DOWN, aligned_edge=LEFT, buff=0.25).to_corner(UP + RIGHT, buff=0.9)
        self.play(Create(ax))
        self.play(Create(w_small), run_time=2.5)
        self.play(FadeIn(leg[0]))
        self.hold(1.5)
        self.play(Create(w_big), run_time=2.5)
        self.play(FadeIn(leg[1]))
        note = ctext("能量视角：盈功存进飞轮动能，亏功从飞轮取出——'削峰填谷'", size=28,
                     color=ACCENT).to_edge(DOWN, buff=0.5)
        self.play(Write(note))
        self.hold(3.5)


class S09_GrandFinale(LessonScene):
    """★全课收官（10min）：L01 那台内燃机重新出场，11 讲知识点在它身上逐一点亮：
    简图/自由度→运动分析→连杆(曲柄滑块)→凸轮(配气)→齿轮(正时)→轮系(减速)→
    摩擦效率→平衡(曲轴配重)→飞轮(必备!)。结语+第13/14章展望。"""

    def construct(self):
        self.header("终点回到起点", "一台内燃机 · 十一讲知识")
        cs = CrankSlider(0.9, 2.6, 0.0, origin=np.array([-4.3, -0.9, 0]))
        th = ValueTracker(0.0)

        def engine():
            O, A, B = cs.solve(th.get_value())
            return VGroup(link_line(O, A, GOLD), link_line(A, B, BLUE_B),
                          slider_block(B, 0.7, 0.5), pin_joint(A), fixed_pin(O))

        self.play(FadeIn(always_redraw(engine)))
        self.play(th.animate.set_value(TAU), run_time=3, rate_func=linear)
        checklist = [
            "简图与自由度 F=1        —— 第 1-2 讲",
            "活塞速度加速度           —— 第 3 讲",
            "曲柄滑块特性与死点        —— 第 4-5 讲",
            "配气凸轮的运动规律        —— 第 6 讲",
            "正时齿轮与渐开线          —— 第 7-8 讲",
            "附件轮系传动              —— 第 9 讲",
            "摩擦损耗与机械效率        —— 第 10 讲",
            "曲轴配重平衡 + 飞轮       —— 第 11 讲",
        ]
        rows = VGroup()
        for i, s in enumerate(checklist):
            row = ctext("✓ " + s, size=26, color=GREEN)
            rows.add(row)
        rows.arrange(DOWN, aligned_edge=LEFT, buff=0.24).shift(RIGHT * 2.7 + DOWN * 0.2)
        for row in rows:
            self.play(FadeIn(row, shift=RIGHT * 0.3), run_time=0.55)
            self.hold(0.7)
        self.play(th.animate.set_value(3 * TAU), run_time=4, rate_func=linear)
        end1 = ctext("一台机器的每个细节，你现在都有名字、有公式、有方法。", size=30, color=ACCENT).to_edge(DOWN, buff=0.75)
        end2 = ctext("机械原理，完结。去造点什么吧。", size=34, weight="BOLD").to_edge(DOWN, buff=0.15)
        self.play(Write(end1))
        self.hold(2)
        self.play(Write(end2))
        self.hold(4)
