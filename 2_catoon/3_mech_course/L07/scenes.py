# -*- coding: utf-8 -*-
"""L07 最完美的曲线（上）——渐开线齿廓与标准齿轮（孙桓八版 §10-1~10-5, p195-208）

目标成片 85-95 min。核心：啮合基本定律、渐开线五性质、标准参数、正确啮合、重合度。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：从手表到风电，齿轮无处不在；本讲回答三个问题——
    什么齿形能定传动比？为什么全世界选了渐开线？'模数'到底是什么？(p195-197)"""

    def construct(self):
        big = ctext("第 7 讲   最完美的曲线（上）", size=56, weight="BOLD")
        sub = ctext("渐开线：统治齿轮世界 250 年", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        g1 = gear_profile(0.28, 15, color=TEAL).shift(LEFT * 4.6 + DOWN * 2.2)
        g2 = gear_profile(0.28, 24, color=BLUE_B).shift(LEFT * 4.6 + DOWN * 2.2 + RIGHT * 5.55)
        self.play(Write(big), FadeIn(sub), Create(g1), Create(g2))
        self.play(Rotate(g1, TAU / 2, about_point=g1.get_center()),
                  Rotate(g2, -TAU / 2 * 15 / 24, about_point=g2.get_center()),
                  run_time=4, rate_func=linear)
        self.hold(2)


class S02_FundamentalLaw(LessonScene):
    """★齿廓啮合基本定律（12min，p197-199）。
    讲稿要点：两齿廓接触点公法线与连心线交点 = 节点 C = 相对速度瞬心（三心定理回归！）；
    要传动比恒定 ⇔ C 固定 ⇔ 任意接触位置的公法线都过同一点。节圆=过 C 的两个'纯滚动圆'。"""

    def construct(self):
        self.header("齿廓啮合基本定律", "公法线必过节点 · p197-199")
        O1 = np.array([-3.5, 0.6, 0]); O2 = np.array([3.0, 0.6, 0])
        c1 = Dot(O1, color=GOLD, radius=0.08); c2 = Dot(O2, color=TEAL, radius=0.08)
        centerline = Line(O1, O2, color=GREY_B, stroke_width=2)
        l1 = MathTex("O_1", font_size=34).next_to(O1, UP); l2 = MathTex("O_2", font_size=34).next_to(O2, UP)
        self.play(FadeIn(c1), FadeIn(c2), Create(centerline), Write(l1), Write(l2))
        K = np.array([-0.6, -0.9, 0])
        contact = Dot(K, color=RED, radius=0.08)
        arc1 = Arc(radius=1.6, angle=1.2, start_angle=-0.9, color=GOLD, stroke_width=4).move_arc_center_to(K + np.array([-1.1, -0.9, 0]))
        arc2 = Arc(radius=1.9, angle=1.1, start_angle=PI * 0.62, color=TEAL, stroke_width=4).move_arc_center_to(K + np.array([1.3, -1.1, 0]))
        t1 = ctext("两齿廓在 K 接触——能不能保持接触？相对速度必须沿公切线！", size=27).to_edge(DOWN, buff=1.25)
        self.play(Create(arc1), Create(arc2), FadeIn(contact), Write(t1))
        self.hold(2)
        # 公法线交连心线于 C
        nline = Line(K + np.array([-0.9, -1.4, 0]) * 0.8, K + np.array([0.9, 1.4, 0]) * 1.6,
                     color=YELLOW, stroke_width=3)
        C = np.array([-0.6 + 0.9 * ((0.6 + 0.9) / 1.4), 0.6, 0])
        dC = Dot(C, color=YELLOW, radius=0.09); lC = MathTex("C", font_size=36, color=YELLOW).next_to(C, UP + RIGHT, buff=0.08)
        t2 = ctext("接触点公法线 交 连心线 于节点 C —— C 正是两轮的相对速度瞬心（三心定理！）",
                   size=26, color=ACCENT).to_edge(DOWN, buff=1.25)
        self.play(Create(nline), FadeIn(dC), Write(lC), ReplacementTransform(t1, t2))
        self.hold(2.2)
        f = [
            MathTex(r"\frac{\omega_1}{\omega_2} = \frac{\overline{O_2C}}{\overline{O_1C}}", font_size=48),
            MathTex(r"\text{传动比恒定} \iff C\ \text{固定} \iff \text{任意接触位置公法线过同一点}",
                    font_size=38, color=YELLOW),
            MathTex(r"\text{过 } C \text{ 的两圆（节圆）作纯滚动}", font_size=38, color=GREEN),
        ]
        self.play(FadeOut(t2))
        formula_reveal(self, f, anchor=DOWN * 2.5, buff=0.32, wait=1.8)
        self.add(page_ref("孙桓八版 p197-199"))
        self.hold(2.5)


class S03_InvoluteGeneration(LessonScene):
    """★渐开线的生成与五大性质（15min，p199-200，全片最经典镜头）。
    动画：绳子绕基圆展开画渐开线。性质逐条验证：①发生线滚过弧长=展直线段长
    ②法线切于基圆 ③曲率半径=NK ④基圆内无渐开线 ⑤形状仅取决于基圆大小。"""

    def construct(self):
        self.header("渐开线的诞生", "一根绳子的几何 · p199-200")
        rb = 1.5
        ctr = np.array([-3.0, -0.5, 0])
        base = Circle(radius=rb, color=GREY_B, stroke_width=3).move_to(ctr)
        blab = ctext("基圆 rb", size=24, color=GREY_B).next_to(base, DOWN, buff=0.3)
        self.play(Create(base), FadeIn(blab))
        tv = ValueTracker(0.01)

        def taut_line():
            t = tv.get_value()
            N = ctr + rb * np.array([np.cos(t), np.sin(t), 0])
            K = ctr + rb * np.array([np.cos(t) + t * np.sin(t), np.sin(t) - t * np.cos(t), 0])
            return VGroup(Line(N, K, color=YELLOW, stroke_width=3),
                          Dot(N, color=GREY_B, radius=0.05), Dot(K, color=RED, radius=0.07))

        def inv_trace():
            t = tv.get_value()
            pts = involute_pts(rb, max(t, 0.02), n=60)
            pts = pts + ctr
            vm = VMobject(color=ORANGE, stroke_width=4)
            vm.set_points_smoothly(list(pts))
            return vm

        self.play(FadeIn(always_redraw(taut_line)), FadeIn(always_redraw(inv_trace)))
        t1 = ctext("拉直的绳端点轨迹 = 渐开线（发生线沿基圆纯滚动）", size=28).to_edge(DOWN, buff=1.25)
        self.play(Write(t1))
        self.play(tv.animate.set_value(2.2), run_time=6, rate_func=linear)
        props = bullets([
            "① 滚过的弧长 NK(弧) = 展直段 NK：|NK| = rb·t",
            "② 发生线 = 渐开线在 K 点的法线，且恒与基圆相切",
            "③ N 是曲率中心：曲率半径 ρ = NK（越远越平直）",
            "④ 基圆之内没有渐开线",
            "⑤ 形状只由 rb 决定：rb→∞ 变成直线（齿条！）",
        ], size=26).shift(RIGHT * 3.1 + UP * 0.4)
        self.play(ReplacementTransform(t1, ctext("五大性质：", size=28, color=ACCENT).to_edge(DOWN, buff=1.25)))
        self.play(FadeIn(props, lag_ratio=0.35), run_time=3.5)
        self.add(page_ref("孙桓八版 p199-200"))
        self.hold(3.5)


class S04_InvFunction(LessonScene):
    """渐开线方程与 inv 函数（8min，p200）。
    讲稿要点：以基圆为参照，K 点压力角 αK（cos αK = rb/rK）；
    展角 θK = inv αK = tan αK − αK 的推导（弧长关系）。"""

    def construct(self):
        self.header("渐开线函数 inv α", "工程师的查表神器 · p200")
        f = [
            MathTex(r"\cos\alpha_K = \frac{r_b}{r_K}\quad(\text{K 点压力角})", font_size=44),
            MathTex(r"\text{性质①: } \widehat{AN} = \overline{NK}\ \Rightarrow\ r_b(\theta_K+\alpha_K) = r_b\tan\alpha_K",
                    font_size=42),
            MathTex(r"\boxed{\theta_K = \mathrm{inv}\,\alpha_K = \tan\alpha_K - \alpha_K}", font_size=52, color=YELLOW),
            MathTex(r"\text{极坐标方程: } r_K = \frac{r_b}{\cos\alpha_K},\ \ \theta_K = \mathrm{inv}\,\alpha_K",
                    font_size=40, color=GREEN),
        ]
        formula_reveal(self, f, anchor=UP * 0.4, buff=0.42, wait=1.9)
        note = ctext("变位齿轮的无侧隙啮合方程全靠它——第 8 讲见", size=27, color=GREY_B).to_edge(DOWN, buff=0.6)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p200"))
        self.hold(3)


class S05_MeshingBeauty(LessonScene):
    """渐开线啮合三大优点（10min，p200-201）。
    动画：一对渐开线齿廓啮合，接触点沿两基圆内公切线（啮合线）移动；
    ①定传动比（公法线=啮合线恒过 C）②啮合线是直线 → 正压力方向不变
    ③中心距可分性：拉开中心距传动比不变（安装误差不敏感——工程杀手锏）。"""

    def construct(self):
        self.header("渐开线为什么完美", "三大啮合特性 · p200-201")
        z1, z2, m = 13, 19, 0.34
        r1, r2 = m * z1 / 2, m * z2 / 2
        a0 = r1 + r2
        O1 = np.array([-a0 / 2 - 0.4, 0, 0]); O2 = np.array([a0 / 2 - 0.4, 0, 0])
        g1 = gear_profile(m, z1, color=GOLD).move_to(O1)
        g2 = gear_profile(m, z2, color=TEAL).rotate(PI / z2).move_to(O2)
        self.play(Create(g1), Create(g2), run_time=2)
        self.play(Rotate(g1, TAU / z1 * 3, about_point=O1),
                  Rotate(g2, -TAU / z2 * 3, about_point=O2), run_time=5, rate_func=linear)
        alpha = np.deg2rad(20)
        # 啮合线：过节点 C 与连心线成 90°-α
        C = O1 + (O2 - O1) * (r1 / a0)
        d = np.array([np.sin(alpha), np.cos(alpha), 0])
        mesh_line = Line(C - d * 1.8, C + d * 1.8, color=YELLOW, stroke_width=3)
        t1 = ctext("接触点始终在这条固定直线（啮合线 = 两基圆内公切线）上移动", size=27,
                   color=ACCENT).to_edge(DOWN, buff=1.25)
        self.play(Create(mesh_line), Write(t1))
        self.hold(2.2)
        pts = bullets([
            "① 公法线=啮合线恒过定点 C → 传动比恒定",
            "② 啮合线是直线 → 齿面正压力方向恒定，传动平稳",
            "③ 可分性：中心距略变，传动比不变（因 rb 不变）",
        ], size=27).to_edge(DOWN, buff=0.12)
        self.play(ReplacementTransform(t1, pts))
        sep = ctext("演示：拉开中心距——", size=26, color=GREY_B).to_corner(UP + RIGHT, buff=1.0).shift(DOWN * 0.8)
        self.play(Write(sep))
        self.play(g2.animate.shift(RIGHT * 0.25), run_time=1.2)
        self.play(Rotate(g1, TAU / z1 * 2, about_point=O1),
                  Rotate(g2, -TAU / z2 * 2, about_point=O2 + RIGHT * 0.25), run_time=4, rate_func=linear)
        self.add(page_ref("孙桓八版 p200-201"))
        self.hold(2.5)


class S06_StandardParams(LessonScene):
    """★标准齿轮参数与几何尺寸（15min，p201-204）。
    讲稿要点：为什么需要模数——分度圆周长 πd = z·p → d = (p/π)z，令 m=p/π 标准化；
    五参数 m,z,α,ha*,c*；全套尺寸公式表；'模数越大牙越壮'的直观。"""

    def construct(self):
        self.header("标准齿轮的'身份证'", "m · z · α · ha* · c* · p201-204")
        f = [
            MathTex(r"\text{分度圆周长: } \pi d = z\,p\ \Rightarrow\ d = \frac{p}{\pi}\,z", font_size=42),
            MathTex(r"\text{令 } m \equiv \frac{p}{\pi}\ \text{(标准化系列值)}\ \Rightarrow\ \boxed{d = m z}",
                    font_size=48, color=YELLOW),
            MathTex(r"h_a = h_a^{*}m,\quad h_f=(h_a^{*}+c^{*})m,\quad (h_a^{*}=1,\ c^{*}=0.25)", font_size=38),
            MathTex(r"d_a = m(z+2h_a^{*}),\quad d_f = m(z-2h_a^{*}-2c^{*}),\quad d_b = mz\cos\alpha",
                    font_size=38),
            MathTex(r"s = e = \frac{\pi m}{2}\ \text{（标准齿轮齿厚=槽宽）},\quad \alpha=20^\circ", font_size=38, color=GREEN),
        ]
        formula_reveal(self, f, anchor=UP * 0.2, buff=0.34, wait=1.8)
        gs = VGroup(gear_profile(0.18, 16, color=GREY_B), gear_profile(0.3, 16, color=TEAL),
                    gear_profile(0.42, 16, color=GOLD)).arrange(RIGHT, buff=0.9).to_edge(DOWN, buff=0.25)
        lab = ctext("同 z 不同 m：模数越大，牙越'壮'，承载越强", size=25, color=ACCENT).next_to(gs, UP, buff=0.15)
        self.play(FadeIn(gs, lag_ratio=0.3), Write(lab))
        self.add(page_ref("孙桓八版 p201-204 表10-2"))
        self.hold(3)


class S07_MeshingConditions(LessonScene):
    """正确啮合条件 + 标准中心距（10min，p204-206）。
    讲稿要点：两轮法向齿距(基圆齿距)必须相等 pb1=pb2 → m1cosα1=m2cosα2 →
    标准化下 m1=m2 且 α1=α2；标准安装：分度圆=节圆，a=m(z1+z2)/2；啮合角=压力角。"""

    def construct(self):
        self.header("能不能咬合？", "正确啮合条件 · p204-206")
        f = [
            MathTex(r"\text{连续啮合: 前后两对齿必须'步调一致'}\ \Rightarrow\ p_{b1} = p_{b2}", font_size=40),
            MathTex(r"p_b = \pi m \cos\alpha\ \Rightarrow\ m_1\cos\alpha_1 = m_2\cos\alpha_2", font_size=42),
            MathTex(r"\text{m、}\alpha\text{ 皆已标准化}\ \Rightarrow\ \boxed{m_1=m_2,\ \ \alpha_1=\alpha_2}",
                    font_size=46, color=YELLOW),
            MathTex(r"\text{标准中心距: } a = r_1 + r_2 = \frac{m(z_1+z_2)}{2}", font_size=42, color=GREEN),
        ]
        formula_reveal(self, f, anchor=UP * 0.3, buff=0.4, wait=1.8)
        note = ctext("传动比只认齿数：i = ω1/ω2 = z2/z1 —— 与模数无关", size=28, color=ACCENT).to_edge(DOWN, buff=0.6)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p204-206"))
        self.hold(3)


class S08_ContactRatio(LessonScene):
    """★重合度 εα（12min，p206-208）。
    讲稿要点：连续传动要求前一对齿未脱啮、后一对已进入 → 实际啮合线段 B1B2 ≥ pb；
    εα = B1B2/pb；εα=1.3 含义：单/双齿对交替，30% 时间双对分担；εα↑ 平稳承载好。"""

    def construct(self):
        self.header("重合度 εα", "接力棒不能掉地上 · p206-208")
        f = [
            MathTex(r"\text{连续传动条件: } \overline{B_1B_2} \ \ge\ p_b", font_size=44),
            MathTex(r"\boxed{\varepsilon_\alpha = \frac{\overline{B_1B_2}}{p_b} \ \ge\ 1}", font_size=52, color=YELLOW),
            MathTex(r"\varepsilon_\alpha = \tfrac{1}{2\pi}\left[z_1(\tan\alpha_{a1}-\tan\alpha') + z_2(\tan\alpha_{a2}-\tan\alpha')\right]",
                    font_size=38),
        ]
        formula_reveal(self, f, anchor=UP * 1.1, buff=0.4, wait=1.9)
        # εα=1.3 的时间轴示意
        bar = Rectangle(width=10, height=0.7, color=GREY_B).shift(DOWN * 1.7)
        segs = VGroup()
        for k in range(4):
            x0 = -5 + k * 2.5
            segs.add(Rectangle(width=0.75, height=0.7, fill_color=TEAL, fill_opacity=0.6,
                               stroke_width=0).move_to([x0 + 0.375 + 1.75, -1.7, 0]))
        lab = ctext("εα=1.3：70% 时间单齿对承载（灰），30% 双齿对接力（青）", size=26,
                    color=ACCENT).next_to(bar, DOWN, buff=0.35)
        self.play(Create(bar), FadeIn(segs, lag_ratio=0.2), Write(lab))
        self.add(page_ref("孙桓八版 p206-208"))
        self.hold(3.5)


class S09_Summary(LessonScene):
    """小结（4min）+ 预告 L08：齿轮怎么造出来？小齿轮为什么会'根切'？"""

    def construct(self):
        self.header("第 7 讲小结")
        pts = bullets([
            "基本定律：公法线过节点 ⇔ 定传动比            (p197-199)",
            "渐开线五性质；inv α = tanα − α               (p199-200)",
            "d=mz；五参数身份证；齿厚=槽宽=πm/2           (p201-204)",
            "正确啮合：m、α 分别相等；i = z2/z1           (p204-206)",
            "重合度 εα≥1；越大越平稳                      (p206-208)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：一把'直线刀'如何切出弯曲的渐开线？根切与变位的攻防战",
                  size=30, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
