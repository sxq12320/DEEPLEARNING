# -*- coding: utf-8 -*-
"""L03 让机构的速度看得见——平面机构的运动分析（孙桓八版 第3章, p35-54）

目标成片 85-95 min。核心推导：K=N(N−1)/2、三心定理、哥氏加速度、解析法求导链。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：牛头刨床为什么'去慢回快'？答案藏在速度分析里。三类方法预览：
    瞬心法(求速度快)/图解法(几何直觉)/解析法(计算机时代主力)。p35"""

    def construct(self):
        big = ctext("第 3 讲   让速度看得见", size=60, weight="BOLD")
        sub = ctext("速度瞬心 · 图解法 · 解析法", size=34, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        m = bullets([
            "瞬心法：一眼看出速度比（几何魔法）",
            "图解法：矢量多边形，手绘时代的智慧",
            "解析法：位置方程求导——仿真软件的心脏",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(m, lag_ratio=0.4))
        self.hold(3)


class S02_InstantCenter(LessonScene):
    """速度瞬心定义与数目（10min，p35-36）。
    讲稿要点：两构件瞬时速度相同的重合点=瞬心 P；绝对瞬心(一构件为机架) vs 相对瞬心；
    N 个构件的机构瞬心数 K=N(N−1)/2（组合数）。动画：转动圆盘上各点速度场 → 圆心速度为零=绝对瞬心。"""

    def construct(self):
        self.header("速度瞬心", "瞬时的'共同不动点' · p35-36")
        disk = Circle(radius=1.4, color=BLUE_B).shift(LEFT * 3.5)
        ctr = Dot(disk.get_center(), color=YELLOW, radius=0.08)
        arrows = VGroup()
        for ang in np.linspace(0, TAU, 8, endpoint=False):
            p = disk.get_center() + 1.0 * np.array([np.cos(ang), np.sin(ang), 0])
            v = 0.6 * np.array([-np.sin(ang), np.cos(ang), 0])
            arrows.add(Arrow(p, p + v, buff=0, color=TEAL, stroke_width=4, max_tip_length_to_length_ratio=0.25))
        self.play(Create(disk), FadeIn(ctr))
        self.play(FadeIn(arrows, lag_ratio=0.1))
        t1 = ctext("绕定轴转动：圆心速度 = 0 → 它就是（绝对）瞬心", size=28).to_edge(DOWN, buff=1.3)
        self.play(Write(t1))
        self.hold(2)
        steps = [
            MathTex(r"\text{瞬心 } P_{ij}:\ \text{构件 } i,j\ \text{上速度相同的重合点}", font_size=40),
            MathTex(r"N \text{ 个构件两两一个瞬心：} K = \binom{N}{2} = \frac{N(N-1)}{2}",
                    font_size=44, color=YELLOW),
            MathTex(r"\text{四杆机构 } N=4\ \Rightarrow\ K = 6", font_size=40, color=GREEN),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 2.4 + UP * 0.4, wait=1.6)
        self.add(page_ref("孙桓八版 p35-36"))
        self.hold(2.5)


class S03_KennedyTheorem(LessonScene):
    """★三心定理及证明（12min，p36-37）。
    讲稿要点：作平面运动的三个构件，其三个瞬心必在同一直线上。
    反证法动画：设 P13 不在 P12-P23 连线上，则该点作为构件1、3 的公共点，
    其速度方向分别垂直于 P12-P13 与 P23-P13 —— 两方向不可能相同 → 矛盾。"""

    def construct(self):
        self.header("三心定理（Kennedy）", "三个瞬心必共线 · p36-37")
        P12 = np.array([-3.2, -0.6, 0]); P23 = np.array([2.6, -0.6, 0]); P13_wrong = np.array([-0.3, 1.6, 0])
        d12 = Dot(P12, color=GOLD, radius=0.09); l12 = MathTex("P_{12}", font_size=36).next_to(d12, DOWN)
        d23 = Dot(P23, color=TEAL, radius=0.09); l23 = MathTex("P_{23}", font_size=36).next_to(d23, DOWN)
        base = Line(P12, P23, color=GREY_B)
        self.play(FadeIn(d12), FadeIn(d23), Write(l12), Write(l23), Create(base))
        t0 = ctext("反证：假设 P13 不在这条线上——", size=30).to_edge(DOWN, buff=1.3)
        d13 = Dot(P13_wrong, color=RED, radius=0.09); l13 = MathTex("P_{13}?", font_size=36, color=RED).next_to(d13, UP)
        self.play(Write(t0), FadeIn(d13), Write(l13))
        self.hold(1.5)
        # 两个矛盾的速度方向
        r1 = Line(P12, P13_wrong, color=GOLD, stroke_width=2)
        r2 = Line(P23, P13_wrong, color=TEAL, stroke_width=2)
        v1_dir = P13_wrong - P12; v1_perp = np.array([-v1_dir[1], v1_dir[0], 0]); v1_perp /= np.linalg.norm(v1_perp)
        v2_dir = P13_wrong - P23; v2_perp = np.array([-v2_dir[1], v2_dir[0], 0]); v2_perp /= np.linalg.norm(v2_perp)
        a1 = Arrow(P13_wrong, P13_wrong + v1_perp * 1.1, buff=0, color=GOLD)
        a2 = Arrow(P13_wrong, P13_wrong + v2_perp * 1.1, buff=0, color=TEAL)
        t1 = ctext("该点速度：作为构件2上的点 ⊥ P12 连线（金）；作为构件2上的点 ⊥ P23 连线（青）",
                   size=26).to_edge(DOWN, buff=1.3)
        self.play(Create(r1), Create(r2), ReplacementTransform(t0, t1))
        self.play(GrowArrow(a1), GrowArrow(a2))
        self.hold(2)
        t2 = ctext("同一点两个不同速度方向？矛盾！→ P13 只能在 P12P23 连线上", size=28,
                   color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(t2))
        self.play(d13.animate.move_to(P12 * 0.45 + P23 * 0.55), FadeOut(a1), FadeOut(a2),
                  FadeOut(r1), FadeOut(r2), l13.animate.become(
                      MathTex("P_{13}", font_size=36, color=GREEN).next_to(P12 * 0.45 + P23 * 0.55, UP)))
        self.add(page_ref("孙桓八版 p36-37"))
        self.hold(3)


class S04_ICExample(LessonScene):
    """瞬心法应用：四杆机构 6 瞬心逐个定位 + 求传动比（12min，p37-39）。
    讲稿要点：4 个铰链=4 个直观瞬心；P13、P24 用三心定理两条线相交定出；
    ω2/ω4 = P14P24 / P12P24（相对瞬心分线段反比）。"""

    def construct(self):
        self.header("瞬心法实战", "四杆机构的 6 个瞬心 · p37-39")
        fb = FourBar(4.0, 1.4, 3.2, 2.6, origin=np.array([-2.6, -1.4, 0]))
        A0, A, B, B0 = fb.solve(1.0)
        mech = VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B), link_line(B, B0, TEAL),
                      *[pin_joint(p) for p in (A0, A, B, B0)],
                      ground_hatch(A0 + DOWN * 0.16, 0.6), ground_hatch(B0 + DOWN * 0.16, 0.6))
        self.play(FadeIn(mech))
        labels = VGroup(
            MathTex("P_{12}", font_size=30).next_to(A0, DOWN + LEFT, buff=0.1),
            MathTex("P_{23}", font_size=30).next_to(A, UP, buff=0.15),
            MathTex("P_{34}", font_size=30).next_to(B, UP, buff=0.15),
            MathTex("P_{14}", font_size=30).next_to(B0, DOWN + RIGHT, buff=0.1),
        )
        t1 = ctext("① 4 个铰链，就是 4 个现成的瞬心", size=28).to_edge(DOWN, buff=1.25)
        self.play(FadeIn(labels, lag_ratio=0.3), Write(t1))
        self.hold(2)
        # P13: 延长 A0A 与 B0B 交点；P24: 延长 AB 与 A0B0 交点
        def line_inter(p1, d1, p2, d2):
            Amat = np.array([[d1[0], -d2[0]], [d1[1], -d2[1]]])
            b = (p2 - p1)[:2]
            t = np.linalg.solve(Amat, b)
            return p1 + t[0] * d1

        P13 = line_inter(A0, A - A0, B0, B - B0)
        P24 = line_inter(A, B - A, A0, B0 - A0)
        ext1 = VGroup(Line(A0, P13, color=GOLD, stroke_width=2).set_opacity(0.6),
                      Line(B0, P13, color=TEAL, stroke_width=2).set_opacity(0.6))
        d13 = Dot(P13, color=YELLOW, radius=0.08); l13 = MathTex("P_{13}", font_size=30, color=YELLOW).next_to(d13, UP)
        t2 = ctext("② P13：三心定理×2 → 两条连线的交点（延长两连架杆）", size=28,
                   color=ACCENT).to_edge(DOWN, buff=1.25)
        self.play(ReplacementTransform(t1, t2), Create(ext1), FadeIn(d13), Write(l13))
        self.hold(2)
        ext2 = VGroup(Line(A, P24, color=BLUE_B, stroke_width=2).set_opacity(0.6),
                      Line(A0, P24, color=GREY_B, stroke_width=2).set_opacity(0.6))
        d24 = Dot(P24, color=ORANGE, radius=0.08); l24 = MathTex("P_{24}", font_size=30, color=ORANGE).next_to(d24, DOWN)
        self.play(Create(ext2), FadeIn(d24), Write(l24))
        self.hold(2)
        ratio = MathTex(r"\frac{\omega_2}{\omega_4} \;=\; \frac{\overline{P_{14}P_{24}}}{\overline{P_{12}P_{24}}}",
                        font_size=48, color=YELLOW).to_corner(UP + RIGHT, buff=0.8).shift(DOWN * 0.9)
        t3 = ctext("③ 传动比 = 相对瞬心到两绝对瞬心的距离反比", size=28, color=GOOD).to_edge(DOWN, buff=0.6)
        self.play(Write(ratio), ReplacementTransform(t2, t3))
        self.add(page_ref("孙桓八版 p37-39"))
        self.hold(3)


class S05_VelocityPolygon(LessonScene):
    """图解法：速度多边形（12min，p39-41）。
    讲稿要点：同一构件两点速度关系 vB = vA + vBA（vBA ⊥ AB，大小 ω·lAB）；
    取极点 p 作速度图；速度影像原理。动画：曲柄滑块速度多边形逐矢量生长。"""

    def construct(self):
        self.header("图解法：速度多边形", "矢量方程的几何解 · p39-41")
        eq = MathTex(r"\vec{v}_B = \vec{v}_A + \vec{v}_{BA},\quad v_{BA} = \omega\,l_{AB},\ \perp AB",
                     font_size=44).to_edge(UP, buff=1.4)
        self.play(Write(eq))
        self.hold(2)
        cs = CrankSlider(1.1, 3.0, 0, origin=np.array([-4.4, -1.0, 0]))
        th0 = 1.0
        O, A, B = cs.solve(th0)
        mech = VGroup(link_line(O, A, GOLD), link_line(A, B, BLUE_B), slider_block(B, 0.6, 0.36),
                      pin_joint(A), pin_joint(B), fixed_pin(O))
        self.play(FadeIn(mech))
        # 数值速度（ω=1）：vA ⊥ OA；vB 沿导路；vBA ⊥ AB
        w = 1.0
        vA = w * 1.1 * np.array([-np.sin(th0), np.cos(th0), 0])
        # 解析：vB = vA + ω3 × AB, 求 ω3 使 vB 沿 x
        ABv = B - A
        # vB_y = vA_y + w3*ABx = 0 -> w3 = -vA_y/ABx
        w3 = -vA[1] / ABv[0]
        vBA = w3 * np.array([-ABv[1], ABv[0], 0])
        vB = vA + vBA
        pole = np.array([2.6, -0.4, 0])
        pA = Arrow(pole, pole + vA * 1.6, buff=0, color=GOLD)
        pBA = Arrow(pole + vA * 1.6, pole + (vA + vBA) * 1.6, buff=0, color=BLUE_B)
        pB = Arrow(pole, pole + vB * 1.6, buff=0, color=ORANGE)
        pl = MathTex("p", font_size=34).next_to(pole, DOWN + LEFT, buff=0.08)
        t1 = ctext("从极点 p 出发：先画已知 vA（⊥曲柄）", size=27).to_edge(DOWN, buff=1.2)
        self.play(FadeIn(pl), Write(t1))
        self.play(GrowArrow(pA)); self.hold(1.5)
        t2 = ctext("过 a 作 ⊥AB 方向线（vBA 方向已知大小未知）", size=27).to_edge(DOWN, buff=1.2)
        self.play(ReplacementTransform(t1, t2), GrowArrow(pBA)); self.hold(1.5)
        t3 = ctext("过 p 作导路方向线 → 交点即 b：vB 被'夹'出来了！", size=27, color=ACCENT).to_edge(DOWN, buff=1.2)
        self.play(ReplacementTransform(t2, t3), GrowArrow(pB)); self.hold(2)
        img = ctext("速度影像：速度图中 △pab ∽ 机构中对应构件图形（同构件可直接放缩取点）",
                    size=25, color=GREY_B).to_edge(DOWN, buff=0.45)
        self.play(Write(img))
        self.add(page_ref("孙桓八版 p39-41"))
        self.hold(3)


class S06_Coriolis(LessonScene):
    """★哥氏加速度（12min，p41-43）。
    讲稿要点：两构件重合点（有相对滑动+牵连转动）加速度关系中多出一项 a^k = 2ω×v_r；
    直观演示：旋转平台上沿径向匀速走的小球，绝对轨迹是螺线——横向不断被'带偏'即哥氏项；
    方向判定：v_r 顺牵连 ω 转 90°。应用：导杆机构必考。"""

    def construct(self):
        self.header("哥氏加速度", "旋转系里多出来的一项 · p41-43")
        eq = MathTex(r"\vec a_{B_2} = \vec a_{B_1} + \vec a_{B_2B_1}^{\,r} + \underbrace{2\vec\omega_1\times\vec v_{B_2B_1}}_{\text{哥氏 } a^k}",
                     font_size=42).to_edge(UP, buff=1.35)
        self.play(Write(eq))
        self.hold(2)
        # 旋转平台 + 径向走点 → 螺线
        ctr = np.array([-2.8, -1.0, 0])
        plat = Circle(radius=2.0, color=GREY_B).move_to(ctr)
        tv = ValueTracker(0.0)

        def ball():
            t = tv.get_value()
            r = 0.25 + 0.28 * t
            a = 0.9 * t
            return Dot(ctr + r * np.array([np.cos(a), np.sin(a), 0]), color=YELLOW, radius=0.09)

        def trace():
            t = tv.get_value()
            ts = np.linspace(0, max(t, 1e-3), 60)
            pts = [ctr + (0.25 + 0.28 * s) * np.array([np.cos(0.9 * s), np.sin(0.9 * s), 0]) for s in ts]
            vm = VMobject(color=TEAL, stroke_width=3)
            vm.set_points_smoothly(pts)
            return vm

        b = always_redraw(ball); tr = always_redraw(trace)
        t1 = ctext("平台匀转 + 小球沿径向匀速外爬 → 绝对轨迹是螺线", size=27).to_edge(DOWN, buff=1.2)
        self.play(Create(plat), FadeIn(b), FadeIn(tr), Write(t1))
        self.play(tv.animate.set_value(6.0), run_time=6, rate_func=linear)
        t2 = ctext("横向速度一直在变 → 需要横向加速度 = 哥氏项 2ω·v_r", size=28, color=ACCENT).to_edge(DOWN, buff=1.2)
        self.play(ReplacementTransform(t1, t2))
        self.hold(2)
        rule = ctext("方向口诀：把相对速度 v_r 顺着牵连角速度 ω 的转向转 90°", size=28,
                     color=GOOD).to_edge(DOWN, buff=0.55)
        mag = MathTex(r"a^k = 2\,\omega\, v_r", font_size=48, color=YELLOW).to_edge(DOWN, buff=0.02)
        self.play(Write(rule)); self.hold(1.5)
        self.play(Write(mag))
        self.add(page_ref("孙桓八版 p41-43"))
        self.hold(3)


class S07_Analytical(LessonScene):
    """解析法（12min，p43-46）。
    讲稿要点：建立封闭矢量方程 → 投影得位置方程 → 对 t 求导得速度/加速度；
    曲柄滑块全推导：x_B = r cosθ + l cos ψ, r sinθ = l sinψ；说明'仿真软件就是把这件事自动化'。"""

    def construct(self):
        self.header("解析法", "位置方程 → 求导 → 一切 · p43-46")
        steps = [
            MathTex(r"\text{封闭矢量方程: } \vec{r}_{OA} + \vec{r}_{AB} = \vec{r}_{OB}", font_size=40),
            MathTex(r"x:\ r\cos\theta + l\cos\psi = x_B", font_size=40),
            MathTex(r"y:\ r\sin\theta - l\sin\psi = 0\ \Rightarrow\ \sin\psi = \tfrac{r}{l}\sin\theta", font_size=40),
            MathTex(r"x_B = r\cos\theta + l\sqrt{1 - (\tfrac{r}{l}\sin\theta)^2}", font_size=42, color=YELLOW),
            MathTex(r"v_B = \dot{x}_B = \frac{dx_B}{d\theta}\,\omega,\qquad a_B = \dot{v}_B", font_size=40, color=GREEN),
        ]
        formula_reveal(self, steps, anchor=UP * 0.3, buff=0.38, wait=1.7)
        note = ctext("每一款机构仿真软件的心脏，就是这三行求导", size=28, color=GREY_B).to_edge(DOWN, buff=0.5)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p44-46"))
        self.hold(3)


class S08_CurvesCrossCheck(LessonScene):
    """解析结果曲线 + 与图解法互验（8min）。
    动画：曲柄滑块运转，同步绘制 x_B(θ)、v_B(θ)、a_B(θ) 三条曲线；
    强调 r/l 越大越偏离简谐（连杆效应）。正确性自校：θ=90° 处 v 极值等特征点核对。"""

    def construct(self):
        self.header("把整圈速度画出来", "解析法的红利：全周期曲线")
        r, l, w = 1.0, 3.0, 1.0
        ax = Axes(x_range=[0, TAU, PI / 2], y_range=[-2.2, 2.2, 1], x_length=7.2, y_length=4.2,
                  axis_config={"include_tip": False, "font_size": 22}).shift(RIGHT * 2.2 + DOWN * 0.3)
        xl = MathTex(r"\theta", font_size=30).next_to(ax.x_axis, RIGHT, buff=0.15)
        def xB(th): return r * np.cos(th) + np.sqrt(l ** 2 - (r * np.sin(th)) ** 2)
        def vB(th):
            s = r * np.sin(th)
            return -r * np.sin(th) * w - (r ** 2 * np.sin(th) * np.cos(th) * w) / np.sqrt(l ** 2 - s ** 2)
        curve_x = ax.plot(lambda t: xB(t) - l, x_range=[0, TAU], color=BLUE_B)
        curve_v = ax.plot(vB, x_range=[0, TAU], color=ORANGE)
        leg = VGroup(ctext("位移 x_B（蓝） / 速度 v_B（橙）", size=24)).next_to(ax, UP, buff=0.2)
        cs = CrankSlider(r, l, 0, origin=np.array([-5.2, -0.6, 0]))
        th = ValueTracker(0.0)

        def mech():
            O, A, B = cs.solve(th.get_value())
            return VGroup(link_line(O, A, GOLD), link_line(A, B, BLUE_B),
                          slider_block(B, 0.55, 0.34), pin_joint(A), fixed_pin(O))

        dot_v = always_redraw(lambda: Dot(ax.c2p(th.get_value() % TAU, vB(th.get_value() % TAU)),
                                          color=ORANGE, radius=0.07))
        self.play(Create(ax), FadeIn(xl), FadeIn(leg), FadeIn(always_redraw(mech)))
        self.play(Create(curve_x), Create(curve_v), run_time=2.5)
        self.play(FadeIn(dot_v))
        self.play(th.animate.set_value(2 * TAU), run_time=8, rate_func=linear)
        note = ctext("自校：θ=0/π 时 v=0（死点）✓；曲线非正弦 = 连杆效应（r/l 越大越明显）",
                     size=25, color=GOOD).to_edge(DOWN, buff=0.25)
        self.play(Write(note))
        self.hold(3)


class S09_Summary(LessonScene):
    """小结（4min）+ 预告：会算速度了，下一站——四杆机构的'性格'（L04 连杆机构）。"""

    def construct(self):
        self.header("第 3 讲小结")
        pts = bullets([
            "瞬心：K = N(N−1)/2；三心定理定共线            (p35-37)",
            "瞬心法求传动比：相对瞬心分线段成反比            (p37-39)",
            "图解法：vB = vA + vBA 矢量多边形；速度影像      (p39-41)",
            "哥氏加速度 a^k = 2ω·v_r，方向 v_r 顺 ω 转 90°  (p41-43)",
            "解析法：位置方程求导——仿真软件的心脏           (p43-46)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：四根杆，凭什么撑起半个机械世界？", size=32, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
