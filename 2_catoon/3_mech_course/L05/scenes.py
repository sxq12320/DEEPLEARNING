# -*- coding: utf-8 -*-
"""L05 四根杆的智慧（下）——连杆机构的设计（孙桓八版 §8-4~8-6, p139-160）

目标成片 80-90 min。核心：三类设计命题、按 K 设计图解全过程、刚化反转法、连杆曲线。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：上讲会'体检'，这讲学'定制'。三类设计命题总览（p139）：
    ①实现连架杆给定位置（函数生成）②实现连杆给定位置（刚体导引）③实现给定轨迹。"""

    def construct(self):
        big = ctext("第 5 讲   把机构'设计'出来", size=58, weight="BOLD")
        sub = ctext("给定要求 → 反求杆长", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        pts = bullets([
            "命题①  函数生成：两连架杆按给定对应角转动",
            "命题②  刚体导引：连杆整体经过几个指定位姿",
            "命题③  轨迹生成：连杆上某点走出指定曲线",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p139"))
        self.hold(3)


class S02_RigidBodyGuidance(LessonScene):
    """刚体导引：按连杆三个位置设计（15min，p139-141）。
    讲稿要点：连杆两端点 B、C 各经过三个位置 → B1B2B3 圆的圆心即固定铰链 A0
    （三点定圆：两条弦的垂直平分线交点）；C 端同理得 D0。动画：逐步作图。"""

    def construct(self):
        self.header("按连杆三位置设计", "垂直平分线的魔法 · p139-141")
        B = [np.array([-3.4, 1.5, 0]), np.array([-2.0, 2.0, 0]), np.array([-0.6, 1.9, 0])]
        C = [np.array([-1.6, 0.6, 0]), np.array([-0.3, 1.15, 0]), np.array([1.1, 1.25, 0])]
        planks = VGroup(*[
            VGroup(link_line(B[i], C[i], BLUE_B, w=5), pin_joint(B[i], 0.06), pin_joint(C[i], 0.06),
                   ctext(f"位置{i+1}", size=20, color=GREY_B).next_to(B[i], UP, buff=0.15))
            for i in range(3)
        ])
        self.play(FadeIn(planks, lag_ratio=0.4), run_time=2)
        t1 = ctext("要求：连杆 BC 依次通过三个指定位姿", size=28).to_edge(DOWN, buff=1.3)
        self.play(Write(t1))
        self.hold(2)

        def circumcenter(p1, p2, p3):
            ax_, ay = p1[0], p1[1]; bx, by = p2[0], p2[1]; cx, cy = p3[0], p3[1]
            d = 2 * (ax_ * (by - cy) + bx * (cy - ay) + cx * (ay - by))
            ux = ((ax_ ** 2 + ay ** 2) * (by - cy) + (bx ** 2 + by ** 2) * (cy - ay)
                  + (cx ** 2 + cy ** 2) * (ay - by)) / d
            uy = ((ax_ ** 2 + ay ** 2) * (cx - bx) + (bx ** 2 + by ** 2) * (ax_ - cx)
                  + (cx ** 2 + cy ** 2) * (bx - ax_)) / d
            return np.array([ux, uy, 0])

        A0 = circumcenter(*B); D0 = circumcenter(*C)
        bis1 = VGroup(Line((B[0] + B[1]) / 2 + (A0 - (B[0] + B[1]) / 2) * 1.6, A0, color=GOLD, stroke_width=2),
                      Line((B[1] + B[2]) / 2 + (A0 - (B[1] + B[2]) / 2) * 1.6, A0, color=GOLD, stroke_width=2))
        t2 = ctext("B 的三个位置定一个圆：两条弦的垂直平分线交点 = 圆心 = 固定铰链 A0",
                   size=27, color=ACCENT).to_edge(DOWN, buff=1.3)
        self.play(ReplacementTransform(t1, t2), Create(bis1))
        dA0 = VGroup(pin_joint(A0, 0.1), ground_hatch(A0 + DOWN * 0.18, 0.55))
        self.play(FadeIn(dA0))
        self.hold(2)
        bis2 = VGroup(Line((C[0] + C[1]) / 2 + (D0 - (C[0] + C[1]) / 2) * 1.6, D0, color=TEAL, stroke_width=2),
                      Line((C[1] + C[2]) / 2 + (D0 - (C[1] + C[2]) / 2) * 1.6, D0, color=TEAL, stroke_width=2))
        dD0 = VGroup(pin_joint(D0, 0.1), ground_hatch(D0 + DOWN * 0.18, 0.55))
        self.play(Create(bis2), FadeIn(dD0))
        links = VGroup(link_line(A0, B[0], GOLD), link_line(C[0], D0, TEAL))
        t3 = ctext("连上 A0B、CD0 —— 四杆机构诞生！（两位置时圆心可任选：无穷多解）", size=27,
                   color=GOOD).to_edge(DOWN, buff=1.3)
        self.play(ReplacementTransform(t2, t3), Create(links))
        self.add(page_ref("孙桓八版 p139-141"))
        self.hold(3)


class S03_DesignByK(LessonScene):
    """★按行程速比系数 K 设计曲柄摇杆（18min，p141-144，本讲核心图解）。
    讲稿要点：K→θ=180°(K−1)/(K+1)；给定摇杆长 CD 与摆角 ψ：作两极限位置 C1D、C2D；
    圆周角定理——对 C1C2 张角为 θ 的点在一段圆弧上 → 作辅助圆；A 在弧上任选（附加条件定唯一）；
    曲柄长 a=(AC1−AC2)/2? 注意 AC1=b+a、AC2=b−a → a=(AC1−AC2)/2, b=(AC1+AC2)/2。"""

    def construct(self):
        self.header("按 K 设计曲柄摇杆机构", "辅助圆图解法 · p141-144")
        steps0 = [
            MathTex(r"K \Rightarrow \theta = 180^\circ\frac{K-1}{K+1}", font_size=44, color=YELLOW),
        ]
        formula_reveal(self, steps0, anchor=UP * 2.4, wait=1.4)
        # 摇杆两极限位置
        D = np.array([1.8, -1.8, 0]); Lcd = 2.4; psi = np.deg2rad(55); base_ang = np.deg2rad(75)
        C1 = D + Lcd * np.array([np.cos(base_ang + psi / 2), np.sin(base_ang + psi / 2), 0])
        C2 = D + Lcd * np.array([np.cos(base_ang - psi / 2), np.sin(base_ang - psi / 2), 0])
        rock = VGroup(link_line(D, C1, TEAL), link_line(D, C2, TEAL).set_opacity(0.5),
                      pin_joint(D, 0.1), ground_hatch(D + DOWN * 0.18, 0.55),
                      Dot(C1, color=WHITE, radius=0.06), Dot(C2, color=WHITE, radius=0.06),
                      MathTex("C_1", font_size=30).next_to(C1, UP, buff=0.1),
                      MathTex("C_2", font_size=30).next_to(C2, RIGHT, buff=0.1))
        t1 = ctext("① 按给定摇杆长与摆角 ψ，画出两个极限位置 C1D、C2D", size=27).to_edge(DOWN, buff=1.25)
        self.play(FadeIn(rock), Write(t1))
        self.hold(2)
        # 辅助圆：对弦 C1C2 张角 θ 的圆
        theta = np.deg2rad(30)
        chord = C2 - C1; Lc = np.linalg.norm(chord)
        R = Lc / (2 * np.sin(theta))
        mid = (C1 + C2) / 2
        nvec = np.array([-chord[1], chord[0], 0]) / Lc
        Oc = mid - nvec * np.sqrt(max(R ** 2 - (Lc / 2) ** 2, 0))
        circ = Circle(radius=R, color=GOLD, stroke_width=2.5).move_to(Oc)
        t2 = ctext("② 圆周角定理：能对 C1C2 张角 θ 的点，都在这个辅助圆上！", size=27,
                   color=ACCENT).to_edge(DOWN, buff=1.25)
        self.play(ReplacementTransform(t1, t2), Create(circ))
        self.hold(2.2)
        ang_A = np.deg2rad(150)
        A = Oc + R * np.array([np.cos(ang_A), np.sin(ang_A), 0])
        dA = VGroup(pin_joint(A, 0.1), ground_hatch(A + DOWN * 0.18, 0.55),
                    MathTex("A", font_size=32).next_to(A, LEFT, buff=0.15))
        sight = VGroup(Line(A, C1, color=GREY_B, stroke_width=2), Line(A, C2, color=GREY_B, stroke_width=2))
        t3 = ctext("③ 在弧上选固定铰链 A（无穷多解——用最小传动角等附加条件挑）", size=27).to_edge(DOWN, buff=1.25)
        self.play(ReplacementTransform(t2, t3), FadeIn(dA), Create(sight))
        self.hold(2.2)
        f = [
            MathTex(r"AC_1 = b + a,\qquad AC_2 = b - a", font_size=42),
            MathTex(r"\Rightarrow\ a = \tfrac{AC_1 - AC_2}{2},\quad b = \tfrac{AC_1 + AC_2}{2}",
                    font_size=44, color=GREEN),
        ]
        self.play(FadeOut(t3))
        formula_reveal(self, f, anchor=DOWN * 3.0 + LEFT * 2.5, buff=0.3, wait=1.8)
        self.add(page_ref("孙桓八版 p141-144"))
        self.hold(3)


class S04_TwoPositionsFunc(LessonScene):
    """按两连架杆对应位置设计：刚化-反转法（12min，p144-147）。
    讲稿要点：给定主动杆两组转角 ↔ 从动杆两组转角；把机构在第二位置'刚化'，
    反转让从动杆回到第一位置 → 问题化为'定长杆端点轨迹' → 作图求 B 点。"""

    def construct(self):
        self.header("刚化 - 反转法", "函数生成设计的钥匙 · p144-147")
        idea = bullets([
            "① 目标：主动杆转 α12，从动杆按要求转 β12",
            "② 把第 2 位置整个机构'冻住'（刚化）",
            "③ 让全体绕 D 反转 −β12：从动杆回到位置 1",
            "④ 此时 B2 落到 B2'——而 CB 定长 ⇒ C1 在 B 点轨迹的圆上",
            "⑤ B1、B2' 的垂直平分线 ∩ 给定条件 → 铰链 C",
        ], size=28).shift(UP * 0.6)
        for i, row in enumerate(idea):
            self.play(FadeIn(row, shift=RIGHT * 0.35), run_time=0.8)
            self.hold(1.4)
        note = ctext("核心思想：相对运动不变——'我动你不动' 等价于 '你反着动我不动'", size=28,
                     color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p144-147"))
        self.hold(3)


class S05_CouplerCurves(LessonScene):
    """连杆曲线：轨迹综合（12min，p147/152 概念 + 视觉高潮）。
    讲稿要点：连杆上不同点画出千姿百态的连杆曲线（6 次代数曲线）；
    Hrones-Nelson 图谱：工程师翻'曲线字典'选机构；应用：搅拌轨迹、步行腿。
    动画：同一四杆机构连杆延长面上 5 个点同时描迹。"""

    def construct(self):
        self.header("连杆曲线", "一根杆上藏着一族曲线 · p147-152")
        fb = FourBar(3.4, 1.1, 2.9, 2.3, origin=np.array([-2.2, -1.6, 0]))
        th = ValueTracker(0.0)
        # 连杆延长面上的点：B + u*(AB方向) + v*(法向)
        params = [(0.5, 0.9, YELLOW), (0.3, 1.6, TEAL), (0.85, 1.3, ORANGE), (0.15, 0.5, GREEN), (0.6, 2.1, RED)]

        def coupler_pt(u, v, a):
            A0, A, B, B0 = fb.solve(a)
            e1 = (B - A) / np.linalg.norm(B - A)
            e2 = np.array([-e1[1], e1[0], 0])
            return A + u * np.linalg.norm(B - A) * e1 + v * e2

        def mech():
            a = th.get_value()
            A0, A, B, B0 = fb.solve(a)
            g = VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B), link_line(B, B0, TEAL),
                       *[pin_joint(p) for p in (A0, A, B, B0)],
                       ground_hatch(A0 + DOWN * 0.15, 0.5), ground_hatch(B0 + DOWN * 0.15, 0.5))
            for u, v, col in params:
                P = coupler_pt(u, v, a)
                g.add(Dot(P, color=col, radius=0.06),
                      Line(A, P, color=GREY_C, stroke_width=1.5), Line(B, P, color=GREY_C, stroke_width=1.5))
            return g

        def traces():
            a_now = th.get_value()
            g = VGroup()
            ts = np.linspace(max(a_now - TAU, 0), a_now, 90)
            for u, v, col in params:
                if len(ts) > 2:
                    pts = [coupler_pt(u, v, s) for s in ts]
                    vm = VMobject(color=col, stroke_width=2.5, stroke_opacity=0.85)
                    vm.set_points_smoothly(pts)
                    g.add(vm)
            return g

        self.play(FadeIn(always_redraw(mech)), FadeIn(always_redraw(traces)))
        t1 = ctext("连杆平面上每个点，都在画一条属于自己的六次曲线", size=28).to_edge(DOWN, buff=0.9)
        self.play(Write(t1))
        self.play(th.animate.set_value(2 * TAU + 0.2), run_time=12, rate_func=linear)
        t2 = ctext("Hrones-Nelson 图谱收录了 7000+ 条——工程师按需'查字典'选机构", size=27,
                   color=ACCENT).to_edge(DOWN, buff=0.25)
        self.play(Write(t2))
        self.add(page_ref("孙桓八版 p147-152"))
        self.hold(3)


class S06_Applications(LessonScene):
    """应用串讲 + 空间连杆一瞥（10min，p152-160）。
    讲稿要点：汽车转向梯形(双摇杆)、Jansen 步行腿(轨迹)、门式起重机(水平直线导引)；
    空间连杆 RSSR、万向节及双万向节等速条件（同平面+两轴夹角相等）。"""

    def construct(self):
        self.header("连杆机构的十八般武艺", "从汽车到机械兽 · p152-160")
        apps = bullets([
            "汽车转向梯形：双摇杆让内外轮转角自动配比（阿克曼原理）",
            "起重机四杆组合：吊钩近似水平直线移动——货物不'荡秋千'",
            "Jansen 机械兽：一条腿 = 一组多杆机构的轨迹设计",
            "空间连杆 RSSR：把运动'拐'到另一个平面",
        ], size=29).shift(UP * 0.9)
        self.play(FadeIn(apps, lag_ratio=0.45), run_time=3)
        self.hold(2.5)
        uj = bullets([
            "万向节：轴间夹角可变的传动（汽车传动轴）",
            "单万向节：瞬时传动比波动！ω2/ω1 随转角脉动",
            "双万向节等速条件：两端夹角相等 + 中间轴两叉面共面",
        ], size=29).shift(DOWN * 1.6)
        self.play(FadeIn(uj, lag_ratio=0.45), run_time=2.5)
        self.add(page_ref("孙桓八版 p157-160"))
        self.hold(3)


class S07_Summary(LessonScene):
    """小结（4min）+ 预告 L06：连杆只能'近似'实现轨迹；要'精确按剧本'运动？——凸轮登场。"""

    def construct(self):
        self.header("第 5 讲小结")
        pts = bullets([
            "三类命题：函数生成 / 刚体导引 / 轨迹生成      (p139)",
            "三位置刚体导引：垂直平分线交点定固定铰链      (p139-141)",
            "按 K 设计：θ → 辅助圆（圆周角定理）→ 选 A    (p141-144)",
            "刚化-反转法：相对运动等价变换                 (p144-147)",
            "连杆曲线族与图谱：轨迹设计的'查字典'          (p147-152)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：想让从动件严格按'剧本'走？——把剧本刻在凸轮上", size=30,
                  color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
