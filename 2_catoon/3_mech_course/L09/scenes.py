# -*- coding: utf-8 -*-
"""L09 齿轮的组合艺术——轮系及其传动比（孙桓八版 第11章, p237-253）

目标成片 85-95 min。核心：定轴轮系符号、转化机构法推导、复合轮系拆解、差速器、装配条件。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：一对齿轮 i≤8 左右；手表要 1:43200？风电要 1:100？——轮系。
    分类：定轴/周转(行星 F=1、差动 F=2)/复合。(p237-238)"""

    def construct(self):
        big = ctext("第 9 讲   齿轮的组合艺术", size=58, weight="BOLD")
        sub = ctext("轮系：传动比的乐高积木", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        pts = bullets([
            "定轴轮系：所有轮轴线都固定",
            "周转轮系：行星轮又自转又公转（有行星架 H）",
            "  — F=1 行星轮系 / F=2 差动轮系",
            "复合轮系：以上混搭（真实变速箱的常态）",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p237-238"))
        self.hold(3)


class S02_FixedAxis(LessonScene):
    """定轴轮系传动比（12min，p238-239）。
    讲稿要点：i1N = 各级连乘 = ∏从动齿数/∏主动齿数；转向：平面轮系 (−1)^m（m=外啮合次数），
    空间轮系只能画箭头；惰轮改向不改大小。动画：三级轮系逐级点亮计算。"""

    def construct(self):
        self.header("定轴轮系", "连乘 + 数负号 · p238-239")
        zs = [(12, GOLD), (24, TEAL), (18, BLUE_B), (36, ORANGE)]
        m0 = 0.16
        gears = VGroup()
        x = -5.4
        centers = []
        for z, col in zs:
            r = m0 * z / 2
            x += r
            g = gear_profile(m0, z, color=col).move_to([x, 0.6, 0])
            centers.append(np.array([x, 0.6, 0]))
            gears.add(g)
            x += r
        self.play(Create(gears, lag_ratio=0.2), run_time=2.5)
        # 演示旋转（速度按传动比）
        w1 = 1.6
        speeds = [w1, -w1 * 12 / 24, -w1 * 12 / 24, w1 * 12 / 24 * 18 / 36]
        anims = [Rotate(gears[i], speeds[i] * 4, about_point=centers[i]) for i in range(4)]
        self.play(*anims, run_time=5, rate_func=linear)
        f = [
            MathTex(r"i_{14} = \frac{\omega_1}{\omega_4} = (-1)^m\,\frac{z_2\,z_4}{z_1\,z_3}\quad(m=\text{外啮合对数})",
                    font_size=44, color=YELLOW),
            MathTex(r"= (-1)^2\frac{24\times 36}{12\times 18} = +4\ \ \text{(同向, 减速 4 倍)}", font_size=42, color=GREEN),
            MathTex(r"\text{惰轮: 只改转向不改大小; 空间轮系转向只能画箭头}", font_size=36),
        ]
        formula_reveal(self, f, anchor=DOWN * 1.9, buff=0.32, wait=1.9)
        self.add(page_ref("孙桓八版 p238-239"))
        self.hold(2.5)


class S03_PlanetaryIntuition(LessonScene):
    """周转轮系的困惑（8min，p239-240）。
    讲稿要点：行星轮既自转又公转——'它的转速到底是多少？'直接观察算不出传动比；
    动画：行星轮系运转（太阳轮-行星轮-内齿圈-行星架），行星轮上标箭头看它的复合运动。"""

    def construct(self):
        self.header("行星轮：又转又跑的'月亮'", "为什么直接算不了 · p239-240")
        z_s, z_p = 12, 9
        m0 = 0.22
        rs, rp = m0 * z_s / 2, m0 * z_p / 2
        ctr = np.array([-2.2, -0.3, 0])
        sun = gear_profile(m0, z_s, color=GOLD).move_to(ctr)
        ring_r = rs + 2 * rp
        ring = Circle(radius=ring_r + 0.16, color=GREY_B, stroke_width=6).move_to(ctr)
        thH = ValueTracker(0.0)

        def planet():
            aH = thH.get_value()
            pc = ctr + (rs + rp) * np.array([np.cos(aH), np.sin(aH), 0])
            # 行星自转角（对机架）：ω_p = -ωH*(zs/zp)*(...) 简化演示：转速比按啮合近似
            spin = -aH * (rs + rp) / rp
            g = gear_profile(m0, z_p, color=TEAL).rotate(spin).move_to(pc)
            arm = Line(ctr, pc, color=ORANGE, stroke_width=7)
            return VGroup(arm, g, pin_joint(pc, 0.07), pin_joint(ctr, 0.09))

        self.play(Create(sun), Create(ring), FadeIn(always_redraw(planet)))
        lab = VGroup(ctext("太阳轮", size=22, color=GOLD).next_to(ctr, DOWN + LEFT, buff=0.7),
                     ctext("行星架 H", size=22, color=ORANGE).next_to(ctr, UP + RIGHT, buff=1.3),
                     ctext("内齿圈", size=22, color=GREY_B).next_to(ring, UP, buff=0.15))
        self.play(FadeIn(lab))
        self.play(thH.animate.set_value(2 * TAU), run_time=8, rate_func=linear)
        q = ctext("行星轮轴线自己在动 → 定轴公式全失效。怎么办？", size=30, color=ACCENT).to_edge(DOWN, buff=0.6)
        self.play(Write(q))
        self.add(page_ref("孙桓八版 p239-240"))
        self.hold(2.5)


class S04_InversionMethod(LessonScene):
    """★转化机构法推导（15min，p240-241，全书最优雅的技巧之一）。
    讲稿要点：给整个轮系叠加 −ωH（坐到行星架上看）→ 行星架'静止' → 变成定轴轮系！
    i^H_1n = (ω1−ωH)/(ωn−ωH) = 定轴公式算；再解出所需未知转速。
    与 L05 刚化反转、L06 反转法同源：相对运动不变原理三连击。"""

    def construct(self):
        self.header("转化机构法", "坐到行星架上看世界 · p240-241")
        f = [
            MathTex(r"\text{全系统叠加公共角速度 } -\omega_H\ \text{（相对运动不变）}", font_size=40),
            MathTex(r"\omega_H^{H} = 0:\ \text{行星架静止} \Rightarrow \text{转化为定轴轮系!}", font_size=42, color=YELLOW),
            MathTex(r"i_{1n}^{H} = \frac{\omega_1 - \omega_H}{\omega_n - \omega_H} = \pm\frac{\prod z_{\text{从}}}{\prod z_{\text{主}}}",
                    font_size=48, color=GREEN),
            MathTex(r"\text{注意: } i^H \text{ 的正负按'转化后的定轴轮系'判定, 代入时带符号!}", font_size=36, color=RED),
        ]
        formula_reveal(self, f, anchor=UP * 0.6, buff=0.42, wait=2.0)
        ex = [
            MathTex(r"\text{例: 内齿圈固定 } (\omega_3=0):\ i_{13}^{H} = \frac{\omega_1-\omega_H}{0-\omega_H} = -\frac{z_3}{z_1}",
                    font_size=40),
            MathTex(r"\Rightarrow\ i_{1H} = \frac{\omega_1}{\omega_H} = 1 + \frac{z_3}{z_1}\quad\text{（行星减速器公式）}",
                    font_size=44, color=GREEN),
        ]
        formula_reveal(self, ex, anchor=DOWN * 2.2, buff=0.3, wait=2.0)
        self.add(page_ref("孙桓八版 p240-241"))
        self.hold(3)


class S05_CompoundTrains(LessonScene):
    """复合轮系拆解（12min，p242-243）。
    讲稿要点：拆解口诀——先找行星轮（轴线动的轮），行星轮+支承它的行星架+
    与行星轮啮合的中心轮 = 一个周转单元；其余为定轴部分；分别列式再联立。"""

    def construct(self):
        self.header("复合轮系：先找行星架！", "拆解三步法 · p242-243")
        steps = bullets([
            "① 找行星轮：哪个轮的轴线在'跑'？",
            "② 圈出它的周转单元：行星轮 + 行星架 + 相啮合的中心轮",
            "③ 剩下的都是定轴部分；分别列式 → 用公共构件转速联立",
        ], size=30).shift(UP * 1.2)
        for row in steps:
            self.play(FadeIn(row, shift=RIGHT * 0.4), run_time=0.8)
            self.hold(1.6)
        warn = bullets([
            "常见错误①：把整个复合轮系直接套一个转化公式（不同行星架不能共用！）",
            "常见错误②：i^H 的 ± 号漏判（转化轮系里照样数外啮合次数）",
        ], size=28, marker="⚠ ").shift(DOWN * 1.4)
        self.play(FadeIn(warn, lag_ratio=0.5), run_time=2)
        self.add(page_ref("孙桓八版 p242-243"))
        self.hold(3.5)


class S06_Differential(LessonScene):
    """★汽车差速器（12min，p243-246 轮系功用之'运动合成分解'，全课最实用高潮）。
    讲稿要点：差动轮系 F=2：两个太阳轮(左右半轴)+行星架(被发动机驱动)；
    锥齿差动关系 ω左+ω右=2ωH；直行两轮同速、转弯自动分配——纯机械的'智能'。"""

    def construct(self):
        self.header("差速器：纯机械的智能", "ω左 + ω右 = 2ωH · p243-246")
        f = [
            MathTex(r"\text{锥齿差动轮系: } i_{12}^{H} = \frac{\omega_L-\omega_H}{\omega_R-\omega_H} = -1",
                    font_size=44),
            MathTex(r"\Rightarrow\ \boxed{\omega_L + \omega_R = 2\,\omega_H}", font_size=52, color=YELLOW),
        ]
        formula_reveal(self, f, anchor=UP * 1.6, buff=0.4, wait=2.0)
        # 直行 vs 转弯的车轮转速演示
        wheelL = Circle(radius=0.55, color=TEAL, stroke_width=6).shift(LEFT * 3.3 + DOWN * 1.6)
        wheelR = Circle(radius=0.55, color=GOLD, stroke_width=6).shift(RIGHT * 3.3 + DOWN * 1.6)
        mkL = Line(wheelL.get_center(), wheelL.get_center() + UP * 0.55, color=WHITE, stroke_width=3)
        mkR = Line(wheelR.get_center(), wheelR.get_center() + UP * 0.55, color=WHITE, stroke_width=3)
        axle = Line(wheelL.get_center(), wheelR.get_center(), color=GREY_C, stroke_width=3)
        t1 = ctext("直行：ωL = ωR = ωH", size=28, color=GOOD).to_edge(DOWN, buff=0.5)
        self.play(Create(wheelL), Create(wheelR), Create(axle), Create(mkL), Create(mkR), Write(t1))
        self.play(Rotate(mkL, TAU, about_point=wheelL.get_center()),
                  Rotate(mkR, TAU, about_point=wheelR.get_center()), run_time=3, rate_func=linear)
        t2 = ctext("右转弯：外(左)轮加速、内(右)轮减速——总和不变，自动分配！", size=28,
                   color=ACCENT).to_edge(DOWN, buff=0.5)
        self.play(ReplacementTransform(t1, t2))
        self.play(Rotate(mkL, TAU * 1.4, about_point=wheelL.get_center()),
                  Rotate(mkR, TAU * 0.6, about_point=wheelR.get_center()), run_time=3.5, rate_func=linear)
        self.add(page_ref("孙桓八版 p243-246"))
        self.hold(3)


class S07_DesignConditions(LessonScene):
    """行星轮系设计四条件（10min，p248-253）。
    讲稿要点：①传动比条件 z3=z1(i1H−1)？按公式配齿 ②同心条件 z3=z1+2z2
    ③装配条件 (z1+z3)/k=整数（k 个行星轮均布）推导 ④邻接条件（相邻行星轮不打架）。"""

    def construct(self):
        self.header("行星轮系设计四关", "配齿不是随便凑 · p248-253")
        f = [
            MathTex(r"\text{① 传动比: } z_3 = (i_{1H}-1)\,z_1", font_size=40),
            MathTex(r"\text{② 同心: } z_3 = z_1 + 2z_2\ \text{（太阳+2行星=内齿圈半径）}", font_size=40),
            MathTex(r"\text{③ 装配: } \frac{z_1+z_3}{k} = \text{整数}\ \text{（k 个行星均布进得去）}",
                    font_size=42, color=YELLOW),
            MathTex(r"\text{④ 邻接: } (z_1+z_2)\sin\frac{\pi}{k} > z_2 + 2h_a^{*}\ \text{（相邻行星不相碰）}",
                    font_size=40),
        ]
        formula_reveal(self, f, anchor=UP * 0.4, buff=0.42, wait=1.9)
        note = ctext("装配条件不满足 → 第 2 个行星轮物理上装不进齿——动画里让它'卡'给你看",
                     size=26, color=ACCENT).to_edge(DOWN, buff=0.55)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p248-253"))
        self.hold(3)


class S08_Applications(LessonScene):
    """轮系功用总览（8min，p243-246）：大传动比/换向/变速/分路/合成分解/大功率分流。
    案例：手表(定轴)、机床变速箱(滑移齿轮)、风电增速箱(行星+平行轴)、轮毂减速。"""

    def construct(self):
        self.header("轮系的六大本领", "p243-246")
        pts = bullets([
            "① 大传动比：行星系单级可达 i>100（风电增速箱反向用）",
            "② 变速换向：机床滑移齿轮 / 汽车手动挡",
            "③ 运动合成：差速器 ω_L+ω_R=2ω_H",
            "④ 运动分解：差速器转弯自动分配",
            "⑤ 大功率分流：k 个行星轮均担载荷（航空减速器）",
            "⑥ 结构紧凑同轴：输入输出同轴线（AT 变速箱行星排）",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3.5)
        self.add(page_ref("孙桓八版 p243-246"))
        self.hold(4)


class S09_Summary(LessonScene):
    """小结（4min）+ 预告 L10：轮系把运动传得漂亮，但摩擦在偷功率——效率与自锁。"""

    def construct(self):
        self.header("第 9 讲小结")
        pts = bullets([
            "定轴：i = (−1)^m ∏z从/∏z主                  (p238-239)",
            "转化机构法：i^H = (ω1−ωH)/(ωn−ωH)            (p240-241)",
            "行星减速: i1H = 1 + z3/z1                    (p241)",
            "复合轮系：先找行星架，分单元列式联立            (p242-243)",
            "差速器 ωL+ωR=2ωH；设计四条件（配齿）           (p243-253)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：机器中的暗力量——摩擦如何偷走功率，又如何被我们利用？",
                  size=30, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
