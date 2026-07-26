# -*- coding: utf-8 -*-
"""L02 机构能不能动、怎么动？——自由度与机构组成原理（孙桓八版 §2-4~2-7, p14-24）

目标成片 90-100 min。核心：F = 3n − 2P_L − P_H 推导 + 三大陷阱 + 杆组。
渲染:  manim -pqh scenes.py <SceneName>
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：上讲回顾 30 秒 + 本讲问题：'给你一张简图，它能不能动？要装几个电机？'
    这是全书第一个'可计算'的问题——自由度 F。"""

    def construct(self):
        big = ctext("第 2 讲   机构能不能动？", size=60, weight="BOLD")
        sub = ctext("自由度 F —— 机械原理的第一个'超能力'", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub, shift=UP * 0.3))
        self.hold(2.5)
        qs = bullets([
            "这张简图，是机构还是死结构？",
            "它需要几个电机（原动件）驱动？",
            "为什么有的'机构'装上电机就卡死/乱动？",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(qs, lag_ratio=0.4))
        self.hold(3)


class S02_DetermineCondition(LessonScene):
    """机构具有确定运动的条件（10min，p14-16）。
    讲稿要点：自由度 F = 机构相对机架的独立运动数；四种情形——
    F≤0 刚性桁架不能动；原动件数=F 确定运动；<F 运动不确定(乱动)；>F 最薄弱处破坏。
    最小阻力定律一句带过(p16)。"""

    def construct(self):
        self.header("机构具有确定运动的条件", "原动件数 vs 自由度 · p14-16")
        table = VGroup(
            ctext("F ≤ 0           →  不能动（刚性桁架）", size=30, color=BAD),
            ctext("原动件数 = F    →  运动确定  ✔", size=30, color=GOOD),
            ctext("原动件数 < F    →  运动不确定（乱动）", size=30, color=ORANGE),
            ctext("原动件数 > F    →  最薄弱环节被破坏", size=30, color=BAD),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(UP * 0.3)
        for row in table:
            self.play(FadeIn(row, shift=RIGHT * 0.4), run_time=0.8)
            self.hold(1.4)
        concl = ctext("所以：先会算 F，才能谈设计", size=34, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(concl))
        self.add(page_ref("孙桓八版 p14-16"))
        self.hold(2.5)


class S03_FormulaDerivation(LessonScene):
    """★核心推导：F = 3n − 2P_L − P_H（15min，p16）。
    讲稿要点：平面内一个自由构件 3 个自由度(x,y,φ)；n 个活动构件共 3n；
    每个低副(转动/移动)引入 2 个约束；每个高副引入 1 个约束(只约束公法线方向移动)；
    逐项扣除得公式。动画：一根杆的 3 个自由度演示 → 逐步加副扣自由度。"""

    def construct(self):
        self.header("自由度公式推导", "F = 3n − 2P_L − P_H · p16")
        bar = link_line(LEFT * 1.2, RIGHT * 1.2, BLUE_B).shift(LEFT * 3.6 + UP * 1.2)
        self.play(Create(bar))
        # 平面 3 自由度演示
        self.play(bar.animate.shift(RIGHT * 0.8), run_time=0.5)
        self.play(bar.animate.shift(LEFT * 0.8), run_time=0.5)
        self.play(bar.animate.shift(UP * 0.5), run_time=0.5)
        self.play(bar.animate.shift(DOWN * 0.5), run_time=0.5)
        self.play(Rotate(bar, PI / 5), run_time=0.6)
        self.play(Rotate(bar, -PI / 5), run_time=0.6)
        t0 = ctext("平面自由构件：x 移动 + y 移动 + 转动 = 3 个自由度", size=28,
                   color=ACCENT).next_to(bar, DOWN, buff=0.6)
        self.play(Write(t0))
        self.hold(2)
        steps = [
            MathTex(r"n\ \text{个活动构件（机架不算！）}\ \Rightarrow\ 3n", font_size=42),
            MathTex(r"\text{每个低副（转动/移动）：约束 } 2\ \Rightarrow\ -2P_L", font_size=42),
            MathTex(r"\text{每个高副：只约束公法线方向}\ \Rightarrow\ -P_H", font_size=42),
            MathTex(r"\boxed{\,F = 3n - 2P_L - P_H\,}", font_size=60, color=YELLOW),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 2.2 + DOWN * 0.6, buff=0.45, wait=1.8)
        self.add(page_ref("孙桓八版 p16 式(2-1)"))
        self.hold(3)


class S04_BasicExamples(LessonScene):
    """标准算例两连发（10min）。
    ① 铰链四杆：n=3, PL=4, PH=0 → F=1；② 曲柄滑块：n=3, PL=4(3转+1移), PH=0 → F=1；
    ③ 凸轮机构：n=2, PL=2, PH=1 → F=1。都=1：一个电机即可确定运动。"""

    def construct(self):
        self.header("算例：三大常用机构的 F", "都等于 1 不是巧合")
        fb = FourBar(3.4, 1.2, 2.9, 2.4, origin=np.array([-5.6, -0.6, 0]))
        A0, A, B, B0 = fb.solve(0.9)
        mech1 = VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B), link_line(B, B0, TEAL),
                       *[pin_joint(p) for p in (A0, A, B, B0)],
                       ground_hatch(A0 + DOWN * 0.16, 0.6), ground_hatch(B0 + DOWN * 0.16, 0.6))
        f1 = MathTex(r"n=3,\ P_L=4,\ P_H=0", font_size=34).next_to(mech1, UP, buff=0.4)
        f1b = MathTex(r"F=3\times3-2\times4-0=1", font_size=36, color=GREEN).next_to(mech1, DOWN, buff=0.45)
        self.play(FadeIn(mech1))
        self.play(Write(f1))
        self.hold(1.5)
        self.play(Write(f1b))
        self.hold(2)
        cs = CrankSlider(0.9, 2.4, 0, origin=np.array([1.2, 0.6, 0]))
        O, A2, B2 = cs.solve(0.8)
        mech2 = VGroup(link_line(O, A2, GOLD), link_line(A2, B2, BLUE_B),
                       slider_block(B2, 0.6, 0.36), pin_joint(A2), fixed_pin(O))
        f2 = MathTex(r"n=3,\ P_L=4\,(3R+1P),\ F=1", font_size=34, color=GREEN).next_to(mech2, DOWN, buff=0.4)
        self.play(FadeIn(mech2), Write(f2))
        self.hold(2)
        cam = cam_profile_knife(lambda d: 0.3 * (1 - np.cos(d)), 0.6).shift(RIGHT * 4.6 + DOWN * 1.9)
        stem2 = Line(cam.get_center() + UP * 0.9, cam.get_center() + UP * 1.7, color=ORANGE, stroke_width=6)
        f3 = MathTex(r"n=2,\ P_L=2,\ P_H=1,\ F=1", font_size=34, color=GREEN).next_to(cam, LEFT, buff=0.4)
        self.play(Create(cam), Create(stem2), Write(f3))
        self.hold(2.5)
        self.add(page_ref("孙桓八版 p16-17"))


class S05_CompoundHinge(LessonScene):
    """陷阱一：复合铰链（8min，p18）。
    讲稿要点：k 个构件汇交于同一转动轴线 → 转动副数 = k−1（不是 1！）。
    动画：三个杆共铰，'拆开看'显示两层铰链叠在一起。"""

    def construct(self):
        self.header("陷阱一：复合铰链", "k 个构件共轴 → (k−1) 个转动副 · p18")
        c = np.array([-1.5, 0.2, 0])
        bars = VGroup(link_line(c, c + np.array([2.4, 0.9, 0]), BLUE_B),
                      link_line(c, c + np.array([2.2, -1.2, 0]), TEAL),
                      link_line(c, c + np.array([-2.4, -0.4, 0]), GOLD))
        j = pin_joint(c, r=0.12)
        self.play(Create(bars), FadeIn(j))
        q = ctext("这里是 1 个转动副吗？", size=32, color=ORANGE).to_edge(DOWN, buff=1.4)
        self.play(Write(q))
        self.hold(2)
        # 拆层动画：三杆端部错开显示两个副
        self.play(bars[0].animate.shift(UP * 0.35), bars[2].animate.shift(DOWN * 0.35), run_time=1.2)
        j2 = VGroup(pin_joint(c + UP * 0.35, 0.1), pin_joint(c + DOWN * 0.0, 0.1))
        arrow_note = ctext("拆开看：杆1-杆2 一个副，杆2-杆3 又一个副", size=28, color=ACCENT).to_edge(DOWN, buff=0.75)
        self.play(FadeIn(j2), ReplacementTransform(q, arrow_note))
        self.hold(2)
        rule = MathTex(r"k\ \text{个构件汇交}\ \Rightarrow\ P_L \mathrel{+}= (k-1)", font_size=46,
                       color=YELLOW).to_edge(DOWN, buff=0.1)
        self.play(Write(rule))
        self.add(page_ref("孙桓八版 p18"))
        self.hold(3)


class S06_LocalDOF(LessonScene):
    """陷阱二：局部自由度（8min，p18-19）。
    讲稿要点：滚子自转不影响整个机构的输出运动 → 计算时把滚子与推杆焊死（去掉该自由度）。
    动画：凸轮-滚子推杆机构，滚子疯转/不转，推杆位移曲线完全一致。"""

    def construct(self):
        self.header("陷阱二：局部自由度", "滚子的'无效自转' · p18-19")
        cam = cam_profile_knife(lambda d: 0.4 * (1 - np.cos(d)), 0.8).shift(DOWN * 1.4 + LEFT * 2.5)
        cam_c = cam.get_center()
        roller = Circle(radius=0.22, color=TEAL, stroke_width=5).move_to(cam_c + UP * 1.45)
        spoke = Line(roller.get_center(), roller.get_center() + UP * 0.22, color=TEAL, stroke_width=3)
        stem = Line(roller.get_center() + UP * 0.22, roller.get_center() + UP * 1.6, color=ORANGE, stroke_width=6)
        self.play(Create(cam), Create(roller), Create(spoke), Create(stem))
        t1 = ctext("滚子绕自身轴的转动：转多快，推杆都不多走一毫米", size=28).to_edge(DOWN, buff=1.2)
        self.play(Write(t1))
        self.play(Rotate(spoke, 6 * TAU, about_point=roller.get_center()), run_time=3, rate_func=linear)
        self.hold(1.5)
        rule = ctext("计算 F 时：把滚子与推杆'焊死'（n−1，PL−1，抵消该局部自由度）", size=28,
                     color=ACCENT).to_edge(DOWN, buff=0.55)
        calc = MathTex(r"n=2,\ P_L=2,\ P_H=1\ \Rightarrow\ F=3\times2-2\times2-1=1", font_size=38,
                       color=GREEN).to_edge(DOWN, buff=0.05)
        self.play(ReplacementTransform(t1, rule))
        self.hold(1.5)
        self.play(Write(calc))
        self.add(page_ref("孙桓八版 p18-19"))
        self.hold(3)


class S07_VirtualConstraint(LessonScene):
    """陷阱三：虚约束（12min，p19-21，最难陷阱）。
    讲稿要点：不起独立限制作用的重复约束=虚约束，计算时必须除去；
    典型情形：①两构件间多条轨迹重合的连线（平行四边形加中间杆 EF）②两副连同一构件且导路平行
    ③机车车轮多个平行曲柄。虚约束的工程价值：改善受力（冗余但有用！）。"""

    def construct(self):
        self.header("陷阱三：虚约束", "重复的约束不算数 · p19-21")
        # 平行四边形 + 中间平行杆
        A0 = np.array([-4.5, -0.8, 0]); B0 = np.array([-0.5, -0.8, 0])
        L = 1.6
        th = ValueTracker(PI / 3)

        def para():
            a = th.get_value()
            A = A0 + L * np.array([np.cos(a), np.sin(a), 0])
            B = B0 + L * np.array([np.cos(a), np.sin(a), 0])
            E0 = A0 * 0.5 + B0 * 0.5
            E = A * 0.5 + B * 0.5
            g = VGroup(link_line(A0, A, GOLD), link_line(B0, B, GOLD), link_line(A, B, BLUE_B),
                       link_line(E0, E, RED),  # 中间杆 = 虚约束
                       *[pin_joint(p) for p in (A0, A, B, B0, E0, E)],
                       ground_hatch(A0 + DOWN * 0.15, 0.55), ground_hatch(B0 + DOWN * 0.15, 0.55))
            return g

        mech = always_redraw(para)
        self.play(FadeIn(mech))
        t1 = ctext("平行四边形机构 + 中间平行杆 EF（红）", size=28).to_edge(DOWN, buff=1.35)
        self.play(Write(t1))
        self.play(th.animate.set_value(PI / 3 + TAU * 0.8), run_time=5, rate_func=there_and_back)
        f_wrong = MathTex(r"\text{硬算: } n=4,P_L=6 \Rightarrow F=0\ ??\ \text{可它明明在动!}",
                          font_size=36, color=RED).to_edge(DOWN, buff=0.75)
        self.play(ReplacementTransform(t1, f_wrong))
        self.hold(2)
        t2 = ctext("EF 连接的两点轨迹本就重合 → 它的约束是重复的 = 虚约束，去掉再算",
                   size=28, color=ACCENT).to_edge(DOWN, buff=0.75)
        f_right = MathTex(r"\text{去 EF: } n=3,P_L=4 \Rightarrow F=1\ \checkmark", font_size=38,
                          color=GREEN).to_edge(DOWN, buff=0.15)
        self.play(ReplacementTransform(f_wrong, t2))
        self.hold(2)
        self.play(Write(f_right))
        self.hold(2)
        note = ctext("虚约束不白给：机车车轮三曲柄——运动上冗余，受力上救命", size=26,
                     color=GREY_B).next_to(f_right, UP, buff=2.9).shift(RIGHT * 3.2)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p19-21"))
        self.hold(3)


class S08_FailureTheater(LessonScene):
    """反例剧场（8min）：F=0 的桁架'假机构' + F=2 单电机乱动。
    讲稿要点：五杆机构 F=2 需要两个原动件；只装一个电机时另一自由度随机漂移。"""

    def construct(self):
        self.header("反例剧场", "F 算错的下场")
        tri = Polygon([-5, -1, 0], [-2.6, -1, 0], [-3.8, 0.9, 0], color=GREY_B, stroke_width=6)
        t1 = ctext("n=2, PL=3 → F=0：三角形桁架，稳如泰山但它不是机构", size=27).to_edge(DOWN, buff=1.3)
        self.play(Create(tri), Write(t1))
        self.hold(2.2)
        # 五杆 F=2 演示（两个相位独立变化）
        a1, a2 = ValueTracker(0.9), ValueTracker(2.2)
        P0 = np.array([1.0, -1.0, 0]); Q0 = np.array([4.6, -1.0, 0])
        L1, L2, L3, L4 = 1.1, 1.7, 1.7, 1.1

        def fivebar():
            A = P0 + L1 * np.array([np.cos(a1.get_value()), np.sin(a1.get_value()), 0])
            D = Q0 + L4 * np.array([np.cos(a2.get_value()), np.sin(a2.get_value()), 0])
            dv = D - A
            d = max(min(float(np.linalg.norm(dv)), L2 + L3 - 1e-6), abs(L2 - L3) + 1e-6)
            aa = (L2 ** 2 - L3 ** 2 + d ** 2) / (2 * d)
            hh = np.sqrt(max(L2 ** 2 - aa ** 2, 0))
            u = dv / d; nvec = np.array([-u[1], u[0], 0])
            C = A + aa * u + hh * nvec
            return VGroup(link_line(P0, A, GOLD), link_line(A, C, BLUE_B), link_line(C, D, TEAL),
                          link_line(D, Q0, ORANGE), *[pin_joint(p) for p in (P0, A, C, D, Q0)],
                          ground_hatch(P0 + DOWN * 0.15, 0.55), ground_hatch(Q0 + DOWN * 0.15, 0.55))

        mech = always_redraw(fivebar)
        f2 = MathTex(r"\text{五杆: } n=4, P_L=5 \Rightarrow F=2", font_size=36, color=ORANGE).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(mech), ReplacementTransform(t1, f2))
        t2 = ctext("只驱动左曲柄：右侧随机漂移——运动不确定！", size=27, color=BAD).to_edge(DOWN, buff=0.12)
        self.play(Write(t2))
        self.play(a1.animate.set_value(0.9 + TAU), a2.animate.set_value(2.2 + 2.1),
                  run_time=4, rate_func=linear)
        self.play(a1.animate.set_value(0.9 + 2 * TAU), a2.animate.set_value(2.2 - 1.3),
                  run_time=4, rate_func=linear)
        t3 = ctext("两个电机同时驱动 → 运动确定（这正是五杆并联机械臂的原理）", size=27,
                   color=GOOD).to_edge(DOWN, buff=0.12)
        self.play(ReplacementTransform(t2, t3))
        self.play(a1.animate.set_value(0.9 + 2.5 * TAU), a2.animate.set_value(2.2 + TAU),
                  run_time=4, rate_func=smooth)
        self.add(page_ref("孙桓八版 p14-16"))
        self.hold(2.5)


class S09_AssurGroups(LessonScene):
    """机构的组成原理与杆组（12min，p21-24）。
    讲稿要点：机构 = 机架 + 原动件 + 若干 F=0 的基本杆组（阿苏尔杆组）；
    杆组条件 3n−2PL=0 → 最简 n=2,PL=3 为Ⅱ级组（5 种形式）；n=4,PL=6 为Ⅲ级组；
    机构级别 = 所含最高杆组级别；拆杆组从远离原动件端开始。"""

    def construct(self):
        self.header("机构的组成原理", "杆组：F=0 的'积木' · p21-24")
        steps = [
            MathTex(r"\text{杆组: 不能再拆的、}F=0\text{ 的构件组}", font_size=40),
            MathTex(r"3n - 2P_L = 0\ \Rightarrow\ n:P_L = 2:3", font_size=44),
            MathTex(r"n=2,\ P_L=3\ \Rightarrow\ \text{II 级杆组（最常用）}", font_size=42, color=YELLOW),
        ]
        formula_reveal(self, steps, anchor=UP * 1.1, wait=1.6)
        # 二级组示意 RRR
        g1 = VGroup(link_line(LEFT * 1.0 + DOWN * 1.6, ORIGIN + DOWN * 0.7, BLUE_B),
                    link_line(ORIGIN + DOWN * 0.7, RIGHT * 1.0 + DOWN * 1.6, TEAL),
                    pin_joint(LEFT * 1.0 + DOWN * 1.6), pin_joint(ORIGIN + DOWN * 0.7),
                    pin_joint(RIGHT * 1.0 + DOWN * 1.6)).shift(LEFT * 3)
        lbl = ctext("RRR 二级组（两杆三副）", size=26, color=BLUE_B).next_to(g1, DOWN, buff=0.3)
        self.play(FadeIn(g1), Write(lbl))
        self.hold(2)
        rule = bullets([
            "机构 = 机架 + 原动件 + 若干杆组",
            "拆组顺序：从离原动件最远的构件开始试拆Ⅱ级组",
            "机构级别 = 其中最高杆组的级别（Ⅱ级机构最普遍）",
        ], size=27).shift(RIGHT * 2.6 + DOWN * 1.7)
        self.play(FadeIn(rule, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p21-24"))
        self.hold(3)


class S10_Summary(LessonScene):
    """小结（4min）：F 公式 + 三大陷阱口诀 + 杆组；下讲预告：知道能动了，那到底怎么动？——运动分析。"""

    def construct(self):
        self.header("第 2 讲小结")
        f = MathTex(r"F = 3n - 2P_L - P_H", font_size=64, color=YELLOW).shift(UP * 1.6)
        self.play(Write(f))
        pts = bullets([
            "复合铰链：k 个构件共轴 → (k−1) 个副       (p18)",
            "局部自由度：滚子自转 → 焊死再算            (p18-19)",
            "虚约束：重复约束 → 去掉再算（但工程上有用）  (p19-21)",
            "杆组：3n=2PL 的 F=0 积木；拆组定机构级别    (p21-24)",
        ], size=28).shift(DOWN * 0.6)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：能动了——每个点的速度、加速度到底是多少？", size=30, color=ACCENT).to_edge(DOWN, buff=0.4)
        self.play(Write(q))
        self.hold(3)
