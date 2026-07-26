# -*- coding: utf-8 -*-
"""L08 最完美的曲线（下）——切齿、变位与空间齿轮（孙桓八版 §10-6~10-10, p208-226）

目标成片 85-95 min。核心：范成法、根切与 zmin=17、变位、斜齿当量齿数、锥齿/蜗杆要点。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：上讲的完美曲线，工厂里怎么造？两条路：仿形法(成形铣刀,精度看刀)
    vs 范成法(展成,一把刀切所有同模数齿轮——工业智慧)。(p208-210)"""

    def construct(self):
        big = ctext("第 8 讲   最完美的曲线（下）", size=56, weight="BOLD")
        sub = ctext("切齿 · 根切 · 变位 · 斜齿与蜗杆", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        pts = bullets([
            "仿形法：铣刀形状=齿槽形状——换齿数就得换刀",
            "范成法：刀具与轮坯'对滚'——一把刀切遍同模数所有齿数",
            "范成法的代价：小齿数会被'切伤'——根切",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p208-210"))
        self.hold(3)


class S02_GenerationCutting(LessonScene):
    """★范成法切齿动画（15min，p209-210，本讲核心动画）。
    讲稿要点：齿条刀与轮坯按 v=ωr 对滚（相当于齿条齿轮啮合），刀刃各位置的
    包络线自然形成渐开线齿面。动画：齿条刀多个相对位置叠印，齿形被'包络'出来。"""

    def construct(self):
        self.header("范成法：包络出渐开线", "刀具与轮坯的共舞 · p209-210")
        m, z = 0.5, 10
        r = m * z / 2
        ctr = np.array([0, -1.1, 0])
        blank = Circle(radius=r + m, color=GREY_C, stroke_width=2).move_to(ctr)
        pitch_line_y = ctr[1] + r
        self.play(Create(blank))
        t1 = ctext("轮坯转 ω，齿条刀平移 v = ω·r（分度圆与刀具中线纯滚动）", size=27).to_edge(DOWN, buff=1.2)
        self.play(Write(t1))

        def rack(x_shift):
            """简化梯形齿条刀（齿形角 20°）。"""
            g = VGroup()
            alpha = np.deg2rad(20)
            pitch = PI * m
            for k in range(-6, 7):
                x0 = x_shift + k * pitch
                ha = m
                pts = [
                    [x0 - pitch / 4 - ha * np.tan(alpha), pitch_line_y + ha, 0],
                    [x0 - pitch / 4 + ha * np.tan(alpha), pitch_line_y - ha, 0],
                    [x0 + pitch / 4 - ha * np.tan(alpha), pitch_line_y - ha, 0],
                    [x0 + pitch / 4 + ha * np.tan(alpha), pitch_line_y + ha, 0],
                ]
                g.add(VMobject(color=ORANGE, stroke_width=2.5).set_points_as_corners(pts))
            return g

        # 包络叠印：把齿条在轮坯参考系中的多个位置画出（轮坯静止，刀绕它滚）
        ghosts = VGroup()
        for i, phi in enumerate(np.linspace(-0.9, 0.9, 13)):
            rk = rack(-phi * r)
            rk.rotate(phi, about_point=ctr)
            rk.set_stroke(opacity=0.28)
            ghosts.add(rk)
        t2 = ctext("把每个相对位置的刀刃都画下来——", size=27, color=ACCENT).to_edge(DOWN, buff=1.2)
        self.play(ReplacementTransform(t1, t2))
        self.play(FadeIn(ghosts, lag_ratio=0.12), run_time=4)
        gear = gear_profile(m, z, color=TEAL, stroke_width=3.5).move_to(ctr)
        t3 = ctext("包络线浮现：渐开线齿形被'挤'了出来！一把直刃刀 → 弯曲齿面", size=27,
                   color=GOOD).to_edge(DOWN, buff=1.2)
        self.play(ReplacementTransform(t2, t3), Create(gear), run_time=3)
        self.add(page_ref("孙桓八版 p209-210"))
        self.hold(3)


class S03_Undercut(LessonScene):
    """★根切与最少齿数 z_min=17 推导（15min，p210-211）。
    讲稿要点：刀具齿顶线若超过啮合极限点 N1，刀刃会切入已成形齿根 → 根切（齿根变瘦、
    εα 降低）。几何条件：ha*·m ≤ N1 到刀具中线的距离 = (mz/2)sin²α →
    z ≥ 2ha*/sin²α = 2/sin²20° = 17.1 → z_min = 17。"""

    def construct(self):
        self.header("根切：小齿轮的'截肢'危机", "z_min = 17 从哪来 · p210-211")
        cmp = VGroup(
            VGroup(gear_profile(0.42, 8, color=RED), ctext("z=8：齿根被掏空", size=24, color=RED)).arrange(DOWN, buff=0.3),
            VGroup(gear_profile(0.42, 17, color=GREEN), ctext("z=17：安全线", size=24, color=GREEN)).arrange(DOWN, buff=0.3),
        ).arrange(RIGHT, buff=1.6).shift(UP * 1.15)
        self.play(FadeIn(cmp, lag_ratio=0.3), run_time=2)
        self.hold(2)
        f = [
            MathTex(r"\text{不根切条件: 刀顶线不超过啮合极限点 } N_1", font_size=38),
            MathTex(r"h_a^{*}m \ \le\ \overline{N_1P}\sin\alpha = \frac{mz}{2}\sin^2\alpha", font_size=44),
            MathTex(r"z \ \ge\ \frac{2h_a^{*}}{\sin^2\alpha}", font_size=48, color=YELLOW),
            MathTex(r"h_a^{*}=1,\ \alpha=20^\circ:\ z_{\min} = \frac{2}{\sin^2 20^\circ} = 17.1 \Rightarrow \boxed{17}",
                    font_size=44, color=GREEN),
        ]
        formula_reveal(self, f, anchor=DOWN * 1.5, buff=0.3, wait=1.9)
        self.add(page_ref("孙桓八版 p210-211"))
        self.hold(3)


class S04_ProfileShift(LessonScene):
    """变位齿轮（15min，p211-216）。
    讲稿要点：把刀具沿径向移距 xm（正变位远离轮心）——同一渐开线取不同段，
    齿根变厚避免根切；最小变位系数 x_min = ha*(z_min−z)/z_min 推导；
    变位传动三类型（零传动/正传动/负传动）与用途（凑中心距、提弯强、修 εα）。"""

    def construct(self):
        self.header("变位：给小齿轮'整形'", "刀具移一移，齿形大不同 · p211-216")
        f = [
            MathTex(r"\text{刀具径向外移 } xm\ (x>0\ \text{正变位})", font_size=42),
            MathTex(r"\text{避免根切: } h_a^{*}m - xm \ \le\ \frac{mz}{2}\sin^2\alpha", font_size=42),
            MathTex(r"\boxed{x_{\min} = h_a^{*}\,\frac{z_{\min}-z}{z_{\min}}}\quad(z<17\ \text{才需要})",
                    font_size=48, color=YELLOW),
            MathTex(r"\text{例: } z=10:\ x_{\min} = \frac{17-10}{17} \approx 0.41", font_size=40, color=GREEN),
        ]
        formula_reveal(self, f, anchor=UP * 0.8, buff=0.38, wait=1.9)
        uses = bullets([
            "凑中心距：a' ≠ 标准 a 时用变位补（无侧隙啮合方程 → inv 函数上场）",
            "提强度：正变位齿根增厚，小轮常取 x>0",
            "代价：齿顶变尖风险、需校核 εα",
        ], size=27).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(uses, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p211-216"))
        self.hold(3.5)


class S05_HelicalGears(LessonScene):
    """斜齿轮（15min，p216-221）。
    讲稿要点：齿廓曲面=渐开螺旋面；接触线斜着逐渐进入/退出 → 平稳、εγ 大（可 >2）；
    法面/端面参数 mn=mt·cosβ、tanαn=tanαt·cosβ；正确啮合加一条 β1=−β2（外啮合）；
    ★当量齿数 zv=z/cos³β 推导（法面剖出椭圆 → 顶点曲率半径 → 当量圆）；轴向力是代价。"""

    def construct(self):
        self.header("斜齿轮：把'突然'变'渐渐'", "螺旋的智慧 · p216-221")
        cmp = bullets([
            "直齿：整条齿宽同时进入啮合——'啪'的一声（冲击）",
            "斜齿：接触线斜着扫入扫出——'嘶'的一声（平稳）",
            "总重合度 εγ = εα + εβ（轴面重合度白赚一项）",
        ], size=29).shift(UP * 1.5)
        self.play(FadeIn(cmp, lag_ratio=0.4), run_time=2.5)
        self.hold(2)
        f = [
            MathTex(r"m_n = m_t\cos\beta,\qquad \tan\alpha_n = \tan\alpha_t\cos\beta", font_size=40),
            MathTex(r"\text{正确啮合: } m_{n1}=m_{n2},\ \alpha_{n1}=\alpha_{n2},\ \beta_1=-\beta_2", font_size=38),
            MathTex(r"\text{当量齿数（法面椭圆顶点曲率半径 } \rho=\tfrac{r}{\cos^2\beta}\text{）:}", font_size=36),
            MathTex(r"\boxed{z_v = \frac{2\rho}{m_n} = \frac{z}{\cos^3\beta}}", font_size=50, color=YELLOW),
            MathTex(r"\Rightarrow\ \text{斜齿不根切: } z_{\min}' = z_{\min}\cos^3\beta < 17\ \text{（还能更小!）}",
                    font_size=38, color=GREEN),
        ]
        formula_reveal(self, f, anchor=DOWN * 1.1, buff=0.3, wait=1.9)
        self.add(page_ref("孙桓八版 p216-221"))
        self.hold(3)


class S06_BevelWorm(LessonScene):
    """锥齿轮与蜗杆蜗轮（12min，p221-226）。
    讲稿要点：锥齿轮传相交轴运动，背锥展开 → 当量直齿轮 zv=z/cosδ；
    蜗杆传动=交错轴、大传动比 i=z2/z1（z1=头数）、正确啮合(ma1=mt2, αa1=αt2, γ1=β2 同旋向)、
    转向判定左右手定则、常自锁（γ<φv，呼应 L10）。"""

    def construct(self):
        self.header("空间传动双雄", "锥齿轮 & 蜗杆蜗轮 · p221-226")
        left = bullets([
            "锥齿轮：两轴相交（常 90°）",
            "节锥纯滚动；大端参数为标准",
            "背锥展开 → 当量齿数 zv = z / cosδ",
        ], size=28).shift(LEFT * 3.3 + UP * 1.0)
        right = bullets([
            "蜗杆蜗轮：两轴交错 90°",
            "i = z2 / z1，z1=1~4 头 → 单级可达 i=80",
            "滑动速度大 → 效率低、发热；常可自锁",
            "转向：左右手定则（右旋用右手）",
        ], size=28).shift(RIGHT * 3.1 + UP * 0.85)
        divider = Line(UP * 2.6, DOWN * 2.0, color=GREY_C, stroke_width=2)
        self.play(Create(divider))
        self.play(FadeIn(left, lag_ratio=0.4), run_time=2)
        self.hold(1.5)
        self.play(FadeIn(right, lag_ratio=0.4), run_time=2)
        note = ctext("蜗杆自锁 = 电梯/卷扬机的安全底牌（第 10 讲从摩擦角证明它）", size=27,
                     color=ACCENT).to_edge(DOWN, buff=0.6)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p221-226"))
        self.hold(3.5)


class S07_Summary(LessonScene):
    """小结（4min）+ 预告 L09：单对齿轮 i 有限，串起来呢？轮系——传动比的乐高。"""

    def construct(self):
        self.header("第 8 讲小结")
        pts = bullets([
            "范成法：对滚包络，一刀通吃同模数        (p209-210)",
            "根切与 z_min = 2ha*/sin²α = 17          (p210-211)",
            "变位 x_min = ha*(17−z)/17；凑距提强     (p211-216)",
            "斜齿：mn=mt·cosβ；zv=z/cos³β；有轴向力  (p216-221)",
            "锥齿 zv=z/cosδ；蜗杆大 i 可自锁         (p221-226)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：手表里几十个齿轮怎么算？行星轮系为什么难住所有初学者？",
                  size=30, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
