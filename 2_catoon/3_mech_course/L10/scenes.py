# -*- coding: utf-8 -*-
"""L10 机器中的暗力量——摩擦、效率与自锁（孙桓八版 第4章+第5章, p55-84）

目标成片 85-95 min。核心：摩擦角/斜面正反行程、螺旋副力矩、摩擦圆、效率、串并联、自锁三判据。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：螺旋千斤顶顶起一吨的车，松手为什么不掉？——摩擦既是小偷也是保安。
    本讲双主题：力分析中的摩擦(第4章) + 效率与自锁(第5章)。"""

    def construct(self):
        big = ctext("第 10 讲   机器中的暗力量", size=58, weight="BOLD")
        sub = ctext("摩擦 · 效率 · 自锁", size=34, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        pts = bullets([
            "摩擦是小偷：偷走 5%~70% 的输入功率",
            "摩擦是保安：千斤顶、夹具、蜗杆——全靠它'锁住'",
            "本讲兵器：摩擦角 φ · 摩擦圆 ρ · 效率 η",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(pts, lag_ratio=0.4))
        self.hold(3)


class S02_FrictionAngle(LessonScene):
    """移动副摩擦与摩擦角（10min，p58-59）。
    讲稿要点：全反力 R = N + F 合成，与法线偏角 φ=arctan f（摩擦角）；
    方向铁律：R 恒偏向阻碍相对运动一侧；摩擦锥：驱动力在锥内怎么推都不动。"""

    def construct(self):
        self.header("摩擦角 φ：全反力的偏转", "p58-59")
        ground = Line(LEFT * 5, RIGHT * 2, color=GREY_B, stroke_width=4).shift(DOWN * 1.4)
        blk = Rectangle(width=1.6, height=1.0, color=BLUE_B, fill_opacity=0.25).shift(LEFT * 1.6 + DOWN * 0.85)
        self.play(Create(ground), Create(blk))
        c = blk.get_bottom()
        N = Arrow(c, c + UP * 1.8, buff=0, color=TEAL)
        Fv = Arrow(c, c + LEFT * 0.9, buff=0, color=RED)
        Rv = Arrow(c, c + UP * 1.8 + LEFT * 0.9, buff=0, color=YELLOW, stroke_width=6)
        labs = VGroup(MathTex("N", font_size=34, color=TEAL).next_to(N, UP, buff=0.1),
                      MathTex("F=fN", font_size=32, color=RED).next_to(Fv, LEFT, buff=0.1),
                      MathTex("R", font_size=36, color=YELLOW).next_to(Rv.get_end(), UP + LEFT, buff=0.1))
        move = Arrow(blk.get_right(), blk.get_right() + RIGHT * 1.2, buff=0.1, color=WHITE)
        mlab = ctext("相对运动 →", size=24).next_to(move, UP, buff=0.1)
        self.play(GrowArrow(move), FadeIn(mlab))
        self.play(GrowArrow(N), GrowArrow(Fv), run_time=1.2)
        self.play(GrowArrow(Rv), FadeIn(labs))
        f = [
            MathTex(r"\tan\varphi = \frac{F}{N} = f\ \Rightarrow\ \boxed{\varphi = \arctan f}",
                    font_size=48, color=YELLOW),
            MathTex(r"\text{方向铁律: } R\ \text{恒向'阻碍相对运动'一侧偏转 } \varphi", font_size=38, color=GREEN),
        ]
        formula_reveal(self, f, anchor=RIGHT * 3.4 + UP * 0.9, buff=0.45, wait=1.8)
        cone = ctext("摩擦锥：合外力作用线落在 2φ 锥内 → 推力再大也推不动（自锁伏笔）",
                     size=26, color=ACCENT).to_edge(DOWN, buff=0.35)
        self.play(Write(cone))
        self.add(page_ref("孙桓八版 p58-59"))
        self.hold(3)


class S03_InclinedPlane(LessonScene):
    """★斜面正反行程推导（15min，p59-60，本讲核心推导一）。
    讲稿要点：滑块沿斜面匀速上滑（正行程），力三角形 → F = G·tan(α+φ)；
    下滑（反行程）摩擦反向 → F' = G·tan(α−φ)；α<φ 时 F'<0：要拉才下来 = 自锁！"""

    def construct(self):
        self.header("斜面的正反行程", "一对'镜像'公式 · p59-60")
        alpha = np.deg2rad(24)
        slope = Polygon([-5.5, -2.2, 0], [-0.5, -2.2, 0], [-0.5, -2.2 + 5 * np.tan(alpha), 0],
                        color=GREY_B, fill_opacity=0.15)
        blk = Square(side_length=0.7, color=BLUE_B, fill_opacity=0.3).rotate(alpha)
        blk.move_to([-2.6, -2.2 + 2.9 * np.tan(alpha) + 0.42, 0])
        self.play(Create(slope), Create(blk))
        f_up = [
            MathTex(r"\text{正行程(推上去): 力平衡 } \vec F + \vec G + \vec R = 0", font_size=38),
            MathTex(r"R \text{ 与法线偏 } \varphi\ \text{(向下坡侧)}\ \Rightarrow\ \text{力三角形}", font_size=38),
            MathTex(r"\boxed{F = G\tan(\alpha+\varphi)}", font_size=52, color=YELLOW),
        ]
        formula_reveal(self, f_up, anchor=RIGHT * 3.1 + UP * 1.5, buff=0.35, wait=1.8)
        f_dn = [
            MathTex(r"\text{反行程(滑下来): 摩擦反向} \Rightarrow \varphi \to -\varphi", font_size=38),
            MathTex(r"\boxed{F' = G\tan(\alpha-\varphi)}", font_size=52, color=GREEN),
            MathTex(r"\alpha<\varphi\ \Rightarrow\ F'<0:\ \text{不拉不下来}=\textbf{自锁!}", font_size=42, color=RED),
        ]
        formula_reveal(self, f_dn, anchor=RIGHT * 3.1 + DOWN * 1.6, buff=0.35, wait=1.8)
        self.add(page_ref("孙桓八版 p59-60"))
        self.hold(3)


class S04_ScrewFriction(LessonScene):
    """螺旋副摩擦（12min，p60-61）。
    讲稿要点：矩形螺纹=斜面绕在圆柱上（升角 λ, tanλ=l/(πd2)）；拧紧=沿斜面推重物上行
    M = G·(d2/2)·tan(λ+φ)；放松 M' = G·(d2/2)·tan(λ−φ)；三角螺纹用当量摩擦角 φv=arctan(f/cosβ)。
    动画：螺旋线'展开'成斜面。"""

    def construct(self):
        self.header("螺旋 = 缠起来的斜面", "千斤顶的数学 · p60-61")
        # 螺旋线展开动画
        helix_pts = [np.array([1.4 * np.cos(t), -1.8 + 0.16 * t, 1.4 * np.sin(t)])[:3] for t in np.linspace(0, 4 * PI, 120)]
        helix2d = [np.array([1.4 * np.cos(t), -1.9 + 0.18 * t, 0]) for t in np.linspace(0, 4 * PI, 120)]
        helix = VMobject(color=TEAL, stroke_width=4).set_points_smoothly(helix2d).shift(LEFT * 3.4)
        self.play(Create(helix), run_time=2)
        tri = Polygon([0.6, -1.9, 0], [5.4, -1.9, 0], [5.4, -0.55, 0], color=TEAL, fill_opacity=0.15)
        tl = MathTex(r"\tan\lambda = \frac{l}{\pi d_2}", font_size=40).next_to(tri, UP, buff=0.3)
        t1 = ctext("把一圈螺纹展开：底边 πd2、高为导程 l 的斜面！", size=27).to_edge(DOWN, buff=1.25)
        self.play(Create(tri), Write(tl), Write(t1))
        self.hold(2.2)
        f = [
            MathTex(r"\text{拧紧: } M = G\,\frac{d_2}{2}\tan(\lambda+\varphi_v)", font_size=46, color=YELLOW),
            MathTex(r"\text{放松: } M' = G\,\frac{d_2}{2}\tan(\lambda-\varphi_v)", font_size=46, color=GREEN),
            MathTex(r"\text{三角螺纹: } \varphi_v=\arctan\frac{f}{\cos\beta}>\varphi\ \text{(更易自锁, 宜连接)}",
                    font_size=36),
        ]
        self.play(FadeOut(t1))
        formula_reveal(self, f, anchor=DOWN * 2.6, buff=0.3, wait=1.9)
        self.add(page_ref("孙桓八版 p60-61"))
        self.hold(2.5)


class S05_FrictionCircle(LessonScene):
    """转动副摩擦圆（10min，p61-62）。
    讲稿要点：轴颈转动时全反力 R 对轴心力矩 = 摩擦力矩 → R 必与半径 ρ=fv·r 的圆相切；
    方向：切向偏向阻碍相对转动一侧。判轴受力：作用线离摩擦圆越远越'省'，
    穿过圆内 → 转不动（转动副自锁）。动画：轴颈转动 R 始终切圆。"""

    def construct(self):
        self.header("摩擦圆 ρ = f_v · r", "转动副的'禁区' · p61-62")
        journal = Circle(radius=1.5, color=GREY_B, stroke_width=4).shift(LEFT * 2.8)
        shaft = Circle(radius=1.34, color=BLUE_B, stroke_width=5).move_to(journal)
        fric = Circle(radius=0.42, color=RED, stroke_width=3).move_to(journal)
        flab = MathTex(r"\rho = f_v r", font_size=36, color=RED).next_to(fric, DOWN, buff=0.1)
        self.play(Create(journal), Create(shaft), Create(fric), Write(flab))
        thv = ValueTracker(0.3)

        def Rline():
            a = thv.get_value()
            # 与摩擦圆相切的一条线（切点随 a 转）
            tp = journal.get_center() + 0.42 * np.array([np.cos(a), np.sin(a), 0])
            d = np.array([-np.sin(a), np.cos(a), 0])
            return VGroup(Line(tp - d * 2.2, tp + d * 2.2, color=YELLOW, stroke_width=4),
                          Dot(tp, color=YELLOW, radius=0.05))

        self.play(FadeIn(always_redraw(Rline)))
        t1 = ctext("轴匀速转动时：全反力 R 的作用线恒与摩擦圆相切", size=28).to_edge(DOWN, buff=1.2)
        self.play(Write(t1))
        self.play(thv.animate.set_value(0.3 + TAU), run_time=5, rate_func=linear)
        rules = bullets([
            "外力作用线切于圆外 → 能转，力臂越大越轻松",
            "作用线穿过摩擦圆内 → 力矩斗不过摩擦：转动副自锁",
        ], size=28).to_edge(DOWN, buff=0.35)
        self.play(ReplacementTransform(t1, rules))
        self.add(page_ref("孙桓八版 p61-62"))
        self.hold(3)


class S06_Efficiency(LessonScene):
    """机械效率与串并联（12min，p75-78）。
    讲稿要点：η = 输出功/输入功 = P_r/P_d = (理想驱动力/实际驱动力) 三种形式；
    串联 η=∏ηi（连乘衰减可怕）、并联 η=∑P_i η_i/∑P_i（加权平均）；功率流'漏斗'动画。"""

    def construct(self):
        self.header("机械效率 η", "功率去哪儿了 · p75-78")
        f = [
            MathTex(r"\eta = \frac{W_r}{W_d} = \frac{P_r}{P_d} = \frac{F_0}{F}\ \ (\text{功/功率/力三种形式})",
                    font_size=44, color=YELLOW),
            MathTex(r"\text{串联: } \eta = \eta_1\eta_2\cdots\eta_k\quad(\text{连乘, 越串越亏})", font_size=42),
            MathTex(r"\text{并联: } \eta = \frac{\sum P_i\eta_i}{\sum P_i}\quad(\text{加权平均})", font_size=42),
        ]
        formula_reveal(self, f, anchor=UP * 1.3, buff=0.42, wait=1.9)
        # 功率漏斗：三级串联 0.95*0.9*0.8
        levels = [1.0, 0.95, 0.855, 0.684]
        cols = [GREEN, TEAL, ORANGE, RED]
        bars = VGroup()
        for i, (v, c) in enumerate(zip(levels, cols)):
            bars.add(VGroup(Rectangle(width=4.2 * v, height=0.5, fill_color=c, fill_opacity=0.7,
                                      stroke_width=1).shift(DOWN * (1.1 + i * 0.72) + LEFT * (2.1 * (1 - v))),
                            ctext(f"{v*100:.1f}%", size=22).shift(DOWN * (1.1 + i * 0.72) + RIGHT * 3.1)))
        labs = ctext("电机 → 带传动0.95 → 齿轮0.9 → 蜗杆0.8：到手只剩 68%", size=25,
                     color=GREY_B).shift(DOWN * 3.4)
        self.play(FadeIn(bars, lag_ratio=0.3), Write(labs), run_time=3)
        self.add(page_ref("孙桓八版 p75-78"))
        self.hold(3.5)


class S07_SelfLocking(LessonScene):
    """★自锁三判据统一（12min，p78-82）。
    讲稿要点：正行程能动、反行程怎么推都不动=自锁；三种形式——移动副 α≤φ、
    转动副 力臂 e≤ρ、螺旋副 λ≤φv；统一判据：反行程 η'≤0；
    应用双面性：千斤顶/夹具要自锁，传动链严禁自锁。"""

    def construct(self):
        self.header("自锁：反向的'单行道'", "三判据归一 · p78-82")
        table = VGroup(
            MathTex(r"\text{移动副: } \alpha \le \varphi", font_size=44),
            MathTex(r"\text{转动副: } e \le \rho\ (\text{力线进摩擦圆})", font_size=44),
            MathTex(r"\text{螺旋副: } \lambda \le \varphi_v", font_size=44),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.5).shift(LEFT * 2.6 + UP * 0.8)
        for r in table:
            self.play(Write(r), run_time=0.9)
            self.hold(1.3)
        box = SurroundingRectangle(table, color=YELLOW, buff=0.4)
        unify = MathTex(r"\text{统一判据: 反行程效率 } \eta' \le 0", font_size=48,
                        color=YELLOW).next_to(box, RIGHT, buff=0.7)
        self.play(Create(box), Write(unify))
        self.hold(2)
        apps = bullets([
            "要自锁：千斤顶(λ<φv)、台虎钳、楔块夹具——'松手不溜'",
            "怕自锁：传动螺旋、蜗杆提升机构需校核，避免卡死或倒转伤人",
            "呼应 L06：凸轮压力角 α>[α] 的卡死，本质就是自锁",
        ], size=28).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(apps, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p78-82"))
        self.hold(3.5)


class S08_ForceAnalysisNote(LessonScene):
    """力分析与惯性力提要（8min，p55-58, 64-69）。
    讲稿要点：动态静力分析思想——把惯性力 F=-ma、M=-Jα 当外力加上，按静力学解；
    质量代换概念（两点代换的三条件）；为 L11 平衡与速度波动铺路。"""

    def construct(self):
        self.header("把动力学'变成'静力学", "动态静力分析 · p55-58, 64-69")
        f = [
            MathTex(r"\text{达朗贝尔: 加上惯性力 } F_I=-ma_S,\ \ M_I=-J_S\alpha", font_size=44),
            MathTex(r"\Rightarrow\ \text{'动'问题按'静'力学求解（杆组逐个分析）}", font_size=40, color=YELLOW),
            MathTex(r"\text{质量代换(两点): } \sum m_i=m,\ \ \sum m_i x_i =0,\ \ \sum m_i x_i^2=J_S",
                    font_size=38),
        ]
        formula_reveal(self, f, anchor=UP * 0.7, buff=0.45, wait=1.9)
        note = ctext("惯性力是 L11 的两位主角（不平衡振动 / 速度波动）共同的根源", size=28,
                     color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p55-58"))
        self.hold(3)


class S09_Summary(LessonScene):
    """小结（4min）+ 预告 L11：摩擦偷功率是'慢性病'，不平衡与速度波动是'心律不齐'——收官之战。"""

    def construct(self):
        self.header("第 10 讲小结")
        pts = bullets([
            "φ=arctan f；R 恒偏向阻碍相对运动一侧          (p58-59)",
            "斜面: F=G·tan(α±φ)；螺旋: M=G(d2/2)tan(λ±φv)  (p59-61)",
            "摩擦圆 ρ=fv·r；力线入圆即卡死                 (p61-62)",
            "η 三形式；串联连乘并联加权                     (p75-78)",
            "自锁: α≤φ / e≤ρ / λ≤φv ⇔ 反行程 η'≤0        (p78-82)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("终章预告：为什么洗衣机会'跳舞'？飞轮如何驯服忽快忽慢的机器？",
                  size=30, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
