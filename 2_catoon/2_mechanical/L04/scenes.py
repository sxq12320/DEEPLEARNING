# -*- coding: utf-8 -*-
"""L04 机器中的暗力量——力分析·摩擦·效率·自锁（第4章+第5章, p55-84）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """片头（~4min）：力分析的两大任务——求运动副反力（强度/寿命）与平衡力
    （选电机）；受力源：驱动力/工作阻力/重力/惯性力。"""

    def construct(self):
        self.header("机器中的暗力量", "平面机构力分析 · p55")
        tasks = bullets([
            "任务① 求运动副反力 → 零件强度与寿命",
            "任务② 求平衡力/力矩 → 该配多大的电机",
            "受力清单：驱动力 · 工作阻力 · 重力 · 惯性力",
        ], size=30).shift(UP * 0.2)
        self.play(FadeIn(tasks, lag_ratio=0.5), run_time=2.5)
        self.hold(2.5)
        self.takeaway("力分析 = 运动分析 + 达朗贝尔（惯性力当外力）",
                        p="孙桓八版 p55-57")
        self.hold(3)


class S02_FrictionCircle(LessonScene):
    """转动副摩擦（~12min, p60-63）：轴颈转动→反力作用线切摩擦圆；
    摩擦角 φ=arctan f、摩擦锥；摩擦圆 ρ=r·sinφ≈f·r。"""

    def construct(self):
        self.header("转动副里的摩擦", "摩擦角 · 摩擦圆 · p60-63")
        # 轴颈 + 摩擦圆
        journal = Circle(radius=1.1, color=LINK_B, stroke_width=5).move_to(
            LEFT * 3.8 + UP * 0.3)
        shaft = Circle(radius=0.42, color=FRAME_C, stroke_width=4).move_to(
            journal.get_center())
        fr_circle = DashedVMobject(
            Circle(radius=0.62, color=BAD, stroke_width=2.5), num_dashes=36)
        fr_circle.move_to(journal.get_center())
        # 反力 R 切摩擦圆
        contact = journal.get_center() + RIGHT * 0.72 + UP * 0.72
        R_arrow = vec(contact, UP * 1.6 + LEFT * 0.9, ACCENT)
        self.play(Create(journal), Create(shaft))
        self.play(Create(fr_circle))
        labs = VGroup(
            ctext("轴颈", size=22).next_to(shaft, DOWN, buff=0.3),
            ctext("摩擦圆 ρ≈f·r", size=22, color=BAD).next_to(fr_circle, LEFT,
                                                              buff=0.4),
        )
        self.play(FadeIn(labs))
        self.play(FadeIn(R_arrow))
        self.play(Write(ctext("反力 R 必切摩擦圆（阻碍相对转动）", size=24,
                              color=ACCENT).next_to(journal, UP, buff=0.5)))
        self.hold(2)
        # 右侧：摩擦角
        blk = Square(side_length=1.1, color=LINK_D, fill_opacity=0.25)
        blk.shift(RIGHT * 2.6 + DOWN * 0.9)
        nrm = vec(blk.get_center() + DOWN * 0.55, UP * 1.5, INK)
        tot = vec(blk.get_center() + DOWN * 0.55, UP * 1.5 + RIGHT * 0.8,
                  BAD)
        phi = angle_mark(blk.get_center() + DOWN * 0.55, PI / 2,
                         np.arctan2(1.5, 0.8), r=0.55, label="φ", color=BAD)
        self.play(FadeIn(blk), FadeIn(nrm), FadeIn(tot), FadeIn(phi))
        eq = mtex(r"\tan\varphi = f,\qquad \rho = r\sin\varphi \approx fr",
                     font_size=42, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(eq))
        self.add(page_ref("孙桓八版 p60-63"))
        self.hold(3)


class S03_ScrewIncline(LessonScene):
    """螺旋副=卷起来的斜面（~12min, p63-66）：圆柱展开→斜面；拧紧/松开力矩
    M = F·d2/2·tan(λ±φ)；斜面滑块受力图。"""

    def construct(self):
        self.header("螺旋 = 卷起来的斜面", "p63-66")
        # 圆柱→斜面展开动画
        cyl = Rectangle(width=1.6, height=2.4, color=LINK_B,
                        fill_opacity=0.12, stroke_width=4)
        helix_pts = []
        for t in np.linspace(0, 4 * PI, 80):
            helix_pts.append([0.8 * np.cos(t), -1.0 + t / (4 * PI) * 2.0,
                              0.8 * np.sin(t)])
        self.play(FadeIn(cyl.shift(LEFT * 4.4 + UP * 0.4)))
        note1 = ctext("沿母线剪开并摊平 →", size=25).shift(LEFT * 2.2 + UP * 0.4)
        self.play(Write(note1))
        # 斜面：底 = πd2，高 = 导程
        tri = Polygon(P(0, -0.9), P(3.4, -0.9), P(3.4, -0.15),
                      color=LINK_B, fill_opacity=0.15, stroke_width=4)
        blk = Square(side_length=0.7, color=LINK_D, fill_opacity=0.3)
        blk.rotate(np.arctan2(0.75, 3.4)).move_to(P(1.7, -0.55) + UP * 0.3)
        lam = angle_mark(P(3.4, -0.9), PI, PI - np.arctan2(0.75, 3.4),
                         r=0.6, label="λ", color=ACCENT)
        self.play(Create(tri), FadeIn(blk), FadeIn(lam))
        t = VGroup(ctext("底边 = πd₂（中径周长）", size=23).next_to(tri, DOWN,
                                                                   buff=0.35),
                   ctext("高 = 导程 s；升角 λ = arctan(s/πd₂)", size=23)
                   .next_to(tri, LEFT, buff=0.5))
        self.play(FadeIn(t))
        self.hold(2)
        steps = [
            mtex(r"M = \frac{d_2}{2}\,F\,\tan(\lambda + \varphi)"
                    r"\quad \text{拧紧}", font_size=42),
            mtex(r"M' = \frac{d_2}{2}\,F\,\tan(\lambda - \varphi)"
                    r"\quad \text{松开}", font_size=42),
        ]
        formula_reveal(self, steps, anchor=DOWN * 2.85, wait=1.6)
        self.add(page_ref("孙桓八版 p63-66"))
        self.hold(3)


class S04_Efficiency(LessonScene):
    """机械效率（~10min, p70-74）：η=输出功/输入功<1；功率'漏斗'动画——
    输入功率条流入，摩擦损耗分流，输出变窄；串联η连乘、并联加权。"""

    def construct(self):
        self.header("机械效率 η", "功率去哪了 · p70-74")
        # 功率漏斗
        src = Rectangle(width=0.5, height=2.6, color=GOOD, fill_opacity=0.5,
                        fill_color=GOOD).shift(LEFT * 5.2 + UP * 0.5)
        loss = Polygon(P(-4.6, 1.8), P(0.6, 1.0), P(0.6, 1.8),
                       color=BAD, fill_opacity=0.35, stroke_width=3)
        keep = Polygon(P(-4.6, -0.8), P(-4.6, 1.8), P(0.6, 1.0),
                       P(0.6, -0.2), color=GOOD, fill_opacity=0.3,
                       stroke_width=3)
        out = Rectangle(width=0.5, height=1.2, color=GOOD, fill_opacity=0.5,
                        fill_color=GOOD).shift(RIGHT * 0.9 + UP * 0.15)
        self.play(FadeIn(src), Create(keep), Create(loss), FadeIn(out))
        labs = VGroup(
            ctext("输入功率 Nd", size=24, color=GOOD).next_to(src, UP,
                                                               buff=0.3),
            ctext("摩擦损耗 Nf", size=24, color=BAD).next_to(loss, UP,
                                                              buff=0.25),
            ctext("输出功率 Nr", size=24, color=GOOD).next_to(out, RIGHT,
                                                               buff=0.35),
        )
        self.play(FadeIn(labs))
        self.hold(2)
        steps = [
            mtex(r"\eta = \frac{N_r}{N_d} = 1 - \frac{N_f}{N_d} < 1",
                    font_size=48, color=ACCENT),
            mtex(r"\text{串联：}\eta=\eta_1\eta_2\cdots\eta_k"
                    r"\qquad\text{并联：}\eta=\frac{\sum N_i\eta_i}{\sum N_i}",
                    font_size=38),
        ]
        formula_reveal(self, steps, anchor=DOWN * 1.9, wait=1.8)
        self.add(page_ref("孙桓八版 p70-74"))
        self.hold(3)


class S05_SelfLock(LessonScene):
    """自锁（~10min, p74-78）：斜面自锁 λ≤φ；螺旋千斤顶'顶得住'演示——
    驱动力撤去后靠摩擦保持；自锁条件=效率 η≤0 的物理解读。"""

    def construct(self):
        self.header("自锁：省力但不落下", "p74-78")
        # 斜面+滑块
        ang = np.deg2rad(14)
        wedge = Polygon(P(-4.6, -1.0), P(-1.0, -1.0), P(-1.0, -1.0 + 3.6 *
                                                        np.tan(ang)),
                        color=LINK_B, fill_opacity=0.15, stroke_width=4)
        blk = Square(side_length=0.62, color=LINK_D, fill_opacity=0.3)
        blk.rotate(ang).move_to(P(-2.7, -0.75))
        self.play(Create(wedge), FadeIn(blk))
        g = vec(blk.get_center(), DOWN * 1.0, INK)
        nrm = vec(blk.get_center(), np.array([-np.sin(ang), np.cos(ang), 0]) * 1.15,
                  NOTE)
        fri = vec(blk.get_center(), np.array([np.cos(ang), np.sin(ang), 0]) * 0.75,
                  BAD)
        self.play(FadeIn(g), FadeIn(nrm), FadeIn(fri))
        self.play(Write(ctext("λ ≤ φ：重力分力 < 最大摩擦 → 滑块不下滑",
                              size=25, color=GOOD).to_edge(UP, buff=1.7)))
        self.hold(2)
        steps = [
            mtex(r"\text{自锁条件（斜面）：}\ \lambda \le \varphi",
                    font_size=44, color=ACCENT),
            mtex(r"\text{本质：主动力无论如何增大，效率}\ \eta \le 0",
                    font_size=38),
        ]
        g = formula_reveal(self, steps, anchor=RIGHT * 2.7 + UP * 0.5,
                           wait=1.8)
        self.focus(g[0])
        demo = ctext("螺旋千斤顶：顶起后撤去手柄，重物不掉——自锁保安全",
                     size=25, color=NOTE).to_edge(DOWN, buff=0.5)
        self.play(Write(demo))
        self.add(page_ref("孙桓八版 p74-78"))
        self.hold(3)


class S06_DynamicStatic(LessonScene):
    """动态静力分析（~8min, p57-60）：四杆机构 + 各构件惯性力（与加速度反向）
    → 达朗贝尔：加惯性力后按静力平衡求解。"""

    def construct(self):
        self.header("动态静力分析", "达朗贝尔原理 · p57-60")
        fb = FourBar(3.8, 1.1, 3.0, 2.5, origin=np.array([-4.3, -1.2, 0]))
        m = AnimatedFourBar(fb)
        self.play(FadeIn(m.group))
        self.play(m.theta.animate.set_value(0.6 + TAU), run_time=4,
                  rate_func=linear)
        self.hold(1)
        # 惯性力箭头（示意：连杆质心处与加速度反向）
        _, A, B, _ = fb.solve(m.theta.get_value())
        G = (A + B) / 2
        Fi = vec(G, DOWN * 0.9 + LEFT * 0.5, BAD)
        Mi = Arc(radius=0.5, angle=PI * 1.2, color=BAD, arc_center=G)
        il = mtex(r"\text{惯性力 }F_i=-m\,a_s\;;\;"
                  r"\text{惯性力偶 }M_i=-J\alpha", font_size=34,
                  color=BAD).next_to(m.group, UP, buff=0.5)
        self.play(FadeIn(Fi), FadeIn(Mi), Write(il))
        self.hold(2)
        self.takeaway("加上惯性力 → 动力学问题按静力学解（达朗贝尔）",
                        p="孙桓八版 p57-60")
        self.hold(3)


class S07_UnitEfficiency(LessonScene):
    """机组效率算例（~6min, p74）：带传动+齿轮对+连杆串联，η 连乘现场算。"""

    def construct(self):
        self.header("算例：一台机器的 η", "串联效率 · p74")
        stages = VGroup()
        names = [("带传动", 0.96), ("齿轮副", 0.97), ("连杆机构", 0.90)]
        for i, (nm, et) in enumerate(names):
            box = RoundedRectangle(corner_radius=0.14, width=3.0, height=0.95,
                                   color=LINK_B, fill_opacity=0.12,
                                   fill_color=LINK_B)
            row = VGroup(box, ctext(nm, size=25).move_to(box),
                         ctext(f"η={et}", size=22, color=NOTE)
                         .next_to(box, DOWN, buff=0.12))
            stages.add(row)
        stages.arrange(RIGHT, buff=0.8).shift(UP * 0.9)
        arrows = VGroup(*[vec(stages[i].get_right(), RIGHT * 0.8, INK,
                              tip=0.14) for i in range(2)])
        self.play(FadeIn(stages, lag_ratio=0.3), FadeIn(arrows))
        self.hold(1.5)
        eq = mtex(r"\eta = 0.96 \times 0.97 \times 0.90 \approx 0.84",
                     font_size=48, color=ACCENT).shift(DOWN * 0.8)
        self.play(Write(eq))
        warn = ctext("每级 95%+ 看似很高，三级下来只剩 84%——节能要抠每一环",
                     size=25, color=NOTE).next_to(eq, DOWN, buff=0.45)
        self.play(Write(warn))
        self.add(page_ref("孙桓八版 p74"))
        self.hold(3)


class S08_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 4 讲小结")
        pts = bullets([
            "力分析两任务：运动副反力 / 平衡力        (p55-57)",
            "转动副：反力切摩擦圆 ρ≈fr               (p60-63)",
            "螺旋=卷起的斜面；M=(d2/2)F·tan(λ±φ)     (p63-66)",
            "η=输出/输入；串联连乘                    (p70-74)",
            "自锁：λ≤φ；本质 η≤0                    (p74-78)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：转起来的机器为什么会抖？——机械的平衡", size=25,
                  color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
