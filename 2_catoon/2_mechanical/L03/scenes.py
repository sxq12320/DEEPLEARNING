# -*- coding: utf-8 -*-
"""L03 让速度看得见——平面机构的运动分析（第3章, p35-54）

瞬心法 + 相对运动图解法 + 哥氏加速度 + 解析法一瞥。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """片头（~3min）：机构能动之后——动多快？往哪动？运动分析三大任务。"""

    def construct(self):
        self.header("让速度看得见", "平面机构运动分析 · p35")
        tasks = bullets([
            "位移 / 轨迹：输出点走到哪（行程、工作空间）",
            "速度：快慢与方向（力的估算、功率基础）",
            "加速度：惯性力的来源（振动、冲击的根源）",
        ], size=30).shift(UP * 0.2)
        self.play(FadeIn(tasks, lag_ratio=0.5), run_time=2.5)
        self.hold(2.5)
        self.takeaway("两条路线：瞬心法（快） vs 相对运动图解法（全）",
                        p="孙桓八版 p35-36")
        self.hold(3)


class S02_InstantCenter(LessonScene):
    """瞬心定义（~10min, p36-38）：两构件瞬心=瞬时等速重合点；转动副中心、
    移动副在无穷远、纯滚动高副在接触点、滑动高副在公法线上——四图并排。"""

    def construct(self):
        self.header("速度瞬心：瞬时的'共同转轴'", "p36-38")
        # 四情形小图卡
        cards = VGroup()
        defs = [
            ("转动副", "瞬心 = 铰链中心", lambda c: VGroup(
                pin_joint(c, 0.12), link_line(c + LEFT * 0.7 + DOWN * 0.4, c),
                link_line(c + RIGHT * 0.7 + UP * 0.4, c))),
            ("移动副", "瞬心在 ⊥导路 无穷远处", lambda c: VGroup(
                guide_rails(c, 1.8, 0.5), slider_block(c, 0.5, 0.3))),
            ("纯滚动高副", "瞬心 = 接触点", lambda c: VGroup(
                Circle(radius=0.55, color=LINK_B).move_to(c + UP * 0.55),
                Line(c + LEFT * 0.9, c + RIGHT * 0.9, color=FRAME_C,
                     stroke_width=4),
                Dot(c, radius=0.06, color=ACCENT))),
            ("滑动高副", "瞬心在接触点公法线上", lambda c: VGroup(
                Arc(radius=0.6, angle=PI * 0.6, color=LINK_B,
                    arc_center=c + DOWN * 0.15),
                Line(c + UP * 0.1 + LEFT * 0.5, c + UP * 1.5 + LEFT * 0.1,
                     color=ACCENT, stroke_width=3),
                Dot(c + UP * 0.45 + LEFT * 0.18, radius=0.06, color=ACCENT))),
        ]
        for i, (name, desc, build) in enumerate(defs):
            cx = P(-5.1 + i * 3.4, 0.9)
            card = VGroup(build(cx),
                          ctext(name, size=24, color=LINK_B).next_to(
                              VGroup(build(cx)), DOWN, buff=0.5),
                          ctext(desc, size=20, color=NOTE).next_to(
                              VGroup(build(cx)), DOWN, buff=1.0))
            cards.add(card)
        self.play(FadeIn(cards, lag_ratio=0.3), run_time=3)
        self.hold(3)
        self.takeaway("瞬心：两构件上瞬时速度相同的重合点", p="孙桓八版 p36-38")
        self.hold(3)


class S03_Kennedy(LessonScene):
    """三心定理（~8min, p38-39）：Kennedy 定理——三构件三瞬心共线。
    反证动画：假设 P13 不在 P12P23 连线上→速度方向矛盾→必须在连线上。"""

    def construct(self):
        self.header("三心定理（Kennedy）", "三个瞬心必共线 · p38-39")
        # 三个构件：1=机架（地）, 2,3 叠放示意
        g1 = ground_hatch(P(-3.5, -1.0), 3.0)
        l1 = ctext("构件1（机架）", size=22).next_to(g1, DOWN, buff=0.3)
        bar2 = link_line(P(-4.5, -0.3), P(-2.4, 0.8), LINK_B)
        bar3 = link_line(P(-3.6, 1.4), P(-1.8, 0.6), LINK_C)
        p12 = pin_joint(P(-4.3, -0.5), color=LINK_A)
        p23 = pin_joint(P(-2.9, 0.4), color=GOOD)
        self.play(FadeIn(g1), FadeIn(l1), Create(bar2), Create(bar3),
                  FadeIn(p12), FadeIn(p23))
        labs = VGroup(
            ctext("P12", size=22, color=LINK_A).next_to(p12, LEFT, buff=0.2),
            ctext("P23", size=22, color=GOOD).next_to(p23, UP, buff=0.2))
        self.play(FadeIn(labs))
        line = DashedLine(P(-4.9, -0.75), P(-2.3, 0.75), color=ACCENT)
        self.play(Create(line))
        self.hold(1)
        # 假设 P13 不在线上 → 速度矛盾
        bad = Dot(P(-1.2, 1.6), radius=0.08, color=BAD)
        bl = ctext("P13？", size=22, color=BAD).next_to(bad, UP, buff=0.15)
        v2 = vec(bad.get_center(), DOWN * 0.7 + RIGHT * 0.5, BAD)
        v3 = vec(bad.get_center(), UP * 0.7 + LEFT * 0.5, LINK_C)
        self.play(FadeIn(bad), Write(bl), FadeIn(VGroup(v2, v3)))
        contra = ctext("对构件2、3 速度方向不同 → 不是瞬心！", size=25,
                       color=BAD).shift(RIGHT * 2.8 + UP * 0.4)
        self.play(Write(contra))
        self.hold(2)
        ok = Dot(P(-4.9, -0.75) + 0.72 * (P(-2.3, 0.75) - P(-4.9, -0.75)),
                 radius=0.09, color=ACCENT)
        ol = ctext("P13 只能在连线（延长线）上", size=25,
                   color=ACCENT).next_to(contra, DOWN, buff=0.4)
        self.play(FadeIn(ok), Write(ol))
        self.focus(ok)
        self.takeaway("三个构件的三个瞬心必在同一直线上", p="孙桓八版 p38-39")
        self.hold(3)


class S04_ICVelocity(LessonScene):
    """瞬心求速度（~10min, p40-42）：四杆机构六瞬心全标→用 P13 求传动比。
    动画：先标 P12/P23/P34/P14，再按三心定理两条连线交得 P13、P24。"""

    def construct(self):
        self.header("瞬心法实战", "四杆机构求传动比 · p40-42")
        fb = FourBar(4.4, 1.3, 3.4, 3.0, origin=np.array([-3.2, -1.6, 0]))
        A0, A, B, B0 = fb.solve(0.85)
        mech = VGroup(link_line(A0, A, LINK_A), link_line(A, B, LINK_B),
                      link_line(B, B0, LINK_C), link_line(A0, B0, FRAME_C, 5),
                      ground_hatch(A0 + DOWN * 0.16, 0.6),
                      ground_hatch(B0 + DOWN * 0.16, 0.6),
                      *[pin_joint(p) for p in (A0, A, B, B0)])
        self.play(FadeIn(mech))
        direct = [(A0, "P14"), (A, "P12"), (B, "P23"), (B0, "P34")]
        for p, name in direct:
            d = Dot(p, radius=0.09, color=LINK_A)
            self.play(FadeIn(d), Write(ctext(name, size=20, color=LINK_A)
                                       .next_to(p, UP + LEFT, buff=0.08)),
                      run_time=0.5)
        self.hold(1)
        # P13：P12P23 连线 ∩ P14P34 连线
        l_a = DashedLine(A0, B0, color=MUTED)                     # P14P34 线
        l_b = DashedLine(A + (A - B) * 1.2, B + (B - A) * 0.4, color=MUTED)
        p13 = self._intersect(A0, B0, A, B)
        self.play(Create(l_a), Create(l_b))
        d13 = Dot(p13, radius=0.1, color=ACCENT)
        t13 = ctext("P13", size=24, color=ACCENT).next_to(p13, DOWN, buff=0.2)
        self.play(FadeIn(d13), Write(t13))
        self.hold(1.5)
        eq = mtex(r"\frac{\omega_3}{\omega_1}=\frac{P_{14}P_{13}}{P_{34}P_{13}}",
                     font_size=48, color=ACCENT).to_edge(DOWN, buff=0.9)
        self.play(Write(eq))
        self.takeaway("P13 上两构件同速 → 传动比 = 瞬心分连心线两段反比",
                        p="孙桓八版 p40-42")
        self.hold(3)

    @staticmethod
    def _intersect(p1, p2, p3, p4):
        d = (p2[0] - p1[0]) * (p4[1] - p3[1]) - (p2[1] - p1[1]) * (p4[0] - p3[0])
        t = ((p3[0] - p1[0]) * (p4[1] - p3[1])
             - (p3[1] - p1[1]) * (p4[0] - p3[0])) / d
        return p1 + t * (p2 - p1)


class S05_RelativeVelocity(LessonScene):
    """相对运动图解法（~12min, p43-46）：v_B=v_A+v_BA 矢量三角形现场作图——
    机构位形→速度图（极点 p、影像原理）；数值与机构动画同步。"""

    def construct(self):
        self.header("相对运动图解法", "速度多边形 · p43-46")
        fb = FourBar(4.0, 1.2, 3.0, 2.6, origin=np.array([-4.8, -1.5, 0]))
        A0, A, B, B0 = fb.solve(1.0)
        mech = VGroup(link_line(A0, A, LINK_A), link_line(A, B, LINK_B),
                      link_line(B, B0, LINK_C), link_line(A0, B0, FRAME_C, 5),
                      ground_hatch(A0 + DOWN * 0.16, 0.6),
                      ground_hatch(B0 + DOWN * 0.16, 0.6),
                      *[pin_joint(p) for p in (A0, A, B, B0)])
        self.play(FadeIn(mech))
        # 速度图（右侧）：v_A⊥曲柄、v_BA⊥连杆、v_B⊥摇杆
        pv = P(2.0, -1.55)                      # 极点 p
        vA = np.array([A[1] - A0[1], -(A[0] - A0[0]), 0.0]) * 0.75  # ⊥A0A
        vBA = np.array([B[1] - A[1], -(B[0] - A[0]), 0.0])
        vBA = vBA / np.linalg.norm(vBA) * 1.2
        tri, pA, pB = vec_triangle(pv, vA, vBA)
        pe = Dot(pv, radius=0.07)
        pl = ctext("极点 p（速度零点）", size=22).next_to(pe, LEFT, buff=0.3)
        self.play(FadeIn(pe), Write(pl))
        # 标注贴各边中点外侧（远离质心一侧），教材式矢量图标注
        cent = (pv + pA + pB) / 3

        def edge_lab(txt, p, q, col):
            mid = (p + q) / 2
            d = q - p
            n = np.array([-d[1], d[0], 0.0])
            n /= np.linalg.norm(n)
            if np.dot(n, cent - mid) > 0:
                n = -n
            return ctext(txt, size=20, color=col).move_to(mid + n * 0.95)
        self.play(FadeIn(tri[0]), Write(edge_lab("vA ⊥ A₀A", pv, pA,
                                                 LINK_A)))   # v_A
        self.play(FadeIn(tri[1]), Write(edge_lab("vBA ⊥ AB", pA, pB,
                                                 LINK_D)))   # v_BA
        self.play(FadeIn(tri[2]), Write(edge_lab("vB ⊥ B₀B", pv, pB,
                                                 INK)))    # v_B
        eq = mtex(r"\vec{v}_B = \vec{v}_A + \vec{v}_{BA}",
                     font_size=46, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(eq))
        img = ctext("速度影像：图上三角形 ∽ 机构上对应构件形", size=24,
                    color=NOTE).next_to(eq, UP, buff=0.3)
        self.play(Write(img))
        self.add(page_ref("孙桓八版 p43-46"))
        self.hold(3)


class S06_Coriolis(LessonScene):
    """加速度与哥氏加速度（~10min, p47-50）：旋转导杆上的滑块——牵连转动+相对
    滑动 → 哥氏 a_k=2ω×v_r 方向演示（转 v_r 90°）。"""

    def construct(self):
        self.header("加速度图与哥氏加速度", "p47-50")
        # 旋转导杆 + 滑动块
        O = P(-4.2, -1.3)
        th = ValueTracker(0.5)

        def guide():
            ang = th.get_value()
            d = np.array([np.cos(ang), np.sin(ang), 0.0])
            end = O + d * 3.2
            blk_pos = O + d * 2.0
            return VGroup(
                link_line(O, end, LINK_C),
                slider_block(blk_pos, 0.5, 0.34, angle=ang, color=LINK_D),
                fixed_pin(O),
                Dot(O + d * 2.0, radius=0.06, color=INK),
            )
        mech = always_redraw(guide)
        self.play(FadeIn(mech))
        self.play(th.animate.set_value(1.1), run_time=2.5)
        self.hold(1)
        # 哥氏分解
        steps = [
            mtex(r"\vec{a}_B = \vec{a}_A + \vec{a}_{BA}^{n} + "
                    r"\vec{a}_{BA}^{t}", font_size=42),
            mtex(r"a^n = \omega^2 l\ (\text{指向转动中心})", font_size=36),
            mtex(r"a_k = 2\,\vec{\omega}\times\vec{v}_r"
                    r"\ (\text{哥氏，方向：}v_r\text{ 顺 }\omega\text{ 转 }90°)",
                    font_size=36, color=ACCENT),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 2.8 + UP * 0.4, wait=1.6)
        demo = ctext("滑块随杆转（牵连）+ 沿杆滑（相对）→ 哥氏加速度",
                     size=24, color=NOTE).to_edge(DOWN, buff=0.6)
        self.play(Write(demo))
        self.add(page_ref("孙桓八版 p47-50"))
        self.hold(3)


class S07_Analytical(LessonScene):
    """解析法一瞥（~6min, p50-54）：闭环矢量方程 → 分量方程 → 线性方程组解 ω。"""

    def construct(self):
        self.header("解析法速览", "矢量闭环 → 矩阵 · p50-54")
        steps = [
            mtex(r"\vec{l}_1 + \vec{l}_4 = \vec{l}_2 + \vec{l}_3",
                    font_size=44),
            mtex(r"l_2\cos\theta_2 + l_3\cos\theta_3 - l_4\cos\theta_4 = l_1",
                    font_size=38),
            mtex(r"l_2\sin\theta_2 + l_3\sin\theta_3 - l_4\sin\theta_4 = 0",
                    font_size=38),
            mtex(r"\begin{bmatrix} -l_3\sin\theta_3 & l_4\sin\theta_4\\"
                    r" l_3\cos\theta_3 & -l_4\cos\theta_4\end{bmatrix}"
                    r"\begin{bmatrix}\omega_3\\ \omega_4\end{bmatrix} = "
                    r"\omega_2\begin{bmatrix} l_2\sin\theta_2\\ -l_2\cos\theta_2"
                    r"\end{bmatrix}", font_size=40, color=ACCENT),
        ]
        formula_reveal(self, steps, anchor=UP * 0.4, buff=0.42, wait=1.6)
        note = ctext("本课前面所有动画，正是这套方程的数值解在驱动",
                     size=26, color=NOTE).to_edge(DOWN, buff=0.7)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p50-54"))
        self.hold(3)


class S08_Summary(LessonScene):
    """小结+下讲悬念（~3min）：瞬心快/图解全/解析狠；下讲：力从哪来？"""

    def construct(self):
        self.header("第 3 讲小结")
        pts = bullets([
            "瞬心=瞬时等速点；三心定理定位           (p36-39)",
            "瞬心法：求传动比最快                     (p40-42)",
            "相对运动图解法：速度/加速度多边形+影像原理  (p43-50)",
            "哥氏加速度：转动+滑动并存时出现 2ω×vr    (p47-50)",
            "解析法：闭环矢量方程求导 → 线性方程组      (p50-54)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：机构里的力有多大？——摩擦与效率",
                  size=25, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
