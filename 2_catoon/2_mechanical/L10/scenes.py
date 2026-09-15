# -*- coding: utf-8 -*-
"""L10 最完美的曲线(上)——渐开线与标准齿轮（§10-1~10-5, p195-208）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_MeshingLaw(LessonScene):
    """齿廓啮合基本定律（~10min, p195-197）：任意齿廓接触点公法线须过定点 P
    （节点）→ 定传动比；i = O₂P/O₁P；节圆=以 OP 为半径的摩擦轮。"""

    def construct(self):
        self.header("齿轮的第一性原理", "齿廓啮合基本定律 · p195-197")
        O1, O2 = P(-2.6, -0.4), P(1.4, -0.4)
        Pn = P(-0.4, -0.4)                      # 节点
        # 两节圆（相切于 P）
        pc1 = DashedVMobject(Circle(radius=2.2, color=MUTED,
                                    stroke_width=1.5),
                             num_dashes=64).move_to(O1)
        pc2 = DashedVMobject(Circle(radius=1.8, color=MUTED,
                                    stroke_width=1.5),
                             num_dashes=56).move_to(O2)
        self.play(FadeIn(pin_joint(O1)), FadeIn(pin_joint(O2)),
                  Create(pc1), Create(pc2))
        # 公法线：过 P、相对竖直倾 20°；接触点 K 在其上
        n = np.array([np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20)), 0.0])
        K = Pn + n * 0.85
        nline = DashedLine(Pn - n * 1.9, Pn + n * 1.6, color=ACCENT)
        # 两齿廓：圆心在法线上的两段弧 → 在 K 相切（公切线 ⊥ 法线）
        a1 = np.arctan2(-n[1], -n[0])
        a2 = np.arctan2(n[1], n[0])
        f1 = Arc(radius=0.95, start_angle=a1 - 0.6, angle=1.2,
                 arc_center=K + n * 0.95, color=LINK_B, stroke_width=6)
        f2 = Arc(radius=0.7, start_angle=a2 - 0.6, angle=1.2,
                 arc_center=K - n * 0.7, color=LINK_C, stroke_width=6)
        self.play(Create(nline))
        self.play(Create(f1), Create(f2),
                  FadeIn(Dot(K, radius=0.08, color=BAD)))
        self.play(Write(ctext("接触点 K", size=22, color=BAD).next_to(
            K, UP + RIGHT, buff=0.2)))
        self.play(FadeIn(Dot(Pn, radius=0.1, color=ACCENT)),
                  Write(ctext("节点 P（两节圆切点）", size=23,
                              color=ACCENT).next_to(Pn, DOWN, buff=0.35)))
        self.play(Write(ctext("接触点公法线必过定点 P → i 恒定", size=26)
                        .to_edge(UP, buff=1.7)))
        eq = mtex(r"i_{12}=\frac{\omega_1}{\omega_2}=\frac{O_2P}{O_1P}",
                     font_size=52, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(eq))
        self.add(page_ref("孙桓八版 p195-197"))
        self.hold(3)


class S02_InvoluteBirth(LessonScene):
    """渐开线生成（~10min, p199-200）：绳绕基圆展开——发生线逐帧展开，
    端点描出渐开线；五条性质逐条验证。"""

    def construct(self):
        self.header("渐开线：一根绳子绷出来的曲线", "生成与性质 · p199-200")
        rb = 1.5
        O = P(-3.0, -0.7)
        base = Circle(radius=rb, color=NOTE, stroke_width=4).move_to(O)
        self.play(Create(base), FadeIn(pin_joint(O)),
                  Write(ctext("基圆 rb", size=24, color=NOTE).next_to(
                      base, DOWN, buff=0.4)))
        t_max = 2.4
        inv = involute_pts(rb, t_max, n=80)
        # 转动渐开线使其顶点向上
        rot = PI / 2
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        inv = inv @ R.T + O
        curve = VMobject(color=ACCENT, stroke_width=4)
        curve.set_points_smoothly(inv)
        # 发生线动画
        tt = ValueTracker(0.0)

        def gen_line():
            tv = tt.get_value() * t_max
            pt0 = O + np.array([rb, 0, 0]) @ np.array(
                [[np.cos(rot), np.sin(rot), 0], [-np.sin(rot), np.cos(rot), 0],
                 [0, 0, 1]])
            tan_pt = O + np.array([rb * np.cos(rot + tv),
                                   rb * np.sin(rot + tv), 0])
            direc = np.array([np.sin(rot + tv), -np.cos(rot + tv), 0])
            end = tan_pt + direc * rb * tv
            return Line(tan_pt, end, color=GOOD, stroke_width=4)
        gline = always_redraw(gen_line)
        tip = always_redraw(lambda: Dot(
            gen_line().get_end(), radius=0.07, color=GOOD))
        self.play(FadeIn(gline), FadeIn(tip))
        # 镜头推进到绳圆接触区，看切点如何扫出曲线
        self.zoom_to(O + UP * rb * 0.55 + LEFT * 0.4, scale=0.6, run_time=1.6)
        self.play(tt.animate.set_value(1.0), Create(curve), run_time=6,
                  rate_func=linear)
        self.zoom_reset()
        self.play(FadeIn(glow(curve)), run_time=1.2)
        self.hold(1.5)
        props = bullets([
            "发生线 = 渐开线法线 = 基圆切线",
            "离基圆越远，压力角越大",
            "基圆以内没有渐开线",
        ], size=24).shift(RIGHT * 3.4 + UP * 0.2)
        self.play(FadeIn(props, lag_ratio=0.5))
        self.emphasize(gline)          # “发生线=法线”时点一下发生线
        self.add(page_ref("孙桓八版 p199-200"))
        self.hold(3)


class S03_InvoluteVirtues(LessonScene):
    """渐开线啮合三美德（~10min, p200-202）：啮合线=两基圆内公切线（直线）；
    传动比恒定；中心距可分性——拉开中心距 i 不变（基圆不变）。"""

    def construct(self):
        self.header("渐开线啮合的三美德", "p200-202")
        gp = AnimatedGearPair(12, 20, m=0.11, c1=np.array([-2.9, -1.0, 0]))
        self.play(FadeIn(gp.group))
        # 啮合线：两基圆内公切线
        self.play(gp.theta.animate.set_value(2 * TAU), run_time=5,
                  rate_func=linear)
        mid = (gp.c1 + gp.c2) / 2          # 节点
        nd = np.array([np.sin(np.deg2rad(20)), np.cos(np.deg2rad(20)), 0.0])
        nline = DashedLine(mid - nd * 2.3, mid + nd * 2.3, color=ACCENT)
        lab = ctext("啮合线：两基圆内公切线（一条直线！）", size=25,
                    color=ACCENT).next_to(gp.group, UP, buff=0.4)
        self.play(Create(nline), Write(lab))
        self.hold(2)
        pts = bullets([
            "i 恒定：ω₁/ω₂ = rb2/rb1 = z2/z1",
            "中心距可分：装偏一点，i 仍不变（基圆没变）",
            "啮合角 α' = 法线与速度方向夹角，恒定",
        ], size=25).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p200-202"))
        self.hold(3)


class S04_StandardGear(LessonScene):
    """标准齿轮几何词典（~12min, p202-205）：一只真渐开线齿轮上标注
    m/z/α=20°/d=mz/da/df/s=e=πm/2/基圆——逐个点亮。"""

    def construct(self):
        self.header("一只标准齿轮的'身份证'", "基本参数 · p202-205")
        m, z = 0.13, 16
        g = gear_profile(m, z, color=GEAR_1, stroke_width=3)
        g.move_to(LEFT * 4.0 + DOWN * 0.5)
        O = g.get_center()
        r, rb = m * z / 2, m * z / 2 * np.cos(np.deg2rad(20))
        ra, rf = r + m, r - 1.25 * m
        self.play(Create(g))
        circs = VGroup(
            DashedVMobject(Circle(radius=ra, color=LINK_D, stroke_width=2),
                           num_dashes=44).move_to(O),
            Circle(radius=r, color=ACCENT, stroke_width=2.5).move_to(O),
            DashedVMobject(Circle(radius=rb, color=NOTE, stroke_width=2),
                           num_dashes=44).move_to(O),
            DashedVMobject(Circle(radius=rf, color=MUTED, stroke_width=2),
                           num_dashes=44).move_to(O),
        )
        self.play(FadeIn(circs, lag_ratio=0.3))
        pitch_lab = ctext("d=mz 分度圆", size=21, color=ACCENT).move_to(
            O + np.array([1.75, -1.15, 0]))
        leader = Line(O + np.array([r * np.cos(-0.6), r * np.sin(-0.6), 0]),
                      pitch_lab.get_center() + UP * 0.28 + LEFT * 0.9,
                      color=ACCENT, stroke_width=1.5)
        labels = VGroup(
            ctext("da 齿顶圆", size=21, color=LINK_D).next_to(circs[0], UP,
                                                              buff=0.12),
            pitch_lab,
            ctext("db 基圆", size=21, color=NOTE).next_to(circs[2], LEFT,
                                                           buff=0.35),
            ctext("df 齿根圆", size=21, color=MUTED).next_to(circs[3], DOWN,
                                                              buff=0.12),
            leader,
        )
        self.play(FadeIn(labels, lag_ratio=0.4))
        self.hold(1.5)
        dict_ = VGroup(
            mtex(r"d=mz,\ \ d_b=d\cos\alpha", font_size=36),
            mtex(r"d_a=d+2h_a^{*}m,\ \ d_f=d-2(h_a^{*}+c^{*})m",
                    font_size=36),
            mtex(r"s=e=\pi m/2,\ \ p=\pi m,\ \ \alpha=20°", font_size=36),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(RIGHT * 2.6)
        formula_reveal(self, dict_, anchor=RIGHT * 2.6 + UP * 0.3, wait=1.3)
        self.takeaway("模数 m 是'字号'：m 定下来，全部尺寸都定", p="孙桓八版 p202-205")
        self.hold(3)


class S05_CorrectMesh(LessonScene):
    """正确啮合与标准中心距（~8min, p205-207）：m1=m2、α1=α2；
    a=m(z1+z2)/2——装对了分度圆相切。"""

    def construct(self):
        self.header("怎样才算'配得上'", "啮合条件与中心距 · p205-207")
        gp = AnimatedGearPair(14, 22, m=0.1, c1=np.array([-2.6, -0.9, 0]))
        self.play(FadeIn(gp.group))
        self.play(gp.theta.animate.set_value(TAU), run_time=4,
                  rate_func=linear)
        steps = [
            mtex(r"m_1 = m_2,\quad \alpha_1 = \alpha_2", font_size=46,
                    color=ACCENT),
            mtex(r"a = \frac{m(z_1+z_2)}{2}\ (\text{分度圆相切})",
                    font_size=44),
            mtex(r"\text{顶隙 }c = c^{*}m\ \text{防止咬死}", font_size=38),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 3.0 + UP * 0.5, wait=1.6)
        self.add(page_ref("孙桓八版 p205-207"))
        self.hold(3)


class S06_ContactRatio(LessonScene):
    """重合度 ε（~10min, p207-208）：啮合区间 B1B2 上一对齿'交接班'动画——
    ε=B1B2/p_b>1 才能保证连续传动；ε=1.35 意味着 35% 时间双齿啮合。"""

    def construct(self):
        self.header("接力赛：重合度 ε", "连续传动条件 · p207-208")
        gp = AnimatedGearPair(14, 22, m=0.11, c1=np.array([-2.7, -1.0, 0]))
        self.play(FadeIn(gp.group))
        self.play(gp.theta.animate.set_value(2 * TAU), run_time=6,
                  rate_func=linear)
        mid = (gp.c1 + gp.c2) / 2
        zone = SurroundingRectangle(
            VGroup(Dot(mid + UP * 0.8), Dot(mid + DOWN * 0.5)),
            color=ACCENT, buff=0.3)
        self.play(Create(zone))
        self.play(Write(ctext("啮合区间 B₁B₂：齿对在此交接", size=24,
                              color=ACCENT).move_to(P(3.4, 0.9))),
                  Create(Arrow(P(1.15, 0.72), mid + P(0.5, 1.0),
                               buff=0.1, color=ACCENT, stroke_width=2,
                               max_tip_length_to_length_ratio=0.18)))
        steps = [
            mtex(r"\varepsilon = \frac{B_1B_2}{p_b} > 1",
                    font_size=50, color=ACCENT),
            mtex(r"\varepsilon=1.35:\ 35\%\text{ 时间两对齿分担载荷}",
                    font_size=36),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 3.4 + DOWN * 1.0,
                       wait=1.8)
        self.add(page_ref("孙桓八版 p207-208"))
        self.hold(3)


class S07_RackInternal(LessonScene):
    """齿条与内齿轮（~5min, p208 附近）：齿条=半径∞的齿轮（直线齿廓）；
    内啮合=小齿轮在大环内同向转。"""

    def construct(self):
        self.header("齿轮家族的亲戚", "齿条 · 内齿轮")
        # 齿条
        rack_pts = []
        for i in range(6):
            x = i * 0.62
            rack_pts += [[x, 0, 0], [x + 0.18, 0.28, 0], [x + 0.44, 0.28, 0],
                         [x + 0.62, 0, 0]]
        rack = VMobject(color=LINK_C, stroke_width=4)
        rack.set_points_as_corners(rack_pts)
        rack.shift(LEFT * 4.2 + UP * 0.6)
        g1 = gear_profile(0.11, 12, color=GEAR_1).move_to(LEFT * 2.6
                                                         + UP * 0.05)
        self.play(Create(rack), Create(g1))
        self.play(Rotate(g1, TAU / 3, about_point=g1.get_center()),
                  rack.animate.shift(RIGHT * 0.6), run_time=3,
                  rate_func=linear)
        t1 = ctext("齿条：半径→∞，渐开线变直线", size=24,
                   color=LINK_C).next_to(rack, UP, buff=0.5)
        self.play(Write(t1))
        # 内齿轮
        ring = VGroup(Circle(radius=1.5, color=GEAR_2, stroke_width=5),
                      Circle(radius=1.32, color=GEAR_2, stroke_width=2,
                             stroke_opacity=0.6)).move_to(RIGHT * 3.6
                                                          + UP * 0.3)
        g2 = gear_profile(0.1, 11, color=GEAR_1).move_to(
            RIGHT * 3.6 + UP * 0.3 + RIGHT * 0.55)
        self.play(FadeIn(ring), Create(g2))
        self.play(Rotate(g2, TAU / 2, about_point=RIGHT * 3.6 + UP * 0.3),
                  Rotate(ring, TAU / 2 * 11 / 30,
                         about_point=RIGHT * 3.6 + UP * 0.3),
                  run_time=3, rate_func=linear)
        t2 = ctext("内啮合：转向相同", size=24, color=GEAR_2).next_to(
            ring, DOWN, buff=0.4)
        self.play(Write(t2))
        self.hold(2)
        self.add(page_ref("孙桓八版 第10章"))
        self.hold(2)


class S08_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 10 讲小结")
        pts = bullets([
            "啮合定律：公法线过节点 → i=O₂P/O₁P        (p195-197)",
            "渐开线=绳展基圆；法线=基圆切线            (p199-200)",
            "三美德：定 i / 中心距可分 / 啮合线是直线    (p200-202)",
            "标准齿轮：d=mz, α=20°, s=e=πm/2          (p202-205)",
            "正确啮合 m、α 相等；重合度 ε>1            (p205-208)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：渐开线怎么'切'出来？——范成·根切·变位",
                  size=25, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
