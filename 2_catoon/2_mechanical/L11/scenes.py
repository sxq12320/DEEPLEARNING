# -*- coding: utf-8 -*-
"""L11 最完美的曲线(下)——范成·根切·变位·空间齿轮（§10-6~10-10, p208-236）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


def rack_tooth_pts(x0, pitch=0.5, h=0.42, tip=0.16):
    """单个齿条齿廓形（梯形齿）：返回四点折线。"""
    return [P(x0, 0), P(x0 + tip, h), P(x0 + pitch / 2 - tip, h),
            P(x0 + pitch / 2, 0)]


class S01_CutMethods(LessonScene):
    """切齿两路线（~6min, p208-210）：仿形法（成形铣刀，低精度）vs
    范成法（展成，滚刀/插齿刀，高精度+一把刀切所有齿数）。"""

    def construct(self):
        self.header("渐开线怎么切出来？", "仿形 vs 范成 · p208-210")
        cards = VGroup(
            VGroup(ctext("仿形法", size=30, weight="BOLD"),
                   bullets(["盘形/指形铣刀照搬槽形", "精度低、每种 z 要换刀"],
                           size=23)),
            VGroup(ctext("范成法（展成）", size=30, weight="BOLD", color=ACCENT),
                   bullets(["刀具与坯按啮合关系对滚", "一把刀切所有齿数",
                            "插齿刀/滚刀/齿条刀"], size=23)),
        )
        for c in cards:
            c.arrange(DOWN, aligned_edge=LEFT, buff=0.35)
            box = SurroundingRectangle(c, color=FRAME_C, buff=0.3)
            c.add(box)
        cards.arrange(RIGHT, buff=1.0).shift(UP * 0.3)
        self.play(FadeIn(cards, lag_ratio=0.4), run_time=2.5)
        self.hold(2.5)
        self.takeaway("范成法：刀刃轨迹的包络线 = 渐开线", p="孙桓八版 p208-210")
        self.hold(3)


class S02_Generating(LessonScene):
    """范成动画（~10min 本讲最美镜头, p210-212）：齿条刀沿节线纯滚动——
    逐帧叠加刀位，渐开线齿廓自然'包络'显现。"""

    def construct(self):
        self.header("范成法：包络出渐开线", "齿条刀滚切 · p210-212")
        O = P(-2.4, -1.5)
        m_t, z_t = 0.2, 19                     # 刀与坯同模数
        r = m_t * z_t / 2                      # 坯分度圆 = 1.9
        blank = Circle(radius=r + m_t, color=FRAME_C, stroke_width=3,
                       fill_opacity=0.06, fill_color=FRAME_C).move_to(O)
        pitch_ln = Line(O + LEFT * 3.2 + UP * r, O + RIGHT * 3.2 + UP * r,
                        color=MUTED, stroke_width=2)
        self.play(Create(blank), Create(pitch_ln), FadeIn(pin_joint(O)))
        self.play(Write(ctext("节线 = 分度圆切线；刀速 v = ω·r 纯滚动",
                              size=24).to_edge(UP, buff=1.7)))
        # 齿条刀多位置叠加（齿尖朝下切入坯料，刀体在节线上方）
        pitch = PI * m_t
        positions = np.linspace(-2.2, 2.2, 9)
        ghost = VGroup()
        for xp in positions:
            pts = []
            for k in range(-2, 3):
                tp = rack_tooth_pts(xp + k * pitch, pitch=pitch)
                pts += [[p[0], -p[1], 0] for p in tp]
            pts += [P(xp + 3 * pitch, 0), P(xp + 3.4 * pitch, 0.5),
                    P(xp - 2.6 * pitch, 0.5), P(xp - 2 * pitch, 0)]
            vm = VMobject(color=LINK_D, stroke_width=1.6,
                          stroke_opacity=0.5)
            vm.set_points_as_corners(pts)
            vm.shift(O + UP * r)
            ghost.add(vm)
        self.play(FadeIn(ghost, lag_ratio=0.3), run_time=3)
        self.hold(1.5)
        # 成品轮廓浮现（与坯分度圆、模数一致）
        gear = gear_profile(m_t, z_t, color=GEAR_1, stroke_width=3.5)
        gear.move_to(O)
        self.play(Create(gear), run_time=2.5)
        note = ctext("刀位族的包络 = 渐开线齿廓", size=26,
                     color=GOOD).to_edge(DOWN, buff=0.55).shift(RIGHT * 1.4)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p210-212"))
        self.hold(3)


class S03_Undercut(LessonScene):
    """根切（~8min, p213-214）：刀具齿顶线超过啮合极限点→已切成的渐开线被
    '啃掉'；根切特写对比正常齿。"""

    def construct(self):
        self.header("根切：多切了一刀", "p213-214")
        # 左正常齿，右根切齿（示意：齿根处内凹）
        g_ok = gear_profile(0.14, 20, color=GOOD, stroke_width=3)
        g_ok.move_to(LEFT * 3.4 + DOWN * 0.4)
        g_bad = gear_profile(0.14, 10, color=BAD, stroke_width=3)
        g_bad.move_to(RIGHT * 3.0 + DOWN * 0.4)
        self.play(Create(g_ok), Write(ctext("z=20：齿根饱满", size=24,
                                            color=GOOD).next_to(g_ok, UP,
                                                                buff=0.4)))
        self.play(Create(g_bad), Write(ctext("z=10：根部被挖去一块", size=24,
                                             color=BAD).next_to(g_bad, UP,
                                                                buff=0.4)))
        zoom = Circle(radius=1.0, color=BAD, stroke_width=2).move_to(
            g_bad.get_center() + DOWN * 1.1)
        self.play(Create(zoom), Write(ctext("根切区", size=22, color=BAD)
                                      .next_to(zoom, RIGHT, buff=0.25)))
        pts = bullets([
            "原因：齿数太少 → 刀顶线越过啮合极限点 N",
            "后果：根部变弱、重合度下降",
        ], size=25).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p213-214"))
        self.hold(3)


class S04_ZminShifted(LessonScene):
    """z_min=17 与变位（~12min, p215-220）：推导 z_min=2h_a*/sin²α=17；
    变位齿轮族——x 从负到正齿形渐变动画；正变位防根切。"""

    def construct(self):
        self.header("最少 17 齿？变位来救场", "zmin 与变位 · p215-220")
        steps = [
            mtex(r"z_{min}=\frac{2h_a^{*}}{\sin^2\alpha}"
                    r"=\frac{2}{\sin^2 20°}\approx 17", font_size=46,
                    color=ACCENT),
            mtex(r"\text{变位：刀具远离/靠近轮心 }xm", font_size=40),
        ]
        formula_reveal(self, steps, anchor=UP * 1.15, wait=1.6)
        # 变位族 x = -0.4, 0, +0.5
        xs = [-0.4, 0.0, 0.5]
        labs = ["x=-0.4", "x=0", "x=+0.5"]
        gears = VGroup()
        for i, x in enumerate(xs):
            g = gear_profile(0.16, 14, x=x,
                             color=[BAD, INK, GOOD][i], stroke_width=3)
            g.move_to(P(-3.6 + i * 3.4, -0.85))
            gears.add(VGroup(g, ctext(labs[i], size=24,
                                      color=[BAD, INK, GOOD][i])
                             .next_to(g, DOWN, buff=0.15)))
        self.play(FadeIn(gears, lag_ratio=0.5), run_time=2.5)
        self.hold(2)
        pts = bullets([
            "正变位 x>0：齿顶变尖、齿根变厚 → 防根切",
            "负变位 x<0：凑中心距用（要小心根切）",
        ], size=24).to_edge(DOWN, buff=0.35)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p215-220"))
        self.hold(3)


class S05_ShiftApps(LessonScene):
    """变位的三大用途（~6min, p220-224）：防根切、凑中心距、强化小齿轮/
    均磨损——齿条刀移动距离 xm 的几何含义回顾。"""

    def construct(self):
        self.header("变位齿轮的三张牌", "p220-224")
        cards = VGroup()
        for i, (t, d) in enumerate([
                ("防根切", "z<17 也能造\n正变位加厚齿根"),
                ("凑中心距", "a'≠a 时\n用变位补偿"),
                ("均寿命", "小轮正变位强化\n大轮微调")]):
            box = RoundedRectangle(corner_radius=0.15, width=3.8, height=1.5,
                                   color=LINK_B, fill_opacity=0.12,
                                   fill_color=LINK_B)
            cards.add(VGroup(box, ctext(t, size=26, weight="BOLD",
                                        color=ACCENT).next_to(box, UP,
                                                              buff=0.12),
                             ctext(d, size=19).move_to(box)))
        cards.arrange(RIGHT, buff=0.55).shift(UP * 0.5)
        self.play(FadeIn(cards, lag_ratio=0.4), run_time=2.5)
        self.hold(2.5)
        self.add(page_ref("孙桓八版 p220-224"))
        self.hold(2)


class S06_Helical(LessonScene):
    """斜齿轮（~8min, p224-229）：螺旋角 β——啮合渐进、重合度大、更平稳；
    法面参数（标准）vs 端面参数；当量齿数 z_v=z/cos³β。"""

    def construct(self):
        self.header("斜齿轮：把直齿轮'拧'一下", "p224-229")
        # 示意：斜齿条齿向斜线
        gear = Rectangle(width=3.6, height=2.2, color=GEAR_1,
                         fill_opacity=0.12, stroke_width=4)
        gear.shift(LEFT * 3.4 + UP * 0.2)
        teeth = VGroup(*[Line(P(-5.0 + i * 0.5, -0.7), P(-4.7 + i * 0.5, 1.1),
                              color=GEAR_1, stroke_width=2.5)
                         for i in range(7)])
        beta = angle_mark(P(-5.0, -0.7), 0, np.arctan2(1.8, 0.3), r=0.6,
                          label="β", color=ACCENT)
        self.play(FadeIn(gear), FadeIn(teeth), FadeIn(beta))
        self.play(Write(ctext("齿向与轴线成螺旋角 β：啮合从点接触渐进成线",
                              size=24).next_to(gear, UP, buff=0.5)))
        steps = [
            mtex(r"\text{法面 }m_n,\alpha_n\text{ 为标准}"
                    r";\quad m_t = m_n/\cos\beta", font_size=40),
            mtex(r"z_v = z/\cos^3\beta\ (\text{当量齿数，选刀用})",
                    font_size=40),
            mtex(r"\varepsilon_\gamma = \varepsilon_\alpha + "
                    r"\varepsilon_\beta\ (\text{更大更稳})", font_size=40,
                    color=ACCENT),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 2.9 + DOWN * 0.4,
                       wait=1.6)
        self.add(page_ref("孙桓八版 p224-229"))
        self.hold(3)


class S07_BevelWorm(LessonScene):
    """锥齿轮·蜗杆蜗轮·交错轴（~6min, p229-236）：转向关系、背锥/当量齿数、
    蜗杆大传动比+自锁——三张小卡片。"""

    def construct(self):
        self.header("空间齿轮三杰", "锥齿轮 · 蜗杆蜗轮 · p229-236")
        cards = VGroup()
        for t, d in [("锥齿轮", "相交轴传动\n背锥当量齿数 zv=z/cosδ"),
                     ("蜗杆蜗轮", "交错90°大减速比 i=z2/z1\n反行程自锁"),
                     ("交错轴斜齿", "点接触承载小\n只传运动不传大力")]:
            box = RoundedRectangle(corner_radius=0.15, width=4.0, height=1.6,
                                   color=LINK_B, fill_opacity=0.12,
                                   fill_color=LINK_B)
            cards.add(VGroup(box, ctext(t, size=26, weight="BOLD",
                                        color=ACCENT).next_to(box, UP,
                                                              buff=0.12),
                             ctext(d, size=19).move_to(box)))
        cards.arrange(RIGHT, buff=0.5).shift(UP * 0.4)
        self.play(FadeIn(cards, lag_ratio=0.4), run_time=2.5)
        self.hold(2.5)
        self.add(page_ref("孙桓八版 p229-236"))
        self.hold(2)


class S08_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 11 讲小结")
        pts = bullets([
            "范成法：刀与坯对滚，包络出渐开线          (p208-212)",
            "根切：z 太少刀顶越线；zmin=17            (p213-215)",
            "变位 xm：防根切/凑中心距/均寿命           (p215-224)",
            "斜齿：螺旋角 β、当量齿数、啮合更平稳       (p224-229)",
            "锥齿交轴、蜗杆大减速可自锁                (p229-236)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：齿轮成排成系——轮系与传动比的艺术", size=25,
                  color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
