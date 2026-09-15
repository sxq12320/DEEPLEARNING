# -*- coding: utf-8 -*-
"""L07 四根杆的智慧(上)——连杆机构类型与特性（§8-1~8-3, p123-139）

Grashof 曲柄存在条件 / 演化谱系 / 急回特性 / 压力角与死点。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """片头（~4min）：连杆机构无处不在——雨刮/破碎机/缝纫机/挖掘机图片式
    快剪（用简笔机构图代表）。"""

    def construct(self):
        self.header("四根杆的智慧", "平面连杆机构 · p123")
        items = bullets([
            "全是低副：承载大、耐磨损、易制造",
            "能走'花路'：连杆上一点可画出任意复杂曲线",
            "代价：设计比凸轮难、平衡差——但无处不在",
        ], size=30).shift(UP * 0.3)
        self.play(FadeIn(items, lag_ratio=0.5), run_time=2.5)
        fb = FourBar(2.8, 0.9, 2.2, 2.0, origin=np.array([-1.4, -3.0, 0]))
        m = AnimatedFourBar(fb, coupler=(0.5, 0.5), trace=True)
        self.play(FadeIn(m.group))
        self.play(m.theta.animate.set_value(0.6 + TAU), run_time=5,
                  rate_func=linear)
        self.hold(1)
        self.takeaway("连杆机构：全部由低副连接的机构", p="孙桓八版 p123")
        self.hold(2)


class S02_ThreeTypes(LessonScene):
    """铰链四杆三基本型（~10min, p124-128）：曲柄摇杆/双曲柄/双摇杆
    三机同屏同步运转对比——同一根'最短杆'三种命运。"""

    def construct(self):
        self.header("铰链四杆的三种基本型", "p124-128")
        params = [(3.0, 0.85, 2.6, 2.4, "曲柄摇杆（最短杆=连架杆）", -4.6),
                  (2.2, 1.1, 2.5, 2.3, "双曲柄（最短杆=机架）", -0.6),
                  (3.2, 1.3, 0.9, 2.6, "双摇杆（最短杆=连杆）", 3.6)]
        mechs = []
        for L1, L2, L3, L4, name, x in params:
            fb = FourBar(L1, L2, L3, L4, origin=np.array([x, -1.5, 0]))
            m = AnimatedFourBar(fb)
            mechs.append(m)
            self.add(m.group)
            self.add(ctext(name, size=21).next_to(m.group, UP, buff=0.3))
        anims = []
        for i, m in enumerate(mechs):
            span = TAU if i < 2 else 0.9
            if i == 2:   # 双摇杆：两连架杆都摆
                anims.append(m.theta.animate.set_value(
                    m.theta.get_value() + 1.2))
            else:
                anims.append(m.theta.animate.set_value(
                    m.theta.get_value() + TAU))
        self.play(*anims, run_time=6, rate_func=linear)
        self.hold(1.5)
        self.takeaway("谁是'最短杆'、谁当机架 → 决定机构性格",
                        p="孙桓八版 p124-128")
        self.hold(3)


class S03_Grashof(LessonScene):
    """曲柄存在条件推导（~12min, p131-133）：曲柄能整周转 ⇔ 能过两'共线位形'
    ⇔ 最长+最短 ≤ 其余两杆之和。两位形作图推导 + 不等式得出。"""

    def construct(self):
        self.header("曲柄存在条件（Grashof）", "两位形推导 · p131-133")
        fb = FourBar(3.6, 1.0, 2.8, 2.4, origin=np.array([-4.2, -1.3, 0]))
        # 位形1：曲柄与机架共线（拉直）
        A0, A1, B1, B0 = fb.solve(0.0)
        pos1 = VGroup(link_line(A0, A1, LINK_A), link_line(A1, B1, LINK_B),
                      link_line(B1, B0, LINK_C), link_line(A0, B0, FRAME_C, 5),
                      *[pin_joint(p) for p in (A0, A1, B1, B0)],
                      ground_hatch(A0 + DOWN * 0.16, 0.6),
                      ground_hatch(B0 + DOWN * 0.16, 0.6))
        self.play(FadeIn(pos1))
        t1 = ctext("位形①：曲柄与机架共线（外拉）", size=25).to_edge(UP,
                                                                     buff=1.7)
        self.play(Write(t1))
        ineq1 = mtex(r"|A_0B| = l_3 + l_2 \le l_1 + l_4",
                        font_size=42).to_edge(DOWN, buff=1.1)
        self.play(Write(ineq1))
        self.hold(1.5)
        # 位形2：曲柄与机架共线（内收）
        A0, A2, B2, B0 = fb.solve(PI)
        pos2 = VGroup(link_line(A0, A2, LINK_A), link_line(A2, B2, LINK_B),
                      link_line(B2, B0, LINK_C), link_line(A0, B0, FRAME_C, 5),
                      *[pin_joint(p) for p in (A0, A2, B2, B0)],
                      ground_hatch(A0 + DOWN * 0.16, 0.6),
                      ground_hatch(B0 + DOWN * 0.16, 0.6))
        self.play(Transform(pos1, pos2))
        t2 = ctext("位形②：曲柄与机架共线（内收）", size=25).to_edge(UP,
                                                                     buff=1.7)
        self.play(Transform(t1, t2))
        ineq2 = mtex(r"|l_3 - l_2| \ge |l_1 - l_4|", font_size=42).to_edge(
            DOWN, buff=1.1)
        self.play(Transform(ineq1, ineq2))
        self.hold(2)
        final = mtex(r"l_{min} + l_{max} \le l' + l''",
                        font_size=56, color=ACCENT).to_edge(DOWN, buff=1.1)
        self.play(Transform(ineq1, final))
        self.takeaway("最短+最长 ≤ 其余之和；最短杆须当连架杆/机架",
                        p="孙桓八版 p131-133")
        self.hold(3)


class S04_Inversion(LessonScene):
    """机架变换=机构变换（~8min, p128-131）：同一条四杆链，轮流高亮四个构件
    当机架 → 得到曲柄摇杆/双曲柄/双摇杆——'相对运动不变'思想。"""

    def construct(self):
        self.header("换谁当机架，就换一台机器", "机构倒置 · p128-131")
        fb = FourBar(3.6, 1.0, 2.8, 2.4, origin=np.array([-2.0, -1.4, 0]))
        m = AnimatedFourBar(fb)
        self.play(FadeIn(m.group))
        names = ["机架=l₁：曲柄摇杆", "机架=l₂(最短)：双曲柄",
                 "机架=l₃(连杆)：双摇杆", "机架=l₄：曲柄摇杆"]
        tag = ctext(names[0], size=27, color=ACCENT).to_edge(DOWN, buff=0.6)
        self.play(Write(tag))
        for i in range(1, 4):
            self.play(Transform(tag, ctext(names[i], size=27, color=ACCENT)
                                .to_edge(DOWN, buff=0.6)),
                      m.theta.animate.set_value(m.theta.get_value() + PI),
                      run_time=2.2)
        self.hold(1)
        self.play(FadeOut(tag))
        self.takeaway("机架变换（倒置）不改变相对运动，只改变'谁是主角'",
                        p="孙桓八版 p128-131")
        self.hold(3)


class S05_Evolution(LessonScene):
    """演化谱系（~10min, p124-128/133-135）：四杆→曲柄滑块（摇杆→弧→滑块）→
    偏心/对心 → 导杆/摇块/定块——'变形记'流程图动画。"""

    def construct(self):
        self.header("一杆四变：演化谱系", "p124-135")
        steps = VGroup(
            ctext("铰链四杆", size=25),
            ctext("→ 摇杆变成无限长", size=25, color=NOTE),
            ctext("曲柄滑块", size=25, color=ACCENT),
            ctext("→ 换机架/换滑块", size=25, color=NOTE),
            ctext("导杆·摇块·定块", size=25, color=ACCENT),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(LEFT * 4.6 + UP * 0.4)
        self.play(FadeIn(steps, lag_ratio=0.4), run_time=2.5)
        # 右上：曲柄滑块动画；右下：摇块
        cs = AnimatedCrankSlider(CrankSlider(0.6, 1.8, 0.0,
                                             origin=np.array([1.2, 1.2, 0])),
                                 cylinder=False)
        self.play(FadeIn(cs.group))
        self.play(cs.theta.animate.set_value(TAU), run_time=4,
                  rate_func=linear)
        # 摇块机构：滑块绕定点摆动
        oc = P(2.6, -1.6)
        th2 = ValueTracker(0.0)

        def wiggle():
            t = th2.get_value()
            return VGroup(
                link_line(P(0.4, -1.6), P(1.2, -0.4), LINK_A),
                link_line(P(1.2, -0.4), P(2.9, -1.0), LINK_B),
                slider_block(P(2.9, -1.0), 0.5, 0.34, angle=0.2 *
                             np.sin(t), color=LINK_D),
                pin_joint(P(2.9, -1.0)),
                ground_hatch(P(2.9, -1.45), 0.6),
                link_line(P(0.4, -1.6), P(0.4, -1.6) + P(0.001, 0.001),
                          FRAME_C, 1),
                fixed_pin(P(0.4, -1.6)),
            )
        wm = always_redraw(wiggle)
        self.play(FadeIn(wm))
        self.play(th2.animate.set_value(TAU), run_time=4)
        self.hold(1)
        self.takeaway("演化的两板斧：转动副↔移动副互换、机架轮换",
                        p="孙桓八版 p124-135")
        self.hold(3)


class S06_QuickReturn(LessonScene):
    """急回特性（~12min, p136-138）：极位夹角 θ、行程速比 K=(180+θ)/(180−θ)；
    摆动导杆（牛头刨）实测：工作行程慢、回程快，角速度表实时显示。"""

    def construct(self):
        self.header("急回特性", "极位夹角 θ 与行程速比 K · p136-138")
        ww = Whitworth(r=0.85, d=1.5, lever=2.6,
                       origin=np.array([-2.6, 0.6, 0]))
        m = AnimatedWhitworth(ww)
        self.play(FadeIn(m.group))
        # 标出两极位（导杆与曲柄圆相切）
        e = ww.extreme_angle()
        self.play(Write(ctext(f"极位夹角 θ = 2·asin(r/d) ≈ "
                              f"{np.degrees(e):.0f}°", size=26,
                              color=ACCENT).to_edge(UP, buff=1.7)))
        # 转两圈：工作行程(慢) vs 回程(快) — 用不等速暗示
        self.play(m.theta.animate.set_value(PI / 2 + TAU), run_time=8,
                  rate_func=linear)
        steps = [
            mtex(r"K = \frac{v_{return}}{v_{work}} = "
                    r"\frac{180° + \theta}{180° - \theta} > 1", font_size=48,
                    color=ACCENT),
            mtex(r"\theta = 180°\,\frac{K-1}{K+1}", font_size=42),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 2.9 + DOWN * 1.5,
                       wait=1.8)
        self.takeaway("曲柄等速转，刨刀去程慢（切削稳）回程快（省时间）",
                        p="孙桓八版 p136-138")
        self.hold(3)


class S07_PressureAngle(LessonScene):
    """压力角α/传动角γ（~10min, p138-139）：四杆运转中 γ 实时显示——
    输出力有效分力∝sin γ；γ_min 出现在曲柄与机架共线位形。"""

    def construct(self):
        self.header("压力角与传动角", "出力'顺不顺'的度量 · p138-139")
        fb = FourBar(3.8, 1.1, 3.0, 2.5, origin=np.array([-3.6, -1.4, 0]))
        m = AnimatedFourBar(fb)
        self.play(FadeIn(m.group))
        # γ 实时标注（连杆-摇杆夹角）
        arc = always_redraw(lambda: angle_mark(
            fb.solve(m.theta.get_value())[2],
            np.arctan2(*(fb.solve(m.theta.get_value())[1]
                         - fb.solve(m.theta.get_value())[2])[1::-1]),
            np.arctan2(*(fb.solve(m.theta.get_value())[3]
                         - fb.solve(m.theta.get_value())[2])[1::-1]),
            r=0.5, color=ACCENT))
        gval = always_redraw(lambda: ctext(
            f"γ={np.degrees(fb.transmission_angle(m.theta.get_value())):.0f}°",
            size=28, color=ACCENT).next_to(
                fb.solve(m.theta.get_value())[2], UP + RIGHT, buff=0.4))
        self.play(FadeIn(arc), FadeIn(gval))
        self.play(m.theta.animate.set_value(0.6 + TAU), run_time=8,
                  rate_func=linear)
        self.hold(1)
        rules = bullets([
            "γ=90° 最佳：力全部用于推动从动件",
            "γmin ≥ 40°~50°（校核准则）",
            "γmin 出现在曲柄与机架共线位形",
        ], size=25).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(rules, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p138-139"))
        self.hold(3)


class S08_DeadPoint(LessonScene):
    """死点（~8min, p139）：摇杆主动时连杆与曲柄共线→γ=0 卡死；
    缝纫机踏板靠飞轮惯性冲过死点——演示'卡住'与'惯性冲过'两结局。"""

    def construct(self):
        self.header("死点：力再好也推不动", "p139")
        fb = FourBar(3.6, 0.9, 3.0, 2.6, origin=np.array([-3.4, -1.3, 0]))
        m = AnimatedFourBar(fb)
        self.play(FadeIn(m.group))
        # 摇杆主动：到共线位形时演示卡住
        self.play(m.theta.animate.set_value(PI), run_time=2)
        stuck = ctext("曲柄-连杆共线：γ=0，推力全压向铰链 → 卡死", size=26,
                      color=BAD).to_edge(UP, buff=1.7)
        self.play(Write(stuck))
        cross = VGroup(Line(P(-0.5, 0.5), P(0.5, -0.5), color=BAD,
                            stroke_width=8),
                       Line(P(-0.5, -0.5), P(0.5, 0.5), color=BAD,
                            stroke_width=8)).scale(0.5).move_to(
            fb.solve(PI)[1])
        self.play(Create(cross))
        self.focus(cross, color=BAD)
        self.hold(2)
        fix = ctext("对策：飞轮惯性冲过 / 多组错相并列（内燃机多缸）", size=25,
                    color=GOOD).to_edge(DOWN, buff=0.7)
        self.play(Write(fix))
        self.add(page_ref("孙桓八版 p139"))
        self.hold(3)


class S09_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 7 讲小结")
        pts = bullets([
            "三基本型：曲柄摇杆/双曲柄/双摇杆——看最短杆当谁   (p124-128)",
            "Grashof：最短+最长 ≤ 其余之和（两位形推导）       (p131-133)",
            "演化：移动副替换、机架倒置 → 滑块/导杆/摇块       (p124-135)",
            "急回：K=(180+θ)/(180−θ)，θ 极位夹角              (p136-138)",
            "γ 越大越好；死点 γ=0 靠惯性/错相通过             (p138-139)",
        ], size=25)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：给定任务怎么'反推'四根杆的尺寸？——连杆机构设计",
                  size=26, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
