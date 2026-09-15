# -*- coding: utf-8 -*-
"""L13 精巧的小机构——棘轮·槽轮·万向节与其他（第12章, p261-287）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Ratchet(LessonScene):
    """棘轮机构（~10min, p261-264）：外啮合棘轮+棘爪——摆杆往复摆动→轮单向
    间歇转；止回爪；可变向棘轮（翻转爪）——千斤顶/自行车飞轮。"""

    def construct(self):
        self.header("棘轮：只许前进不许后退", "p261-264")
        center = P(-3.0, -0.9)
        wheel, pawl, pv = ratchet_assembly(center, r=1.25, z=12)
        arm_th = ValueTracker(0.0)

        def pawl_arm():
            a = 0.18 * np.sin(arm_th.get_value())
            rot = pawl.copy().rotate(a, about_point=pv)
            return VGroup(link_line(pv, pv + LEFT * 1.8, LINK_A, 5), rot,
                          pin_joint(pv))
        pa = always_redraw(pawl_arm)
        self.play(FadeIn(wheel), FadeIn(pa), FadeIn(fixed_pin(center)))
        # 驱动：摆臂摆半周→轮进一齿
        for _ in range(3):
            self.play(arm_th.animate.set_value(arm_th.get_value() + PI),
                      Rotate(wheel, TAU / 12, about_point=center),
                      run_time=1.6)
            self.play(arm_th.animate.set_value(arm_th.get_value() + PI),
                      run_time=1.2)   # 回程轮不动
        self.hold(1)
        pts = bullets([
            "主动摆杆往复摆 → 棘爪推棘轮单向间歇转",
            "止回爪防倒转（千斤顶/卷扬刹车）",
            "可变向棘轮：翻转棘爪即换向",
        ], size=26).shift(RIGHT * 3.0 + DOWN * 0.3)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p261-264"))
        self.hold(3)


class S02_Geneva(LessonScene):
    """槽轮机构（~12min, p264-267）：马耳他十字——主动盘匀转，槽轮转位后
    锁止；τ=(z−2)/2z 推导（运动系数<1/2，槽数越多占空比越高）。"""

    def construct(self):
        self.header("槽轮：电影放映机的'心跳'", "p264-267")
        g = AnimatedGeneva(z=4, a=2.6, c1=np.array([-4.2, -0.7, 0]))
        self.play(FadeIn(g.group))
        state = always_redraw(lambda: ctext(
            "槽轮：转位中" if geneva_state(g.theta.get_value(), 4, 2.6)[1]
            else "槽轮：锁止中", size=26,
            color=GOOD if geneva_state(g.theta.get_value(), 4, 2.6)[1]
            else MUTED).to_edge(UP, buff=1.7))
        self.play(FadeIn(state))
        # 第一圈先看全景；入槽瞬间推镜看销-槽啮合细节，再拉回
        self.play(g.theta.animate.set_value(g.theta.get_value() + TAU),
                  run_time=2.5, rate_func=linear)
        engage_pt = (g.c1 + g.c2) / 2 + DOWN * 0.3
        self.zoom_to(engage_pt, scale=0.55, run_time=1.2)
        self.play(g.theta.animate.set_value(g.theta.get_value() + TAU),
                  run_time=2.5, rate_func=linear)
        self.zoom_reset()
        self.play(g.theta.animate.set_value(g.theta.get_value() + 2 * TAU),
                  run_time=5, rate_func=linear)
        self.hold(1)
        steps = [
            mtex(r"\tau = \frac{t_{\text{动}}}{t_{\text{总}}} = \frac{z-2}{2z} < "
                    r"\frac{1}{2}", font_size=48, color=ACCENT),
            mtex(r"z=4:\ \tau=0.25;\quad z=6:\ \tau=1/3;\quad "
                    r"z\uparrow\Rightarrow\tau\uparrow", font_size=38),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 3.4 + DOWN * 0.6,
                       wait=1.8)
        self.add(page_ref("孙桓八版 p264-267"))
        self.hold(3)


class S03_IncompleteGear(LessonScene):
    """不完全齿轮与凸轮间歇（~6min, p267-269）：有齿段啮合、无齿段停；
    锁止弧配合——电影机与计数器应用。"""

    def construct(self):
        self.header("不完全齿轮", "p267-269")
        # 有齿段齿轮 + 全齿轮
        driver = gear_profile(0.1, 18, color=LINK_A, stroke_width=2.5)
        driver.move_to(LEFT * 3.0 + UP * 0.35)
        # 挖掉下半圈齿：用局部圆弧近似“无齿段”
        cover = Sector(radius=1.35, angle=PI * 0.9, color=BG,
                       fill_opacity=1, stroke_width=0).move_to(
            driver.get_center()).rotate(PI * 1.05,
                                        about_point=driver.get_center())
        driven = gear_profile(0.1, 30, color=LINK_C, stroke_width=2.5)
        driven.move_to(LEFT * 0.6 + UP * 0.35)  # 分度圆与主动轮相切
        self.play(Create(driver), FadeIn(cover), Create(driven))
        self.play(Rotate(driver, TAU / 2,
                         about_point=driver.get_center()),
                  run_time=2.5)
        pts = bullets([
            "主动轮只有一段齿：啮合段从动轮转，脱开段停",
            "锁止弧保证停歇位姿精确",
            "与槽轮相比：冲击大，只用于低速轻载",
        ], size=25).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p267-269"))
        self.hold(3)


class S04_Screw(LessonScene):
    """螺旋机构（~6min, p269-272）：螺母-螺杆——传动（丝杠）/微调（千分尺）/
    自锁（千斤顶）三用；滚动螺旋提一句。"""

    def construct(self):
        self.header("螺旋机构：一转走多远", "p269-272")
        O = P(-3.6, -0.2)
        th = ValueTracker(0.0)

        def screw():
            a = th.get_value()
            x = a / TAU * 0.5                      # 每圈走一个导程单位
            return VGroup(
                Line(O + LEFT * 1.6, O + RIGHT * 3.4, color=FRAME_C,
                     stroke_width=6),              # 螺杆
                Rectangle(width=0.8, height=0.9, color=LINK_D,
                          fill_opacity=0.35,
                          fill_color=LINK_D).move_to(O + RIGHT * x),
                dashed(O + RIGHT * x + UP * 0.45, O + RIGHT * x + UP * 0.9,
                       color=MUTED),
            )
        m = always_redraw(screw)
        self.play(FadeIn(m))
        self.play(th.animate.set_value(4 * TAU), run_time=5, rate_func=linear)
        self.hold(1)
        pts = bullets([
            "s = n·ph：一转一导程，天然减速微调",
            "λ≤φ 时自锁：千斤顶、台虎钳",
            "滚珠丝杠：滚动摩擦，效率 90%+",
        ], size=24).shift(RIGHT * 2.4 + DOWN * 0.9)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p269-272"))
        self.hold(3)


class S05_UniversalJoint(LessonScene):
    """万向联轴节（~8min, p272-275）：单万向——输入匀速输出波动（α 越大波动
    越大）；双万向等速条件：中间轴两叉共面+两 α 相等。"""

    def construct(self):
        self.header("万向联轴节", "单万向不等速 · p272-275")
        ax = Axes(x_range=[0, 6.3, 1], y_range=[0.5, 1.6, 0.5],
                  x_length=9.5, y_length=3.2,
                  axis_config={"color": FRAME_C})
        ax.shift(UP * 0.1)
        alpha = np.deg2rad(30)
        curve = ax.plot(
            lambda t: np.cos(alpha) / (1 - np.sin(alpha) ** 2 *
                                       np.cos(t) ** 2),
            x_range=[0, TAU], color=LINK_D, stroke_width=4)
        flat = ax.plot(lambda t: 1.0, x_range=[0, TAU], color=MUTED,
                       stroke_width=2.5)
        self.play(Create(ax), Create(flat))
        self.play(Write(ctext("单万向：输入匀速 → 输出周期性快慢波动", size=25)
                        .next_to(ax, UP, buff=0.3)))
        self.play(Create(curve))
        self.hold(2)
        fix = bullets([
            "双万向等速条件①：中间轴两端叉头共面",
            "② 主动、从动轴与中间轴夹角 α₁=α₂",
            "汽车传动轴因此总是成对出现",
        ], size=25).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(fix, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p272-275"))
        self.hold(3)


class S06_Combined(LessonScene):
    """组合机构一瞥（~5min, p275-287）：齿轮-连杆/凸轮-连杆/凸轮-齿轮组合
    可实现单一机构给不出的复杂运动——引出机构创新概念。"""

    def construct(self):
        self.header("1+1>2：组合机构", "p275-287")
        pts = bullets([
            "齿轮-连杆：连杆曲线 + 齿轮约束 → 复杂轨迹",
            "凸轮-连杆：精确停顿/变速回摆",
            "凸轮-齿轮：实现非匀速回转",
            "机构创新路径：演化 · 倒置 · 组合 · 变异",
        ], size=29).shift(UP * 0.3)
        self.play(FadeIn(pts, lag_ratio=0.5), run_time=2.8)
        self.hold(3)
        self.add(page_ref("孙桓八版 p275-287"))
        self.hold(2)


class S07_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 13 讲小结")
        pts = bullets([
            "棘轮：往复摆 → 单向间歇转 + 止回         (p261-264)",
            "槽轮：τ=(z−2)/2z；锁止弧保停歇位姿       (p264-267)",
            "不完全齿轮/凸轮间歇：简单但冲击大         (p267-269)",
            "螺旋：减速微调+自锁三用                  (p269-272)",
            "万向节：单个不等速，成对才等速            (p272-275)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("最后一讲：把零件攒成机器——传动系统方案设计",
                  size=25, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
