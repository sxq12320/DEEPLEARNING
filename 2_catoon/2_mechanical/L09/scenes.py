# -*- coding: utf-8 -*-
"""L09 按剧本运动的机器——凸轮机构及其设计（第9章, p167-190）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_CamFamily(LessonScene):
    """凸轮机构组成与分类（~8min, p167-169）：盘形/移动/圆柱 × 直动/摆动 ×
    尖顶/滚子/平底——分类矩阵动画。"""

    def construct(self):
        self.header("凸轮：把'剧本'刻在轮廓上", "组成与分类 · p167-169")
        cam = AnimatedCamKnife(0.7, cam_law("cos", 0.5, PI * 1.3),
                               origin=np.array([-4.2, -1.0, 0]))
        self.play(FadeIn(cam.group))
        self.play(cam.theta.animate.set_value(TAU), run_time=4,
                  rate_func=linear)
        tree = bullets([
            "按形状：盘形 / 移动 / 圆柱",
            "按推杆：直动（对心/偏置） / 摆动",
            "按端部：尖顶 / 滚子 / 平底",
            "按封闭：几何（沟槽） / 力（弹簧）",
        ], size=25).shift(RIGHT * 2.2 + UP * 0.4)
        self.play(FadeIn(tree, lag_ratio=0.4), run_time=2.8)
        self.hold(2.5)
        self.takeaway("凸轮=主动件，轮廓=运动剧本；推杆=照章办事",
                        p="孙桓八版 p167-169")
        self.hold(3)


class S02_MotionLaws(LessonScene):
    """推程四规律（~16min, p170-176 核心）：等速/等加速等减速/余弦/正弦——
    每规律一行 s|v|a 三小图 + 冲击标注（刚性 vs 柔性冲击）。"""

    def construct(self):
        self.header("推程运动规律：四种'剧本'", "s/v/a 三联图 · p170-176")
        laws = [("const", "等速", BAD), ("para", "等加速等减速", LINK_D),
                ("cos", "余弦·简谐", NOTE), ("sine", "正弦·摆线", GOOD)]
        col_names = ["s(δ) 位移", "v(δ) 速度", "a(δ) 加速度"]
        col_x = [-2.9, 0.3, 3.5]
        row_y = [1.85, 0.5, -0.85, -2.2]
        shock = ["刚性冲击×2", "柔性冲击×2", "柔性冲击×2", "无冲击"]
        shock_col = [BAD, LINK_D, LINK_D, GOOD]
        W, H = 2.7, 1.0
        # 列标题
        for cn, cx in zip(col_names, col_x):
            self.add(ctext(cn, size=20, color=MUTED).move_to(
                np.array([cx, 2.6, 0])))
        def mini_axes(c, r):
            ax = Axes(x_range=[0, 1.02, 0.5], y_range=[-1.05, 1.05, 1],
                      x_length=W, y_length=H,
                      axis_config={"color": FRAME_C, "stroke_width": 2},
                      tips=False)
            ax.move_to(np.array([c, r, 0]))
            return ax
        rows = []
        for i, (key, name, col) in enumerate(laws):
            nm = ctext(name, size=22, color=col).move_to(
                np.array([-5.3, row_y[i], 0]))
            axs = [mini_axes(col_x[k], row_y[i]) for k in range(3)]
            tag = ctext(shock[i], size=19, color=shock_col[i]).move_to(
                np.array([5.6, row_y[i], 0]))
            rows.append((nm, axs, tag))
            self.play(FadeIn(VGroup(nm, *axs, tag)), run_time=0.7)
            s, v, a = cam_law_va(key, 1.0, 1.0)
            xs = np.linspace(0, 1, 80)
            for k, f, cc in ((0, s, INK), (1, v, LINK_D), (2, a, col)):
                fs = np.array([f(x) for x in xs], dtype=float)
                fs = fs / (np.abs(fs).max() + 1e-9) * 0.88
                pts = [axs[k].c2p(x, y) for x, y in zip(xs, fs)]
                crv = VMobject(color=cc, stroke_width=2.5)
                crv.set_points_smoothly(pts)
                self.play(Create(crv), run_time=0.55)
        self.hold(1.5)
        self.takeaway("选规律=选冲击：低速用等速，高速必须正弦/组合",
                        p="孙桓八版 p170-176")
        self.hold(3)


class S03_CombinedLaws(LessonScene):
    """组合规律（~5min, p176）：等速段两头用圆弧/摆线过渡削冲击——改进等速
    位移图前后对比。"""

    def construct(self):
        self.header("给剧本'修圆角'", "组合运动规律 · p176")
        ax = Axes(x_range=[0, 1.05, 0.25], y_range=[0, 1.25, 0.5],
                  x_length=7.5, y_length=3.4,
                  axis_config={"color": FRAME_C})
        ax.shift(UP * 0.1)
        raw = ax.plot(lambda u: np.clip(u, 0, 1), x_range=[0, 1],
                      color=BAD, stroke_width=4)
        mod = ax.plot(lambda u: np.clip(u - 0.12 * np.sin(TAU * u), 0, 1),
                      x_range=[0, 1], color=GOOD, stroke_width=4)
        self.play(Create(ax))
        self.play(Create(raw))
        cap = ctext("等速：两端直角=刚性冲击", size=24, color=BAD).next_to(
            ax, UP, buff=0.3)
        self.play(Write(cap))
        self.play(Transform(raw, mod))
        self.play(Transform(cap, ctext("首末用正弦段过渡 → 速度连续、无冲击",
                                       size=24,
                                       color=GOOD).next_to(ax, UP,
                                                           buff=0.3)))
        self.add(page_ref("孙桓八版 p176"))
        self.hold(3)


class S04_InversionDraw(LessonScene):
    """反转法作图（~14min, p177-179 核心）：凸轮别动，推杆绕中心'倒着转'——
    基圆→等分→各位移点→包络成廓线全过程现场画。"""

    def construct(self):
        self.header("反转法：把凸轮'按住'作图", "§9-3 · p177-179")
        O = P(-1.4, -1.1)
        r0 = 1.3
        self.play(Create(Circle(radius=r0, color=MUTED).move_to(O)))
        self.play(Write(ctext("① 画基圆 r₀", size=24).next_to(
            Circle(radius=r0).move_to(O), LEFT, buff=0.6)))
        h, beta = 0.9, PI * 1.1
        sfun = cam_law("sine", h, beta)
        n = 9
        rays = VGroup()
        tips = []
        for i in range(n + 1):
            ang = PI / 2 + i * beta / n  # 推程角内等分
            rays.add(DashedLine(O, O + P(np.cos(ang), np.sin(ang)) *
                                (r0 + h + 0.2), color=MUTED,
                                stroke_width=1.5))
            rr = r0 + sfun(i * beta / n)
            tips.append(O + P(np.cos(ang), np.sin(ang)) * rr)
        cap = ctext("② 反转等分推程角 δ", size=24).to_edge(UP, buff=1.75)
        self.play(Create(rays, lag_ratio=0.15))
        self.play(Write(cap))
        dots = VGroup(*[Dot(t, radius=0.07, color=LINK_D) for t in tips])
        self.play(FadeIn(dots, lag_ratio=0.2))
        self.play(Transform(cap, ctext("③ 各反转位置量取位移 s(δ)", size=24)
                              .to_edge(UP, buff=1.75)))
        prof_pts = tips + [tips[-1]]
        prof = VMobject(color=LINK_D, stroke_width=4)
        prof.set_points_smoothly(tips)
        self.play(Create(prof), run_time=2)
        self.play(Transform(cap, ctext("④ 光滑包络 → 凸轮廓线", size=24,
                                       color=LINK_D).to_edge(UP, buff=1.75)))
        # 真值对照：完整轮廓淡显
        full = cam_profile_knife(lambda d: sfun(d) if d < beta else
                                 (h if d < PI * 1.6 else
                                  sfun(TAU - d) if d > TAU - beta * 0.6 else h),
                                 r0, color=GOOD, stroke_width=2)
        full.move_to(O)
        self.play(Create(full))
        self.add(page_ref("孙桓八版 p177-179"))
        self.hold(3)


class S05_RollerCam(LessonScene):
    """滚子从动件（~8min, p179-181）：理论廓线（滚子中心轨迹）→实际廓线=
    内等距线；滚子半径过大→尖点/失真特写。"""

    def construct(self):
        self.header("滚子推杆：理论廓线 ≠ 实际廓线", "等距线与失真 · p179-181")
        sfun = cam_law("sine", 0.55, PI * 1.2)
        O = P(-3.4, -1.1)
        theory = cam_profile_knife(lambda d: 0.18 + sfun(d), 0.62,
                                   color=MUTED, stroke_width=2.5)
        theory.move_to(O)
        real = cam_profile_roller(sfun, 0.62, 0.18, color=LINK_D,
                                  stroke_width=4)
        real.move_to(O)
        self.play(Create(theory))
        self.play(Write(ctext("理论廓线 = 滚子中心轨迹", size=24)
                        .next_to(theory, UP, buff=0.5)))
        self.play(Create(real))
        self.play(Write(ctext("实际廓线 = 内偏滚子半径 rr 的等距线", size=24,
                              color=LINK_D).next_to(theory, DOWN, buff=0.5)))
        self.hold(2)
        warn = VGroup(
            mtex(r"\rho_{min} > r_r\ \text{否则廓线变尖/失真}",
                    font_size=42, color=BAD),
            ctext("滚子半径宜取 rr ≤ 0.8·ρmin", size=24, color=NOTE),
        ).arrange(DOWN, buff=0.4).shift(RIGHT * 3.2 + UP * 0.4)
        self.play(FadeIn(warn, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p179-181"))
        self.hold(3)


class S06_PressureAngle(LessonScene):
    """凸轮压力角（~8min, p181-183）：α=接触点受力方向与推杆速度方向夹角；
    α 过大→自锁；α 与基圆 r0 的权衡：基圆大→α 小但凸轮大。"""

    def construct(self):
        self.header("凸轮的压力角", "α 与基圆半径 · p181-183")
        cam = AnimatedCamKnife(0.8, cam_law("sine", 0.5, PI * 1.2),
                               origin=np.array([-3.6, -1.5, 0]))
        self.play(FadeIn(cam.group))
        th = ValueTracker(0.0)
        # α 标注：接触点法线（轮廓法向）与竖直导路夹角——示意
        alpha_lab = always_redraw(lambda: ctext(
            f"α≈{abs(28 * np.sin(th.get_value())):.0f}°", size=30,
            color=ACCENT).next_to(cam.group, UP, buff=0.5))
        self.play(FadeIn(alpha_lab))
        self.play(cam.theta.animate.set_value(TAU),
                  th.animate.set_value(TAU), run_time=6, rate_func=linear)
        pts = bullets([
            "α 过大 → 推杆被'别住'（自锁风险）",
            "推程 [α]≈30°~38°；回程 [α]'≈70°~80°",
            "基圆 r0 越大 α 越小，但凸轮越肥大——权衡",
        ], size=26).shift(RIGHT * 2.4 + DOWN * 0.4)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p181-183"))
        self.hold(3)


class S07_OtherCams(LessonScene):
    """其他凸轮速览（~5min, p183-190）：摆动推杆/平底推杆/偏置/圆柱凸轮——
    各给一张示意+一句话。"""

    def construct(self):
        self.header("凸轮家族其他成员", "p183-190")
        items = bullets([
            "偏置直动：借偏距 e 改善压力角（反程增α慎用）",
            "摆动推杆：输出摆角，结构更紧凑",
            "平底推杆：α≡0 传动最好，但不能走'凹'廓线",
            "圆柱凸轮：空间凸轮，输出轴向直动/摆动",
        ], size=28).shift(UP * 0.3)
        self.play(FadeIn(items, lag_ratio=0.5), run_time=2.8)
        self.hold(3)
        self.add(page_ref("孙桓八版 p183-190"))
        self.hold(2)


class S08_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 9 讲小结")
        pts = bullets([
            "凸轮=运动剧本；三轴分类法                (p167-169)",
            "四规律：等速(刚冲)/等加等减/余弦(柔冲)/正弦(无冲) (p170-176)",
            "反转法：凸轮不动、推杆倒转、包络成廓       (p177-179)",
            "滚子：实际廓线=内等距线；rr<0.8ρmin    (p179-181)",
            "压力角 α 与基圆权衡；[α]≈30°~38°        (p181-183)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：最优雅的传动——渐开线齿轮", size=25,
                  color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
