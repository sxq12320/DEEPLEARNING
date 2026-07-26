# -*- coding: utf-8 -*-
"""L06 按剧本运动的机器——凸轮机构及其设计（孙桓八版 第9章, p167-190）

目标成片 85-95 min。核心：运动规律推导与冲击、反转法轮廓设计、压力角与基圆、滚子失真。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


def s_uniform(d, h=1.0, d0=PI):
    """等速运动规律（推程段）。"""
    return h * d / d0 if d <= d0 else h


def s_parab(d, h=1.0, d0=PI):
    """等加速等减速（分段抛物线）。"""
    if d >= d0:
        return h
    if d < d0 / 2:
        return 2 * h * (d / d0) ** 2
    return h - 2 * h * (1 - d / d0) ** 2


def s_cos(d, h=1.0, d0=PI):
    """余弦（简谐）：s = h/2 (1−cos(πδ/δ0))。"""
    return h / 2 * (1 - np.cos(PI * d / d0)) if d <= d0 else h


def s_cyc(d, h=1.0, d0=PI):
    """正弦（摆线）：s = h(δ/δ0 − sin(2πδ/δ0)/2π)。"""
    return h * (d / d0 - np.sin(2 * PI * d / d0) / (2 * PI)) if d <= d0 else h


class S01_Opening(LessonScene):
    """开场（3min）：连杆只能近似轨迹；配气门要求'20°内开到 8mm、保持、再关'——
    这种'剧本式'运动只有凸轮能精确胜任。分类速览（p167-170）。"""

    def construct(self):
        big = ctext("第 6 讲   按剧本运动的机器", size=58, weight="BOLD")
        sub = ctext("凸轮：把运动规律'刻'在轮廓上", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        pts = bullets([
            "优点：任意运动规律都能精确实现，结构紧凑",
            "缺点：高副点线接触易磨损、加工较难",
            "分类：盘形/移动/圆柱凸轮 × 尖顶/滚子/平底推杆 × 力封闭/形封闭",
        ], size=29)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p167-170"))
        self.hold(3)


class S02_Terminology(LessonScene):
    """基本名词（8min，p170-171）：基圆 r0、推程/推程运动角 δ0、远休止角、回程角、近休止角、行程 h。
    动画：凸轮转一圈，右侧 s-δ 曲线同步走完'推-远休-回-近休'四段。"""

    def construct(self):
        self.header("凸轮机构的语言", "基圆 · 行程 · 四个运动角 · p170-171")

        def s_full(d):
            d = d % TAU
            if d < PI / 2:
                return s_cos(d, 0.9, PI / 2)
            if d < PI * 0.75:
                return 0.9
            if d < PI * 1.25:
                return 0.9 * (1 - (d - PI * 0.75) / (PI * 0.5))
            return 0.0

        cam = cam_profile_knife(s_full, 1.0).shift(LEFT * 3.6 + DOWN * 0.6)
        cc = cam.get_center()
        ax = Axes(x_range=[0, TAU, PI / 2], y_range=[0, 1.2, 0.5], x_length=6.0, y_length=2.8,
                  axis_config={"include_tip": False, "font_size": 20}).shift(RIGHT * 2.8 + UP * 0.8)
        curve = ax.plot(s_full, x_range=[0, TAU], color=ORANGE)
        seglabels = VGroup(
            ctext("推程 δ0", size=20, color=GOOD).move_to(ax.c2p(PI / 4, 1.1)),
            ctext("远休", size=20, color=GREY_B).move_to(ax.c2p(PI * 0.62, 1.1)),
            ctext("回程", size=20, color=ORANGE).move_to(ax.c2p(PI, 1.1)),
            ctext("近休", size=20, color=GREY_B).move_to(ax.c2p(PI * 1.6, 1.1)),
        )
        base_c = Circle(radius=1.0, color=GREY_B, stroke_width=2).move_to(cc)
        r0lab = ctext("基圆 r0（最小向径圆）", size=24, color=GREY_B).next_to(base_c, DOWN, buff=0.4)
        th = ValueTracker(0.0)
        dot = always_redraw(lambda: Dot(ax.c2p(th.get_value() % TAU, s_full(th.get_value())),
                                        color=YELLOW, radius=0.07))
        cam_g = VGroup(cam, base_c)
        self.play(Create(cam), Create(base_c), FadeIn(r0lab), Create(ax), Create(curve),
                  FadeIn(seglabels), FadeIn(dot))
        self.play(Rotate(cam_g, -TAU, about_point=cc), th.animate.set_value(TAU),
                  run_time=8, rate_func=linear)
        self.add(page_ref("孙桓八版 p170-171"))
        self.hold(2.5)


class S03_MotionLaws(LessonScene):
    """★运动规律与冲击（18min，p171-176，本讲第一核心）。
    讲稿要点：四种常用规律的 s/v/a 三联图推导——等速(a 在端点无穷大→刚性冲击)、
    等加等减(a 有限跳变→柔性冲击)、余弦简谐(端点仍柔性冲击)、正弦摆线(a 连续→无冲击)；
    选用原则：低速轻载可等速，高速选摆线/组合规律。"""

    def construct(self):
        self.header("推杆的运动规律", "s → v → a：冲击藏在导数里 · p171-176")
        laws = [("等速", s_uniform, RED, "a→∞  刚性冲击"),
                ("等加等减速", s_parab, ORANGE, "a 跳变  柔性冲击"),
                ("余弦(简谐)", s_cos, TEAL, "端点 a 跳变  柔性冲击"),
                ("正弦(摆线)", s_cyc, GREEN, "a 连续  无冲击")]
        for name, sf, col, verdict in laws:
            title = ctext(f"{name} 运动规律", size=32, color=col).to_edge(UP, buff=1.5)
            axs = VGroup()
            eps = 1e-4
            def vf(d, sf=sf): return (sf(min(d + eps, PI)) - sf(max(d - eps, 0))) / (2 * eps)
            def af(d, vf=vf): return (vf(min(d + eps, PI)) - vf(max(d - eps, 0))) / (2 * eps)
            for i, (fn, lab, yr) in enumerate([(sf, "s", 1.2), (vf, "v", 1.6), (af, "a", 6.5)]):
                a = Axes(x_range=[0.02, PI - 0.02], y_range=[-yr, yr], x_length=3.6, y_length=2.0,
                         axis_config={"include_tip": False, "include_ticks": False})
                cur = a.plot(fn, x_range=[0.03, PI - 0.03, 0.02], color=col, use_smoothing=False)
                axs.add(VGroup(a, cur, ctext(lab, size=24).next_to(a, UP, buff=0.1)))
            axs.arrange(RIGHT, buff=0.7).shift(DOWN * 0.3)
            verdict_t = ctext("→ " + verdict, size=30,
                              color=GREEN if "无冲击" in verdict else (RED if "刚性" in verdict else ORANGE)
                              ).to_edge(DOWN, buff=0.8)
            self.play(FadeIn(title), FadeIn(axs, lag_ratio=0.2), run_time=1.6)
            self.play(Write(verdict_t))
            self.hold(2.8)
            self.play(FadeOut(title), FadeOut(axs), FadeOut(verdict_t), run_time=0.5)
        concl = bullets([
            "刚性冲击：理论加速度无穷大——低速轻载才敢用等速",
            "柔性冲击：加速度有限跳变——中速可用等加等减/简谐",
            "无冲击：摆线/高次多项式/组合规律——高速凸轮标配",
        ], size=29)
        self.play(FadeIn(concl, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p171-176"))
        self.hold(3.5)


class S04_InversionPrinciple(LessonScene):
    """★反转法原理（10min，p177）。
    讲稿要点：给整个机构加 −ω 公共角速度，相对运动不变 → 凸轮'静止'，
    推杆连同导路绕凸轮反转，同时按规律伸缩——推杆尖端扫出的就是凸轮轮廓。"""

    def construct(self):
        self.header("反转法", "让凸轮静止，世界反着转 · p177")
        idea = [
            MathTex(r"\text{原系统: 凸轮 } \omega,\ \text{推杆按 } s(\delta) \text{ 升降}", font_size=40),
            MathTex(r"\text{全体叠加 } -\omega:\ \text{相对运动完全不变}", font_size=40),
            MathTex(r"\Rightarrow\ \text{凸轮静止, 推杆绕它公转} -\omega \text{ 且沿导路伸缩 } s(\delta)",
                    font_size=40, color=YELLOW),
            MathTex(r"\text{推杆尖端的轨迹} \;=\; \text{凸轮轮廓!}", font_size=46, color=GREEN),
        ]
        formula_reveal(self, idea, anchor=UP * 0.6, buff=0.42, wait=1.8)
        note = ctext("与 L05 刚化-反转法同一个灵魂：相对运动不变原理", size=28, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p177"))
        self.hold(3)


class S05_ProfileConstruction(LessonScene):
    """★反转法作图全过程（15min，p177-179，本讲压轴动画）。
    动画：基圆分度 → 各反转位置画导路径向线 → 按 s(δi) 量取升程点 → 光滑连接成轮廓；
    同一动画流演示尖顶理论轮廓；滚子实际轮廓=理论轮廓的内等距包络。"""

    def construct(self):
        self.header("一步步画出凸轮", "反转法作图 · p177-179")
        r0, h = 1.1, 0.8
        ctr = np.array([-2.2, -0.4, 0])
        base = Circle(radius=r0, color=GREY_B, stroke_width=2).move_to(ctr)
        self.play(Create(base))
        t1 = ctext("① 画基圆，按 Δδ 分度（反转方向 = −ω）", size=27).to_edge(DOWN, buff=1.2)
        self.play(Write(t1))
        N = 12
        s_law = lambda d: s_cos(d % TAU, h, PI) if (d % TAU) < PI else s_cos(TAU - (d % TAU), h, PI)
        rays = VGroup(); pts = []
        for i in range(N):
            d = i * TAU / N
            rr = r0 + s_law(d)
            direction = np.array([np.cos(-d + PI / 2), np.sin(-d + PI / 2), 0])
            rays.add(Line(ctr, ctr + direction * (r0 + h + 0.15), color=GREY_C, stroke_width=1.5))
            pts.append(ctr + direction * rr)
        self.play(Create(rays, lag_ratio=0.08), run_time=2.5)
        t2 = ctext("② 每条径向线上，从基圆向外量取该角对应的升程 s(δi)", size=27,
                   color=ACCENT).to_edge(DOWN, buff=1.2)
        dots = VGroup(*[Dot(p, color=YELLOW, radius=0.055) for p in pts])
        self.play(ReplacementTransform(t1, t2), FadeIn(dots, lag_ratio=0.12), run_time=2.5)
        self.hold(1.5)
        prof = cam_profile_knife(lambda d: s_law(-d + PI / 2), 1e-9)  # 占位，改用平滑连接:
        smooth_prof = VMobject(color=ORANGE, stroke_width=4)
        dense = [ctr + np.array([np.cos(-d + PI / 2), np.sin(-d + PI / 2), 0]) * (r0 + s_law(d))
                 for d in np.linspace(0, TAU, 180)]
        smooth_prof.set_points_smoothly([*dense, dense[0]])
        t3 = ctext("③ 光滑连接 → 尖顶推杆的（理论）轮廓完成！", size=27, color=GOOD).to_edge(DOWN, buff=1.2)
        self.play(ReplacementTransform(t2, t3), Create(smooth_prof), run_time=2.5)
        self.hold(2)
        # 滚子：实际轮廓 = 内等距线
        rr_roll = 0.18
        inner = VMobject(color=TEAL, stroke_width=3)
        inner_pts = []
        for k, d in enumerate(np.linspace(0, TAU, 180)):
            p = dense[k]
            # 法向近似：向心方向修正（教学演示用等距近似）
            nrm = (p - ctr) / np.linalg.norm(p - ctr)
            inner_pts.append(p - nrm * rr_roll)
        inner.set_points_smoothly([*inner_pts, inner_pts[0]])
        t4 = ctext("④ 滚子推杆：滚子中心走理论轮廓 → 实际轮廓 = 内等距包络线", size=26,
                   color=TEAL).to_edge(DOWN, buff=1.2)
        self.play(ReplacementTransform(t3, t4), Create(inner), run_time=2)
        self.add(page_ref("孙桓八版 p177-179"))
        self.hold(3)


class S06_PressureAngle(LessonScene):
    """★压力角与基圆半径（12min，p181-184）。
    讲稿要点：α = 推杆受力方向(轮廓法线)与速度方向夹角；tanα = |ds/dδ − e|/(s0+s)
    推导（瞬心法）；α↑ → 有效分力↓，α>[α] 会自锁卡死（推程 [α]≈30°）；
    r0↑ → α↓ 但体积↑——设计权衡。动画：滑块联动展示 r0 与 α 的博弈。"""

    def construct(self):
        self.header("压力角 α 与基圆半径 r0", "小凸轮与不卡死的博弈 · p181-184")
        f = [
            MathTex(r"\alpha:\ \text{轮廓法线（受力线）与推杆速度方向的夹角}", font_size=38),
            MathTex(r"\text{瞬心法推导（相对瞬心 P 在法线与导路交点）:}", font_size=36),
            MathTex(r"\tan\alpha = \frac{|\,ds/d\delta - e\,|}{s_0 + s}\qquad s_0=\sqrt{r_0^2-e^2}",
                    font_size=48, color=YELLOW),
            MathTex(r"r_0 \uparrow\ \Rightarrow\ \alpha \downarrow\ \text{（但凸轮变大）}", font_size=40, color=GREEN),
            MathTex(r"\text{推程 } [\alpha]\approx 30^\circ,\ \text{回程 } [\alpha]\approx 70^\circ\ \text{（力封闭）}",
                    font_size=36),
        ]
        formula_reveal(self, f, anchor=UP * 0.3, buff=0.36, wait=1.8)
        note = ctext("α 太大 → 有用分力小、侧压大 → 效率骤降甚至自锁（呼应第 10 讲）", size=27,
                     color=ACCENT).to_edge(DOWN, buff=0.4)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p181-184"))
        self.hold(3)


class S07_RollerUndercut(LessonScene):
    """滚子半径与运动失真（8min，p184-185）。
    讲稿要点：外凸段实际轮廓曲率半径 ρa = ρ − rr；若 rr>ρmin 实际轮廓自交变尖 →
    推杆运动失真。规则 rr < ρmin（常取 ≤0.8ρmin）。动画：滚子渐大，轮廓从圆滑→变尖→自交。"""

    def construct(self):
        self.header("滚子不能太胖", "运动失真 · p184-185")
        f = [
            MathTex(r"\text{外凸处: } \rho_a = \rho - r_r", font_size=46),
            MathTex(r"r_r < \rho_{\min}\ \text{否则 } \rho_a \le 0:\ \text{轮廓变尖/自交}", font_size=42, color=RED),
            MathTex(r"\text{工程取 } r_r \le 0.8\,\rho_{\min}", font_size=40, color=GREEN),
        ]
        formula_reveal(self, f, anchor=UP * 1.4, buff=0.4, wait=1.6)
        # 演示：理论轮廓固定，等距线随 rr 增大逐渐变尖
        ctr = np.array([0, -1.7, 0])
        s_law = lambda d: 0.65 * (1 - np.cos(2 * d)) / 2
        dense = [ctr + np.array([np.cos(d), np.sin(d), 0]) * (1.0 + s_law(d))
                 for d in np.linspace(0, TAU, 240)]
        theo = VMobject(color=GREY_B, stroke_width=2)
        theo.set_points_smoothly([*dense, dense[0]])
        self.play(Create(theo))
        rrv = ValueTracker(0.05)

        def actual():
            rr = rrv.get_value()
            pts = []
            for k in range(240):
                p = np.array(dense[k])
                p2 = np.array(dense[(k + 1) % 240])
                tvec = p2 - p
                nrm = np.array([tvec[1], -tvec[0], 0.0])
                nn = np.linalg.norm(nrm)
                if nn > 1e-9:
                    nrm = nrm / nn
                pts.append(p - nrm * rr * (-1))
            vm = VMobject(color=ORANGE, stroke_width=3)
            vm.set_points_as_corners([*pts, pts[0]])
            return vm

        act = always_redraw(actual)
        lab = always_redraw(lambda: ctext(f"滚子半径 rr = {rrv.get_value():.2f}", size=26,
                                          color=ORANGE).to_edge(DOWN, buff=0.4))
        self.play(FadeIn(act), FadeIn(lab))
        self.play(rrv.animate.set_value(0.45), run_time=5, rate_func=linear)
        self.add(page_ref("孙桓八版 p184-185"))
        self.hold(3)


class S08_Summary(LessonScene):
    """小结（4min）+ 预告 L07：凸轮定制'任意规律'，但要传递大功率定传动比——齿轮时间到。"""

    def construct(self):
        self.header("第 6 讲小结")
        pts = bullets([
            "四种运动规律：看 a 曲线认冲击——刚性/柔性/无        (p171-176)",
            "反转法：凸轮静止、推杆公转伸缩，尖端即轮廓          (p177-179)",
            "tanα = |ds/dδ−e| / (s0+s)；r0 与 α 的权衡          (p181-184)",
            "滚子 rr < ρmin，否则运动失真                        (p184-185)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：世界上最'完美'的曲线——渐开线，为什么统治了齿轮 250 年？",
                  size=30, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
