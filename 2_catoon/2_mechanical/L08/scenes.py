# -*- coding: utf-8 -*-
"""L08 四根杆的智慧(下)——四杆机构设计（§8-4~8-6, p139-160）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_DesignTasks(LessonScene):
    """三类设计问题（~6min, p139-141）：函数生成（转角对应）/轨迹复演（连杆点
    走给定曲线）/刚体导引（连杆过若干位置）——三张示意图。"""

    def construct(self):
        self.header("设计：从'要它干什么'反推杆长", "设计问题分类 · p139-141")
        cards = VGroup()
        names = ["函数生成\nθout = f(θin)", "轨迹复演\n连杆点走指定曲线",
                 "刚体导引\n连杆过给定位置序列"]
        for i, nm in enumerate(names):
            box = RoundedRectangle(corner_radius=0.15, width=3.6, height=1.7,
                                   color=LINK_B, fill_opacity=0.1,
                                   fill_color=LINK_B)
            cards.add(VGroup(box, ctext(nm, size=24).move_to(box)))
        cards.arrange(RIGHT, buff=0.6).shift(UP * 0.6)
        self.play(FadeIn(cards, lag_ratio=0.4), run_time=2.5)
        demo = ctext("共同灵魂：把'动件的位置要求'折算成对铰链点的约束",
                     size=27, color=ACCENT).to_edge(DOWN, buff=1.2)
        self.play(Write(demo))
        self.add(page_ref("孙桓八版 p139-141"))
        self.hold(3)


class S02_PositionSynthesis(LessonScene):
    """按连杆位置设计（~12min, p141-146）：给定连杆两位置 B1C1、B2C2——
    作 B1B2、C1C2 中垂线交得铰链点 A0、B0；现场尺规作图。"""

    def construct(self):
        self.header("按连杆位置设计", "中垂线法 · p141-146")
        B1, C1 = P(-3.6, -0.5), P(-1.4, 0.2)
        B2, C2 = P(-3.0, 1.4), P(-0.8, 1.9)
        pos1 = link_line(B1, C1, LINK_B)
        pos2 = link_line(B2, C2, LINK_C)
        self.play(Create(pos1), Create(pos2))
        self.play(FadeIn(VGroup(Dot(B1), Dot(C1), Dot(B2), Dot(C2))))
        self.play(Write(ctext("给定连杆两位置 B₁C₁、B₂C₂", size=25)
                        .to_edge(UP, buff=1.7)))
        # 中垂线作图
        def midperp(p, q, ext=1.6):
            mid = (p + q) / 2
            d = q - p
            nv = np.array([-d[1], d[0], 0.0])
            nv /= np.linalg.norm(nv)
            return DashedLine(mid - nv * ext, mid + nv * ext, color=ACCENT)
        mp_b = midperp(B1, B2)
        mp_c = midperp(C1, C2)
        seg_b = DashedLine(B1, B2, color=MUTED)
        seg_c = DashedLine(C1, C2, color=MUTED)
        self.play(Create(seg_b), Create(mp_b))
        lab_b = ctext("B₁B₂ 中垂线：A₀ 必在其上", size=23, color=ACCENT)
        lab_c = ctext("C₁C₂ 中垂线：B₀ 必在其上", size=23, color=NOTE)
        legend = VGroup(lab_b, lab_c).arrange(DOWN, aligned_edge=LEFT,
                                              buff=0.3).to_corner(
                                                  DL, buff=0.5).shift(
                                                  UP * 1.7)
        self.play(Write(lab_b))
        self.play(Create(seg_c), Create(mp_c))
        self.play(Write(lab_c))
        self.hold(2)
        self.takeaway("选线上一点当铰链 → 有无穷多解（再按结构/γ 筛选）",
                        p="孙桓八版 p141-146")
        self.hold(3)


class S03_KDesign(LessonScene):
    """按行程速比 K 设计（~14min, p146-150）：K→θ=180(K−1)/(K+1)→作辅助圆
    （弦 C1C2 对圆心角 2θ）→圆上任取 A0 → 曲柄连杆长=(AC2±AC1)/2。"""

    def construct(self):
        self.header("按急回要求设计", "K → 辅助圆 → 杆长 · p146-150")
        steps = [
            mtex(r"\theta = 180°\,\frac{K-1}{K+1}", font_size=46,
                    color=ACCENT),
            mtex(r"\text{作弦 }C_1C_2,\ \text{圆心角 }2\theta"
                    r"\ \Rightarrow\ \text{辅助圆}", font_size=38),
            mtex(r"l_2 = \frac{AC_2 - AC_1}{2},\quad "
                    r"l_3 = \frac{AC_2 + AC_1}{2}", font_size=44, color=ACCENT),
        ]
        formula_reveal(self, steps, anchor=LEFT * 3.0 + UP * 0.8, wait=1.6)
        # 右侧尺规作图
        C1, C2 = P(1.6, -1.4), P(4.2, -1.4)
        Oc = P(2.9, -0.1)
        aux = DashedVMobject(Circle(radius=np.linalg.norm(C1 - Oc),
                                    color=NOTE, stroke_width=3),
                             num_dashes=42)
        aux.move_to(Oc)
        rocker1 = Line(Oc, C1, color=MUTED, stroke_width=2)
        rocker2 = Line(Oc, C2, color=MUTED, stroke_width=2)
        self.play(Create(VGroup(rocker1, rocker2)), Create(aux),
                  FadeIn(Dot(C1)), FadeIn(Dot(C2)),
                  Write(ctext("摇杆两极位 C₁、C₂", size=23).move_to(
                      P(2.9, -2.75))))
        A0 = Oc + np.array([np.cos(PI * 0.82), np.sin(PI * 0.82), 0]) * \
            np.linalg.norm(C1 - Oc)
        self.play(FadeIn(Dot(A0, radius=0.09, color=ACCENT)),
                  Write(ctext("圆上任取 A₀", size=23, color=ACCENT)
                        .next_to(A0, UP, buff=0.2)))
        self.play(FadeIn(VGroup(DashedLine(A0, C1, color=LINK_A),
                                DashedLine(A0, C2, color=LINK_A))))
        self.hold(2)
        self.takeaway("极位共线：AC₁=l₃−l₂、AC₂=l₃+l₂ → 解杆长",
                        p="孙桓八版 p146-150")
        self.hold(3)


class S04_GammaDesign(LessonScene):
    """按传动角/死点设计一瞥（~5min, p150-153）：γ_min≥[γ] 反约束杆长；
    给定 γ_min 的几何含义。"""

    def construct(self):
        self.header("按传动角条件校核/设计", "p150-153")
        fb = FourBar(3.6, 1.0, 2.8, 2.4, origin=np.array([-3.8, -1.3, 0]))
        m = AnimatedFourBar(fb)
        self.play(FadeIn(m.group))
        gline = always_redraw(lambda: ctext(
            f"γ={np.degrees(fb.transmission_angle(m.theta.get_value())):.0f}°",
            size=30, color=ACCENT).to_edge(UP, buff=1.7))
        self.play(FadeIn(gline))
        self.play(m.theta.animate.set_value(0.6 + TAU), run_time=7,
                  rate_func=linear)
        pts = bullets([
            "设计约束：γmin ≥ [γ]（常取 40°~50°）",
            "γmin 出现在曲柄与机架共线位形——画图量取即可校核",
        ], size=26).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p150-153"))
        self.hold(3)


class S05_CouplerAtlas(LessonScene):
    """连杆曲线图谱（~8min, p155-158）：同一四杆、连杆上不同点的轨迹族
    同屏绽放——近似直线/圆弧段可被'借用'做间歇/导引。"""

    def construct(self):
        self.header("连杆曲线：一支笔画万种轨迹", "图谱 · p155-158")
        fb = FourBar(3.6, 1.0, 2.8, 2.4, origin=np.array([-3.2, -1.6, 0]))
        m = AnimatedFourBar(fb, coupler=(0.5, 0.7), trace=True)
        self.play(FadeIn(m.group))
        # 采样多条连杆曲线（不同描点）
        specs = [(0.3, 0.4), (0.5, 0.9), (0.7, -0.3), (0.5, 0.0)]
        colors = [LINK_D, ACCENT, NOTE, GOOD]
        curves = VGroup(*[coupler_curve(fb, s, h, n=160, color=c)
                          for (s, h), c in zip(specs, colors)])
        self.play(FadeIn(curves, lag_ratio=0.4), run_time=2.5)
        self.play(m.theta.animate.set_value(0.6 + TAU), run_time=6,
                  rate_func=linear)
        self.hold(1)
        note = ctext("近似直线段→直线导引；近似圆弧段→'停顿'间歇机构",
                     size=26, color=NOTE).to_edge(DOWN, buff=0.6)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p155-158"))
        self.hold(3)


class S06_Methods(LessonScene):
    """实验法与解析法概览（~4min, p153-155/158-160）：模板试凑、连杆曲线图谱
    查表、解析定杆长——各自适用场景。"""

    def construct(self):
        self.header("还有两把刷子", "实验法 · 解析法 · p153-160")
        pts = bullets([
            "图解法：直观、答案立现——精度靠作图",
            "实验法：实物模型试凑 + 连杆曲线图谱查表",
            "解析法：闭环方程精确解——可控任意位置数",
            "多杆机构（六杆及以上）= 四杆模块的组合",
        ], size=29).shift(UP * 0.2)
        self.play(FadeIn(pts, lag_ratio=0.5), run_time=2.8)
        self.hold(3)
        self.add(page_ref("孙桓八版 p153-160"))
        self.hold(2)


class S07_Summary(LessonScene):
    """小结+下讲悬念（~3min）。"""

    def construct(self):
        self.header("第 8 讲小结")
        pts = bullets([
            "三类设计问题：函数/轨迹/导引              (p139-141)",
            "按位置设计：中垂线交出铰链点              (p141-146)",
            "按 K 设计：θ→辅助圆→(AC₂∓AC₁)/2 得杆长    (p146-150)",
            "校核 γmin≥[γ]；连杆曲线可'借用'          (p150-158)",
        ], size=27)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.6)
        self.hold(3)
        q = ctext("下一讲：想让输出严格按'剧本'运动？——凸轮机构", size=25,
                  color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
