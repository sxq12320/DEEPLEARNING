# -*- coding: utf-8 -*-
"""L04 四根杆的智慧（上）——连杆机构的类型与特性（孙桓八版 §8-1~8-3, p123-139）

目标成片 85-95 min。核心推导：Grashof 曲柄存在条件、行程速比系数 K、最小传动角。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """开场（3min）：从雨刮器、缝纫机到抽油机——都是四杆机构。为什么偏爱低副？
    (p123-124: 面接触压强低、易制造润滑、承载大；缺点：路径误差累积、惯性力难平衡)"""

    def construct(self):
        big = ctext("第 4 讲   四根杆的智慧（上）", size=56, weight="BOLD")
        sub = ctext("类型 · 曲柄条件 · 急回 · 传动角 · 死点", size=32, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.5)
        self.play(Write(big), FadeIn(sub))
        self.hold(2.5)
        pts = bullets([
            "优点：低副面接触——承载大、耐磨、好制造",
            "优点：能实现转动↔摆动↔移动的花式转换",
            "缺点：累积误差、高速惯性力不易平衡",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p123-124"))
        self.hold(3)


class S02_ThreeTypes(LessonScene):
    """三种基本形式（12min，p124-131）：曲柄摇杆/双曲柄/双摇杆 + 应用。
    动画：三套机构并排运转——雷达天线俯仰(曲柄摇杆)、惯性筛(双曲柄)、起重机(双摇杆)。"""

    def construct(self):
        self.header("铰链四杆机构的三种形式", "以连架杆能否整转分类 · p124-131")
        # 曲柄摇杆
        fb1 = FourBar(2.6, 0.8, 2.4, 1.8, origin=np.array([-5.9, -0.6, 0]))
        # 双曲柄（Grashof 且最短为机架）
        fb2 = FourBar(0.9, 2.0, 1.6, 2.2, origin=np.array([-1.5, -0.6, 0]))
        # 双摇杆（非 Grashof）
        fb3 = FourBar(2.4, 1.7, 1.5, 1.7, origin=np.array([2.9, -0.6, 0]))
        th = ValueTracker(0.6)

        def mk(fb, crank_full=True):
            def _f():
                a = th.get_value() if crank_full else 0.6 + 0.7 * np.sin(th.get_value())
                A0, A, B, B0 = fb.solve(a)
                return VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B), link_line(B, B0, TEAL),
                              *[pin_joint(p) for p in (A0, A, B, B0)],
                              ground_hatch(A0 + DOWN * 0.14, 0.5), ground_hatch(B0 + DOWN * 0.14, 0.5))
            return _f

        m1 = always_redraw(mk(fb1)); m2 = always_redraw(mk(fb2)); m3 = always_redraw(mk(fb3, crank_full=False))
        labels = VGroup(
            ctext("曲柄摇杆\n雷达俯仰 · 雨刮", size=24, color=GOLD).shift(LEFT * 4.6 + DOWN * 2.6),
            ctext("双曲柄\n惯性筛 · 旋转泵", size=24, color=BLUE_B).shift(LEFT * 0.2 + DOWN * 2.6),
            ctext("双摇杆\n起重机 · 电风扇摇头", size=24, color=TEAL).shift(RIGHT * 4.2 + DOWN * 2.6),
        )
        self.play(FadeIn(m1), FadeIn(m2), FadeIn(m3), FadeIn(labels, lag_ratio=0.3))
        self.play(th.animate.set_value(0.6 + 2 * TAU), run_time=10, rate_func=linear)
        self.add(page_ref("孙桓八版 p124-131"))
        self.hold(2)


class S03_Grashof(LessonScene):
    """★曲柄存在条件推导（15min，p131-133，本讲最重要推导）。
    讲稿要点：整转副存在 ⇔ 曲柄能通过与机架共线的两个极限位置 ⇔ 由三角形两边之和
    大于第三边，在两共线位形分别列不等式 → 最短杆+最长杆 ≤ 其余两杆之和（Grashof），
    且曲柄是最短杆、其邻杆为机架 → 曲柄摇杆；最短杆为机架 → 双曲柄；对边 → 双摇杆。"""

    def construct(self):
        self.header("曲柄存在的条件", "Grashof 判据推导 · p131-133")
        fb = FourBar(3.4, 1.0, 3.0, 2.2, origin=np.array([-4.6, -1.5, 0]))
        # 两个共线极限位形
        A0 = fb.o; B0 = fb.o + np.array([fb.L1, 0, 0])
        t0 = ctext("曲柄要整转，必须能'挤过'与机架共线的两个最难位形：", size=28).to_edge(UP, buff=1.5)
        self.play(Write(t0))
        # 位形1：曲柄与机架重叠方向（A 在 A0 左侧？→ 拉直: AB0 = L2+L1? 教材推导用 A0A 与 A0B0 共线两种）
        A1 = A0 + np.array([-fb.L2, 0, 0])   # 曲柄反向共线：AB0 距离 = L1+L2
        A2 = A0 + np.array([fb.L2, 0, 0])    # 曲柄正向共线：AB0 距离 = L1-L2
        def frame(A, col):
            _, _, B, _ = None, None, None, None
            d_vec = B0 - A; d = np.linalg.norm(d_vec)
            a = (fb.L3 ** 2 - fb.L4 ** 2 + d ** 2) / (2 * d)
            h = np.sqrt(max(fb.L3 ** 2 - a ** 2, 0.01))
            u = d_vec / d; nv = np.array([-u[1], u[0], 0])
            B = A + a * u + h * nv
            return VGroup(link_line(A0, A, GOLD), link_line(A, B, col), link_line(B, B0, TEAL),
                          *[pin_joint(p) for p in (A0, A, B, B0)])
        f1 = frame(A1, BLUE_B)
        lbl1 = MathTex(r"\triangle:\ l_3 + l_4 \ge l_1 + l_2", font_size=40, color=YELLOW).shift(RIGHT * 3.4 + UP * 1.1)
        self.play(FadeIn(f1), Write(lbl1))
        self.hold(2.2)
        f2 = frame(A2, ORANGE)
        lbl2 = MathTex(r"\triangle:\ |l_3 - l_4| \le l_1 - l_2", font_size=40, color=YELLOW).next_to(lbl1, DOWN, buff=0.5)
        self.play(FadeIn(f2), Write(lbl2))
        self.hold(2.2)
        concl = [
            MathTex(r"\text{整理两组三角形不等式} \Rightarrow", font_size=38),
            MathTex(r"\boxed{l_{\min} + l_{\max} \le \text{其余两杆之和}}", font_size=50, color=GREEN),
            MathTex(r"\text{且整转副由最短杆与其邻杆构成}", font_size=38),
        ]
        formula_reveal(self, concl, anchor=DOWN * 2.2, buff=0.35, wait=1.8)
        self.add(page_ref("孙桓八版 p131-133"))
        self.hold(2.5)


class S04_GrashofCorollary(LessonScene):
    """Grashof 推论：机架取法决定机构类型（8min，p132-133）。
    动画：同一 Grashof 运动链，高亮不同构件为机架 → 曲柄摇杆/双曲柄/双摇杆；
    不满足 Grashof → 无论谁当机架都是双摇杆。（呼应 L01 S08 的伏笔）"""

    def construct(self):
        self.header("同一运动链，三种命运", "机架取法定类型 · p132-133")
        rows = VGroup(
            ctext("满足 Grashof + 最短杆的邻杆为机架  → 曲柄摇杆", size=30, color=GOLD),
            ctext("满足 Grashof + 最短杆自己为机架    → 双曲柄", size=30, color=BLUE_B),
            ctext("满足 Grashof + 最短杆的对杆为机架  → 双摇杆", size=30, color=TEAL),
            ctext("不满足 Grashof → 谁当机架都是双摇杆", size=30, color=GREY_B),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.55).shift(UP * 0.4)
        for r in rows:
            self.play(FadeIn(r, shift=RIGHT * 0.4), run_time=0.8)
            self.hold(1.5)
        note = ctext("这就是 L01 埋的伏笔：固定不同构件 = 得到不同机构", size=28, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p132-133"))
        self.hold(3)


class S05_QuickReturn(LessonScene):
    """★急回特性与行程速比系数 K（15min，p133-134）。
    讲稿要点：摇杆两极限位置对应曲柄两共线位形；极位夹角 θ；工作行程扫过 180°+θ、
    回程 180°−θ，等速转动 → 时间比 = K = (180°+θ)/(180°−θ)；θ>0 才有急回。
    动画：曲柄摇杆运转，摇杆到极限位置时定格标注 θ，双向行程计时条对比。"""

    def construct(self):
        self.header("急回特性", "行程速比系数 K · p133-134")
        fb = FourBar(3.6, 1.1, 3.2, 2.2, origin=np.array([-4.2, -1.2, 0]))
        th = ValueTracker(0.0)

        def mech():
            A0, A, B, B0 = fb.solve(th.get_value())
            return VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B), link_line(B, B0, TEAL),
                          *[pin_joint(p) for p in (A0, A, B, B0)],
                          ground_hatch(A0 + DOWN * 0.15, 0.55), ground_hatch(B0 + DOWN * 0.15, 0.55))

        self.play(FadeIn(always_redraw(mech)))
        t1 = ctext("摇杆的两个极限位置 ↔ 曲柄与连杆两次共线", size=28).to_edge(DOWN, buff=1.25)
        self.play(Write(t1))
        self.play(th.animate.set_value(TAU), run_time=6, rate_func=linear)
        steps = [
            MathTex(r"\text{极位夹角 } \theta:\ \text{两共线位置间曲柄的锐角}", font_size=38),
            MathTex(r"t_\text{工作} : t_\text{回程} = (180^\circ + \theta) : (180^\circ - \theta)", font_size=40),
            MathTex(r"\boxed{K = \frac{v_\text{回均}}{v_\text{工均}} = \frac{180^\circ + \theta}{180^\circ - \theta}}",
                    font_size=48, color=YELLOW),
            MathTex(r"\theta = 180^\circ\frac{K-1}{K+1}\quad(\text{设计用反解})", font_size=38, color=GREEN),
        ]
        self.play(FadeOut(t1))
        formula_reveal(self, steps, anchor=RIGHT * 2.9 + UP * 0.3, buff=0.35, wait=1.8)
        note = ctext("牛头刨床：切削慢而稳、空回快省时——白捡的效率", size=27, color=ACCENT).to_edge(DOWN, buff=0.25)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p133-134"))
        self.hold(3)


class S06_TransmissionAngle(LessonScene):
    """压力角与传动角（12min，p134-136）。
    讲稿要点：压力角 α = 从动件受力方向与受力点速度方向夹角；γ = 90°−α 便于度量
    （连杆与摇杆的夹角或其补角）；γ 越大传力越好；γ_min 出现在曲柄与机架两次共线处，
    用余弦定理求两处 γ 取小者；设计许用 [γ] ≈ 40°(一般)/50°(高速大功率)。"""

    def construct(self):
        self.header("压力角 α 与传动角 γ", "传力好坏的'体检指标' · p134-136")
        fb = FourBar(3.6, 1.1, 3.2, 2.2, origin=np.array([-4.4, -1.4, 0]))
        th = ValueTracker(0.4)

        def mech():
            A0, A, B, B0 = fb.solve(th.get_value())
            g = VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B), link_line(B, B0, TEAL),
                       *[pin_joint(p) for p in (A0, A, B, B0)],
                       ground_hatch(A0 + DOWN * 0.15, 0.55), ground_hatch(B0 + DOWN * 0.15, 0.55))
            # γ = 连杆与摇杆夹角
            v1 = A - B; v2 = B0 - B
            ang = np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)), -1, 1)))
            gamma = ang if ang <= 90 else 180 - ang
            col = GREEN if gamma > 40 else RED
            g.add(ctext(f"γ = {gamma:4.1f}°", size=30, color=col).next_to(B, UP + RIGHT, buff=0.3))
            return g

        self.play(FadeIn(always_redraw(mech)))
        t1 = ctext("γ = 连杆与摇杆夹角（或其补角，取锐角）——实时看它变化：", size=27).to_edge(DOWN, buff=1.3)
        self.play(Write(t1))
        self.play(th.animate.set_value(0.4 + 2 * TAU), run_time=10, rate_func=linear)
        f = [
            MathTex(r"\gamma_{\min}\ \text{出现在曲柄与机架共线的两位置之一}", font_size=36),
            MathTex(r"\cos\angle B = \frac{l_3^2+l_4^2-(l_1\mp l_2)^2}{2\,l_3 l_4}\ \text{（两处取小）}",
                    font_size=40, color=YELLOW),
            MathTex(r"[\gamma] \approx 40^\circ\ (\text{一般})\,/\,50^\circ\ (\text{高速大功率})", font_size=36, color=GREEN),
        ]
        self.play(FadeOut(t1))
        formula_reveal(self, f, anchor=RIGHT * 2.9 + UP * 1.0, buff=0.4, wait=1.7)
        self.add(page_ref("孙桓八版 p134-136"))
        self.hold(3)


class S07_DeadPoint(LessonScene):
    """死点（10min，p136-137）。
    讲稿要点：从动件与连杆共线 → γ=0 → 再大的力也只产生轴向压力不产生力矩 → 卡死或运动不确定；
    发生条件：摇杆(或滑块)主动、曲柄从动；克服：飞轮惯性/多组机构错列；
    利用：飞机起落架锁死、夹具自锁夹紧。动画：缝纫机踏板蹬不动的瞬间。"""

    def construct(self):
        self.header("死点：卡住的瞬间", "γ = 0 的灾难与妙用 · p136-137")
        fb = FourBar(3.4, 1.0, 3.0, 2.1, origin=np.array([-4.2, -1.0, 0]))
        # 死点位形：曲柄与连杆共线（B0B 主动时）
        th_dead = 0.0  # 近似演示
        A0, A, B, B0 = fb.solve(th_dead)
        mech = VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B), link_line(B, B0, TEAL),
                      *[pin_joint(p) for p in (A0, A, B, B0)])
        self.play(FadeIn(mech))
        force = Arrow(B + UP * 1.2, B, buff=0, color=RED, stroke_width=8)
        t1 = ctext("摇杆主动使劲推，但力线正好穿过曲柄转轴——力矩为零！", size=28).to_edge(DOWN, buff=1.3)
        self.play(GrowArrow(force), Write(t1))
        self.hold(2.5)
        sol = bullets([
            "克服①：装飞轮，靠惯性'冲'过去（缝纫机脚踏轮）",
            "克服②：多组机构错开相位（机车双侧车轮错 90°）",
            "利用：起落架/夹具在死点位置'锁死'——越压越紧",
        ], size=27).to_edge(DOWN, buff=0.15)
        self.play(ReplacementTransform(t1, sol))
        self.add(page_ref("孙桓八版 p136-137"))
        self.hold(3.5)


class S08_Evolutions(LessonScene):
    """演化家族（10min，p128-131）：转动副→移动副的演化谱系。
    动画：曲柄摇杆 → 摇杆无限长变滑块（曲柄滑块）→ 再演化导杆/摇块/定块；偏心轮回收 L01。"""

    def construct(self):
        self.header("四杆机构的演化家族", "把转动副'拉直' · p128-131")
        rows = bullets([
            "摇杆 → 无限长 ⇒ 转动副退化为移动副：曲柄滑块",
            "取不同构件为机架：导杆机构（牛头刨）· 摇块机构（自卸车油缸）· 定块机构（手压泵）",
            "扩大转动副 ⇒ 偏心轮（L01 见过：偏心距 = 曲柄长）",
        ], size=28).shift(UP * 1.3)
        self.play(FadeIn(rows, lag_ratio=0.4), run_time=2.5)
        cs = CrankSlider(0.9, 2.6, 0, origin=np.array([-2.6, -1.8, 0]))
        th = ValueTracker(0)

        def mech():
            O, A, B = cs.solve(th.get_value())
            return VGroup(link_line(O, A, GOLD), link_line(A, B, BLUE_B),
                          slider_block(B, 0.6, 0.36), pin_joint(A), fixed_pin(O),
                          Line(O + RIGHT * 0.8 + DOWN * 0.24, O + RIGHT * 4.3 + DOWN * 0.24,
                               color=GREY_B, stroke_width=4))

        self.play(FadeIn(always_redraw(mech)))
        self.play(th.animate.set_value(2 * TAU), run_time=6, rate_func=linear)
        self.add(page_ref("孙桓八版 p128-131"))
        self.hold(2.5)


class S09_Summary(LessonScene):
    """小结（4min）+ 预告 L05：知道'性格'了，如何按需求'定制'一个四杆机构？"""

    def construct(self):
        self.header("第 4 讲小结")
        pts = bullets([
            "三种基本形式：看两连架杆能否整转             (p124-131)",
            "Grashof：l_min + l_max ≤ 其余两杆之和        (p131-133)",
            "急回：K = (180°+θ)/(180°−θ)                 (p133-134)",
            "传动角 γ ≥ [γ]≈40°；γ_min 在曲柄-机架共线处   (p134-136)",
            "死点：γ=0；飞轮冲过 / 错列避开 / 夹具利用      (p136-137)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：给定 K、给定位置、给定轨迹——怎么把机构'设计'出来？", size=30,
                  color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(Write(q))
        self.hold(3)
