# -*- coding: utf-8 -*-
"""L02 机构能不能动——自由度·三大陷阱·组成原理（§2-4~2-8, p14-24）

渲染：manim -pqh scenes.py <Scene>；整课：manim -qh scenes.py -a
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """片头+三连反例（~6min）：F=0 桁架纹丝不动 / F=2 单输入乱动 / F=1 确定动。
    讲稿：上节会画简图了——但画出来的东西一定能动吗？怎么动？（p14-15 机构具有
    确定运动的条件）"""

    def construct(self):
        self.header("画出来的机构，一定能动吗？", "确定运动的条件 · p14-16")
        # 反例1：桁架 F=0（三根杆三角刚性）
        tri = VGroup(*[link_line(P(-5.4, -0.5), P(-4.2, -0.5), FRAME_C),
                       link_line(P(-4.2, -0.5), P(-4.8, 0.6), FRAME_C),
                       link_line(P(-4.8, 0.6), P(-5.4, -0.5), FRAME_C)],
                     *[pin_joint(p) for p in (P(-5.4, -0.5), P(-4.2, -0.5),
                                              P(-4.8, 0.6))],
                     ground_hatch(P(-4.8, -0.66), 1.6))
        l1 = ctext("F=0：桁架，纹丝不动", size=24, color=BAD).next_to(
            tri, DOWN, buff=0.5)
        # 反例2：五杆 F=2 单输入乱动
        five = VGroup(
            link_line(P(0.4, -0.6), P(1.1, 0.3), LINK_A),
            link_line(P(1.1, 0.3), P(2.0, 0.5), LINK_B),
            link_line(P(2.0, 0.5), P(2.7, -0.2), LINK_C),
            link_line(P(2.7, -0.2), P(3.4, -0.6), LINK_D),
            link_line(P(0.4, -0.6), P(3.4, -0.6), FRAME_C, 5),
            *[pin_joint(p) for p in (P(0.4, -0.6), P(1.1, 0.3), P(2.0, 0.5),
                                     P(2.7, -0.2), P(3.4, -0.6))],
            ground_hatch(P(0.4, -0.76), 0.6), ground_hatch(P(3.4, -0.76), 0.6),
        )
        l2 = ctext("F=2：一个电机，动作'飘忽'", size=22,
                   color=BAD).next_to(five, DOWN, buff=0.45).shift(LEFT * 0.5)
        # 正例：四杆 F=1（标注放低一行，与 F=2 说明错层）
        fb = FourBar(1.5, 0.7, 1.6, 0.85, origin=np.array([4.5, -0.6, 0]))
        m = AnimatedFourBar(fb)
        l3 = ctext("F=1：一个输入，确定运动", size=22,
                   color=GOOD).next_to(m.group, DOWN, buff=1.15).shift(LEFT * 0.8)
        self.play(FadeIn(tri), FadeIn(l1))
        self.play(FadeIn(five), FadeIn(l2))
        self.play(FadeIn(m.group), FadeIn(l3))
        self.play(m.theta.animate.set_value(m.theta.get_value() + TAU),
                  run_time=5, rate_func=linear)
        self.hold(1)
        self.takeaway("机构具有确定运动的条件：原动件数 = 自由度数 F",
                        p="孙桓八版 p14-16")
        self.hold(3)


class S02_DeriveFormula(LessonScene):
    """F=3n−2PL−PH 推导（~12min, p16）：从 3n 个自由度逐项扣除——每个低副锁 2、
    每个高副锁 1。动画：构件堆叠亮出 3n → 运动副图标逐个'扣减'。"""

    def construct(self):
        self.header("平面机构自由度公式", "推导 · p16")
        steps = [
            mtex(r"\text{每个活动构件：3 个自由度（}x,y,\varphi\text{）}",
                    font_size=36),
            mtex(r"n\text{ 个活动构件} \Rightarrow 3n", font_size=40),
            mtex(r"\text{每个低副}(-2),\ \text{每个高副}(-1)", font_size=40),
            mtex(r"F = 3n - 2P_L - P_H", font_size=58, color=ACCENT),
        ]
        # 左侧图示：活动杆 vs 机架
        bar = link_line(ORIGIN, RIGHT * 1.6, LINK_B)
        bar.shift(LEFT * 4.6 + UP * 1.2)
        axes = VGroup(vec(bar.get_center(), RIGHT * 0.5, INK, tip=0.12),
                      vec(bar.get_center(), UP * 0.5, INK, tip=0.12),
                      Arc(radius=0.5, angle=PI / 3, color=INK,
                          arc_center=bar.get_center()))
        lab = ctext("活动构件：x, y, φ", size=23).next_to(bar, DOWN, buff=0.4)
        self.play(FadeIn(bar), FadeIn(axes), FadeIn(lab))
        formula_reveal(self, steps, anchor=RIGHT * 2.4 + UP * 0.6, wait=1.5)
        cond = bullets([
            "F≤0：不能动（桁架）",
            "F>0 且原动件数=F：确定运动",
            "原动件数<F：运动不确定；>F：拉坏",
        ], size=25).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(cond, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p16"))
        self.hold(3)


class S03_BasicExamples(LessonScene):
    """标准算例三连（~8min, p17）：四杆/曲柄滑块/凸轮机构各现场数 F=1。"""

    def construct(self):
        self.header("热身：三个标准算例", "F = 3n − 2PL − PH · p17")
        fb = FourBar(2.5, 0.85, 2.1, 1.9, origin=np.array([-4.9, -0.7, 0]))
        m1 = AnimatedFourBar(fb)
        cs = AnimatedCrankSlider(CrankSlider(0.55, 1.7, 0.0,
                                             origin=np.array([0.2, -0.8, 0])),
                                 cylinder=False)
        cam = AnimatedCamKnife(0.45, cam_law("cos", 0.35, PI),
                               origin=np.array([4.0, -1.3, 0]), scale=0.85)
        for m, x, name, f in ((m1, -3.6, "铰链四杆", "n=3, PL=4 → F=1"),
                              (cs, 1.0, "曲柄滑块", "n=3, PL=4 → F=1"),
                              (cam, 4.0, "凸轮", "n=2, PL=2, PH=1 → F=1")):
            self.add(m.group)
            self.add(ctext(name, size=23).next_to(m.group, UP, buff=0.25))
            self.add(ctext(f, size=20, color=NOTE).next_to(m.group, DOWN,
                                                          buff=0.35))
        self.play(m1.theta.animate.set_value(TAU),
                  cs.theta.animate.set_value(TAU),
                  cam.theta.animate.set_value(TAU), run_time=6,
                  rate_func=linear)
        self.hold(1)
        self.takeaway("数构件 n → 数低副 PL → 数高副 PH → 套公式",
                        p="孙桓八版 p17")
        self.hold(3)


class S04_CompoundHinge(LessonScene):
    """陷阱1：复合铰链（~8min, p18）：三构件共铰处放大——轴上其实有 2 个转动副。"""

    def construct(self):
        self.header("陷阱① 复合铰链", "一处铰链 ≠ 一个运动副 · p18")
        # 三构件共铰特写：同心两圆叠放 → 爆炸分离显示两根销
        jpt = P(-2.5, 0.4)
        a = link_line(jpt, jpt + P(-1.7, 1.5), LINK_A)
        b = link_line(jpt, jpt + P(1.8, 1.4), LINK_B)
        c = link_line(jpt, jpt + P(0.3, -1.9), LINK_C)
        j = pin_joint(jpt, 0.13)
        self.play(*[Create(x) for x in (a, b, c)], FadeIn(j))
        lab = ctext("三构件共铰：藏着几个转动副？", size=26,
                    color=ACCENT).move_to(P(-2.7, 2.15))
        self.play(Write(lab))
        self.hold(1.5)
        # 放大圆：显示销1(构件a-b) + 销2(构件b-c)
        zoom = Circle(radius=1.9, color=MUTED, stroke_width=1.5).move_to(
            RIGHT * 3.6 + DOWN * 0.15)
        pin1 = Circle(radius=0.75, color=LINK_A, stroke_width=5).move_to(
            zoom.get_center() + LEFT * 0.0)
        pin2 = Circle(radius=1.15, color=LINK_C, stroke_width=5,
                      stroke_opacity=0.85).move_to(zoom.get_center())
        t1 = ctext("销 A：构件①② 相对转动", size=24,
                   color=LINK_A).next_to(zoom, UP, buff=0.35)
        t2 = ctext("销 B：构件②③ 相对转动", size=24,
                   color=LINK_C).next_to(zoom, DOWN, buff=0.35)
        self.play(Create(zoom), FadeIn(pin1), Write(t1))
        self.play(FadeIn(pin2), Write(t2))
        self.hold(2)
        ans = mtex(r"k\text{ 构件共铰} \Rightarrow P_L = k-1",
                      font_size=46,
                      color=BAD).to_edge(DOWN, buff=0.9).shift(LEFT * 2.6)
        self.play(Write(ans))
        self.add(page_ref("孙桓八版 p18"))
        self.hold(3)


class S05_LocalDOF(LessonScene):
    """陷阱2：局部自由度（~8min, p19）：凸轮滚子自转不影响输出——'焊死'滚子
    动画：滚子旋转被锁，机构照常运动，F 应减去 1。"""

    def construct(self):
        self.header("陷阱② 局部自由度", "滚子的自转是'摆设' · p19")
        cam = AnimatedCamRoller(0.55, 0.18, cam_law("cos", 0.5, PI * 1.2),
                                origin=np.array([-3.0, -1.6, 0]), scale=1.0,
                                pitch_line=True)
        self.play(FadeIn(cam.group))
        self.play(cam.theta.animate.set_value(TAU), run_time=4,
                  rate_func=linear)
        # 右侧公式：未修正 F=2 → 焊死滚子 F=1
        f_bad = mtex(r"F = 3{\times}3 - 2{\times}3 - 1 = 2\ ?",
                        font_size=44)
        f_ok = mtex(r"F = 3{\times}2 - 2{\times}2 - 1 = 1\ \checkmark",
                       font_size=46, color=GOOD)
        VGroup(f_bad, f_ok).arrange(DOWN, buff=0.6).shift(RIGHT * 2.6 + UP * 0.9)
        note = ctext("把滚子与推杆'焊'为一体：减去局部自由度 1",
                     size=24, color=NOTE).next_to(f_ok, DOWN, buff=0.5)
        self.play(Write(f_bad))
        self.hold(1.5)
        self.play(Write(f_ok), FadeIn(note))
        # 滚子“焊死”标记（跟随滚子顶与推杆的连接处）
        def _weld():
            c = cam.roller_center() + UP * (cam.rr * cam.scale + 0.24)
            return VGroup(Line(LEFT * 0.2 + UP * 0.2, RIGHT * 0.2 + DOWN * 0.2,
                               color=BAD, stroke_width=5),
                          Line(LEFT * 0.2 + DOWN * 0.2, RIGHT * 0.2 + UP * 0.2,
                               color=BAD, stroke_width=5)).shift(c)
        weld = always_redraw(_weld)
        self.play(FadeIn(weld))
        self.hold(2)
        self.takeaway("局部自由度：与输出无关的自身运动，计算时排除",
                        p="孙桓八版 p19")
        self.hold(3)


class S06_VirtualConstraint(LessonScene):
    """陷阱3：虚约束（~12min, p19-21）：平行四边形机构加'平行杆'——多加的约束
    是重复的；机车车轮联动同款。六种典型情形列表收尾。"""

    def construct(self):
        self.header("陷阱③ 虚约束", "重复的限制不算数 · p19-21")
        # 平行四边形机构
        fb = FourBar(3.2, 1.3, 3.2, 1.3, origin=np.array([-3.6, -1.0, 0]))
        th = ValueTracker(0.5)

        def para(extra=True):
            A0, A, B, B0 = fb.solve(th.get_value())
            g = [link_line(A0, A, LINK_A), link_line(A, B, LINK_B),
                 link_line(B, B0, LINK_C), link_line(A0, B0, FRAME_C, 5),
                 ground_hatch(A0 + DOWN * 0.16, 0.6),
                 ground_hatch(B0 + DOWN * 0.16, 0.6),
                 *[pin_joint(p) for p in (A0, A, B, B0)]]
            if extra:
                M0 = (A0 + A) / 2
                M1 = M0 + (B - A)
                g += [link_line(M0, M1, BAD, 5), pin_joint(M0), pin_joint(M1)]
            return VGroup(*g)
        mech = always_redraw(lambda: para(True))
        self.play(FadeIn(mech))
        t1 = ctext("中间加一根'平行杆'：轨迹其实已经被原机构限定",
                   size=26, color=BAD).to_edge(UP, buff=1.7)
        self.play(Write(t1))
        self.play(th.animate.set_value(0.5 + TAU), run_time=5,
                  rate_func=linear)
        self.hold(1)
        # 杆淡出=虚约束
        mech2 = always_redraw(lambda: para(False))
        self.play(FadeOut(mech), FadeIn(mech2))
        t2 = ctext("去掉它，运动不变 → 那是虚约束（重复约束）", size=26,
                   color=GOOD).to_edge(UP, buff=1.7)
        self.play(Transform(t1, t2))
        self.play(th.animate.set_value(th.get_value() + TAU), run_time=4,
                  rate_func=linear)
        lst = bullets([
            "两构件多处构成移动副且导路平行",
            "两点距离不变处再加连接（本例）",
            "对称/重复部分引入的约束",
            "高副处加'带低副'的等宽接触",
        ], size=23).to_edge(DOWN, buff=0.45)
        self.play(FadeIn(lst, lag_ratio=0.35))
        self.add(page_ref("孙桓八版 p19-21"))
        self.hold(3)


class S07_BigExample(LessonScene):
    """综合大算例（~10min, p21 图）：含复合铰链+局部自由度+虚约束的机构
    现场数 F——逐项标注修正，得 F=1。"""

    def construct(self):
        self.header("大算例：陷阱全踩一遍", "p21")
        eq = mtex(
            r"F = 3n - 2P_L - P_H = 3{\times}7 - 2{\times}9 - 1 = 2\ ?",
            font_size=44).shift(UP * 1.6)
        self.play(Write(eq))
        fixes = bullets([
            "C 处三构件共铰：PL 记 2（已含）",
            "滚子局部自由度：n、PL 各减 1 → F=1",
            "E 处虚约束：不计",
        ], size=25).shift(LEFT * 2.4 + DOWN * 1.25)
        for f in fixes:
            self.play(FadeIn(f, shift=RIGHT * 0.3))
            self.hold(1.2)
        ans = mtex(r"F = 3{\times}6 - 2{\times}8 - 1 = 1",
                      font_size=52, color=GOOD).next_to(eq, DOWN, buff=1.0)
        self.play(Write(ans))
        self.focus(ans)
        self.hold(2)
        self.takeaway("先排陷阱（共铰/局部/虚约束），再套公式", p="孙桓八版 p21")
        self.hold(3)


class S08_AssurGroups(LessonScene):
    """机构组成原理（~10min, p21-24）：机构 = 机架+原动件+基本杆组。
    Ⅱ级组条件 3n−2P_L=0 → n=2,P_L=3；拼接动画：机架+曲柄→外接Ⅱ级组→得四杆。"""

    def construct(self):
        self.header("机构是怎么'搭'出来的", "基本杆组 · p21-24")
        # 步骤1：机架+原动件
        A0, B0 = P(-4.5, -0.8), P(0.5, -0.8)
        crank = link_line(A0, A0 + P(0.9, 1.3), LINK_A)
        g1 = VGroup(link_line(A0, B0, FRAME_C, 5), crank,
                    ground_hatch(A0 + DOWN * 0.16, 0.6),
                    ground_hatch(B0 + DOWN * 0.16, 0.6),
                    pin_joint(A0), pin_joint(B0),
                    pin_joint(A0 + P(0.9, 1.3)))
        t1 = ctext("① 机架 + 原动件（F=1）", size=25).to_edge(UP, buff=1.7)
        self.play(FadeIn(g1), Write(t1))
        self.hold(1.5)
        # 步骤2：Ⅱ级组 = 2 构件 3 低副
        A = A0 + P(0.9, 1.3)
        B = B0 + P(-0.5, 1.5)
        grp = VGroup(link_line(A, (A + B) / 2 + P(0, 0.35), LINK_B),
                     link_line((A + B) / 2 + P(0, 0.35), B, LINK_B),
                     link_line(B, B0, LINK_C),
                     pin_joint((A + B) / 2 + P(0, 0.35)), pin_joint(B))
        t2 = ctext("② 外接Ⅱ级杆组：n=2, PL=3（3n−2PL=0，不改变 F）",
                   size=25, color=ACCENT).to_edge(UP, buff=1.7)
        self.play(FadeIn(grp), Transform(t1, t2))
        self.hold(2)
        # 拼接结果=六杆? 此处示意四杆+一组
        steps = [
            mtex(r"\text{杆组条件：}\ 3n - 2P_L = 0", font_size=42),
            mtex(r"n=2,\ P_L=3\ \Rightarrow\ \mathrm{II}\ \text{级组}",
                    font_size=40),
            mtex(r"\text{机构级别} = \text{最高杆组级别}", font_size=40),
        ]
        formula_reveal(self, steps, anchor=RIGHT * 4.0 + DOWN * 2.2,
                       wait=1.4)
        self.add(page_ref("孙桓八版 p21-24"))
        self.hold(3)


class S09_Summary(LessonScene):
    """小结+下讲悬念（~3min）：F 公式+三陷阱+杆组；下讲：会动了，动多快？"""

    def construct(self):
        self.header("第 2 讲小结")
        pts = bullets([
            "F = 3n − 2PL − PH；原动件数=F 才确定动   (p16)",
            "陷阱①：复合铰链 k 构件 → k−1 个副          (p18)",
            "陷阱②：局部自由度（滚子自转）要扣除        (p19)",
            "陷阱③：虚约束（重复限制）不计入           (p19-21)",
            "机构 = 机架+原动件+基本杆组；级别=最高组级  (p21-24)",
        ], size=26)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：能动之后——它动多快？瞬心与运动分析", size=25,
                  color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
