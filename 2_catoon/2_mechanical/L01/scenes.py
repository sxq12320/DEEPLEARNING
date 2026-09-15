# -*- coding: utf-8 -*-
"""L01 什么是机械——绪论·构件与运动副·运动简图（第1章 + §2-1~2-3, p1-14）

渲染：manim -pqh scenes.py <Scene>；整课：manim -qh scenes.py -a
每 Scene docstring = 配音讲稿骨架（含教材页码）。
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """片头（~2min）：课程定位——机械专业的'内功心法'；内燃机贯穿预告。"""

    def construct(self):
        big = ctext("机械原理", size=100, weight="BOLD")
        sub = ctext("动画公益课 · 第 1 讲", size=36, color=MUTED)
        eng = ctext("一切机器，都是机构的组合", size=30, color=ACCENT)
        VGroup(big, sub, eng).arrange(DOWN, buff=0.55)
        self.play(Write(big), run_time=1.6)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.play(FadeIn(eng, shift=UP * 0.2))
        self.hold(2.5)
        agenda = bullets([
            "机器 · 机构 · 机械：三个词的关系",
            "构件与运动副：机构的'原子'与'关节'",
            "运动简图：把一台机器画成几根线",
        ], size=28).shift(DOWN * 0.2)
        # 大标题收缩到左上角化作常驻角标（不消失，连续感）
        self.play(big.animate.scale(0.32).to_corner(UP + LEFT, buff=0.55),
                  FadeOut(sub, eng), run_time=1.2)
        self.play(FadeIn(agenda, lag_ratio=0.3), run_time=1.8)
        self.hold(3)


class S02_EngineDissect(LessonScene):
    """从内燃机说起（~10min，图1-1, p1）：整机拆成 曲柄滑块+正时齿轮+配气凸轮
    三机构同屏运转 → 机器=机构的组合。此机构贯穿全课，L14 收官回收。"""

    def construct(self):
        self.header("一台单缸内燃机里藏着什么？", "教材图 1-1 · p1")
        # —— 曲柄滑块（左）
        cs = AnimatedCrankSlider(CrankSlider(0.8, 2.3, 0.0,
                                           origin=np.array([-4.6, -1.6, 0])))
        lb1 = ctext("① 曲柄滑块机构\n活塞·连杆·曲轴", size=24,
                    color=LINK_A).next_to(cs.group, UP, buff=0.5)
        # —— 正时齿轮（中）
        gp = AnimatedGearPair(12, 24, m=0.075, c1=np.array([-0.4, -2.2, 0]))
        lb2 = ctext("② 齿轮机构\n曲轴→凸轮轴 正时 1:2", size=24,
                    color=GEAR_1).next_to(gp.group, UP, buff=0.55)
        # —— 配气凸轮（右上）
        cam = AnimatedCamKnife(0.55, cam_law("cos", 0.45, PI * 1.1),
                               origin=np.array([3.6, -0.35, 0]))
        lb3 = ctext("③ 凸轮机构\n按'剧本'开闭气门", size=24,
                    color=LINK_D).next_to(cam.group, LEFT, buff=0.5)
        self.play(FadeIn(cs.group), FadeIn(lb1))
        self.play(FadeIn(gp.group), FadeIn(lb2))
        self.play(FadeIn(cam.group), FadeIn(lb3))
        self.hold(1)
        self.play(cs.theta.animate.set_value(2 * TAU),
                  gp.theta.animate.set_value(2 * TAU),
                  cam.theta.animate.set_value(4 * TAU),
                  run_time=8, rate_func=linear)
        self.hold(1)
        self.takeaway("机器 = 若干机构的组合（协同完成能量转换）", p="孙桓八版 p1-2")
        self.hold(3)


class S03_Definitions(LessonScene):
    """机器/机构/机械三定义（~8min, p1-2）：三特征逐条点亮→框选分级。"""

    def construct(self):
        self.header("机器 · 机构 · 机械", "定义与三特征 · p1-2")
        feats = bullets([
            "① 人为的实体（构件）组合",
            "② 各部分之间具有确定的相对运动",
            "③ 能转换或传递 能量 · 物料 · 信息",
        ], size=32).shift(UP * 0.7)
        self.play(FadeIn(feats, lag_ratio=0.5), run_time=2.5)
        self.hold(2)
        b1 = SurroundingRectangle(feats, color=GOOD, buff=0.35)
        t1 = ctext("三条全满足 → 机器 machine", size=28, color=GOOD)
        t1.next_to(b1, DOWN, buff=0.4)
        self.play(Create(b1), Write(t1))
        self.hold(2)
        b2 = SurroundingRectangle(VGroup(feats[0], feats[1]), color=NOTE, buff=0.28)
        t2 = ctext("只满足①② → 机构 mechanism", size=28, color=NOTE)
        t2.next_to(t1, DOWN, buff=0.32)
        self.play(ReplacementTransform(b1.copy(), b2), Write(t2))
        self.hold(2)
        t3 = ctext("机械 machinery = 机器 + 机构 的总称", size=30,
                   color=ACCENT).next_to(t2, DOWN, buff=0.5)
        self.play(Write(t3))
        self.hold(2)
        eg = ctext("例：内燃机/电动机/机床是机器；台虎钳、千斤顶是机构",
                   size=24, color=MUTED).next_to(t3, DOWN, buff=0.4)
        self.play(FadeIn(eg))
        self.add(page_ref("孙桓八版 p1-2"))
        self.hold(3)


class S04_CourseMap(LessonScene):
    """本课地图（~5min, p2）：五大研究板块 → 对应讲次预告。"""

    def construct(self):
        self.header("这门课研究什么？", "五大板块 · p2")
        items = ["机构的结构分析", "机构的运动分析", "机器动力学",
                 "常用机构分析与设计", "机械系统方案设计"]
        maps = ["→ 第 1-2 讲", "→ 第 3 讲", "→ 第 4-6 讲", "→ 第 7-13 讲",
                "→ 第 14 讲"]
        cards = VGroup()
        for s, m in zip(items, maps):
            box = RoundedRectangle(corner_radius=0.15, width=4.9, height=0.8,
                                   color=LINK_B, fill_opacity=0.12,
                                   fill_color=LINK_B)
            row = VGroup(box, ctext(s, size=27).move_to(box),
                         ctext(m, size=24, color=ACCENT).next_to(box, RIGHT,
                                                                buff=0.55))
            cards.add(row)
        cards.arrange(DOWN, buff=0.32).shift(LEFT * 1.6 + DOWN * 0.35)
        for c in cards:
            self.play(FadeIn(c, shift=RIGHT * 0.4), run_time=0.55)
            self.hold(0.7)
        self.add(page_ref("孙桓八版 p2"))
        self.hold(3)


class S05_LinkVsPart(LessonScene):
    """构件≠零件（~8min, p5 图2-1）：连杆爆炸拆解→刚连为一构件。"""

    def construct(self):
        self.header("构件 ≠ 零件", "运动单元 vs 制造单元 · p5")
        # 连杆爆炸图：杆身/大头盖/螺栓
        body = Polygon([-1.9, 0.45, 0], [1.9, 0.45, 0], [1.4, -0.45, 0],
                       [-1.4, -0.45, 0], color=LINK_B, fill_opacity=0.25,
                       fill_color=LINK_B, stroke_width=3)
        cap = Polygon([-1.5, -0.45, 0], [1.5, -0.45, 0], [1.3, -1.05, 0],
                      [-1.3, -1.05, 0], color=LINK_D, fill_opacity=0.25,
                      fill_color=LINK_D, stroke_width=3)
        eye_l = Circle(radius=0.42, color=LINK_B, stroke_width=4).shift(LEFT * 1.55)
        eye_r = Circle(radius=0.55, color=LINK_B, stroke_width=4).shift(RIGHT * 1.62)
        bolts = VGroup(*[Dot(p, radius=0.09, color=MUTED)
                         for p in ([-0.9, -0.75, 0], [0.9, -0.75, 0])])
        conn = VGroup(body, cap, eye_l, eye_r, bolts).shift(UP * 0.6)
        self.play(FadeIn(conn, lag_ratio=0.4), run_time=2)
        labels = VGroup(
            ctext("连杆体", size=23, color=LINK_B).next_to(body, UP, buff=0.35),
            ctext("大头盖+轴瓦", size=23, color=LINK_D).next_to(cap, DOWN, buff=0.35),
            ctext("螺栓·螺母", size=23, color=MUTED).next_to(
                bolts, DOWN, buff=0.5).shift(LEFT * 2.0),
        )
        self.play(FadeIn(labels, lag_ratio=0.4))
        self.hold(2)
        box = SurroundingRectangle(conn, color=GOOD, buff=0.35)
        t = ctext("刚连成一体 → 一个构件 link", size=22, color=GOOD)
        t.next_to(box, RIGHT, buff=0.35)
        t.shift(LEFT * max(0, t.get_right()[0] - 6.9))
        self.play(Create(box), Write(t))
        self.hold(2.5)
        concl = bullets([
            "零件 part：独立的制造单元",
            "构件 link：独立的运动单元（可由多零件刚连）",
            "机构分析只关心构件",
        ], size=27).to_edge(DOWN, buff=0.55)
        self.play(FadeIn(concl, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p5"))
        self.hold(3)


class S06_KinematicPair(LessonScene):
    """运动副与 f=6−s（~14min, p6）：空间构件 6 自由度逐个点亮→约束逐个锁死。"""

    def construct(self):
        self.header("运动副：构件的关节", "自由度与约束 · p6")
        cube = Square(side_length=1.3, color=LINK_B, fill_opacity=0.15)
        cube.shift(LEFT * 4.4 + UP * 0.5)
        self.play(Create(cube))
        names = ["移动 x", "移动 y", "移动 z", "转动 x", "转动 y", "转动 z"]
        labs = VGroup(*[ctext(s, size=24) for s in names])
        labs.arrange_in_grid(rows=2, cols=3, buff=0.55).shift(RIGHT * 1.4 + UP * 1.6)
        # 演示三个可见自由度
        self.play(cube.animate.shift(RIGHT * 0.7), run_time=0.55)
        self.play(cube.animate.shift(LEFT * 0.7), run_time=0.55)
        self.play(cube.animate.shift(UP * 0.5), run_time=0.55)
        self.play(cube.animate.shift(DOWN * 0.5), run_time=0.55)
        self.play(Rotate(cube, PI / 6), Rotate(cube, -PI / 6), run_time=1.1)
        self.play(FadeIn(labs, lag_ratio=0.25), run_time=1.8)
        t6 = ctext("自由的空间构件：6 个自由度", size=28,
                   color=ACCENT).next_to(labs, DOWN, buff=0.5)
        self.play(Write(t6))
        self.hold(2)
        # 逐个“锁死”
        for i, lab in enumerate(labs):
            self.play(lab.animate.set_color(MUTED).set_opacity(0.45),
                      run_time=0.4)
        lock = ctext("每加 1 个约束 s，就少 1 个自由度", size=26,
                     color=MUTED).next_to(t6, DOWN, buff=0.35)
        self.play(Write(lock))
        steps = [
            mtex(r"f = 6 - s", font_size=58, color=ACCENT),
            mtex(r"s=1,\dots,5 \Rightarrow \mathrm{I}\sim\mathrm{V}\ "
                    r"\text{级副}", font_size=38),
        ]
        formula_reveal(self, steps, anchor=DOWN * 1.9, wait=1.6)
        self.add(page_ref("孙桓八版 p6"))
        self.hold(2.5)


class S07_PairTypes(LessonScene):
    """运动副分类（~10min, p6-7）：低副面接触 vs 高副点线接触 vs 封闭方式。"""

    def construct(self):
        self.header("运动副的分类", "低副 · 高副 · 封闭 · p6-7")
        # 左：转动副
        rev = VGroup(fixed_pin(LEFT * 5.0 + UP * 0.3),
                     link_line(LEFT * 5.0 + UP * 0.3, LEFT * 3.6 + UP * 1.3))
        rl = ctext("转动副 R", size=25, color=LINK_B).next_to(rev, DOWN, buff=0.55)
        self.play(Create(rev), FadeIn(rl))
        self.play(Rotate(rev[1], PI / 5, about_point=rev[1].get_start()),
                  Rotate(rev[1], -PI / 5, about_point=rev[1].get_start()),
                  run_time=1.5)
        # 中：移动副
        gc = LEFT * 0.6 + UP * 0.55
        sld = VGroup(guide_rails(gc, 3.0, 0.62), slider_block(gc))
        sl = ctext("移动副 P", size=25, color=LINK_D).next_to(sld, DOWN, buff=0.5)
        self.play(Create(sld), FadeIn(sl))
        self.play(sld[1].animate.shift(RIGHT * 0.8),
                  sld[1].animate.shift(LEFT * 0.8), run_time=1.4)
        low = ctext("低副：面接触 → 压强小、耐磨、承载大（R/P 各约束 s=2）",
                    size=26, color=GOOD)
        high = ctext("高副：点线接触 → 易实现复杂运动规律（约束 s=1）",
                     size=26, color=NOTE)
        clo = ctext("封闭方式：几何封闭（结构保证） vs 力封闭（弹簧/重力压紧）",
                    size=25, color=MUTED)
        stack = VGroup(low, high, clo).arrange(DOWN, buff=0.34).to_edge(
            DOWN, buff=0.3)
        self.play(Write(low))
        self.hold(2)
        # 右：高副 两廓线点接触
        a1 = Arc(radius=0.95, angle=PI * 0.55, color=NOTE,
                 stroke_width=5).shift(RIGHT * 3.9 + UP * 0.15)
        a2 = Arc(radius=0.8, angle=PI * 0.55, color=LINK_A,
                 stroke_width=5).rotate(PI).shift(RIGHT * 4.28 + UP * 1.32)
        touch = higher_pair_mark(RIGHT * 4.05 + UP * 0.92)
        hl = ctext("高副（点/线接触）", size=25, color=NOTE).next_to(a1, DOWN,
                                                                    buff=0.55)
        self.play(Create(a1), Create(a2), FadeIn(touch), FadeIn(hl))
        self.play(Write(high))
        self.hold(2)
        self.play(Write(clo))
        self.add(page_ref("孙桓八版 p6-7"))
        self.hold(3)


class S08_ChainToMechanism(LessonScene):
    """运动链→机构（~8min, p8-10）：闭链取一构件为机架+指定原动件。"""

    def construct(self):
        self.header("运动链 → 机构", "机架的诞生 · p8-10")
        fb = FourBar(4.0, 1.4, 3.4, 2.8, origin=np.array([-2.0, -1.0, 0]))
        th = ValueTracker(0.7)

        def chain():
            A0, A, B, B0 = fb.solve(th.get_value())
            return VGroup(link_line(A0, A, LINK_A), link_line(A, B, LINK_B),
                          link_line(B, B0, LINK_C), link_line(B0, A0, MUTED),
                          *[pin_joint(p) for p in (A0, A, B, B0)])
        mech = always_redraw(chain)
        t1 = ctext("四个构件 + 四个转动副 = 闭式运动链（还不在'工作'）",
                   size=27).to_edge(DOWN, buff=1.5)
        self.play(FadeIn(mech), Write(t1))
        self.play(th.animate.set_value(0.7 + 0.6), run_time=1.5)
        self.hold(1.5)
        A0, _, _, B0 = fb.solve(th.get_value())
        gnd = VGroup(ground_hatch(A0 + DOWN * 0.16, 0.7),
                     ground_hatch(B0 + DOWN * 0.16, 0.7))
        t2 = ctext("固定一构件为机架 + 指定原动件 → 机构！", size=28,
                   color=ACCENT).to_edge(DOWN, buff=0.75)
        self.play(FadeIn(gnd), Write(t2))
        self.play(th.animate.set_value(th.get_value() + TAU), run_time=6,
                  rate_func=linear)
        t3 = ctext("同一链固定不同构件 = 不同机构（第 7 讲展开）", size=24,
                   color=MUTED).to_edge(DOWN, buff=0.18)
        self.play(Write(t3))
        self.add(page_ref("孙桓八版 p8-10"))
        self.hold(3)


class S09_Schematic(LessonScene):
    """机构运动简图（~16min, p11-14 压轴）：符号表→绘制步骤→偏心轮抽象成
    曲柄滑块（实物渐隐、简图浮现、两者同步运动验证等效）。"""

    def construct(self):
        self.header("机构运动简图", "把一台机器画成几根线 · p11-14")
        sym = VGroup(
            VGroup(pin_joint(ORIGIN), ctext("转动副", size=23)),
            VGroup(slider_block(ORIGIN, 0.55, 0.32), ctext("移动副", size=23)),
            VGroup(ground_hatch(ORIGIN, 0.7), ctext("机架", size=23)),
            VGroup(link_line(LEFT * 0.4, RIGHT * 0.4), ctext("构件", size=23)),
        )
        for row in sym:
            row.arrange(RIGHT, buff=0.5)
        sym.arrange(DOWN, aligned_edge=LEFT, buff=0.42).shift(LEFT * 5.0
                                                            + DOWN * 0.5)
        cap = ctext("常用符号（只保留运动要素）", size=24,
                    color=MUTED).move_to(P(-4.6, 1.95))
        self.play(FadeIn(sym, lag_ratio=0.3), FadeIn(cap), run_time=2.2)
        self.hold(2)
        # 偏心轮实物 → 等效曲柄滑块
        oc = np.array([2.6, 0.1, 0])
        e = 0.42
        disk = Circle(radius=1.05, color=LINK_A, stroke_width=5,
                      fill_opacity=0.12, fill_color=LINK_A).move_to(oc + RIGHT * e)
        shaft = Dot(oc, radius=0.07)
        t1 = ctext("案例：偏心轮传动", size=24, color=LINK_A).next_to(
            disk, UP, buff=0.35)
        self.play(Create(disk), FadeIn(shaft), Write(t1))
        self.play(Rotate(disk, TAU, about_point=oc), run_time=2.5,
                  rate_func=linear)
        steps = bullets([
            "① 数构件：偏心轮·连杆·滑块·机架",
            "② 定运动副：3 转动副 + 1 移动副",
            "③ 选比例尺 μ，按符号作图",
        ], size=24).to_edge(DOWN, buff=0.35).shift(LEFT * 2.6)
        self.play(FadeIn(steps, lag_ratio=0.4))
        self.hold(2)
        cs = AnimatedCrankSlider(CrankSlider(e, 2.0, 0.0, origin=oc),
                                 cylinder=False)
        t2 = ctext("偏心轮 ≡ 曲柄长=偏心距 e 的曲柄滑块", size=25,
                   color=ACCENT).next_to(disk, DOWN, buff=0.5)
        self.play(disk.animate.set_stroke(opacity=0.15).set_fill(opacity=0.03),
                  FadeIn(cs.group), Write(t2))
        self.play(cs.theta.animate.set_value(TAU),
                  Rotate(disk, TAU, about_point=oc), run_time=5,
                  rate_func=linear)
        self.add(page_ref("孙桓八版 p11-14"))
        self.hold(3)


class S10_Summary(LessonScene):
    """小结+下讲悬念（~4min）：五个关键词回收；抛出自由度问题。"""

    def construct(self):
        self.header("第 1 讲小结")
        pts = bullets([
            "机器三特征；机构只管运动；机械是总称      (p1-2)",
            "构件=运动单元 ≠ 零件=制造单元             (p5)",
            "运动副：f=6−s；低副面接触/高副点线接触    (p6-7)",
            "运动链 + 机架 + 原动件 = 机构             (p8-10)",
            "运动简图：只留运动要素的机构'X 光片'      (p11-14)",
        ], size=27)
        self.play(FadeIn(pts, lag_ratio=0.4), run_time=2.8)
        self.hold(3)
        q = ctext("下一讲：它到底能不能动？要几个电机？——自由度",
                  size=25, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
