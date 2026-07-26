# -*- coding: utf-8 -*-
"""L01 什么是机械？——绪论、机构的组成与运动简图（孙桓八版 第1章 + §2-1~2-3, p1-14）

渲染示例:  manim -pqh scenes.py S01_Opening
批量渲染:  manim -qh scenes.py S01_Opening S02_EngineDissect ... （或用根目录 render_all.py）
每个 Scene 的 docstring = 分镜讲稿要点（含教材页码，供配音与校对）。
目标成片 85-95 min：每场景动画已留讲解空白（hold），实际时长由配音节奏拉伸。
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_Opening(LessonScene):
    """片头（约 2min 配音）：课程定位——机械专业的'内功心法'；本节课看点预告。
    讲稿要点：机械原理是什么(p1)；为什么值得学(p2-3)；本课路线：从一台内燃机说起。"""

    def construct(self):
        big = ctext("机械原理", size=96, weight="BOLD")
        sub = ctext("公益课 · 第 1 讲   什么是机械？", size=40, color=GREY_B)
        VGroup(big, sub).arrange(DOWN, buff=0.6)
        self.play(Write(big), run_time=1.5)
        self.play(FadeIn(sub, shift=UP * 0.3))
        self.hold(2)
        line = ctext("—— 一切机器，都是机构的组合 ——", size=32, color=ACCENT).shift(DOWN * 2.6)
        self.play(Write(line))
        self.hold(2.5)
        agenda = bullets([
            "① 机器 · 机构 · 机械：三个词到底啥关系？",
            "② 构件与运动副：机构的'原子'",
            "③ 运动简图：把一台机器画成几根线",
        ], size=30)
        self.play(FadeOut(big), FadeOut(sub), FadeOut(line))
        self.play(FadeIn(agenda, lag_ratio=0.3), run_time=2)
        self.hold(3)
        self.play(FadeOut(agenda))


class S02_EngineDissect(LessonScene):
    """从内燃机说起（约 10min）：整机 → 拆出三个机构（教材图 1-1, p1）。
    讲稿要点：内燃机=曲柄滑块(活塞-连杆-曲轴)+齿轮机构(正时齿轮)+凸轮机构(配气)；
    引出'机器是机构的组合'(p2)。本机构将贯穿全课程，L11 结尾回收。"""

    def construct(self):
        h = self.header("一台单缸内燃机", "孙桓《机械原理》图 1-1 · p1")
        # --- 曲柄滑块（活塞-连杆-曲轴）
        cs = CrankSlider(0.9, 2.6, 0.0, origin=np.array([-3.5, -1.2, 0]))
        theta = ValueTracker(0.0)

        def build_cs():
            O, A, B = cs.solve(theta.get_value())
            cyl = VGroup(  # 气缸壁
                Line(B + LEFT * 1.2 + UP * 0.42, B + RIGHT * 1.6 + UP * 0.42, color=GREY_B, stroke_width=5),
                Line(B + LEFT * 1.2 + DOWN * 0.42, B + RIGHT * 1.6 + DOWN * 0.42, color=GREY_B, stroke_width=5),
            )
            piston = Rectangle(width=0.8, height=0.76, color=ORANGE, fill_opacity=0.35,
                               fill_color=ORANGE).move_to(B)
            rod = link_line(A, B, color=BLUE_B)
            crank = link_line(O, A, color=GOLD)
            return VGroup(cyl, rod, crank, piston, pin_joint(A), pin_joint(B),
                          fixed_pin(O))

        mech = always_redraw(build_cs)
        lbl1 = ctext("曲柄滑块机构：活塞 · 连杆 · 曲轴", size=28, color=GOLD).to_edge(DOWN, buff=1.4)
        self.play(FadeIn(mech), FadeIn(lbl1))
        self.play(theta.animate.set_value(2 * TAU), run_time=6, rate_func=linear)
        self.hold(1)
        # --- 齿轮机构（正时齿轮 1:2）
        g_small = gear_profile(0.22, 12, scale=1.0, color=TEAL).shift(RIGHT * 2.2 + DOWN * 1.8)
        g_big = gear_profile(0.22, 24, scale=1.0, color=BLUE_B).shift(RIGHT * 2.2 + DOWN * 1.8 + RIGHT * 4.05)
        lbl2 = ctext("齿轮机构：曲轴 → 凸轮轴，转速减半（正时）", size=28, color=TEAL).to_edge(DOWN, buff=0.7)
        self.play(Create(g_small), Create(g_big), FadeIn(lbl2), run_time=2)
        self.play(Rotate(g_small, TAU, about_point=g_small.get_center()),
                  Rotate(g_big, -TAU / 2, about_point=g_big.get_center()),
                  run_time=4, rate_func=linear)
        self.hold(1)
        # --- 凸轮机构（配气）
        cam = cam_profile_knife(lambda d: 0.35 * (1 - np.cos(d)) if d < PI else 0.35 * (1 - np.cos(d)),
                                0.65, scale=1.0).shift(RIGHT * 5.2 + UP * 1.6)
        stem = Line(cam.get_center() + UP * 1.0, cam.get_center() + UP * 2.0, color=ORANGE, stroke_width=6)
        lbl3 = ctext("凸轮机构：按'剧本'开闭气门", size=28, color=ORANGE).next_to(cam, LEFT, buff=0.8)
        self.play(Create(cam), Create(stem), FadeIn(lbl3))
        self.play(Rotate(cam, TAU, about_point=cam.get_center()), run_time=4, rate_func=linear)
        self.hold(1)
        concl = ctext("机器 = 若干机构的组合（p2）", size=34, color=ACCENT, weight="BOLD").to_edge(DOWN, buff=0.15)
        self.play(Write(concl))
        self.add(page_ref("孙桓八版 p1-2"))
        self.hold(3)


class S03_Definitions(LessonScene):
    """机器·机构·机械（约 8min，p1-2）。
    讲稿要点：机器三特征（人为实体组合/确定相对运动/转换或传递能量物料信息）；
    机构只满足前两条；机械 = 机器 + 机构 的总称。举例：电动机/加工机械/计算机 vs 台虎钳。"""

    def construct(self):
        self.header("机器 · 机构 · 机械", "定义与三特征 · p1-2")
        feats = bullets([
            "① 人为的实体（构件）组合",
            "② 各部分之间具有确定的相对运动",
            "③ 能转换或传递  能量 · 物料 · 信息",
        ], size=32).shift(UP * 0.6)
        self.play(FadeIn(feats, lag_ratio=0.4), run_time=2.5)
        self.hold(2)
        box_machine = SurroundingRectangle(feats, color=GOOD, buff=0.35)
        t1 = ctext("三条全满足 → 机器 (machine)", size=30, color=GOOD).next_to(box_machine, DOWN, buff=0.35)
        self.play(Create(box_machine), Write(t1))
        self.hold(2)
        box_mech = SurroundingRectangle(VGroup(feats[0], feats[1]), color=TEAL, buff=0.3)
        t2 = ctext("只满足①② → 机构 (mechanism)", size=30, color=TEAL).next_to(t1, DOWN, buff=0.3)
        self.play(ReplacementTransform(box_machine.copy(), box_mech), Write(t2))
        self.hold(2)
        t3 = ctext("机械 (machinery) = 机器 与 机构 的总称", size=32, color=ACCENT).next_to(t2, DOWN, buff=0.45)
        self.play(Write(t3))
        self.add(page_ref("孙桓八版 p1-2"))
        self.hold(3)


class S04_CourseMap(LessonScene):
    """本课程研究什么（约 6min，p2）：五大板块地图——结构分析/运动分析/动力学/常用机构设计/方案设计；
    并预告 11 讲课程与板块的对应关系（观众的'全书地图'）。"""

    def construct(self):
        self.header("这门课研究什么？", "五大板块 · p2")
        items = ["① 机构的结构分析", "② 机构的运动分析", "③ 机器动力学",
                 "④ 常用机构分析与设计", "⑤ 机械系统方案设计"]
        cards = VGroup(*[
            VGroup(Rectangle(width=4.6, height=0.85, color=BLUE_B, fill_opacity=0.12,
                             fill_color=BLUE_B), ctext(s, size=28))
            for s in items
        ])
        for c in cards:
            c[1].move_to(c[0])
        cards.arrange(DOWN, buff=0.28).shift(LEFT * 3 + DOWN * 0.3)
        maps = ["→ 第 1-2 讲", "→ 第 3 讲", "→ 第 10-11 讲", "→ 第 4-9 讲", "→ 结课展望"]
        tags = VGroup(*[ctext(m, size=26, color=ACCENT).next_to(cards[i], RIGHT, buff=0.6)
                        for i, m in enumerate(maps)])
        for i in range(5):
            self.play(FadeIn(cards[i], shift=RIGHT * 0.4), run_time=0.7)
            self.play(Write(tags[i]), run_time=0.6)
            self.hold(0.8)
        self.add(page_ref("孙桓八版 p2"))
        self.hold(3)


class S05_LinkVsPart(LessonScene):
    """构件 vs 零件（约 8min，p5 + 图 2-1）。
    讲稿要点：零件=制造单元，构件=运动单元；连杆由连杆体/连杆头/螺栓螺母等多个零件
    刚性连接为一个构件；本课只关心刚性构件（脚注：弹性/挠性/气液构件）。"""

    def construct(self):
        self.header("构件 ≠ 零件", "运动单元 vs 制造单元 · p5")
        body = Polygon([-1.6, 0.5, 0], [1.6, 0.5, 0], [1.2, -0.5, 0], [-1.2, -0.5, 0],
                       color=BLUE_B, fill_opacity=0.2).shift(UP * 0.8)
        cap = Polygon([-1.2, -0.5, 0], [1.2, -0.5, 0], [1.0, -1.1, 0], [-1.0, -1.1, 0],
                      color=ORANGE, fill_opacity=0.2).shift(UP * 0.8)
        bolts = VGroup(Dot([-1.1, 0.3, 0], color=GREY_B, radius=0.08),
                       Dot([1.1, 0.3, 0], color=GREY_B, radius=0.08))
        labels = VGroup(
            ctext("连杆体（1 个零件）", size=24, color=BLUE_B).next_to(body, LEFT, buff=0.5),
            ctext("连杆头 + 轴瓦", size=24, color=ORANGE).next_to(cap, RIGHT, buff=0.5),
            ctext("螺栓 · 螺母 · 垫圈", size=24, color=GREY_B).next_to(bolts, UP, buff=0.4),
        )
        self.play(Create(body), Create(cap), FadeIn(bolts))
        self.play(FadeIn(labels, lag_ratio=0.4))
        self.hold(2.5)
        whole = SurroundingRectangle(VGroup(body, cap, bolts), color=GOOD, buff=0.3)
        t = ctext("刚性连接成整体运动 → 一个构件 (link)", size=30, color=GOOD).next_to(whole, DOWN, buff=0.5)
        self.play(Create(whole), Write(t))
        self.hold(2)
        concl = bullets([
            "零件 (part)：独立的  制造  单元",
            "构件 (link)：独立的  运动  单元",
            "机构分析只看构件——机器 = 若干构件的组合",
        ], size=28).to_edge(DOWN, buff=0.5)
        self.play(FadeIn(concl, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p5"))
        self.hold(3)


class S06_KinematicPair(LessonScene):
    """运动副与 f = 6 − s（约 15min，p6，本节第一处推导）。
    讲稿要点：直接接触+保持相对运动的可动连接=运动副；自由空间构件 6 个自由度
    （3 移动 x/y/z + 3 转动）；每加一个约束少一个自由度 → f = 6 − s；
    约束度 1~5 → Ⅰ~Ⅴ级副。推导动画：立方体 6 个自由度逐个点亮再逐个锁死。"""

    def construct(self):
        self.header("运动副：构件的'关节'", "自由度与约束 · p6")
        cube = Square(side_length=1.4, color=BLUE_B, fill_opacity=0.15).shift(LEFT * 4 + UP * 0.4)
        self.play(Create(cube))
        dof_labels = VGroup(*[ctext(s, size=24) for s in
                              ["移动 x", "移动 y", "移动 z", "转动 x", "转动 y", "转动 z"]])
        dof_labels.arrange_in_grid(rows=2, cols=3, buff=0.5).shift(RIGHT * 1.8 + UP * 1.7)
        # 演示三个平面内可见自由度 + 列出全部 6 个
        self.play(cube.animate.shift(RIGHT * 0.7), run_time=0.6)
        self.play(cube.animate.shift(LEFT * 0.7), run_time=0.6)
        self.play(cube.animate.shift(UP * 0.5), run_time=0.6)
        self.play(cube.animate.shift(DOWN * 0.5), run_time=0.6)
        self.play(Rotate(cube, PI / 6), Rotate(cube, -PI / 6), run_time=1.2)
        self.play(FadeIn(dof_labels, lag_ratio=0.2), run_time=2)
        t6 = ctext("自由的空间构件：6 个自由度", size=30, color=ACCENT).next_to(dof_labels, DOWN, buff=0.5)
        self.play(Write(t6))
        self.hold(2)
        # f = 6 - s 分步
        steps = [
            MathTex(r"\text{约束度 } s:\ \text{运动副夺走的自由度数}", font_size=40),
            MathTex(r"f \;=\; 6 - s", font_size=56, color=YELLOW),
            MathTex(r"s=1,\dots,5\ \Rightarrow\ \text{I}\sim\text{V 级副}", font_size=40),
        ]
        formula_reveal(self, steps, anchor=DOWN * 1.6, wait=1.6)
        self.add(page_ref("孙桓八版 p6"))
        self.hold(2.5)


class S07_PairTypes(LessonScene):
    """运动副分类（约 10min，p6-7）。
    讲稿要点：按接触——低副(面接触：转动副/移动副) vs 高副(点线接触：齿廓/凸轮)；
    按封闭——几何封闭 vs 力封闭（弹簧/重力压紧）。低副耐磨承载大、高副能实现复杂运动。"""

    def construct(self):
        self.header("运动副的分类", "低副 · 高副 · 封闭方式 · p6-7")
        # 低副：转动副 + 移动副
        rev = VGroup(fixed_pin(LEFT * 4.6 + UP * 0.6), link_line(LEFT * 4.6 + UP * 0.6, LEFT * 3.0 + UP * 1.4))
        rev_l = ctext("转动副 R（面接触）", size=26, color=BLUE_B).next_to(rev, DOWN, buff=0.5)
        sld = VGroup(
            Line(LEFT * 1.4 + UP * 0.4, RIGHT * 1.4 + UP * 0.4, color=GREY_B, stroke_width=5),
            Line(LEFT * 1.4 + UP * 1.2, RIGHT * 1.4 + UP * 1.2, color=GREY_B, stroke_width=5),
            slider_block(UP * 0.8),
        )
        sld_l = ctext("移动副 P（面接触）", size=26, color=ORANGE).next_to(sld, DOWN, buff=0.5)
        self.play(Create(rev), FadeIn(rev_l))
        self.play(Rotate(rev[1], PI / 5, about_point=rev[1].get_start()),
                  Rotate(rev[1], -PI / 5, about_point=rev[1].get_start()), run_time=1.6)
        self.play(Create(sld), FadeIn(sld_l))
        self.play(sld[2].animate.shift(RIGHT * 0.8), sld[2].animate.shift(LEFT * 0.8), run_time=1.6)
        low_tag = ctext("低副：面接触 → 压强小、耐磨、承载大", size=28, color=GOOD).to_edge(DOWN, buff=1.5)
        self.play(Write(low_tag))
        self.hold(2)
        # 高副：两齿廓 / 凸轮
        arc1 = Arc(radius=1.0, angle=PI / 2, color=TEAL, stroke_width=5).shift(RIGHT * 3.6 + UP * 0.2)
        arc2 = Arc(radius=0.8, angle=PI / 2, color=GOLD, stroke_width=5).rotate(PI).shift(RIGHT * 4.35 + UP * 1.35)
        touch = Dot(RIGHT * 3.95 + UP * 0.9, color=RED, radius=0.07)
        high_l = ctext("高副（点/线接触）", size=26, color=TEAL).next_to(arc1, DOWN, buff=0.6)
        self.play(Create(arc1), Create(arc2), FadeIn(touch), FadeIn(high_l))
        high_tag = ctext("高副：点线接触 → 能实现复杂运动规律", size=28, color=TEAL).to_edge(DOWN, buff=0.8)
        self.play(Write(high_tag))
        self.hold(2)
        close_tag = ctext("封闭：几何封闭（结构保证） vs 力封闭（弹簧/重力压紧）", size=28,
                          color=GREY_B).to_edge(DOWN, buff=0.15)
        self.play(Write(close_tag))
        self.add(page_ref("孙桓八版 p6-7"))
        self.hold(3)


class S08_ChainToMechanism(LessonScene):
    """运动链 → 机构（约 8min，p8-10）。
    讲稿要点：构件经运动副连接成运动链（闭式/开式）；取一构件为机架、
    一个或几个构件为原动件 → 机构。动画：同一四杆闭链，固定不同构件的瞬间'变成'不同机构（为 L04 机架变换埋伏笔）。"""

    def construct(self):
        self.header("运动链 → 机构", "机架的诞生 · p8-10")
        fb = FourBar(4.0, 1.4, 3.4, 2.8, origin=np.array([-2, -1.1, 0]))
        th = ValueTracker(0.7)

        def chain():
            A0, A, B, B0 = fb.solve(th.get_value())
            return VGroup(link_line(A0, A, GOLD), link_line(A, B, BLUE_B),
                          link_line(B, B0, TEAL), link_line(B0, A0, GREY_B),
                          *[pin_joint(p) for p in (A0, A, B, B0)])

        mech = always_redraw(chain)
        t1 = ctext("四个构件 + 四个转动副 = 闭式运动链", size=28).to_edge(DOWN, buff=1.4)
        self.play(FadeIn(mech), Write(t1))
        self.hold(2)
        A0, _, _, B0 = fb.solve(th.get_value())
        gnd = VGroup(ground_hatch(A0 + DOWN * 0.15, 0.7), ground_hatch(B0 + DOWN * 0.15, 0.7))
        t2 = ctext("固定一个构件作机架 + 指定原动件 → 机构！", size=28, color=ACCENT).to_edge(DOWN, buff=0.7)
        self.play(FadeIn(gnd), Write(t2))
        self.play(th.animate.set_value(0.7 + TAU), run_time=6, rate_func=linear)
        t3 = ctext("固定不同的构件 → 不同的机构（第 4 讲揭晓）", size=26, color=GREY_B).to_edge(DOWN, buff=0.1)
        self.play(Write(t3))
        self.add(page_ref("孙桓八版 p8-10"))
        self.hold(3)


class S09_Schematic(LessonScene):
    """机构运动简图（约 18min，p11-14，本节压轴）。
    讲稿要点：为什么要简图（只留与运动有关的要素：运动副类型/相对位置/尺寸）；
    常用符号表(p11)；绘制步骤：定构件数→定运动副类型位置→选比例尺→按符号绘制。
    压轴动画：偏心轮传动实物轮廓渐隐 → 等效曲柄滑块简图浮现（'偏心轮=放大的曲柄'）。"""

    def construct(self):
        self.header("机构运动简图", "把一台机器画成几根线 · p11-14")
        sym_rows = VGroup(
            VGroup(pin_joint(ORIGIN), ctext("转动副", size=24)).arrange(RIGHT, buff=0.5),
            VGroup(slider_block(ORIGIN, 0.55, 0.32), ctext("移动副", size=24)).arrange(RIGHT, buff=0.5),
            VGroup(ground_hatch(ORIGIN, 0.7), ctext("机架", size=24)).arrange(RIGHT, buff=0.5),
            VGroup(link_line(LEFT * 0.4, RIGHT * 0.4), ctext("构件", size=24)).arrange(RIGHT, buff=0.5),
        ).arrange(DOWN, aligned_edge=LEFT, buff=0.4).shift(LEFT * 4.4 + DOWN * 0.4)
        self.play(FadeIn(sym_rows, lag_ratio=0.3), run_time=2.5)
        self.hold(2)
        # 偏心轮 → 等效曲柄滑块
        ecc_center = np.array([2.2, 0.6, 0])
        e = 0.45  # 偏心距 = 等效曲柄长
        disk = Circle(radius=1.1, color=GOLD, stroke_width=5).move_to(ecc_center + RIGHT * e)
        shaft = Dot(ecc_center, color=WHITE, radius=0.07)
        t1 = ctext("案例：偏心轮传动（实物）", size=26, color=GOLD).next_to(disk, UP, buff=0.6)
        self.play(Create(disk), FadeIn(shaft), Write(t1))
        self.play(Rotate(disk, TAU, about_point=ecc_center), run_time=3, rate_func=linear)
        self.hold(1)
        steps = bullets([
            "① 数构件：偏心轮 · 连杆 · 滑块 · 机架",
            "② 找运动副：3 个转动副 + 1 个移动副",
            "③ 选比例尺 μl，按符号作图",
        ], size=26).to_edge(DOWN, buff=0.35).shift(LEFT * 2.5)
        self.play(FadeIn(steps, lag_ratio=0.4))
        self.hold(2)
        cs = CrankSlider(e, 2.0, 0.0, origin=ecc_center)
        th = ValueTracker(0.0)

        def simple():
            O, A, B = cs.solve(th.get_value())
            return VGroup(link_line(O, A, GOLD), link_line(A, B, BLUE_B),
                          slider_block(B, 0.6, 0.36), pin_joint(A), fixed_pin(O))

        sk = always_redraw(simple)
        t2 = ctext("简图：偏心轮 ≡ 曲柄长 = 偏心距 e 的曲柄滑块！", size=26, color=ACCENT).next_to(t1, DOWN, buff=0.2)
        self.play(disk.animate.set_stroke(opacity=0.18), FadeIn(sk), Write(t2))
        self.play(th.animate.set_value(TAU), Rotate(disk, TAU, about_point=ecc_center),
                  run_time=5, rate_func=linear)
        self.add(page_ref("孙桓八版 p11-14"))
        self.hold(3)


class S10_Summary(LessonScene):
    """小结 + 下讲预告（约 4min）。
    讲稿要点：本讲五个关键词；抛出下讲问题——'画好简图后，怎么判断它能不能动、要几个马达？'"""

    def construct(self):
        self.header("第 1 讲小结")
        pts = bullets([
            "机器三特征；机构只管运动；机械是总称        (p1-2)",
            "构件 = 运动单元 ≠ 零件 = 制造单元           (p5)",
            "运动副：f = 6 − s；低副面接触 / 高副点线接触  (p6-7)",
            "运动链 + 机架 + 原动件 = 机构               (p8-10)",
            "运动简图：只留运动要素的'机构 X 光片'        (p11-14)",
        ], size=28)
        self.play(FadeIn(pts, lag_ratio=0.35), run_time=3)
        self.hold(3)
        q = ctext("下一讲：这台机构，到底能不能动？——自由度", size=32, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(q))
        self.hold(3)
