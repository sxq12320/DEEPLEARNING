# -*- coding: utf-8 -*-
"""L14 让机器成为系统——机械传动系统方案设计 + 全课收官（第14章, p296-316；
第13章机器人机构* 简介）"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from manim import *  # noqa: E402,F403
from mechlib import *  # noqa: E402,F403


class S01_DesignFlow(LessonScene):
    """设计总流程（~8min, p296-300）：功能→原理→机构选型→运动协调（循环图）
    →尺度综合——流程图动画。"""

    def construct(self):
        self.header("从需求到机器", "传动系统设计流程 · p296-300")
        steps = ["功能分析", "工作原理选择", "机构选型与组合", "运动协调设计",
                 "尺度综合"]
        boxes = VGroup(*[
            VGroup(RoundedRectangle(corner_radius=0.14, width=4.6, height=0.85,
                                    color=LINK_B, fill_opacity=0.12,
                                    fill_color=LINK_B),
                   ) for s in steps])
        for i, (b, s) in enumerate(zip(boxes, steps)):
            b.add(ctext(f"{'①②③④⑤'[i]} {s}", size=26).move_to(b[0]))
        boxes.arrange(DOWN, buff=0.3).shift(LEFT * 3.2 + DOWN * 0.6)
        arrows = VGroup(*[vec(boxes[i][0].get_bottom(), DOWN * 0.34, ACCENT,
                              tip=0.12) for i in range(4)])
        self.play(FadeIn(boxes, lag_ratio=0.3), FadeIn(arrows), run_time=3)
        note = ctext("运动循环图：各执行机构的'节拍表'\n相位对齐，互不打架",
                     size=23, color=NOTE).shift(RIGHT * 3.0 + DOWN * 0.3)
        self.play(Write(note))
        self.add(page_ref("孙桓八版 p296-300"))
        self.hold(3)


class S02_TransmissionChoice(LessonScene):
    """原动机与传动方案比较（~8min, p300-308）：带/链/齿轮/连杆特性对照表——
    减速比/效率/成本/距离；方案比选决策动画。"""

    def construct(self):
        self.header("传动力选谁？", "方案比较 · p300-308")
        rows = [
            ("带传动", "远距·缓冲·会打滑·i≤7"),
            ("链传动", "远距·不打滑·有噪声"),
            ("齿轮", "紧凑·高效·i≤8/级·要润滑"),
            ("连杆", "低副承载大·难做定传动比"),
            ("蜗杆", "单级大减速·可自锁·效率低"),
        ]
        tab = VGroup(*[bullets([f"{n}：{d}"], size=25)
                       for n, d in rows])
        tab.arrange(DOWN, aligned_edge=LEFT, buff=0.38).shift(UP * 0.5)
        self.play(FadeIn(tab, lag_ratio=0.35), run_time=2.8)
        self.hold(2.5)
        rule = ctext("选型口诀：看功率、看距离、看精度、看成本——没有最好，"
                     "只有最合适", size=26, color=ACCENT).to_edge(DOWN,
                                                                   buff=0.7)
        self.play(Write(rule))
        self.add(page_ref("孙桓八版 p300-308"))
        self.hold(3)


class S03_Combination(LessonScene):
    """机构组合与创新（~6min, p308-313）：串联/并联/复合——多级减速示意 +
    机构创新四法（演化/倒置/组合/变异）。"""

    def construct(self):
        self.header("机构的排列组合", "组合与创新 · p308-313")
        # 串联示意：电机→带→齿轮→执行
        chain = VGroup(*[
            VGroup(RoundedRectangle(corner_radius=0.12, width=2.3, height=0.9,
                                    color=LINK_B, fill_opacity=0.12,
                                    fill_color=LINK_B),
                   ) for s in ["电动机", "带传动", "齿轮减速", "执行机构"]])
        labs = ["电动机", "带传动", "齿轮减速", "执行机构"]
        for b, s in zip(chain, labs):
            b.add(ctext(s, size=23).move_to(b[0]))
        chain.arrange(RIGHT, buff=0.7).shift(UP * 0.9)
        arrows = VGroup(*[vec(chain[i][0].get_right(), RIGHT * 0.7, INK,
                              tip=0.12) for i in range(3)])
        self.play(FadeIn(chain, lag_ratio=0.3), FadeIn(arrows))
        pts = bullets([
            "串联：前级输出=后级输入（传动比连乘）",
            "并联/复合：运动合成与分解（差速器就是）",
            "创新四法：演化 · 倒置 · 组合 · 变异",
        ], size=26).to_edge(DOWN, buff=0.6)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.add(page_ref("孙桓八版 p308-313"))
        self.hold(3)


class S04_Robotics(LessonScene):
    """机器人机构学 3 分钟（选学, 第13章）：开链机械臂（串联）vs 并联平台；
    自由度=电机数；工作空间示意。"""

    def construct(self):
        self.header("机器人机构学一瞥", "开链 vs 闭链 · 第13章*")
        # 串联臂：两杆机械臂
        O = P(-4.2, -1.2)
        a1, a2 = ValueTracker(0.7), ValueTracker(0.9)

        def arm():
            e1 = O + P(2.0 * np.cos(a1.get_value()),
                       2.0 * np.sin(a1.get_value()))
            e2 = e1 + P(1.6 * np.cos(a1.get_value() + a2.get_value()),
                        1.6 * np.sin(a1.get_value() + a2.get_value()))
            return VGroup(fixed_pin(O), link_line(O, e1, LINK_B),
                          link_line(e1, e2, LINK_C), pin_joint(e1),
                          Dot(e2, radius=0.09, color=LINK_D))
        m = always_redraw(arm)
        self.play(FadeIn(m))
        self.play(a1.animate.set_value(1.4), a2.animate.set_value(1.2),
                  run_time=2.5)
        self.play(a1.animate.set_value(0.4), a2.animate.set_value(-0.5),
                  run_time=2.5)
        pts = bullets([
            "开链（串联）臂：关节即自由度、空间大刚度低",
            "闭链（并联）台：多支链共托、刚度大空间小",
            "学到这里，你已经能看懂它们的自由度账",
        ], size=25).shift(RIGHT * 2.6 + DOWN * 0.3)
        self.play(FadeIn(pts, lag_ratio=0.4))
        self.hold(3)


class S05_EngineReunion(LessonScene):
    """收官（~6min）：L01 拆开的内燃机重新装回去——曲柄滑块+齿轮+凸轮+飞轮
    同屏齐转：你学到的每一章都是这台机器的一个零件。"""

    def construct(self):
        self.header("把内燃机装回去", "全课回收")
        cs = AnimatedCrankSlider(CrankSlider(0.75, 2.2, 0.0,
                                             origin=np.array([-4.8, -1.5, 0])))
        gp = AnimatedGearPair(12, 24, m=0.07, c1=np.array([-0.7, -2.0, 0]))
        cam = AnimatedCamKnife(0.5, cam_law("cos", 0.4, PI * 1.1),
                               origin=np.array([3.0, -0.6, 0]))
        fw = flywheel(P(-4.8, -1.5), r=1.2)
        lb = VGroup(
            ctext("曲柄滑块 §8", size=22, color=LINK_A).move_to(
                P(-2.9, -2.55)),
            ctext("齿轮 §10-11", size=22, color=GEAR_1).next_to(gp.group,
                                                               LEFT,
                                                               buff=0.4),
            ctext("凸轮 §9", size=22, color=LINK_D).next_to(cam.group, LEFT,
                                                            buff=0.4),
            ctext("飞轮 §7", size=22, color=NOTE).next_to(fw, UP, buff=0.35),
        )
        self.play(FadeIn(cs.group), FadeIn(gp.group), FadeIn(cam.group),
                  FadeIn(fw), FadeIn(lb))
        if getattr(self.camera.frame, "savedstate", None) is None:
            self.camera.frame.save_state()
        self._zoomed = True
        self.play(cs.theta.animate.set_value(3 * TAU),
                  gp.theta.animate.set_value(3 * TAU),
                  cam.theta.animate.set_value(6 * TAU),
                  Rotate(fw, 3 * TAU, about_point=P(-4.8, -1.5)),
                  self.camera.frame.animate.scale(0.82).move_to(
                      P(-0.9, -0.9)),
                  run_time=9, rate_func=linear)
        self.hold(1)
        self.takeaway("一台机器 = 结构(2)·运动(3)·力(4-7)·常用机构(8-13)·"
                      "方案(14) 的合体", p="全书")
        self.hold(3)


class S06_Closing(LessonScene):
    """结语（~3min）：课程地图回顾 + 往后学什么（机械设计/动力学/机器人）。"""

    def construct(self):
        big = ctext("机械原理 · 全课终", size=60, weight="BOLD").shift(
            UP * 0.8)
        self.play(Write(big))
        self.hold(1.5)
        nxt = bullets([
            "向前：机械设计（把机构落成零件）",
            "向深：机械动力学 / 振动 / 机器人学",
            "向广：机构创新设计——下一个精巧机构等你发明",
        ], size=28).shift(DOWN * 1.4)
        self.play(FadeIn(nxt, lag_ratio=0.4))
        self.hold(3)
        thx = ctext("谢谢观看——愿你对每台机器都多问一句：它怎么动？",
                    size=26, color=ACCENT).to_edge(DOWN, buff=0.8)
        self.play(Write(thx))
        self.hold(3)
