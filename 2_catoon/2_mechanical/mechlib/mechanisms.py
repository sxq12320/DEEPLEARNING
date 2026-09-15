# -*- coding: utf-8 -*-
"""mechlib.mechanisms — 即取即演的机构动画组件.

统一约定：每个组件暴露 .theta (ValueTracker) 与 .group (always_redraw VGroup)；
场景里 `self.play(mech.theta.animate.set_value(TAU), run_time=6, rate_func=linear)`
即可驱动整机构连续运转。
"""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, PI, RIGHT, TAU, UP, Circle,
    DashedVMobject, Dot, Line, Rectangle, TracedPath, VGroup, ValueTracker,
    always_redraw,
)

from .curves import cam_profile_knife, cam_profile_roller, gear_pair, gear_profile
from .primitives import (
    P, fixed_pin, ground_hatch, guide_rails, link_line, pin_joint, saw_wheel,
    slider_block,
)
from .solvers import CrankSlider, FourBar, Whitworth, geneva_state
from .style import (
    BG, FRAME_C, GEAR_1, GEAR_2, INK, LINK_A, LINK_B, LINK_C, LINK_D, MUTED,
)


# ---------------------------------------------------------------- 四杆机构
class AnimatedFourBar:
    """铰链四杆动画组件。

    fb: FourBar；theta 为曲柄角。
    coupler=(s,h) 时在连杆上画描点；trace=True 追加轨迹 TracedPath。
    """

    def __init__(self, fb: FourBar, coupler=None, trace: bool = False,
                 show_ground: bool = True, joint_r: float = 0.08):
        self.fb = fb
        self.coupler = coupler
        self.theta = ValueTracker(0.6)
        self.show_ground = show_ground
        self.joint_r = joint_r
        self.group = always_redraw(self._build)
        self.trace_path = None
        if coupler is not None and trace:
            self.trace_path = TracedPath(
                self._cpt, stroke_color=LINK_D, stroke_width=3)
            self.group = VGroup(self.trace_path, self.group)

    def _cpt(self):
        s, h = self.coupler
        return self.fb.coupler_point(self.theta.get_value(), s, h)

    def _build(self):
        A0, A, B, B0 = self.fb.solve(self.theta.get_value())
        parts = [
            link_line(A0, A, LINK_A),
            link_line(A, B, LINK_B),
            link_line(B, B0, LINK_C),
        ]
        if self.show_ground:
            parts += [link_line(A0, B0, FRAME_C, 5),
                      ground_hatch(A0 + DOWN * 0.16, 0.6),
                      ground_hatch(B0 + DOWN * 0.16, 0.6)]
        parts += [pin_joint(p, self.joint_r) for p in (A0, A, B, B0)]
        if self.coupler is not None:
            parts.append(Dot(self._cpt(), radius=0.07, color=LINK_D))
        return VGroup(*parts)


# ---------------------------------------------------------------- 曲柄滑块
class AnimatedCrankSlider:
    """曲柄滑块 + 气缸壁。theta 曲柄角。"""

    def __init__(self, cs: CrankSlider, cylinder: bool = True,
                 piston_color=LINK_D, show_ground: bool = True):
        self.cs = cs
        self.theta = ValueTracker(0.0)
        self.cylinder = cylinder
        self.piston_color = piston_color
        self.show_ground = show_ground
        self.group = always_redraw(self._build)

    def _build(self):
        O, A, B = self.cs.solve(self.theta.get_value())
        parts = [link_line(O, A, LINK_A), link_line(A, B, LINK_B),
                 slider_block(B, 0.72, 0.5, color=self.piston_color),
                 pin_joint(A), pin_joint(B)]
        if self.show_ground:
            parts.append(fixed_pin(O))
        if self.cylinder:
            parts += [
                Line(B + LEFT * 1.5 + UP * 0.33, B + RIGHT * 1.6 + UP * 0.33,
                     color=FRAME_C, stroke_width=5),
                Line(B + LEFT * 1.5 + DOWN * 0.33, B + RIGHT * 1.6 + DOWN * 0.33,
                     color=FRAME_C, stroke_width=5),
            ]
        return VGroup(*parts)


# ---------------------------------------------------------------- 摆动导杆急回
class AnimatedWhitworth:
    """曲柄摆动导杆机构（牛头刨）。theta 曲柄角；导杆摆角自动。"""

    def __init__(self, ww: Whitworth, ram: bool = True):
        self.ww = ww
        self.ram = ram
        self.theta = ValueTracker(PI / 2)
        self.group = always_redraw(self._build)

    def _build(self):
        O1, A, O2, phi = self.ww.solve(self.theta.get_value())
        L = self.ww.lever
        top = O2 + np.array([L * np.cos(phi), L * np.sin(phi), 0.0])
        parts = [
            link_line(O1, A, LINK_A),                    # 曲柄
            link_line(O2, top, LINK_C),                  # 摆动导杆
            Dot(A, radius=0.1, color=LINK_D),            # 滑块销
            Circle(radius=self.ww.r, color=FRAME_C, stroke_width=1.5,
                   stroke_opacity=0.5).move_to(O1),      # 销轨迹圆
            fixed_pin(O1), fixed_pin(O2),
        ]
        if self.ram:
            # 刨刀滑枕：导杆顶端小铰链 + 水平滑块
            ram_y = top[1]
            ram = slider_block(top + UP * 0.12, 1.1, 0.42, color=LINK_D)
            parts += [
                guide_rails(top + UP * 0.12, length=2.6, gap=0.56, color=FRAME_C),
                pin_joint(top),
                ram,
            ]
        return VGroup(*parts)


# ---------------------------------------------------------------- 凸轮
class AnimatedCamKnife:
    """对心尖顶直动推杆盘形凸轮。theta=凸轮转角；推杆升程 s(θ)。"""

    def __init__(self, r0: float, s_func, scale: float = 1.0,
                 origin=ORIGIN, cam_color=LINK_D):
        self.r0, self.s_func, self.scale = r0, s_func, scale
        self.o = np.array(origin, dtype=float)
        self.theta = ValueTracker(0.0)
        self.cam_color = cam_color
        self.group = always_redraw(self._build)

    def follower_h(self):
        # 凸轮顺时针转 θ 后，位于正上方（接触角 π/2）的廓线点原相角为 π/2+θ
        return (self.r0 + self.s_func(self.theta.get_value() + PI / 2)
                ) * self.scale

    def _build(self):
        cam = cam_profile_knife(self.s_func, self.r0, scale=self.scale,
                                color=self.cam_color)
        cam.rotate(-self.theta.get_value(), about_point=ORIGIN).shift(self.o)
        h = self.follower_h()
        tip = self.o + UP * h
        stem_top = self.o + UP * (h + 1.5)
        guide = guide_rails(self.o + UP * (h + 1.15), length=1.1, gap=0.34,
                            angle=PI / 2)
        parts = [cam,
                 fixed_pin(self.o),
                 Line(tip, stem_top, color=LINK_B, stroke_width=7),
                 Dot(tip, radius=0.07, color=INK),
                 guide]
        return VGroup(*parts)


class AnimatedCamRoller:
    """滚子直动推杆盘形凸轮：实际廓线 + 滚子圆 + 推杆。"""

    def __init__(self, r0: float, rr: float, s_func, scale: float = 1.0,
                 origin=ORIGIN, cam_color=LINK_D, pitch_line: bool = False):
        self.r0, self.rr, self.s_func, self.scale = r0, rr, s_func, scale
        self.o = np.array(origin, dtype=float)
        self.theta = ValueTracker(0.0)
        self.cam_color = cam_color
        self.pitch_line = pitch_line
        self.group = always_redraw(self._build)

    def _pitch_r(self, d):
        return self.r0 + self.rr + self.s_func(d)

    def roller_center(self):
        """当前滚子中心世界坐标（接触角恒在正上方）。"""
        return self.o + UP * (self._pitch_r(self.theta.get_value() + PI / 2)
                              * self.scale)

    def _build(self):
        th = self.theta.get_value()
        cam = cam_profile_roller(self.s_func, self.r0, self.rr,
                                 scale=self.scale, color=self.cam_color)
        cam.rotate(-th, about_point=ORIGIN).shift(self.o)
        parts = [cam, fixed_pin(self.o)]
        if self.pitch_line:
            pitch = cam_profile_knife(lambda d: self.rr + self.s_func(d),
                                      self.r0, scale=self.scale,
                                      color=MUTED, stroke_width=1.5)
            pitch.rotate(-th, about_point=ORIGIN).shift(self.o)
            pitch = DashedVMobject(pitch, num_dashes=46)
            parts.append(pitch)
        rc = self.roller_center()
        roller = Circle(radius=self.rr * self.scale, color=LINK_B,
                        stroke_width=3, fill_opacity=0.25,
                        fill_color=LINK_B).move_to(rc)
        stem = Line(rc + UP * self.rr * self.scale,
                    rc + UP * (self.rr * self.scale + 1.4),
                    color=LINK_B, stroke_width=7)
        guide = guide_rails(rc + UP * 1.1, length=1.1, gap=0.34, angle=PI / 2)
        parts += [roller, stem, guide]
        return VGroup(*parts)


# ---------------------------------------------------------------- 齿轮对
class AnimatedGearPair:
    """一对标准外啮合齿轮。theta=轮1转角；轮2按 -z1/z2 反转。"""

    def __init__(self, z1: int, z2: int, m: float = 0.12, c1=ORIGIN):
        self.z1, self.z2, self.m = z1, z2, m
        self.c1 = np.array(c1, dtype=float)
        self.a = m * (z1 + z2) / 2
        self.c2 = self.c1 + np.array([self.a, 0.0, 0.0])
        self.theta = ValueTracker(0.0)
        self.phase2 = PI - PI / z2       # 齿槽对齿心相位
        self.group = always_redraw(self._build)

    def _build(self):
        g1 = gear_profile(self.m, self.z1, color=GEAR_1)
        g1.rotate(self.theta.get_value(), about_point=ORIGIN).shift(self.c1)
        g2 = gear_profile(self.m, self.z2, color=GEAR_2)
        g2.rotate(self.phase2 - self.theta.get_value() * self.z1 / self.z2,
                  about_point=ORIGIN).shift(self.c2)
        return VGroup(g1, g2, fixed_pin(self.c1), fixed_pin(self.c2))


# ---------------------------------------------------------------- 槽轮
class AnimatedGeneva:
    """外槽轮机构。theta=主动盘角；槽轮角自动（含锁止）。

    z 槽数；a 中心距；r_pin = a·sin(π/z)。
    """

    def __init__(self, z: int = 4, a: float = 2.4, c1=ORIGIN):
        self.z, self.a = z, a
        self.c1 = np.array(c1, dtype=float)
        self.c2 = self.c1 + np.array([a, 0.0, 0.0])
        self.r_pin = a * np.sin(PI / z)
        self.wheel_r = a * np.cos(PI / z)      # 槽轮外圆
        self.theta = ValueTracker(-PI / 4)
        self.group = always_redraw(self._build)

    def _build(self):
        th = self.theta.get_value()
        wang, engaged = geneva_state(th, self.z, self.a)
        # 主动盘 + 销
        driver = Circle(radius=self.r_pin, color=LINK_A, stroke_width=3,
                        fill_opacity=0.08, fill_color=LINK_A).move_to(self.c1)
        pin_pos = self.c1 + np.array([self.r_pin * np.cos(th),
                                      self.r_pin * np.sin(th), 0.0])
        pin = Dot(pin_pos, radius=0.09, color=LINK_D)
        lock = Circle(radius=self.r_pin * 0.55, color=LINK_A,
                      stroke_width=3).move_to(self.c1)
        # 槽轮：圆盘 + z 条径向槽（黑底描边读出“槽”形）
        wheel = VGroup(Circle(radius=self.wheel_r, color=LINK_C, stroke_width=3,
                              fill_opacity=0.45, fill_color=LINK_C))
        for i in range(self.z):
            ang = i * TAU / self.z
            slot = Rectangle(width=self.wheel_r * 0.72, height=0.2,
                             color=LINK_C, fill_opacity=1.0, fill_color=BG,
                             stroke_width=1.5)
            slot.move_to(np.array([np.cos(ang), np.sin(ang), 0.0])
                         * self.wheel_r * 0.64 + self.c2)
            slot.rotate(ang)
            wheel.add(slot)
        wheel.rotate(wang, about_point=self.c2)
        # 销弧（与锁止弧配合的圆弧段提示）
        parts = [driver, lock, pin, wheel,
                 fixed_pin(self.c1), fixed_pin(self.c2)]
        return VGroup(*parts)


# ---------------------------------------------------------------- 棘轮
def ratchet_assembly(point, r: float = 1.1, z: int = 12):
    """棘轮 + 棘爪（静件，场景自行驱动摆臂/轮）。

    返回 (wheel, pawl, pawl_pivot)。pawl 绕 pawl_pivot 摆动推轮。
    """
    wheel = saw_wheel(point, r, z, color=LINK_C)
    pv = np.asarray(point, dtype=float) + UP * (r * 1.35)
    tip = np.asarray(point, dtype=float) + np.array([r * 0.28, r * 0.98, 0.0])
    pawl = VGroup(link_line(pv, tip, LINK_D, 6), Dot(tip, radius=0.06, color=LINK_D))
    return wheel, pawl, pv


# ---------------------------------------------------------------- 行星轮系
class AnimatedPlanetary:
    """单排行星轮系示意动画（2K-H）。

    参数化：tracker tH(转臂角), t1(太阳轮角), t3(齿圈角)——由场景按
    Planetary.speeds() 算好关系后驱动。行星轮画简化圆+自转标记线。
    """

    def __init__(self, pl, m: float = 0.09, origin=ORIGIN, n_planets: int = 3):
        self.pl = pl
        self.m = m
        self.o = np.array(origin, dtype=float)
        self.np = n_planets
        self.r1 = m * pl.z1 / 2
        self.r2 = m * pl.z2 / 2
        self.r3 = m * pl.z3 / 2
        self.tH = ValueTracker(0.0)
        self.t1 = ValueTracker(0.0)
        self.t3 = ValueTracker(0.0)
        self.group = always_redraw(self._build)

    def _planet_center(self, i):
        a = self.tH.get_value() + i * TAU / self.np
        return self.o + (self.r1 + self.r2) * np.array([np.cos(a), np.sin(a), 0])

    def _build(self):
        sun = gear_profile(self.m, self.pl.z1, color=GEAR_1, stroke_width=2)
        sun.rotate(self.t1.get_value(), about_point=ORIGIN).shift(self.o)
        # 齿圈：外环 + 内齿示意短刻线
        ring = VGroup(
            Circle(radius=self.r3 + self.m * 2.2, color=GEAR_2, stroke_width=4),
            Circle(radius=self.r3 + self.m * 1.0, color=GEAR_2, stroke_width=2),
        ).shift(self.o)
        ticks = VGroup(*[
            Line(self.o + (self.r3 - 0.02) * np.array([np.cos(a), np.sin(a), 0]),
                 self.o + (self.r3 + self.m) * np.array([np.cos(a), np.sin(a), 0]),
                 color=GEAR_2, stroke_width=2)
            for a in np.linspace(0, TAU, self.pl.z3, endpoint=False)
        ])
        ticks.rotate(self.t3.get_value(), about_point=self.o)
        carrier = VGroup(*[
            link_line(self.o, self._planet_center(i), FRAME_C, 6)
            for i in range(self.np)
        ])
        planets = VGroup(*[
            gear_profile(self.m, self.pl.z2, color=LINK_D, stroke_width=2)
            .move_to(self._planet_center(i))
            for i in range(self.np)
        ])
        return VGroup(ring, ticks, carrier, sun, planets, fixed_pin(self.o))
