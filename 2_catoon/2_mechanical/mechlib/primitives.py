# -*- coding: utf-8 -*-
"""mechlib.primitives — 机构制图元件：机架/转动副/移动副/滑块/弹簧/箭头/尺寸标注/带轮."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, GREY_B, LEFT, ORIGIN, PI, RIGHT, TAU, UP, Arc, Arrow, Circle,
    DashedLine, Dot, DoubleArrow, Line, Polygon, Rectangle, VGroup, VMobject,
    angle_of_vector,
)

from .style import FRAME_C, INK, LINK_D, MUTED


def P(x, y):
    """快捷三维点。"""
    return np.array([float(x), float(y), 0.0])


# ---------------------------------------------------------------- 机架与运动副
def ground_hatch(point=ORIGIN, width: float = 0.9, angle: float = 0.0,
                 n: int = 6, color=FRAME_C) -> VGroup:
    """机架符号：基准线 + 斜排短影线。"""
    base = Line(LEFT * width / 2, RIGHT * width / 2, color=color, stroke_width=3)
    hatches = VGroup(*[
        Line(ORIGIN, DOWN * 0.2 + LEFT * 0.11, color=color, stroke_width=2)
        .move_to(base.point_from_proportion(i / (n - 1)), aligned_edge=UP)
        for i in range(n)
    ])
    g = VGroup(base, hatches).rotate(angle, about_point=ORIGIN)
    return g.shift(point)


def pin_joint(point, r: float = 0.09, color=INK) -> VGroup:
    """转动副符号：空心圆 + 中心点。"""
    return VGroup(
        Circle(radius=r, color=color, stroke_width=3).move_to(point),
        Dot(point, radius=0.03, color=color),
    )


def fixed_pin(point, r: float = 0.1, color=INK, hatch_color=FRAME_C) -> VGroup:
    """固定铰链：铰链 + 下方机架剖线。"""
    return VGroup(pin_joint(point, r, color),
                  ground_hatch(point + DOWN * (r + 0.05), width=0.62, color=hatch_color))


def link_line(p1, p2, color=LINK_D, w: float = 7) -> Line:
    """构件（杆）：粗线段。"""
    return Line(p1, p2, color=color, stroke_width=w)


def bone_link(p1, p2, color=LINK_D, w: float = 0.16) -> VGroup:
    """实体感构件：矩形杆身 + 两端圆头（比 link_line 更'机械'）。"""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    d = p2 - p1
    L = np.linalg.norm(d)
    ang = np.arctan2(d[1], d[0])
    body = Rectangle(width=L, height=w, color=color, fill_opacity=0.85,
                     fill_color=color, stroke_width=2)
    c1 = Circle(radius=w * 0.9, color=color, fill_opacity=0.85, fill_color=color,
                stroke_width=2)
    c2 = c1.copy()
    g = VGroup(body, c1.move_to(LEFT * L / 2), c2.move_to(RIGHT * L / 2))
    g.rotate(ang, about_point=ORIGIN).shift((p1 + p2) / 2)
    return g


def slider_block(point, wdt: float = 0.7, hgt: float = 0.4, angle: float = 0.0,
                 color=LINK_D) -> Rectangle:
    """移动副滑块。"""
    return Rectangle(width=wdt, height=hgt, color=color, stroke_width=4,
                     fill_opacity=0.3, fill_color=color).rotate(angle).move_to(point)


def guide_rails(center, length: float = 4.0, gap: float = 0.56, angle: float = 0.0,
                color=FRAME_C) -> VGroup:
    """滑块导路：两根平行线 + 两端机架短影。"""
    half = UP * gap / 2
    top = Line(LEFT * length / 2, RIGHT * length / 2, color=color, stroke_width=4).shift(half)
    bot = Line(LEFT * length / 2, RIGHT * length / 2, color=color, stroke_width=4).shift(-half)
    g = VGroup(top, bot).rotate(angle, about_point=ORIGIN).shift(center)
    return g


def higher_pair_mark(point, r: float = 0.07):
    """高副接触点强调点。"""
    return Dot(point, radius=r, color="#EF476F")


# ---------------------------------------------------------------- 弹簧 / 阻尼 / 动力件
def spring(start, end, coils: int = 8, amp: float = 0.16, color=INK,
           stroke_width: float = 2.5) -> VMobject:
    """螺旋弹簧示意（锯齿折线）。"""
    start, end = np.asarray(start, dtype=float), np.asarray(end, dtype=float)
    d = end - start
    L = np.linalg.norm(d)
    u = d / L
    nv = np.array([-u[1], u[0], 0.0])
    lead = 0.18 * L
    pts = [start, start + u * lead]
    seg = (L - 2 * lead) / coils
    for i in range(coils):
        t = start + u * (lead + seg * (i + 0.5))
        pts.append(t + nv * (amp if i % 2 == 0 else -amp))
    pts += [end - u * lead, end]
    vm = VMobject(color=color, stroke_width=stroke_width)
    vm.set_points_as_corners(pts)
    return vm


def motor(point, label: str = "M", size: float = 0.62, color=INK) -> VGroup:
    """电动机符号：方框 + M。"""
    from .style import ctext
    box = Rectangle(width=size, height=size, color=color, stroke_width=3,
                    fill_opacity=0.15, fill_color=color)
    t = ctext(label, size=int(size * 55), color=color)
    return VGroup(box, t).move_to(point)


def flywheel(point, r: float = 0.9, color=FRAME_C, spokes: int = 6) -> VGroup:
    """飞轮：外圈 + 轮辐 + 轮毂。"""
    rim = Circle(radius=r, color=color, stroke_width=6)
    hub = Circle(radius=r * 0.18, color=color, fill_opacity=0.6, fill_color=color)
    sp = VGroup(*[
        Line(ORIGIN, RIGHT * r * 0.82, color=color, stroke_width=3)
        .rotate(i * TAU / spokes, about_point=ORIGIN)
        for i in range(spokes)
    ])
    return VGroup(rim, hub, sp).move_to(point)


# ---------------------------------------------------------------- 箭头与标注
def vec(start, vector, color=INK, buff: float = 0.0, tip: float = 0.18,
        stroke: float = 5) -> Arrow:
    """矢量箭头。"""
    return Arrow(np.asarray(start, dtype=float),
                 np.asarray(start, dtype=float) + np.asarray(vector, dtype=float),
                 buff=buff, color=color, tip_length=tip, stroke_width=stroke)


def darrow(p1, p2, color=INK, tip: float = 0.14, stroke: float = 3) -> DoubleArrow:
    """双头尺寸箭头。"""
    return DoubleArrow(p1, p2, buff=0, color=color, tip_length=tip,
                       stroke_width=stroke)


def dim_line(p1, p2, label=None, offset: float = 0.35, color=MUTED,
             size: int = 22) -> VGroup:
    """尺寸标注：延长线 + 双头箭 + 标注文字。"""
    p1, p2 = np.asarray(p1, dtype=float), np.asarray(p2, dtype=float)
    d = p2 - p1
    u = d / np.linalg.norm(d)
    nv = np.array([-u[1], u[0], 0.0])
    off = nv * offset
    ext1 = DashedLine(p1 + nv * offset * 0.3, p1 + off + nv * 0.12, color=color,
                      stroke_width=1.5, dash_length=0.06)
    ext2 = DashedLine(p2 + nv * offset * 0.3, p2 + off + nv * 0.12, color=color,
                      stroke_width=1.5, dash_length=0.06)
    arr = darrow(p1 + off, p2 + off, color=color)
    g = VGroup(ext1, ext2, arr)
    if label is not None:
        lab = label if not isinstance(label, str) else None
        if lab is None:
            from .style import ctext
            lab = ctext(label, size=size, color=color)
        lab.next_to(arr, nv, buff=0.1)
        g.add(lab)
    return g


def angle_mark(center, dir1: float, dir2: float, r: float = 0.5, color=INK,
               label=None, stroke: float = 3) -> VGroup:
    """角度标注圆弧（dir1→dir2 逆时针）+ 可选文字。"""
    a2 = dir2
    while a2 < dir1:
        a2 += TAU
    sweep = a2 - dir1
    if sweep > PI:
        sweep -= TAU
    arc = Arc(radius=r, start_angle=dir1, angle=sweep, color=color,
              stroke_width=stroke, arc_center=center)
    g = VGroup(arc)
    if label is not None:
        mid = dir1 + sweep / 2
        lab = label
        if isinstance(label, str):
            from .style import ctext
            lab = ctext(label, size=24, color=color)
        lab.move_to(center + np.array([np.cos(mid), np.sin(mid), 0]) * (r + 0.32))
        g.add(lab)
    return g


def dashed(p1, p2, color=MUTED, dash: float = 0.09, stroke: float = 2.5) -> DashedLine:
    return DashedLine(p1, p2, color=color, dash_length=dash, stroke_width=stroke)


# ---------------------------------------------------------------- 带传动
def _tangent_points(c1, r1, c2, r2, side=1):
    """两圆外公切线切点（同向带）。side=±1 取上下两条。"""
    d = np.linalg.norm(c2 - c1)
    u = (c2 - c1) / d
    nv = np.array([-u[1], u[0], 0.0]) * side
    return c1 + nv * r1, c2 + nv * r2


def belt_drive(c1, r1, c2, r2, color=MUTED, crossed: bool = False,
               stroke: float = 4) -> VGroup:
    """带传动：两轮 + 两段公切线（crossed=True 画交叉带，内公切线）。"""
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    w1 = Circle(radius=r1, color=FRAME_C, stroke_width=5).move_to(c1)
    w2 = Circle(radius=r2, color=FRAME_C, stroke_width=5).move_to(c2)
    if crossed:
        d = np.linalg.norm(c2 - c1)
        u = (c2 - c1) / d
        base = np.arctan2(u[1], u[0])
        beta = np.arccos(np.clip((r1 + r2) / d, -1, 1))
        segs = []
        for s in (1, -1):
            ang_n = base + s * (PI - beta)      # 法线方向 n·u = -(r1+r2)/d
            nv = np.array([np.cos(ang_n), np.sin(ang_n), 0.0])
            t1 = c1 - r1 * nv
            t2 = c2 + r2 * nv
            segs.append(Line(t1, t2, color=color, stroke_width=stroke))
        b1 = segs
    else:
        t1a, t2a = _tangent_points(c1, r1, c2, r2, side=1)
        t1b, t2b = _tangent_points(c1, r1, c2, r2, side=-1)
        b1 = [Line(t1a, t2a, color=color, stroke_width=stroke),
              Line(t1b, t2b, color=color, stroke_width=stroke)]
    return VGroup(w1, w2, *b1)


def belt_path(c1, r1, c2, r2):
    """带闭合路径点列（供 TracedPath/循环动点用）：切线+圆弧。"""
    c1, c2 = np.asarray(c1, dtype=float), np.asarray(c2, dtype=float)
    t1a, t2a = _tangent_points(c1, r1, c2, r2, 1)
    t1b, t2b = _tangent_points(c1, r1, c2, r2, -1)
    ang2a = np.arctan2((t2a - c2)[1], (t2a - c2)[0])
    ang2b = np.arctan2((t2b - c2)[1], (t2b - c2)[0])
    ang1a = np.arctan2((t1a - c1)[1], (t1a - c1)[0])
    ang1b = np.arctan2((t1b - c1)[1], (t1b - c1)[0])

    def arc_pts(c, r, a0, a1, n=24):
        while a1 < a0:
            a1 += TAU
        ts = np.linspace(a0, a1, n)
        return np.stack([c[0] + r * np.cos(ts), c[1] + r * np.sin(ts),
                         np.zeros(n)], axis=1)

    pts = [np.linspace(t1a, t2a, 12), arc_pts(c2, r2, ang2a, ang2b, 30),
           np.linspace(t2b, t1b, 12), arc_pts(c1, r1, ang1b, ang1a, 30)]
    return np.vstack(pts)


def saw_wheel(point, r: float, z: int, color=LINK_D, ratchet: bool = True) -> VMobject:
    """棘轮/锯齿轮廓：缓升沿 + 陡降沿。"""
    pts = []
    for k in range(z):
        a_root = k * TAU / z
        a_tip = (k + 0.85) * TAU / z
        r_root, r_tip = r * 0.78, r
        for a in np.linspace(a_root, a_tip, 5, endpoint=False):
            rr = r_root + (r_tip - r_root) * (a - a_root) / (a_tip - a_root)
            pts.append([rr * np.cos(a), rr * np.sin(a), 0])
        pts.append([r_tip * np.cos(a_tip), r_tip * np.sin(a_tip), 0])
        a_next = (k + 1) * TAU / z
        pts.append([r_root * np.cos(a_next), r_root * np.sin(a_next), 0])
    pts = np.array(pts)
    vm = VMobject(color=color, stroke_width=3)
    vm.set_points_as_corners([*pts, pts[0]])
    return vm.shift(point)


def star_polygon(point, r_out: float, r_in: float, n: int = 5, color=INK) -> VMobject:
    """星形（凸轮轮廓装饰/飞轮标志用）。"""
    pts = []
    for k in range(2 * n):
        r = r_out if k % 2 == 0 else r_in
        a = k * PI / n
        pts.append([r * np.cos(a), r * np.sin(a), 0])
    vm = VMobject(color=color, stroke_width=2.5)
    vm.set_points_as_corners([*np.array(pts), np.array(pts[0])])
    return vm.shift(point)
