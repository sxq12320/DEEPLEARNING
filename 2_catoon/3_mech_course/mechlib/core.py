# -*- coding: utf-8 -*-
"""mechlib.core — 共享原语：版式 / 公式推导 / 机构绘制 / 运动学求解 / 齿轮凸轮轮廓.

正确性注记：所有运动学公式以孙桓《机械原理》第八版为准（页码见调用处注释）。
渐开线：x = rb(cos t + t sin t), y = rb(sin t − t cos t)（教材 §10-3, p199）。
"""

from __future__ import annotations

import numpy as np
from manim import (
    BLUE, BLUE_B, GREY_B, GREY_C, GOLD, GREEN, ORANGE, RED, TEAL, WHITE, YELLOW,
    DOWN, LEFT, ORIGIN, RIGHT, UP, PI, TAU,
    Circle, Dot, FadeIn, Line, MathTex, Polygon, Rectangle, ReplacementTransform,
    Scene, Square, Text, VGroup, VMobject, Write, config,
)

# ---------------------------------------------------------------- 版式与配色
CJK = "Microsoft YaHei"   # Windows 自带中文字体
ACCENT = YELLOW           # 强调色（当前讲到的元素）
GOOD = GREEN              # 正确/成立
BAD = RED                 # 错误/陷阱
NOTE = TEAL               # 注记/页码


def ctext(s: str, size: int = 34, color=WHITE, weight="NORMAL") -> Text:
    """中文文本（统一字体）。"""
    return Text(s, font=CJK, font_size=size, color=color, weight=weight)


def title_bar(title: str, sub: str = "") -> VGroup:
    """页首标题条：主标题 + 可选小标题，置于画面顶部。"""
    t = ctext(title, size=40, color=WHITE, weight="BOLD")
    g = VGroup(t)
    if sub:
        g.add(ctext(sub, size=24, color=GREY_B))
    g.arrange(DOWN, aligned_edge=LEFT, buff=0.12).to_corner(UP + LEFT, buff=0.45)
    underline = Line(g.get_corner(DOWN + LEFT), g.get_corner(DOWN + LEFT) + RIGHT * 11.5,
                     stroke_width=2, color=GREY_C).shift(DOWN * 0.12)
    return VGroup(g, underline)


def bullets(items: list[str], size: int = 30, buff: float = 0.32, marker: str = "▸ ") -> VGroup:
    """竖排要点列表。"""
    rows = VGroup(*[ctext(marker + s, size=size) for s in items])
    rows.arrange(DOWN, aligned_edge=LEFT, buff=buff)
    return rows


def page_ref(p: str) -> Text:
    """教材页码水印（正确性锚点），右下角。例：page_ref("孙桓八版 p16")。"""
    return ctext(p, size=20, color=NOTE).to_corner(DOWN + RIGHT, buff=0.3)


def formula_reveal(scene: Scene, steps: list[MathTex], anchor=ORIGIN, buff: float = 0.5,
                   run_time: float = 1.0, wait: float = 1.2, keep_last_only: bool = False):
    """公式分步推导：逐条出现（供配音逐步讲解，每步之间留白）。

    steps: 已排版好的 MathTex 列表；本函数负责纵向排列并逐条 Write。
    keep_last_only=True 时每步替换上一步（长推导省空间）。
    返回最终 VGroup 便于后续 FadeOut。
    """
    group = VGroup(*steps).arrange(DOWN, aligned_edge=LEFT, buff=buff).move_to(anchor)
    shown = []
    for i, st in enumerate(steps):
        if keep_last_only and shown:
            scene.play(ReplacementTransform(shown[-1], st), run_time=run_time)
            shown[-1] = st
        else:
            scene.play(Write(st), run_time=run_time)
            shown.append(st)
        scene.wait(wait)
    return VGroup(*shown)


# ---------------------------------------------------------------- 机构绘制原语
def ground_hatch(point: np.ndarray, width: float = 0.9, angle: float = 0.0,
                 n: int = 6, color=GREY_B) -> VGroup:
    """机架符号：一条横线 + 斜排短线（标准制图画法）。angle 为整体旋转角。"""
    base = Line(LEFT * width / 2, RIGHT * width / 2, color=color, stroke_width=3)
    hatches = VGroup(*[
        Line(ORIGIN, DOWN * 0.22 + LEFT * 0.12, color=color, stroke_width=2)
        .move_to(base.point_from_proportion(i / (n - 1)), aligned_edge=UP)
        for i in range(n)
    ])
    g = VGroup(base, hatches).rotate(angle).shift(point)
    return g


def pin_joint(point: np.ndarray, r: float = 0.09, color=WHITE) -> VGroup:
    """转动副（铰链）符号：空心圆 + 中心点。"""
    return VGroup(
        Circle(radius=r, color=color, stroke_width=3).move_to(point),
        Dot(point, radius=0.03, color=color),
    )


def fixed_pin(point: np.ndarray, r: float = 0.1, color=WHITE) -> VGroup:
    """固定铰链：铰链 + 机架斜线（地面固定端）。"""
    return VGroup(pin_joint(point, r, color), ground_hatch(point + DOWN * (r + 0.05), width=0.6))


def link_line(p1: np.ndarray, p2: np.ndarray, color=BLUE_B, w: float = 7) -> Line:
    """构件（杆）：粗线段。"""
    return Line(p1, p2, color=color, stroke_width=w)


def slider_block(point: np.ndarray, wdt: float = 0.7, hgt: float = 0.4,
                 angle: float = 0.0, color=ORANGE) -> Rectangle:
    """移动副滑块符号。"""
    return Rectangle(width=wdt, height=hgt, color=color, stroke_width=4,
                     fill_opacity=0.25, fill_color=color).rotate(angle).move_to(point)


# ---------------------------------------------------------------- 运动学求解器
class FourBar:
    """铰链四杆机构位置求解（教材 §3-3 解析法思想, p43-46）。

    机架 A0(原点)-B0(L1,0)；曲柄 A0A=L2、连杆 AB=L3、摇杆 B0B=L4。
    solve(theta2) 返回 (A0, A, B, B0) 三维坐标（z=0）；branch=+1/-1 选装配分支。
    位置解 = 圆(A, L3) 与 圆(B0, L4) 交点（余弦定理法，见 p44 式(3-13)同源推导）。
    """

    def __init__(self, L1: float, L2: float, L3: float, L4: float,
                 origin=ORIGIN, branch: int = 1):
        self.L1, self.L2, self.L3, self.L4 = L1, L2, L3, L4
        self.o = np.array(origin, dtype=float)
        self.branch = branch

    def grashof(self) -> bool:
        Ls = sorted([self.L1, self.L2, self.L3, self.L4])
        return Ls[0] + Ls[3] <= Ls[1] + Ls[2]  # 教材 §8-3 曲柄存在条件 (p131-133)

    def solve(self, theta2: float):
        A0 = self.o
        B0 = self.o + np.array([self.L1, 0, 0])
        A = A0 + self.L2 * np.array([np.cos(theta2), np.sin(theta2), 0])
        d_vec = B0 - A
        d = float(np.linalg.norm(d_vec))
        d = max(min(d, self.L3 + self.L4 - 1e-9), abs(self.L3 - self.L4) + 1e-9)  # 数值夹取防出域
        # 圆-圆交点：沿 AB0 方向前进 a，再垂直偏移 h
        a = (self.L3 ** 2 - self.L4 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(max(self.L3 ** 2 - a ** 2, 0.0))
        u = d_vec / d
        n = np.array([-u[1], u[0], 0.0])
        B = A + a * u + self.branch * h * n
        return A0, A, B, B0


class CrankSlider:
    """曲柄滑块机构位置求解（教材 §3-3 例, p44-46）。

    曲柄 r（原点转动）、连杆 l、偏距 e（滑块导路 y=e）。
    solve(theta) 返回 (O, A, B)：A 曲柄销，B 滑块销。
    x_B = r cosθ + sqrt(l² − (r sinθ − e)²)。
    """

    def __init__(self, r: float, l: float, e: float = 0.0, origin=ORIGIN):
        self.r, self.l, self.e = r, l, e
        self.o = np.array(origin, dtype=float)

    def solve(self, theta: float):
        O = self.o
        A = O + np.array([self.r * np.cos(theta), self.r * np.sin(theta), 0.0])
        s = self.r * np.sin(theta) - self.e
        xB = self.r * np.cos(theta) + np.sqrt(max(self.l ** 2 - s ** 2, 1e-9))
        B = O + np.array([xB, self.e, 0.0])
        return O, A, B


# ---------------------------------------------------------------- 齿轮与凸轮轮廓
def involute_pts(rb: float, t_max: float, n: int = 40, sign: int = 1) -> np.ndarray:
    """渐开线离散点（教材 §10-3, p199-200）：
    x = rb(cos t + t sin t), y = rb(sin t − t cos t)；sign=-1 生成镜像侧齿廓。
    """
    t = np.linspace(0, t_max, n)
    x = rb * (np.cos(t) + t * np.sin(t))
    y = rb * (np.sin(t) - t * np.cos(t)) * sign
    return np.stack([x, y, np.zeros_like(x)], axis=1)


def gear_profile(m: float, z: int, alpha_deg: float = 20.0, ha_star: float = 1.0,
                 c_star: float = 0.25, scale: float = 1.0, color=BLUE_B,
                 stroke_width: float = 2.5) -> VMobject:
    """渐开线标准直齿轮完整轮廓（教学精度：真实渐开线齿面 + 圆弧齿根过渡省略为径向线）。

    几何关系（教材 §10-4 表 10-2, p202-203）：
    d=mz, db=d·cosα, da=d+2ha*·m, df=d−2(ha*+c*)·m；
    分度圆齿厚 s=πm/2 → 半齿角 = s/d = π/(2z)；
    渐开线在分度圆处的展角修正用 inv α = tanα − α（p200）。
    """
    alpha = np.deg2rad(alpha_deg)
    r = m * z / 2.0
    rb = r * np.cos(alpha)
    ra = r + ha_star * m
    rf = max(r - (ha_star + c_star) * m, 0.35 * r)
    inv_a = np.tan(alpha) - alpha

    def involute_polar(rr):
        """半径 rr 处渐开线点的 (rr, 极角相对基圆起始)。t = 展开角参数。"""
        t = np.sqrt(max((rr / rb) ** 2 - 1.0, 0.0))
        theta = t - np.arctan(t)  # = inv(压力角 at rr)
        return theta

    # 半齿：从 rf 到 ra 的渐开线（rf<rb 时用径向线补到 rb）
    n_pts = 26
    rr_list = np.linspace(max(rb, rf), ra, n_pts)
    half_tooth_ang = PI / (2 * z) + inv_a  # 齿厚中心线到分度圆齿面点的角（对称基准）
    pts_right = []
    for rr in rr_list:
        th = half_tooth_ang - involute_polar(rr)
        pts_right.append([rr * np.cos(th), rr * np.sin(th), 0.0])
    pts_right = np.array(pts_right)
    if rf < rb:  # 径向补根
        th0 = half_tooth_ang
        pts_right = np.vstack([[rf * np.cos(th0), rf * np.sin(th0), 0.0], pts_right])
    pts_left = pts_right.copy()
    pts_left[:, 1] *= -1
    tooth = np.vstack([pts_left[::-1], pts_right])  # 左齿面(倒序) + 右齿面

    prof = []
    pitch_ang = TAU / z
    for k in range(z):
        rot = k * pitch_ang
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        tk = tooth @ R.T
        prof.append(tk)
        # 齿槽根部圆弧（相邻齿之间沿 rf 的圆弧）
        a0 = np.arctan2(tk[-1, 1], tk[-1, 0])
        a1 = rot + pitch_ang - (np.arctan2(tk[-1, 1], tk[-1, 0]) - rot)
        arc_t = np.linspace(a0, a1, 8)
        prof.append(np.stack([rf * np.cos(arc_t), rf * np.sin(arc_t),
                              np.zeros_like(arc_t)], axis=1))
    pts = np.vstack(prof) * scale
    vm = VMobject(color=color, stroke_width=stroke_width)
    vm.set_points_as_corners([*pts, pts[0]])
    return vm


def cam_profile_knife(s_func, r0: float, n: int = 360, scale: float = 1.0,
                      color=ORANGE, stroke_width: float = 3.0) -> VMobject:
    """对心尖顶直动推杆盘形凸轮理论轮廓（反转法，教材 §9-3, p177-179）：
    极坐标 r(δ) = r0 + s(δ)。s_func: [0,2π)->位移。
    （滚子推杆实际轮廓 = 该理论轮廓的内等距线，课程动画中单独演示。）
    """
    d = np.linspace(0, TAU, n, endpoint=False)
    r = r0 + np.array([s_func(x) for x in d])
    pts = np.stack([r * np.cos(d), r * np.sin(d), np.zeros_like(d)], axis=1) * scale
    vm = VMobject(color=color, stroke_width=stroke_width)
    vm.set_points_as_corners([*pts, pts[0]])
    return vm


# ---------------------------------------------------------------- 课程 Scene 基类
class LessonScene(Scene):
    """课程场景基类：统一背景/片尾留白；子类 docstring 即讲稿要点（含教材页码）。"""

    lesson = ""   # 如 "L01"
    seg = ""      # 如 "S1 从内燃机说起"

    def header(self, title: str, sub: str = ""):
        h = title_bar(title, sub)
        self.play(FadeIn(h), run_time=0.6)
        return h

    def hold(self, t: float = 1.5):
        """讲解留白（配音空间）。"""
        self.wait(t)
