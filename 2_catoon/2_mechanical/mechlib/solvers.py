# -*- coding: utf-8 -*-
"""mechlib.solvers — 运动学解算器（所有公式以孙桓《机械原理》八版为准）.

- FourBar：铰链四杆位置解 + Grashof 判定 + 传动角 + 极位
- CrankSlider：曲柄滑块位置解
- Whitworth：曲柄摆动导杆急回机构
- Geneva：外槽轮机构转角函数
- Planetary：简单行星轮系转速关系
"""
from __future__ import annotations

import numpy as np
from manim import ORIGIN, PI, TAU


def P(x, y):
    return np.array([float(x), float(y), 0.0])


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


# ---------------------------------------------------------------- 铰链四杆
class FourBar:
    """铰链四杆机构位置求解（教材 §3-3 解析法思想, p43-46）。

    机架 A0(原点)-B0(L1,0)；曲柄 A0A=L2、连杆 AB=L3、摇杆 B0B=L4。
    solve(theta2) 返回 (A0, A, B, B0)；branch=±1 选装配分支。
    """

    def __init__(self, L1, L2, L3, L4, origin=ORIGIN, branch: int = 1):
        self.L1, self.L2, self.L3, self.L4 = map(float, (L1, L2, L3, L4))
        self.o = np.array(origin, dtype=float)
        self.branch = branch

    # --- 特性判定（教材 §8-3, p131-135）
    def grashof(self) -> bool:
        s = sorted([self.L1, self.L2, self.L3, self.L4])
        return s[0] + s[3] <= s[1] + s[2]

    def link_type(self) -> str:
        """以 L2 为连架杆时机型名称（供教学标注）。"""
        Ls = sorted([self.L1, self.L2, self.L3, self.L4])
        g = Ls[0] + Ls[3] <= Ls[1] + Ls[2]
        if not g:
            return "双摇杆机构（非 Grashof）"
        if abs(self.L2 - Ls[0]) < 1e-9:
            return "曲柄摇杆机构（最短杆为连架杆）"
        if abs(self.L1 - Ls[0]) < 1e-9:
            return "双曲柄机构（最短杆为机架）"
        return "双摇杆机构（最短杆为连杆）"

    def solve(self, theta2: float):
        A0 = self.o
        B0 = self.o + np.array([self.L1, 0.0, 0.0])
        A = A0 + self.L2 * np.array([np.cos(theta2), np.sin(theta2), 0.0])
        d_vec = B0 - A
        d = float(np.linalg.norm(d_vec))
        d = max(min(d, self.L3 + self.L4 - 1e-9),
                abs(self.L3 - self.L4) + 1e-9)
        a = (self.L3 ** 2 - self.L4 ** 2 + d ** 2) / (2 * d)
        h = np.sqrt(max(self.L3 ** 2 - a ** 2, 0.0))
        u = d_vec / d
        n = np.array([-u[1], u[0], 0.0])
        B = A + a * u + self.branch * h * n
        return A0, A, B, B0

    def coupler_point(self, theta2: float, s: float = 0.5, h: float = 0.0):
        """连杆上描点：s=沿 AB 比例，h=垂直 AB 偏移（左正）。"""
        _, A, B, _ = self.solve(theta2)
        u = _unit(B - A)
        n = np.array([-u[1], u[0], 0.0])
        return A + s * (B - A) + h * n

    def transmission_angle(self, theta2: float) -> float:
        """传动角 γ = 连杆与摇杆夹角（0~π，理想 90°）。"""
        _, A, B, B0 = self.solve(theta2)
        v1 = _unit(A - B)
        v2 = _unit(B0 - B)
        cosg = np.clip(np.dot(v1, v2), -1, 1)
        g = np.arccos(cosg)
        return min(g, PI - g) if g > PI / 2 else g  # 取锐角补角规范见 p136

    def crank_limit_angles(self):
        """曲柄摇杆（L2 为曲柄）两极限位置曲柄角：曲柄与连杆共线。
        返回 (θ_工作行程末, θ_回程末) 或 None（非曲柄摇杆）。
        依据：B 在圆(B0,L4) 上且 |A0B| = L3+L2 或 |L3−L2|。"""
        out = []
        for s in (+1, -1):
            dAB = self.L3 + s * self.L2
            if dAB <= 0:
                continue
            cos_t = (self.L1 ** 2 + dAB ** 2 - self.L4 ** 2) / (2 * self.L1 * dAB)
            if abs(cos_t) > 1:
                continue
            t = np.arccos(np.clip(cos_t, -1, 1))
            out.append(t)
        if len(out) < 2:
            return None
        return tuple(sorted(out))


# ---------------------------------------------------------------- 曲柄滑块
class CrankSlider:
    """曲柄滑块位置解（教材 §3-3 例, p44-46）。
    曲柄 r（原点转动）、连杆 l、偏距 e（导路 y=e）。solve(θ)→(O,A,B)。"""

    def __init__(self, r, l, e: float = 0.0, origin=ORIGIN):
        self.r, self.l, self.e = float(r), float(l), float(e)
        self.o = np.array(origin, dtype=float)

    def solve(self, theta: float):
        O = self.o
        A = O + np.array([self.r * np.cos(theta), self.r * np.sin(theta), 0.0])
        s = self.r * np.sin(theta) - self.e
        xB = self.r * np.cos(theta) + np.sqrt(max(self.l ** 2 - s ** 2, 1e-9))
        B = O + np.array([xB, self.e, 0.0])
        return O, A, B

    def stroke(self) -> float:
        """滑块行程（近似 2r，偏置略大）。"""
        return (self.solve(0.0)[2][0] - self.solve(PI)[2][0])


# ---------------------------------------------------------------- 摆动导杆急回机构
class Whitworth:
    """曲柄摆动导杆（牛头刨）急回机构（教材 §8-4, p137-138）。
    曲柄中心 O1、导杆摆轴 O2（在 O1 正下方 d 处）、曲柄半径 r<d。
    solve(θ)→(O1, A, O2, φ)：A 曲柄销位置，φ 导杆角（自 +x 轴）。
    急回极位：导杆与曲柄圆相切，极位夹角 θ_p = 2·arcsin(r/d)。"""

    def __init__(self, r, d, lever: float = 2.6, origin=ORIGIN):
        self.r, self.d, self.lever = float(r), float(d), float(lever)
        self.o = np.array(origin, dtype=float)

    @property
    def O2(self):
        return self.o + P(0, -self.d)

    def solve(self, theta: float):
        O1 = self.o
        A = O1 + P(self.r * np.cos(theta), self.r * np.sin(theta))
        O2 = self.O2
        v = A - O2
        phi = np.arctan2(v[1], v[0])
        return O1, A, O2, phi

    def extreme_angle(self) -> float:
        """极位夹角 θ = 2·asin(r/d)（教材式 8-x）。"""
        return 2 * np.arcsin(self.r / self.d)


# ---------------------------------------------------------------- 外槽轮机构
def geneva_state(theta: float, z: int, a: float = 2.4, r=None):
    """外槽轮机构状态（教材 §12-2, p264-267）。

    主动盘中心 C1=(0,0)，槽轮中心 C2=(a,0)，销圆半径 r=a·sin(π/z)。
    θ=0 为销指向 C2 的啮合中点。啮合半角 θ_e=π/2−π/z；每次啮合槽轮转 −2π/z。
    返回 (wheel_angle, engaged)：槽轮绝对角（连续多圈可解卷）与啮合标志。
    """
    if r is None:
        r = a * np.sin(PI / z)
    th_e = PI / 2 - PI / z
    k = int(np.floor((theta + th_e) / TAU))   # 最近一个啮合窗口编号
    t = theta - TAU * k                        # 相对窗口中心的角（t≥−θ_e）
    px, py = r * np.cos(theta), r * np.sin(theta)
    raw = np.arctan2(py, px - a)               # 销相对槽轮中心方向
    if raw < 0:
        raw += TAU                             # 连续分支 ∈ (π−π/z, π+π/z)
    if abs(t) <= th_e:
        return raw - TAU * k / z, True
    return (PI - PI / z) - TAU * k / z, False  # 锁止：保持窗口末值


# ---------------------------------------------------------------- 行星轮系
class Planetary:
    """单排行星轮系（2K-H，教材 §11-2~11-4, p239-248）。

    z1 太阳轮、z2 行星轮、z3 内齿圈（z3 = z1 + 2 z2 同心条件）。
    转化机构法：(n1−nH)/(n3−nH) = −z3/z1。
    speeds(n1, n3) → (n1, n2, n3, nH)。
    """

    def __init__(self, z1, z2, z3=None):
        self.z1, self.z2 = int(z1), int(z2)
        self.z3 = int(z3) if z3 else self.z1 + 2 * self.z2

    def carrier(self, n1: float, n3: float) -> float:
        """由太阳轮/齿圈转速求转臂转速。"""
        k = -self.z3 / self.z1
        # (n1−nH) = k·(n3−nH) → nH = (k·n3 − n1)/(k − 1)
        return (k * n3 - n1) / (k - 1)

    def speeds(self, n1: float = None, n3: float = None, nH: float = None):
        """三选二求解；返回 dict(n1,n2,n3,nH)。缺项自动解。"""
        k = -self.z3 / self.z1
        if nH is None:
            nH = self.carrier(n1, n3)
        elif n3 is None:
            n3 = nH + (n1 - nH) / k if abs(k) > 1e-9 else nH
        elif n1 is None:
            n1 = nH + k * (n3 - nH)
        # 行星轮自转（相对转臂）：(n1−nH)·z1 + (n2−nH)·z2 = 0
        n2 = nH - (n1 - nH) * self.z1 / self.z2
        return dict(n1=n1, n2=n2, n3=n3, nH=nH)


# ---------------------------------------------------------------- 差速器
def diff_speeds(n_carrier: float, n_left: float = None, n_right: float = None):
    """汽车差速器：n_L + n_R = 2·n_H。给一边求另一边。"""
    if n_left is None:
        n_left = 2 * n_carrier - n_right
    if n_right is None:
        n_right = 2 * n_carrier - n_left
    return n_left, n_right
