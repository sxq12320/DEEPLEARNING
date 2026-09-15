# -*- coding: utf-8 -*-
"""mechlib.curves — 渐开线/齿轮轮廓/凸轮运动规律与廓线/连杆曲线.

渐开线（教材 §10-3, p199）：x = rb(cos t + t sin t), y = rb(sin t − t cos t)
齿轮几何（§10-4 表10-2, p202-203）：d=mz, db=d·cosα, da=d+2ha*m, df=d−2(ha*+c*)m
凸轮规律（§9-2, p170-176）：等速/等加速等减速/余弦/正弦
"""
from __future__ import annotations

import numpy as np
from manim import PI, TAU, VMobject


# ---------------------------------------------------------------- 渐开线
def involute_pts(rb: float, t_max: float, n: int = 40, sign: int = 1) -> np.ndarray:
    """渐开线离散点（§10-3, p199）。sign=−1 生成镜像侧。"""
    t = np.linspace(0, t_max, n)
    x = rb * (np.cos(t) + t * np.sin(t))
    y = rb * (np.sin(t) - t * np.cos(t)) * sign
    return np.stack([x, y, np.zeros_like(x)], axis=1)


def involute_polar_angle(rb: float, rr: float) -> float:
    """半径 rr 处渐开线点的极角（相对基圆起始点）：θ = inv(α_r)。"""
    t = np.sqrt(max((rr / rb) ** 2 - 1.0, 0.0))
    return t - np.arctan(t)


# ---------------------------------------------------------------- 齿轮轮廓
def gear_profile(m: float, z: int, alpha_deg: float = 20.0, ha_star: float = 1.0,
                 c_star: float = 0.25, scale: float = 1.0, color="#4CC9F0",
                 stroke_width: float = 2.5, x: float = 0.0) -> VMobject:
    """渐开线直齿轮完整轮廓（教学精度：齿根过渡简化为径向线+根圆弧）。

    x: 变位系数（§10-7, p216-220；x>0 齿顶变尖齿根变宽）。
    """
    alpha = np.deg2rad(alpha_deg)
    r = m * z / 2.0
    rb = r * np.cos(alpha)
    ra = r + (ha_star + x) * m
    rf = max(r - (ha_star + c_star - x) * m, 0.35 * r)
    inv_a = np.tan(alpha) - alpha
    # 分度圆齿厚 s = m(π/2 + 2x·tanα) → 半齿角 s/(2r)·? 半齿圆心角 = s/d
    s_half_ang = (PI / 2 + 2 * x * np.tan(alpha)) / (2 * z) + 0.0
    half_tooth_ang = s_half_ang + inv_a  # 分度圆齿面点→齿槽对称线的角距

    n_pts = 26
    rr_list = np.linspace(max(rb, rf), ra, n_pts)
    pts_right = []
    for rr in rr_list:
        th = half_tooth_ang - involute_polar_angle(rb, rr)
        pts_right.append([rr * np.cos(th), rr * np.sin(th), 0.0])
    pts_right = np.array(pts_right)
    if rf < rb:
        th0 = half_tooth_ang
        pts_right = np.vstack([[rf * np.cos(th0), rf * np.sin(th0), 0.0], pts_right])
    pts_left = pts_right.copy()
    pts_left[:, 1] *= -1
    tooth = np.vstack([pts_left[::-1], pts_right])

    prof = []
    pitch_ang = TAU / z
    for k in range(z):
        rot = k * pitch_ang
        c, s = np.cos(rot), np.sin(rot)
        R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
        tk = tooth @ R.T
        prof.append(tk)
        rel = np.arctan2(tk[-1, 1], tk[-1, 0]) - rot
        rel = (rel + PI) % TAU - PI
        arc_t = np.linspace(rot + rel, rot + pitch_ang - rel, 8)
        prof.append(np.stack([rf * np.cos(arc_t), rf * np.sin(arc_t),
                              np.zeros_like(arc_t)], axis=1))
    pts = np.vstack(prof) * scale
    vm = VMobject(color=color, stroke_width=stroke_width)
    vm.set_points_as_corners([*pts, pts[0]])
    return vm


def gear_pair(z1: int, z2: int, m: float = 0.12, c1=np.zeros(3),
              direction: int = -1, phase2: float = 0.0, **kw):
    """一对外啮合齿轮轮廓：按标准中心距摆放并相位对齐（齿对槽）。

    返回 (g1, g2, a)：a 为中心距。direction=-1 外啮合反向。
    """
    from .style import GEAR_1, GEAR_2
    a = m * (z1 + z2) / 2.0
    g1 = gear_profile(m, z1, color=kw.get("color1", GEAR_1),
                      stroke_width=kw.get("stroke_width", 2.5)).move_to(c1)
    c2 = c1 + np.array([a, 0.0, 0.0])
    # 轮2 相位：连心线（角 π 处）应为齿槽 → 齿心在 π∓π/z2，即旋转 π−π/z2
    g2 = gear_profile(m, z2, color=kw.get("color2", GEAR_2),
                      stroke_width=kw.get("stroke_width", 2.5))
    g2.rotate(PI - PI / z2 + phase2, about_point=np.zeros(3)).move_to(c2)
    return g1, g2, a


# ---------------------------------------------------------------- 凸轮运动规律
def cam_law(name: str, h: float, beta: float):
    """推程位移规律 s(δ), δ∈[0,beta]（教材 §9-2, p170-176）。

    name: 'const' 等速 | 'para' 等加速等减速 | 'cos' 余弦加速度(简谐)
          | 'sine' 正弦加速度(摆线位移)
    返回 s(delta) 标量函数。
    """
    if name == "const":
        return lambda d: h * np.clip(d / beta, 0, 1)
    if name == "para":     # 等加速等减速（前半加速后半减速）
        def s(d):
            u = np.clip(d / beta, 0, 1)
            return h * (2 * u * u if u < 0.5 else -2 * u * u + 4 * u - 1)
        return s
    if name == "cos":      # 余弦加速度 = 简谐位移
        return lambda d: h / 2 * (1 - np.cos(PI * np.clip(d / beta, 0, 1)))
    if name == "sine":     # 正弦加速度 = 摆线位移
        return lambda d: h * (np.clip(d / beta, 0, 1)
                              - np.sin(TAU * np.clip(d / beta, 0, 1)) / TAU)
    raise ValueError(name)


def cam_law_va(name: str, h: float, beta: float, w: float = 1.0):
    """返回 (s(δ), v(δ), a(δ))：v=ds/dt, a=d²s/dt²（角速度 w 常数）。

    教材 p170-176 无量纲形式（以 δ 为自变量：v'=ds/dδ=v/w, a'=d²s/dδ²=a/w²）。
    这里直接返回以 δ 为自变量的导数（教学用，形状相同）。
    """
    if name == "const":
        return (lambda d: h * np.clip(d / beta, 0, 1),
                lambda d: h / beta * (0 <= d <= beta),
                lambda d: 0.0)
    if name == "para":
        def s(d):
            u = np.clip(d / beta, 0, 1)
            return h * (2 * u * u if u < 0.5 else -2 * u * u + 4 * u - 1)
        def v(d):
            u = np.clip(d / beta, 0, 1)
            if not (0 <= d <= beta):
                return 0.0
            return h / beta * (4 * u if u < 0.5 else 4 - 4 * u)
        def a(d):
            u = np.clip(d / beta, 0, 1)
            if not (0 <= d <= beta):
                return 0.0
            return 4 * h / beta ** 2 * (1 if u < 0.5 else -1)
        return s, v, a
    if name == "cos":
        return (lambda d: h / 2 * (1 - np.cos(PI * np.clip(d / beta, 0, 1))),
                lambda d: (h * PI / (2 * beta)) * np.sin(PI * np.clip(d / beta, 0, 1))
                if 0 <= d <= beta else 0.0,
                lambda d: (h * PI ** 2 / (2 * beta ** 2)) * np.cos(PI * np.clip(d / beta, 0, 1))
                if 0 <= d <= beta else 0.0)
    if name == "sine":
        return (lambda d: h * (np.clip(d / beta, 0, 1)
                               - np.sin(TAU * np.clip(d / beta, 0, 1)) / TAU),
                lambda d: (h / beta) * (1 - np.cos(TAU * np.clip(d / beta, 0, 1)))
                if 0 <= d <= beta else 0.0,
                lambda d: (h * TAU / beta ** 2) * np.sin(TAU * np.clip(d / beta, 0, 1))
                if 0 <= d <= beta else 0.0)
    raise ValueError(name)


def cam_profile_knife(s_func, r0: float, n: int = 360, scale: float = 1.0,
                      color="#F78C6B", stroke_width: float = 3.0) -> VMobject:
    """对心尖顶直动推杆盘形凸轮理论廓线（反转法, §9-3, p177-179）：
    极坐标 r(δ) = r0 + s(δ)。"""
    d = np.linspace(0, TAU, n, endpoint=False)
    r = r0 + np.array([s_func(x) for x in d])
    pts = np.stack([r * np.cos(d), r * np.sin(d), np.zeros_like(d)], axis=1) * scale
    vm = VMobject(color=color, stroke_width=stroke_width)
    vm.set_points_as_corners([*pts, pts[0]])
    return vm


def cam_profile_roller(s_func, r0: float, rr: float, n: int = 360,
                       scale: float = 1.0, color="#F78C6B",
                       stroke_width: float = 3.0) -> VMobject:
    """滚子推杆实际廓线：理论廓线的内等距线（近似法向偏置）。

    理论廓线（滚子中心轨迹）r_t(δ) = r0 + rr + s(δ)；实际廓线沿法向内偏 rr。
    """
    d = np.linspace(0, TAU, n, endpoint=False)
    rt = r0 + rr + np.array([s_func(x) for x in d])
    pts_t = np.stack([rt * np.cos(d), rt * np.sin(d), np.zeros_like(d)], axis=1)
    # 数值法向（朝向凸轮中心一侧）
    tang = np.gradient(pts_t, axis=0)
    nrm = np.stack([-tang[:, 1], tang[:, 0], np.zeros_like(d)], axis=1)
    nrm /= np.linalg.norm(nrm, axis=1, keepdims=True) + 1e-12
    inward = -pts_t  # 指向中心
    sign = np.sign(np.einsum("ij,ij->i", nrm, inward))[:, None]
    pts = (pts_t + nrm * sign * rr) * scale
    vm = VMobject(color=color, stroke_width=stroke_width)
    vm.set_points_as_corners([*pts, pts[0]])
    return vm


# ---------------------------------------------------------------- 连杆曲线
def coupler_curve(fourbar, s: float = 0.5, h: float = 0.0, n: int = 200,
                  theta_range=(0, TAU), color="#F78C6B",
                  stroke_width: float = 3.0) -> VMobject:
    """连杆曲线：连杆上描点随曲柄一周的轨迹（§8-6, p155-158）。"""
    ths = np.linspace(theta_range[0], theta_range[1], n)
    pts = np.array([fourbar.coupler_point(t, s, h) for t in ths])
    vm = VMobject(color=color, stroke_width=stroke_width)
    vm.set_points_smoothly(pts)
    return vm
