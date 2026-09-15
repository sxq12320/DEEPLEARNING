# -*- coding: utf-8 -*-
"""mechlib.formulas — 公式分步推导 / 矢量图作图助手."""
from __future__ import annotations

import numpy as np
from manim import (
    DOWN, LEFT, ORIGIN, RIGHT, UP, Arrow, MathTex,
    ReplacementTransform, Scene, SurroundingRectangle, VGroup, Write,
)

from .style import ACCENT, INK, ctext


def formula_reveal(scene: Scene, steps, anchor=ORIGIN, buff: float = 0.5,
                   run_time: float = 1.0, wait: float = 1.2,
                   keep_last_only: bool = False):
    """公式分步推导：逐条 Write，供配音逐步讲解。

    steps: MathTex/Text/VGroup 列表；keep_last_only=True 时每步替换上一步。
    返回最终 VGroup 便于 FadeOut。
    """
    group = VGroup(*steps).arrange(DOWN, aligned_edge=LEFT, buff=buff).move_to(anchor)
    shown = []
    for st in steps:
        if keep_last_only and shown:
            scene.play(ReplacementTransform(shown[-1], st), run_time=run_time)
            shown[-1] = st
        else:
            scene.play(Write(st), run_time=run_time)
            shown.append(st)
        scene.wait(wait)
    return VGroup(*shown)


def eq_box(formula, label: str = "", color=ACCENT):
    """结论公式加框 + 中文小标注（右侧）。"""
    box = SurroundingRectangle(formula, color=color, buff=0.25)
    g = VGroup(formula, box)
    if label:
        g.add(ctext(label, size=24, color=color).next_to(box, RIGHT, buff=0.3))
    return g


def tag(formula_or_mob, text: str, direction=DOWN, color=INK, size: int = 24):
    """给公式/图元贴中文标签。"""
    return ctext(text, size=size, color=color).next_to(formula_or_mob,
                                                     direction, buff=0.25)


def vec_triangle(origin, v_a, v_ba, color_a=ACCENT, color_ba="#F78C6B",
                 color_b=INK, tip: float = 0.16):
    """速度矢量三角形：v_B = v_A + v_BA。

    v_a: 从 origin 出发的矢量；v_ba: 接在 v_a 末端的相对矢量。
    返回 (VGroup, pA_end, pB_end)。
    """
    o = np.asarray(origin, dtype=float)
    pA = o + np.asarray(v_a, dtype=float)
    pB = pA + np.asarray(v_ba, dtype=float)
    a1 = Arrow(o, pA, buff=0, color=color_a, tip_length=tip, stroke_width=5)
    a2 = Arrow(pA, pB, buff=0, color=color_ba, tip_length=tip, stroke_width=5)
    a3 = Arrow(o, pB, buff=0, color=color_b, tip_length=tip, stroke_width=5)
    return VGroup(a1, a2, a3), pA, pB
