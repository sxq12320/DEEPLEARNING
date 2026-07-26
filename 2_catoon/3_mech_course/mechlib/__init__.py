# -*- coding: utf-8 -*-
"""mechlib — 机械原理公益课共享动画库（Manim CE 0.19）.

11 节课共用：中文文本/版式助手、公式分步推导、机构绘制原语（铰链/移动副/机架）、
运动学求解器（四杆/曲柄滑块）、渐开线齿轮轮廓、凸轮轮廓生成。
用法（各课 scenes.py 顶部）:
    import sys, os; sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from mechlib import *
"""

from .core import (
    CJK,
    ACCENT,
    GOOD,
    BAD,
    NOTE,
    ctext,
    title_bar,
    bullets,
    page_ref,
    formula_reveal,
    ground_hatch,
    pin_joint,
    fixed_pin,
    link_line,
    slider_block,
    FourBar,
    CrankSlider,
    involute_pts,
    gear_profile,
    cam_profile_knife,
    LessonScene,
)

__all__ = [
    "CJK", "ACCENT", "GOOD", "BAD", "NOTE",
    "ctext", "title_bar", "bullets", "page_ref", "formula_reveal",
    "ground_hatch", "pin_joint", "fixed_pin", "link_line", "slider_block",
    "FourBar", "CrankSlider", "involute_pts", "gear_profile", "cam_profile_knife",
    "LessonScene",
]
