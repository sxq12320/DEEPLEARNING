# -*- coding: utf-8 -*-
"""mechlib v2 — 《机械原理》动画课程共享库.

用法（每个 L*/scenes.py 开头）：
    import os, sys
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    from manim import *
    from mechlib import *
"""
from .curves import *          # noqa: F401,F403
from .formulas import *        # noqa: F401,F403
from .mechanisms import *      # noqa: F401,F403
from .primitives import *      # noqa: F401,F403
from .solvers import *         # noqa: F401,F403
from .style import *           # noqa: F401,F403
