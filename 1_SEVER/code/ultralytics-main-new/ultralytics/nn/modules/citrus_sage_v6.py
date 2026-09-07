# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""SAGE-v6 persistent-detail backbone and budgeted cross-resolution exchange.

Independent task adaptation of dual-resolution representation/semantic guidance,
not a reproduction of DDRNet, Lite-HRNet or PIDNet. No PID or stability claim.
Topology is explicit in YAML; these modules do not hide an extra pyramid.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("SAGEV6Stage", "SAGEV6Exchange")


class SAGEV6Residual(nn.Module):
    """Narrow dense spatial bottleneck, with no attention/unfold/custom operator."""

    def __init__(self, channels):
        super().__init__()
        hidden = max(8, channels // 4)
        self.reduce = Conv(channels, hidden, 1)
        self.spatial = Conv(hidden, hidden, 3)
        self.expand = Conv(hidden, channels, 1, act=False)
        self.act = nn.SiLU()

    def forward(self, x):
        return self.act(x + self.expand(self.spatial(self.reduce(x))))


class SAGEV6Stage(nn.Module):
    """Residual stage replacing C3k2 split/concatenate topology; parser owns repeats."""

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.project = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(SAGEV6Residual(c2) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.project(x))


class SAGEV6Exchange(nn.Module):
    """Compress before upsampling; pool before widening on downward exchange.

    The add/select pair has the same identity-preserving residual and initial
    gain. Select starts at an effective gate of one (2*sigmoid(0)), so it does
    not silently halve cross-scale information at initialization.
    """

    def __init__(self, channels, c2, mode="add"):
        super().__init__()
        if len(channels) != 2 or mode not in {"add", "select"}:
            raise ValueError("SAGEV6Exchange requires two input features and add/select mode")
        self.mode = mode
        self.anchor = Conv(channels[0], c2, 1, act=False) if channels[0] != c2 else nn.Identity()
        self.source = Conv(channels[1], c2, 1, act=False)
        self.gain = nn.Parameter(torch.full((1, c2, 1, 1), 0.1))
        if mode == "select":
            # One spatial gate, not a quadratic pixel-attention matrix.
            self.gate = nn.Conv2d(2 * c2, 1, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)

    def forward(self, inputs):
        anchor = self.anchor(inputs[0])
        source = inputs[1]
        target = anchor.shape[-2:]
        if source.shape[-2:] != target and source.shape[-2] >= target[0] and source.shape[-1] >= target[1]:
            # Do not expand a narrow stride-4 tensor to 128 channels before pooling.
            source = F.adaptive_avg_pool2d(source, target)
        source = self.source(source)
        if source.shape[-2:] != target:
            source = F.interpolate(source, size=target, mode="nearest")
        if self.mode == "select":
            source = source * (2.0 * self.gate(torch.cat((anchor, source), 1)).sigmoid())
        return anchor + self.gain.tanh() * source
