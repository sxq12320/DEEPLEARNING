# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Minimal C2-to-P3 evidence relay with an otherwise native YOLO11 Segment head.

This is an unvalidated task-specific combination of existing primitives, not a
claim of inventing space-to-depth, gating or detail relay. See the R_MINIMAL docs.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .head import Segment

__all__ = ("SegmentCitrusMinimal",)


class ShallowEvidenceRelay(nn.Module):
    """Read a narrow C2 feature once and gate its residual at the P3 prediction input.

    A spatial sigmoid can suppress all packed channels at a location. It is not
    calibrated objectness. No fixed high-pass filter discards the DC component.
    Zero channel gain preserves the native function at initialization; internal
    relay gradients start after gain updates, rather than on the first step.
    """

    def __init__(self, c_detail, c_prediction, width=8):
        super().__init__()
        if width < 1:
            raise ValueError("Relay width must be positive")
        self.detail = Conv(c_detail, width, 1)
        self.context = Conv(c_prediction, width, 1)
        self.gate = nn.Conv2d(5 * width, 1, 1)
        self.project = nn.Conv2d(4 * width, c_prediction, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(1, c_prediction, 1, 1))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, detail, prediction):
        """Return P3 with a learned residual; inputs must have an exact 2:1 spatial ratio."""
        packed = F.pixel_unshuffle(self.detail(detail), 2)
        context = self.context(prediction)
        gate = torch.sigmoid(self.gate(torch.cat((packed, context), dim=1)))
        return prediction + 0.1 * self.gain.tanh() * gate * self.project(packed)


class SegmentCitrusMinimal(Segment):
    """Keep native candidate towers, prototype decoder, state keys and segmentation loss.

    Inputs are [P3, P4, P5, C2]. Only the first three create candidates. C2 is
    read directly from the original backbone, with no persistent detail stream.
    """

    def __init__(self, nc=80, nm=32, npr=256, relay_width=8, reg_max=16, end2end=False, ch=()):
        if len(ch) != 4:
            raise ValueError("SegmentCitrusMinimal requires [P3, P4, P5, C2]")
        if end2end:
            raise ValueError("R_MINIMAL is defined for the native YOLO11 non-end2end protocol")
        super().__init__(nc, nm, npr, reg_max, end2end, ch[:3])
        self.relay = ShallowEvidenceRelay(ch[3], ch[0], relay_width)

    def forward(self, x):
        features = list(x[:3])
        features[0] = self.relay(x[3], features[0])
        return super().forward(features)
