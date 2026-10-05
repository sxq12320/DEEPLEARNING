# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J V2 glance head: standard YOLO segment candidates plus a small-fruit density map.

The density map answers only "where would a closer look help": centres of fruits whose visible mask is
smaller than 32 x 32 input pixels in the CURRENT view. It never filters glance candidates; it is consumed
after inference by the gaze evaluator (window selection and optional score gating).
Design adapted from CenterNet heatmap supervision and ESOD/QueryDet-style sparse high-resolution search;
not a reproduction of either complete method.
"""

import math

import torch
import torch.nn as nn

from .conv import Conv
from .head import Segment

__all__ = ("SegmentCitrusJ2",)


class SegmentCitrusJ2(Segment):
    """Segment head with an optional P3 small-fruit density branch.

    Args:
        nc (int): Number of classes.
        nm (int): Number of mask coefficients.
        npr (int): Prototype channels.
        density (bool): Build the density branch.
        detach (bool): Read P3 without back-propagating density gradients into the shared network.
        gain (float): Effective density loss weight, read by J2SegmentationLoss (not multiplied by box gain).
        reg_max (int): DFL bins.
        end2end (bool): Must be False (one-to-many assignment only).
        ch (tuple): Input channels of P3/P4/P5.
    """

    prior = 0.01

    def __init__(self, nc=1, nm=32, npr=256, density=True, detach=False, gain=0.5, reg_max=16, end2end=False, ch=()):
        if end2end or len(ch) != 3:
            raise ValueError("SegmentCitrusJ2 expects P3/P4/P5 and one-to-many assignment")
        if gain < 0:
            raise ValueError("Density gain must be nonnegative")
        super().__init__(nc, nm, npr, reg_max, end2end, tuple(ch))
        self.density_enabled, self.density_detach = bool(density), bool(detach)
        self.density_gain = float(gain) if density else 0.0
        if self.density_enabled:
            hidden = max(ch[0] // 2, 16)
            self.density = nn.Sequential(Conv(ch[0], hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, 1, 1))
            nn.init.constant_(self.density[-1].bias, -math.log((1 - self.prior) / self.prior))

    def forward(self, x):
        """Return the standard Segment contract; density is an extra dict entry ``j2_density`` (logits)."""
        density = self.density(x[0].detach() if self.density_detach else x[0]) if self.density_enabled else None
        outputs = super().forward(x)
        if density is not None:
            preds = outputs if self.training else outputs[1]
            if isinstance(preds, dict):
                preds["j2_density"] = density
        return outputs

    @staticmethod
    def density_probability(preds):
        """Sigmoid density from a raw prediction dict, or None for heads without the branch."""
        logits = preds.get("j2_density") if isinstance(preds, dict) else None
        return None if logits is None else torch.sigmoid(logits.float())
