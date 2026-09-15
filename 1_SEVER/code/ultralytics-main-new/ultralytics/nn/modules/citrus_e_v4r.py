"""Evidence-led E V4 reconstruction: narrow detail correction and bounded mask ranking.

These are testable adaptations, not reproductions of Gated-SCNN or Mask Scoring
R-CNN and not a dynamic control system or stability guarantee. No recurrent pass,
new dense P2 detection tower, FFT, custom CUDA, or image-dependent Python loop.
"""

import math

import torch
import torch.nn as nn

from .citrus_e_v3 import SegmentCitrusEV3Quality
from .citrus_e_v4 import EV4PhaseLead
from .citrus_sage_v4r import SAGEMaskCorrection
from .citrus_sage_v5 import SegmentCitrusSAGEV5

__all__ = ("SegmentCitrusEV4Detail", "SegmentCitrusEV4Quality")


class EV4BoundedDetail(EV4PhaseLead):
    """A zero-initialized local residual on the 16-channel semantic-conditioned P2 path."""

    def forward(self, x):
        return x + self.gain.tanh() * self.compensate(x - self.low(x))


class EV4DetailCorrection(SAGEMaskCorrection):
    """Preserve the existing refiner's modules, values and state-dict key paths."""

    def __init__(self, base):
        nn.Module.__init__(self)
        if base.mode != "semantic":
            raise ValueError("E V4R correction requires the semantic P2 refiner")
        self.mode = base.mode
        for name, module in base.named_children():
            self.add_module(name, module)
        channels = base.detail[0].conv.out_channels
        self.local_correction = EV4BoundedDetail(channels)

    def forward(self, c2, p3):
        return self.local_correction(super().forward(c2, p3))


class SegmentCitrusEV4Detail(SegmentCitrusSAGEV5):
    """Share the corrected narrow P2 feature with both the P3 relay and mask prototypes."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.refiner = EV4DetailCorrection(self.refiner)


class SegmentCitrusEV4Quality(SegmentCitrusEV3Quality):
    """Detached mask-IoU supervision with bounded score attenuation.

    floor=0 recovers V3's multiplication; floor=1 disables score attenuation while
    retaining the same trained weights. The default 0.5 is a predeclared experiment,
    not a fitted optimum. It cannot recover candidates that were never proposed.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        relay=True,
        late_proto=False,
        boundary_gain=0.0,
        neighbor_gain=0.0,
        quality_gain=1.0,
        detail_correction=False,
        quality_floor=0.5,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        super().__init__(
            nc,
            nm,
            npr,
            detail_channels,
            relay,
            late_proto,
            boundary_gain,
            neighbor_gain,
            quality_gain,
            reg_max,
            end2end,
            ch,
        )
        if not math.isfinite(quality_floor) or not 0 <= quality_floor <= 1:
            raise ValueError("quality_floor must be finite and in [0, 1]")
        self.quality_floor = float(quality_floor)
        if detail_correction:
            self.refiner = EV4DetailCorrection(self.refiner)

    def _inference(self, preds):
        boxes = self._get_decode_boxes(preds)
        scores = preds["scores"].sigmoid()
        if self.quality_calibration:
            quality = preds["ev3_quality"].sigmoid()
            scores = scores * (self.quality_floor + (1.0 - self.quality_floor) * quality)
        return torch.cat((boxes, scores, preds["mask_coefficient"]), 1)
