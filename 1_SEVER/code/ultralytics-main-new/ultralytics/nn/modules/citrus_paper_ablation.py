# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Controlled decoder ablation without changing any historical defaults."""

import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v12 import SegmentCitrusEV12

__all__ = ("SegmentCitrusPaperAblation",)


class _ZeroQuality(nn.Module):
    """Parameter-free placeholder; quality is NOT used to score this control."""

    def forward(self, x):
        return x.new_zeros((x.shape[0], 1, x.shape[2], x.shape[3]))


class SegmentCitrusPaperAblation(SegmentCitrusEV12):
    """Remove learned phase updates but preserve the stride-2 mask output grid."""

    def __init__(
        self, nc=80, nm=32, npr=256, detail_channels=16, persistent=False,
        legacy_transport=True, factorized_box=False, assignment_mix=0.0,
        ring_gain=0.0, tiny_gain=0.25, boundary_gain=0.5, neighbor_gain=0.25,
        route=2, difference=True, phase_refinement=True, quality_enabled=True, reg_max=16,
        end2end=False, ch=(),
    ):
        super().__init__(
            nc, nm, npr, detail_channels, persistent, legacy_transport, factorized_box,
            assignment_mix, ring_gain, tiny_gain, boundary_gain, neighbor_gain,
            route, difference, reg_max, end2end, ch,
        )
        self.paper_phase_refinement = bool(phase_refinement)
        if not self.paper_phase_refinement:
            del self.fine_detail
            del self.fine_to_proto
            del self.fine_scale
        if not quality_enabled:
            self.quality_gain = 0.0
            self.quality_calibration = False
            self.quality_predictor = nn.ModuleList(_ZeroQuality() for _ in range(self.nl))

    def refine_fine_proto(self, proto, mask_detail, stem):
        if self.paper_phase_refinement:
            return super().refine_fine_proto(proto, mask_detail, stem)
        return F.interpolate(proto, scale_factor=2, mode="bilinear", align_corners=False)
