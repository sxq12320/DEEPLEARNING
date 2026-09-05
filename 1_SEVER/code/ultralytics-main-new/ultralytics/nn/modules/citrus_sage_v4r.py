# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Reconstructed SAGE-v4: semantic-guided, one-step mask-feature correction.

Independent adaptation of semantic shape guidance (Gated-SCNN/RefineMask) and
back-projection error correction (DBPN). Not their original implementation, a
PID controller, or a stability guarantee. The asymmetric neck is explicit in
SAGE40--48 YAMLs, not an additional pyramid hidden in this module.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .head import Detect, Segment

__all__ = ("SegmentCitrusSAGEV4R",)


class SAGEMaskCorrection(nn.Module):
    """A fixed-width P2 branch; optional semantic measurement and ONE back-projection."""

    def __init__(self, c_detail: int, c_semantic: int, channels: int = 16, mode: str = "semantic"):
        super().__init__()
        if mode not in {"direct", "semantic", "reproject"} or channels < 8:
            raise ValueError("Use direct/semantic/reproject and at least eight detail channels")
        self.mode = mode
        self.detail = nn.Sequential(Conv(c_detail, channels, 1), Conv(channels, channels, 3, g=channels))
        if mode != "direct":
            self.semantic = Conv(c_semantic, channels, 1, act=False)
            self.gate = nn.Conv2d(3 * channels, 1, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.zeros_(self.gate.bias)
        if mode == "reproject":
            self.down_project = nn.Conv2d(channels, channels, 1, bias=False)
            self.up_error = nn.Conv2d(channels, channels, 1, bias=False)
            nn.init.dirac_(self.down_project.weight)
            nn.init.dirac_(self.up_error.weight)
            self.correction_scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, c2: torch.Tensor, p3: torch.Tensor) -> torch.Tensor:
        detail = self.detail(c2)
        if self.mode == "direct":
            return detail
        semantic = self.semantic(p3)
        estimate = F.interpolate(semantic, size=detail.shape[-2:], mode="nearest")
        gain = self.gate(torch.cat((detail, estimate, (detail - estimate).abs()), 1)).sigmoid()
        refined = estimate + gain * (detail - estimate)
        if self.mode == "reproject":
            # Explicit low-resolution discrepancy, not a repeated backbone pass.
            projected = self.down_project(F.adaptive_avg_pool2d(refined, semantic.shape[-2:]))
            error = self.up_error(semantic - projected)
            error = F.interpolate(error, size=detail.shape[-2:], mode="nearest")
            refined = refined + self.correction_scale.tanh() * error
        return refined


class SegmentCitrusSAGEV4R(Segment):
    """Standard detection/coefficient towers plus semantic-conditioned stride-4 prototypes.

    Boundary and neighbor weights configure a criterion acting on the SAME
    per-instance mask logits, not separate semantic auxiliary heads. These
    weights add no inference parameters. P2 has no dense detection tower.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        mode="semantic",
        boundary_gain=0.0,
        neighbor_gain=0.0,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if len(ch) != 4 or end2end:
            raise ValueError("SAGE-v4r requires P3/P4/C5/C2 and the YOLO11 one-to-many head")
        if min(boundary_gain, neighbor_gain) < 0:
            raise ValueError("Geometry gains must be nonnegative")
        super().__init__(nc, nm, npr, reg_max, end2end, tuple(ch[:3]))
        self.boundary_gain, self.neighbor_gain = float(boundary_gain), float(neighbor_gain)
        self.refiner = SAGEMaskCorrection(ch[3], ch[0], detail_channels, mode)
        self.detail_to_proto = nn.Conv2d(detail_channels, nm, 1, bias=False)
        self.detail_scale = nn.Parameter(torch.full((1, nm, 1, 1), 0.01))

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        proto = self.proto(features[0])
        if detail.shape[-2:] != proto.shape[-2:]:
            detail = F.interpolate(detail, size=proto.shape[-2:], mode="nearest")
        proto = proto + self.detail_scale.tanh() * self.detail_to_proto(detail)
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
