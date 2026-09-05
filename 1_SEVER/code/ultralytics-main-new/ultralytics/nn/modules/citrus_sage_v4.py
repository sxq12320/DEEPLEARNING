# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""SAGE-v4: bounded adjacent-scale fusion and a direct stride-4 mask-detail route.

This is an independent adaptation of PIDNet's detail/context separation, ReZero's
small residual initialization, and gated CNNs. It is NOT a PID/Kalman controller,
an implementation of MambaOut, or a claim of closed-loop stability. See the
source/ablation report in docs/SAGE_V4_RESULTS_AND_DESIGN_20260903.md.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .head import Detect, Segment

__all__ = ("CitrusSAGEBoundedP3", "SAGEGatedStage", "SegmentCitrusSAGEV4")


class BoundedScaleUpdate(nn.Module):
    """Convex update between two projected tensors; not a calibrated uncertainty estimate."""

    def __init__(self, channels: int):
        super().__init__()
        self.gain = nn.Conv2d(channels * 3, 1, 1)
        nn.init.zeros_(self.gain.weight)
        nn.init.zeros_(self.gain.bias)

    def forward(self, measurement: torch.Tensor, prediction: torch.Tensor) -> torch.Tensor:
        if prediction.shape[-2:] != measurement.shape[-2:]:
            prediction = F.interpolate(prediction, size=measurement.shape[-2:], mode="nearest")
        difference = measurement - prediction
        gain = self.gain(torch.cat((measurement, prediction, difference.abs()), 1)).sigmoid()
        return prediction + gain * difference


class CitrusSAGEBoundedP3(nn.Module):
    """Correct only PAN-P3, preserving native P4/P5 and the complete pretrained PAN.

    At each scale the blend lies between its two inputs. That property does not
    imply bounded end-to-end activations through learned projections. Small
    channel gains are applied AFTER the final projection, which has no BN.
    """

    def __init__(self, in_channels, route_channels: int = 24, initial_scale: float = 0.01):
        super().__init__()
        if len(in_channels) != 4 or route_channels < 8 or not 0 < initial_scale <= 0.1:
            raise ValueError("Expected C3/C4/C5/PAN-P3 channels, width >= 8, and scale in (0, .1].")
        c3, c4, c5, out = in_channels
        self.measure3 = Conv(c3, route_channels, 1, act=False)
        self.measure4 = Conv(c4, route_channels, 1, act=False)
        self.measure5 = Conv(c5, route_channels, 1, act=False)
        self.predict4 = Conv(route_channels, route_channels, 1, act=False)
        self.predict3 = Conv(route_channels, route_channels, 1, act=False)
        self.update4 = BoundedScaleUpdate(route_channels)
        self.update3 = BoundedScaleUpdate(route_channels)
        self.output = nn.Conv2d(route_channels, out, 1, bias=False)
        self.scale = nn.Parameter(torch.full((1, out, 1, 1), initial_scale))

    def forward(self, features):
        c3, c4, c5, base = features
        local3 = self.measure3(c3)
        state4 = self.update4(self.measure4(c4), self.predict4(self.measure5(c5)))
        state3 = self.update3(local3, self.predict3(state4))
        return base + self.scale.tanh() * self.output(state3 - local3)


class _SAGEGatedBlock(nn.Module):
    """NCHW gated convolution; no sequence scan, tensor-layout switching or attention matrix."""

    def __init__(self, channels: int, expansion: float):
        super().__init__()
        hidden = max(16, int(channels * expansion))
        self.input = Conv(channels, hidden * 2, 1, act=False)
        self.spatial = Conv(hidden, hidden, 3, g=hidden, act=False)
        self.output = nn.Conv2d(hidden, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.full((1, channels, 1, 1), 0.01))

    def forward(self, x):
        value, gate = self.input(x).chunk(2, 1)
        update = self.output(self.spatial(value) * F.silu(gate))
        return x + self.scale.tanh() * update


class SAGEGatedStage(nn.Module):
    """Actual replacement of a CSP/C3k2 stage, not an appended attention block.

    Use as an OPTIONAL P4 ablation: unlike the conservative models, this stage
    cannot reuse the corresponding C3k2 checkpoint weights. BN/SiLU/3x3 and the
    narrow expansion deliberately differ from the original MambaOut block.
    """

    def __init__(self, c1: int, c2: int, n: int = 1, expansion: float = 1.0):
        super().__init__()
        if n < 1 or expansion <= 0:
            raise ValueError("n and expansion must be positive")
        self.project = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(_SAGEGatedBlock(c2, expansion) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.project(x))


class SegmentCitrusSAGEV4(Segment):
    """Native P3/P4/P5 detections with a narrow C2-to-prototype residual.

    No P2 detection tower is added. Auxiliary binary fields are independent and
    training-only: foreground, visible boundary, nearby-instance separator.
    The standard per-instance mask loss remains responsible for instance IDs.
    """

    def __init__(self, nc=80, nm=32, npr=256, detail_channels=16, structure_gain=0.0, reg_max=16, end2end=False, ch=()):
        if len(ch) != 4 or detail_channels < 8 or structure_gain < 0:
            raise ValueError("Expected P3/P4/P5/C2, detail width >= 8 and nonnegative structure gain")
        if end2end:
            raise ValueError("SAGE-v4 currently supports the standard YOLO11 one-to-many segmentation protocol only")
        super().__init__(nc, nm, npr, reg_max, end2end, tuple(ch[:3]))
        self.structure_gain = float(structure_gain)
        self.detail = nn.Sequential(
            Conv(ch[3], detail_channels, 1), Conv(detail_channels, detail_channels, 3, g=detail_channels)
        )
        self.detail_to_proto = nn.Conv2d(detail_channels, nm, 1, bias=False)
        self.detail_scale = nn.Parameter(torch.full((1, nm, 1, 1), 0.01))
        self.structure = nn.Conv2d(detail_channels, 3, 1) if structure_gain > 0 else None

    def forward(self, x):
        features = list(x[:3])
        detail = self.detail(x[3])
        proto = self.proto(features[0])
        if detail.shape[-2:] != proto.shape[-2:]:
            detail = F.interpolate(detail, size=proto.shape[-2:], mode="nearest")
        proto = proto + self.detail_scale.tanh() * self.detail_to_proto(detail)
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                if self.structure is not None:
                    preds["sage_structure"] = self.structure(detail)
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
