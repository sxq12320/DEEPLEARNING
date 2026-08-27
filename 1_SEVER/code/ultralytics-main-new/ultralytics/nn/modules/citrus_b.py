# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Evidence-grounded dual-resolution blocks for CitrusB instance segmentation.

The implementation adapts three verified ideas while keeping the active
Ultralytics fork free of MMCV, custom CUDA kernels, and Mamba:

* Lite-HRNet's stride-one ShuffleUnit for an inexpensive persistent detail path.
* PIDNet's PagFM for guarded semantic-to-detail exchange.
* DDRNet's repeated high/low-resolution exchange, with reversible pixel
  rearrangement before learned channel projection in the detail-to-pyramid direction.

These blocks are deliberately small. The architectural contribution is the
long-lived stride-4 path and its repeated exchanges, not a stack of attention
operators inside every YOLO block.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ("CitrusDetailInject", "CitrusPagFM", "LiteHRDetailBlock")


def _channel_shuffle(x: torch.Tensor, groups: int = 2) -> torch.Tensor:
    """Shuffle channels exactly as used by ShuffleNetV2/Lite-HRNet."""
    batch, channels, height, width = x.shape
    if channels % groups:
        raise ValueError(f"Channel count {channels} must be divisible by groups={groups}")
    return x.view(batch, groups, channels // groups, height, width).transpose(1, 2).reshape(x.shape)


class _ConvBNAct(nn.Sequential):
    """Minimal convolution-BN-SiLU helper local to the evidence blocks."""

    def __init__(
        self,
        c1: int,
        c2: int,
        kernel_size: int = 1,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        activate: bool = True,
    ):
        layers: list[nn.Module] = [
            nn.Conv2d(c1, c2, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(c2),
        ]
        if activate:
            layers.append(nn.SiLU(inplace=True))
        super().__init__(*layers)


class _SpatialWeighting(nn.Module):
    """Dependency-free port of Lite-HRNet's spatial weighting."""

    def __init__(self, channels: int, ratio: int = 4):
        super().__init__()
        hidden = max(channels // ratio, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply sample-dependent channel weights to the transformed half."""
        weights = torch.sigmoid(self.expand(F.silu(self.reduce(self.pool(x)))))
        return x * weights


class LiteHRDetailBlock(nn.Module):
    """Stride-one Lite-HRNet/ShuffleNetV2 unit for a persistent P2 detail stream.

    Half of the channels bypass all convolutions while the other half receives a
    pointwise-depthwise-pointwise transform and spatial weighting. This preserves
    fine pixels with substantially less P2 computation than a full C3 block.
    """

    def __init__(self, c1: int, c2: int):
        super().__init__()
        if c2 % 2:
            raise ValueError(f"LiteHRDetailBlock requires an even output width, got {c2}")
        self.project = _ConvBNAct(c1, c2) if c1 != c2 else nn.Identity()
        branch = c2 // 2
        self.branch = nn.Sequential(
            _ConvBNAct(branch, branch, 1),
            _ConvBNAct(branch, branch, 3, padding=1, groups=branch, activate=False),
            _ConvBNAct(branch, branch, 1),
        )
        self.weighting = _SpatialWeighting(branch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Update half of the channels and shuffle the two paths."""
        x = self.project(x)
        bypass, transformed = x.chunk(2, dim=1)
        transformed = self.weighting(self.branch(transformed))
        return _channel_shuffle(torch.cat((bypass, transformed), dim=1), 2)


class CitrusPagFM(nn.Module):
    """Pixel-attention-guided semantic-to-detail fusion adapted from PIDNet PagFM.

    The first input is the high-resolution detail stream and the second is a
    lower-resolution semantic tensor. Projected feature agreement determines how
    much semantic context is accepted at every P2 location, which prevents the
    low-frequency orchard background from uniformly overwhelming fruit edges.
    """

    def __init__(self, channels: list[int] | tuple[int, int], mid_channels: int = 16):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"CitrusPagFM expects [detail, semantic] channels, got {channels}")
        detail_channels, semantic_channels = channels
        mid_channels = max(int(mid_channels), 4)
        self.detail_key = _ConvBNAct(detail_channels, mid_channels, activate=False)
        self.semantic_query = _ConvBNAct(semantic_channels, mid_channels, activate=False)
        self.semantic_value = _ConvBNAct(semantic_channels, detail_channels, activate=False)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Blend aligned semantic values into P2 according to pixel similarity."""
        detail, semantic = features
        size = detail.shape[-2:]
        query = F.interpolate(self.semantic_query(semantic), size=size, mode="bilinear", align_corners=False)
        key = self.detail_key(detail)
        similarity = torch.sigmoid((key * query).sum(dim=1, keepdim=True))
        value = F.interpolate(self.semantic_value(semantic), size=size, mode="bilinear", align_corners=False)
        return (1.0 - similarity) * detail + similarity * value


class CitrusDetailInject(nn.Module):
    """Inject a P2 detail stream into a lower-resolution pyramid tensor.

    PixelUnshuffle first performs a reversible 2x spatial rearrangement at every
    step, after which a learned 1x1 projection compresses the rearranged channels.
    A high-pass gate admits the projected signal primarily near boundaries. The
    zero-initialized residual scale keeps the original pretrained pyramid function
    intact at initialization while allowing the detail residual to learn.
    """

    def __init__(self, channels: list[int] | tuple[int, int], steps: int = 1):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"CitrusDetailInject expects [pyramid, detail] channels, got {channels}")
        pyramid_channels, detail_channels = channels
        self.steps = int(steps)
        if self.steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        rearrange: list[nn.Module] = []
        current = detail_channels
        for index in range(self.steps):
            output = pyramid_channels if index == self.steps - 1 else min(pyramid_channels, max(current, 32))
            rearrange.extend((nn.PixelUnshuffle(2), _ConvBNAct(current * 4, output, 1)))
            current = output
        self.rearrange = nn.Sequential(*rearrange)
        self.boundary_gate = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 3, padding=1, groups=detail_channels, bias=False),
            nn.BatchNorm2d(detail_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(detail_channels, 1, 1),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1, pyramid_channels, 1, 1))

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Return the pretrained pyramid tensor plus boundary-gated P2 detail."""
        pyramid, detail = features
        injected = self.rearrange(detail)
        high_frequency = detail - F.avg_pool2d(detail, 5, stride=1, padding=2)
        gate = torch.sigmoid(self.boundary_gate(high_frequency))
        gate = F.interpolate(gate, size=pyramid.shape[-2:], mode="bilinear", align_corners=False)
        if injected.shape[-2:] != pyramid.shape[-2:]:
            raise ValueError(
                f"CitrusDetailInject steps={self.steps} produced {injected.shape[-2:]}, expected {pyramid.shape[-2:]}"
            )
        return pyramid + torch.tanh(self.residual_scale) * gate * injected
