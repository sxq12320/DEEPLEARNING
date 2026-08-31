# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Pretrained-compatible structure-preserving components for the G_0830 citrus series.

The modules in this file form one coherent architecture rather than a collection
of interchangeable attention blocks.  A narrow stride-4 shape representation is
kept alive through the whole backbone, semantic stages decide which local
structures are useful, and neck fusion explicitly aligns low-frequency semantics
with high-frequency detail.  Every exchange is zero-initialized so a YAML loaded
from ``yolo11n-seg.pt`` starts from the pretrained YOLO function instead of
destroying it with randomly initialized residuals.

The implementation is a compact, independently written adaptation of the design
principles in Lite-HRNet (persistent high resolution), PIDNet (detail/semantic
gating), FreqFusion and SFM (frequency-aware sampling/fusion), and RepViT
(re-parameterizable depthwise token mixing).  It does not copy code from those
repositories.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("CitrusBilateralExchange", "CitrusFrequencyAlignedConcat", "CitrusRepMixerStage")


class _SqueezeExcite(nn.Module):
    """Small squeeze-excitation gate used only inside the optional deep mixer."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.reduce = nn.Conv2d(channels, hidden, 1)
        self.expand = nn.Conv2d(hidden, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reweight channels using global context."""
        gate = F.adaptive_avg_pool2d(x, 1)
        gate = F.silu(self.reduce(gate), inplace=True)
        return x * self.expand(gate).sigmoid()


class _RepDepthwiseMixer(nn.Module):
    """RepViT-style local mixer with a residual channel MLP.

    The 3x3 depthwise, 1x1 depthwise and identity branches can be fused for a
    deployment implementation later.  Keeping the explicit branches during
    training avoids adding a custom CUDA dependency.
    """

    def __init__(self, channels: int, expansion: float = 2.0, use_se: bool = False):
        super().__init__()
        hidden = max(channels, int(round(channels * expansion)))
        self.dw3 = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.dw1 = nn.Conv2d(channels, channels, 1, groups=channels, bias=False)
        self.spatial_norm = nn.BatchNorm2d(channels)
        self.spatial_gate = _SqueezeExcite(channels) if use_se else nn.Identity()
        self.expand = Conv(channels, hidden, k=1)
        self.project = Conv(hidden, channels, k=1, act=False)
        nn.init.zeros_(self.project.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix local tokens and channels while retaining an identity start."""
        spatial = self.spatial_norm(self.dw3(x) + self.dw1(x) + x)
        spatial = self.spatial_gate(spatial)
        return spatial + self.project(self.expand(spatial))


class CitrusRepMixerStage(nn.Module):
    """Optional non-CSP deep stage for testing C3k2 replacement independently."""

    def __init__(
        self,
        c1: int,
        c2: int,
        repeats: int = 2,
        expansion: float = 2.0,
        use_se: bool = True,
    ):
        super().__init__()
        if repeats < 1:
            raise ValueError(f"repeats must be positive, got {repeats}")
        self.input_project = Conv(c1, c2, k=1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(_RepDepthwiseMixer(c2, expansion, use_se) for _ in range(repeats)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the mixed stage feature."""
        return self.blocks(self.input_project(x))


class CitrusBilateralExchange(nn.Module):
    """Bidirectionally exchange persistent P2 shape and one semantic feature.

    The first input is always the stride-4 detail tensor; the second is a P3, P4
    or P5 semantic tensor.  High-pass magnitude supplies colour-insensitive shape
    evidence, while semantic/detail agreement rejects leaf texture.  Both output
    residual scales start at zero, making the module an exact identity at model
    initialization and preserving mapped YOLO11 pretrained weights.
    """

    def __init__(self, channels: list[int] | tuple[int, int], ratio: int, gate_channels: int = 16):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"CitrusBilateralExchange expects [detail, semantic] channels, got {channels}")
        if ratio not in {2, 4, 8}:
            raise ValueError(f"P2-to-semantic ratio must be 2, 4, or 8, got {ratio}")
        detail_channels, semantic_channels = (int(value) for value in channels)
        gate_channels = max(8, int(gate_channels))
        self.ratio = int(ratio)

        self.detail_key = Conv(detail_channels, gate_channels, k=1, act=False)
        self.semantic_key = Conv(semantic_channels, gate_channels, k=1, act=False)
        self.detail_update = nn.Sequential(
            DWConv(detail_channels, detail_channels, k=3),
            Conv(detail_channels, detail_channels, k=1, act=False),
        )
        self.shape_to_semantic = nn.Sequential(
            Conv(detail_channels, semantic_channels, k=1),
            DWConv(semantic_channels, semantic_channels, k=3),
        )
        self.detail_scale = nn.Parameter(torch.zeros(1, detail_channels, 1, 1))
        self.semantic_scale = nn.Parameter(torch.zeros(1, semantic_channels, 1, 1))

    @staticmethod
    def _high_pass(x: torch.Tensor) -> torch.Tensor:
        """Return local AC content without assuming that citrus is dark or bright."""
        return x - F.avg_pool2d(x, 3, stride=1, padding=1)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> list[torch.Tensor]:
        """Return updated ``[P2 detail, semantic]`` tensors."""
        if len(features) != 2:
            raise ValueError(f"Expected two input tensors, received {len(features)}")
        detail, semantic = features
        detail_high = self._high_pass(detail)

        detail_key = self.detail_key(detail_high)
        semantic_key = F.interpolate(
            self.semantic_key(semantic), size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        detail_gate = torch.sigmoid((detail_key * semantic_key).sum(1, keepdim=True) / math.sqrt(detail_key.shape[1]))
        detail_out = detail + torch.tanh(self.detail_scale) * detail_gate * self.detail_update(detail_high)

        pooled_key = F.adaptive_avg_pool2d(self.detail_key(detail_out), semantic.shape[-2:])
        semantic_key_native = self.semantic_key(semantic)
        semantic_gate = torch.sigmoid(
            (pooled_key * semantic_key_native).sum(1, keepdim=True) / math.sqrt(pooled_key.shape[1])
        )
        # Max pooling of absolute AC response retains narrow edges that average
        # pooling would cancel, but the learned projection still carries sign and
        # channel information from the updated detail tensor.
        edge_gate = F.adaptive_max_pool2d(detail_high.abs().mean(1, keepdim=True), semantic.shape[-2:])
        edge_gate = edge_gate / edge_gate.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        pooled_detail = F.adaptive_avg_pool2d(detail_out, semantic.shape[-2:]) * (0.5 + edge_gate)
        semantic_update = self.shape_to_semantic(pooled_detail)
        semantic_out = semantic + torch.tanh(self.semantic_scale) * semantic_gate * semantic_update
        return [detail_out, semantic_out]


class CitrusFrequencyAlignedConcat(nn.Module):
    """Identity-initialized frequency-aware replacement for two-input Concat.

    ``topdown`` treats the first input as upsampled semantics and the second as a
    lateral detail feature.  ``bottomup`` reverses those roles.  The output keeps
    exactly the original input order and channel count, so the following C3k2 can
    reuse its official pretrained weights.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int],
        dimension: int = 1,
        direction: str = "topdown",
        gate_channels: int = 16,
    ):
        super().__init__()
        if len(channels) != 2:
            raise ValueError(f"CitrusFrequencyAlignedConcat expects two channel widths, got {channels}")
        if direction not in {"topdown", "bottomup"}:
            raise ValueError(f"direction must be 'topdown' or 'bottomup', got {direction!r}")
        first_channels, second_channels = (int(value) for value in channels)
        semantic_channels, detail_channels = (
            (first_channels, second_channels) if direction == "topdown" else (second_channels, first_channels)
        )
        gate_channels = max(8, int(gate_channels))
        self.dimension = int(dimension)
        self.direction = direction
        self.semantic_key = Conv(semantic_channels, gate_channels, k=1, act=False)
        self.detail_key = Conv(detail_channels, gate_channels, k=1, act=False)
        # Depthwise-only refiners keep the frequency alignment cheaper than a
        # second PAN bottleneck; channel mixing remains in the pretrained C3k2
        # immediately following this identity-initialized fusion.
        self.semantic_refine = DWConv(semantic_channels, semantic_channels, k=3, act=False)
        self.detail_refine = DWConv(detail_channels, detail_channels, k=3, act=False)
        self.semantic_scale = nn.Parameter(torch.zeros(1, semantic_channels, 1, 1))
        self.detail_scale = nn.Parameter(torch.zeros(1, detail_channels, 1, 1))

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
        """Align semantic consistency and detail boundaries before concatenation."""
        if len(features) != 2:
            raise ValueError(f"Expected two input tensors, received {len(features)}")
        first, second = features
        semantic, detail = (first, second) if self.direction == "topdown" else (second, first)
        if semantic.shape[-2:] != detail.shape[-2:]:
            raise ValueError(f"Fusion inputs must share spatial size, got {semantic.shape[-2:]} and {detail.shape[-2:]}")

        semantic_low = F.avg_pool2d(semantic, 3, stride=1, padding=1)
        detail_high = detail - F.avg_pool2d(detail, 3, stride=1, padding=1)
        semantic_key = self.semantic_key(semantic_low)
        detail_key = self.detail_key(detail_high)
        agreement = torch.sigmoid((semantic_key * detail_key).sum(1, keepdim=True) / math.sqrt(semantic_key.shape[1]))
        semantic = semantic + torch.tanh(self.semantic_scale) * agreement * self.semantic_refine(semantic_low)
        detail = detail + torch.tanh(self.detail_scale) * agreement * self.detail_refine(detail_high)

        ordered = (semantic, detail) if self.direction == "topdown" else (detail, semantic)
        return torch.cat(ordered, dim=self.dimension)
