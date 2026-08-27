# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Task-specific building blocks for topology-aware immature-citrus segmentation.

The blocks keep the original YOLO11 tensor shapes and are zero-residual at
initialization. This preserves the pretrained baseline function while the new
branches learn high-resolution detail, cross-scale weighting, and global context.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import SPPF, RepVGGDW
from .conv import Conv

__all__ = (
    "CitrusBoundaryFusion",
    "CitrusScaleFusion",
    "CitrusTrainAux",
    "SPPFLSKAResidual",
    "SPPFRepContext",
)


class LargeSeparableKernelAttention(nn.Module):
    """Official LSKA-23 operator adapted from Lau et al. without external dependencies."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv0h = nn.Conv2d(channels, channels, (1, 5), padding=(0, 2), groups=channels)
        self.conv0v = nn.Conv2d(channels, channels, (5, 1), padding=(2, 0), groups=channels)
        self.conv_spatial_h = nn.Conv2d(
            channels, channels, (1, 7), padding=(0, 9), dilation=(1, 3), groups=channels
        )
        self.conv_spatial_v = nn.Conv2d(
            channels, channels, (7, 1), padding=(9, 0), dilation=(3, 1), groups=channels
        )
        self.project = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the factorized 23x23-equivalent spatial attention."""
        attention = self.conv0h(x)
        attention = self.conv0v(attention)
        attention = self.conv_spatial_h(attention)
        attention = self.conv_spatial_v(attention)
        return x * self.project(attention)


class SPPFLSKAResidual(SPPF):
    """Pretraining-compatible SPPF followed by a zero-initialized LSKA residual.

    The inherited ``cv1``, ``cv2``, and max-pooling keys exactly match YOLO11's
    original SPPF. Therefore replacing layer 9 does not discard its pretrained
    parameters. LSKA is used only at P5, where its large effective kernel supplies
    orchard-level context at modest cost.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__(c1, c2, k)
        self.context = LargeSeparableKernelAttention(c2)
        self.context_scale = nn.Parameter(torch.zeros(1, c2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the pretrained SPPF path and learn context as a residual."""
        y = super().forward(x)
        return y + torch.tanh(self.context_scale) * self.context(y)


class SPPFRepContext(SPPF):
    """Pretraining-compatible SPPF with deploy-time reparameterized 7x7 context.

    The training graph uses the official Ultralytics ``RepVGGDW`` 7x7 and 3x3
    depthwise branches. ``model.fuse()`` merges them into one 7x7 depthwise
    convolution, avoiding the multiple-kernel inference overhead that made
    several earlier citrus blocks slow despite modest FLOPs.
    """

    def __init__(self, c1: int, c2: int, k: int = 5):
        super().__init__(c1, c2, k)
        self.context = RepVGGDW(c2)
        self.context_scale = nn.Parameter(torch.zeros(1, c2, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run SPPF and learn the reparameterizable context as a residual."""
        y = super().forward(x)
        return y + torch.tanh(self.context_scale) * self.context(y)


class CitrusTrainAux(nn.Module):
    """Training-only P2/P3 supervision for citrus detail and camouflage cues.

    This branch does not alter the inference feature path. It teaches the shared
    backbone features to retain (1) visible-mask boundaries, (2) sparse tiny-fruit
    centers, and (3) fruit-versus-nearby-leaf contrast. The owning segmentation
    head skips the module entirely in evaluation/export mode.
    """

    def __init__(self, p2_channels: int, p3_channels: int):
        super().__init__()
        hidden = max(16, min(48, p2_channels))
        self.p2_embed = Conv(p2_channels, hidden, 3)
        self.p3_embed = Conv(p3_channels, hidden, 1)
        self.boundary_predictor = nn.Sequential(Conv(hidden, hidden, 3), nn.Conv2d(hidden, 1, 1))
        self.query_predictor = nn.Conv2d(hidden, 1, 1)
        self.contrast_predictor = nn.Sequential(Conv(hidden, hidden, 3), nn.Conv2d(hidden, 1, 1))

        # Sparse-query prior from QueryDet/RetinaNet-style focal heads.
        nn.init.constant_(self.query_predictor.bias, -4.595)

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict P2-resolution boundary, tiny-center, and local-contrast logits."""
        detail = self.p2_embed(p2)
        context = F.interpolate(self.p3_embed(p3), size=p2.shape[-2:], mode="nearest")
        shared = detail + context
        high_frequency = detail - F.avg_pool2d(detail, 5, stride=1, padding=2)
        boundary_logits = self.boundary_predictor(shared + high_frequency)
        query_logits = self.query_predictor(context)
        contrast_logits = self.contrast_predictor(shared + high_frequency)
        return boundary_logits, query_logits, contrast_logits


class CitrusScaleFusion(nn.Module):
    """Data-dependent cross-scale reweighting that is an exact Concat at initialization."""

    def __init__(self, channels: list[int] | tuple[int, ...], dimension: int = 1):
        super().__init__()
        if len(channels) < 2:
            raise ValueError(f"CitrusScaleFusion expects at least two feature levels, got {channels}")
        self.dimension = dimension
        self.num_inputs = len(channels)
        hidden = max(4, self.num_inputs * 2)
        self.gate = nn.Sequential(
            nn.Linear(self.num_inputs * 2, hidden),
            nn.SiLU(inplace=True),
            nn.Linear(hidden, self.num_inputs),
        )
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

    def forward(self, features: list[torch.Tensor]) -> torch.Tensor:
        """Fuse same-resolution pyramid features with bounded sample-wise gates."""
        if len(features) != self.num_inputs:
            raise ValueError(f"Expected {self.num_inputs} inputs, received {len(features)}")
        means = torch.stack([feature.mean(dim=(1, 2, 3)) for feature in features], dim=1)
        maxima = torch.stack([feature.amax(dim=(1, 2, 3)) for feature in features], dim=1)
        gates = 1.0 + 0.5 * torch.tanh(self.gate(torch.cat((means, maxima), dim=1)))
        weighted = [feature * gates[:, i, None, None, None] for i, feature in enumerate(features)]
        return torch.cat(weighted, dim=self.dimension)


class CitrusBoundaryFusion(nn.Module):
    """P2/P3 mask-boundary mutual fusion inspired by Boundary-preserving Mask R-CNN.

    P2 is rearranged with PixelUnshuffle instead of stride convolution, so every
    2x2 spatial sample reaches the P3-resolution branch. A QueryDet-style candidate
    map suppresses high-resolution leaf clutter. The boundary-to-mask residual is
    zero at initialization, leaving the pretrained prototype path unchanged.
    """

    def __init__(self, p2_channels: int, p3_channels: int):
        super().__init__()
        boundary_channels = max(32, p3_channels // 2)
        self.query = nn.Sequential(
            nn.Conv2d(p2_channels, p2_channels, 3, padding=1, groups=p2_channels, bias=False),
            nn.BatchNorm2d(p2_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(p2_channels, 1, 1),
        )
        nn.init.constant_(self.query[-1].bias, -4.595)  # 1% prior, as in focal-style dense heads
        self.p2_to_boundary = nn.Sequential(
            nn.Conv2d(p2_channels, boundary_channels, 1, bias=False),
            nn.BatchNorm2d(boundary_channels),
            nn.SiLU(inplace=True),
        )
        self.mask_to_boundary = nn.Conv2d(p3_channels, boundary_channels, 1, bias=False)
        self.boundary_refine = nn.Sequential(
            nn.Conv2d(
                boundary_channels, boundary_channels, 3, padding=1, groups=boundary_channels, bias=False
            ),
            nn.BatchNorm2d(boundary_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(boundary_channels, boundary_channels, 1, bias=False),
            nn.BatchNorm2d(boundary_channels),
            nn.SiLU(inplace=True),
        )
        self.boundary_predictor = nn.Conv2d(boundary_channels, 1, 1)
        self.boundary_to_mask = nn.Sequential(
            nn.PixelUnshuffle(2),
            nn.Conv2d(boundary_channels * 4, p3_channels, 1, bias=False),
        )
        self.mask_scale = nn.Parameter(torch.zeros(1, p3_channels, 1, 1))

    def forward(self, p2: torch.Tensor, p3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return refined P3 mask features, boundary logits, and P2 query logits."""
        query_logits = self.query(p2)
        candidate_gate = 0.5 + torch.sigmoid(query_logits)
        mask_boundary = F.interpolate(
            self.mask_to_boundary(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False
        )
        boundary = self.p2_to_boundary(p2 * candidate_gate) + mask_boundary
        boundary = self.boundary_refine(boundary)
        boundary_logits = self.boundary_predictor(boundary)
        refined_p3 = p3 + torch.tanh(self.mask_scale) * self.boundary_to_mask(boundary)
        return refined_p3, boundary_logits, query_logits
