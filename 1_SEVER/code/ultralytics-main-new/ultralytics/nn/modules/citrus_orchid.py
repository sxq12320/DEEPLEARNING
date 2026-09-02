# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Object-region-conditioned feature routing for the ORCHID citrus series.

ORCHID changes the information-flow graph instead of inserting another attention
block into PAN.  The stable detection pyramid and the mask-evidence path are
separated.  A coarse semantic query decides where native P2 detail may enter the
mask prototype, while a single-canvas variant replaces recurrent PAN fusion.

The design is grounded in three published observations:

* QueryDet (CVPR 2022) uses coarse queries to activate high-resolution features
  only where small objects are plausible.
* Mask2Former (CVPR 2022) restricts feature aggregation to predicted foreground
  regions instead of attending to the complete background.
* DCNet (CVPR 2023) suppresses camouflage through target-versus-background
  difference evidence at both pixel and instance levels.

Only regular PyTorch operators are used; no Mamba, custom CUDA, or deformable
operator is required.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv

__all__ = ("CitrusORCHIDMaskRouter", "CitrusORCHIDNeck")


class _DepthwiseRefine(nn.Module):
    """One regular depthwise/pointwise residual refinement block."""

    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Refine one feature map without changing its shape."""
        return self.block(x)


class CitrusORCHIDMaskRouter(nn.Module):
    """Build a mask-only evidence path from raw backbone and PAN features.

    Detection continues to consume the pretrained PAN P3/P4/P5 tensors.  This
    module sees raw C2/C3/C4/C5 features and modifies only the P3 input of the
    prototype generator.  Consequently leaf texture admitted by the mask route
    cannot perturb box/class predictions.

    Modes:
        1: task-decoupled, ungated detail evidence.
        2: coarse semantic query gates P2 detail.
        3: mode 2 plus local-reference subtraction for camouflage suppression.
    """

    def __init__(
        self,
        channels: list[int] | tuple[int, int, int, int, int],
        mode: int = 2,
        route_channels: int = 48,
        initial_scale: float = 0.1,
    ):
        super().__init__()
        if len(channels) != 5:
            raise ValueError(f"CitrusORCHIDMaskRouter expects C2/C3/C4/C5/PAN-P3 channels, got {channels}")
        if mode not in {1, 2, 3}:
            raise ValueError(f"ORCHID router mode must be 1, 2, or 3, got {mode}")
        if route_channels < 8:
            raise ValueError(f"route_channels must be at least 8, got {route_channels}")
        if not 0.0 < initial_scale <= 1.0:
            raise ValueError(f"initial_scale must be in (0, 1], got {initial_scale}")

        c2, c3, c4, c5, mask_channels = (int(value) for value in channels)
        width = int(route_channels)
        self.mode = int(mode)

        # P2 keeps only channel projection and gating. Spatial mixing happens
        # after the one anti-aliased transition at P3, avoiding a costly dense
        # 3x3 pass over the full high-resolution orchard background.
        self.p2_detail = Conv(c2, width, 1)
        self.detail_refine = _DepthwiseRefine(width)
        self.p3_semantic = Conv(c3, width, 1)
        self.p4_context = Conv(c4, width, 1)
        self.p5_context = Conv(c5, width, 1)
        self.semantic_refine = _DepthwiseRefine(width)

        self.query_predictor = None
        if self.mode >= 2:
            self.query_predictor = nn.Conv2d(width, 1, 1)
            nn.init.normal_(self.query_predictor.weight, std=0.01)
            nn.init.constant_(self.query_predictor.bias, -4.595)  # 1% foreground prior used by focal heads

        self.evidence_refine = _DepthwiseRefine(width)
        self.evidence_to_mask = nn.Sequential(
            nn.Conv2d(width, mask_channels, 1, bias=False),
            nn.BatchNorm2d(mask_channels),
        )
        self.route_scale = nn.Parameter(torch.full((1, mask_channels, 1, 1), float(initial_scale)))

        self.contrast_predictor = None
        if self.mode == 3:
            self.contrast_predictor = nn.Sequential(
                _DepthwiseRefine(width),
                nn.Conv2d(width, 1, 1),
            )

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Resize only when needed to avoid redundant interpolation kernels."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(
        self,
        features: list[torch.Tensor] | tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Return refined prototype input, tiny-query logits, and optional contrast logits."""
        if len(features) != 5:
            raise ValueError(f"Expected C2/C3/C4/C5/PAN-P3 tensors, received {len(features)}")
        p2, c3, c4, c5, pan_p3 = features
        p3_size = c3.shape[-2:]

        semantic = self.p3_semantic(c3)
        semantic = semantic + self._resize(self.p4_context(c4), p3_size)
        # Normalize while spatial support is still larger than 1x1 so batch-size-one
        # smoke runs remain valid, then pool the projected context.
        global_context = F.adaptive_avg_pool2d(self.p5_context(c5), output_size=1)
        semantic = self.semantic_refine(semantic + global_context)

        detail = self.p2_detail(p2)
        query_logits = None
        if self.query_predictor is not None:
            query_p3 = self.query_predictor(semantic)
            query_logits = self._resize(query_p3, detail.shape[-2:])
            detail = detail * query_logits.sigmoid()

        # One anti-aliased transition is the only P2-to-P3 resampling in this path.
        detail_p3 = F.avg_pool2d(detail, kernel_size=2, stride=2)
        detail_p3 = self._resize(detail_p3, p3_size)
        detail_p3 = self.detail_refine(detail_p3)

        contrast_logits = None
        if self.mode == 3:
            # A local ring is a cheap reference for the leaf/branch appearance surrounding a candidate.
            local_reference = F.avg_pool2d(detail_p3, kernel_size=7, stride=1, padding=3)
            detail_p3 = detail_p3 - local_reference
            if self.training:
                contrast_p3 = self.contrast_predictor(detail_p3 + semantic)
                contrast_logits = self._resize(contrast_p3, p2.shape[-2:])

        evidence = self.evidence_refine(detail_p3 + semantic)
        residual = self.evidence_to_mask(evidence)
        refined_p3 = pan_p3 + torch.tanh(self.route_scale) * residual
        return refined_p3, query_logits, contrast_logits


class CitrusORCHIDNeck(nn.Module):
    """Single-canvas alternative to recurrent FPN/PAN fusion.

    All raw levels contribute once to a canonical P3 evidence canvas.  P2 detail
    is admitted by a P3-derived tiny-object query.  P4/P5 then retain their own
    lateral identities and receive a small residual from the canvas.  This makes
    the ablation structurally different from top-down/bottom-up concatenate loops.
    """

    def __init__(
        self,
        in_channels: list[int] | tuple[int, int, int, int],
        out_channels: list[int] | tuple[int, int, int] = (64, 128, 256),
        route_channels: int = 48,
        initial_scale: float = 0.1,
    ):
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError(f"CitrusORCHIDNeck expects C2/C3/C4/C5 channels, got {in_channels}")
        if len(out_channels) != 3:
            raise ValueError(f"CitrusORCHIDNeck expects P3/P4/P5 output channels, got {out_channels}")
        c2, c3, c4, c5 = (int(value) for value in in_channels)
        o3, o4, o5 = (int(value) for value in out_channels)
        width = int(route_channels)

        self.p2_detail = Conv(c2, o3, 1)
        self.p3_lateral = Conv(c3, o3, 1)
        self.p4_query = Conv(c4, width, 1)
        self.p5_query = Conv(c5, width, 1)
        self.query_refine = _DepthwiseRefine(width)
        self.query_predictor = nn.Conv2d(width, 1, 1)
        nn.init.normal_(self.query_predictor.weight, std=0.01)
        nn.init.constant_(self.query_predictor.bias, -4.595)
        self.context_to_p3 = nn.Sequential(
            nn.Conv2d(width, o3, 1, bias=False),
            nn.BatchNorm2d(o3),
        )

        self.p4_lateral = Conv(c4, o4, 1)
        self.p5_lateral = Conv(c5, o5, 1)
        self.p3_to_p4 = nn.Sequential(Conv(o3, o4, 1), nn.AvgPool2d(2, 2))
        self.p4_to_p5 = nn.Sequential(Conv(o4, o5, 1), nn.AvgPool2d(2, 2))

        self.detail_scale = nn.Parameter(torch.full((1, o3, 1, 1), float(initial_scale)))
        self.context_scale = nn.Parameter(torch.full((1, o3, 1, 1), float(initial_scale)))
        self.p4_scale = nn.Parameter(torch.full((1, o4, 1, 1), float(initial_scale)))
        self.p5_scale = nn.Parameter(torch.full((1, o5, 1, 1), float(initial_scale)))

    @staticmethod
    def _resize(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
        """Bilinearly align a source to the canonical canvas."""
        return x if x.shape[-2:] == size else F.interpolate(x, size=size, mode="bilinear", align_corners=False)

    def forward(self, features: list[torch.Tensor] | tuple[torch.Tensor, ...]) -> list[torch.Tensor]:
        """Return P3/P4/P5 predictions features and a P2-resolution query map."""
        if len(features) != 4:
            raise ValueError(f"Expected four backbone tensors, received {len(features)}")
        p2, c3, c4, c5 = features
        p3 = self.p3_lateral(c3)
        p3_size = p3.shape[-2:]

        context = self._resize(self.p4_query(c4), p3_size)
        context = context + F.adaptive_avg_pool2d(self.p5_query(c5), output_size=1)
        context = self.query_refine(context)
        query_p3 = self.query_predictor(context)
        query_p2 = self._resize(query_p3, p2.shape[-2:])

        detail = self.p2_detail(p2) * query_p2.sigmoid()
        detail = F.avg_pool2d(detail, kernel_size=2, stride=2)
        detail = self._resize(detail, p3_size)
        p3 = p3 + torch.tanh(self.detail_scale) * detail
        p3 = p3 + torch.tanh(self.context_scale) * self.context_to_p3(context)

        p4_identity = self.p4_lateral(c4)
        p4_from_canvas = self._resize(self.p3_to_p4(p3), p4_identity.shape[-2:])
        p4 = p4_identity + torch.tanh(self.p4_scale) * p4_from_canvas

        p5_identity = self.p5_lateral(c5)
        p5_from_canvas = self._resize(self.p4_to_p5(p4), p5_identity.shape[-2:])
        p5 = p5_identity + torch.tanh(self.p5_scale) * p5_from_canvas
        return [p3, p4, p5, query_p2]
