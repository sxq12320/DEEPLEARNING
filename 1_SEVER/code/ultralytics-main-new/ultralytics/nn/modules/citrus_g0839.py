# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Shared Search--Discriminate--Refine support for the G_0839 citrus series.

This module is deliberately a single task-specific path rather than a collection
of interchangeable attention blocks. A coarse P3/P4 query locates likely tiny
fruit, a P2 inner-versus-ring comparison rejects leaf-colour camouflage, and the
same high-resolution representation predicts visible boundaries and neighbouring
instance topology. The resulting support only refines mask prototypes; detection
remains on P3--P5 to avoid the cost and false positives of a dense P2 detector.

The design follows the experimentally testable ideas in QueryDet (coarse-to-fine
small-object search), camouflage segmentation (target/context discrimination),
boundary-aware mask refinement, and topology/offset-style instance separation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv
from .citrus_d import CitrusStructureStem

__all__ = ("CitrusDualResolutionBackbone", "CitrusSDRSupport")


class _LiteResidualBlock(nn.Module):
    """Depthwise residual feature extractor without CSP/C3k2 topology."""

    def __init__(self, channels: int, expansion: float = 2.0):
        super().__init__()
        hidden = max(channels, int(round(channels * expansion)))
        self.expand = Conv(channels, hidden, k=1)
        self.spatial = DWConv(hidden, hidden, k=3)
        self.project = Conv(hidden, channels, k=1, act=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply local spatial extraction and residual channel mixing."""
        return x + self.project(self.spatial(self.expand(x)))


class _SemanticStage(nn.Module):
    """Low-pass downsampling followed by lightweight residual extraction."""

    def __init__(self, c1: int, c2: int, repeats: int):
        super().__init__()
        self.downsample = nn.Sequential(nn.AvgPool2d(2, stride=2), Conv(c1, c2, k=1))
        self.blocks = nn.Sequential(*(_LiteResidualBlock(c2) for _ in range(repeats)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract the next lower-resolution semantic stage."""
        return self.blocks(self.downsample(x))


class _DualResolutionExchange(nn.Module):
    """Exchange evidence between persistent P2 detail and one semantic scale."""

    def __init__(self, detail_channels: int, semantic_channels: int, ratio: int):
        super().__init__()
        if ratio not in {2, 4, 8}:
            raise ValueError(f"P2-to-semantic ratio must be 2, 4, or 8, got {ratio}")
        self.ratio = ratio
        self.detail_update = nn.Sequential(
            DWConv(detail_channels, detail_channels, k=3),
            Conv(detail_channels, detail_channels, k=1),
        )
        self.semantic_gate = Conv(semantic_channels, detail_channels, k=1, act=False)
        self.detail_to_semantic = Conv(detail_channels, semantic_channels, k=1, act=False)
        self.detail_scale = nn.Parameter(torch.tensor(0.1))
        self.semantic_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, detail: torch.Tensor, semantic: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Gate P2 updates with semantics and inject pooled shape evidence back."""
        gate = torch.sigmoid(
            F.interpolate(self.semantic_gate(semantic), size=detail.shape[-2:], mode="bilinear", align_corners=False)
        )
        detail = detail + torch.tanh(self.detail_scale) * gate * self.detail_update(detail)
        pooled = F.avg_pool2d(detail, kernel_size=self.ratio, stride=self.ratio)
        if pooled.shape[-2:] != semantic.shape[-2:]:
            pooled = F.interpolate(pooled, size=semantic.shape[-2:], mode="area")
        semantic = semantic + torch.tanh(self.semantic_scale) * self.detail_to_semantic(pooled)
        return detail, semantic


class CitrusDualResolutionBackbone(nn.Module):
    """Non-C3k2 dual-resolution backbone for RGB immature-citrus imagery.

    The P2 stream remains at stride 4 through the complete backbone. A separate
    low-pass semantic stream forms P3--P5 using depthwise residual blocks. Three
    bidirectional exchanges let global semantics reject leaf texture while local
    shape evidence survives every downsampling stage. This is a compact adaptation
    of the structural principles in Lite-HRNet, PIDNet, and Gated-SCNN.

    Args:
        channels: Fixed output widths for P2, P3, P4 and P5.
        repeats: Number of lightweight residual blocks in each semantic stage.
    """

    def __init__(self, channels: list[int] | tuple[int, int, int, int] = (48, 96, 160, 256), repeats: int = 2):
        super().__init__()
        if len(channels) != 4:
            raise ValueError(f"Expected P2/P3/P4/P5 channel widths, got {channels}")
        if repeats < 1:
            raise ValueError(f"repeats must be positive, got {repeats}")
        p2_channels, p3_channels, p4_channels, p5_channels = (int(value) for value in channels)
        stem_channels = max(24, p2_channels // 2)

        self.stem = CitrusStructureStem(3, stem_channels, k=3, s=2, structure_ratio=0.25)
        self.p2 = _SemanticStage(stem_channels, p2_channels, repeats=1)
        self.p3 = _SemanticStage(p2_channels, p3_channels, repeats=repeats)
        self.p4 = _SemanticStage(p3_channels, p4_channels, repeats=repeats)
        self.p5 = _SemanticStage(p4_channels, p5_channels, repeats=repeats)
        self.exchanges = nn.ModuleList(
            (
                _DualResolutionExchange(p2_channels, p3_channels, ratio=2),
                _DualResolutionExchange(p2_channels, p4_channels, ratio=4),
                _DualResolutionExchange(p2_channels, p5_channels, ratio=8),
            )
        )

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return persistent P2 detail and the P3--P5 semantic pyramid."""
        p2 = self.p2(self.stem(x))
        p3 = self.p3(p2)
        p2, p3 = self.exchanges[0](p2, p3)
        p4 = self.p4(p3)
        p2, p4 = self.exchanges[1](p2, p4)
        p5 = self.p5(p4)
        p2, p5 = self.exchanges[2](p2, p5)
        return [p2, p3, p4, p5]


class CitrusSDRSupport(nn.Module):
    """Generate query-gated high-resolution prototype evidence and auxiliary logits.

    Args:
        channels: P2, P3 and P4 channel counts.
        prototype_channels: Number of YOLO mask prototypes.
        stage: Enabled SDR stage in ``[2, 5]``.
        topk: Number of coarse P3 cells retained by the inference support mask.
        detail_channels: Width of the shared stride-4 detail representation.

    Training uses a differentiable query probability. Evaluation uses a dilated
    top-k support mask, so the information flow is sparse even though ordinary
    PyTorch still executes the lightweight P2 convolutions densely. Consequently
    this module does not claim sparse-kernel wall-clock savings.
    """

    topology_classes = 4

    def __init__(
        self,
        channels: list[int] | tuple[int, int, int],
        prototype_channels: int,
        stage: int = 2,
        topk: int = 64,
        detail_channels: int = 32,
    ):
        super().__init__()
        if len(channels) != 3:
            raise ValueError(f"CitrusSDRSupport expects P2/P3/P4 channels, got {channels}")
        if stage not in {2, 3, 4, 5}:
            raise ValueError(f"SDR stage must be 2, 3, 4, or 5, got {stage}")
        if topk < 1:
            raise ValueError(f"topk must be positive, got {topk}")

        p2_channels, p3_channels, p4_channels = channels
        hidden = max(16, int(detail_channels))
        query_channels = max(8, hidden // 2)
        self.stage = int(stage)
        self.topk = int(topk)

        self.p2_detail = nn.Sequential(
            DWConv(p2_channels, p2_channels, k=3),
            Conv(p2_channels, hidden, k=1),
        )
        self.p3_context = Conv(p3_channels, hidden, k=1, act=False)
        self.p3_query = Conv(p3_channels, query_channels, k=1, act=False)
        self.p4_query = Conv(p4_channels, query_channels, k=1, act=False)
        self.query_predictor = nn.Sequential(
            DWConv(query_channels, query_channels, k=3),
            nn.Conv2d(query_channels, 1, 1),
        )

        # One local representation supplies every later stage. This keeps the
        # ablation causal and avoids paying for parallel attention branches.
        self.context_predictor = (
            nn.Sequential(DWConv(hidden, hidden, k=3), nn.Conv2d(hidden, 1, 1)) if stage >= 3 else None
        )
        self.boundary_predictor = (
            nn.Sequential(DWConv(hidden, hidden, k=3), nn.Conv2d(hidden, 1, 1)) if stage >= 4 else None
        )
        self.topology_predictor = nn.Conv2d(hidden, self.topology_classes, 1) if stage >= 5 else None
        self.prototype_residual = nn.Sequential(
            DWConv(hidden, hidden, k=3),
            nn.Conv2d(hidden, prototype_channels, 1),
        )

        # A sparse focal-head prior and an exact standard-prototype initialization.
        nn.init.constant_(self.query_predictor[-1].bias, -4.595)
        nn.init.zeros_(self.prototype_residual[-1].weight)
        nn.init.zeros_(self.prototype_residual[-1].bias)

    def _query_support(self, query_logits: torch.Tensor, p2_size: tuple[int, int]) -> torch.Tensor:
        """Return soft training support or dilated top-k evaluation support at P2."""
        probability = query_logits.sigmoid()
        if self.training:
            support = probability
        else:
            batch, _, height, width = probability.shape
            count = min(self.topk, height * width)
            indices = probability.flatten(1).topk(count, dim=1).indices
            support = probability.new_zeros((batch, height * width))
            support.scatter_(1, indices, 1.0)
            support = F.max_pool2d(support.view(batch, 1, height, width), 3, stride=1, padding=1)
        return F.interpolate(support, size=p2_size, mode="nearest")

    def forward(
        self,
        p2: torch.Tensor,
        p3: torch.Tensor,
        p4: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Return a zero-initialized prototype residual and enabled auxiliary logits."""
        query_feature = self.p3_query(p3) + F.interpolate(
            self.p4_query(p4), size=p3.shape[-2:], mode="bilinear", align_corners=False
        )
        query_logits = self.query_predictor(query_feature)
        query_support = self._query_support(query_logits, p2.shape[-2:])

        detail = self.p2_detail(p2)
        semantic = F.interpolate(self.p3_context(p3), size=p2.shape[-2:], mode="bilinear", align_corners=False)
        shared = detail + semantic

        # The residual floor preserves gradients outside selected cells while the
        # query map still doubles the contribution inside likely tiny-fruit ROIs.
        support_gate = 0.5 + query_support
        auxiliary: dict[str, torch.Tensor] = {"citrus_query": query_logits}

        if self.stage >= 3:
            # Inner-minus-ring evidence is insensitive to absolute green colour.
            inner = F.avg_pool2d(shared, 3, stride=1, padding=1)
            outer = F.avg_pool2d(shared, 7, stride=1, padding=3)
            context_logits = self.context_predictor(inner - outer)
            support_gate = support_gate * (0.75 + 0.5 * context_logits.sigmoid())
            auxiliary["citrus_contrast"] = context_logits

        if self.stage >= 4:
            high_frequency = shared - F.avg_pool2d(shared, 5, stride=1, padding=2)
            boundary_logits = self.boundary_predictor(high_frequency)
            support_gate = support_gate * (0.75 + 0.5 * boundary_logits.sigmoid())
            auxiliary["citrus_boundary"] = boundary_logits

        if self.stage >= 5:
            topology_logits = self.topology_predictor(shared)
            topology_probability = topology_logits.softmax(dim=1)
            fruit_probability = topology_probability[:, 1:].sum(dim=1, keepdim=True)
            contour_probability = topology_probability[:, 2:4].sum(dim=1, keepdim=True)
            support_gate = support_gate * (0.5 + 0.5 * fruit_probability) * (0.75 + 0.5 * contour_probability)
            auxiliary["citrus_topology"] = topology_logits

        residual = self.prototype_residual(shared * support_gate)
        return residual, auxiliary
