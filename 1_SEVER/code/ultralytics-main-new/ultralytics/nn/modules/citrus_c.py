# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Topology-decoupled prototype decoding for visible immature-citrus masks.

The module addresses three coupled failure modes in the citrus dataset instead of
adding generic attention blocks:

* P3 semantics should keep one strip-occluded fruit as one instance.
* P2 detail should retain deeply concave visible boundaries and narrow separators.
* A topology map should reject high-frequency leaf texture that is not a fruit edge.

The output remains a standard ``nm``-channel YOLO prototype tensor. Therefore the
existing mask-coefficient loss, post-processing, export path, and NMS remain intact.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv, DWConv

__all__ = ("CitrusTopologyPrototype",)


class CitrusTopologyPrototype(nn.Module):
    """Generate complementary semantic and topology-detail mask prototypes.

    Most prototype channels are generated from P3 to preserve instance semantics.
    A small channel budget is generated directly at P2 resolution for visible
    boundaries. Per-instance mask coefficients select the useful mixture, avoiding
    a fixed global fusion rule across the dataset's large within-image scale span.

    The same P2 representation predicts a four-state topology map:
    local context, fruit interior, visible boundary, and inter-instance separator.
    Its probabilities gate the detail prototypes and a zero-initialized P3 residual,
    suppressing unrelated leaf texture while allowing small-fruit evidence to reach
    the detector without adding a dense P2 detection level.
    """

    topology_classes = 4

    def __init__(
        self,
        p2_channels: int,
        p3_channels: int,
        hidden_channels: int = 256,
        prototype_channels: int = 32,
        detail_channels: int = 8,
    ):
        super().__init__()
        if not 0 < detail_channels < prototype_channels:
            raise ValueError(
                f"detail_channels must be in (0, {prototype_channels}), got {detail_channels}"
            )
        semantic_channels = prototype_channels - detail_channels

        # Keep the stock Proto names so cv1/upsample/cv2 transfer directly from
        # YOLO11n-seg. Only cv3 changes shape because the prototype budget is split.
        self.cv1 = Conv(p3_channels, hidden_channels, k=3)
        self.upsample = nn.ConvTranspose2d(hidden_channels, hidden_channels, 2, 2, bias=True)
        self.cv2 = Conv(hidden_channels, hidden_channels, k=3)
        self.cv3 = Conv(hidden_channels, semantic_channels)

        self.detail_encoder = DWConv(p2_channels, p2_channels, k=3)
        self.detail_prototypes = Conv(p2_channels, detail_channels, k=1)
        self.topology_predictor = nn.Conv2d(p2_channels, self.topology_classes, 1)

        # Topology-gated Space-to-Depth retains all P2 samples while keeping the
        # detector on P3-P5. Zero scale reproduces the pretrained P3 path at init.
        self.p2_to_p3 = nn.Sequential(
            nn.PixelUnshuffle(2),
            Conv(p2_channels * 4, p3_channels, k=1),
        )
        self.p3_residual_scale = nn.Parameter(torch.zeros(1, p3_channels, 1, 1))

    def forward(
        self, p2: torch.Tensor, p3: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return prototypes, topology logits, and topology-refined P3 features."""
        detail_features = self.detail_encoder(p2)
        topology_logits = self.topology_predictor(detail_features)
        topology_probability = topology_logits.softmax(dim=1)

        # Boundary and separator probabilities emphasize useful contour evidence;
        # the 0.5 floor avoids suppressing gradients before topology is learned.
        contour_gate = 0.5 + topology_probability[:, 2:4].sum(dim=1, keepdim=True)
        detail_prototypes = self.detail_prototypes(detail_features) * contour_gate

        # Interior/boundary/separator probability rejects unrelated orchard texture.
        fruit_gate = topology_probability[:, 1:].sum(dim=1, keepdim=True)
        p3_detail = self.p2_to_p3(detail_features * fruit_gate)
        if p3_detail.shape[-2:] != p3.shape[-2:]:
            raise ValueError(f"P2 detail produced {p3_detail.shape[-2:]}, expected P3 shape {p3.shape[-2:]}")
        refined_p3 = p3 + torch.tanh(self.p3_residual_scale) * p3_detail

        semantic_prototypes = self.cv3(self.cv2(self.upsample(self.cv1(refined_p3))))
        if semantic_prototypes.shape[-2:] != detail_prototypes.shape[-2:]:
            detail_prototypes = F.interpolate(
                detail_prototypes,
                size=semantic_prototypes.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        prototypes = torch.cat((semantic_prototypes, detail_prototypes), dim=1)
        return prototypes, topology_logits, refined_p3
