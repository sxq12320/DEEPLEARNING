# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V6: stride-2 mask evidence for strict-IoU quality on small instances.

The inherited topology is the V5 control graph (V4R05): asymmetric neck, 16ch
semantic-conditioned P2 detail, bounded mask-quality ranking. V6 adds an
optional second prototype stage that decodes masks at stride 2 instead of
stride 4, gated by the already-refined detail estimate. Task adaptation of the
fine-feature fusion idea in RefineMask/Mask Transfiner; not a reproduction, no
quadtree, RoI loop, per-instance crops, dense P2 detection tower, or inference
branches beyond one extra depthwise/pointwise pair.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v5 import SegmentCitrusEV5
from .conv import Conv
from .head import Detect

__all__ = ("EV6FineDetail", "SegmentCitrusEV6")


class EV6FineDetail(nn.Module):
    """Gate stem-resolution (stride-2) texture against the refined P2 estimate.

    Mirrors the SAGE semantic-correction form: an upsampled semantic estimate is
    the reference, the raw fine detail is only admitted where a learned gate
    agrees with it. ``channels`` stays narrow; all spatial work is depthwise.
    """

    def __init__(self, c_stem: int, channels: int = 16):
        super().__init__()
        if channels < 8:
            raise ValueError("Use at least eight fine-detail channels")
        self.detail = nn.Sequential(Conv(c_stem, channels, 1), Conv(channels, channels, 3, g=channels))
        self.gate = nn.Conv2d(3 * channels, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, stem: torch.Tensor, estimate: torch.Tensor) -> torch.Tensor:
        """Return estimate corrected by gated stride-2 detail at estimate's size."""
        fine = self.detail(stem)
        if fine.shape[-2:] != estimate.shape[-2:]:
            fine = F.interpolate(fine, estimate.shape[-2:], mode="nearest")
        gain = self.gate(torch.cat((fine, estimate, (fine - estimate).abs()), 1)).sigmoid()
        return estimate + gain * (fine - estimate)


class SegmentCitrusEV6(SegmentCitrusEV5):
    """V5 head plus an optional stride-2 prototype refinement path.

    With ``fine_mask`` off the graph, parameters and state-dict keys are exactly
    the V5 control. When enabled the head additionally consumes the stride-2
    stem feature (fifth YAML input), produces ``proto`` at twice the historical
    mask resolution and declares ``proto_stride = 2`` so the validator scales
    ground-truth rasters to the actual output grid instead of a hardcoded /4.
    The extra path is a bounded residual: ``tanh`` gain starts at 0.01, so early
    training is not dominated by an untrained high-resolution branch.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        mask_route=False,
        region_gain=0.0,
        fine_mask=False,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if len(ch) < 4:
            raise ValueError("E V6 requires at least P3/P4/C5/C2 head inputs")
        stem_channels = ch[4] if len(ch) > 4 else 0
        super().__init__(
            nc, nm, npr, detail_channels, mask_route, region_gain, reg_max, end2end, tuple(ch[:4])
        )
        self.fine_mask = bool(fine_mask)
        if fine_mask not in (False, True, "phase"):
            raise ValueError("fine_mask must be False, True (dense) or 'phase'")
        self.fine_mode = "phase" if fine_mask == "phase" else "dense"
        if self.fine_mask and not stem_channels:
            raise ValueError("fine_mask requires the stride-2 stem feature as head input index 4")
        self.proto_stride = 2 if self.fine_mask else 4
        if self.fine_mask:
            # Phase mode transports all four stem phases to P2 before learned
            # compression/gating, then emits four mask subpixels per location.
            # Rearrangement is exact; learned channel compression is NOT lossless.
            phase = self.fine_mode == "phase"
            self.fine_detail = EV6FineDetail(stem_channels * (4 if phase else 1), detail_channels)
            self.fine_to_proto = nn.Conv2d(detail_channels, nm * (4 if phase else 1), 1, bias=False)
            self.fine_scale = nn.Parameter(torch.full((1, nm, 1, 1), 0.01))

    def refine_fine_proto(self, proto, mask_detail, stem):
        """Shared dense/phase decoder, also used by subclasses with extra scales."""
        if getattr(self, "fine_mode", "dense") == "phase":
            refined = self.fine_detail(F.pixel_unshuffle(stem, 2), mask_detail)
            update = F.pixel_shuffle(self.fine_to_proto(refined), 2)
        else:
            estimate = F.interpolate(mask_detail, scale_factor=2, mode="nearest")
            refined = self.fine_detail(stem, estimate)
            update = self.fine_to_proto(refined)
        proto = F.interpolate(proto, update.shape[-2:], mode="bilinear", align_corners=False)
        return proto + self.fine_scale.tanh() * update

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        mask_detail = detail
        if self.mask_route:
            context = F.interpolate(self.mask_context(features[0]), detail.shape[-2:], mode="nearest")
            update = self.mask_update(torch.cat((detail, context), 1))
            mask_detail = detail + self.mask_route_scale.tanh() * update
        target = (features[0].shape[-2] * 2, features[0].shape[-1] * 2)
        routed = detail if detail.shape[-2:] == target else F.interpolate(detail, target, mode="nearest")
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(routed, 2))
        proto = self.proto(features[0])
        if mask_detail.shape[-2:] != proto.shape[-2:]:
            mask_detail = F.interpolate(mask_detail, proto.shape[-2:], mode="nearest")
        proto = proto + self.detail_scale.tanh() * self.detail_to_proto(mask_detail)
        if self.fine_mask:
            proto = self.refine_fine_proto(proto, mask_detail, x[4])
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                if self.region_gain:
                    preds["ev5_region_logits"] = self.region_classifier(mask_detail)
                    preds["ev5_region_embedding"] = self.region_embedding(mask_detail)
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
