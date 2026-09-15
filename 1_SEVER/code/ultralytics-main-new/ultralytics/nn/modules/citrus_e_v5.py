# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V5: mask-specific detail adaptation and training-only regional discrimination.

The inherited deep8/quality topology is V4R05. This is a task adaptation, not
ReCo/ConDSeg reproduction. No color thresholds, convexity prior or learned crop
selector. Auxiliary pixel classifiers/embeddings are never evaluated at inference.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v4r import SegmentCitrusEV4Quality
from .conv import Conv
from .head import Detect

__all__ = ("SegmentCitrusEV5",)


class SegmentCitrusEV5(SegmentCitrusEV4Quality):
    """Keep the candidate relay unchanged; optionally adapt only mask detail.

    The additional residual is bounded in gain, not a stability-guaranteed
    controller. Base features remain shared; this is not gradient isolation.
    All experimental switches are serialized in the official model YAML.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        mask_route=False,
        region_gain=0.0,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        super().__init__(nc, nm, npr, detail_channels, True, False, 0.0, 0.0, 1.0, False, 0.5, reg_max, end2end, ch)
        if not math.isfinite(region_gain) or region_gain < 0:
            raise ValueError("region_gain must be finite and nonnegative")
        self.mask_route = bool(mask_route)
        self.region_gain = float(region_gain)
        if self.mask_route:
            self.mask_context = Conv(ch[0], detail_channels, 1, act=False)
            self.mask_update = nn.Sequential(
                Conv(2 * detail_channels, detail_channels, 1),
                Conv(detail_channels, detail_channels, 3, g=detail_channels, act=False),
            )
            self.mask_route_scale = nn.Parameter(torch.full((1, detail_channels, 1, 1), 0.01))
        if self.region_gain:
            self.region_classifier = nn.Conv2d(detail_channels, 1, 1)
            self.region_embedding = nn.Conv2d(detail_channels, 8, 1, bias=False)

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
