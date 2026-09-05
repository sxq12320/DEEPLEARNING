# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""SAGE-v5: dual-use detail routing and late-upsample prototype decoding.

Independent task adaptation, not a transplanted PIDNet/GSCNN/QueryDet model.
See docs/SAGE_V5_EVIDENCE_AND_DESIGN.md for source-code provenance and limits.
There is no temporal PID, iterative inference, new dense P2 detection tower,
custom CUDA operation or inference-time GT input.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_sage_v4r import SegmentCitrusSAGEV4R
from .conv import Conv
from .head import Detect

__all__ = ("SegmentCitrusSAGEV5",)


class SAGELateProto(nn.Module):
    """Perform both spatial convolutions at P3, then resize and project at P2.

    cv1/cv2/cv3 shapes preserve their official initialization keys, but changing
    the operation order is NOT function equivalence to the pretrained decoder.
    Fine spatial evidence arrives through the separately learned C2 branch.
    """

    def __init__(self, c1, hidden, nm):
        super().__init__()
        self.cv1 = Conv(c1, hidden, 3)
        self.cv2 = Conv(hidden, hidden, 3)
        self.cv3 = Conv(hidden, nm, 1)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.cv3(self.upsample(self.cv2(self.cv1(x))))


class SegmentCitrusSAGEV5(SegmentCitrusSAGEV4R):
    """One gated C2/P3 detail estimate, two consumers: candidates and prototypes.

    Relay reindexes the stride-4 estimate into stride-8 channels before a 1x1
    projection. Reindexing alone is invertible; the projection is not lossless.
    The relay residual has a small nonzero initial gain so it can learn from
    the first backward pass. No tensor is detached and features are not mutated.
    Inherited geometry acts on visible per-instance masks, never their hulls.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        relay=False,
        late_proto=False,
        boundary_gain=0.0,
        neighbor_gain=0.0,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        super().__init__(nc, nm, npr, detail_channels, "semantic", boundary_gain, neighbor_gain, reg_max, end2end, ch)
        self.relay_enabled, self.late_proto = bool(relay), bool(late_proto)
        if late_proto:
            self.proto = SAGELateProto(ch[0], npr, nm)
        if relay:
            self.detail_relay = Conv(4 * detail_channels, ch[0], 1, act=False)
            self.relay_scale = nn.Parameter(torch.full((1, ch[0], 1, 1), 0.01))

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        if self.relay_enabled:
            # Exact 2x spatial relation in normal /32-padded YOLO inputs.
            # Explicit sizing also supports independently supplied feature maps.
            target = (features[0].shape[-2] * 2, features[0].shape[-1] * 2)
            routed = detail if detail.shape[-2:] == target else F.interpolate(detail, size=target, mode="nearest")
            routed = self.detail_relay(F.pixel_unshuffle(routed, 2))
            features[0] = features[0] + self.relay_scale.tanh() * routed
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
