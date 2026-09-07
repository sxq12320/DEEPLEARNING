# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Budgeted candidate-resolution experiment; not an official QueryDet/RTMDet reproduction.

Preserve the V5 prototype and semantic detail estimator. Move spatial prediction
work into a narrow per-scale stem, then predict boxes/classes/coefficients with
separate linear projections. Optional P2 adds actual candidates, not just masks.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_sage_v5 import SegmentCitrusSAGEV5
from .conv import Conv
from .head import Detect

__all__ = ("SegmentCitrusSAGEV7",)


class SAGEV7LocalContext(nn.Module):
    """PKINet-inspired short-range multi-kernel mixer, not its full PKI/CAA block.

    Only used in the narrow P2 prediction stem. No spatial expansion, dilated
    holes, attention matrix or extra full-resolution feature pyramid.
    """

    def __init__(self, channels):
        super().__init__()
        self.local = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.context = nn.Conv2d(channels, channels, 5, padding=2, groups=channels, bias=False)
        self.mix = Conv(channels, channels, 1)

    def forward(self, x):
        local = self.local(x)
        return self.mix(local + self.context(local))


class SegmentCitrusSAGEV7(SegmentCitrusSAGEV5):
    """Shared-task spatial stems, independent output projections, optional dense P2.

    P2 is FIRST in the output lattice so TAL sees ascending [4,8,16,32]
    strides. Prototype generation remains at P3 -> stride4, never stride2.
    New predictors have distinct state keys: do not accidentally initialize a
    P2 predictor from a shape-compatible but semantically different P3 tower.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        width=32,
        p2=True,
        relay=True,
        p2_mixer="dense",
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if width < 8 or width % 8:
            raise ValueError("Prediction width must be a positive multiple of eight")
        super().__init__(nc, nm, npr, detail_channels, relay, False, 0, 0, reg_max, end2end, ch)
        self.p2_enabled = bool(p2)
        candidate_channels = ([detail_channels] if p2 else []) + list(ch[:3])
        self.nl = len(candidate_channels)
        self.stride = torch.zeros(self.nl)
        # No unused old prediction parameters; proto/refiner/relay retain their keys.
        del self.cv2, self.cv3, self.cv4
        self.candidate_stems = nn.ModuleList(
            nn.Sequential(Conv(c, width, 1), Conv(width, width, 3)) for c in candidate_channels
        )
        if p2_mixer not in {"dense", "local_context"} or (p2_mixer != "dense" and not p2):
            raise ValueError("Use dense, or local_context with P2 enabled")
        if p2_mixer == "local_context":
            self.candidate_stems[0][1] = SAGEV7LocalContext(width)
        self.candidate_boxes = nn.ModuleList(
            nn.Sequential(nn.Conv2d(width, 4 * reg_max, 1)) for _ in candidate_channels
        )
        self.candidate_classes = nn.ModuleList(nn.Sequential(nn.Conv2d(width, nc, 1)) for _ in candidate_channels)
        self.candidate_masks = nn.ModuleList(nn.Sequential(nn.Conv2d(width, nm, 1)) for _ in candidate_channels)

    @property
    def one2many(self):
        return dict(box_head=self.candidate_boxes, cls_head=self.candidate_classes, mask_head=self.candidate_masks)

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        if self.relay_enabled:
            target = (2 * features[0].shape[-2], 2 * features[0].shape[-1])
            routed = detail if detail.shape[-2:] == target else F.interpolate(detail, size=target, mode="nearest")
            features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(routed, 2))
        proto = self.proto(features[0])
        detail = (
            detail
            if detail.shape[-2:] == proto.shape[-2:]
            else F.interpolate(detail, size=proto.shape[-2:], mode="nearest")
        )
        proto = proto + self.detail_scale.tanh() * self.detail_to_proto(detail)
        candidates = ([detail] if self.p2_enabled else []) + features
        candidates = [stem(feature) for stem, feature in zip(self.candidate_stems, candidates)]
        outputs = Detect.forward(self, candidates)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
