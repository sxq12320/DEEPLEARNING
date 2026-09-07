# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V7R: preserve fine evidence and route coarse context by prediction task.

Independent adaptation of SAFM/STAM multiscale context and PIDNet's selective
detail/context fusion principle. Not a reproduction of either architecture,
a temporal network, a PID controller or a guarantee of increased recall.
See docs/SAGE_V7R_DESIGN.md for sources, ablations and limitations.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_sage_v4r import SAGEMaskCorrection
from .citrus_sage_v7 import SegmentCitrusSAGEV7
from .conv import Conv

__all__ = ("SegmentCitrusSAGEV7R",)


class SAGEV7ContextPyramid(nn.Module):
    """Four channel groups at P3; NEVER pool the high-resolution detail path.

    SAFM-inspired grouped pooling/convolution/resize, with bounded residual
    modulation instead of unbounded multiplicative replacement. No SE, ASPP,
    FFT, unfold, grid sampling, custom CUDA, or quadratic attention matrix.
    Single-scale control retains exactly the same trainable parameter shapes.
    """

    def __init__(self, channels: int, multiscale: bool = True):
        super().__init__()
        if channels < 8 or channels % 4:
            raise ValueError("Context width must be >=8 and divisible by four")
        self.factors = (1, 2, 4, 8) if multiscale else (1, 1, 1, 1)
        group = channels // 4
        self.filters = nn.ModuleList(nn.Conv2d(group, group, 3, padding=1, groups=group) for _ in range(4))
        self.aggregate = nn.Conv2d(channels, channels, 1)

    def forward(self, x):
        h, w = x.shape[-2:]
        groups = []
        for part, conv, factor in zip(x.chunk(4, 1), self.filters, self.factors):
            if factor != 1:
                part = F.adaptive_avg_pool2d(part, (max(1, h // factor), max(1, w // factor)))
            part = conv(part)
            if factor != 1:
                part = F.interpolate(part, size=(h, w), mode="nearest")
            groups.append(part)
        return x * (1.0 + 0.5 * self.aggregate(torch.cat(groups, 1)).tanh())


class SAGEV7SemanticRoute(nn.Module):
    """Learn a bounded-gain context residual; discrepancy is NOT calibrated uncertainty."""

    def __init__(self, semantic_channels: int, width: int, multiscale: bool):
        super().__init__()
        self.project = Conv(semantic_channels, width, 1, act=False)
        self.pyramid = SAGEV7ContextPyramid(width, multiscale)
        self.gate = nn.Conv2d(3 * width, 1, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gain = nn.Parameter(torch.full((1, width, 1, 1), 0.1))

    def forward(self, local, semantic):
        context = self.pyramid(self.project(semantic))
        context = F.interpolate(context, size=local.shape[-2:], mode="nearest")
        weight = self.gate(torch.cat((local, context, (local - context).abs()), 1)).sigmoid()
        return local + self.gain.tanh() * weight * context


class SegmentCitrusSAGEV7R(SegmentCitrusSAGEV7):
    """Direct detail -> boxes/mask coefficients/prototypes; context -> chosen tasks.

    `none`: direct-detail control. `shared`: context residual to all P2 predictors.
    `cls_only`: SAME residual/parameters, but only P2 classification consumes it.
    Prototype and P3 relay always consume direct detail in these three variants.
    Semantic routing reads pre-relay P3; there is no repeated or circular pass.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        width=32,
        route="cls_only",
        multiscale=True,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if route not in {"none", "shared", "cls_only"}:
            raise ValueError("route must be none/shared/cls_only")
        super().__init__(nc, nm, npr, detail_channels, width, True, True, "dense", reg_max, end2end, ch)
        self.route = route
        self.refiner = SAGEMaskCorrection(ch[3], ch[0], detail_channels, "direct")
        if route != "none":
            self.semantic_route = SAGEV7SemanticRoute(ch[0], width, multiscale)

    def forward(self, x):
        features = list(x[:3])
        semantic = features[0]
        detail = self.refiner(x[3], semantic)
        target = (2 * semantic.shape[-2], 2 * semantic.shape[-1])
        relay = detail if detail.shape[-2:] == target else F.interpolate(detail, size=target, mode="nearest")
        features[0] = semantic + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(relay, 2))
        proto = self.proto(features[0])
        detail = (
            F.interpolate(detail, size=proto.shape[-2:], mode="nearest")
            if detail.shape[-2:] != proto.shape[-2:]
            else detail
        )
        proto = proto + self.detail_scale.tanh() * self.detail_to_proto(detail)
        candidates = [stem(feature) for stem, feature in zip(self.candidate_stems, [detail, *features])]
        class_features = list(candidates)
        if self.route != "none":
            routed = self.semantic_route(candidates[0], semantic)
            class_features[0] = routed
            if self.route == "shared":
                candidates[0] = routed

        # Independent feature inputs, identical anchors and official output contract.
        # No temporary tensors stored on self: safe for repeated/checkpointed forwards.
        bs = detail.shape[0]
        preds = dict(
            boxes=torch.cat(
                [head(f).view(bs, 4 * self.reg_max, -1) for head, f in zip(self.candidate_boxes, candidates)], 2
            ),
            scores=torch.cat(
                [head(f).view(bs, self.nc, -1) for head, f in zip(self.candidate_classes, class_features)], 2
            ),
            mask_coefficient=torch.cat(
                [head(f).view(bs, self.nm, -1) for head, f in zip(self.candidate_masks, candidates)], 2
            ),
            feats=candidates,
            proto=proto,
        )
        if self.training:
            return preds
        decoded = self._inference(preds)
        return (decoded, proto) if self.export else ((decoded, proto), preds)
