# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V9: task-separated mask fusion, preserving the empirically useful detector.

Independent adaptations of separate mask/instance paths (SparseInst), semantic
shape guidance (Gated-SCNN/PIDNet) and fine-feature decoding (RefineMask).
No CPU edge extraction, recurrent controller, sparse CUDA op or extra P2 anchors.
The optional mask neck is AFTER the detail-to-detection relay: it cannot replace
the P4 spatial reconstruction, whose removal hurt E V8 in the supplied results.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v7 import SegmentCitrusEV7
from .conv import Conv
from .head import Detect

__all__ = ("EV9MaskNeck", "EV9CompactProto", "SegmentCitrusEV9")


class EV9MaskNeck(nn.Module):
    """Native C4/C5 context corrects a narrow mask stream, not detection features.

    A local residual band is kept separate from the semantic discrepancy until
    the last gated update. High-frequency energy is NOT assumed to be fruit:
    the gate also sees the semantic reference. All high-resolution spatial
    operators work on 16 channels, never on a full C4/C5 feature tensor.
    """

    def __init__(self, c4, c5, channels=16):
        super().__init__()
        self.c4 = Conv(c4, channels, 1)
        self.c5 = Conv(c5, channels, 1)
        self.context = Conv(channels, channels, 3)
        self.edge = Conv(channels, channels, 3, g=channels)
        self.gate = nn.Conv2d(3 * channels, 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, detail, c4, c5):
        native = self.c4(c4)
        deep = F.interpolate(self.c5(c5), native.shape[-2:], mode="nearest")
        semantic = self.context((native + deep) * 0.5)
        semantic = F.interpolate(semantic, detail.shape[-2:], mode="bilinear", align_corners=False)
        low = F.avg_pool2d(detail, 3, 1, 1, count_include_pad=False)
        residual = detail - low
        weights = self.gate(torch.cat((semantic, low, residual.abs()), 1)).softmax(dim=1)
        update = weights[:, :1] * (semantic - low) + weights[:, 1:] * self.edge(residual)
        return detail + self.gain.tanh() * update


class EV9CompactProto(nn.Module):
    """Decode a shared mask basis with a 32-channel P3 -> P2 path.

    Replaces, rather than supplements, the original wide Proto stack. Detail
    enters before the spatial refinement. Final stride-2 phase refinement is
    inherited and unchanged. New parameter names prevent accidental transfer
    from unrelated same-shaped layers in the official pretrained Proto.
    """

    def __init__(self, c_p3, c_detail=16, nm=32, width=32):
        super().__init__()
        self.semantic = Conv(c_p3, width, 1)
        self.detail = Conv(c_detail, width, 1)
        self.spatial = nn.Sequential(Conv(width, width, 3, g=width), Conv(width, width, 1))
        self.basis = nn.Conv2d(width, nm, 1)

    def forward(self, p3, detail):
        context = F.interpolate(self.semantic(p3), detail.shape[-2:], mode="bilinear", align_corners=False)
        return self.basis(self.spatial(context + self.detail(detail)))


class SegmentCitrusEV9(SegmentCitrusEV7):
    """Preserve P3/P4/P5 towers; independently ablate mask neck, basis and losses.

    Inputs always [P3, P4, C5, C2, stem, C4]. Loss gains are YAML settings so the
    official YOLO(YAML) entry constructs the correct criterion. Dataset mixing,
    copy-paste and the optional cosine schedule belong to the batch recipe.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        mask_neck=False,
        compact_proto=False,
        tiny_dice_gain=0.0,
        negative_quality_gain=0.0,
        boundary_gain=0.5,
        neighbor_gain=0.25,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if len(ch) != 6 or end2end:
            raise ValueError("E V9 requires [P3,P4,C5,C2,stem,C4], one-to-many segmentation")
        super().__init__(
            nc,
            nm,
            npr,
            detail_channels,
            False,
            0.0,
            "phase",
            False,
            False,
            False,
            boundary_gain,
            neighbor_gain,
            reg_max,
            end2end,
            tuple(ch[:5]),
        )
        self.mask_neck = bool(mask_neck)
        self.compact_proto = bool(compact_proto)
        for name, value in (("tiny_dice_gain", tiny_dice_gain), ("negative_quality_gain", negative_quality_gain)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, float(value))
        if self.mask_neck:
            self.mask_fusion = EV9MaskNeck(ch[5], ch[2], detail_channels)
        if self.compact_proto:
            self.proto = EV9CompactProto(ch[0], detail_channels, nm)
            del self.detail_to_proto
            del self.detail_scale

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        # The inherited detection relay is kept identical, before mask-only updates.
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(detail, 2))
        mask_detail = self.mask_fusion(detail, x[5], x[2]) if self.mask_neck else detail
        if self.compact_proto:
            proto = self.proto(features[0], mask_detail)
        else:
            proto = self.proto(features[0]) + self.detail_scale.tanh() * self.detail_to_proto(mask_detail)
        proto = self.refine_fine_proto(proto, mask_detail, x[4])
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
