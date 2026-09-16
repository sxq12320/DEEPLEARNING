# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V10: colour-complementary stem and a narrow early-detail transport path.

Task adaptations of context/texture separation (DGNet), high-resolution
transport (Lite-HRNet), and fine mask decoding (RefineMask). Not reproductions,
not PID, not colour invariance, and not proven gains. RGB and the useful
spatial P4 reconstruction are preserved. No sparse/DCN/Canny/CPU operators.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v9 import SegmentCitrusEV9
from .conv import Conv
from .head import Detect

__all__ = ("EV10ContrastStem", "EV10DetailTransport", "EV10BalancedProto", "SegmentCitrusEV10")


class EV10ContrastStem(Conv):
    """Keep pretrained RGB Conv keys and add a zero-start luminance-contrast path.

    This is complementary evidence, not removal of colour or a guarantee of
    colour invariance. Variance floor and bounded residual avoid amplifying
    flat-region noise. Replicate padding avoids artificial image-edge cues.
    """

    def __init__(self, c1, c2, k=3, s=2):
        if c1 != 3 or k != 3 or s != 2:
            raise ValueError("EV10ContrastStem expects RGB and a 3x3 stride-2 stem")
        super().__init__(c1, c2, k, s)
        self.structure = Conv(1, c2, 3, 2)
        self.structure.register_parameter("gain", nn.Parameter(torch.zeros(1, c2, 1, 1)))
        self.register_buffer("luminance", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    def contrast(self, x):
        gray = (x.float() * self.luminance.float()).sum(1, keepdim=True)
        mean = F.avg_pool2d(F.pad(gray, (3, 3, 3, 3), mode="replicate"), 7, 1)
        second = F.avg_pool2d(F.pad(gray.square(), (3, 3, 3, 3), mode="replicate"), 7, 1)
        variance = (second - mean.square()).clamp_min(0)
        return ((gray - mean) / (variance + 0.05**2).sqrt()).tanh().to(x.dtype)

    def forward(self, x):
        return super().forward(x) + self.structure.gain.tanh() * self.structure(self.contrast(x))

    def forward_fuse(self, x):
        # BaseModel.fuse rebinds Conv.forward; the extra path must remain present.
        return super().forward_fuse(x) + self.structure.gain.tanh() * self.structure(self.contrast(x))


class EV10DetailTransport(nn.Module):
    """Stem phases reach P2 before the detection relay, not only final masks.

    A 16ch P2 stream combines early spatial evidence with a C4 context gate.
    Local centre-surround differences are explicit signals, NOT subtracting
    one shared constant from softmax logits (which has no effect). Context
    guides a spatial gate, with one bounded update and no repeated backbone.
    """

    def __init__(self, c_stem, c4, channels=16):
        super().__init__()
        self.early = Conv(c_stem, 8, 1)
        self.phase = Conv(32, channels, 1)
        self.context = Conv(c4, channels, 1)
        self.mix = nn.Sequential(Conv(3 * channels, channels, 1), Conv(channels, channels, 3, g=channels))
        self.gate = nn.Conv2d(3 * channels, channels, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.05))

    def forward(self, detail, stem, c4):
        early = self.phase(F.pixel_unshuffle(self.early(stem), 2))
        if early.shape[-2:] != detail.shape[-2:]:
            raise ValueError("V10 stem/P2 grids must have stride ratio two")
        context = F.interpolate(self.context(c4), detail.shape[-2:], mode="bilinear", align_corners=False)
        near = detail - F.avg_pool2d(detail, 3, 1, 1, count_include_pad=False)
        far = detail - F.avg_pool2d(detail, 7, 1, 3, count_include_pad=False)
        update = self.mix(torch.cat((early - detail, near, far), 1))
        gate = self.gate(torch.cat((detail, context, (early - detail).abs()), 1)).sigmoid()
        return detail + self.gain.tanh() * gate * update


class EV10BalancedProto(nn.Module):
    """Moderate 64ch decoder: compress at P3, retain ordinary spatial mixing at P2.

    V9's 32ch depthwise-only compact basis lost AP. This replaces the wide
    prototype stack with a less aggressive alternative, not another branch.
    New state keys prevent accidental same-shape transfer from old Proto.
    """

    def __init__(self, c_p3, c_detail=16, nm=32, width=64):
        super().__init__()
        self.semantic = Conv(c_p3, width, 1)
        self.detail = Conv(c_detail, width, 1)
        self.spatial = Conv(width, width, 3)
        self.basis = nn.Conv2d(width, nm, 1)

    def forward(self, p3, detail):
        semantic = F.interpolate(self.semantic(p3), detail.shape[-2:], mode="bilinear", align_corners=False)
        return self.basis(self.spatial(semantic + self.detail(detail)))


class SegmentCitrusEV10(SegmentCitrusEV9):
    """Preserve detection topology; separate early-detail, decoder and supervision factors.

    Inputs [P3,P4,C5,C2,stem,C4]. Auxiliary visible-foreground prediction is
    TRAINING ONLY and supervises all nonempty GT masks, even without TAL
    positives. It is not a replacement for instance assignment or segmentation.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        transport=False,
        balanced_proto=False,
        tiny_gain=0.25,
        visibility_gain=0.0,
        boundary_gain=0.5,
        neighbor_gain=0.25,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        super().__init__(
            nc,
            nm,
            npr,
            detail_channels,
            False,
            False,
            tiny_gain,
            0.0,
            boundary_gain,
            neighbor_gain,
            reg_max,
            end2end,
            ch,
        )
        if not math.isfinite(visibility_gain) or visibility_gain < 0:
            raise ValueError("visibility_gain must be finite and nonnegative")
        self.transport = bool(transport)
        self.balanced_proto = bool(balanced_proto)
        self.visibility_gain = float(visibility_gain)
        if self.transport:
            self.detail_transport = EV10DetailTransport(ch[4], ch[5], detail_channels)
        if self.balanced_proto:
            self.proto = EV10BalancedProto(ch[0], detail_channels, nm)
            del self.detail_to_proto
            del self.detail_scale
        if self.visibility_gain:
            self.visibility = nn.Conv2d(detail_channels, 1, 1)

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        if self.transport:
            detail = self.detail_transport(detail, x[4], x[5])
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(detail, 2))
        if self.balanced_proto:
            proto = self.proto(features[0], detail)
        else:
            proto = self.proto(features[0]) + self.detail_scale.tanh() * self.detail_to_proto(detail)
        proto = self.refine_fine_proto(proto, detail, x[4])
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                if self.visibility_gain:
                    preds["ev10_visibility"] = self.visibility(detail)
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
