# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V2: spatial/channel separation, shared context hub, and contrast detail routing.

Independent citrus adaptations of RepViT (CVPR24), Gold-YOLO (NeurIPS23),
FreqFusion (TPAMI24) and DGNet (MIR23) ideas; not official reproductions.
Only ordinary Conv/BN/RepConv, pooling, resize and pointwise gates. No custom
CUDA, deformable resampling, external Mamba, or extra target-dependent head.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .citrus_sage_v5 import SegmentCitrusSAGEV5
from .conv import Conv, RepConv

__all__ = ("EV2RepStage", "EV2Down", "EV2ContextHub", "EV2ContextInject", "SegmentCitrusEV2")


class EV2Mixer(nn.Module):
    """Reparameterizable spatial mixing followed by a separate residual channel MLP."""

    def __init__(self, channels):
        super().__init__()
        self.spatial = RepConv(channels, channels, 3, g=channels, bn=True, act=False)
        # Equal-width mixing is intentional: expanding the persistent C2/C3
        # channels made the first prototype more expensive than its control.
        self.expand = Conv(channels, channels, 1, act=nn.GELU())
        self.contract = Conv(channels, channels, 1, act=False)
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        x = self.spatial(x)
        return x + self.gain.tanh() * self.contract(self.expand(x))


class EV2RepStage(nn.Module):
    """Replace a whole CSP stage; no channel split/concatenation or nested C3k2 blocks."""

    def __init__(self, c1, c2, n=1):
        super().__init__()
        self.entry = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(EV2Mixer(c2) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.entry(x))


class EV2Down(nn.Module):
    """Separate spatial reduction and channel projection; not a lossless downsampler."""

    def __init__(self, c1, c2):
        super().__init__()
        self.reduce = Conv(c1, c1, 5, 2, g=c1)
        self.project = Conv(c1, c2, 1)

    def forward(self, x):
        return self.project(self.reduce(x))


def resize_feature(x, size):
    if x.shape[-2:] == size:
        return x
    if x.shape[-2] >= size[0] and x.shape[-1] >= size[1]:
        if x.shape[-2] % size[0] == 0 and x.shape[-1] % size[1] == 0:
            factor = (x.shape[-2] // size[0], x.shape[-1] // size[1])
            return F.avg_pool2d(x, factor, factor)
        return F.adaptive_avg_pool2d(x, size)
    return F.interpolate(x, size=size, mode="nearest")


class EV2ContextHub(nn.Module):
    """Gather C3/C4/C5 once at C4; P3 details bypass this low-resolution summary."""

    def __init__(self, ch, c2):
        super().__init__()
        if len(ch) != 3:
            raise ValueError("Hub requires C3, C4, C5")
        self.projections = nn.ModuleList(Conv(c, c2, 1) for c in ch)
        self.mix = nn.Sequential(Conv(3 * c2, c2, 1), EV2Mixer(c2))

    def forward(self, x):
        size = x[1].shape[-2:]
        return self.mix(torch.cat([resize_feature(p(f), size) for p, f in zip(self.projections, x)], 1))


class EV2ContextInject(nn.Module):
    """Parallel hub-to-scale distribution with a local bypass, not another PAN round trip."""

    def __init__(self, ch, c2):
        super().__init__()
        self.local = Conv(ch[0], c2, 1)
        self.context = Conv(ch[1], c2, 1, act=False)
        self.gate = nn.Conv2d(ch[1], c2, 1)
        self.refine = nn.Sequential(Conv(c2, c2, 3, g=c2), Conv(c2, c2, 1, act=False))
        self.gain = nn.Parameter(torch.full((1, c2, 1, 1), 0.1))

    def forward(self, x):
        local, hub = x
        local = self.local(local)
        context = resize_feature(self.context(hub), local.shape[-2:])
        gate = resize_feature(self.gate(hub).sigmoid(), local.shape[-2:])
        return local + self.gain.tanh() * self.refine(context * gate)


class EV2ContrastDetail(nn.Module):
    """Retain C2 while adding semantic-conditioned centre-surround feature contrast.

    A mean residual is an explicit high-pass feature, not a fruit contour label.
    The RGB main path and semantic context remain necessary for leaf rejection.
    """

    def __init__(self, c_detail, c_semantic, channels):
        super().__init__()
        self.local = Conv(c_detail, channels, 1)
        self.context = Conv(c_semantic, channels, 1, act=False)
        self.gate = nn.Conv2d(3 * channels, channels, 1)
        self.correction = nn.Sequential(Conv(2 * channels, channels, 1), Conv(channels, channels, 3, g=channels))
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, c2, p3):
        local = self.local(c2)
        context = resize_feature(self.context(p3), local.shape[-2:])
        smooth = F.avg_pool2d(local, 5, 1, 2, count_include_pad=False)
        detail = local - smooth
        gate = self.gate(torch.cat((local, context, (local - context).abs()), 1)).sigmoid()
        correction = self.correction(torch.cat((context - smooth, gate * detail), 1))
        return local + self.gain.tanh() * correction


class SegmentCitrusEV2(SegmentCitrusSAGEV5):
    """Contrast detail shared by P3 candidate relay and stride4 instance prototypes."""

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        relay=True,
        late_proto=False,
        boundary_gain=0.0,
        neighbor_gain=0.0,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        super().__init__(
            nc, nm, npr, detail_channels, relay, late_proto, boundary_gain, neighbor_gain, reg_max, end2end, ch
        )
        self.refiner = EV2ContrastDetail(ch[3], ch[0], detail_channels)
