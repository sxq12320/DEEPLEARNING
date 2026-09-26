# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""I V2: evidence-driven high-resolution candidate recovery for tiny citrus.

I V1 changed the mask prototypes after candidates had already been generated.
Its best synchronized arm was tied with the V12 replay control on one leaked
validation split, while misses remained strongly concentrated below 16 pixels.
I V2 therefore moves the primary intervention upstream: a narrow stride-4
candidate feature is constructed and is allowed to enter the detector.

The P2 fusion is an independent task adaptation of two published observations:
SET (CVPR 2025) shows that background high-frequency energy can be harmful, and
LSNet (CVPR 2025) separates broad perception from local aggregation. It does
not copy either full module. Only ordinary PyTorch convolutions are used; there
is no Triton, FFT, Mamba, unfold, deformable convolution, or second training
forward pass. Accuracy, camouflage discrimination, and speed remain empirical
hypotheses to test on the leakage-free grouped_dedup split.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Sequence

from .block import C3k2
from .citrus_e_v12 import SegmentCitrusEV12
from .conv import Conv, DWConv
from .head import Detect

__all__ = ("IV2LargeSmallStage", "IV2P2Candidate", "SegmentCitrusIV2")


class IV2LargeSmallMixer(nn.Module):
    """Broad depthwise perception gates a local depthwise aggregation residual."""

    def __init__(self, channels: int, kernel: int = 7):
        super().__init__()
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("IV2 large-small kernel must be an odd integer >=3")
        self.broad = Conv(channels, channels, kernel, p=kernel // 2, g=channels)
        self.local = Conv(channels, channels, 3, g=channels)
        self.gate = nn.Conv2d(channels, channels, 1)
        self.out = Conv(channels, channels, 1, act=False)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.out.bn.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Keep an identity start while learning context-conditioned local aggregation."""
        weight = self.gate(self.broad(x)).sigmoid()
        return x + self.out(weight * self.local(x))


class IV2LargeSmallStage(C3k2):
    """C3k2-compatible stage whose inner mixers use the I V2 large-small rule."""

    def __init__(self, c1: int, c2: int, n: int = 1, c3k: bool = True, e: float = 0.5, kernel: int = 7):
        super().__init__(c1, c2, n, c3k, e)
        self.m = nn.ModuleList(IV2LargeSmallMixer(self.c, kernel) for _ in range(n))


class IV2P2Candidate(nn.Module):
    """Build one narrow P2 candidate map from persistent detail and P3 semantics.

    ``mode=0`` is a raw projected-detail control. ``mode=1`` gates the local
    high-frequency residual using an upsampled semantic reference. ``mode=2``
    first applies a large depthwise field to that reference, separating broad
    perception from local residual aggregation. The gate is initialized at
    0.2 so an untrained branch does not automatically amplify every leaf edge.
    """

    def __init__(self, ch: Sequence[int], out_channels: int = 64, mode: int = 1, kernel: int = 7):
        super().__init__()
        if len(ch) != 2:
            raise ValueError("IV2P2Candidate expects [detail, P3] inputs")
        if mode not in (0, 1, 2):
            raise ValueError("IV2 P2 mode must be 0 (raw), 1 (semantic gate), or 2 (large-small gate)")
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("IV2 P2 context kernel must be an odd integer >=3")
        self.mode = int(mode)
        self.detail = Conv(ch[0], out_channels, 1)
        if self.mode:
            self.reference = Conv(ch[1], out_channels, 1)
            self.broad = (
                Conv(out_channels, out_channels, kernel, p=kernel // 2, g=out_channels)
                if self.mode == 2
                else nn.Identity()
            )
            self.local = Conv(out_channels, out_channels, 3, g=out_channels)
            self.gate = nn.Conv2d(3 * out_channels, 1, 1)
            self.fuse = Conv(2 * out_channels, out_channels, 1)
            nn.init.zeros_(self.gate.weight)
            nn.init.constant_(self.gate.bias, -1.38629436)  # sigmoid -> 0.2

    def forward(self, x: Sequence[torch.Tensor]) -> torch.Tensor:
        detail, p3 = x
        detail = self.detail(detail)
        if not self.mode:
            return detail
        reference = F.interpolate(self.reference(p3), detail.shape[-2:], mode="nearest")
        reference = self.broad(reference)
        low = F.avg_pool2d(detail, 3, 1, 1, count_include_pad=False)
        high = detail - low
        keep = self.gate(torch.cat((low, reference, high.abs()), 1)).sigmoid()
        filtered = low + keep * self.local(high)
        return self.fuse(torch.cat((filtered, reference), 1))


class SegmentCitrusIV2(SegmentCitrusEV12):
    """V12 head with an optional appended P2 detection scale.

    A three-scale control uses ``[P3, P4, P5, C2, stem, C4, persistent_P2]``.
    A P2 arm inserts ``P2_candidate`` after P5. P2 is appended after the
    inherited P3/P4/P5 prediction towers so their state keys remain aligned.
    The official quality-aware mask head and V12 recognition route are retained;
    no I V1 dual prototype is carried forward because its one-seed gain was not
    material.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        persistent=True,
        legacy_transport=False,
        factorized_box=False,
        assignment_mix=0.0,
        ring_gain=0.0,
        tiny_gain=0.25,
        boundary_gain=0.5,
        neighbor_gain=0.25,
        route=2,
        difference=True,
        p2_head=False,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        parent_expected = 7 if persistent else 6
        expected = parent_expected + int(bool(p2_head))
        if len(ch) != expected:
            raise ValueError(f"I V2 expects {expected} inputs for p2_head={bool(p2_head)}, got {len(ch)}")
        parent_ch = tuple(ch[:3]) + tuple(ch[4:]) if p2_head else tuple(ch)
        super().__init__(
            nc,
            nm,
            npr,
            detail_channels,
            persistent,
            legacy_transport,
            factorized_box,
            assignment_mix,
            ring_gain,
            tiny_gain,
            boundary_gain,
            neighbor_gain,
            route,
            difference,
            reg_max,
            end2end,
            parent_ch,
        )
        self.p2_head = bool(p2_head)
        if self.p2_head:
            self._append_p2_towers(ch[3])
            self.nl = 4
            self.stride = torch.zeros(self.nl)

    def _append_p2_towers(self, channels: int) -> None:
        """Append ordinary lightweight towers without shifting pretrained scales."""
        hidden = max(16, channels // 2)
        self.cv2.append(
            nn.Sequential(Conv(channels, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, 4 * self.reg_max, 1))
        )
        self.cv3.append(
            nn.Sequential(
                nn.Sequential(DWConv(channels, channels, 3), Conv(channels, hidden, 1)),
                nn.Sequential(DWConv(hidden, hidden, 3), Conv(hidden, hidden, 1)),
                nn.Conv2d(hidden, self.nc, 1),
            )
        )
        self.cv4.append(nn.Sequential(Conv(channels, hidden, 3), nn.Conv2d(hidden, self.nm, 1)))
        quality = nn.Sequential(Conv(channels + self.nm, 16, 1), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(quality[-1].weight)
        nn.init.constant_(quality[-1].bias, 1.38629436)
        self.quality_predictor.append(quality)

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None):
        """Extend V12 recognition correction to a fourth, unchanged P2 scale."""
        if not self.recognition_route:
            return super().forward_head(x, box_head, cls_head, mask_head)
        recognized = [self.recognition[i](x[i], x[i + 1]) for i in range(2)] + list(x[2:])
        geometry = recognized if self.recognition_route == 2 else x
        batch = x[0].shape[0]
        boxes = torch.cat(
            [box_head[i](geometry[i]).view(batch, 4 * self.reg_max, -1) for i in range(self.nl)], -1
        )
        scores = torch.cat([cls_head[i](recognized[i]).view(batch, self.nc, -1) for i in range(self.nl)], -1)
        maps = [mask_head[i](geometry[i]) for i in range(self.nl)]
        coefficients = torch.cat([item.view(batch, self.nm, -1) for item in maps], -1)
        quality = [
            self.quality_predictor[i](torch.cat((geometry[i].detach(), item.detach()), 1))
            for i, item in enumerate(maps)
        ]
        return dict(
            boxes=boxes,
            scores=scores,
            feats=x,
            mask_coefficient=coefficients,
            ev3_quality=torch.cat([item.view(batch, 1, -1) for item in quality], -1),
        )

    def forward(self, x):
        """Construct prototypes from the proven shared path and optionally detect on P2."""
        features = list(x[:3])
        base = 4 if self.p2_head else 3
        detail = self.refiner(x[base], features[0])
        if self.transport:
            detail = self.detail_transport(detail, x[base + 1], x[base + 2])
        if self.persistent:
            detail = detail + self.persistent_gain.tanh() * (x[base + 3] - detail)
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(detail, 2))
        proto = self.proto(features[0]) + self.detail_scale.tanh() * self.detail_to_proto(detail)
        proto = self.refine_fine_proto(proto, detail, x[base + 1])
        if self.p2_head:
            features.append(x[3])
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
