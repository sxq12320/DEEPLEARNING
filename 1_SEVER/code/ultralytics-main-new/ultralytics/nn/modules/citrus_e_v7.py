# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V7: eight screening arms on the inherited V5/V6 control backbone and neck.

P2 adds a narrow stride-4 detection scale. DCNv2 adapts sampling on the
16-channel mask-detail stream. PMCE splits pooled low/high residual features
on fused P3/P4 (and P2), leaving the raw P5 path unchanged. Geometry reuses
SAGEV4R boundary and neighboring-instance losses as one paired intervention.

These are task hypotheses, not demonstrated gains or a full factorial design.
Background errors alone do not establish colour confusion. Geometry excludes
many tiny instances through its area/interior eligibility gate. See
docs/E_V7_RECONSTRUCTION.md for evidence, operator costs and limitations.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v6 import SegmentCitrusEV6
from .conv import Conv, DWConv
from .head import Detect

__all__ = ("EV7DeformBlock", "EV7PMCE", "SegmentCitrusEV7")


class EV7DeformBlock(nn.Module):
    """One DCNv2-style local deformable sampling on the narrow detail stream.

    Zero-initialized offsets make the block start as an ordinary 3x3
    convolution whose contribution is further bounded by ``tanh(scale)``, so
    modulation starts at 0.5 (not a full ordinary convolution). The matched
    torchvision build must supply this compiled op on the selected device.
    """

    def __init__(self, channels: int, kernel: int = 3):
        super().__init__()
        if kernel != 3:
            raise ValueError("EV7DeformBlock is implemented for a single 3x3 sampling")
        self.kernel = kernel
        self.offset = nn.Conv2d(channels, 3 * kernel * kernel, kernel, padding=kernel // 2)
        nn.init.zeros_(self.offset.weight)
        nn.init.zeros_(self.offset.bias)
        self.weight = nn.Conv2d(channels, channels, kernel, padding=kernel // 2)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` plus a bounded deformable residual at the same resolution."""
        from torchvision.ops import deform_conv2d

        field = self.offset(x)
        points = self.kernel * self.kernel
        offset, modulation = field[:, : 2 * points], field[:, 2 * points :].sigmoid()
        deformed = deform_conv2d(
            x, offset, self.weight.weight, self.weight.bias, padding=self.kernel // 2, mask=modulation
        )
        return x + self.scale.tanh() * deformed


class EV7PMCE(nn.Module):
    """Bounded frequency-split multi-scale channel enhancement for detection features.

    A parameter-free Laplacian decomposition separates the reduced feature into
    a low-frequency semantic view (``avg_pool``) and a high-frequency
    boundary/texture view (``f - low``). Each band gets its own spatial
    operator -- dilated context on the low band, local detail on the high band
    -- then a 1x1 fuse and a squeezed global channel gate. The result re-enters
    as a ``tanh``-bounded residual, so the pretrained feature flow is preserved
    at initialization. Task adaptation of the frequency-decomposition idea in
    FEDER/camouflage literature and HWD-style fixed filters; ordinary ops only,
    no wavelet package or FFT.
    """

    def __init__(self, channels: int, hidden: int = 0):
        super().__init__()
        h = hidden or max(16, channels // 2)
        squeeze = max(8, h // 4)
        self.pre = Conv(channels, h, 1)
        self.local = Conv(h, h, 3, g=h)
        self.context = Conv(h, h, 3, p=2, d=2, g=h)
        self.fuse = nn.Conv2d(2 * h, h, 1)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(h, squeeze, 1), nn.SiLU(), nn.Conv2d(squeeze, h, 1), nn.Sigmoid()
        )
        self.out = nn.Conv2d(h, channels, 1, bias=False)
        self.scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return ``x`` enhanced by the gated frequency-split channel residual."""
        f = self.pre(x)
        # Do not turn crop/image padding into a false high-frequency contour.
        low = F.avg_pool2d(f, 3, 1, 1, count_include_pad=False)
        high = f - low
        fused = self.fuse(torch.cat((self.local(high), self.context(low)), 1))
        return x + self.scale.tanh() * self.out(fused * self.gate(fused))


class SegmentCitrusEV7(SegmentCitrusEV6):
    """V6 head plus four separable factors: P2 scale, deformable detail, PMCE, geometry loss.

    Input layout is ``[P3, P4, P5, (P2), detail, (stem)]``: ``p2_head`` appends
    a stride-4 feature AFTER the three inherited scales so pretrained
    ``cv2``/``cv3``/``cv4``/``quality_predictor`` indices 0--2 stay aligned with
    P3/P4/P5, and the appended fourth branch trains from scratch. ``boundary``
    and ``neighbor`` gains are read by :class:`SAGEV4RSegmentationLoss` from
    head attributes; both default to the historical zero.
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
        p2_head=False,
        deform_detail=False,
        pmce=False,
        boundary_gain=0.0,
        neighbor_gain=0.0,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        self.p2_head = bool(p2_head)
        self.n_det = 4 if self.p2_head else 3
        expected = self.n_det + 1 + (1 if fine_mask else 0)
        if len(ch) != expected:
            raise ValueError(
                f"EV7 expects {expected} head inputs ({self.n_det} scales + detail"
                f"{' + stem' if fine_mask else ''}), got {len(ch)}"
            )
        detail_ch = ch[self.n_det]
        stem_ch = ch[self.n_det + 1] if fine_mask else 0
        parent_ch = tuple(ch[:3]) + (detail_ch,) + ((stem_ch,) if fine_mask else ())
        super().__init__(
            nc, nm, npr, detail_channels, mask_route, region_gain, fine_mask, reg_max, end2end, parent_ch
        )
        for name, value in (("boundary_gain", boundary_gain), ("neighbor_gain", neighbor_gain)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
            setattr(self, name, float(value))
        self.deform_detail = bool(deform_detail)
        self.pmce = bool(pmce)
        if self.deform_detail:
            self.deform = EV7DeformBlock(detail_channels)
        if self.pmce:
            # Enhance only PAN-fused neck features (P3, P4, and P2 when present);
            # the raw backbone C2PSA map is left untouched.
            self.pmce_idx = (0, 1) + ((3,) if self.p2_head else ())
            self.enhance = nn.ModuleList(EV7PMCE(ch[i]) for i in self.pmce_idx)
        if self.p2_head:
            self._append_p2_towers(ch[3])
            self.nl = self.n_det
            self.stride = torch.zeros(self.nl)

    def _append_p2_towers(self, c_p2: int):
        """Append one narrow stride-4 branch to every per-scale tower and the quality head."""
        hidden = max(16, c_p2 // 2)
        self.cv2.append(
            nn.Sequential(Conv(c_p2, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, 4 * self.reg_max, 1))
        )
        self.cv3.append(
            nn.Sequential(
                nn.Sequential(DWConv(c_p2, c_p2, 3), Conv(c_p2, hidden, 1)),
                nn.Sequential(DWConv(hidden, hidden, 3), Conv(hidden, hidden, 1)),
                nn.Conv2d(hidden, self.nc, 1),
            )
        )
        self.cv4.append(nn.Sequential(Conv(c_p2, hidden, 3), nn.Conv2d(hidden, self.nm, 1)))
        predictor = nn.Sequential(Conv(c_p2 + self.nm, 16, 1), nn.Conv2d(16, 1, 1))
        nn.init.zeros_(predictor[-1].weight)
        nn.init.constant_(predictor[-1].bias, 1.38629436)
        self.quality_predictor.append(predictor)

    def forward(self, x):
        n = self.n_det
        features = list(x[:n])
        if self.pmce:
            for enhance, idx in zip(self.enhance, self.pmce_idx):
                features[idx] = enhance(features[idx])
        detail = self.refiner(x[n], features[0])
        mask_detail = detail
        if self.mask_route:
            context = F.interpolate(self.mask_context(features[0]), detail.shape[-2:], mode="nearest")
            update = self.mask_update(torch.cat((detail, context), 1))
            mask_detail = detail + self.mask_route_scale.tanh() * update
        if self.deform_detail:
            mask_detail = self.deform(mask_detail)
        target = (features[0].shape[-2] * 2, features[0].shape[-1] * 2)
        routed = detail if detail.shape[-2:] == target else F.interpolate(detail, target, mode="nearest")
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(routed, 2))
        proto = self.proto(features[0])
        if mask_detail.shape[-2:] != proto.shape[-2:]:
            mask_detail = F.interpolate(mask_detail, proto.shape[-2:], mode="nearest")
        proto = proto + self.detail_scale.tanh() * self.detail_to_proto(mask_detail)
        if self.fine_mask:
            proto = self.refine_fine_proto(proto, mask_detail, x[n + 1])
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
