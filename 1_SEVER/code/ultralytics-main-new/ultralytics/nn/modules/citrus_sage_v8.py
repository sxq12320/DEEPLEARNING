# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V8: preserve task capacity, redistribute candidate scales, test luminance detail.

Independent task adaptation, not a reproduction of RTMDet, SPD-Conv or PiDiNet.
The P3/P4 towers retain their original state keys and weights. Removing P5
prediction does NOT remove C5 context from the backbone/neck. No custom CUDA,
spatial attention matrix, auxiliary loss or GT-dependent inference is used.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_sage_v5 import SegmentCitrusSAGEV5
from .conv import Conv
from .head import Detect

__all__ = ("SAGEV8PhaseStem", "SegmentCitrusSAGEV8")


class SAGEV8PhaseStem(Conv):
    """Pretrained RGB stem plus a narrow, brightness-offset-invariant detail path.

    Reindex four luminance phases before subtracting their block mean. The
    differences are insensitive to a uniform additive brightness offset, NOT
    invariant to illumination/colour in general. RGB semantic information is
    retained in the original convolution. Projection is not lossless.
    """

    def __init__(self, c1, c2, k=3, s=2):
        if c1 != 3 or k != 3 or s != 2:
            raise ValueError("PhaseStem is an RGB 3x3 stride2 entrance only")
        super().__init__(c1, c2, k, s)
        self.register_buffer("luma_weights", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))
        self.shape_path = nn.Sequential(Conv(4, 8, 1), Conv(8, c2, 3))
        # Keep the RGB convolution as the first parameter: the existing FLOPs
        # utility infers input channels from that parameter's second dimension.
        self.gains = nn.ParameterList([nn.Parameter(torch.full((1, c2, 1, 1), 0.01))])

    @property
    def shape_gain(self):
        return self.gains[0]

    def phase_detail(self, x):
        luma = (x * self.luma_weights).sum(1, keepdim=True)
        # Replicate on odd sizes to match the original stride2 convolution.
        luma = F.pad(luma, (0, luma.shape[-1] % 2, 0, luma.shape[-2] % 2), mode="replicate")
        phases = F.pixel_unshuffle(luma, 2)
        return phases - phases.mean(1, keepdim=True)

    def forward(self, x):
        return super().forward(x) + self.shape_gain.tanh() * self.shape_path(self.phase_detail(x))

    def forward_fuse(self, x):
        # BaseModel.fuse replaces forward with forward_fuse; do not lose the bypass.
        return super().forward_fuse(x) + self.shape_gain.tanh() * self.shape_path(self.phase_detail(x))


class SegmentCitrusSAGEV8(SegmentCitrusSAGEV5):
    """Independent narrow P2 towers + preserved P3/P4(/P5) task towers.

    retain_p5=False reallocates prediction work to P2/P3/P4. C5 still feeds
    the asymmetric semantic neck. Coefficients and prototypes remain separate;
    the usual TAL, DFL, box and instance-mask objectives are unchanged.
    """

    def __init__(
        self, nc=80, nm=32, npr=256, detail_channels=16, width=16,
        retain_p5=True, reg_max=16, end2end=False, ch=(),
    ):
        if width < 8 or width % 8:
            raise ValueError("P2 width must be a positive multiple of eight")
        super().__init__(nc, nm, npr, detail_channels, True, False, 0, 0, reg_max, end2end, ch)
        self.retain_p5 = bool(retain_p5)
        # C5 still exists even if P5 prediction is removed. Letterbox must use
        # /32 padding, while anchor decoding/TAL use the actual [4,8,16] strides.
        self.required_input_stride = 32
        self.coarse_levels = 3 if retain_p5 else 2
        if not retain_p5:
            self.cv2 = self.cv2[:2]
            self.cv3 = self.cv3[:2]
            self.cv4 = self.cv4[:2]
        self.nl = self.coarse_levels + 1
        self.stride = torch.zeros(self.nl)
        # Do NOT insert into cv2/cv3/cv4: that shifts the pretrained P3/P4 keys.
        self.p2_box = nn.Sequential(Conv(detail_channels, width, 3), nn.Conv2d(width, 4 * reg_max, 1))
        self.p2_cls = nn.Sequential(Conv(detail_channels, width, 3), nn.Conv2d(width, nc, 1))
        self.p2_mask = nn.Sequential(Conv(detail_channels, width, 3), nn.Conv2d(width, nm, 1))

    @property
    def one2many(self):
        return dict(
            box_head=[self.p2_box, *self.cv2],
            cls_head=[self.p2_cls, *self.cv3],
            mask_head=[self.p2_mask, *self.cv4],
        )

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        target = (2 * features[0].shape[-2], 2 * features[0].shape[-1])
        routed = detail if detail.shape[-2:] == target else F.interpolate(detail, size=target, mode="nearest")
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(routed, 2))
        proto = self.proto(features[0])
        if detail.shape[-2:] != proto.shape[-2:]:
            detail = F.interpolate(detail, size=proto.shape[-2:], mode="nearest")
        proto = proto + self.detail_scale.tanh() * self.detail_to_proto(detail)
        outputs = Detect.forward(self, [detail, *features[:self.coarse_levels]])
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
