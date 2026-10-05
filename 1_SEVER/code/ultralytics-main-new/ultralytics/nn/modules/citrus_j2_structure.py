# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J2 structural hypotheses: deployable spatial mixing and a short detail-first decoder.

Independent task adaptation of structural reparameterization (RepViT/RepNeXt),
high-resolution discovery (QueryDet/DM-EFS), and complementary structure cues (BDNet).
These are NOT reproductions of the cited models or a model of human perception.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .block import Proto
from .head import Segment

__all__ = ("J2RepMix", "J2RepStage", "J2Contrast", "J2ResidualFuse", "SegmentCitrusJ2Structural")


class J2RepMix(nn.Module):
    """Three linear depthwise branches; exactly mergeable in evaluation mode.

    No nonlinearities occur inside the branch sum. BN statistics must be frozen.
    The equivalence does not hold for training batch-dependent BN.
    """

    def __init__(self, channels):
        super().__init__()
        self.dw3 = Conv(channels, channels, 3, g=channels, act=False)
        self.dw1 = Conv(channels, channels, 1, g=channels, act=False)
        self.identity = nn.BatchNorm2d(channels)
        self.channels = channels

    def forward(self, x):
        if hasattr(self, "deploy"):
            return self.deploy(x)
        return self.dw3(x) + self.dw1(x) + self.identity(x)

    @staticmethod
    def _fold(branch):
        conv = branch.conv
        bias = conv.bias if conv.bias is not None else torch.zeros_like(conv.weight[:, 0, 0, 0])
        if not hasattr(branch, "bn"):
            return conv.weight, bias
        bn = branch.bn
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        return conv.weight * scale[:, None, None, None], bn.bias + (bias - bn.running_mean) * scale

    @torch.no_grad()
    def switch_to_deploy(self):
        """Destructively convert only this module, for inference; caller should copy the model first."""
        if hasattr(self, "deploy"):
            return self
        if self.training:
            raise RuntimeError("J2RepMix fusion requires eval() and frozen BatchNorm statistics")
        w3, b3 = self._fold(self.dw3)
        w1, b1 = self._fold(self.dw1)
        bn = self.identity
        scale = bn.weight / (bn.running_var + bn.eps).sqrt()
        kernel = w3 + F.pad(w1, (1, 1, 1, 1))
        kernel[:, 0, 1, 1] += scale
        bias = b3 + b1 + bn.bias - bn.running_mean * scale
        deploy = nn.Conv2d(self.channels, self.channels, 3, padding=1, groups=self.channels, bias=True)
        deploy = deploy.to(device=kernel.device, dtype=kernel.dtype)
        deploy.weight.copy_(kernel)
        deploy.bias.copy_(bias)
        self.deploy = deploy
        del self.dw3, self.dw1, self.identity
        return self


class _J2Block(nn.Module):
    """Separate spatial and channel mixing; no CSP split/concat or multiplicative GLU."""

    def __init__(self, channels, expansion):
        super().__init__()
        hidden = max(channels, int(channels * expansion))
        self.spatial = J2RepMix(channels)
        self.channel = nn.Sequential(Conv(channels, hidden, 1), Conv(hidden, channels, 1, act=False))
        # Nonzero initial residual permits gradients on both channel convolutions from step one.
        nn.init.constant_(self.channel[-1].bn.weight, 0.1)

    def forward(self, x):
        value = self.spatial(x)
        return value + self.channel(value)


class J2RepStage(nn.Module):
    """Non-CSP stage. Internal repetitions are handled explicitly by parse_model()."""

    def __init__(self, c1, c2, n=1, expansion=2.0):
        super().__init__()
        if expansion < 1 or n < 1:
            raise ValueError("Positive repeats and expansion >= 1 required")
        self.project = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(_J2Block(c2, expansion) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.project(x))


class J2Contrast(nn.Module):
    """Optional bounded, zero-initialized local-contrast residual; does NOT discard RGB.

    Cancellation of a constant offset holds for the centre-minus-local-mean operator
    before the learned convolution, not for arbitrary leaves or the complete network.
    """

    def __init__(self, channels, kernel=5):
        super().__init__()
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("Odd local kernel >= 3 required")
        self.kernel = kernel
        self.project = Conv(channels, channels, 1, act=False)
        self.gain = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        surround = F.avg_pool2d(x, self.kernel, 1, self.kernel // 2, count_include_pad=False)
        return x + 0.25 * self.gain.tanh() * self.project(x - surround)


class J2ResidualFuse(nn.Module):
    """Preserve the detail stream; inject projected semantics with bounded gain.

    Projection is performed at the source resolution. This is one top-down path,
    not a repeated PAN round trip. mode=0 is unweighted addition; mode=1 a bounded
    learnable residual, initially 0.1, so the semantic branch learns immediately.
    """

    def __init__(self, ch, c2, mode=1):
        super().__init__()
        if len(ch) != 2 or mode not in (0, 1):
            raise ValueError("J2ResidualFuse expects [detail, semantics] and mode 0/1")
        self.detail = Conv(ch[0], c2, 1)
        self.semantic = Conv(ch[1], c2, 1)
        self.mode = mode
        self.gain = nn.Parameter(torch.full((1, c2, 1, 1), math.atanh(0.2))) if mode else None
        self.refine = Conv(c2, c2, 3)

    def forward(self, x):
        detail, semantic = self.detail(x[0]), self.semantic(x[1])
        semantic = F.interpolate(semantic, size=detail.shape[-2:], mode="nearest")
        value = detail + (0.5 * self.gain.tanh() * semantic if self.mode else semantic)
        return self.refine(value)


class SegmentCitrusJ2Structural(Segment):
    """Three-scale standard output contract, with optional scale-shared prediction tower.

    P2/P3/P4 candidates address discovery, unlike a P2 prototype-only branch.
    The backbone still includes P5 semantic context. Shared tower and losses are
    separate ablations; no hard colour/roundness gate or hidden score correction.
    """

    def __init__(self, nc=1, nm=32, npr=64, shared=False, tiny_gain=0.0, density=False,
                 proto_stride=2, hidden=32, compact_proto=False, reg_max=16, end2end=False, ch=()):
        if len(ch) != 3 or end2end or tiny_gain < 0 or proto_stride not in (2, 4):
            raise ValueError("Three candidate scales, one-to-many and proto stride 2/4 required")
        super().__init__(nc, nm, npr, reg_max, end2end, tuple(ch))
        self.shared = bool(shared)
        self.tiny_gain, self.boundary_gain, self.scene_gain = float(tiny_gain), 0.0, 0.0
        self.proto_stride = int(proto_stride)
        self.proto_index = 1 if proto_stride == 4 else 0
        if self.proto_index:
            self.proto = Proto(ch[self.proto_index], npr, nm)
        self.density_enabled, self.density_detach = bool(density), False
        self.density_gain = 0.1 if density else 0.0
        if compact_proto:
            # Only a 1x1 projection runs after upsampling. Avoid quadratic-width
            # high-resolution 3x3 prototype convolutions, not just fewer parameters.
            self.proto = nn.Sequential(Conv(ch[self.proto_index], 32, 1), Conv(32, 32, 3, g=32),
                                       nn.Upsample(scale_factor=2, mode="nearest"), nn.Conv2d(32, nm, 1))
        if shared:
            del self.cv2, self.cv3, self.cv4
            self.adapters = nn.ModuleList(Conv(c, hidden, 1) for c in ch)
            # BN is scale-specific; the two spatial kernels are shared without BN.
            self.kernel1 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
            self.kernel2 = nn.Conv2d(hidden, hidden, 3, padding=1, bias=False)
            self.norm1 = nn.ModuleList(nn.BatchNorm2d(hidden) for _ in ch)
            self.norm2 = nn.ModuleList(nn.BatchNorm2d(hidden) for _ in ch)
            self.box_out = nn.ModuleList(nn.Conv2d(hidden, 4 * reg_max, 1) for _ in ch)
            self.cls_out = nn.ModuleList(nn.Conv2d(hidden, nc, 1) for _ in ch)
            self.mask_out = nn.ModuleList(nn.Conv2d(hidden, nm, 1) for _ in ch)
        if density:
            self.density = nn.Sequential(Conv(ch[0], 16, 1), Conv(16, 16, 3), nn.Conv2d(16, 1, 1))
            nn.init.constant_(self.density[-1].bias, -math.log(99.0))

    def bias_init(self):
        if not self.shared:
            return super().bias_init()
        for box, cls, stride in zip(self.box_out, self.cls_out, self.stride):
            box.bias.data.fill_(1.0)
            cls.bias.data[:self.nc] = math.log(5 / self.nc / (640 / stride) ** 2)

    def forward(self, x):
        density = self.density(x[0]) if self.density_enabled else None
        if not self.shared:
            raw = self.forward_head(x, **self.one2many)
            raw["proto"] = self.proto(x[self.proto_index])
        else:
            features = [F.silu(self.norm2[i](self.kernel2(F.silu(self.norm1[i](self.kernel1(a(feat)))))))
                        for i, (a, feat) in enumerate(zip(self.adapters, x))]
            raw = dict(boxes=torch.cat([m(f).flatten(2) for m, f in zip(self.box_out, features)], 2),
                       scores=torch.cat([m(f).flatten(2) for m, f in zip(self.cls_out, features)], 2),
                       mask_coefficient=torch.cat([m(f).flatten(2) for m, f in zip(self.mask_out, features)], 2),
                       feats=x, proto=self.proto(x[self.proto_index]))
        if density is not None and isinstance(raw, dict):
            raw["j2_density"] = density
        if self.training:
            return raw
        decoded = self._inference(raw)
        return (decoded, raw["proto"]) if self.export else ((decoded, raw["proto"]), raw)
