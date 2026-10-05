# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J series: phase-aware encoding, competitive fusion and contrastive context.

Independent task adaptations of FasterNet, GLU, PIDNet and differential attention.
Not reproductions of those complete networks, nor a model of human perception.
Only standard PyTorch operations; no custom CUDA, Mamba, FFT or spatial unfold.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .conv import Conv
from .head import Segment

__all__ = ("JStem", "JDown", "JStage", "JFusion", "JContext", "SegmentCitrusJ")


class JStem(nn.Module):
    """RGB stem; optional invertible rearrangement BEFORE learned compression."""

    def __init__(self, c1, c2, mode=1):
        super().__init__()
        if c1 != 3 or mode not in (0, 1, 2):
            raise ValueError("JStem requires RGB and mode 0/1/2")
        self.mode = mode
        self.register_buffer("luma", torch.tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1))
        self.project = Conv(
            3 if mode == 0 else (20 if mode == 2 else 12), c2, 3 if mode == 0 else 1, 2 if mode == 0 else 1
        )

    def forward(self, x):
        if not self.mode:
            return self.project(x)
        if self.mode == 2:
            luminance = (x * self.luma.to(x.dtype)).sum(1, keepdim=True)
            surround = F.avg_pool2d(luminance, 5, 1, 2, count_include_pad=False)
            x = torch.cat((x, luminance, luminance - surround), 1)
        return self.project(F.pixel_unshuffle(x, 2))


class JDown(nn.Module):
    """Stride-two control or space-to-depth plus learned channel projection."""

    def __init__(self, c1, c2, mode=1):
        super().__init__()
        if mode not in (0, 1):
            raise ValueError("JDown mode must be 0/1")
        self.mode = mode
        self.project = Conv(c1 * (4 if mode else 1), c2, 1 if mode else 3, 1 if mode else 2)

    def forward(self, x):
        return self.project(F.pixel_unshuffle(x, 2) if self.mode else x)


class _JBlock(nn.Module):
    """Quarter-channel spatial mixing with a narrow multiplicative GLU residual."""

    def __init__(self, channels, kernel=5):
        super().__init__()
        self.active = max(channels // 4, 1)
        self.local = Conv(self.active, self.active, 3)
        self.value = Conv(channels, channels, 1)
        self.gate = nn.Sequential(Conv(channels, channels, kernel, g=channels), nn.Conv2d(channels, channels, 1))
        self.out = Conv(channels, channels, 1, act=False)
        self.gain = nn.Parameter(torch.full((1, channels, 1, 1), 0.1))

    def forward(self, x):
        mixed = torch.cat((self.local(x[:, : self.active]), x[:, self.active :]), 1)
        # Unlike SwiGLU's unbounded gate, sigmoid makes this gate lie in (0, 1).
        residual = self.out(self.value(mixed) * self.gate(mixed).sigmoid())
        return x + self.gain.tanh() * residual


class JStage(nn.Module):
    """A new non-CSP stage, with internal repetitions registered in parse_model."""

    def __init__(self, c1, c2, n=1, kernel=5):
        super().__init__()
        if kernel < 3 or kernel % 2 == 0:
            raise ValueError("JStage kernel must be odd and >=3")
        self.project = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(_JBlock(c2, kernel) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.project(x))


class JFusion(nn.Module):
    """Project before resampling; fuse sources once at a specified target scale.

    mode=0: uniform mixture. mode=1: per-pixel convex competitive mixture.
    The convex-hull bound applies BEFORE the final learned projection only.
    """

    def __init__(self, ch, c2, target=0, mode=1):
        super().__init__()
        if not 0 <= target < len(ch) or mode not in (0, 1):
            raise ValueError("Invalid JFusion target/mode")
        self.target, self.mode = target, mode
        self.project = nn.ModuleList(Conv(c, c2, 1) for c in ch)
        self.score = nn.Conv2d(c2 * len(ch), len(ch), 1)
        nn.init.zeros_(self.score.weight)
        nn.init.zeros_(self.score.bias)
        self.refine = nn.Sequential(Conv(c2, c2, 3, g=c2), Conv(c2, c2, 1))

    def aligned(self, x):
        size = x[self.target].shape[-2:]
        values = []
        for feat, project in zip(x, self.project):
            value = project(feat)
            if value.shape[-2:] != size:
                value = (
                    F.adaptive_avg_pool2d(value, size)
                    if value.shape[-2] > size[0]
                    else F.interpolate(value, size, mode="nearest")
                )
            values.append(value)
        return values

    def forward(self, x):
        values = self.aligned(x)
        scores = self.score(torch.cat(values, 1))
        weights = scores.float().softmax(1).to(values[0].dtype)
        if not self.mode:
            weights = weights * 0 + 1.0 / len(values)
        mixed = sum(value * weights[:, i : i + 1] for i, value in enumerate(values))
        return self.refine(mixed)


class JContext(nn.Module):
    """Fixed-size pooled memory; optional zero-row-sum differential reading.

    Attention memory is O(N*m), m=grid**2, NOT O(N**2). Constant values cancel
    exactly in differential mode apart from finite-precision roundoff. This
    does not prove leaves will cancel: real background is spatially variable.
    """

    def __init__(self, channels, mode=2, grid=4, dim=16):
        super().__init__()
        if mode not in (0, 1, 2) or grid < 1 or dim < 1:
            raise ValueError("Invalid JContext configuration")
        self.mode, self.grid, self.dim = mode, grid, dim
        self.qk = nn.Conv2d(channels, 4 * dim, 1, bias=False)
        self.value = nn.Conv2d(channels, dim, 1, bias=False)
        self.out = nn.Conv2d(dim, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(1))

    @staticmethod
    def read(q1, k1, q2, k2, value, differential=True):
        dim = q1.shape[-1]
        a = (q1.float() @ k1.float().transpose(-1, -2) / dim**0.5).softmax(-1)
        b = (q2.float() @ k2.float().transpose(-1, -2) / dim**0.5).softmax(-1)
        return ((a - b) if differential else (a + b * 0)) @ value.float()

    def forward(self, x):
        if not self.mode:
            # Identity ablation does not compute context or add hidden latency.
            return x
        q1, k1, q2, k2 = self.qk(x).chunk(4, 1)
        pool = lambda z: F.adaptive_avg_pool2d(z, (self.grid, self.grid)).flatten(2).transpose(1, 2)
        value = pool(self.value(x))
        result = self.read(
            q1.flatten(2).transpose(1, 2), pool(k1), q2.flatten(2).transpose(1, 2), pool(k2), value, self.mode == 2
        )
        result = result.transpose(1, 2).reshape(x.shape[0], self.dim, *x.shape[-2:]).to(x.dtype)
        return x + 0.25 * self.gain.tanh() * self.out(result)


class SegmentCitrusJ(Segment):
    """Standard candidate contract with a task-separated high-resolution decoder.

    Inputs: P3/P4/P5, preserved P2, stem (stride 2). proto_mode 0/1/2 uses
    standard P3-upsampled, native P2, or P2+stem prototypes. Scene supervision
    produces soft residual recognition modulation, never a hard foreground mask.
    Loss coefficients are explicit YAML arguments, usable through YOLO(yaml).train.
    """

    def __init__(
        self,
        nc=1,
        nm=32,
        npr=64,
        proto_mode=2,
        scene=False,
        tiny_gain=0.0,
        boundary_gain=0.0,
        scene_gain=0.1,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        if len(ch) != 5 or proto_mode not in (0, 1, 2) or end2end:
            raise ValueError("J expects P3/P4/P5/P2/stem; proto 0/1/2; one-to-many only")
        if min(tiny_gain, boundary_gain, scene_gain) < 0:
            raise ValueError("J loss gains must be nonnegative")
        super().__init__(nc, nm, npr, reg_max, end2end, tuple(ch[:3]))
        self.proto_mode, self.scene_enabled = int(proto_mode), bool(scene)
        self.tiny_gain, self.boundary_gain = float(tiny_gain), float(boundary_gain)
        self.scene_gain = float(scene_gain) if scene else 0.0
        self.proto_stride = 2 if proto_mode == 2 else 4
        if proto_mode:
            del self.proto
            self.detail = Conv(ch[3], 32, 1)
            self.semantic = Conv(ch[0], 32, 1)
            if proto_mode == 2:
                self.stem = Conv(ch[4], 16, 1)
            width = 48 if proto_mode == 2 else 32
            self.decoder = nn.Sequential(Conv(width, width, 3, g=width), nn.Conv2d(width, nm, 1))
        if scene:
            self.scene = nn.Sequential(Conv(ch[3], 16, 1), Conv(16, 16, 3, g=16), nn.Conv2d(16, 2, 1))
            self.scene_scale = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        features = list(x[:3])
        # Geometry stays on unmodified features; recognition receives the soft scene prior.
        recognized = features
        scene = None
        if self.scene_enabled:
            scene = self.scene(x[3])
            prior = scene[:, :1].sigmoid()
            recognized = [
                feat * (1 + 0.25 * self.scene_scale.tanh() * F.adaptive_avg_pool2d(prior, feat.shape[-2:]))
                for feat in features
            ]
        preds = dict(
            boxes=torch.cat([self.cv2[i](feat).flatten(2) for i, feat in enumerate(features)], 2),
            scores=torch.cat([self.cv3[i](feat).flatten(2) for i, feat in enumerate(recognized)], 2),
            mask_coefficient=torch.cat([self.cv4[i](feat).flatten(2) for i, feat in enumerate(features)], 2),
            feats=features,
        )
        if scene is not None:
            preds["j_scene"] = scene
        if not self.proto_mode:
            proto = self.proto(features[0])
        else:
            detail = self.detail(x[3])
            semantic = F.interpolate(self.semantic(features[0]), detail.shape[-2:], mode="nearest")
            basis = detail + semantic
            if self.proto_mode == 2:
                basis = torch.cat((F.interpolate(basis, x[4].shape[-2:], mode="nearest"), self.stem(x[4])), 1)
            proto = self.decoder(basis)
        preds["proto"] = proto
        if self.training:
            return preds
        output = self._inference(preds)
        return (output, proto) if self.export else ((output, proto), preds)
