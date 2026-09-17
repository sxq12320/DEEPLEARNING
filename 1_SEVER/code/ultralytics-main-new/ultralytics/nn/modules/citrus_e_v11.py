# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V11: a thin persistent detail backbone, native-scale neck and tiny-aware matching.

Independent task adaptations of Lite-HRNet's parallel resolution transport,
RepViT's separated spatial/channel mixing, and FreqFusion's consistency/detail
separation. No CARAFE, unfold, scan, deformable/CPU edge operators or new P2
anchor tower. A bounded feed-forward correction is NOT a PID controller or a
proof of closed-loop stability. Accuracy and GPU speed require experiments.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C2PSA, C3k2
from .citrus_e_v10 import SegmentCitrusEV10
from .conv import Conv, RepConv
from .head import Detect

__all__ = (
    "EV11RepStage",
    "EV11ContextStage",
    "EV11DetailSeed",
    "EV11DetailExchange",
    "EV11DetailInject",
    "EV11NativeFusion",
    "SegmentCitrusEV11",
)


class EV11StablePool(nn.Module):
    """Channel-frequency selection adapted from LAST-ViT's author implementation.

    Author reference: ChengShiest/LAST-ViT, cls_pretrain/conf.py dense_vit.forward.
    Repository uses ORIGINAL features in the numerator, unlike paper Eq. 5's
    filtered numerator. Follow the released code, add denominator epsilon, and
    keep top-1 per channel. Channel frequencies are NOT image frequencies.
    CNN channel ordering has no natural frequency interpretation: this is an
    empirical transfer hypothesis, not a foreground or colour-invariance proof.
    Selection indices are nondifferentiable; gradients flow through gathered
    original features. No spatial tokens are removed from the dense path.
    """

    def __init__(self, channels, selective=True):
        super().__init__()
        self.selective = bool(selective)
        grid = torch.arange(-channels // 2 + 1, channels // 2 + 1).float()
        kernel = torch.exp(-0.5 * (grid / math.sqrt(channels)).square())
        self.register_buffer("kernel", kernel / kernel.max())

    def forward(self, x):
        if not self.selective:
            return x.mean((2, 3), keepdim=True)
        original = x.flatten(2).transpose(1, 2)  # B,N,C; FFT is along C, not N.
        with torch.no_grad():
            tokens = original.float()
            spectrum = torch.fft.fftshift(torch.fft.fft(tokens, dim=-1), dim=-1)
            filtered = torch.fft.ifft(torch.fft.ifftshift(spectrum * self.kernel.float(), dim=-1), dim=-1).real
            score = tokens / (filtered - tokens).abs().clamp_min(1e-6)
            indices = score.topk(1, dim=1, sorted=False).indices
        return original.gather(1, indices).mean(1).unsqueeze(-1).unsqueeze(-1)


class EV11ContextAttention(nn.Module):
    """Replace all-pairs attention with one pooled descriptor and dense alignment.

    Both GAP and selective arms have identical learned parameters. This dense
    residual bridge is OUR task adaptation, not an author LAST-ViT module. It
    preserves every position and starts at zero correction. No CLS token or
    additional image-level classification loss is claimed.
    """

    def __init__(self, channels, selective):
        super().__init__()
        self.pool = EV11StablePool(channels, selective)
        self.context_value = nn.Conv2d(channels, channels, 1, bias=False)
        self.context_gain = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        descriptor = self.pool(x)
        # Cosine is bounded, but does not certify that foreground beats leaves.
        alignment = F.cosine_similarity(x.float(), descriptor.float(), dim=1, eps=1e-6).unsqueeze(1)
        return self.context_gain.tanh() * alignment.sigmoid().to(x.dtype) * self.context_value(descriptor)


class EV11ContextStage(C2PSA):
    """C5 only: replace existing attention, retain CSP projections and pretrained FFN."""

    def __init__(self, c1, c2, n=1, selective=True, e=0.5):
        super().__init__(c1, c2, n, e)
        for block in self.m:
            block.attn = EV11ContextAttention(self.c, bool(selective))


class EV11RepMixer(nn.Module):
    """Reparameterizable depthwise spatial mixing, then a pointwise channel MLP."""

    def __init__(self, channels):
        super().__init__()
        self.token = RepConv(channels, channels, 3, g=channels, bn=True)
        self.expand = Conv(channels, 2 * channels, 1)
        self.contract = Conv(2 * channels, channels, 1, act=False)
        nn.init.zeros_(self.contract.bn.weight)

    def forward(self, x):
        spatial = self.token(x)
        return spatial + self.contract(self.expand(spatial))


class EV11RepStage(C3k2):
    """Preserve CSP projection transfer; replace C3/C4/C5 interior spatial mixers."""

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5):
        super().__init__(c1, c2, n, c3k, e)
        self.m = nn.ModuleList(EV11RepMixer(self.c) for _ in range(n))


class EV11DetailSeed(nn.Module):
    """Start a 16ch stride-4 branch from C2 and the four stride-2 stem phases."""

    def __init__(self, ch, channels=16):
        super().__init__()
        if len(ch) != 2:
            raise ValueError("DetailSeed expects [C2, stem]")
        self.shallow = Conv(ch[0], channels, 1)
        self.early = Conv(ch[1], 8, 1)
        self.phase = Conv(32, channels, 1)
        self.mix = Conv(2 * channels, channels, 1)

    def forward(self, x):
        shallow, stem = x
        return self.mix(torch.cat((self.shallow(shallow), self.phase(F.pixel_unshuffle(self.early(stem), 2))), 1))


class EV11DetailExchange(nn.Module):
    """Keep P2 throughout C3/C4 extraction; correct low band without overwriting edges.

    Reference-minus-measurement is borrowed as an error representation only.
    Two spatially varying routes distinguish semantic discrepancy and local
    residual; filters are fixed pooling/ordinary convolutions, not FreqFusion's
    learned reassembly/resampling. No absolute colour-invariance claim.
    """

    def __init__(self, ch):
        super().__init__()
        c, context = ch
        self.context = Conv(context, c, 1)
        self.local = Conv(c, c, 3, g=c)
        self.gate = nn.Conv2d(3 * c, 2, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.gain = nn.Parameter(torch.full((1, c, 1, 1), 0.1))

    def forward(self, x):
        detail, context = x
        reference = F.interpolate(self.context(context), detail.shape[-2:], mode="nearest")
        low = F.avg_pool2d(detail, 3, 1, 1, count_include_pad=False)
        residual = detail - low
        error = reference - low
        weights = self.gate(torch.cat((low, error, residual.abs()), 1)).softmax(1)
        update = weights[:, :1] * error + weights[:, 1:] * self.local(residual)
        return detail + self.gain.tanh() * update


class EV11DetailInject(nn.Module):
    """P2 phases re-enter native C4 BEFORE C5 extraction; projection is not lossless."""

    def __init__(self, ch):
        super().__init__()
        c4, c_detail = ch
        self.phase = Conv(16 * c_detail, c4, 1)
        self.gain = nn.Parameter(torch.zeros(1, c4, 1, 1))

    def forward(self, x):
        native, detail = x
        if detail.shape[-2:] != (4 * native.shape[-2], 4 * native.shape[-1]):
            raise ValueError("P2/C4 grids must have stride ratio four")
        return native + self.gain.tanh() * self.phase(F.pixel_unshuffle(detail, 4))


class EV11NativeFusion(nn.Module):
    """Spatially route native C4 or top-down N4; keep the P3-down stream separate.

    Output width and the following ordinary C3k2 spatial reconstruction remain
    unchanged. Semantic features are compressed before route prediction. The
    initial weights favour the inherited N4 path, instead of averaging blindly.
    """

    def __init__(self, ch, hidden=16):
        super().__init__()
        if len(ch) != 3 or ch[1] != ch[2]:
            raise ValueError("NativeFusion expects [P3down,N4,C4] with equal latter widths")
        self.fine = Conv(ch[0], hidden, 1)
        self.semantic = Conv(ch[1], hidden, 1)
        self.native = Conv(ch[2], hidden, 1)
        self.route = nn.Conv2d(3 * hidden, 2, 1)
        nn.init.zeros_(self.route.weight)
        with torch.no_grad():
            self.route.bias.copy_(torch.tensor([0.0, -4.0]))

    def forward(self, x):
        fine, semantic, native = x
        a, b = self.semantic(semantic), self.native(native)
        weights = self.route(torch.cat((self.fine(fine), a, b - a), 1)).softmax(1)
        return torch.cat((fine, weights[:, :1] * semantic + weights[:, 1:] * native), 1)


class SegmentCitrusEV11(SegmentCitrusEV10):
    """Keep the successful mask decoder and 8400 P3/P4/P5 candidates at 640.

    Layout [P3,P4,C5,C2,stem,C4,(persistent P2)]. Baselines preserve V10's
    state keys and exact objectives. Persistent arms replace the head-only V10
    transport, rather than accumulating it. Assignment/ring gains live in YAML
    so YOLO(YAML).train() constructs the correct criterion without monkeypatches.
    """

    def __init__(
        self,
        nc=80,
        nm=32,
        npr=256,
        detail_channels=16,
        persistent=False,
        legacy_transport=True,
        factorized_box=False,
        assignment_mix=0.0,
        ring_gain=0.0,
        tiny_gain=0.25,
        boundary_gain=0.5,
        neighbor_gain=0.25,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
        expected = 7 if persistent else 6
        if len(ch) != expected or (persistent and legacy_transport):
            raise ValueError("V11 expects six inputs or seven with persistent detail and no duplicate transport")
        super().__init__(
            nc,
            nm,
            npr,
            detail_channels,
            legacy_transport,
            False,
            tiny_gain,
            0.0,
            boundary_gain,
            neighbor_gain,
            reg_max,
            end2end,
            tuple(ch[:6]),
        )
        if not math.isfinite(assignment_mix) or not 0 <= assignment_mix <= 1:
            raise ValueError("assignment_mix must be finite in [0,1]")
        if not math.isfinite(ring_gain) or ring_gain < 0:
            raise ValueError("ring_gain must be finite and nonnegative")
        self.assignment_mix = float(assignment_mix)
        self.ring_gain = float(ring_gain)
        self.persistent = bool(persistent)
        self.factorized_box = bool(factorized_box)
        if self.factorized_box:
            # Factor the FIRST expensive box convolution into DW spatial + PW
            # channel mixing. Preserve the second ordinary spatial convolution,
            # DFL predictor, mask/cls towers and their pretrained keys. New
            # nested keys prevent wrong transfer from the old first 3x3 layer.
            for i, c in enumerate(ch[:3]):
                old = self.cv2[i]
                width = old[1].conv.out_channels
                self.cv2[i] = nn.Sequential(nn.Sequential(Conv(c, c, 3, g=c), Conv(c, width, 1)), old[1], old[2])
        if self.persistent:
            if ch[6] != detail_channels:
                raise ValueError("Persistent P2 width must match the mask-detail width")
            self.persistent_gain = nn.Parameter(torch.full((1, detail_channels, 1, 1), 0.1))

    def forward(self, x):
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        if self.transport:
            detail = self.detail_transport(detail, x[4], x[5])
        if self.persistent:
            detail = detail + self.persistent_gain.tanh() * (x[6] - detail)
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(detail, 2))
        proto = self.proto(features[0]) + self.detail_scale.tanh() * self.detail_to_proto(detail)
        proto = self.refine_fine_proto(proto, detail, x[4])
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
