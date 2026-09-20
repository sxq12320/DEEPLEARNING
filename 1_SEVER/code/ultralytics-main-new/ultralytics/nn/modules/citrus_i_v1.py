# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""I V1: synchronized dual prototypes inside the mask decoder.

V12 evidence: classification-side semantic correction reduced background errors
but did not raise tiny-fruit recall; the shared-route arm scored best on the fine
budget. The single prototype path still mixes "what it is" with "where it is".
This series tests a second, semantic prototype built from P4 context, then a
per-position gate that arbitrates between it and the validated detail prototype.

Context choices are independent task adaptations of SegNeXt's decomposed
multi-scale convolutional attention (NeurIPS22) and PKINet's context-anchor
attention (CVPR24). They are NOT the authors' complete blocks, no CAA pretrained
weights are reused, and no colour-invariance or leaf-discrimination proof is
implied. Every added term is bounded and starts near zero so the parent's decode
function is preserved at initialization. This is not a PID controller, not a
recurrent state, and carries no stability guarantee; accuracy and latency must
be measured, matching the reviewer-frozen protocol in docs/I_V1_REVIEW_20260920.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v12 import SegmentCitrusEV12
from .conv import Conv
from .head import Detect

__all__ = ("IV1StripContext", "IV1AnchorContext", "IV1SemanticProto", "IV1ProtoSync", "SegmentCitrusIV1")


class IV1StripContext(nn.Module):
    """Multi-scale strip depthwise context, adapted from SegNeXt's MSCA idea.

    One 5x5 local term plus two 1xK/Kx1 decomposed larger kernels are summed and
    used as a convolutional attention factor on the reduced feature. Kernels are
    kept at 7 and 11 (not the author's full set) because P4 is only 40x40 at 640
    and the widest strip adds little at this resolution. Attention multiplies
    the feature; it does not remove positions or claim foreground separation.
    """

    def __init__(self, channels, width=64):
        super().__init__()
        self.reduce = Conv(channels, width, 1)
        self.local = nn.Conv2d(width, width, 5, padding=2, groups=width, bias=False)
        self.h7 = nn.Conv2d(width, width, (1, 7), padding=(0, 3), groups=width, bias=False)
        self.v7 = nn.Conv2d(width, width, (7, 1), padding=(3, 0), groups=width, bias=False)
        self.h11 = nn.Conv2d(width, width, (1, 11), padding=(0, 5), groups=width, bias=False)
        self.v11 = nn.Conv2d(width, width, (11, 1), padding=(5, 0), groups=width, bias=False)
        self.attn = nn.Conv2d(width, width, 1)

    def forward(self, x):
        x = self.reduce(x)
        context = self.local(x) + self.h7(self.v7(x)) + self.h11(self.v11(x))
        return x * self.attn(context)


class IV1AnchorContext(nn.Module):
    """Context-anchor strip attention, adapted from PKINet's CAA idea.

    Average pooling smooths local noise before the horizontal/vertical strip
    factors; a sigmoid factor then scales the reduced feature. Only the strip
    decomposition is borrowed: no PKI inception branch, no author weights.
    """

    def __init__(self, channels, width=64):
        super().__init__()
        self.reduce = Conv(channels, width, 1)
        self.pool = nn.AvgPool2d(7, 1, 3)
        self.pre = Conv(width, width, 1)
        self.h = nn.Conv2d(width, width, (1, 11), padding=(0, 5), groups=width, bias=False)
        self.v = nn.Conv2d(width, width, (11, 1), padding=(5, 0), groups=width, bias=False)
        self.post = Conv(width, width, 1)

    def forward(self, x):
        x = self.reduce(x)
        attn = self.post(self.v(self.h(self.pre(self.pool(x))))).sigmoid()
        return x * attn


class IV1SemanticProto(nn.Module):
    """A second mask-prototype basis fed by P4 context instead of P2 detail.

    The semantic basis answers "where does fruit evidence persist at context
    scale"; it never replaces the detail basis inside this module. `context=0`
    keeps a parameter-matched plain projection so ablations can separate the
    context block from the mere existence of a second prototype.
    """

    def __init__(self, c_p4, nm=32, width=64, context=1):
        super().__init__()
        if context == 1:
            self.context = IV1StripContext(c_p4, width)
        elif context == 2:
            self.context = IV1AnchorContext(c_p4, width)
        elif context == 0:
            self.context = Conv(c_p4, width, 1)
        else:
            raise ValueError("IV1SemanticProto context must be 0 (plain), 1 (strip) or 2 (anchor)")
        self.spatial = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False),
            Conv(width, width, 1),
        )
        self.basis = nn.Conv2d(width, nm, 1)

    def forward(self, p4, size):
        # All convolutions stay at P4 resolution; only the nm basis channels are
        # upsampled, which keeps the second prototype nearly free at stride-2 protos.
        semantic = self.basis(self.spatial(self.context(p4)))
        return F.interpolate(semantic, size, mode="bilinear", align_corners=False)


class IV1ProtoSync(nn.Module):
    """Per-position gate field arbitrating between detail and semantic protos.

    The gate observes compressed views of both prototypes and the live detail
    stream, including their pointwise discrepancy. A spatial softmax was
    considered and rejected: positions should not compete with each other, so a
    sigmoid field per channel is returned. The scalar mixture gain lives in the
    head, keeping this module a pure direction field.
    """

    def __init__(self, nm, detail_channels, width=8):
        super().__init__()
        self.detail = Conv(detail_channels, width, 1)
        self.left = Conv(nm, width, 1)
        self.right = Conv(nm, width, 1)
        self.gate = nn.Conv2d(3 * width + 1, nm, 1)

    def forward(self, proto_d, proto_s, detail):
        # Arbitration is a low-frequency signal: pool both prototypes to the
        # detail grid, compute the gate there, then upsample the nm gate field.
        target = proto_d.shape[-2:]
        size = detail.shape[-2:]
        proto_d = F.adaptive_avg_pool2d(proto_d, size)
        proto_s = F.adaptive_avg_pool2d(proto_s, size)
        evidence = torch.cat(
            (
                self.detail(detail),
                self.left(proto_d),
                self.right(proto_s),
                (proto_s - proto_d).abs().mean(1, keepdim=True),
            ),
            1,
        )
        gate = self.gate(evidence).sigmoid()
        return F.interpolate(gate, target, mode="nearest") if gate.shape[-2:] != target else gate


class SegmentCitrusIV1(SegmentCitrusEV12):
    """Dual synchronized prototypes on top of the validated V12 recognition route.

    `dual` selects the prototype recipe: 0 replays the parent exactly (anchors);
    1 mixes the semantic basis with one learned scalar; 2 adds the per-position
    arbitration gate; 3 substitutes the detail prototype entirely. `feedback`
    adds a bounded depthwise term on the prototype discrepancy. `lead` sets the
    initial scalar mixture so direction-biased arms start away from the parent
    deliberately, which the protocol counts as a hypothesis, not free accuracy.
    Detection towers, anchors, the persistent detail path and every loss gain
    are inherited unchanged from V12/V11/V10.
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
        route=1,
        difference=True,
        dual=2,
        context=1,
        feedback=0,
        lead=0.0,
        reg_max=16,
        end2end=False,
        ch=(),
    ):
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
            ch,
        )
        if dual not in (0, 1, 2, 3):
            raise ValueError("IV1 dual must be 0 (replay), 1 (scalar), 2 (gated) or 3 (semantic only)")
        if not math.isfinite(lead):
            raise ValueError("IV1 lead must be finite")
        self.dual = int(dual)
        self.has_feedback = bool(feedback)
        if self.dual:
            self.semantic_proto = IV1SemanticProto(ch[1], nm, 64, context)
            self.mix = nn.Parameter(torch.tensor(float(lead)))
            if self.dual == 2:
                self.sync = IV1ProtoSync(nm, detail_channels)
            if self.has_feedback:
                self.feedback = nn.Conv2d(nm, nm, 3, padding=1, groups=nm, bias=False)
                self.feedback_gain = nn.Parameter(torch.zeros(1, nm, 1, 1))

    def synchronize(self, proto, p4, detail):
        """Combine the detail prototype with the semantic basis at P2 resolution."""
        semantic = self.semantic_proto(p4, proto.shape[-2:])
        if self.dual == 3:
            combined = semantic + self.mix.tanh() * (proto - semantic)
        elif self.dual == 2:
            combined = proto + self.mix.tanh() * self.sync(proto, semantic, detail) * (semantic - proto)
        else:
            combined = proto + self.mix.tanh() * (semantic - proto)
        if self.has_feedback:
            combined = combined + self.feedback_gain.tanh() * self.feedback(proto - semantic)
        return combined

    def forward(self, x):
        """V11 forward with a single added synchronization step at the end."""
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        if self.transport:
            detail = self.detail_transport(detail, x[4], x[5])
        if self.persistent:
            detail = detail + self.persistent_gain.tanh() * (x[6] - detail)
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(detail, 2))
        proto = self.proto(features[0]) + self.detail_scale.tanh() * self.detail_to_proto(detail)
        proto = self.refine_fine_proto(proto, detail, x[4])
        if self.dual:
            proto = self.synchronize(proto, x[1], detail)
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        if isinstance(preds, dict):
            preds["proto"] = proto
            if self.training:
                return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
