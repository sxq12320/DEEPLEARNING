# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J7R: real observation evidence enters one mature-structure instance network.

An untrained task hypothesis, not a new pixel-rearrangement/residual algorithm.
ESOD motivates separating observation cost from deep recognition; EV11/J7
supply the inherited instance/visible-mask path. Views are explicit, not cached.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_j7 import SegmentCitrusJ7
from .conv import Conv
from .head import Detect

__all__ = ("J7RPackedInput", "J7RHighEvidence", "J7RStemEvidence", "J7REvidenceRead", "SegmentCitrusJ7R")


class J7RPackedInput(nn.Module):
    """Split [global RGB, pixel-unshuffled raw high RGB], not modalities."""

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 15:
            raise ValueError(
                "J7R requires packed15=[global RGB, pixel_unshuffle(aligned high RGB,2)]. "
                "Use J7RTrainer/J7RValidator or explicitly pack both views; RGB-only fallback is forbidden."
            )
        if min(x.shape[-2:]) < 32 or any(size % 32 for size in x.shape[-2:]):
            raise ValueError("J7R canonical spatial dimensions must be positive multiples of 32")
        return [x[:, :3], x[:, 3:]]


class J7RHighEvidence(nn.Module):
    """Compress raw phases; projection is lossy, not guaranteed anti-aliasing.

    packed12@H,W comes from RGB@2H,2W. Outputs are width@H/2,H/4,H/8.
    Fixed phase-block support is known; effective sampling centres are not
    asserted identical to those of learned strided convolutions.
    """

    def __init__(self, c1=12, width=16):
        super().__init__()
        if c1 != 12 or not isinstance(width, int) or width < 8:
            raise ValueError("J7RHighEvidence expects packed12 and integer width >= 8")
        self.width = width
        self.fine = nn.Sequential(Conv(48, width, 1), Conv(width, width, 3, g=width))
        self.middle = nn.Sequential(Conv(4 * width, width, 1), Conv(width, width, 3, g=width))
        self.coarse = nn.Sequential(Conv(width, width, 3, 2, g=width), Conv(width, width, 1))

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 12 or any(size % 8 for size in x.shape[-2:]):
            raise ValueError("J7RHighEvidence requires BCHW packed12 with spatial multiples of 8")
        fine = self.fine(F.pixel_unshuffle(x, 2))
        middle = self.middle(F.pixel_unshuffle(fine, 2))
        return [fine, middle, self.coarse(middle)]


class J7RStemEvidence(nn.Module):
    """Read raw phases BEFORE native C2-C5 instead of compressing them to C3 first.

    The optional phase-centering is a removable hypothesis, not a fruit filter.
    For each RGB channel Q=I_4-11^T/4 removes a 2x2 block's constant component.
    Q is an orthogonal projection; the following 48->width projection is lossy.
    This standard signal decomposition is not claimed as a new transform.
    No dynamic sampling, ROI loop or CUDA extension is required.
    """

    def __init__(self, c1=12, width=16, phase_center=False):
        super().__init__()
        if c1 != 12 or isinstance(width, bool) or not isinstance(width, int) or width < 8:
            raise ValueError("J7RStemEvidence expects packed12 and integer width >= 8")
        if not isinstance(phase_center, bool):
            raise ValueError("phase_center must be a boolean")
        self.width, self.phase_center = width, phase_center
        self.encode = nn.Sequential(Conv(48, width, 1), Conv(width, width, 3, g=width))

    @staticmethod
    def center_phases(packed):
        """Zero-DC residual in each RGB 2x2 block, preserving phase order."""
        b, _, h, w = packed.shape
        phases = packed.reshape(b, 3, 4, h, w)
        return (phases - phases.mean(2, keepdim=True)).reshape(b, 12, h, w)

    def forward(self, x):
        if x.ndim != 4 or x.shape[1] != 12 or any(size % 2 for size in x.shape[-2:]):
            raise ValueError("J7RStemEvidence requires BCHW packed12 with even spatial dimensions")
        if self.phase_center:
            x = self.center_phases(x)
        return self.encode(F.pixel_unshuffle(x, 2))


class J7REvidenceRead(nn.Module):
    """Ordinary same-grid concat residual; no saliency/quality predictor.

    Only gain starts at zero, not the output projection: gain can learn on step
    one. The familiar formula is an interface control, not a novelty claim.
    """

    def __init__(self, ch, width=16):
        super().__init__()
        if len(ch) != 2 or not isinstance(width, int) or width < 8:
            raise ValueError("J7REvidenceRead expects [native,evidence] widths and integer width >= 8")
        native, evidence = map(int, ch)
        self.native = Conv(native, width, 1)
        self.evidence = Conv(evidence, width, 1)
        self.mix = nn.Sequential(Conv(2 * width, width, 1), Conv(width, width, 3, g=width))
        self.expand = nn.Conv2d(width, native, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(1, native, 1, 1))

    def forward(self, x):
        native, evidence = x
        if native.shape[0] != evidence.shape[0] or native.shape[-2:] != evidence.shape[-2:]:
            raise ValueError("J7R evidence must already be aligned to the native feature grid")
        update = self.expand(self.mix(torch.cat((self.native(native), self.evidence(evidence)), 1)))
        return native + self.gain.tanh() * update


class SegmentCitrusJ7R(SegmentCitrusJ7):
    """Same candidates; optional high input affects ONLY final Proto at inference.

    Joint training still couples gradients through the mask loss. Early-C3 or
    old-EV11 inputs are explicit YAML edges, never hidden inside this head.
    """

    def __init__(self, nc=1, nm=32, npr=256, detail_channels=16, persistent=True,
                 mask_evidence=False, reg_max=16, end2end=False, ch=()):
        if len(ch) != (8 if mask_evidence else 7):
            raise ValueError("J7R head needs seven inherited inputs plus final-Proto evidence if enabled")
        super().__init__(nc=nc, nm=nm, npr=npr, detail_channels=detail_channels,
                         persistent=persistent, shared=False, fine_mask=True,
                         reg_max=reg_max, end2end=end2end, ch=ch[:7])
        self.mask_evidence = bool(mask_evidence)
        if self.mask_evidence:
            self.high_mask = J7REvidenceRead([nm, ch[7]], detail_channels)

    def forward(self, x):
        if self.export:
            raise ValueError("J7R export is not verified; use native packed-tensor inference")
        features = list(x[:3])
        detail = self.refiner(x[3], features[0])
        if self.transport:
            detail = self.detail_transport(detail, x[4], x[5])
        if self.persistent:
            detail = detail + self.persistent_gain.tanh() * (x[6] - detail)
        features[0] = features[0] + self.relay_scale.tanh() * self.detail_relay(F.pixel_unshuffle(detail, 2))
        proto = self.proto(features[0]) + self.detail_scale.tanh() * self.detail_to_proto(detail)
        proto = self.refine_fine_proto(proto, detail, x[4])
        if self.mask_evidence:
            proto = self.high_mask([proto, x[7]])
        outputs = Detect.forward(self, features)
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        preds["proto"] = proto
        return preds if self.training else ((outputs[0], proto), preds)
