# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Explicit J2 decoder repairs; old structural/checkpoint graphs remain unchanged.

Learned sub-pixel phases follow ESPCN (Shi et al., CVPR2016). Repeated phase
initialization follows the ICNR idea (Aitken et al., arXiv:1707.02937).
This breaks a verified nearest-copy constraint, not a proof of higher AP.
"""

import torch
import torch.nn as nn

from .citrus_j2_structure import SegmentCitrusJ2Structural

__all__ = ("J2PhaseProto", "SegmentCitrusJ2Expressive")


class J2PhaseProto(nn.Module):
    """Replace nearest-upsample + 1x1 with independently learnable sub-pixel phases.

    At initialization the four phases reproduce the previous decoder exactly
    (floating-point roundoff aside). They are NOT tied during optimization.
    The final convolution has equal MACs and four times its previous parameters;
    the rest of the decoder is reused without changing its initial state.
    """

    def __init__(self, repeated_decoder):
        super().__init__()
        if len(repeated_decoder) != 4 or not isinstance(repeated_decoder[2], nn.Upsample):
            raise ValueError("Expected the original compact nearest x2 decoder")
        previous = repeated_decoder[3]
        if (previous.kernel_size != (1, 1) or previous.stride != (1, 1)
                or previous.padding != (0, 0) or previous.groups != 1
                or repeated_decoder[2].mode != "nearest" or repeated_decoder[2].scale_factor != 2
                or repeated_decoder[2].size is not None):
            raise ValueError("Phase conversion requires nearest x2 followed by a 1x1 convolution")
        self.project, self.spatial = repeated_decoder[0], repeated_decoder[1]
        # All random initial weights are immediately overwritten. Do not consume
        # extra CPU RNG draws in the otherwise identical seeded control graph.
        with torch.random.fork_rng(devices=[]):
            self.phase = nn.Conv2d(previous.in_channels, 4 * previous.out_channels, 1,
                                   bias=previous.bias is not None).to(previous.weight)
        with torch.no_grad():
            self.phase.weight.copy_(previous.weight.repeat_interleave(4, dim=0))
            if previous.bias is not None:
                self.phase.bias.copy_(previous.bias.repeat_interleave(4, dim=0))
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        return self.shuffle(self.phase(self.spatial(self.project(x))))


class SegmentCitrusJ2Expressive(SegmentCitrusJ2Structural):
    """Same head/loss/validation contract; only remove the compact proto copy constraint.

    proto_stride=4 retains P3 as its source for the clean decoder-only comparison.
    proto_stride=2 deliberately uses P2: source and output resolution change
    together, so it is not described as a pure stride ablation.
    """

    def __init__(self, nc=1, nm=32, npr=64, shared=False, tiny_gain=0.0, density=False,
                 proto_stride=2, hidden=32, compact_proto=True, reg_max=16, end2end=False, ch=()):
        if not compact_proto:
            raise ValueError("Expressive repair applies only to the compact prototype decoder")
        super().__init__(nc, nm, npr, shared, tiny_gain, density, proto_stride, hidden,
                         compact_proto, reg_max, end2end, ch)
        self.proto = J2PhaseProto(self.proto)
