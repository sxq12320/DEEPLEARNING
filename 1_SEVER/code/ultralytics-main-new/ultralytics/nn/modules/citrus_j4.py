# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J4: amortize prediction parameters across scales, retain the tested detail decoder.

This is an independently written, untrained hypothesis, not a reproduction of
DEIMv2/EdgeCrafter. Projections and output layers are scale-specific; only the
middle spatial operator is shared. There is NO shared BatchNorm across scales,
custom CUDA, recurrent controller or hard grey/roundness gate.
"""

import torch.nn as nn

from .citrus_e_v12 import SegmentCitrusEV12
from .conv import Conv

__all__ = ("SegmentCitrusJ4",)


class SegmentCitrusJ4(SegmentCitrusEV12):
    """Retain P3/P4/P5 candidates and stride-2 prototypes; use narrow shared towers.

    Each task has its own shared middle convolution, not one shared task output.
    The per-level 1x1 Conv includes its own BN. The shared 3x3 is unnormalized,
    avoiding a mixture of stride-specific BN statistics. Final logits, priors
    and projections remain independent for each level.
    """

    def __init__(self, nc=1, nm=32, npr=256, detail_channels=16, persistent=True,
                 legacy_transport=False, factorized_box=False, assignment_mix=0.0,
                 ring_gain=0.0, tiny_gain=0.25, boundary_gain=0.5, neighbor_gain=0.25,
                 route=2, difference=True, tower_width=48, shared=True, quality=True,
                 reg_max=16, end2end=False, ch=()):
        if end2end or tower_width < 16 or tower_width % 8:
            raise ValueError("J4 requires one-to-many prediction and tower_width >=16 divisible by 8")
        super().__init__(nc, nm, npr, detail_channels, persistent, legacy_transport, factorized_box,
                         assignment_mix, ring_gain, tiny_gain, boundary_gain, neighbor_gain,
                         route, difference, reg_max, end2end, ch)
        self.shared_towers = bool(shared)
        self.tower_width = int(tower_width)
        self.quality_calibration = bool(quality)
        self.quality_gain = 1.0 if quality else 0.0
        if shared:
            def tower(output_channels):
                shared_weight = nn.Conv2d(tower_width, tower_width, 3, padding=1, bias=False).weight
                levels = nn.ModuleList()
                for c in ch[:3]:
                    # Distinct operator objects, tied Parameter: profilers count
                    # each call once rather than recursively recounting a reused
                    # module's cumulative counters at every parent path.
                    spatial = nn.Conv2d(tower_width, tower_width, 3, padding=1, bias=False)
                    spatial.weight = shared_weight
                    levels.append(nn.Sequential(Conv(c, tower_width, 1),
                                                nn.Sequential(spatial, nn.SiLU()),
                                                nn.Conv2d(tower_width, output_channels, 1)))
                return levels

            self.cv2 = tower(4 * reg_max)
            self.cv3 = tower(nc)
            self.cv4 = tower(nm)
