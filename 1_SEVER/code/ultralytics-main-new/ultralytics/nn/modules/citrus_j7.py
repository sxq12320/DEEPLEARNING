# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J7: factor deep computation, share context once, retain native instance evidence.

HIRI-ViT motivates low-grid channel mixing; multiscale fusion and the inherited
V12/J4 detail/phase decoder are acknowledged prior mechanisms. These are
untrained task hypotheses, not author reproductions or proven AP improvements.
There is no shallow discovery head, extra candidate source, or quality reweight.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_j4 import SegmentCitrusJ4
from .conv import Conv
from .head import Detect

__all__ = ("J7DeepStage", "J7ScaleHub", "SegmentCitrusJ7")


class _J7DeepBlock(nn.Module):
    """Keep spatial evidence on its native grid; mix channels on half the grid."""

    def __init__(self, channels, expansion=2.0):
        super().__init__()
        hidden = max(channels, int(round(channels * expansion)))
        self.spatial = Conv(channels, channels, 5, g=channels)
        self.channel = nn.Sequential(Conv(channels, hidden, 1), Conv(hidden, channels, 1, act=False))

    def forward(self, x):
        # The low route is not asserted to reconstruct discarded high frequencies.
        low = self.channel(F.avg_pool2d(x, 2, 2, ceil_mode=True))
        return x + self.spatial(x) + F.interpolate(low, size=x.shape[-2:], mode="nearest")


class J7DeepStage(nn.Module):
    """Optional C4/C5-only stage; parse_model supplies channels and internal repeats."""

    def __init__(self, c1, c2, n=1, expansion=2.0):
        super().__init__()
        if n < 1 or not 1 <= expansion <= 4:
            raise ValueError("J7DeepStage requires positive repeats and expansion in [1, 4]")
        self.project = Conv(c1, c2, 1) if c1 != c2 else nn.Identity()
        self.blocks = nn.Sequential(*(_J7DeepBlock(c2, expansion) for _ in range(n)))

    def forward(self, x):
        return self.blocks(self.project(x))


class J7ScaleHub(nn.Module):
    """One C4-grid context cache, native-scale readers, and a short N3 -> N4 return.

    Inputs are [C3, C4, C5]. All source projections occur before resizing.
    The N3 spatial return is semantically enriched, not raw C3 used as a claim
    of equivalence to PAN. The complete neck replacement needs its own ablation.
    Outputs are a three-element list for explicit YAML Index nodes.
    """

    def __init__(self, ch, output_channels, hub_channels=24, spatial_return=True):
        super().__init__()
        if len(ch) != 3 or len(output_channels) != 3 or hub_channels < 8:
            raise ValueError("J7ScaleHub requires three input/output widths and hub width >= 8")
        if any(int(c) != c or c < 8 for c in (*ch, *output_channels, hub_channels)):
            raise ValueError("J7ScaleHub widths must be positive integer channel counts")
        p3, p4, p5 = map(int, output_channels)
        h = int(hub_channels)
        self.spatial_return = bool(spatial_return)
        self.projections = nn.ModuleList(Conv(c, h, 1) for c in ch)
        self.context = nn.Sequential(Conv(3 * h, h, 1), Conv(h, h, 3, g=h))

        def reader(cin, cout):
            return nn.Sequential(Conv(cin, cout, 1), Conv(cout, cout, 3, g=cout), Conv(cout, cout, 1))

        self.read3 = reader(ch[0] + h, p3)
        self.read4 = reader(ch[1] + h * (2 if self.spatial_return else 1), p4)
        self.read5 = reader(ch[2] + h, p5)
        if self.spatial_return:
            self.return3 = nn.Sequential(Conv(p3, p3, 3, 2, g=p3), Conv(p3, h, 1))

    def forward(self, x):
        if len(x) != 3:
            raise ValueError("J7ScaleHub expects C3/C4/C5 tensors")
        c3, c4, c5 = x
        if not (c3.shape[-2] >= c4.shape[-2] >= c5.shape[-2] and
                c3.shape[-1] >= c4.shape[-1] >= c5.shape[-1]):
            raise ValueError("J7ScaleHub inputs must be ordered fine to coarse")
        a, b, c = [project(value) for project, value in zip(self.projections, x)]
        hub = self.context(torch.cat((F.adaptive_avg_pool2d(a, b.shape[-2:]), b,
                                      F.interpolate(c, size=b.shape[-2:], mode="nearest")), 1))
        n3 = self.read3(torch.cat((c3, F.interpolate(hub, size=c3.shape[-2:], mode="nearest")), 1))
        p4_inputs = [c4, hub]
        if self.spatial_return:
            returned = self.return3(n3)
            if returned.shape[-2:] != c4.shape[-2:]:
                raise ValueError("J7 native N3/N4 grids must have stride ratio two")
            p4_inputs.append(returned)
        n4 = self.read4(torch.cat(p4_inputs, 1))
        n5 = self.read5(torch.cat((c5, F.adaptive_avg_pool2d(hub, c5.shape[-2:])), 1))
        return [n3, n4, n5]


class SegmentCitrusJ7(SegmentCitrusJ4):
    """Same-origin three-scale candidates with inherited wide/fine mask evidence.

    Unlike setting an old quality gain to zero, this head removes its predictor
    and never evaluates its loss. Route=0 explicitly excludes EV12's additional
    per-scale recognition correction. Shared/unshared spatial towers and fine
    masks remain independent YAML factors, not claimed innovations.
    """

    def __init__(self, nc=1, nm=32, npr=256, detail_channels=16, persistent=True,
                 shared=False, tower_width=48, fine_mask=True, tiny_gain=0.0,
                 boundary_gain=0.0, neighbor_gain=0.0, reg_max=16, end2end=False, ch=()):
        super().__init__(nc=nc, nm=nm, npr=npr, detail_channels=detail_channels,
                         persistent=persistent, legacy_transport=False, factorized_box=False,
                         assignment_mix=0.0, ring_gain=0.0, tiny_gain=tiny_gain,
                         boundary_gain=boundary_gain, neighbor_gain=neighbor_gain,
                         route=0, difference=True, tower_width=tower_width, shared=shared,
                         quality=False, reg_max=reg_max, end2end=end2end, ch=ch)
        del self.quality_predictor
        self.quality_calibration = False
        self.quality_gain = 0.0
        self.fine_mask = bool(fine_mask)
        self.proto_stride = 2 if self.fine_mask else 4
        if not self.fine_mask:
            del self.fine_detail, self.fine_to_proto, self.fine_scale

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None):
        output = Detect.forward_head(self, x, box_head, cls_head)
        batch = x[0].shape[0]
        output["mask_coefficient"] = torch.cat(
            [layer(value).reshape(batch, self.nm, -1) for layer, value in zip(mask_head, x)], 2)
        return output

    def refine_fine_proto(self, proto, mask_detail, stem):
        if not self.fine_mask:
            return proto
        return super().refine_fine_proto(proto, mask_detail, stem)
