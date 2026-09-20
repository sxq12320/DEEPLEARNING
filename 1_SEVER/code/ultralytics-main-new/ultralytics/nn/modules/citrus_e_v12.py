# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V12 task-routed recognition correction, keeping the validated geometry path.

Task decomposition is motivated by TOOD (ICCV21); protection of local detail
during semantic fusion by PIDNet (CVPR23) and FreqFusion (TPAMI24). This is an
independent adaptation, NOT any author's complete module or a PID controller.
There is no recurrent state, temporal integral or stability guarantee.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_e_v11 import SegmentCitrusEV11
from .conv import Conv

__all__ = ("EV12RecognitionCorrection", "SegmentCitrusEV12")


class EV12RecognitionCorrection(nn.Module):
    """A narrow reference/discrepancy route at P3/P4, not another dense P2 tower.

    Local and upper-level features are projected BEFORE upsampling. A learned
    spatial gate observes local contrast as well as semantic context; high-frequency
    energy is not assumed to mean fruit. The zero-start, bounded channel gain keeps
    the initial pretrained function intact. It bounds gain, NOT feature magnitude.
    The `difference=False` arm uses the same parameter count and sum evidence to
    test whether explicit discrepancy representation is actually beneficial.
    """

    def __init__(self, channels, context_channels, width=16, difference=True):
        super().__init__()
        self.difference = bool(difference)
        self.local = Conv(channels, width, 1)
        self.reference = Conv(context_channels, width, 1)
        self.gate = nn.Conv2d(3 * width, 1, 1)
        self.correct = nn.Sequential(Conv(2 * width, width, 1), Conv(width, width, 3, g=width))
        self.expand = nn.Conv2d(width, channels, 1, bias=False)
        self.gain = nn.Parameter(torch.zeros(1, channels, 1, 1))
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, local, context):
        measurement = self.local(local)
        reference = F.interpolate(self.reference(context), local.shape[-2:], mode="nearest")
        low = F.avg_pool2d(measurement, 5, 1, 2, count_include_pad=False)
        contrast = measurement - low
        evidence = reference - low if self.difference else reference + low
        confidence = self.gate(torch.cat((measurement, reference, contrast.abs()), 1)).sigmoid()
        correction = self.expand(confidence * self.correct(torch.cat((evidence, contrast), 1)))
        return local + 0.5 * self.gain.tanh() * correction


class SegmentCitrusEV12(SegmentCitrusEV11):
    """Recognition-specific semantic correction before classification towers.

    Layout/objectives are inherited from V11. `route=0` is an exact V11 replay;
    `route=1` corrects classification ONLY, keeping box/mask/quality features and
    prototype construction unchanged; `route=2` tests shared correction of all
    prediction towers (not the prototype decoder). All keep 8400 candidates@640.
    Correction is computed once per level and never changes anchor coordinates.
    Gradient coupling through the shared backbone/TAL remains; task separation is
    not a claim of complete gradient independence.
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
            reg_max,
            end2end,
            ch,
        )
        if route not in (0, 1, 2):
            raise ValueError("V12 route must be 0 (replay), 1 (classification), or 2 (shared towers)")
        self.recognition_route = int(route)
        if route:
            self.recognition = nn.ModuleList(
                EV12RecognitionCorrection(ch[i], ch[i + 1], 16, difference) for i in range(2)
            )

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None):
        if not self.recognition_route:
            return super().forward_head(x, box_head, cls_head, mask_head)
        # Use the ORIGINAL next-level reference for each correction, avoiding
        # an implicit recurrent/cascaded path and feature-cache side effects.
        recognized = [self.recognition[i](x[i], x[i + 1]) for i in range(2)] + [x[2]]
        geometry = recognized if self.recognition_route == 2 else x
        bs = x[0].shape[0]
        boxes = torch.cat([box_head[i](geometry[i]).view(bs, 4 * self.reg_max, -1) for i in range(self.nl)], -1)
        scores = torch.cat([cls_head[i](recognized[i]).view(bs, self.nc, -1) for i in range(self.nl)], -1)
        maps = [mask_head[i](geometry[i]) for i in range(self.nl)]
        coefficients = torch.cat([m.view(bs, self.nm, -1) for m in maps], -1)
        quality = [
            self.quality_predictor[i](torch.cat((geometry[i].detach(), m.detach()), 1)) for i, m in enumerate(maps)
        ]
        return dict(
            boxes=boxes,
            scores=scores,
            feats=x,
            mask_coefficient=coefficients,
            ev3_quality=torch.cat([q.view(bs, 1, -1) for q in quality], -1),
        )
