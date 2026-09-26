# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""I V4: supervised scene layers replace the serial PAN neck.

Inspired by region aggregation (SparseInst), search/identification (SINet), and
detail/context separation (PIDNet), not a reproduction of their modules. The
two scene regions are NOT instance slots. Instances still use separate YOLO
coefficients and TAL. No hard foreground rejection, amodal filling or new labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .citrus_i_v2 import SegmentCitrusIV2
from .conv import Conv

__all__ = ("SegmentCitrusIV4",)


class LayeredSceneNeck(nn.Module):
    """Parallel multi-scale projection, soft scene partition, bounded feedback.

    Mode 0: parallel context only; 1: auxiliary discovery only; 2: region
    feedback; 3: boundary-protected region feedback. A second shared state step
    is optional, not a second backbone pass. The raw local feature always survives.
    """

    def __init__(self, channels, outputs, detail_channels, width=16, mode=3, steps=1):
        super().__init__()
        if mode not in (0, 1, 2, 3) or steps not in (1, 2):
            raise ValueError("I V4 mode must be 0..3 and steps 1 or 2")
        self.mode, self.steps = mode, steps
        self.local = nn.ModuleList(Conv(c, out, 1) for c, out in zip(channels, outputs))
        self.context = nn.ModuleList(Conv(c, width, 1) for c in channels)
        self.detail = Conv(detail_channels, width, 1)
        self.mix = nn.Sequential(Conv(width * 4, width, 1), Conv(width, width, 3, g=width))
        self.broadcast = nn.ModuleList(Conv(width, out, 1, act=False) for out in outputs)
        if mode:
            self.readout = nn.Conv2d(width, 2, 1)  # visible fruit occupancy; instance transition band
        if mode >= 2:
            self.correct = nn.Sequential(Conv(width * 2, width, 1), Conv(width, width, 3, g=width))
            self.gain = nn.Parameter(torch.zeros(1, width, 1, 1))

    @staticmethod
    def region_difference(state, probability):
        """Two normalized soft region summaries; accumulate in FP32 under AMP."""
        values, p = state.float().flatten(2), probability.float().flatten(2)
        regions = torch.cat((p, 1 - p), 1)
        weights = regions / regions.sum(-1, keepdim=True).clamp_min(1e-6)
        tokens = torch.bmm(weights, values.transpose(1, 2))
        difference = tokens[:, 0] - tokens[:, 1]
        return difference[:, :, None, None].to(state.dtype).expand_as(state)

    def forward(self, features, detail):
        size = detail.shape[-2:]
        context = [F.interpolate(project(x), size, mode="nearest") for project, x in zip(self.context, features)]
        state = self.mix(torch.cat([self.detail(detail), *context], 1))
        logits_history = []
        if self.mode:
            for _ in range(self.steps):
                logits = self.readout(state)
                logits_history.append(logits)
                if self.mode >= 2:
                    p = logits[:, :1].sigmoid()
                    contrast = self.region_difference(state, p)
                    protection = 1 - logits[:, 1:2].sigmoid() if self.mode == 3 else 1.0
                    delta = self.correct(torch.cat((state, contrast), 1))
                    state = state + 0.5 * self.gain.tanh() * protection * delta
        outputs = []
        for x, local, broadcast in zip(features, self.local, self.broadcast):
            # Downsample the narrow state BEFORE projecting to wide detection channels.
            summary = F.adaptive_avg_pool2d(state, x.shape[-2:])
            outputs.append(local(x) + broadcast(summary))
        return outputs, logits_history


class SegmentCitrusIV4(SegmentCitrusIV2):
    """Raw C3/C4/C5 + C2/stem/C4/detail -> layered neck -> instance head.

    YAML args: nc, nm, npr, width, mode, steps, auxiliary_gain. The established
    mask decoder/quality/tiny losses are held fixed. Serial PAN is absent from
    these YAMLs; mode 0 isolates this structural change from layer supervision.
    """

    def __init__(self, nc=80, nm=32, npr=256, width=16, mode=3, steps=1,
                 auxiliary_gain=0.2, reg_max=16, end2end=False, ch=()):
        if len(ch) != 7 or nc != 1:
            raise ValueError("I V4 is single-class citrus and requires seven feature inputs")
        outputs = (max(16, ch[0] // 2), ch[1], ch[2])
        super().__init__(nc, nm, npr, ch[-1], True, False, False, 0.0, 0.0,
                         0.25, 0.5, 0.25, 2, True, False, reg_max, end2end,
                         (*outputs, *ch[3:]))
        self.scene_neck = LayeredSceneNeck(ch[:3], outputs, ch[-1], width, mode, steps)
        self.layer_aux_gain = float(auxiliary_gain)

    def forward(self, x):
        features, layers = self.scene_neck(x[:3], x[-1])
        result = super().forward([*features, *x[3:]])
        predictions = result if self.training else (result[1] if not self.export else None)
        if isinstance(predictions, dict):
            predictions["iv4_scene_layers"] = layers
        return result
