# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V3: selective deep compression, native-scale bypass and mask-IoU ranking.

Task adaptations of FasterNet PConv, EfficientDet normalized fusion and Mask
Scoring R-CNN. Not complete reproductions of those networks. Shallow stages,
SPPF/C2PSA, spatial neck refiners and standard prediction towers are retained.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import C3k2
from .citrus_far import FasterBlock
from .citrus_sage_v5 import SegmentCitrusSAGEV5
from .conv import Conv
from .head import Detect

__all__ = ("EV3DeepStage", "EV3DetailDown", "EV3NativeFusion", "SegmentCitrusEV3Quality")


class EV3DeepStage(C3k2):
    """Keep CSP projection keys; replace ONLY deep inner blocks with PConv mixers."""

    def __init__(self, c1, c2, n=1, c3k=True, e=0.5):
        super().__init__(c1, c2, n, c3k, e)
        self.m = nn.ModuleList(FasterBlock(self.c, self.c, e=2.0) for _ in range(n))


class EV3DetailDown(nn.Module):
    """Learned 2x2 phase aggregation; reindexing is lossless, projection is NOT."""

    def __init__(self, c1, c2):
        super().__init__()
        self.project = Conv(c1 * 4, c2, 1)

    def forward(self, x):
        return self.project(F.pixel_unshuffle(x, 2))


class EV3NativeFusion(nn.Module):
    """C4-native bypass into P4 reconstruction, without collapsing P3 into a hub.

    The first two inputs are the historical P3-down and top-down P4. The third
    is the native C4 feature previously absent from this fusion. Start close to
    the pretrained path; spatial C3k2 refinement follows in YAML unchanged.
    """

    def __init__(self):
        super().__init__()
        self.route_logits = nn.Parameter(torch.tensor([0.0, -4.0]))

    def forward(self, x):
        fine, semantic, native = x
        weights = self.route_logits.softmax(0)
        return torch.cat((fine, weights[0] * semantic + weights[1] * native), 1)


class SegmentCitrusEV3Quality(SegmentCitrusSAGEV5):
    """Retain pretrained towers; train a detached, cheap dense mask-quality observer.

    Only ranking is calibrated; no new anchors, masks, recurrent controller or
    target access during inference. Features/coefficient inputs are detached so
    the auxiliary ranking objective cannot silently reshape the segmentation
    representation. This ablation is not a claim of improved colour recognition.
    """

    def __init__(self, nc=80, nm=32, npr=256, detail_channels=16, relay=True,
                 late_proto=False, boundary_gain=0.0, neighbor_gain=0.0,
                 quality_gain=1.0, reg_max=16, end2end=False, ch=()):
        if end2end:
            raise ValueError("E V3 quality calibration uses the YOLO11 one-to-many segmentation protocol")
        super().__init__(nc, nm, npr, detail_channels, relay, late_proto,
                         boundary_gain, neighbor_gain, reg_max, end2end, ch)
        self.quality_gain = float(quality_gain)
        self.quality_calibration = True
        self.quality_predictor = nn.ModuleList(
            nn.Sequential(Conv(c + nm, 16, 1), nn.Conv2d(16, 1, 1)) for c in ch[:3]
        )
        for predictor in self.quality_predictor:
            nn.init.zeros_(predictor[-1].weight)
            nn.init.constant_(predictor[-1].bias, 1.38629436)

    def forward_head(self, x, box_head=None, cls_head=None, mask_head=None):
        preds = Detect.forward_head(self, x, box_head, cls_head)
        maps = [mask_head[i](x[i]) for i in range(self.nl)]
        batch_size = x[0].shape[0]
        preds["mask_coefficient"] = torch.cat([m.view(batch_size, self.nm, -1) for m in maps], 2)
        quality = [self.quality_predictor[i](torch.cat((x[i].detach(), m.detach()), 1))
                   for i, m in enumerate(maps)]
        preds["ev3_quality"] = torch.cat([q.view(batch_size, 1, -1) for q in quality], 2)
        return preds

    def _inference(self, preds):
        boxes = self._get_decode_boxes(preds)
        scores = preds["scores"].sigmoid()
        if self.quality_calibration:
            # Optional same-checkpoint diagnostic; historical checkpoints default
            # to their exact original multiplication (floor=0).
            floor = getattr(self, "quality_floor", 0.0)
            scores = scores * (floor + (1.0 - floor) * preds["ev3_quality"].sigmoid())
        return torch.cat((boxes, scores, preds["mask_coefficient"]), 1)
