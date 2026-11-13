# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Two independent E12 hypotheses on the standard Segment head and loss."""
import torch.nn as nn

from .conv import Conv
from .head import Detect, Segment
from .citrus_e_v12 import EV12RecognitionCorrection


class SegmentCitrusE12Simple(Segment):
    """Optional P2 prototype residual and optional P3/P4 semantic correction.

    Detection strides and prototype resolution stay 8/16/32 and 4. Neither
    optional route adds a loss, quality score, assignment rule or postprocessor.
    Inputs: P3/P4/P5, then P2 only when detail=True. Width is explicitly 16.
    """

    def __init__(self, nc=80, nm=32, npr=256, detail=False, semantic=False,
                 reg_max=16, end2end=False, ch=()):
        if len(ch) != (4 if detail else 3):
            raise ValueError("Expected P3/P4/P5 and optional P2")
        if end2end:
            raise ValueError("This controlled YOLO11 study uses standard one-to-many segmentation")
        super().__init__(nc, nm, npr, reg_max, end2end, ch[:3])
        self.detail_enabled = bool(detail)
        self.semantic_enabled = bool(semantic)
        if detail:
            self.detail_residual = nn.Sequential(
                Conv(ch[3], 16, 1), Conv(16, 16, 3, g=16), nn.Conv2d(16, nm, 1)
            )
            nn.init.zeros_(self.detail_residual[-1].weight)
            nn.init.zeros_(self.detail_residual[-1].bias)
        if semantic:
            self.semantic = nn.ModuleList(EV12RecognitionCorrection(ch[i], ch[i+1], 16) for i in range(2))

    def forward_head(self, x, box_head, cls_head, mask_head):
        if self.semantic_enabled:
            x = [self.semantic[i](x[i], x[i+1]) for i in range(2)] + [x[2]]
        return super().forward_head(x, box_head, cls_head, mask_head)

    def forward(self, x):
        outputs = Detect.forward(self, list(x[:3]))
        preds = outputs[1] if isinstance(outputs, tuple) else outputs
        proto = self.proto(x[0])
        if self.detail_enabled:
            residual = self.detail_residual(x[3])
            if residual.shape != proto.shape:
                raise ValueError("P2 detail and prototype grids must match; no implicit resize")
            proto = proto + residual
        if isinstance(preds, dict):
            preds["proto"] = proto
        if self.training:
            return preds
        return (outputs, proto) if self.export else ((outputs[0], proto), preds)
