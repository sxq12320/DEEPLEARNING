# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Align mask-quality targets with the mask loss supervision grid in V6 only."""

import torch.nn.functional as F

from .citrus_e_v5_loss import EV5SegmentationLoss


class EV6SegmentationLoss(EV5SegmentationLoss):
    """Resize detached logits, never downsample GT instance IDs for quality."""

    def _quality_loss(self, preds, batch):
        size = batch["masks"].shape[-2:]
        if preds["proto"].shape[-2:] != size:
            preds = dict(preds)
            preds["proto"] = F.interpolate(preds["proto"].detach(), size, mode="bilinear", align_corners=False)
        return super()._quality_loss(preds, batch)
