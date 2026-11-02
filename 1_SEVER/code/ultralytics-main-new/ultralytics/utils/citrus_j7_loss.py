# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Optional J7 reuse of existing tiny/visible-boundary terms, without quality work.

The zero-gain arms use v8SegmentationLoss directly. This is an objective ablation,
not a new loss, pseudo-labeling scheme, or proof of colour invariance.
"""

from .citrus_e_v11_loss import EV11SegmentationLoss


class J7SegmentationLoss(EV11SegmentationLoss):
    """Keep inherited ordinary/Tiny-Dice/visible geometry, skip removed quality head."""

    def _quality_loss(self, preds, batch):
        return preds["proto"].sum() * 0
