# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V7 scale-order-safe assignment, preserving all inherited loss weights."""

from .citrus_e_v6_loss import EV6SegmentationLoss


class EV7SegmentationLoss(EV6SegmentationLoss):
    """Use spatial scale order only for TAL's small-GT candidate expansion.

    V7 appends P2 after P3/P4/P5 to preserve pretrained tower indices. TAL's
    expansion logic assumes stride[0:2] are the two finest levels; sorting its
    metadata fixes this without reordering predictions, anchors or target boxes.
    The detection loss's own stride tensor remains in original feature order.
    This is a compatibility correction, not a guaranteed tiny-object gain.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        ordered = sorted(self.assigner.stride)
        self.assigner.stride = ordered
        self.assigner.stride_val = ordered[1] if len(ordered) > 1 else ordered[0]
