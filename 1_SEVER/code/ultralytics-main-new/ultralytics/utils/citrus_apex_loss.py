# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Loss criterion for SegmentCitrusBSD (`APEX` series).

Extends `v8SegmentationLoss` without altering TAL anchor assignment, box regression, or
classification objectives. Enables mild boundary supervision (`citrus_boundary_gain=0.25`)
for the multiplicative strip-boundary gate and detached Mask-IoU regression
(`citrus_quality_gain=0.25`) when `SegmentCitrusBSD.use_quality_calib` is enabled.
"""

from __future__ import annotations

from ultralytics.utils.loss import v8SegmentationLoss


class CitrusBSDLoss(v8SegmentationLoss):
    """Segmentation loss wrapper enabling BSD boundary and detached Mask-IoU calibration."""

    def __init__(self, model, tal_topk: int = 10, tal_topk2: int | None = None) -> None:
        super().__init__(model, tal_topk=tal_topk, tal_topk2=tal_topk2)
        head = model.model[-1]
        self.citrus_boundary_gain = max(self.citrus_boundary_gain, 0.25)
        if getattr(head, "use_quality_calib", False):
            self.citrus_quality_gain = max(self.citrus_quality_gain, 0.25)
