# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Reuse visible-instance loss and density supervision; no new GT or topology assumptions."""

import torch

from ultralytics.utils.citrus_j_loss import JSegmentationLoss
from ultralytics.utils.citrus_j2_loss import J2SegmentationLoss, density_focal_loss


class J2StructuralLoss(JSegmentationLoss):
    """Unchanged base YOLO losses; optional bounded tiny-Dice and density terms."""

    density_targets = J2SegmentationLoss.density_targets

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.density_gain = model.model[-1].density_gain

    def loss(self, preds, batch):
        total, components = super().loss(preds, batch)
        density = preds.get("j2_density")
        if density is None:
            return total, components
        imgsz = [float(v) * float(self.stride[0]) for v in preds["feats"][0].shape[2:]]
        target = self.density_targets(batch["masks"].to(density.device).float(),
                                      batch["batch_idx"].view(-1).to(density.device),
                                      density.shape[0], tuple(density.shape[-2:]), imgsz)
        term = self.density_gain * density_focal_loss(density, target)
        zero = term * 0.0
        addition = torch.stack((zero, zero, zero, zero, term)).to(total.dtype)
        return total + addition * density.shape[0], components + addition.detach()
