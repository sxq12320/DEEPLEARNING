# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Independent, normalized geometry supervision for SAGE-v4; legacy losses are unchanged."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .loss import v8SegmentationLoss


@torch.no_grad()
def structure_targets(masks, batch_idx, batch_size, size, overlap=True):
    """Return overlapping binary foreground/boundary/separator targets and a local active band.

    Fast overlap-mask path uses ID-map morphology: no N-instance expansion, CPU
    roundtrips, hull fitting, or pairwise instance loops. Separator is restricted
    to a boundary/gap; no deep-interior separator labels. Widths are defined on
    the stride-4 target grid. These are local proxies, NOT topological guarantees.
    """
    if overlap:
        ids = masks.reshape(batch_size, 1, *masks.shape[-2:]).float()
    else:
        # Optional non-overlap compatibility, not the fixed experimental protocol.
        ids = masks.new_zeros((batch_size, 1, *masks.shape[-2:]))
        for image_index in range(batch_size):
            instances = masks[batch_idx.flatten() == image_index].float()
            if len(instances):
                numbers = torch.arange(1, len(instances) + 1, device=masks.device)[:, None, None]
                ids[image_index, 0] = (instances * numbers).amax(0)
    if tuple(ids.shape[-2:]) != tuple(size):
        ids = F.interpolate(ids, size=size, mode="nearest")
    fruit = ids > 0
    maximum3 = F.max_pool2d(ids, 3, 1, 1)
    minimum3 = -F.max_pool2d(-F.pad(ids, (1, 1, 1, 1)), 3, 1, 0)
    inner_edge = fruit & ((maximum3 != ids) | (minimum3 != ids))
    boundary = F.max_pool2d(inner_edge.float(), 3, 1, 1) > 0
    maximum5 = F.max_pool2d(ids, 5, 1, 2)
    sentinel = ids.amax(dim=(2, 3), keepdim=True) + 1
    positive_ids = torch.where(fruit, ids, sentinel)
    minimum_positive5 = -F.max_pool2d(-positive_ids, 5, 1, 2)
    nearby_distinct = (maximum5 > 0) & (minimum_positive5 < maximum5)
    separator = nearby_distinct & (boundary | ~fruit)
    active = F.max_pool2d(fruit.float(), 7, 1, 3)
    # A background-only image supplies bounded negative supervision, not sum/zero-positive normalization.
    active = torch.where(active.amax(dim=(2, 3), keepdim=True) > 0, active, torch.ones_like(active))
    return torch.cat((fruit, boundary, separator), 1).float(), active


def normalized_structure_loss(logits, targets, active):
    """Balanced BCE and positive-present Dice, averaged per image and binary channel."""
    logits = logits.float()
    targets, active = targets.float(), active.float()
    positive, negative = targets * active, (1 - targets) * active
    dims = (2, 3)
    pcount, ncount = positive.sum(dims), negative.sum(dims)
    positive_bce = (F.softplus(-logits) * positive).sum(dims) / pcount.clamp_min(1)
    negative_bce = (F.softplus(logits) * negative).sum(dims) / ncount.clamp_min(1)
    present_p, present_n = (pcount > 0).float(), (ncount > 0).float()
    bce = (positive_bce + negative_bce) / (present_p + present_n).clamp_min(1)
    probability = logits.sigmoid() * active
    dice = 1 - (2 * (probability * targets).sum(dims) + 1) / (probability.sum(dims) + pcount + 1)
    return (bce + dice * present_p).mean()


class SAGEV4SegmentationLoss(v8SegmentationLoss):
    """Keep official losses untouched and log the new geometry term in sem_loss."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.structure_gain = float(model.model[-1].structure_gain)

    def loss(self, preds, batch):
        total, components = super().loss(preds, batch)
        logits = preds.get("sage_structure")
        if logits is not None and self.structure_gain > 0:
            batch_size = logits.shape[0]
            targets, active = structure_targets(
                batch["masks"].to(logits.device),
                batch["batch_idx"].to(logits.device),
                batch_size,
                logits.shape[-2:],
                self.overlap,
            )
            auxiliary = self.structure_gain * normalized_structure_loss(logits, targets, active)
            addition = torch.stack((auxiliary * 0, auxiliary * 0, auxiliary * 0, auxiliary * 0, auxiliary))
            total = total + addition * batch_size
            components = components + addition.detach()
        return total, components
