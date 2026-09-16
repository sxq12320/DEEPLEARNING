# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V10 adds training-only, instance-balanced visible-foreground supervision.

Actual instance masks, box regression, TAL and PR metrics stay unchanged.
This is an independent adaptation of auxiliary dense supervision, not a new
definition of GT or a claim that a semantic auxiliary separates instances.
"""

import torch
import torch.nn.functional as F

from .citrus_e_v9_loss import EV9SegmentationLoss


def visible_foreground_loss(logits, masks, batch_idx, overlap):
    """Area-balanced positives and balanced background at a narrow P2 branch.

    Max-pool reduction retains occupied cells for auxiliary supervision only.
    Each surviving GT contributes equal total positive weight, irrespective
    of its visible area or how many anchors TAL assigns. Empty images train
    background; empty/collapsed GT does not create a fabricated positive.
    """
    size = logits.shape[-2:]
    terms = []
    for i in range(len(logits)):
        if overlap:
            labels = F.adaptive_max_pool2d(masks[i : i + 1, None].float(), size)[0, 0].long()
            counts = torch.bincount(labels.flatten()).float()
            weights = counts.clamp_min(1).reciprocal()[labels] * (labels > 0)
            target = labels > 0
        else:
            selected = masks[batch_idx.flatten() == i].float()
            if len(selected):
                selected = F.adaptive_max_pool2d(selected[:, None], size)[:, 0] > 0
                area = selected.sum((1, 2)).clamp_min(1)
                weights = (selected.float() / area[:, None, None]).sum(0)
                target = selected.any(0)
            else:
                weights = logits.new_zeros(size)
                target = torch.zeros(size, dtype=torch.bool, device=logits.device)
        per_pixel = F.binary_cross_entropy_with_logits(logits[i, 0].float(), target.float(), reduction="none")
        positive = (per_pixel * weights).sum() / weights.sum().clamp_min(1)
        negative = (per_pixel * (~target)).sum() / (~target).sum().clamp_min(1)
        # Background-only images must still have a nonzero gradient.
        present = target.any().to(per_pixel.dtype)
        terms.append((positive + negative) / (1 + present))
    return torch.stack(terms).mean()


class EV10SegmentationLoss(EV9SegmentationLoss):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.visibility_gain = model.model[-1].visibility_gain
        self.last_visibility = None

    def loss(self, preds, batch):
        total, components = super().loss(preds, batch)
        self.last_visibility = None
        # Validation computes a loss from eval-mode predictions too. The
        # auxiliary head is deliberately absent there and in deployment.
        if self.visibility_gain and "ev10_visibility" in preds:
            logits = preds["ev10_visibility"]
            value = visible_foreground_loss(
                logits, batch["masks"].to(logits.device), batch["batch_idx"].to(logits.device), self.overlap
            )
            self.last_visibility = value.detach()
            value = self.visibility_gain * value
            extra = torch.stack((value * 0, value * 0, value * 0, value * 0, value))
            total = total + extra * len(logits)
            components = components + extra.detach()
        return total, components
