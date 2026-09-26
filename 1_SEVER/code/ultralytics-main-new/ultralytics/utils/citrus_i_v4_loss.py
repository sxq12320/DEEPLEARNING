# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Dense layer supervision independent of whether TAL found positive candidates."""

import torch
import torch.nn.functional as F

from .citrus_e_v11_loss import EV11SegmentationLoss


def layer_targets(masks, batch_idx, batch_size, overlap, size):
    """Visible union and instance transitions; never infer concealed/full fruit shape.

    Compute transitions BEFORE reduction. Max pooling retains occupied cells for
    auxiliary discovery only; official instance targets are not changed. The band
    includes outer contours and touching-instance interfaces, not a convex prior.
    """
    unions, boundaries = [], []
    for i in range(batch_size):
        if overlap:
            labels = masks[i]
            union = labels > 0
            band = torch.zeros_like(union)
            horizontal = (labels[:, 1:] != labels[:, :-1]) & (union[:, 1:] | union[:, :-1])
            vertical = (labels[1:] != labels[:-1]) & (union[1:] | union[:-1])
            band[:, 1:] |= horizontal
            band[:, :-1] |= horizontal
            band[1:] |= vertical
            band[:-1] |= vertical
        else:
            instances = masks[batch_idx.flatten() == i] > 0
            union = instances.any(0)
            if len(instances):
                raw = instances[:, None].float()
                outer = F.max_pool2d(raw, 3, 1, 1)
                inner = -F.max_pool2d(-raw, 3, 1, 1)
                band = ((outer - inner)[:, 0] > 0).any(0)
            else:
                band = torch.zeros_like(union)
        unions.append(union)
        boundaries.append(band)
    target = torch.stack((torch.stack(unions), torch.stack(boundaries)), 1).float()
    return F.adaptive_max_pool2d(target, size)


def balanced_layer_bce(logits, targets):
    """Balance occupied/background pixels per image/channel; handle empty classes."""
    values = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    positive = targets.sum((-2, -1))
    negative = (1 - targets).sum((-2, -1))
    pos_loss = (values * targets).sum((-2, -1)) / positive.clamp_min(1)
    neg_loss = (values * (1 - targets)).sum((-2, -1)) / negative.clamp_min(1)
    active = (positive > 0).float() + (negative > 0).float()
    return ((pos_loss + neg_loss) / active.clamp_min(1)).mean()


class IV4SegmentationLoss(EV11SegmentationLoss):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.layer_gain = model.model[-1].layer_aux_gain
        self.boundary_enabled = model.model[-1].scene_neck.mode == 3

    def loss(self, preds, batch):
        total, components = super().loss(preds, batch)
        maps = preds.get("iv4_scene_layers", [])
        if not maps:
            return total, components
        target = layer_targets(batch["masks"].to(self.device), batch["batch_idx"].to(self.device),
                               len(preds["proto"]), self.overlap, maps[0].shape[-2:])
        terms = []
        for logits in maps:
            term = balanced_layer_bce(logits[:, :1], target[:, :1])
            if self.boundary_enabled:
                term = term + 0.5 * balanced_layer_bce(logits[:, 1:], target[:, 1:])
            else:
                term = term + logits[:, 1:].sum() * 0
            terms.append(term)
        value = self.layer_gain * torch.stack(terms).mean()
        extra = torch.stack((value * 0, value * 0, value * 0, value * 0, value))
        return total + extra * len(preds["proto"]), components + extra.detach()
