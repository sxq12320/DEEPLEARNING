# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Mask-IoU score supervision with predicted-box crops and stable instance IDs."""

import torch
import torch.nn.functional as F

from .ops import crop_mask
from .sage_v4r_loss import SAGEV4RSegmentationLoss


@torch.no_grad()
def quality_targets(coefficients, prototype, boxes, gt):
    """No gradients through targets; boxes are predictions in prototype pixels."""
    logits = torch.einsum("nc,chw->nhw", coefficients, prototype)
    binary = crop_mask((logits > 0).float(), boxes) > 0
    positive = gt > 0
    intersection = (binary & positive).sum((1, 2)).float()
    union = (binary | positive).sum((1, 2)).float().clamp_min(1)
    return intersection / union


class EV3SegmentationLoss(SAGEV4RSegmentationLoss):
    """Reuse TAL once; at most 64 positive anchors/image for the auxiliary only.

    Standard losses retain ALL positive anchors/labels. Missing rasterized ID 1
    never renumbers ID 2. Quality loss uses only surviving assigned GT masks.
    Work is bounded and no candidate-by-candidate Python mask rendering is used.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.quality_gain = model.model[-1].quality_gain
        self._assignment = None
        self.last_quality = None

    def get_assigned_targets_and_loss(self, preds, batch):
        result = super().get_assigned_targets_and_loss(preds, batch)
        self._assignment = result[0]
        return result

    def loss(self, preds, batch):
        self._assignment = None
        try:
            total, components = super().loss(preds, batch)
            quality = self._quality_loss(preds, batch)
            self.last_quality = quality.detach()
            q = quality * self.quality_gain
            addition = torch.stack((q * 0, q * 0, q * 0, q * 0, q))
            return total + addition * preds["proto"].shape[0], components + addition.detach()
        finally:
            self._assignment = None

    def _quality_loss(self, preds, batch):
        quality = preds["ev3_quality"][:, 0]
        fg, assigned, _, anchors, strides = self._assignment
        proto = preds["proto"].detach()
        coefficients = preds["mask_coefficient"].detach().permute(0, 2, 1)
        with torch.no_grad():
            boxes = self.bbox_decode(anchors, preds["boxes"].detach().permute(0, 2, 1)) * strides
            masks = batch["masks"].to(proto.device).float()
            if masks.shape[-2:] != proto.shape[-2:]:
                masks = F.interpolate(masks[:, None], proto.shape[-2:], mode="nearest")[:, 0]
            ih, iw = batch["img"].shape[-2:]
            mh, mw = proto.shape[-2:]
            boxes = boxes * boxes.new_tensor([mw / iw, mh / ih, mw / iw, mh / ih])
        pieces = []
        for i in range(len(proto)):
            indices = fg[i].nonzero(as_tuple=False).flatten()
            if not len(indices):
                continue
            # Deterministic strided sample, independent of model score; no extra RNG consumption.
            if len(indices) > 64:
                positions = torch.linspace(0, len(indices)-1, 64, device=indices.device).long()
                indices = indices[positions]
            ids = assigned[i, indices]
            if self.overlap:
                gt = masks[i] == (ids + 1)[:, None, None]
            else:
                gt = masks[batch["batch_idx"].view(-1) == i][ids] > 0
            valid = gt.flatten(1).any(1)
            if not valid.any():
                continue
            indices, gt = indices[valid], gt[valid]
            target = quality_targets(coefficients[i, indices], proto[i], boxes[i, indices], gt)
            pieces.append(F.mse_loss(quality[i, indices].sigmoid(), target))
        return torch.stack(pieces).mean() if pieces else quality.sum() * 0
