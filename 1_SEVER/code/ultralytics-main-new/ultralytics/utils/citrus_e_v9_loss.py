# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""E V9: tiny-instance overlap and cautious negative quality supervision.

Base TAL, box, classification, BCE, geometry and positive-quality losses remain
unchanged. These are separate testable interventions, not a claim to recover
objects with no assigned anchor, nor a change to PR/AP interpolation.
"""

import torch
import torch.nn.functional as F

from .citrus_e_v7_loss import EV7SegmentationLoss
from .ops import crop_mask, xywh2xyxy


def outside_gt_boxes(centers, boxes, margin=8.0):
    """Exclude centers inside ANY expanded GT box, including unassigned tiny GTs."""
    if not len(boxes):
        return torch.ones(len(centers), dtype=torch.bool, device=centers.device)
    lo, hi = boxes[:, :2] - margin, boxes[:, 2:] + margin
    inside = (centers[:, None] >= lo).all(-1) & (centers[:, None] <= hi).all(-1)
    return ~inside.any(1)


def tiny_instance_dice(coefficients, proto, gt, inverse, boxes, pixel_size):
    """One mean-coefficient mask per tiny GT; no anchor-multiplicity weighting.

    Only nonempty visible masks below 256 input-pixel area are eligible. This
    complements the inherited boundary term that excludes those tiny instances.
    A two-mask-pixel box margin preserves local negative pixels. GT is never
    dilated, filled, made convex, or overwritten. All ordinary mask losses
    still supervise every positive anchor; this auxiliary uses their mean.
    """
    area = gt.sum((1, 2)) * pixel_size[0] * pixel_size[1]
    eligible = (area > 0) & (area < 256)
    zero = coefficients.sum() * 0 + proto.sum() * 0
    if not eligible.any():
        return zero, gt.new_zeros(())
    count = torch.bincount(inverse, minlength=len(gt)).to(coefficients.dtype).clamp_min(1)
    mean = coefficients.new_zeros((len(gt), coefficients.shape[1])).index_add(0, inverse, coefficients) / count[:, None]
    mean_boxes = boxes.new_zeros((len(gt), 4)).index_add(0, inverse, boxes) / count[:, None]
    logits = torch.einsum("nc,chw->nhw", mean[eligible], proto).float()
    region_boxes = mean_boxes[eligible] + boxes.new_tensor([-2, -2, 2, 2])
    # crop_mask's CPU fast path is in-place; crop a fresh tensor, never sigmoid's output.
    region = crop_mask(torch.ones_like(logits), region_boxes)
    probability = logits.sigmoid() * region
    target = gt[eligible].float() * region
    numerator = 2 * (probability * target).sum((1, 2))
    denominator = probability.square().sum((1, 2)) + target.square().sum((1, 2))
    value = 1 - (numerator + 1e-6) / (denominator + 1e-6)
    return value.sum(), eligible.sum()


class EV9SegmentationLoss(EV7SegmentationLoss):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.tiny_gain = model.model[-1].tiny_dice_gain
        self.negative_gain = model.model[-1].negative_quality_gain
        self._tiny = None
        self.last_tiny = None
        self.last_negative_quality = None

    def loss(self, preds, batch):
        self._tiny = None
        try:
            total, components = super().loss(preds, batch)
            tiny = self._tiny if self._tiny is not None else preds["proto"].sum() * 0
            self.last_tiny = tiny.detach()
            q = self.tiny_gain * tiny
            addition = torch.stack((q * 0, q * 0, q * 0, q * 0, q))
            return total + addition * preds["proto"].shape[0], components + addition.detach()
        finally:
            self._tiny = None

    def calculate_segmentation_loss(
        self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    ):
        base = super().calculate_segmentation_loss(
            fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
        )
        if not self.tiny_gain:
            return base
        h, w = proto.shape[-2:]
        total, count = proto.sum() * 0, proto.new_zeros(())
        boxes = target_bboxes / imgsz[[1, 0, 1, 0]] * proto.new_tensor([w, h, w, h])
        pixel_size = (self._input_size[0] / h, self._input_size[1] / w)
        for i in range(len(proto)):
            selected = fg_mask[i]
            ids, inverse = torch.unique(target_gt_idx[i, selected], sorted=True, return_inverse=True)
            if not len(ids):
                continue
            gt = (
                (masks[i] == (ids + 1)[:, None, None]).float()
                if self.overlap
                else masks[batch_idx.flatten() == i][ids].float()
            )
            value, n = tiny_instance_dice(
                pred_masks[i, selected], proto[i], gt, inverse, boxes[i, selected], pixel_size
            )
            total, count = total + value, count + n
        self._tiny = total / count.clamp_min(1)
        return base

    def _quality_loss(self, preds, batch):
        positive = super()._quality_loss(preds, batch)
        quality = preds["ev3_quality"][:, 0]
        negative = quality.sum() * 0
        if self.negative_gain:
            fg, _, _, anchors, strides = self._assignment
            centers = (anchors * strides).detach()
            ih, iw = batch["img"].shape[-2:]
            boxes = xywh2xyxy(batch["bboxes"].to(centers.device)) * centers.new_tensor([iw, ih, iw, ih])
            pieces = []
            for i in range(len(quality)):
                image_boxes = boxes[batch["batch_idx"].flatten().to(boxes.device) == i]
                # TAL's empty-target fast path may return floating zeros.
                eligible = (~fg[i].bool()) & outside_gt_boxes(centers, image_boxes)
                idx = eligible.nonzero(as_tuple=False).flatten()
                if len(idx):
                    scores = preds["scores"][i, :, idx].detach().sigmoid().amax(0)
                    idx = idx[scores.topk(min(32, len(idx))).indices]
                    pieces.append(F.mse_loss(quality[i, idx].sigmoid(), torch.zeros_like(quality[i, idx])))
            if pieces:
                negative = torch.stack(pieces).mean()
        self.last_negative_quality = negative.detach()
        return positive + self.negative_gain * negative
