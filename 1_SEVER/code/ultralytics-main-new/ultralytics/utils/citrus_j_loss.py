# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J supervised visible-mask losses; no circular completion or invented labels."""

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask


def binary_boundary(mask):
    """Three-pixel morphological band of each VISIBLE instance, including concavities."""
    padded = F.pad(mask[:, None].float(), (1, 1, 1, 1))
    maximum = F.max_pool2d(padded, 3, 1)
    minimum = -F.max_pool2d(-padded, 3, 1)
    return (maximum - minimum)[:, 0].clamp(0, 1)


class JSegmentationLoss(v8SegmentationLoss):
    """Reuse the standard assignment/box/class losses; isolate mask-side hypotheses."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        head = model.model[-1]
        self.tiny_gain, self.boundary_gain, self.scene_gain = head.tiny_gain, head.boundary_gain, head.scene_gain
        self.extra_mask = None

    def instance_targets(self, masks, batch_idx, image, ids=None):
        if self.overlap:
            labels = masks[image]
            if ids is None:
                ids = labels.unique()
                ids = ids[ids > 0] - 1
            return (labels[None] == (ids + 1)[:, None, None]).float()
        values = masks[batch_idx.flatten() == image]
        return values if ids is None else values[ids]

    def calculate_segmentation_loss(
        self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    ):
        result = super().calculate_segmentation_loss(
            fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
        )
        if not (self.tiny_gain or self.boundary_gain):
            return result
        terms = []
        h, w = proto.shape[-2:]
        for image in range(len(proto)):
            positive = fg_mask[image]
            ids, inverse = target_gt_idx[image, positive].unique(sorted=True, return_inverse=True)
            if not len(ids):
                continue
            # Equal instance contribution: average positive-anchor coefficients BEFORE loss.
            coeff = pred_masks[image, positive]
            mean_coeff = coeff.new_zeros((len(ids), coeff.shape[1])).index_add(0, inverse, coeff)
            count = torch.bincount(inverse, minlength=len(ids)).to(coeff.dtype).clamp_min(1)
            mean_coeff = mean_coeff / count[:, None]
            target = self.instance_targets(masks, batch_idx, image, ids)
            logit = torch.einsum("in,nhw->ihw", mean_coeff, proto[image])
            # All matched anchors for one GT share its target box; mean is deterministic.
            boxes = target_bboxes[image, positive]
            gt_boxes = boxes.new_zeros((len(ids), 4)).index_add(0, inverse, boxes) / count[:, None]
            scaled = gt_boxes / imgsz[[1, 0, 1, 0]] * boxes.new_tensor([w, h, w, h])
            # crop_mask uses in-place slicing for small CPU batches; preserve the
            # sigmoid output required by autograd, including CPU smoke tests.
            probability = crop_mask(logit.sigmoid().clone(), scaled)
            target = crop_mask(target, scaled)
            smooth = 1.0
            dice = 1 - (2 * (probability * target).sum((1, 2)) + smooth) / (
                probability.sum((1, 2)) + target.sum((1, 2)) + smooth
            )
            area_pixels = target.sum((1, 2)) * imgsz.prod() / (h * w)
            size_weight = (256 / area_pixels.clamp_min(1)).sqrt().clamp(1, 2)
            term = self.tiny_gain * (size_weight * dice).sum() / size_weight.sum()
            if self.boundary_gain:
                band = binary_boundary(target)
                bce = F.binary_cross_entropy_with_logits(logit.float(), target.float(), reduction="none")
                # Fixed support is derived from GT only; missing boundaries cannot hide loss.
                valid_band = crop_mask(band, scaled)
                term = (
                    term
                    + self.boundary_gain * ((bce * valid_band).sum((1, 2)) / valid_band.sum((1, 2)).clamp_min(1)).mean()
                )
            terms.append(term)
        self.extra_mask = torch.stack(terms).mean() if terms else proto.sum() * 0
        return result

    def loss(self, preds, batch):
        self.extra_mask = None
        try:
            total, components = super().loss(preds, batch)
            extra = self.extra_mask if self.extra_mask is not None else preds["proto"].sum() * 0
            if self.scene_gain:
                scene = preds["j_scene"]
                masks = batch["masks"].to(scene.device)
                unions, boundaries = [], []
                for image in range(len(scene)):
                    target = self.instance_targets(masks, batch["batch_idx"], image)
                    if len(target):
                        union = target.amax(0)
                        boundary = binary_boundary(target).amax(0)
                    else:
                        union = masks.new_zeros(masks.shape[-2:]).float()
                        boundary = union
                    unions.append(union)
                    boundaries.append(boundary)
                target = torch.stack((torch.stack(unions), torch.stack(boundaries)), 1)
                # Max-pool keeps a positive tiny support rather than nearest downsample disappearance.
                target = F.adaptive_max_pool2d(target, scene.shape[-2:])
                extra = extra + self.scene_gain * F.binary_cross_entropy_with_logits(scene.float(), target.float())
            elif "j_scene" in preds:
                extra = extra + preds["j_scene"].sum() * 0
            # Aux gains are effective gains, NOT secretly multiplied by box=7.5.
            additions = torch.stack((extra * 0, extra, extra * 0, extra * 0, extra * 0))
            return total + additions * len(preds["proto"]), components + additions.detach()
        finally:
            self.extra_mask = None
