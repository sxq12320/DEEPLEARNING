# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""V11: isolate tiny-box assignment from all-object regression and visible mask GT.

NWD authors' Gaussian box distance inspires a selective TAL quality blend.
This is not a faithful RFLA/RKA reproduction. Candidate geometry, box targets,
CIoU/DFL regression and official evaluation stay unchanged. A small training-only
local contrast term excludes ALL labelled fruit from each exterior ring.
"""

import torch
import torch.nn.functional as F

from .citrus_e_v9_loss import EV9SegmentationLoss
from .tal import TaskAlignedAssigner


class EV11TinyAssigner(TaskAlignedAssigner):
    """Blend NWD only for tiny GT; retain exact standard TAL for other GT.

    Boxes with area<32² input pixels use q=(1-mix)*CIoU_+ + mix*NWD,
    NWD=exp(-sqrt(||dc||²+||dwh/2||²)/12.8). This quality drives both ranking
    and soft labels, explicitly NOT just matching. No unconditional label floor,
    forced positive, or guarantee that overlapping/colliding GT all get anchors.
    """

    def __init__(self, mix=0.2, **kwargs):
        super().__init__(**kwargs)
        self.mix = float(mix)

    def iou_calculation(self, gt_bboxes, pd_bboxes):
        base = super().iou_calculation(gt_bboxes, pd_bboxes)
        if not self.mix:
            return base
        size = (gt_bboxes[:, 2:] - gt_bboxes[:, :2]).clamp_min(0)
        center = (gt_bboxes[:, 2:] + gt_bboxes[:, :2]) * 0.5
        pred_size = (pd_bboxes[:, 2:] - pd_bboxes[:, :2]).clamp_min(0)
        pred_center = (pd_bboxes[:, 2:] + pd_bboxes[:, :2]) * 0.5
        distance = ((center - pred_center).square().sum(-1) + 0.25 * (size - pred_size).square().sum(-1)).clamp_min(
            1e-7
        )
        nwd = (-distance.sqrt() / 12.8).exp()
        tiny = (size.prod(-1) > 0) & (size.prod(-1) < 1024)
        return torch.where(tiny, (1 - self.mix) * base + self.mix * nwd, base)


def local_mask_contrast(coefficients, prototype, targets, inverse, union):
    """At most eight smallest assigned GT/image, narrow grid, with authentic visible contours.

    Mean positive-anchor coefficients prevent GT with many anchors dominating.
    Auxiliary max-pooling retains occupied cells; the ordinary GT mask loss is
    untouched. Exterior rings exclude every other labelled instance, so touching
    fruit are not trained as background. Unlabelled fruit cannot be distinguished
    from background; this limitation is recorded in the experiment document.
    """
    if not len(targets):
        return prototype.sum() * 0 + coefficients.sum() * 0
    size = (max(1, prototype.shape[-2] // 2), max(1, prototype.shape[-1] // 2))
    gt = F.adaptive_max_pool2d(targets[:, None].float(), size)[:, 0] > 0
    whole = F.adaptive_max_pool2d(union[None, None].float(), size)[0, 0] > 0
    area = gt.sum((1, 2))
    valid = (area > 0).nonzero(as_tuple=False).flatten()
    if not len(valid):
        return prototype.sum() * 0 + coefficients.sum() * 0
    selected = valid[area[valid].argsort()[:8]]
    count = torch.bincount(inverse, minlength=len(targets)).to(coefficients.dtype).clamp_min(1)
    mean = (
        coefficients.new_zeros((len(targets), coefficients.shape[1])).index_add(0, inverse, coefficients)
        / count[:, None]
    )
    proto = F.interpolate(prototype[None], size, mode="bilinear", align_corners=False)[0]
    logits = torch.einsum("nc,chw->nhw", mean[selected], proto).float()
    inside = gt[selected]
    ring = (F.max_pool2d(inside[:, None].float(), 5, 1, 2)[:, 0] > 0) & (~whole)
    n_ring = ring.sum((1, 2))
    active = n_ring > 0
    if not active.any():
        return logits.sum() * 0
    positive = (logits * inside).sum((1, 2)) / inside.sum((1, 2)).clamp_min(1)
    negative = (logits * ring).sum((1, 2)) / n_ring.clamp_min(1)
    # Relative evidence and background suppression, not a roundness/convexity prior.
    margin = F.softplus(0.5 - positive + negative)
    background = (F.softplus(logits) * ring).sum((1, 2)) / n_ring.clamp_min(1)
    return (margin[active] + 0.5 * background[active]).mean()


class EV11SegmentationLoss(EV9SegmentationLoss):
    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        head = model.model[-1]
        self.ring_gain = head.ring_gain
        self.last_ring = None
        self._ring = None
        if head.assignment_mix:
            a = self.assigner
            self.assigner = EV11TinyAssigner(
                mix=head.assignment_mix,
                topk=a.topk,
                num_classes=a.num_classes,
                alpha=a.alpha,
                beta=a.beta,
                stride=a.stride,
                topk2=a.topk2,
                eps=a.eps,
            )

    def loss(self, preds, batch):
        self._ring = None
        try:
            total, components = super().loss(preds, batch)
            value = self._ring if self._ring is not None else preds["proto"].sum() * 0
            self.last_ring = value.detach()
            term = self.ring_gain * value
            extra = torch.stack((term * 0, term * 0, term * 0, term * 0, term))
            return total + extra * len(preds["proto"]), components + extra.detach()
        finally:
            self._ring = None

    def calculate_segmentation_loss(
        self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    ):
        value = super().calculate_segmentation_loss(
            fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
        )
        if not self.ring_gain:
            return value
        terms = []
        for i in range(len(proto)):
            positive = fg_mask[i]
            ids, inverse = torch.unique(target_gt_idx[i, positive], sorted=True, return_inverse=True)
            if not len(ids):
                continue
            if self.overlap:
                gt = masks[i] == (ids + 1)[:, None, None]
                union = masks[i] > 0
            else:
                all_gt = masks[batch_idx.flatten() == i]
                gt, union = all_gt[ids], (all_gt > 0).any(0)
            terms.append(local_mask_contrast(pred_masks[i, positive], proto[i], gt, inverse, union))
        self._ring = torch.stack(terms).mean() if terms else proto.sum() * 0
        return value
