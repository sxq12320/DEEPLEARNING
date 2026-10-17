# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""J V3 loss: YOLO segmentation terms under incomplete annotation, driven by per-instance label-trust codes.

Every instance carries one float ``cls`` code, so the trust level survives Mosaic, perspective, copy-paste and the
overlap-mask sort exactly like a class id would:

* ``0``                 human ground truth (weight 1);
* ``1 + w``, w in (0,1] teacher pseudo-positive with weight ``w`` (scales its classification target, box/DFL and
  mask terms);
* ``3 + s``, s in [0,1] teacher ignore region: never a positive, and anchors whose centres fall inside it are removed
  from the background (negative) classification loss.

With only ``0`` codes the loss is numerically identical to ``v8SegmentationLoss`` (contract-tested).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.ops import crop_mask, xywh2xyxy, xyxy2xywh
from ultralytics.utils.tal import make_anchors

GT_CODE = 0.0
PSEUDO_BASE = 1.0
IGNORE_BASE = 3.0
_PSEUDO_FROM, _IGNORE_FROM = 0.5, 2.5  # decision boundaries between the three code bands


def pseudo_code(weight):
    """Code of a pseudo-positive instance with trust ``weight`` in (0, 1]."""
    if not 0.0 < weight <= 1.0:
        raise ValueError("pseudo-positive weight must be in (0, 1]")
    return PSEUDO_BASE + float(weight)


def ignore_code(score=0.0):
    """Code of an ignore region; the teacher score is kept only for bookkeeping."""
    return IGNORE_BASE + min(max(float(score), 0.0), 1.0)


def decode_codes(codes):
    """Return (kind, weight): kind 0=GT, 1=pseudo-positive, 2=ignore; weight is 1 for GT, w for pseudo, 0 for ignore.

    Works for numpy arrays and torch tensors of any shape.
    """
    if isinstance(codes, torch.Tensor):
        kind = (codes >= _PSEUDO_FROM).long() + (codes >= _IGNORE_FROM).long()
        weight = torch.where(kind == 0, torch.ones_like(codes), (codes - PSEUDO_BASE).clamp(0.0, 1.0))
        return kind, torch.where(kind == 2, torch.zeros_like(codes), weight)
    import numpy as np

    codes = np.asarray(codes, dtype=np.float64)
    kind = (codes >= _PSEUDO_FROM).astype(np.int64) + (codes >= _IGNORE_FROM).astype(np.int64)
    weight = np.where(kind == 0, 1.0, np.clip(codes - PSEUDO_BASE, 0.0, 1.0))
    return kind, np.where(kind == 2, 0.0, weight)


class J3SegmentationLoss(v8SegmentationLoss):
    """Task-aligned assignment over trusted positives only; weighted pseudo-positives; ignore regions mask negatives."""

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        unsupported = {
            name: value
            for name, value in (
                ("citrus_vfl", self.citrus_vfl),
                ("citrus_exclusive", self.citrus_exclusive_gain),
                ("citrus_quality", self.citrus_quality_gain),
            )
            if value
        }
        if unsupported:
            raise ValueError(f"J3 re-indexes positives; these auxiliary terms are not supported: {unsupported}")
        self._positive_index = None
        self._anchor_weight = None
        self.stats = dict(batches=0, gt=0, pseudo=0, ignore_regions=0, ignored_anchors=0, positive_anchors=0)

    @staticmethod
    def within_image_index(batch_idx, batch_size):
        """Position of every instance inside its own image, in batch order (= overlap-mask id minus one)."""
        batch_idx = batch_idx.long()
        if not len(batch_idx):
            return batch_idx
        counts = torch.bincount(batch_idx, minlength=batch_size)
        starts = torch.cumsum(counts, 0) - counts
        return torch.arange(len(batch_idx), device=batch_idx.device) - starts[batch_idx]

    @staticmethod
    def ignored_anchors(batch_idx, boxes, batch_size, points, imgsz):
        """(B, A) anchors whose centre lies inside any ignore box of the same image."""
        out = torch.zeros((batch_size, points.shape[0]), dtype=torch.bool, device=points.device)
        if not len(boxes):
            return out
        xyxy = xywh2xyxy(boxes * imgsz[[1, 0, 1, 0]])
        x, y = points[:, 0], points[:, 1]
        for image in batch_idx.long().unique().tolist():
            b = xyxy[batch_idx.long() == image]
            inside = (x[None] >= b[:, :1]) & (x[None] <= b[:, 2:3]) & (y[None] >= b[:, 1:2]) & (y[None] <= b[:, 3:4])
            out[image] = inside.any(0)
        return out

    def get_assigned_targets_and_loss(self, preds, batch):
        """Same structure as ``v8DetectionLoss.get_assigned_targets_and_loss`` with trust-aware targets."""
        loss = torch.zeros(3, device=self.device)  # box, cls, dfl
        pred_distri, pred_scores = (
            preds["boxes"].permute(0, 2, 1).contiguous(),
            preds["scores"].permute(0, 2, 1).contiguous(),
        )
        anchor_points, stride_tensor = make_anchors(preds["feats"], self.stride, 0.5)
        dtype = pred_scores.dtype
        batch_size = pred_scores.shape[0]
        imgsz = torch.tensor(preds["feats"][0].shape[2:], device=self.device, dtype=dtype) * self.stride[0]

        batch_idx = batch["batch_idx"].view(-1).to(self.device)
        codes = batch["cls"].view(-1).to(self.device).float()
        boxes = batch["bboxes"].to(self.device)
        kind, weight = decode_codes(codes)
        within = self.within_image_index(batch_idx, batch_size)
        positive, ignore = kind < 2, kind == 2

        targets = torch.cat(
            (
                batch_idx[positive, None].float(),
                torch.zeros_like(codes[positive, None]),
                boxes[positive],
                weight[positive, None],
                within[positive, None].float(),
            ),
            1,
        )
        targets = self.preprocess(targets, batch_size, scale_tensor=imgsz[[1, 0, 1, 0]])
        gt_labels, gt_bboxes, gt_weight, gt_within = targets.split((1, 4, 1, 1), 2)
        mask_gt = gt_bboxes.sum(2, keepdim=True).gt_(0.0)

        pred_bboxes = self.bbox_decode(anchor_points, pred_distri)  # xyxy, (b, h*w, 4)
        _, target_bboxes, target_scores, fg_mask, target_gt_idx = self.assigner(
            pred_scores.detach().sigmoid(),
            (pred_bboxes.detach() * stride_tensor).type(gt_bboxes.dtype),
            anchor_points * stride_tensor,
            gt_labels,
            gt_bboxes,
            mask_gt,
        )
        fg_mask = fg_mask.bool()  # the assigner returns a float zero mask when an image batch has no positives
        if gt_weight.shape[1]:
            anchor_weight = torch.gather(gt_weight.squeeze(-1), 1, target_gt_idx)
            anchor_weight = torch.where(fg_mask, anchor_weight, torch.ones_like(anchor_weight))
            target_scores = target_scores * anchor_weight.unsqueeze(-1).to(target_scores.dtype)
        else:
            anchor_weight = torch.ones(fg_mask.shape, device=self.device)
        ignored = self.ignored_anchors(batch_idx[ignore], boxes[ignore], batch_size, anchor_points * stride_tensor,
                                       imgsz) & ~fg_mask

        target_scores_sum = max(target_scores.sum(), 1)
        bce_loss = self.bce(pred_scores, target_scores.to(dtype))  # (bs, num_anchors, nc)
        if self.class_weights is not None:
            bce_loss *= self.class_weights
        loss[1] = bce_loss.masked_fill(ignored.unsqueeze(-1), 0.0).sum() / target_scores_sum

        if fg_mask.sum():
            loss[0], loss[2] = self.bbox_loss(
                pred_distri,
                pred_bboxes,
                anchor_points,
                target_bboxes / stride_tensor,
                target_scores,
                target_scores_sum,
                fg_mask,
                imgsz,
                stride_tensor,
            )
        loss[0] *= self.hyp.box
        loss[1] *= self.hyp.cls
        loss[2] *= self.hyp.dfl

        self._positive_index = gt_within.squeeze(-1).long()
        self._anchor_weight = anchor_weight
        self.stats["batches"] += 1
        self.stats["gt"] += int((kind == 0).sum())
        self.stats["pseudo"] += int((kind == 1).sum())
        self.stats["ignore_regions"] += int(ignore.sum())
        self.stats["ignored_anchors"] += int(ignored.sum())
        self.stats["positive_anchors"] += int(fg_mask.sum())
        return (fg_mask, target_gt_idx, target_bboxes, anchor_points, stride_tensor), loss, loss.detach()

    def calculate_segmentation_loss(
        self, fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
    ):
        """Standard per-anchor mask BCE; GT ids are mapped back past filtered ignore regions and weighted by trust."""
        _, _, mask_h, mask_w = proto.shape
        loss = 0
        normalized = target_bboxes / imgsz[[1, 0, 1, 0]]
        marea = xyxy2xywh(normalized)[..., 2:].prod(2)
        mxyxy = normalized * torch.tensor([mask_w, mask_h, mask_w, mask_h], device=proto.device)
        index, weight = self._positive_index, self._anchor_weight
        for i in range(len(fg_mask)):
            fg = fg_mask[i]
            if fg.any():
                mask_idx = index[i][target_gt_idx[i][fg]]
                if self.overlap:
                    gt_mask = (masks[i] == (mask_idx + 1).view(-1, 1, 1)).float()
                else:
                    gt_mask = masks[batch_idx.view(-1) == i][mask_idx]
                pred_mask = torch.einsum("in,nhw->ihw", pred_masks[i][fg], proto[i])
                term = F.binary_cross_entropy_with_logits(pred_mask, gt_mask, reduction="none")
                term = crop_mask(term, mxyxy[i][fg]).mean(dim=(1, 2)) / marea[i][fg]
                loss += (term * weight[i][fg].to(term.dtype)).sum()
            else:  # keep DDP gradients defined, as in the stock loss
                loss += (proto * 0).sum() + (pred_masks * 0).sum()
        return loss / fg_mask.sum()

    def loss(self, preds, batch):
        try:
            return super().loss(preds, batch)
        finally:
            self._positive_index = None
            self._anchor_weight = None
