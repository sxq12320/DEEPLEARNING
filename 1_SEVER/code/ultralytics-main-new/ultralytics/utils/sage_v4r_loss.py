# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Scale-resolvable visible-boundary and neighboring-instance mask supervision.

These are optional GT-guided reweightings of the existing instance logits, not
new topology guarantees. No convex-hull filling, roundness prior, CPU morphology,
pairwise instance matrix, or extra full-resolution mask head is used.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from .loss import v8SegmentationLoss
from .ops import crop_mask, xyxy2xywh


@torch.no_grad()
def instance_geometry_regions(gt, foreground, pixel_size=(4.0, 4.0)):
    """Build regions once per distinct assigned GT, NOT once per positive anchor.

    Args:
        gt: Unique binary instance masks (N,H,W), in visible-mask convention.
        foreground: Union of all labeled instances in this image (H,W).
        pixel_size: Input-image pixels per mask pixel (height,width).

    Returns:
        Boundary band, visible neighbor region, and boundary eligibility (N).
        Boundary radius is ~4 input px and neighbor radius ~8 input px. At least
        25% resolvable interior and area >=16**2 input pixels are required for the
        EXTRA boundary term. Ineligible instances KEEP their official mask loss.
    """
    sy, sx = (float(v) for v in pixel_size)
    if min(sy, sx) <= 0:
        raise ValueError("Mask pixel size must be positive")
    ry, rx = max(1, round(4 / sy)), max(1, round(4 / sx))
    ny, nx = max(1, round(8 / sy)), max(1, round(8 / sx))
    own = gt.float()[:, None]
    eroded = -F.max_pool2d(-F.pad(own, (rx, rx, ry, ry)), (2 * ry + 1, 2 * rx + 1), 1)
    dilated = F.max_pool2d(own, (2 * ry + 1, 2 * rx + 1), 1, (ry, rx))
    area = own.sum((1, 2, 3))
    interior = eroded.sum((1, 2, 3))
    eligible = (area * sy * sx >= 256) & (interior >= 0.25 * area) & (interior > 0)
    boundary = (dilated - eroded)[:, 0] * eligible[:, None, None]
    vicinity = F.max_pool2d(own, (2 * ny + 1, 2 * nx + 1), 1, (ny, nx))[:, 0]
    # Another fruit is negative for THIS instance, but our own visible pixels never are.
    neighbor = vicinity * (foreground[None] > 0).float() * (1 - gt.float())
    return boundary, neighbor, eligible


def _instance_average(values, valid, inverse, count):
    """Average anchors within each GT, then valid GTs; avoid anchor-multiplicity bias."""
    sums = values.new_zeros(count).scatter_add(0, inverse, values)
    counts = values.new_zeros(count).scatter_add(0, inverse, torch.ones_like(values))
    per_gt = sums / counts.clamp_min(1)
    return (per_gt * valid).sum(), valid.sum()


def geometry_from_bce(bce, assigned_gt, unique_gt, inverse, foreground, pixel_size):
    """Reuse official mask BCE; return sums/counts for the two optional terms."""
    boundary, neighbor, _ = instance_geometry_regions(unique_gt, foreground, pixel_size)
    positive = boundary * unique_gt
    negative = boundary * (1 - unique_gt)
    pcount, ncount = positive.sum((1, 2)), negative.sum((1, 2))
    # Matched logits have the SAME GT target as the official mask BCE.
    boundary_value = 0.5 * (
        (bce * positive[inverse]).sum((1, 2)) / pcount[inverse].clamp_min(1)
        + (bce * negative[inverse]).sum((1, 2)) / ncount[inverse].clamp_min(1)
    )
    boundary_sum, boundary_count = _instance_average(
        boundary_value, (pcount > 0) & (ncount > 0), inverse, len(unique_gt)
    )
    neighbor_count = neighbor.sum((1, 2))
    # assigned_gt is intentionally checked through the target construction, never inverted labels.
    neighbor_region = neighbor[inverse] * (1 - assigned_gt)
    neighbor_value = (bce * neighbor_region).sum((1, 2)) / neighbor_count[inverse].clamp_min(1)
    neighbor_sum, neighbor_valid = _instance_average(neighbor_value, neighbor_count > 0, inverse, len(unique_gt))
    return boundary_sum, boundary_count, neighbor_sum, neighbor_valid


class SAGEV4RSegmentationLoss(v8SegmentationLoss):
    """Official loss plus optional per-instance terms, logged together in sem_loss.

    Geometry gains are effective weights, not implicitly multiplied by box=7.5.
    The zero-gain route calls the original segmentation implementation unchanged.
    """

    def __init__(self, model, *args, **kwargs):
        super().__init__(model, *args, **kwargs)
        self.boundary_gain = float(model.model[-1].boundary_gain)
        self.neighbor_gain = float(model.model[-1].neighbor_gain)
        self.input_stride = int(model.model[-1].stride[0])
        self._geometry = None
        self.last_geometry = None

    def loss(self, preds, batch):
        self._geometry = None
        self._input_size = tuple(v * self.input_stride for v in preds["feats"][0].shape[-2:])
        total, components = super().loss(preds, batch)
        if self._geometry is not None:
            boundary, neighbor = self._geometry
            auxiliary = self.boundary_gain * boundary + self.neighbor_gain * neighbor
            addition = torch.stack((auxiliary * 0, auxiliary * 0, auxiliary * 0, auxiliary * 0, auxiliary))
            total = total + addition * preds["proto"].shape[0]
            components = components + addition.detach()
            self.last_geometry = torch.stack((boundary.detach(), neighbor.detach()))
        else:
            self.last_geometry = total.detach().new_zeros(2)
        self._geometry = None  # Do not retain an old autograd graph between batches.
        return total, components

    def calculate_segmentation_loss(
        self,
        fg_mask,
        masks,
        target_gt_idx,
        target_bboxes,
        batch_idx,
        proto,
        pred_masks,
        imgsz,
    ):
        if self.boundary_gain == 0 and self.neighbor_gain == 0:
            return super().calculate_segmentation_loss(
                fg_mask, masks, target_gt_idx, target_bboxes, batch_idx, proto, pred_masks, imgsz
            )
        _, _, mask_h, mask_w = proto.shape
        normalized_boxes = target_bboxes / imgsz[[1, 0, 1, 0]]
        area = xyxy2xywh(normalized_boxes)[..., 2:].prod(2)
        boxes = normalized_boxes * proto.new_tensor([mask_w, mask_h, mask_w, mask_h])
        # Input shape is metadata. Avoid device->host conversion in each per-image iteration.
        pixel_size = (self._input_size[0] / mask_h, self._input_size[1] / mask_w)
        loss = proto.sum() * 0 + pred_masks.sum() * 0
        boundary_sum = neighbor_sum = loss
        boundary_count = neighbor_count = proto.new_zeros(())
        for i in range(len(proto)):
            selected = fg_mask[i]
            if not selected.any():
                continue
            indices = target_gt_idx[i, selected]
            unique, inverse = torch.unique(indices, sorted=True, return_inverse=True)
            if self.overlap:
                unique_gt = (masks[i] == (unique + 1)[:, None, None]).float()
                foreground = masks[i] > 0
            else:
                image_masks = masks[batch_idx.flatten() == i]
                unique_gt = image_masks[unique].float()
                foreground = image_masks.amax(0) > 0
            gt = unique_gt[inverse]
            logits = torch.einsum("in,nhw->ihw", pred_masks[i, selected], proto[i])
            bce = F.binary_cross_entropy_with_logits(logits, gt, reduction="none")
            # crop_mask modifies its input IN PLACE only on the small-N CPU path.
            # Geometry must see uncropped logits, including visible neighboring GT.
            base_bce = bce.clone() if len(bce) < 50 and not bce.is_cuda else bce
            loss = loss + (crop_mask(base_bce, boxes[i, selected]).mean((1, 2)) / area[i, selected]).sum()
            bs, bc, ns, nc = geometry_from_bce(bce, gt, unique_gt, inverse, foreground, pixel_size)
            boundary_sum, neighbor_sum = boundary_sum + bs, neighbor_sum + ns
            boundary_count, neighbor_count = boundary_count + bc, neighbor_count + nc
        self._geometry = (boundary_sum / boundary_count.clamp_min(1), neighbor_sum / neighbor_count.clamp_min(1))
        return loss / fg_mask.sum().clamp_min(1)
