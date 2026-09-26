"""One-to-one visible-instance supervision for the I_V4 independent segmenter."""

import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def _visible_instances(batch, image_index, size, overlap):
    """Read the official visible masks; never infer hidden or convex fruit regions."""
    indices = batch["batch_idx"].view(-1) == image_index
    count = int(indices.sum().item())
    if overlap:
        raster = batch["masks"][image_index]
        if count:
            masks = torch.arange(1, count + 1, device=raster.device).view(-1, 1, 1) == raster
        else:
            masks = raster.new_zeros((0, *raster.shape), dtype=torch.bool)
    else:
        masks = batch["masks"][indices] > 0.5
    masks = masks.float()
    if masks.shape[-2:] != size and count:
        masks = (F.adaptive_max_pool2d(masks.unsqueeze(1), size)[:, 0]
                 if size[0] < masks.shape[-2] or size[1] < masks.shape[-1]
                 else F.interpolate(masks.unsqueeze(1), size=size, mode="nearest")[:, 0])
    return masks


def _balanced_bce(logits, targets):
    values = F.binary_cross_entropy_with_logits(logits.float(), targets.float(), reduction="none")
    positive = targets.sum((-2, -1)).clamp_min(1)
    negative = (1 - targets).sum((-2, -1)).clamp_min(1)
    return ((values * targets).sum((-2, -1)) / positive +
            (values * (1 - targets)).sum((-2, -1)) / negative).mean()


class CitrusNativeLoss:
    """Hungarian set matching with scene, evidence, visible-mask and mild review supervision."""

    def __init__(self, model):
        head = model.model[-1]
        self.overlap = bool(getattr(model.args, "overlap_mask", True)) if not isinstance(
            getattr(model, "args", None), dict
        ) else bool(model.args.get("overlap_mask", True))
        self.scene_enabled = head.scene_enabled
        self.evidence_enabled = head.evidence_enabled
        self.review_enabled = head.review_enabled

    @staticmethod
    @torch.no_grad()
    def _match(logits, scores, truth):
        if len(truth) == 0:
            empty = torch.zeros(0, device=logits.device, dtype=torch.long)
            return empty, empty
        low = F.interpolate(logits[:, None].float(), (32, 32), mode="bilinear", align_corners=False)[:, 0]
        target = F.adaptive_max_pool2d(truth[:, None].float(), (32, 32))[:, 0]
        prob = low.sigmoid().flatten(1)
        gt = target.flatten(1)
        intersection = prob @ gt.T
        dice = (2 * intersection + 1) / (prob.sum(1)[:, None] + gt.sum(1)[None, :] + 1)
        cost = 1 - dice - 0.2 * scores.sigmoid()[:, None]
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        return (torch.as_tensor(row, device=logits.device, dtype=torch.long),
                torch.as_tensor(col, device=logits.device, dtype=torch.long))

    def __call__(self, predictions, batch):
        if isinstance(predictions, tuple):
            predictions = predictions[2]  # evaluation carries raw heads for loss, alongside formatted masks
        mask_logits, class_logits = predictions["masks"], predictions["scores"]
        scene_logits, evidence_logits = predictions["scene"], predictions["evidence"]
        batch_size = len(mask_logits)
        terms = mask_logits.new_zeros(5)
        scene_targets = []
        for image_index in range(batch_size):
            gt = _visible_instances(batch, image_index, mask_logits.shape[-2:], self.overlap)
            if len(gt) > mask_logits.shape[1]:
                raise ValueError("Image has {} instances but only {} object slots".format(len(gt), mask_logits.shape[1]))
            row, col = self._match(mask_logits[image_index], class_logits[image_index], gt)
            label = torch.zeros_like(class_logits[image_index])
            label[row] = 1
            cls_bce = F.binary_cross_entropy_with_logits(class_logits[image_index], label, reduction="none")
            terms[1] = terms[1] + (cls_bce * (0.15 + 0.85 * label)).sum() / max(1, len(row))
            if len(row):
                pred = mask_logits[image_index, row].float()
                target = gt[col].float()
                per_mask_bce = F.binary_cross_entropy_with_logits(pred, target, reduction="none").mean((-2, -1))
                p = pred.sigmoid().flatten(1)
                t = target.flatten(1)
                dice = 1 - (2 * (p * t).sum(1) + 1) / (p.sum(1) + t.sum(1) + 1)
                # Bounded tiny-instance weighting; never remove tiny labeled objects.
                area = t.sum(1).clamp_min(1)
                weight = (area.mean() / area).sqrt().clamp(0.5, 2.0).detach()
                terms[0] = terms[0] + ((per_mask_bce + dice) * weight).mean()
            else:
                terms[0] = terms[0] + mask_logits[image_index].sum() * 0

            union = gt.amax(0, keepdim=True) if len(gt) else torch.zeros_like(mask_logits[image_index, :1])
            if len(gt):
                raw = gt.unsqueeze(1)
                outer = F.max_pool2d(raw, 3, 1, 1)
                inner = -F.max_pool2d(-raw, 3, 1, 1)
                boundary = (outer - inner).amax(0)
            else:
                boundary = union.clone()
            scene_targets.append(torch.cat((union, boundary), 0))

            if self.evidence_enabled:
                evidence = evidence_logits[image_index, 0].float()
                evidence_union = F.interpolate(union[None], evidence.shape[-2:], mode="nearest")[0, 0]
                background = F.softplus(evidence) * (1 - evidence_union)
                value = 0.15 * background.mean()
                for instance in gt:
                    region = F.interpolate(instance[None, None], evidence.shape[-2:], mode="nearest")[0, 0] > 0
                    if region.any():
                        value = value + F.softplus(-evidence[region].amax()) / len(gt)
                terms[3] = terms[3] + value

        if self.scene_enabled:
            targets = torch.stack(scene_targets)
            terms[2] = _balanced_bce(scene_logits, targets)
            if self.review_enabled:
                # Low-weight agreement between object union and scene foreground; scene labels remain primary.
                object_union = 1 - torch.prod(1 - mask_logits.sigmoid(), dim=1, keepdim=True)
                terms[4] = 0.05 * F.l1_loss(object_union, scene_logits[:, :1].sigmoid())
        components = terms / batch_size
        total = components.sum() * batch_size
        return total, components.detach()
