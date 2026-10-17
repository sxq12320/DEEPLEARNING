"""J5: GT-filtered, basis-independent instance distillation; no teacher at deployment.

The teacher sees the SAME augmented RGB view as the student. This is neither
privileged-resolution supervision nor pseudo-label correction. Do not compare
prototype channels directly: coefficients and bases have a change-of-basis
ambiguity. We compare their composed instance logits at bounded ROI samples.
"""

from __future__ import annotations

import csv
import math
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.nn.functional as F

from citrus_e_v6_training import EV6TrainingTrainer
from ultralytics.utils.tal import dist2bbox, make_anchors
from ultralytics.utils.torch_utils import unwrap_model


def raw_predictions(output):
    """Extract the ordinary one-to-many raw dictionary without decoded scores."""
    if isinstance(output, dict) and {"boxes", "scores", "mask_coefficient", "proto"} <= output.keys():
        return output
    if isinstance(output, (list, tuple)):
        for item in reversed(output):
            try:
                return raw_predictions(item)
            except ValueError:
                pass
    raise ValueError("J5 requires one-to-many boxes/scores/mask_coefficient/proto outputs")


def bernoulli_kl(logits, probability):
    """D_KL(Ber(teacher)||Ber(student)), stable in FP32 and zero at equality."""
    probability = probability.detach().float().clamp(1e-6, 1 - 1e-6)
    entropy = -(probability * probability.log() + (1 - probability) * torch.log1p(-probability))
    return (F.binary_cross_entropy_with_logits(logits.float(), probability, reduction="none") - entropy).clamp_min(0)


def composed_logits(prototype, coefficients, grid):
    """Sample C-channel basis BEFORE composition: O(C * points), not a full ROI mask.

    grid: [instances, points, 2], normalized grid_sample coordinates. The basis
    All instance grids are packed into ONE sampling call. Expanding the basis
    to an instance batch would allocate N dense basis gradients during backward,
    despite its cheap forward view. The packed form has one dense input gradient.
    The dot product is invariant to invertible basis transformations, up to
    floating-point error and interpolation's linearity.
    """
    count = len(coefficients)
    points = grid.shape[1]
    features = F.grid_sample(prototype[None].float(), grid.reshape(1, 1, count * points, 2).float(),
                             mode="bilinear", padding_mode="border", align_corners=False)
    features = features[0, :, 0].reshape(prototype.shape[0], count, points).permute(1, 0, 2)
    return torch.einsum("ic,icp->ip", coefficients.float(), features)


def _decode_boxes(predictions, stride):
    anchors, step = make_anchors(predictions["feats"], stride, 0.5)
    logits = predictions["boxes"].float().transpose(1, 2)
    bins = logits.shape[-1] // 4
    if bins > 1:
        distances = logits.reshape(*logits.shape[:2], 4, bins).softmax(-1)
        distances = distances @ torch.arange(bins, device=logits.device, dtype=logits.dtype)
    else:
        distances = logits
    return dist2bbox(distances, anchors, xywh=False) * step, anchors * step


def instance_distillation(student, teacher, batch, stride, point_grid=12, max_instances=32,
                          teacher_min_iou=0.5, teacher_min_score=0.25, overlap=True):
    """Average GT-filtered terms, then images with accepted instances.

    Annotated instances only; no background pseudo-positive is introduced.
    Each GT selects ONE teacher anchor with its centre inside the GT box,
    maximizing score * box IoU. If a tiny box has no centre, consider its nearest
    anchor. Reject weak box/score matches AND teacher masks failing sampled GT
    IoU. This gate can exclude the hardest tiny fruit: acceptance is reported.
    Supervised GT losses remain active for every original GT, including rejects.
    """
    zero = student["proto"].float().sum() * 0 + student["scores"].float().sum() * 0
    counter_names = ("annotated", "considered", "accepted", "no_anchor_centre",
                     "considered_small_box", "accepted_small_box", "rejected_box_or_score", "rejected_mask")
    counter = torch.zeros(len(counter_names), device=student["scores"].device, dtype=torch.long)
    if student["scores"].shape != teacher["scores"].shape:
        raise ValueError("Teacher/student class counts and P3/P4/P5 grids must match")
    if student["mask_coefficient"].shape[-1] != teacher["mask_coefficient"].shape[-1]:
        raise ValueError("Teacher/student candidate ordering differs")
    with torch.no_grad():
        boxes, centres = _decode_boxes({**teacher, "feats": student["feats"]}, stride)
    height, width = batch["img"].shape[-2:]
    scale = boxes.new_tensor([width, height, width, height])
    axis = (torch.arange(point_grid, device=boxes.device).float() + 0.5) / point_grid
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    unit = torch.stack((xx.flatten(), yy.flatten()), -1)
    image_mask_terms, image_class_terms, active_images = [], [], []
    for image in range(len(batch["img"])):
        ids = (batch["batch_idx"].flatten().long() == image).nonzero(as_tuple=False).flatten()
        counter[0] += len(ids)
        if not len(ids):
            continue
        sizes = batch["bboxes"][ids, 2:].float().prod(-1)
        selected = sizes.argsort()[:max_instances]  # bounded work; smallest GT first
        with torch.no_grad():
            # Per-image vectorization avoids Python CUDA scalar synchronizations
            # for every annotated instance (potentially thousands per step).
            xywh = batch["bboxes"][ids[selected]].float() * scale
            target = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), 1)
            inside = ((centres[None] > target[:, None, :2]) &
                      (centres[None] < target[:, None, 2:])).all(-1)
            no_centre = ~inside.any(1)
            nearest = (centres[None] - xywh[:, None, :2]).square().sum(-1).argmin(1)
            inside.scatter_(1, nearest[:, None], inside.gather(1, nearest[:, None]) | no_centre[:, None])
            cls = batch["cls"][ids[selected]].flatten().long()
            scores = teacher["scores"][image, cls].float().sigmoid()
            intersection = (torch.minimum(boxes[image, None, :, 2:], target[:, None, 2:]) -
                            torch.maximum(boxes[image, None, :, :2], target[:, None, :2])).clamp_min(0).prod(-1)
            union = (boxes[image, :, 2:] - boxes[image, :, :2]).clamp_min(0).prod(-1)[None] + xywh[:, 2:].prod(-1)[:, None]
            quality = intersection / (union - intersection).clamp_min(1e-7)
            anchors_chosen = (scores * quality).masked_fill(~inside, -1).argmax(1)
            row = torch.arange(len(selected), device=boxes.device)
            accept_box = ((quality[row, anchors_chosen] >= teacher_min_iou) &
                          (scores[row, anchors_chosen] >= teacher_min_score))
            lo = (xywh[:, :2] - 0.6 * xywh[:, 2:]).clamp_min(0)
            hi = torch.minimum(xywh[:, :2] + 0.6 * xywh[:, 2:], scale[:2])
            grid = 2 * (lo[:, None] + unit[None] * (hi - lo)[:, None]) / scale[:2] - 1
            if overlap:
                sampled_ids = F.grid_sample(batch["masks"][image][None, None].float(), grid[None],
                                            mode="nearest", padding_mode="zeros", align_corners=False)[0, 0]
                sampled_gt = sampled_ids == (selected + 1)[:, None]
            else:
                gt = batch["masks"][ids[selected]].float()
                sampled_gt = F.grid_sample(gt[:, None], grid[:, None], mode="nearest",
                                            padding_mode="zeros", align_corners=False)[:, 0, 0] > .5
            logits_teacher = composed_logits(teacher["proto"][image],
                teacher["mask_coefficient"][image, :, anchors_chosen].T, grid)
            positive = logits_teacher > 0
            mask_iou = (positive & sampled_gt).sum(-1) / (positive | sampled_gt).sum(-1).clamp_min(1)
            accept = accept_box & sampled_gt.any(-1) & (mask_iou >= teacher_min_iou)
            small = xywh[:, 2:].prod(-1) < 1024  # INPUT box area, not public640 tiny-mask metric
            counter[1] += len(selected)
            counter[2] += accept.sum()
            counter[3] += no_centre.sum()
            counter[4] += small.sum()
            counter[5] += (accept & small).sum()
            counter[6] += (~accept_box).sum()
            counter[7] += (accept_box & ~accept).sum()
        logits_student = composed_logits(student["proto"][image],
            student["mask_coefficient"][image, :, anchors_chosen].T, grid)
        denominator = accept.sum().clamp_min(1)
        image_mask_terms.append((bernoulli_kl(logits_student, logits_teacher.sigmoid()).mean(1) * accept).sum() / denominator)
        cls_terms = bernoulli_kl(student["scores"][image, :, anchors_chosen],
                                 teacher["scores"][image, :, anchors_chosen].sigmoid()).mean(0)
        image_class_terms.append((cls_terms * accept).sum() / denominator)
        active_images.append(accept.any().float())
    counts = dict(zip(counter_names, counter.detach().cpu().tolist()))  # one bounded stats transfer per batch
    if not image_mask_terms:
        return zero, zero, counts
    denominator = torch.stack(active_images).sum().clamp_min(1)
    return torch.stack(image_mask_terms).sum() / denominator, torch.stack(image_class_terms).sum() / denominator, counts


class J5DistillationCriterion:
    """Serializable ordinary loss wrapper: contains NO teacher model or cached predictions."""

    def __init__(self, base, stride, config):
        self.base = base
        self.stride = tuple(float(s) for s in stride)
        self.config = dict(config)
        self.last = {}
        self.cumulative = dict(active_batches=0, annotated=0, considered=0, accepted=0, no_anchor_centre=0,
                               considered_small_box=0, accepted_small_box=0, rejected_box_or_score=0,
                               rejected_mask=0, mask_kl_sum=0.0, class_kl_sum=0.0, ramp_sum=0.0)

    def __call__(self, predictions, batch):
        total, components = self.base(predictions, batch)
        weight = float(batch.get("_j5_ramp", 0))
        teacher = batch.get("_j5_teacher")
        if teacher is None or weight == 0:
            self.last = dict(active=False, ramp=weight)
            return total, components
        student = raw_predictions(predictions)
        mask, cls, counts = instance_distillation(student, teacher, batch, self.stride,
            point_grid=self.config["point_grid"], max_instances=self.config["max_instances"],
            teacher_min_iou=self.config["teacher_min_iou"], teacher_min_score=self.config["teacher_min_score"],
            overlap=self.config["overlap"])
        term = weight * (self.config["kd_mask"] * mask + self.config["kd_cls"] * cls)
        extra = torch.stack((term * 0, term * 0, term * 0, term * 0, term))
        if len(total) != 5:
            raise ValueError("J5 criterion expects the five-component segmentation objective")
        self.last = dict(active=True, ramp=weight, mask=float(mask.detach()), cls=float(cls.detach()), **counts)
        self.cumulative["active_batches"] += 1
        for key, count in counts.items():
            self.cumulative[key] += count
        self.cumulative["mask_kl_sum"] += self.last["mask"]
        self.cumulative["class_kl_sum"] += self.last["cls"]
        self.cumulative["ramp_sum"] += weight
        return total + extra * len(batch["img"]), components + extra.detach()


def ramp_factor(epoch, warmup_epochs=10, ramp_epochs=20):
    """No auxiliary loss for warmup_epochs full epochs; reach one linearly."""
    return min(1.0, max(0.0, (epoch + 1 - warmup_epochs) / ramp_epochs))


def save_distillation_log(trainer):
    """Persist denominators; repeated augmented GT exposures are not unique fruits."""
    criterion = getattr(unwrap_model(trainer.model), "criterion", None)
    if not isinstance(criterion, J5DistillationCriterion):
        return
    current = criterion.cumulative
    previous = getattr(trainer, "_j5_previous_counts", {key: 0 for key in current})
    row = dict(epoch=int(trainer.epoch) + 1, **{key: current[key] - previous[key] for key in current})
    row["acceptance"] = row["accepted"] / max(row["considered"], 1)
    row["small_box_acceptance"] = row["accepted_small_box"] / max(row["considered_small_box"], 1)
    other_denominator = row["considered"] - row["considered_small_box"]
    row["non_small_box_acceptance"] = ((row["accepted"] - row["accepted_small_box"]) / other_denominator
                                        if other_denominator else "")
    for source, target in (("mask_kl_sum", "mean_mask_kl"), ("class_kl_sum", "mean_class_kl"),
                           ("ramp_sum", "mean_ramp")):
        row[target] = row[source] / max(row["active_batches"], 1)
    path = Path(trainer.save_dir) / "j5_distillation.csv"
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    trainer._j5_previous_counts = dict(current)


def make_trainer(teacher_path=None, kd_mask=0.5, kd_cls=0.1, kd_box=0.0, warmup_epochs=10,
                 ramp_epochs=20, point_grid=12, max_instances=32, teacher_min_iou=0.5,
                 teacher_min_score=0.25):
    """EV6/source-balanced view training, with a frozen training-only RGB teacher.

    No teacher reference is attached to the student or its criterion. Validation
    batches contain no auxiliary predictions and retain the original loss/EV6.
    Single-GPU only in this runner; no DDP/compile claim is made.
    """
    numbers = (kd_mask, kd_cls, kd_box, teacher_min_iou, teacher_min_score)
    if not all(math.isfinite(v) for v in numbers) or min(kd_mask, kd_cls) < 0 or kd_box != 0:
        raise ValueError("Finite nonnegative mask/cls weights required; box KD is not implemented")
    if warmup_epochs < 0 or ramp_epochs < 1 or point_grid < 2 or max_instances < 1:
        raise ValueError("Invalid J5 ramp or bounded-sampling configuration")
    if not 0 <= teacher_min_iou <= 1 or not 0 <= teacher_min_score <= 1:
        raise ValueError("Teacher reliability thresholds must be in [0,1]")
    if teacher_path is None:
        if kd_mask or kd_cls:
            raise ValueError("Nonzero KD requires an explicit trained teacher checkpoint")
        return EV6TrainingTrainer
    teacher_path = Path(teacher_path).expanduser().resolve()
    if not teacher_path.is_file():
        raise FileNotFoundError(f"Teacher checkpoint not found: {teacher_path}")

    class J5Trainer(EV6TrainingTrainer):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.add_callback("on_train_epoch_end", save_distillation_log)

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            student = unwrap_model(self.model)
            if not getattr(self, "_j5_installed", False):
                student.criterion = J5DistillationCriterion(student.init_criterion(), student.model[-1].stride,
                    dict(kd_mask=kd_mask, kd_cls=kd_cls, point_grid=point_grid, max_instances=max_instances,
                         teacher_min_iou=teacher_min_iou, teacher_min_score=teacher_min_score,
                         overlap=bool(self.args.overlap_mask)))
                self._j5_installed = True
            weight = ramp_factor(self.epoch, warmup_epochs, ramp_epochs)
            batch["_j5_ramp"] = weight
            if weight == 0:
                return batch
            if not hasattr(self, "j5_teacher"):
                from ultralytics import YOLO

                self.j5_teacher = YOLO(str(teacher_path), task="segment").model.to(self.device).float().eval()
                self.j5_teacher.requires_grad_(False)
                if list(self.j5_teacher.model[-1].stride) != list(student.model[-1].stride):
                    raise ValueError("Teacher/student candidate grids must use identical strides")
            context = torch.cuda.amp.autocast(enabled=False) if batch["img"].is_cuda else nullcontext()
            with torch.no_grad(), context:
                raw = raw_predictions(self.j5_teacher(batch["img"].float()))
                batch["_j5_teacher"] = {key: raw[key].detach() for key in
                                        ("boxes", "scores", "mask_coefficient", "proto")}
            return batch

    J5Trainer.__name__ = "J5DistillationTrainer"
    return J5Trainer
