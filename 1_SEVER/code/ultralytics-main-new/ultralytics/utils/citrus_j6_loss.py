"""J6 bounded auxiliary supervision; ordinary global segmentation loss retained.

Gain supervision is an ONLINE detached, in-sample finite-model Brier difference
from a new instance's zero-logit initialization, not a global-mask comparison.
It is not cross-fitted value-of-information supervision and carries no Bayes/AP
guarantee. All assignment uses training labels; inference receives none of them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.citrus_j6 import expand_roi
from ultralytics.nn.modules.citrus_j6_ops import (bounded_pair_edges, masked_relation_loss, normalized_roi_grid,
                                               roi_logits_on_grid, sparse_observable_pair_features)
from ultralytics.utils.loss import v8SegmentationLoss
from ultralytics.utils.tal import make_anchors


def one_image_raw(raw, image):
    """Keep all sampled evidence in exactly the same image/frame, with batch axis."""
    result = {key: (value[image:image + 1] if isinstance(value, torch.Tensor) else value)
              for key, value in raw.items() if key != "j6"}
    result["j6"] = {key: (value[image:image + 1] if isinstance(value, torch.Tensor) else value)
                    for key, value in raw["j6"].items() if key != "discovery"}
    result["j6"]["discovery"] = {key: value[image:image + 1]
                                 for key, value in raw["j6"]["discovery"].items()}
    return result


def instance_targets(batch, image, ids, overlap=True):
    """Decode existing raster ids; never changes polygon annotations or split."""
    if overlap:
        mask = batch["masks"][image]
        return mask[None] == torch.arange(1, len(ids) + 1, device=mask.device)[:, None, None]
    return batch["masks"][ids] > .5


def visible_points(masks, boxes):
    """Nearest visible pixel to bbox centre; explicit bbox fallback if raster empty."""
    n, height, width = masks.shape
    if not n:
        return boxes.new_empty(0, 2), torch.zeros(0, device=boxes.device, dtype=torch.bool)
    y = (torch.arange(height, device=boxes.device, dtype=boxes.dtype) + .5) / height
    x = (torch.arange(width, device=boxes.device, dtype=boxes.dtype) + .5) / width
    centre = boxes[:, :2]
    distance = (x[None, None] - centre[:, 0, None, None]).square()
    distance = distance + (y[None, :, None] - centre[:, 1, None, None]).square()
    best = distance.masked_fill(~masks, float("inf")).flatten(1).argmin(-1)
    points = torch.stack(((best % width + .5) / width, (best // width + .5) / height), -1)
    nonempty = masks.flatten(1).any(-1)
    return torch.where(nonempty[:, None], points, centre), nonempty


def assign_cells(points, sizes, height, width):
    """Deterministic bounded collision repair, explicitly counting unresolved GT.

    Primary cells represent visible points. A collided GT tries the 8 adjacent
    cells (smallest distance first), without discarding another GT. Regression
    uses an UNBOUNDED bbox-centre offset so an off-centre visible cell or a
    relocated cell can still describe the correct box. This assignment changes
    neither the existence-point target nor the GT bbox. It is a one-slot scout,
    not a dense crowd detector.
    """
    occupied, cells, accepted = set(), [], []
    order = sizes.prod(-1).argsort(stable=True).detach().cpu().tolist()
    values = (points.detach() * points.new_tensor([width, height])).cpu().tolist()
    for index in order:
        px, py = values[index]
        cx, cy = min(width - 1, max(0, int(px))), min(height - 1, max(0, int(py)))
        choices = [(x, y) for y in range(max(0, cy - 1), min(height, cy + 2))
                   for x in range(max(0, cx - 1), min(width, cx + 2))]
        choices.sort(key=lambda xy: ((xy[0] + .5 - px) ** 2 + (xy[1] + .5 - py) ** 2, xy[1], xy[0]))
        free = next((xy for xy in choices if xy not in occupied), None)
        if free is not None:
            occupied.add(free)
            cells.append((free[1] * width + free[0]))
            accepted.append(index)
    device = points.device
    return torch.tensor(accepted, device=device, dtype=torch.long), torch.tensor(cells, device=device, dtype=torch.long)


def _box_iou(a, b):
    intersection = (torch.minimum(a[:, None, 2:], b[None, :, 2:]) -
                    torch.maximum(a[:, None, :2], b[None, :, :2])).clamp_min(0).prod(-1)
    areas = (a[:, 2:] - a[:, :2]).prod(-1)[:, None] + (b[:, 2:] - b[:, :2]).prod(-1)[None]
    return intersection / (areas - intersection).clamp_min(1e-7)


class J6Loss:
    """Five original components plus scout/local/identity/gain, identically named in all arms."""

    def __init__(self, model):
        self.base = v8SegmentationLoss(model)
        self.head = model.model[-1]
        self.loss_names = self.head.loss_names
        self.overlap = bool(model.args.overlap_mask)
        self.last = {}
        self.pending = dict(local_forwards=0, gain_forwards=0, identity_positive=0, identity_negative=0)
        self.cumulative = {}

    def commit_updates(self, success=True):
        """Commit readiness only after a real optimizer update, discard AMP-skipped work."""
        if success and self.head.enabled:
            self.head.auxiliary_updates.add_(self.pending["local_forwards"])
            self.head.gain_updates.add_(self.pending["gain_forwards"])
            self.head.identity_positive_updates.add_(self.pending["identity_positive"])
            self.head.identity_negative_updates.add_(self.pending["identity_negative"])
            self.head.relation.mark_trained(int(self.head.identity_positive_updates) >= 20 and
                                             int(self.head.identity_negative_updates) >= 20)
        self.pending = dict(local_forwards=0, gain_forwards=0, identity_positive=0, identity_negative=0)

    def __call__(self, predictions, batch):
        base, detached = self.base(predictions, batch)
        raw = predictions[1] if isinstance(predictions, tuple) else predictions
        zero = base.sum() * 0
        if not self.head.enabled or not raw.get("j6", {}).get("training_mode", False):
            extra = torch.stack((zero, zero, zero, zero))
            return torch.cat((base, extra)), torch.cat((detached, extra.detach()))
        if not raw["j6"]["native_high"]:
            raise ValueError("Enabled J6 training requires original-pixel paired high_img; use J6Trainer")
        scout_terms, local_terms, identity_terms, gain_terms = [], [], [], []
        counts = dict(annotated=0, scout_assigned=0, scout_collision_unresolved=0,
                      empty_gt_raster=0, empty_sampled_gt=0, local_positive=0, local_negative=0, known_identity_pairs=0,
                      identity_positive=0, identity_negative=0, valid_local_images=0, valid_gain_images=0,
                      unassigned_source1=0)
        anchors, strides = make_anchors(raw["feats"], self.head.stride, .5)
        image_h, image_w = batch["img"].shape[-2:]
        anchor_centres = anchors * strides / anchors.new_tensor([image_w, image_h])
        for image in range(len(batch["img"])):
            ids = (batch["batch_idx"].reshape(-1).long() == image).nonzero(as_tuple=False).flatten()
            target_masks = instance_targets(batch, image, ids, self.overlap)
            xywh = batch["bboxes"][ids].float()
            boxes_gt = torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), -1).clamp(0, 1)
            points, nonempty = visible_points(target_masks, xywh)
            heat = raw["j6"]["discovery"]["existence"][image, 0]
            hh, ww = heat.shape
            chosen, cells = assign_cells(points, xywh[:, 2:], hh, ww)
            # Keep the true supervised existence cell for later gain regression.
            # Unresolved collisions have no cell and must not invent gain labels.
            gt_cells = torch.full((len(xywh),), -1, device=heat.device, dtype=torch.long)
            gt_cells[chosen] = cells
            positive = torch.zeros_like(heat).flatten()
            positive[cells] = 1
            pos = positive.reshape_as(heat)
            probability = heat.sigmoid()
            focal = -pos * (1 - probability).square() * F.logsigmoid(heat)
            focal -= (1 - pos) * probability.square() * F.logsigmoid(-heat)
            scout = focal.sum() / max(1, len(chosen))
            if len(chosen):
                offset = raw["j6"]["discovery"]["offset"][image].flatten(1)[:, cells].T
                predicted_size = raw["j6"]["discovery"]["log_box_size"][image].flatten(1)[:, cells].T
                cell_xy = torch.stack((cells % ww, cells // ww), -1)
                # The peak is assigned from a visible mask point, but the box
                # decoder interprets offsets as bbox CENTRES. These are different
                # for concave/occluded masks; using points here shifts perfect boxes.
                offsets_gt = xywh[chosen, :2] * xywh.new_tensor([ww, hh]) - cell_xy
                scout = scout + F.smooth_l1_loss(offset, offsets_gt)
                scout = scout + F.smooth_l1_loss(
                    predicted_size, xywh[chosen, 2:].clamp_min(self.head.scout_min_box_size).log())
            scout_terms.append(scout)
            counts["annotated"] += len(ids)
            counts["scout_assigned"] += len(chosen)
            counts["scout_collision_unresolved"] += len(ids) - len(chosen)
            counts["empty_gt_raster"] += int((~nonempty).sum())

            # Two candidates per selected GT provide real duplicate identity labels.
            # All global GT remain supervised by the ordinary detector loss.
            selected = xywh[:, 2:].prod(-1).argsort(stable=True)[:max(1, (self.head.max_rois - 4) // 2)]
            gt_local = selected.repeat_interleave(2)
            sources = torch.arange(len(gt_local), device=heat.device) % 2
            # Different prediction domains prevent identity learning from seeing
            # only exact-IoU=1 duplicate positives. Jitter is TRAINING metadata.
            margins = (1.2 + .4 * sources.to(xywh.dtype))[:, None]
            domains = expand_roi(boxes_gt[gt_local], scale=margins)
            if len(domains):
                shift = (torch.rand_like(domains[:, :2]) - .5) * .1 * (domains[:, 2:] - domains[:, :2])
                domains = torch.cat(((domains[:, :2] + shift).clamp(0, 1 - 1e-5),
                                     (domains[:, 2:] + shift).clamp(1e-5, 1)), -1)
            negative_boxes, negative_scores, _ = self.head.discover(raw, image, .0)
            if len(negative_boxes):
                negative_valid = ((_box_iou(negative_boxes, boxes_gt).amax(-1) < .05)
                                  if len(boxes_gt) else torch.ones(len(negative_boxes), device=heat.device, dtype=torch.bool))
                negative_boxes = negative_boxes[negative_valid][:4]
                negative_scores = negative_scores[negative_valid][:4]
            npositive, nnegative = len(domains), len(negative_boxes)
            domains = torch.cat((domains, expand_roi(negative_boxes)), 0).detach()
            if not len(domains):
                continue
            sources = torch.cat((sources, torch.ones(nnegative, device=heat.device, dtype=torch.long)))
            has_own = sources == 0
            coefficients = raw["proto"].new_zeros(len(domains), self.head.nm)
            scores = heat.new_zeros(len(domains))
            if npositive:
                # Match deployment inputs: global-origin ROIs own global mask
                # coefficients/scores; new-instance ROIs start with zero mask
                # coefficients and their assigned scout-cell existence score.
                global_slots = (sources[:npositive] == 0).nonzero(as_tuple=False).flatten()
                nearest = (anchor_centres[None] - xywh[gt_local[global_slots], None, :2]).square().sum(-1).argmin(1)
                coefficients[global_slots] = raw["mask_coefficient"][image, :, nearest].T
                scores[global_slots] = raw["scores"][image, 0, nearest].sigmoid()
                new_slots = (sources[:npositive] == 1).nonzero(as_tuple=False).flatten()
                new_cells = gt_cells[gt_local[new_slots]]
                assigned_new = new_cells >= 0
                scores[new_slots[assigned_new]] = probability.flatten()[new_cells[assigned_new]]
                counts["unassigned_source1"] += int((~assigned_new).sum())
            scores[npositive:] = negative_scores
            part = one_image_raw(raw, image)
            decoded = self.head.decode_rois(part, domains[None], coefficients[None], has_own[None],
                                            scores[None], sources[None])
            grid = normalized_roi_grid(domains[None], self.head.roi_side, self.head.max_rois)[0]
            local_target = torch.zeros_like(decoded["logits"])
            if npositive:
                sampled_ids = F.grid_sample(target_masks[gt_local, None].float(),
                                            2 * grid[:npositive] - 1, mode="nearest", align_corners=False)
                local_target[0, :npositive] = sampled_ids
            # A known bbox with a vanished raster is NOT a negative mask label.
            local_known = torch.ones(len(domains), device=heat.device, dtype=torch.bool)
            if npositive:
                local_known[:npositive] = nonempty[gt_local] & local_target[0, :npositive].flatten(1).bool().any(-1)
                counts["empty_sampled_gt"] += int((~local_known[:npositive]).sum())
                # A source-one GT without a collision-assigned discovery cell
                # has no deployment-aligned scout state; treat it as unknown.
                local_known[:npositive] &= has_own[:npositive] | (gt_cells[gt_local] >= 0)
            predicted = decoded["logits"]
            error = F.binary_cross_entropy_with_logits(predicted, local_target, reduction="none").mean((2, 3, 4))[0]
            weights = torch.ones_like(error)
            if self.head.gt_balanced and npositive:
                area = xywh[gt_local, 2:].prod(-1)
                weights[:npositive] = (.001 / area.clamp_min(1e-7)).sqrt().clamp(1, 2)
            weights *= local_known
            counts["valid_local_images"] += int(local_known.any())
            mask_loss = (error * weights).sum() / weights.sum().clamp_min(1)
            with torch.no_grad():
                predicted_binary, gt_binary = predicted > 0, local_target > .5
                quality = ((predicted_binary & gt_binary).sum((2, 3, 4)) /
                           (predicted_binary | gt_binary).sum((2, 3, 4)).clamp_min(1))
            quality_errors = F.binary_cross_entropy_with_logits(decoded["quality_logits"], quality, reduction="none")[0]
            quality_loss = (quality_errors * local_known).sum() / local_known.sum().clamp_min(1)
            local_terms.append(mask_loss + .25 * quality_loss)
            counts["local_positive"] += npositive
            counts["local_negative"] += nnegative

            if self.head.selection and len(domains) > 1:
                axis = (torch.arange(128, device=heat.device, dtype=domains.dtype) + .5) / 128
                yy, xx = torch.meshgrid(axis, axis, indexing="ij")
                world = torch.stack((xx.flatten(), yy.flatten()), -1)[None]
                field, support = roi_logits_on_grid(predicted.detach(), domains[None], world,
                                                    max_rois=self.head.max_rois)
                edges = bounded_pair_edges(domains[None], max_edges=128, max_candidates=self.head.max_rois)["edges"]
                evidence = sparse_observable_pair_features(field > 0, support, edges,
                    queries=decoded["query"], boxes=domains[None], scores=scores[None],
                    max_edges=128, max_candidates=self.head.max_rois)
                relation = self.head.relation(evidence)
                identity = torch.cat((gt_local, torch.full((nnegative,), -1, device=heat.device, dtype=torch.long)))
                left, right = edges.clamp_min(0).unbind(-1)
                pair_known = ((edges[..., 0] >= 0) & (identity[left] >= 0) & (identity[right] >= 0)
                              & local_known[left] & local_known[right])
                same = identity[left] == identity[right]
                identity_terms.append(masked_relation_loss(relation, same, pair_known))
                valid_pair = pair_known & relation["known"]
                counts["known_identity_pairs"] += int(valid_pair.sum())
                counts["identity_positive"] += int((valid_pair & same).sum())
                counts["identity_negative"] += int((valid_pair & ~same).sum())
            gain_cells = torch.full((len(domains),), -1, device=heat.device, dtype=torch.long)
            gain_cells[:npositive] = gt_cells[gt_local]
            gain_known = local_known & (sources == 1) & (gain_cells >= 0)
            if self.head.gain_route and gain_known.any():
                # Fixed aligned pixel event from new-instance zero logits, NOT
                # a global-mask gain, VOI target or unmatched-set subtraction.
                coarse = decoded["reference_logits"].detach().sigmoid()
                fine = predicted.detach().sigmoid()
                risk_difference = ((local_target - coarse).square() - (local_target - fine).square()).mean((2, 3, 4))[0]
                # Inference reads gain at the existence-map cell, NOT the ROI
                # centre (which differs after occlusion and domain jitter).
                # Only assigned positive GT provide a well-defined action cell;
                # background proposals and unresolved collisions are unknown.
                gain = part["j6"]["gain"].flatten()[gain_cells[gain_known]].tanh()
                gain_errors = F.smooth_l1_loss(gain, risk_difference[gain_known], reduction="none")
                gain_terms.append(gain_errors.mean())
                counts["valid_gain_images"] += 1
        self.last = counts
        if counts["valid_local_images"]:
            self.pending["local_forwards"] += 1
        self.pending["identity_positive"] += counts["identity_positive"]
        self.pending["identity_negative"] += counts["identity_negative"]
        if counts["valid_gain_images"]:
            self.pending["gain_forwards"] += 1
        mean = lambda terms: torch.stack(terms).mean() if terms else zero
        extra = torch.stack((.25 * mean(scout_terms), .5 * mean(local_terms),
                             .1 * mean(identity_terms), .1 * mean(gain_terms)))
        values = dict(counts, active_batches=1, scout_loss_sum=float(extra[0].detach()),
                      local_loss_sum=float(extra[1].detach()), identity_loss_sum=float(extra[2].detach()),
                      gain_loss_sum=float(extra[3].detach()))
        for key, value in values.items():
            self.cumulative[key] = self.cumulative.get(key, 0) + value
        return torch.cat((base, extra * len(batch["img"]))), torch.cat((detached, extra.detach()))
