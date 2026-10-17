"""Bounded J6 ROI decoder, followed by the UNCHANGED EV6 metric interface.

View coverage and finite ROI-logit validity are different objects. This shared
prefix implementation has one full high image, not multiple independent crops.
Its common comparison support is the *effective prediction domain*, never a
claim that a whole fruit outside it was observed or that a box is an amodal mask.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from ultralytics.nn.modules.citrus_j6 import expand_roi
from ultralytics.nn.modules.citrus_j6_ops import (bounded_pair_edges, bounded_set_selection, project_roi_logits,
                                               sparse_observable_pair_features, sparse_relation_matrix)
from ultralytics.utils import nms, ops
from ultralytics.utils.citrus_j6_loss import one_image_raw


J6_SCOUT_GEOMETRY = "bbox_center_offset_v2"
J6_CANDIDATE_POLICY = "cell_topk_no_peak_nms_v2"
J6_SOURCE1_POLICY = "zero_coeff_cell_heat_v2"


def assert_j6_scout_geometry(head, operation="J6 execution"):
    """Reject legacy enabled heads whose equal-shaped scout tensors have incompatible learned semantics."""
    if not bool(getattr(head, "enabled", False)):
        return
    expected = {
        "scout_geometry": J6_SCOUT_GEOMETRY,
        "scout_candidate_policy": J6_CANDIDATE_POLICY,
        "scout_source1_policy": J6_SOURCE1_POLICY,
    }
    mismatches = {name: {"received": getattr(head, name, None), "expected": version}
                  for name, version in expected.items() if getattr(head, name, None) != version}
    if mismatches:
        raise RuntimeError(
            f"{operation}: incompatible J6 scout semantics: {mismatches}. "
            "Legacy enabled J6 weights may use visible-point-centred box offsets, peak-suppressed cells, "
            "or differently supervised discovery-source candidates; their tensor shapes match but their "
            "learned coordinate meaning and candidate/source policies do not. "
            "Do not add markers to old weights or reuse "
            "them for resume/validation/prediction. Start a NEW PROJECT from a current J6 YAML and retrain. "
            "Legacy J90 global-only heads with enabled=False do not use this scout and remain evaluable."
        )


def _iou(left, right):
    if not len(left) or not len(right):
        return left.new_zeros(len(left), len(right))
    intersection = (torch.minimum(left[:, None, 2:], right[None, :, 2:]) -
                    torch.maximum(left[:, None, :2], right[None, :, :2])).clamp_min(0).prod(-1)
    areas = (left[:, 2:] - left[:, :2]).prod(-1)[:, None] + (right[:, 2:] - right[:, :2]).prod(-1)[None]
    return intersection / (areas - intersection).clamp_min(1e-7)


def _base_parts(preds):
    if not isinstance(preds, (tuple, list)) or len(preds) != 2 or not isinstance(preds[1], dict):
        raise ValueError("J6 requires native PyTorch Segment predictions, not exported arrays")
    decoded, proto = preds[0]
    return decoded, proto, preds[1]


def _crop_like_global(binary, boxes, original_count):
    """Keep the complete global decoder's CPU rounding policy for a small subset."""
    if binary.device.type == "cpu" and original_count < 50:
        return ops.crop_mask(binary, boxes)
    height, width = binary.shape[-2:]
    x = torch.arange(width, device=binary.device, dtype=boxes.dtype)[None, None]
    y = torch.arange(height, device=binary.device, dtype=boxes.dtype)[None, :, None]
    allowed = ((x >= boxes[:, 0, None, None]) & (x < boxes[:, 2, None, None]) &
               (y >= boxes[:, 1, None, None]) & (y < boxes[:, 3, None, None]))
    return binary & allowed


@torch.no_grad()
def postprocess_j6(head, preds, conf=.001, iou=.7, max_det=300, mask_ratio=2):
    """Return EV6 dictionaries with global-input pixel boxes and input/2 masks.

    Ordinary global NMS runs once. Existing unrefined detections are preserved;
    at most max_rois local candidates are refined/compared. There is no second
    box-NMS pass that could undo identity-aware retention of touching fruit.
    """
    assert_j6_scout_geometry(head, "J6 postprocess")
    if mask_ratio != 2 or not 0 <= conf <= 1 or not 0 <= iou <= 1 or max_det < 1:
        raise ValueError("J6 fixes EV6 mask_ratio=2 and requires valid thresholds")
    decoded, proto, raw = _base_parts(preds)
    shape = (int(raw["feats"][0].shape[-2] * head.stride[0]),
             int(raw["feats"][0].shape[-1] * head.stride[0]))
    evaluation_shape = tuple(value // 2 for value in shape)
    proto = F.interpolate(proto.float(), evaluation_shape, mode="bilinear", align_corners=False)
    base = nms.non_max_suppression(decoded, conf, iou, nc=1, multi_label=True, max_det=max_det, end2end=False)
    if head.enabled and not raw.get("j6", {}).get("native_high", False):
        raise ValueError("Enabled J6 evaluation requires an original-pixel high view, not resized global RGB")
    results = []
    stats = dict(global_candidates=0, new_candidates=0, refined=0, relation_edges=0,
                 known_relations=0, removed_by_relation=0)
    for image, detection in enumerate(base):
        boxes, scores, classes, coefficients = detection[:, :4], detection[:, 4], detection[:, 5], detection[:, 6:]
        masks = (ops.process_mask(proto[image], coefficients, boxes, shape)
                 if len(detection) else torch.zeros(0, *evaluation_shape, device=proto.device, dtype=torch.uint8))
        stats["global_candidates"] += len(boxes)
        # No random untrained ROI/quality/relation head may perturb the control.
        if not head.enabled or int(head.auxiliary_updates) == 0:
            results.append(dict(bboxes=boxes, conf=scores, cls=classes, masks=masks))
            continue
        part = one_image_raw(raw, image)
        new_boxes, new_scores, _ = head.discover(raw, image, conf)
        scale = boxes.new_tensor([shape[1], shape[0], shape[1], shape[0]])
        global_normalized = (boxes / scale).clamp(0, 1)
        # Prefer global instances competing spatially with discoveries. Remaining
        # slots use confidence. This is explicit deterministic routing, not GT.
        priority = scores
        if len(new_boxes) and len(boxes):
            priority = scores + (_iou(global_normalized, new_boxes).amax(-1) > .05).to(scores.dtype)
        slots = min(len(boxes), head.max_rois - len(new_boxes))
        selected_global = priority.argsort(descending=True, stable=True)[:slots]
        local_boxes = torch.cat((global_normalized[selected_global], new_boxes), 0)
        if not len(local_boxes):
            results.append(dict(bboxes=boxes, conf=scores, cls=classes, masks=masks))
            continue
        domains = expand_roi(local_boxes)
        global_count, new_count = len(selected_global), len(new_boxes)
        local_coefficients = coefficients.new_zeros(len(domains), head.nm)
        local_coefficients[:global_count] = coefficients[selected_global]
        has_own = torch.arange(len(domains), device=boxes.device) < global_count
        sources = (~has_own).long()
        local_scores = torch.cat((scores[selected_global], new_scores))
        local = head.decode_rois(part, domains[None], local_coefficients[None], has_own[None],
                                 local_scores[None], sources[None])
        delta = local["logits"] - local["own_logits"]
        delta_field, support = project_roi_logits(delta, domains[None], evaluation_shape,
                                                   max_rois=head.max_rois, outside_value=0.)
        delta_field, support = delta_field[0, :, 0], support[0, :, 0]
        # Project ONLY residuals for global instances. Re-rendering coarse logits
        # through a 16x16 ROI would destroy exact zero-update identity.
        if global_count:
            base_logits = (local_coefficients[:global_count].float() @ proto[image].flatten(1)).reshape(
                global_count, *evaluation_shape)
            refined_logits = base_logits + delta_field[:global_count]
            ratios = scale.new_tensor([evaluation_shape[1], evaluation_shape[0],
                                      evaluation_shape[1], evaluation_shape[0]])
            refined_masks = _crop_like_global(refined_logits > 0, global_normalized[selected_global] * ratios,
                                              len(detection))
            masks[selected_global] = refined_masks.to(masks.dtype)
        if new_count:
            new_masks = (delta_field[global_count:] > 0) & support[global_count:]
            boxes = torch.cat((boxes, new_boxes * scale))
            quality = local["quality_logits"][0].sigmoid()
            scores = torch.cat((scores, new_scores * quality[global_count:]))
            classes = torch.cat((classes, classes.new_zeros(new_count)))
            masks = torch.cat((masks, new_masks.to(masks.dtype)))
        local_indices = torch.cat((selected_global, torch.arange(len(detection), len(boxes), device=boxes.device)))
        if head.selection:
            # Mask quality is a learned reliability cue, not a new AP metric.
            scores[selected_global] *= local["quality_logits"][0, :global_count].sigmoid()
        keep = (scores >= conf) & masks.bool().flatten(1).any(-1)
        if head.selection and len(domains) > 1 and bool(head.relation.relation_trained):
            relation_masks = F.interpolate(masks[local_indices, None].float(), (128, 128), mode="nearest")[:, 0] > .5
            relation_support = F.interpolate(support[:, None].float(), (128, 128), mode="nearest")[:, 0] > .5
            edges = bounded_pair_edges(domains[None], max_edges=128, max_candidates=head.max_rois)
            evidence = sparse_observable_pair_features(relation_masks.flatten(1)[None],
                relation_support.flatten(1)[None], edges["edges"], queries=local["query"], boxes=domains[None],
                scores=scores[local_indices][None], max_edges=128, max_candidates=head.max_rois)
            relation = head.relation(evidence)
            pair_logits, known = sparse_relation_matrix(relation, len(domains), max_candidates=head.max_rois)
            stats["relation_edges"] += int(edges["counts"].sum())
            stats["known_relations"] += int(known.sum()) // 2
            # Confidence-scaled soft duplicate cost avoids a fixed large penalty
            # overwhelming every low-score small fruit. lambda and all caps are
            # fixed implementation constants, not tuned on this test set.
            confidence = scores[local_indices].float()
            duplicate = pair_logits[0].sigmoid() * torch.minimum(confidence[:, None], confidence[None])
            weighted_logits = torch.logit(duplicate.clamp(1e-7, 1 - 1e-7))
            selection = bounded_set_selection(confidence - conf, weighted_logits, known[0],
                budget=len(domains), penalty=1.25, relation_trained=True,
                max_steps=head.max_rois + 4, improvement_rounds=2, max_candidates=head.max_rois)
            keep[local_indices] &= selection["selected"]
            stats["removed_by_relation"] += int((~selection["selected"]).sum())
        order = scores.masked_fill(~keep, -1).argsort(descending=True, stable=True)
        order = order[keep[order]][:max_det]
        results.append(dict(bboxes=boxes[order], conf=scores[order], cls=classes[order], masks=masks[order]))
        stats["new_candidates"] += new_count
        stats["refined"] += len(domains)
    head.last_inference = stats
    return results
