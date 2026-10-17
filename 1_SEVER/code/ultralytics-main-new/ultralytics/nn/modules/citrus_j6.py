"""J6: bounded high-detail computation with an ordinary global YOLO segmenter.

The fourth YAML input is the untouched GLOBAL RGB tensor, not a fourth detection
scale. Original-pixel high views are supplied by the paired dataset/predictor.
Plain tensor forward permits a resize-only fallback for construction/profiling;
the J6 evaluation adapter rejects that fallback for an enabled model.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .citrus_j6_ops import (IndependentDiscovery, InstanceResidual, ObservableRelation, SharedRGBPrefix,
                            packed_roi_sample)
from .head import Segment


def expand_roi(boxes, scale=1.4, minimum=0.025):
    """Return a positive normalized local prediction domain, distinct from a box."""
    centre = (boxes[..., :2] + boxes[..., 2:]) / 2
    size = ((boxes[..., 2:] - boxes[..., :2]) * scale).clamp(minimum, 1.0)
    lo = (centre - size / 2).clamp(0, 1 - 1e-5)
    hi = torch.maximum((centre + size / 2).clamp(0, 1), lo + 1e-5)
    return torch.cat((lo, hi), -1)


class SegmentCitrusJ6(Segment):
    """Single-class global detector plus small shared high-detail instance head.

    Bounding boxes, DFL and global prototype supervision remain unchanged.
    Auxiliary masks are explicit ROI logits, never forced back into the original
    prototype span. The inference adapter projects them before EV6 matching.
    """

    loss_names = ("box_loss", "seg_loss", "cls_loss", "dfl_loss", "sem_loss",
                  "scout_loss", "local_loss", "identity_loss", "gain_loss")
    # Output tensor shapes remain unchanged, but old visible-point-offset
    # checkpoints are NOT semantically compatible with this regression target.
    scout_geometry_protocol = "visible_cell_bbox_center_v2"
    scout_min_box_size = 1e-5

    def __init__(self, nc=1, nm=32, npr=256, enabled=True, neighbor=True, selection=False,
                 shuffle=False, use_d2=True, gt_balanced=False, gain_route=False,
                 detail_channels=8, scout_channels=16, roi_side=16, max_rois=32,
                 reg_max=16, end2end=False, ch=()):
        if len(ch) != 4 or ch[-1] != 3 or nc != 1 or end2end:
            raise ValueError("J6 requires three detection scales plus raw RGB, one class, one-to-many head")
        if not 4 <= roi_side <= 32 or not 4 <= max_rois <= 64:
            raise ValueError("J6 requires ROI side 4..32 and ROI cap 4..64")
        super().__init__(nc, nm, npr, reg_max, end2end, tuple(ch[:3]))
        self.enabled, self.neighbor, self.selection = bool(enabled), bool(neighbor), bool(selection)
        self.shuffle, self.use_d2 = bool(shuffle), bool(use_d2)
        self.gt_balanced, self.gain_route = bool(gt_balanced), bool(gain_route)
        self.roi_side, self.max_rois = int(roi_side), int(max_rois)
        self.max_new = self.max_rois // 2
        self.max_relation_candidates, self.max_relation_edges = 128, 256
        # Stored on the pickled head so old same-shaped J6 checkpoints can be
        # rejected by the runner instead of silently reinterpreting offsets.
        self.scout_geometry = "bbox_center_offset_v2"
        self.scout_candidate_policy = "cell_topk_no_peak_nms_v2"
        self.scout_source1_policy = "zero_coeff_cell_heat_v2"
        self._pending_high = None
        self.register_buffer("auxiliary_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("gain_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("identity_positive_updates", torch.tensor(0, dtype=torch.long))
        self.register_buffer("identity_negative_updates", torch.tensor(0, dtype=torch.long))
        if self.enabled:
            self.high_prefix = SharedRGBPrefix(detail_channels, scout_channels)
            self.discovery = IndependentDiscovery(scout_channels)
            self.semantic_projection = nn.Conv2d(ch[0], 8, 1)
            # Detail/semantic pooling, normalized ROI geometry, score, source one-hot.
            self.query_projection = nn.Sequential(nn.Linear(detail_channels + 8 + 7, 32), nn.SiLU())
            self.roi_decoder = InstanceResidual(detail_channels, 8, 32, 16, max_rois=max_rois)
            self.quality_head = nn.Linear(32, 1)
            self.gain_head = nn.Conv2d(scout_channels, 1, 1)
            self.relation = ObservableRelation(appearance_channels=32, include_geometry=True,
                                               include_scores=True)
            with torch.no_grad():
                self.discovery.output.bias[0] = -4.6
                self.discovery.output.bias[1:3] = .5
                self.discovery.output.bias[3:] = -3.0
                self.gain_head.weight.zero_()
                self.gain_head.bias.zero_()
                self.quality_head.weight.zero_()
                self.quality_head.bias.zero_()

    def bind_high_view(self, high):
        """Bind exactly one paired view; consumed immediately by the next forward."""
        if high is not None and (high.ndim != 4 or high.shape[1] != 3):
            raise ValueError("J6 high view must be BCHW RGB")
        self._pending_high = high

    def forward(self, x):
        """Return standard Segment outputs with explicit auxiliary evidence metadata."""
        if self.export:
            raise NotImplementedError("J6 requires its paired-view Python decoder; single-input export is not implemented")
        global_rgb, features = x[-1], x[:3]
        high, self._pending_high = self._pending_high, None
        output = super().forward(features)
        raw = output if isinstance(output, dict) else output[1]
        if not self.enabled:
            return output
        native = high is not None
        expected = (global_rgb.shape[0], 3, global_rgb.shape[-2] * 2, global_rgb.shape[-1] * 2)
        if high is None:
            high = F.interpolate(global_rgb, scale_factor=2, mode="bilinear", align_corners=False)
        if high.shape != expected or high.device != global_rgb.device:
            raise ValueError(f"Paired high/global geometry or device mismatch; expected {expected}, got {high.shape}")
        detail, scout = self.high_prefix(high.to(global_rgb.dtype))
        raw["j6"] = dict(detail=detail, scout=scout, semantic=self.semantic_projection(features[0]),
                         discovery=self.discovery(scout), gain=self.gain_head(scout),
                         input_shape=tuple(global_rgb.shape[-2:]), native_high=native,
                         training_mode=self.training)
        return output

    def decode_rois(self, raw, boxes, coefficients, has_own, scores, sources):
        """Decode BN ROIs in one normalized global frame with predicted rival state.

        boxes are the explicit local prediction domains, not original image-view
        supports. Neighbor logits are sampled in the CURRENT ROI's coordinates.
        No GT mask or GT identity is read by this method.
        """
        aux = raw["j6"]
        count = boxes.shape[1]
        detail = packed_roi_sample(aux["detail"], boxes, self.roi_side, self.max_rois)
        if not self.use_d2:
            detail = detail * 0
        semantic = packed_roi_sample(aux["semantic"], boxes, self.roi_side, self.max_rois)
        basis = packed_roi_sample(raw["proto"], boxes, self.roi_side, self.max_rois)
        reference = torch.einsum("bnchw,bnc->bnhw", basis, coefficients)[:, :, None]
        own = reference * has_own[..., None, None, None]
        rival = torch.zeros_like(own)
        rival_known = torch.zeros_like(own, dtype=torch.bool)
        if self.neighbor and count > 1:
            centres = (boxes[..., :2] + boxes[..., 2:]) / 2
            distance = (centres[:, :, None] - centres[:, None]).square().sum(-1)
            intersection = (torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:]) -
                            torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])).clamp_min(0).prod(-1)
            area = (boxes[..., 2:] - boxes[..., :2]).prod(-1)
            overlap = intersection / (area[:, :, None] + area[:, None] - intersection).clamp_min(1e-7)
            valid = has_own[:, None].expand_as(distance) & (overlap < .8)
            valid &= ~torch.eye(count, device=boxes.device, dtype=torch.bool)[None]
            k = min(4, count - 1)
            ids = distance.masked_fill(~valid, float("inf")).argsort(stable=True)[..., :k]
            coefficient_ids = (ids + max(1, count // 2)) % count if self.shuffle else ids
            b = torch.arange(len(boxes), device=boxes.device)[:, None, None]
            neighbor_coefficients = coefficients[b, coefficient_ids]
            neighbor_logits = torch.einsum("bnchw,bnkc->bnkhw", basis, neighbor_coefficients)
            allowed = valid.gather(2, ids)
            # A coefficient has no valid prediction outside its instance domain.
            # Keep domains/availability fixed in the wrong-neighbor control.
            axis = (torch.arange(self.roi_side, device=boxes.device, dtype=boxes.dtype) + .5) / self.roi_side
            yy, xx = torch.meshgrid(axis, axis, indexing="ij")
            unit = torch.stack((xx, yy), -1)
            points = boxes[..., None, None, :2] + unit * (boxes[..., None, None, 2:] - boxes[..., None, None, :2])
            neighbor_boxes = boxes[b, ids]
            support = ((points[:, :, None] >= neighbor_boxes[..., None, None, :2]) &
                       (points[:, :, None] < neighbor_boxes[..., None, None, 2:])).all(-1)
            support &= allowed[..., None, None]
            neighbor_logits = neighbor_logits.masked_fill(~support, -20.)
            rival = neighbor_logits.amax(2, keepdim=True)
            rival_known = support.any(2, keepdim=True)
            rival = torch.where(rival_known, rival, torch.zeros_like(rival))
        pooled = torch.cat((detail.mean((-2, -1)), semantic.mean((-2, -1)), boxes,
                            scores[..., None], F.one_hot(sources.long(), 2).to(detail.dtype)), -1)
        query = self.query_projection(pooled)
        logits = self.roi_decoder(detail, semantic, query, own, rival)
        return dict(logits=logits, own_logits=own, reference_logits=reference, rival_logits=rival,
                    rival_known=rival_known, query=query,
                    quality_logits=self.quality_head(query).squeeze(-1))

    def discover(self, raw, image, threshold=.001):
        """Decode boxes from visible-cell peaks and unbounded BBOX-CENTRE offsets.

        The existence peak may lie on an off-centre visible fragment or a cell
        relocated by collision repair. Its regression offset nevertheless points
        to the GT bounding-box centre, never implicitly to that visible point.
        Sizes below one scout cell remain representable; this is not an integer
        raster box. No global proposals or GT are read during this decoding.
        """
        values = raw["j6"]["discovery"]
        heat = values["existence"][image, 0].sigmoid()
        rank = heat
        if self.gain_route and int(self.gain_updates) > 0:
            rank = heat + .25 * raw["j6"]["gain"][image, 0].tanh()
        # Adjacent positive cells may represent different fruits, including a GT
        # relocated during collision repair. Local peak NMS cannot distinguish
        # that case from duplicate peaks; retain thresholded cells within budget.
        # This can increase duplicates, which is an empirical trade-off, not a
        # guarantee of improved recall or precision.
        rank = rank.masked_fill(heat < threshold, -float("inf"))
        # Stable ties; flatten index is the documented deterministic key.
        ids = rank.flatten().argsort(descending=True, stable=True)[:self.max_new]
        keep = torch.isfinite(rank.flatten()[ids])
        ids = ids[keep]
        height, width = heat.shape
        # Unbounded centre offsets handle both off-centre visible fragments and
        # collision relocation without forcing an existence peak into a mask hole.
        offsets = values["offset"][image].flatten(1)[:, ids].T
        centre = torch.stack((ids % width, ids // width), -1).to(offsets.dtype) + offsets
        centre = (centre / centre.new_tensor([width, height])).clamp(0, 1)
        size = values["log_box_size"][image].flatten(1)[:, ids].T.clamp(
            math.log(self.scout_min_box_size), 0).exp()
        boxes = torch.cat(((centre - size / 2).clamp(0, 1), (centre + size / 2).clamp(0, 1)), -1)
        return boxes, heat.flatten()[ids], ids
