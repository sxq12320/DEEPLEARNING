# Ultralytics AGPL-3.0 License - https://ultralytics.com/license
"""Bounded J6 evidence, ROI and candidate-set operators.

All boxes use the caller's SAME normalized xyxy coordinate frame. No raw-image
letterbox transform, detection assignment or hidden image enlargement occurs.
Relation predictions are usable for training immediately, but cannot suppress
candidates until the caller explicitly marks the relation head trained. The
bounded set solver optimizes its stated surrogate, not AP, and has no ordinary
monotone-submodular greedy guarantee.
"""

from itertools import combinations
import math

import torch
from torch import nn
from torch.nn import functional as F

__all__ = (
    "SharedRGBPrefix", "IndependentDiscovery", "InstanceResidual", "ObservableRelation",
    "normalized_roi_grid", "packed_roi_sample", "roi_logits_on_grid", "project_roi_logits", "rival_roi_sample",
    "observable_pair_features", "bounded_pair_edges", "sparse_observable_pair_features",
    "masked_relation_loss", "sparse_relation_matrix", "set_energy", "bounded_set_selection", "exact_reference_selection",
)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _boxes_valid(boxes, batch=None, max_rois=128):
    if boxes.ndim != 3 or boxes.shape[-1] != 4 or not boxes.is_floating_point():
        raise ValueError("Expected floating BN4 normalized xyxy boxes")
    if (batch is not None and boxes.shape[0] != batch) or boxes.shape[1] > max_rois:
        raise ValueError("Batch mismatch or candidate cap exceeded")
    if not torch.isfinite(boxes).all() or (boxes < 0).any() or (boxes > 1).any():
        raise ValueError("Finite boxes in [0,1] required")
    if (boxes[..., 2:] <= boxes[..., :2]).any():
        raise ValueError("Positive box width and height required")


class SharedRGBPrefix(nn.Module):
    """One narrow RGB prefix producing lossy stride-2 and stride-4 evidence."""

    def __init__(self, detail_channels=8, scout_channels=16):
        super().__init__()
        _positive_int(detail_channels, "detail_channels")
        _positive_int(scout_channels, "scout_channels")
        self.detail = nn.Sequential(
            nn.Conv2d(3, detail_channels, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(detail_channels), nn.SiLU(),
        )
        self.scout = nn.Sequential(
            nn.Conv2d(detail_channels, detail_channels, 3, stride=2, padding=1,
                      groups=detail_channels, bias=False),
            nn.BatchNorm2d(detail_channels), nn.SiLU(),
            nn.Conv2d(detail_channels, scout_channels, 1, bias=False),
            nn.BatchNorm2d(scout_channels), nn.SiLU(),
        )

    def forward(self, rgb):
        """Return (B,D,H/2,W/2), (B,S,H/4,W/4), rounding odd sizes upward."""
        if rgb.ndim != 4 or rgb.shape[1] != 3 or not rgb.is_floating_point():
            raise ValueError("Expected floating BCHW RGB")
        detail = self.detail(rgb)
        return detail, self.scout(detail)


class IndependentDiscovery(nn.Module):
    """Five raw cell maps; target construction and box decoding belong to trainer."""

    def __init__(self, channels=16):
        super().__init__()
        _positive_int(channels, "channels")
        self.spatial = nn.Conv2d(channels, channels, 3, padding=1, groups=channels, bias=False)
        self.output = nn.Conv2d(channels, 5, 1)
        # A low initial existence prior limits untrained proposal floods.
        nn.init.constant_(self.output.bias[:1], -4.59511985013459)

    def forward(self, scout):
        raw = self.output(F.silu(self.spatial(scout)))
        return {"existence": raw[:, :1], "offset": raw[:, 1:3], "log_box_size": raw[:, 3:]}


def normalized_roi_grid(boxes, side=16, max_rois=32):
    """Return pixel-centre B,N,R,R,2 coordinates in the common [0,1] frame."""
    _positive_int(side, "side")
    _positive_int(max_rois, "max_rois")
    _boxes_valid(boxes, max_rois=max_rois)
    axis = (torch.arange(side, device=boxes.device, dtype=boxes.dtype) + 0.5) / side
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    unit = torch.stack((xx, yy), -1)
    return boxes[..., None, None, :2] + unit * (boxes[..., None, None, 2:] - boxes[..., None, None, :2])


def packed_roi_sample(feature, boxes, side=16, max_rois=32):
    """BCHW + BN4 -> BNCRR with ONE grid_sample and no N-fold full-image copy.

    Input boxes already refer to the feature's normalized frame. Border padding
    is used only for feature sampling; observation support remains caller-owned.
    Strict finite checks synchronize CUDA and must be included in profiling.
    """
    if feature.ndim != 4 or not feature.is_floating_point():
        raise ValueError("Expected floating BCHW feature")
    _boxes_valid(boxes, feature.shape[0], max_rois)
    if feature.device != boxes.device:
        raise ValueError("Boxes and features must share a device")
    grid = normalized_roi_grid(boxes, side, max_rois)
    batch, channels = feature.shape[:2]
    count = boxes.shape[1]
    if count == 0:
        return feature.new_zeros(batch, 0, channels, side, side) + feature.sum() * 0
    grid = (2 * grid - 1).reshape(batch, count * side * side, 1, 2)
    # grid_sample requires a matching floating dtype; CPU half is not supported
    # by PyTorch, so promote only this sampling op and restore the output dtype.
    compute = feature.float() if feature.device.type == "cpu" and feature.dtype == torch.float16 else feature
    sampled = F.grid_sample(compute, grid.to(compute.dtype), mode="bilinear",
                            padding_mode="border", align_corners=False).to(feature.dtype)
    return sampled.reshape(batch, channels, count, side, side).permute(0, 2, 1, 3, 4)


def roi_logits_on_grid(logits, boxes, world_points, max_rois=128, outside_value=-20.0):
    """Sample each canonical ROI logit at ONE common B,P,2 world grid.

    Returns logits B,N,P and Boolean geometric support B,N,P. Only an ROI's
    canonical r*r tensor is batched over instances, never a full image feature.
    Support denotes the supplied observed crop, not an inferred amodal shape.
    For partial-view predictions, boxes MUST describe their actual observation
    extent; apply a finer valid-padding support separately when needed.
    """
    if logits.ndim != 5 or logits.shape[2] != 1 or not logits.is_floating_point():
        raise ValueError("Expected BN1RR ROI logits")
    batch, count = logits.shape[:2]
    _boxes_valid(boxes, batch, max_rois)
    if boxes.shape[1] != count or world_points.ndim != 3 or world_points.shape[0] != batch:
        raise ValueError("ROI count or B,P,2 points mismatch")
    if world_points.shape[-1] != 2 or not world_points.is_floating_point():
        raise ValueError("Expected floating world points BP2")
    if any(x.device != logits.device for x in (boxes, world_points)):
        raise ValueError("Sampling inputs must share a device")
    if not torch.isfinite(world_points).all() or not math.isfinite(outside_value):
        raise ValueError("Finite world points and outside value required")
    points = world_points.shape[1]
    if points < 1:
        raise ValueError("Sampling grid cannot be empty")
    if count == 0:
        empty = logits.new_zeros(batch, 0, points) + logits.sum() * 0
        return empty, torch.zeros(batch, 0, points, dtype=torch.bool, device=logits.device)
    local = (world_points[:, None] - boxes[..., None, :2]) / (boxes[..., None, 2:] - boxes[..., None, :2])
    support = ((local >= 0) & (local <= 1)).all(-1)
    grid = (2 * local - 1).reshape(batch * count, points, 1, 2)
    canonical = logits.reshape(batch * count, 1, *logits.shape[-2:])
    compute = canonical.float() if canonical.device.type == "cpu" and canonical.dtype == torch.float16 else canonical
    values = F.grid_sample(compute, grid.to(compute.dtype), mode="bilinear", padding_mode="border",
                           align_corners=False).reshape(batch, count, points).to(logits.dtype)
    return torch.where(support, values, values.new_full((), outside_value)), support


def project_roi_logits(logits, boxes, output_size, max_rois=32, outside_value=-20.0):
    """Project local logits onto a shared output raster, returning BN1HW/support.

    This allocates the requested final masks (not copies of a backbone feature).
    Caller chooses the output size and must count its memory and latency.
    """
    if len(output_size) != 2:
        raise ValueError("output_size must be (height,width)")
    height, width = output_size
    _positive_int(height, "height")
    _positive_int(width, "width")
    yy = (torch.arange(height, device=logits.device, dtype=boxes.dtype) + .5) / height
    xx = (torch.arange(width, device=logits.device, dtype=boxes.dtype) + .5) / width
    gy, gx = torch.meshgrid(yy, xx, indexing="ij")
    points = torch.stack((gx, gy), -1).reshape(1, -1, 2).expand(logits.shape[0], -1, -1)
    sampled, support = roi_logits_on_grid(logits, boxes, points, max_rois, outside_value)
    shape = (*sampled.shape[:2], 1, height, width)
    return sampled.reshape(shape), support.reshape(shape)


def rival_roi_sample(logits, boxes, neighbors, side=16, max_rois=32, max_neighbors=4, outside_value=-20.0):
    """Sample BNK neighboring canonical masks in each TARGET ROI's coordinates.

    neighbors is long BNK, -1 for absent. Returns BNK1RR values/support. This
    copies only at most K small canonical masks, not a full-image feature. The
    caller can reduce valid rival logits by max/logsumexp, retaining a separate
    absence indicator instead of confusing an absent rival with foreground.
    """
    if logits.ndim != 5 or logits.shape[2] != 1:
        raise ValueError("Expected BN1RR logits")
    batch, count = logits.shape[:2]
    _boxes_valid(boxes, batch, max_rois)
    _positive_int(side, "side")
    if boxes.shape[1] != count or neighbors.ndim != 3 or neighbors.shape[:2] != (batch, count):
        raise ValueError("Neighbor/box axes must match ROI logits")
    if neighbors.dtype != torch.long or neighbors.device != logits.device or boxes.device != logits.device:
        raise ValueError("Long neighbors and boxes must share the logit device")
    if neighbors.shape[2] > max_neighbors or not ((neighbors >= -1) & (neighbors < count)).all():
        raise ValueError("Neighbor cap exceeded or invalid neighbor index")
    count_neighbors = neighbors.shape[2]
    result_shape = (batch, count, count_neighbors, 1, side, side)
    if count == 0 or count_neighbors == 0:
        empty = logits.new_zeros(result_shape) + logits.sum() * 0
        return empty, torch.zeros(result_shape, dtype=torch.bool, device=logits.device)
    world = normalized_roi_grid(boxes, side, max_rois)
    batches = torch.arange(batch, device=logits.device)[:, None, None]
    safe = neighbors.clamp_min(0)
    rival_boxes = boxes[batches, safe]
    rival = logits[batches, safe]
    local = (world[:, :, None] - rival_boxes[..., None, None, :2]) / (
        rival_boxes[..., None, None, 2:] - rival_boxes[..., None, None, :2])
    support = ((local >= 0) & (local <= 1)).all(-1) & (neighbors >= 0)[..., None, None]
    # Self-neighbor is not valid rival evidence.
    support &= (neighbors != torch.arange(count, device=logits.device)[None, :, None])[..., None, None]
    canonical = rival.reshape(batch * count * count_neighbors, 1, *logits.shape[-2:])
    compute = canonical.float() if canonical.device.type == "cpu" and canonical.dtype == torch.float16 else canonical
    sampled = F.grid_sample(compute, (2 * local - 1).reshape(-1, side, side, 2).to(compute.dtype),
                            mode="bilinear", padding_mode="border", align_corners=False).to(logits.dtype)
    values = sampled.reshape(result_shape)
    support = support.unsqueeze(3)
    return torch.where(support, values, values.new_full((), outside_value)), support


class InstanceResidual(nn.Module):
    """One bounded, zero-initialized signed ROI update with explicit rival state."""

    def __init__(self, detail_channels=8, semantic_channels=8, query_channels=32,
                 hidden_channels=16, max_update=2.0, max_rois=32):
        super().__init__()
        for name, value in (("detail_channels", detail_channels), ("semantic_channels", semantic_channels),
                            ("query_channels", query_channels), ("hidden_channels", hidden_channels),
                            ("max_rois", max_rois)):
            _positive_int(value, name)
        if not 0 < max_update < float("inf"):
            raise ValueError("Finite positive update bound required")
        self.max_update, self.max_rois = float(max_update), max_rois
        self.detail_channels, self.semantic_channels, self.query_channels = (
            detail_channels, semantic_channels, query_channels)
        self.condition = nn.Linear(query_channels, 2 * hidden_channels)
        self.input = nn.Conv2d(detail_channels + semantic_channels + 4, hidden_channels, 1)
        self.spatial = nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1, groups=hidden_channels)
        self.output = nn.Conv2d(hidden_channels, 1, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, detail, semantic, query, own_logits, rival_logits):
        if detail.ndim != 5 or semantic.ndim != 5 or query.ndim != 3:
            raise ValueError("Expected BNCHW features and BNQ query")
        batch, count, _, height, width = detail.shape
        if count > self.max_rois or semantic.shape != (batch, count, self.semantic_channels, height, width):
            raise ValueError("ROI cap or semantic shape mismatch")
        if detail.shape[2] != self.detail_channels or query.shape != (batch, count, self.query_channels):
            raise ValueError("Detail/query channel mismatch")
        expected = (batch, count, 1, height, width)
        if own_logits.shape != expected or rival_logits.shape != expected:
            raise ValueError("Logits must be BN1HW in THIS ROI's coordinates")
        items = (detail, semantic, query, own_logits, rival_logits)
        if any(x.device != detail.device or not x.is_floating_point() for x in items):
            raise ValueError("Floating residual inputs must share a device")
        if count == 0:
            return own_logits + detail.sum() * 0
        yy = 2 * (torch.arange(height, device=detail.device, dtype=detail.dtype) + .5) / height - 1
        xx = 2 * (torch.arange(width, device=detail.device, dtype=detail.dtype) + .5) / width - 1
        gy, gx = torch.meshgrid(yy, xx, indexing="ij")
        coordinates = torch.stack((gx, gy), 0).expand(batch, count, -1, -1, -1)
        merged = torch.cat((detail, semantic, own_logits, rival_logits, coordinates), 2)
        hidden = self.input(merged.reshape(batch * count, -1, height, width))
        gamma, beta = self.condition(query).reshape(batch * count, -1, 1, 1).chunk(2, 1)
        hidden = F.silu(hidden * (1 + .1 * gamma.tanh()) + beta)
        delta = self.output(F.silu(self.spatial(hidden)))
        return own_logits + self.max_update * delta.tanh().reshape(expected)


def _mask_support_check(masks, supports, max_candidates=128, max_samples=16384):
    if masks.shape != supports.shape or masks.ndim != 3:
        raise ValueError("Expected equal B,N,P masks/supports on one common grid")
    if masks.dtype != torch.bool or supports.dtype != torch.bool or masks.device != supports.device:
        raise ValueError("Boolean masks/supports must share a device")
    if masks.shape[1] > max_candidates or not 1 <= masks.shape[2] <= max_samples:
        raise ValueError("Candidate/sample cap exceeded or empty sampling grid")


def observable_pair_features(masks, supports, min_common=4, min_foreground=2, max_candidates=64):
    """Dense symmetric five-feature diagnostic API retained for small N."""
    _mask_support_check(masks, supports, max_candidates)
    _positive_int(min_common, "min_common")
    _positive_int(min_foreground, "min_foreground")
    observed, foreground = supports.float(), (masks & supports).float()
    common = observed @ observed.transpose(1, 2)
    intersection = foreground @ foreground.transpose(1, 2)
    left = foreground @ observed.transpose(1, 2)
    right = left.transpose(1, 2)
    union = left + right - intersection
    il, ir = intersection / left.clamp_min(1), intersection / right.clamp_min(1)
    features = torch.stack((intersection / union.clamp_min(1), (il + ir) / 2, (il - ir).abs(),
                            common / masks.shape[2], union / common.clamp_min(1)), -1)
    known = (common >= min_common) & (union >= min_foreground)
    known &= ~torch.eye(masks.shape[1], device=masks.device, dtype=torch.bool)[None]
    return {"features": features, "known": known, "common_samples": common,
            "common_foreground_union": union}


def _box_pair_score(boxes):
    lo = torch.maximum(boxes[:, :, None, :2], boxes[:, None, :, :2])
    hi = torch.minimum(boxes[:, :, None, 2:], boxes[:, None, :, 2:])
    intersection = (hi - lo).clamp_min(0).prod(-1)
    area = (boxes[..., 2:] - boxes[..., :2]).prod(-1)
    iou = intersection / (area[:, :, None] + area[:, None, :] - intersection).clamp_min(1e-12)
    centre = (boxes[..., :2] + boxes[..., 2:]) / 2
    distance = (centre[:, :, None] - centre[:, None, :]).square().sum(-1)
    scale = ((boxes[..., 2:] - boxes[..., :2]).square().sum(-1))
    distance = distance / (scale[:, :, None] + scale[:, None, :]).clamp_min(1e-12)
    return 2 * iou + 1 / (1 + distance)


def bounded_pair_edges(boxes, max_neighbors=4, max_edges=256, valid=None, max_candidates=128):
    """Deterministic K-neighbor union followed by an explicit GLOBAL edge cap.

    Returns padded edges BE2 (i<j; -1 means padding), valid BE, counts B. Ties
    prefer lower input indices. This is deterministic for fixed ordering, NOT
    permutation-equivariant under exact geometric ties. Union degrees can be
    larger than K; max_edges is the actual total bound.
    """
    _boxes_valid(boxes, max_rois=max_candidates)
    _positive_int(max_neighbors, "max_neighbors")
    _positive_int(max_edges, "max_edges")
    batch, count = boxes.shape[:2]
    if valid is None:
        valid = torch.ones(batch, count, device=boxes.device, dtype=torch.bool)
    if valid.shape != (batch, count) or valid.dtype != torch.bool or valid.device != boxes.device:
        raise ValueError("Expected Boolean BN validity")
    edge_count = min(max_edges, count * (count - 1) // 2)
    if edge_count == 0:
        return {"edges": torch.empty(batch, 0, 2, dtype=torch.long, device=boxes.device),
                "valid": torch.empty(batch, 0, dtype=torch.bool, device=boxes.device),
                "counts": torch.zeros(batch, dtype=torch.long, device=boxes.device)}
    score = _box_pair_score(boxes)
    pair_valid = valid[:, :, None] & valid[:, None, :] & ~torch.eye(count, device=boxes.device, dtype=torch.bool)
    ranked = torch.argsort(score.masked_fill(~pair_valid, -float("inf")), dim=-1, descending=True, stable=True)
    chosen = torch.zeros_like(pair_valid)
    chosen.scatter_(2, ranked[..., :min(max_neighbors, count - 1)], True)
    chosen = (chosen | chosen.transpose(1, 2)) & pair_valid
    left, right = torch.triu_indices(count, count, 1, device=boxes.device)
    candidate_scores = score[:, left, right].masked_fill(~chosen[:, left, right], -float("inf"))
    order = torch.argsort(candidate_scores, dim=-1, descending=True, stable=True)[:, :edge_count]
    active = torch.gather(chosen[:, left, right], 1, order)
    edges = torch.stack((left[order], right[order]), -1).masked_fill(~active[..., None], -1)
    return {"edges": edges, "valid": active, "counts": active.sum(-1)}


def sparse_observable_pair_features(masks, supports, edges, min_common=4, min_foreground=2,
                                    queries=None, boxes=None, scores=None, max_edges=256,
                                    max_candidates=128, max_samples=16384):
    """Bounded BEP common-support evidence and optional symmetric identity cues.

    Base feature width is 5. Optional normalized queries add 2D (abs difference,
    elementwise product); geometry adds 5 (relative centre 2, log size ratio 2,
    box IoU 1); scores add 2 (min,max). No N*N*P tensor is allocated. Unknown
    common domains and empty common foreground remain unknown, not negatives.
    """
    _mask_support_check(masks, supports, max_candidates, max_samples)
    _positive_int(min_common, "min_common")
    _positive_int(min_foreground, "min_foreground")
    batch, count, points = masks.shape
    if edges.ndim != 3 or edges.shape[0] != batch or edges.shape[-1] != 2 or edges.dtype != torch.long:
        raise ValueError("Expected long B,E,2 edges")
    if edges.device != masks.device or edges.shape[1] > max_edges:
        raise ValueError("Edge device mismatch or edge cap exceeded")
    padded = (edges == -1).all(-1)
    good = ((edges >= 0) & (edges < count)).all(-1) & (edges[..., 0] < edges[..., 1])
    if not (padded | good).all():
        raise ValueError("Edges must be i<j or exactly (-1,-1) padding")
    if count == 0:
        if not padded.all():
            raise ValueError("Nonempty edges for empty candidates")
        extra = 0
        if queries is not None:
            if (queries.ndim != 3 or queries.shape[:2] != (batch, count)
                    or not 1 <= queries.shape[-1] <= 64 or queries.device != masks.device):
                raise ValueError("Expected empty BNQ appearance queries with 1<=Q<=64")
            extra += 2 * queries.shape[-1]
        if boxes is not None:
            _boxes_valid(boxes, batch, max_candidates)
            if boxes.shape[1] != count or boxes.device != masks.device:
                raise ValueError("Geometry count/device mismatch")
            extra += 5
        if scores is not None:
            if scores.shape != (batch, count) or scores.device != masks.device:
                raise ValueError("Score count/device mismatch")
            extra += 2
        base = torch.zeros(*edges.shape[:2], 5 + extra, device=masks.device)
        return {"features": base, "known": good, "edges": edges,
                "common_samples": base[..., 0], "common_foreground_union": base[..., 0]}
    indices = edges.clamp_min(0)
    batches = torch.arange(batch, device=masks.device)[:, None]
    left, right = indices.unbind(-1)
    sl, sr = supports[batches, left], supports[batches, right]
    ml, mr = masks[batches, left], masks[batches, right]
    observed = sl & sr
    lf, rf = ml & observed, mr & observed
    common = observed.sum(-1).float()
    intersection = (lf & rf).sum(-1).float()
    la, ra = lf.sum(-1).float(), rf.sum(-1).float()
    union = la + ra - intersection
    il, ir = intersection / la.clamp_min(1), intersection / ra.clamp_min(1)
    components = [torch.stack((intersection / union.clamp_min(1), (il + ir) / 2, (il - ir).abs(),
                               common / points, union / common.clamp_min(1)), -1)]
    if queries is not None:
        if queries.ndim != 3 or queries.shape[:2] != (batch, count) or not 1 <= queries.shape[-1] <= 64:
            raise ValueError("Expected BNQ appearance queries with 1<=Q<=64")
        if queries.device != masks.device or not queries.is_floating_point() or not torch.isfinite(queries).all():
            raise ValueError("Finite appearance queries must share the mask device")
        normalized = F.normalize(queries, dim=-1, eps=1e-6)
        ql, qr = normalized[batches, left], normalized[batches, right]
        components.extend(((ql - qr).abs(), ql * qr))
    if boxes is not None:
        _boxes_valid(boxes, batch, max_candidates)
        if boxes.shape[1] != count or boxes.device != masks.device:
            raise ValueError("Geometry count/device mismatch")
        bl, br = boxes[batches, left], boxes[batches, right]
        wl, wr = bl[..., 2:] - bl[..., :2], br[..., 2:] - br[..., :2]
        centre = ((bl[..., :2] + bl[..., 2:] - br[..., :2] - br[..., 2:]) / 2).abs()
        relative = centre / (wl + wr).clamp_min(1e-6)
        log_size = (wl.clamp_min(1e-6).log() - wr.clamp_min(1e-6).log()).abs()
        intersection_box = (torch.minimum(bl[..., 2:], br[..., 2:]) -
                            torch.maximum(bl[..., :2], br[..., :2])).clamp_min(0).prod(-1)
        iou = intersection_box / (wl.prod(-1) + wr.prod(-1) - intersection_box).clamp_min(1e-12)
        components.append(torch.cat((relative, log_size, iou[..., None]), -1))
    if scores is not None:
        if scores.shape != (batch, count) or scores.device != masks.device or not scores.is_floating_point():
            raise ValueError("Expected floating BN scores on the mask device")
        if not torch.isfinite(scores).all():
            raise ValueError("Finite scores required")
        lscore, rscore = scores[batches, left], scores[batches, right]
        components.append(torch.stack((torch.minimum(lscore, rscore), torch.maximum(lscore, rscore)), -1))
    features = torch.cat(components, -1).masked_fill(~good[..., None], 0)
    known = good & (common >= min_common) & (union >= min_foreground)
    return {"features": features, "known": known, "edges": edges, "common_samples": common,
            "common_foreground_union": union}


class ObservableRelation(nn.Module):
    """Symmetric identity MLP with a persistent, explicit deployment readiness gate."""

    def __init__(self, hidden=16, appearance_channels=0, include_geometry=False, include_scores=False, trained=False):
        super().__init__()
        _positive_int(hidden, "hidden")
        if (isinstance(appearance_channels, bool) or not isinstance(appearance_channels, int)
                or not 0 <= appearance_channels <= 64):
            raise ValueError("appearance_channels must be between 0 and 64")
        self.input_channels = 5 + 2 * appearance_channels + 5 * bool(include_geometry) + 2 * bool(include_scores)
        self.net = nn.Sequential(nn.Linear(self.input_channels, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.register_buffer("relation_trained", torch.tensor(bool(trained), dtype=torch.bool))

    def mark_trained(self, trained=True):
        """Caller marks this only after supervised updates, not at random initialization."""
        self.relation_trained.fill_(bool(trained))

    def forward(self, evidence):
        features, known = evidence["features"], evidence["known"]
        if features.shape[-1] != self.input_channels or known.shape != features.shape[:-1]:
            raise ValueError("Relation input width/known shape mismatch")
        if known.dtype != torch.bool or known.device != features.device:
            raise ValueError("Boolean relation evidence must share the feature device")
        if not torch.isfinite(features).all():
            raise ValueError("Finite relation features required")
        logits = self.net(features).squeeze(-1)
        result = {"logits": logits, "known": known,
                  "selection_known": known & self.relation_trained, "trained": self.relation_trained}
        if "edges" in evidence:
            result["edges"] = evidence["edges"]
        return result


def masked_relation_loss(relation, same_instance, label_known):
    """BCE on labelled observable pairs; supports dense BNN or unique sparse BE."""
    logits, evidence_known = relation["logits"], relation["known"]
    if logits.ndim not in (2, 3) or any(x.shape != logits.shape for x in
                                                     (same_instance, label_known, evidence_known)):
        raise ValueError("Expected equal BE or square BNN relation/label shapes")
    indicators = (same_instance, label_known, evidence_known)
    if any(x.dtype != torch.bool or x.device != logits.device for x in indicators):
        raise ValueError("Boolean labels/evidence must share the logits device")
    if logits.ndim == 3:
        if logits.shape[-1] != logits.shape[-2]:
            raise ValueError("Dense relations must be square")
        if any(not torch.equal(x, x.transpose(1, 2)) for x in (label_known, evidence_known)):
            raise ValueError("Dense known indicators must be symmetric")
        labelled = evidence_known & label_known
        if not torch.equal(same_instance[labelled], same_instance.transpose(1, 2)[labelled]):
            raise ValueError("Known identity labels must be symmetric")
        valid = torch.triu(labelled, diagonal=1)
    else:
        valid = evidence_known & label_known
    selected = logits[valid]
    if not torch.isfinite(selected).all():
        raise ValueError("Supervised relation logits must be finite")
    if selected.numel() == 0:
        return selected.sum()  # Unknown NaN positions are never multiplied by zero.
    return F.binary_cross_entropy_with_logits(selected, same_instance[valid].to(logits.dtype))


def sparse_relation_matrix(relation, count, use_selection_gate=True, max_candidates=128):
    """Scatter unique sparse BE relations to symmetric BNN without inventing edges.

    Gate defaults to selection_known, not raw evidence-known. Duplicate sparse
    indices are rejected because conflicting last-writer behavior is undefined.
    """
    if isinstance(count, bool) or not isinstance(count, int) or not 0 <= count <= max_candidates:
        raise ValueError("Invalid candidate count")
    edges, logits = relation["edges"], relation["logits"]
    known = relation["selection_known"] if use_selection_gate else relation["known"]
    if (logits.ndim != 2 or edges.shape != (*logits.shape, 2) or known.shape != logits.shape
            or edges.dtype != torch.long or known.dtype != torch.bool):
        raise ValueError("Expected BE logits/known and BE2 edges")
    if any(x.device != logits.device for x in (edges, known)):
        raise ValueError("Sparse relation inputs must share a device")
    padded = (edges == -1).all(-1)
    valid = ((edges >= 0) & (edges < count)).all(-1) & (edges[..., 0] < edges[..., 1])
    if not (padded | valid).all() or (known & ~valid).any():
        raise ValueError("Known sparse edges must be valid i<j pairs")
    batch = logits.shape[0]
    values = logits.new_zeros(batch, count, count)
    observed = torch.zeros(batch, count, count, device=logits.device, dtype=torch.bool)
    if count == 0:
        return values, observed
    keys = edges[..., 0] * count + edges[..., 1]
    sorted_keys = keys.masked_fill(~valid, count * count).sort(-1).values
    if ((sorted_keys[:, 1:] == sorted_keys[:, :-1]) & (sorted_keys[:, 1:] < count * count)).any():
        raise ValueError("Sparse relation pairs must be unique")
    b = torch.arange(batch, device=logits.device)[:, None].expand_as(known)[known]
    left, right = edges[known].unbind(-1)
    values[b, left, right] = logits[known]
    values[b, right, left] = logits[known]
    observed[b, left, right] = True
    observed[b, right, left] = True
    return values, observed


def _selection_check(unary, relation_logits, known, penalty, max_candidates=128):
    count = unary.numel()
    if unary.ndim != 1 or count > max_candidates or not unary.is_floating_point():
        raise ValueError("Expected floating N unaries under candidate cap")
    if relation_logits.shape != (count, count) or known.shape != (count, count) or known.dtype != torch.bool:
        raise ValueError("Expected NN pair logits and Boolean known")
    if any(x.device != unary.device for x in (relation_logits, known)):
        raise ValueError("Selection inputs must share a device")
    if not math.isfinite(penalty) or penalty < 0:
        raise ValueError("Finite nonnegative penalty required")
    if not torch.isfinite(unary).all() or not torch.isfinite(relation_logits[known]).all():
        raise ValueError("Finite unary/known relations required")
    if not torch.equal(known, known.T) or not torch.allclose(relation_logits[known], relation_logits.T[known]):
        raise ValueError("Symmetric relation evidence required")


def set_energy(unary, relation_logits, known, selected, penalty=1.0):
    """Fixed-candidate nonmonotone submodular surrogate; unknown means no penalty."""
    _selection_check(unary, relation_logits, known, penalty)
    if selected.shape != unary.shape or selected.dtype != torch.bool or selected.device != unary.device:
        raise ValueError("Expected Boolean N selection")
    active = torch.triu(selected[:, None] & selected[None] & known, diagonal=1)
    return unary[selected].sum() - float(penalty) * relation_logits[active].sigmoid().sum()


@torch.no_grad()
def bounded_set_selection(unary, relation_logits, known, budget, penalty=1.0, relation_trained=False,
                          max_steps=128, improvement_rounds=4, max_candidates=128, tolerance=1e-7):
    """Bounded deterministic greedy + add/remove/one-swap improvement.

    Returns a dict with selected Boolean N, indices, energy, greedy_energy and
    actual steps. All pair/candidate deltas are vectorized on the input device;
    loops are bounded by max_steps and improvement_rounds, never 2**N. Stable
    ties prefer add, then remove, then lexicographic swap and lower input index.
    Exact ties are NOT permutation-equivariant. This is a local surrogate solver
    with NO AP or optimum guarantee. Random relation heads have zero influence:
    relation_trained must be explicitly True to activate the pair penalty.
    """
    _selection_check(unary, relation_logits, known, penalty, max_candidates)
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("budget must be a nonnegative integer")
    _positive_int(max_steps, "max_steps")
    if isinstance(improvement_rounds, bool) or not isinstance(improvement_rounds, int) or improvement_rounds < 0:
        raise ValueError("improvement_rounds must be a nonnegative integer")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("Finite nonnegative tolerance required")
    count, limit = unary.numel(), min(budget, unary.numel())
    selected = torch.zeros(count, device=unary.device, dtype=torch.bool)
    effective_penalty = float(penalty) if bool(relation_trained) else 0.0
    pair = unary.new_zeros(count, count)
    pair[known] = relation_logits[known].sigmoid().to(unary.dtype) * effective_penalty
    pair.fill_diagonal_(0)
    steps = 0
    for _ in range(min(limit, max_steps)):
        gain = unary - pair[:, selected].sum(-1)
        gain = gain.masked_fill(selected, -float("inf"))
        index = gain.argmax() if count else None
        if index is None or gain[index] <= tolerance:
            break
        selected[index] = True
        steps += 1
    greedy = unary[selected].sum() - pair[selected][:, selected].sum() / 2
    for _ in range(improvement_rounds):
        if count == 0 or steps >= max_steps:
            break
        costs = pair[:, selected].sum(-1)
        add = (unary - costs).masked_fill(selected, -float("inf"))
        if int(selected.sum()) >= limit:
            add.fill_(-float("inf"))
        remove = (-unary + costs).masked_fill(~selected, -float("inf"))
        # delta(remove i, add j) = remove_i + add_j + pair_ij.
        swap = remove[:, None] + (unary - costs)[None] + pair
        swap.masked_fill_(~(selected[:, None] & ~selected[None]), -float("inf"))
        gains = torch.cat((add, remove, swap.flatten()))
        operation = gains.argmax()
        if gains[operation] <= tolerance:
            break
        position = int(operation)
        if position < count:
            selected[position] = True
        elif position < 2 * count:
            selected[position - count] = False
        else:
            left, right = divmod(position - 2 * count, count)
            selected[left], selected[right] = False, True
        steps += 1
    value = unary[selected].sum() - pair[selected][:, selected].sum() / 2
    return {"selected": selected, "indices": selected.nonzero(as_tuple=False).flatten(), "energy": value,
            "greedy_energy": greedy, "steps": steps, "relation_active": bool(relation_trained)}


def exact_reference_selection(unary, relation_logits, known, budget, penalty=1.0):
    """Small N<=12 CPU exhaustive oracle, NEVER called by deployment selection."""
    if unary.is_cuda or unary.numel() > 12 or isinstance(budget, bool) or not isinstance(budget, int) or budget < 0:
        raise ValueError("CPU reference only: N<=12 and nonnegative integer budget")
    best = torch.zeros_like(unary, dtype=torch.bool)
    value = float(set_energy(unary, relation_logits, known, best, penalty))
    for size in range(1, min(budget, unary.numel()) + 1):
        for indices in combinations(range(unary.numel()), size):
            selected = torch.zeros_like(best)
            selected[list(indices)] = True
            new_value = float(set_energy(unary, relation_logits, known, selected, penalty))
            if new_value > value:
                best, value = selected, new_value
    return best, value
