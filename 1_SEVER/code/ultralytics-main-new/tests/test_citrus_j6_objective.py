"""Independent objective/decoder regressions; synthetic CPU, not AP/GPU evidence."""

from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from citrus_j6_inference import postprocess_j6
from ultralytics import YOLO
from ultralytics.nn.modules.citrus_j6 import SegmentCitrusJ6
from ultralytics.nn.modules.citrus_j6_ops import normalized_roi_grid
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace, ops
from ultralytics.utils.citrus_j6_loss import assign_cells, instance_targets, visible_points


ROOT = Path(__file__).resolve().parents[1]
YAML = ROOT / "0_orange_yaml/J_V6_series/J94_full.yaml"


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17102026)
    yield
    torch.set_num_threads(old)


def make_model():
    model = YOLO(str(YAML), task="segment", verbose=False).model.train()
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    return model


def make_batch(empty=False, vanished=False):
    masks = torch.zeros(2, 64, 80)
    masks[0, 8:24, 8:24] = 1
    masks[0, 32:52, 48:68] = 2
    boxes = torch.tensor([[.2, .25, .2, .25], [.725, .65625, .25, .3125]])
    if vanished or empty:
        masks.zero_()
    if empty:
        boxes = boxes[:0]
    return dict(img=torch.rand(2, 3, 128, 160), high_img=torch.rand(2, 3, 256, 320),
                masks=masks, bboxes=boxes, cls=torch.zeros(len(boxes), 1), batch_idx=torch.zeros(len(boxes)))


def no_discovery(raw, image, threshold):
    device = raw["proto"].device
    return torch.empty(0, 4, device=device), torch.empty(0, device=device), torch.empty(0, dtype=torch.long, device=device)


def test_instance_targets_and_visible_point_uses_observed_mask():
    batch = make_batch()
    masks = instance_targets(batch, 0, torch.tensor([0, 1]), overlap=True)
    assert masks.shape == (2, 64, 80) and masks[0].sum() == 256 and masks[1].sum() == 400
    # Bbox centre is hidden, so nearest existing foreground is required.
    masks = torch.zeros(2, 5, 7, dtype=torch.bool)
    masks[0, 1, 3] = True
    boxes = torch.tensor([[.5, .5, .2, .2], [.7, .6, .1, .1]])
    points, nonempty = visible_points(masks, boxes)
    assert torch.allclose(points[0], torch.tensor([.5, .3]))
    assert torch.equal(points[1], boxes[1, :2])
    assert nonempty.tolist() == [True, False]


def test_cell_collisions_bounded_repair_and_explicit_unresolved_count():
    points = torch.full((12, 2), .5)
    sizes = torch.arange(1, 13).float()[:, None].expand(-1, 2) / 100
    chosen, cells = assign_cells(points, sizes, 6, 6)
    assert len(chosen) == len(cells) == 9
    assert len(cells.unique()) == 9 and chosen.tolist() == list(range(9))
    assert len(points) - len(chosen) == 3
    again = assign_cells(points, sizes, 6, 6)
    assert torch.equal(chosen, again[0]) and torch.equal(cells, again[1])


def test_relocated_cells_decode_same_unbounded_original_point():
    head = SegmentCitrusJ6(nm=4, npr=8, ch=(8, 16, 32, 3))
    points, sizes = torch.full((9, 2), .55), torch.full((9, 2), .1)
    chosen, cells = assign_cells(points, sizes, 6, 6)
    offset_target = points[chosen] * 6 - torch.stack((cells % 6, cells // 6), -1)
    assert (offset_target < 0).any() and (offset_target > 1).any()
    heat = torch.full((1, 1, 6, 6), -100.)
    heat.flatten()[cells] = 8.
    offset = torch.full((1, 2, 6, 6), .5)
    offset[0].flatten(1)[:, cells] = offset_target.T
    raw = {"j6": {"discovery": {"existence": heat, "offset": offset,
                                  "log_box_size": torch.full((1, 2, 6, 6), -3.)},
                   "gain": torch.zeros(1, 1, 6, 6)}}
    boxes, _, ids = head.discover(raw, 0, .1)
    assert len(ids) == 9
    assert torch.allclose((boxes[:, :2] + boxes[:, 2:]) / 2, torch.full((9, 2), .55), atol=1e-7)


@pytest.mark.parametrize("empty", [False, True])
def test_full_objective_multiple_gt_or_empty_images_backward_and_pending_gate(empty):
    model, batch = make_model(), make_batch(empty=empty)
    total, components = model.loss(batch)
    assert len(total) == len(components) == 9
    assert torch.isfinite(total).all() and torch.isfinite(components).all()
    total.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    criterion, head = model.criterion, model.model[-1]
    assert criterion.last["annotated"] == (0 if empty else 2)
    assert criterion.last["scout_assigned"] == (0 if empty else 2)
    assert criterion.last["local_positive"] == (0 if empty else 4)
    assert head.auxiliary_updates.item() == 0 and not head.relation.relation_trained.item()
    assert criterion.pending["local_forwards"] == 1
    criterion.commit_updates(False)
    assert head.auxiliary_updates.item() == 0 and not any(criterion.pending.values())
    model.zero_grad()
    model.loss(batch)
    criterion.commit_updates(True)
    assert head.auxiliary_updates.item() == 1 and not any(criterion.pending.values())


def test_vanished_gt_raster_not_negative_mask_or_readiness_evidence(monkeypatch):
    model = make_model()
    monkeypatch.setattr(model.model[-1], "discover", no_discovery)
    total, components = model.loss(make_batch(vanished=True))
    criterion = model.criterion
    assert criterion.last["empty_gt_raster"] == 2
    assert criterion.last["scout_assigned"] == 2  # retain known bbox/existence supervision
    assert components[6].item() == 0  # local mask + quality
    assert components[7].item() == 0  # identity
    assert components[8].item() == 0  # gain
    assert torch.isfinite(total).all()
    criterion.commit_updates(True)
    assert model.model[-1].auxiliary_updates.item() == 0
    assert model.model[-1].gain_updates.item() == 0


def test_local_gt_sampling_uses_same_world_frame_as_roi_features(monkeypatch):
    model, batch = make_model(), make_batch()
    monkeypatch.setattr(model.model[-1], "discover", no_discovery)
    # Freeze training-domain jitter for an independent analytic coordinate oracle.
    monkeypatch.setattr(torch, "rand_like", lambda tensor, *a, **kw: torch.full_like(tensor, .5))
    original = F.binary_cross_entropy_with_logits
    captured = []

    def record(input, target, *args, **kwargs):
        if input.ndim == 5:
            captured.append(target.detach().clone())
        return original(input, target, *args, **kwargs)

    monkeypatch.setattr(F, "binary_cross_entropy_with_logits", record)
    model.loss(batch)
    assert len(captured) == 1
    axis = (torch.arange(16) + .5) / 16
    # First box xyxy=(.1,.125,.3,.375), source-zero domain expanded 1.2x.
    world_x, world_y = .08 + .24 * axis, .1 + .3 * axis
    expected = ((world_x[None] >= .1) & (world_x[None] < .3) &
                (world_y[:, None] >= .125) & (world_y[:, None] < .375))
    assert torch.equal(captured[0][0, 0, 0].bool(), expected)
    # The second same-GT prediction deliberately uses a different 1.6x domain;
    # agreement is evaluated in world coordinates, not by equating local grids.
    world_x, world_y = .04 + .32 * axis, .05 + .4 * axis
    second = ((world_x[None] >= .1) & (world_x[None] < .3) &
              (world_y[:, None] >= .125) & (world_y[:, None] < .375))
    assert torch.equal(captured[0][0, 1, 0].bool(), second)


def test_gain_target_retains_both_signed_risk_differences(monkeypatch):
    model, batch = make_model(), make_batch()
    head = model.model[-1]
    monkeypatch.setattr(head, "discover", no_discovery)
    original_decode = head.decode_rois
    original_smooth = F.smooth_l1_loss
    targets = []

    def signed_decode(raw, domains, coefficients, has_own, scores, sources):
        decoded = original_decode(raw, domains, coefficients, has_own, scores, sources)
        grid = normalized_roi_grid(domains, head.roi_side, head.max_rois)[0]
        labels = torch.tensor([1, 1, 2, 2], device=domains.device)
        binary = batch["masks"][0][None] == labels[:, None, None]
        target = F.grid_sample(binary[:, None].float(), 2 * grid - 1, mode="nearest", align_corners=False)[None]
        # Different GT positions, one improved and one worsened finite-model
        # pixel-risk event. This is a unit oracle, not a trained model claim.
        sign = 2 * target - 1
        better = domains[0, :, 0] < .5
        decoded["reference_logits"] = torch.zeros_like(target)
        decoded["logits"] = sign * torch.where(better, .5, -.5)[None, :, None, None, None]
        return decoded

    def record_smooth(input, target, *args, **kwargs):
        if input.ndim == 1 and target.ndim == 1 and kwargs.get("reduction") == "none":
            targets.append(target.detach().clone())
        return original_smooth(input, target, *args, **kwargs)

    monkeypatch.setattr(head, "decode_rois", signed_decode)
    monkeypatch.setattr(F, "smooth_l1_loss", record_smooth)
    total, _ = model.loss(batch)
    assert torch.isfinite(total).all() and targets
    recorded = torch.cat(targets)
    assert (recorded > 0).any() and (recorded < 0).any()
    assert ((recorded >= -1) & (recorded <= 1)).all()


def make_inference_scene(count=64):
    head = SegmentCitrusJ6(nm=4, npr=8, selection=False, ch=(8, 16, 32, 3)).eval()
    head.stride = torch.tensor([8., 16., 32.])
    head.auxiliary_updates.fill_(1)
    boxes = torch.tensor([[12 * (i % 8) + .3, 8 * (i // 8) + .3,
                           12 * (i % 8) + 8.3, 8 * (i // 8) + 6.3] for i in range(count)])
    scores = torch.linspace(.99, .2, count)
    coefficients = torch.zeros(count, 4)
    coefficients[:, 0] = 1
    detections = torch.cat((boxes, scores[:, None], torch.zeros(count, 1), coefficients), -1)
    proto = torch.zeros(1, 4, 16, 24)
    proto[:, 0] = 1
    raw = {"proto": proto, "feats": [torch.zeros(1, 8, 8, 12), torch.zeros(1, 16, 4, 6),
                                      torch.zeros(1, 32, 2, 3)],
           "j6": {"native_high": True, "detail": torch.zeros(1, 8, 64, 96),
                   "semantic": torch.zeros(1, 8, 8, 12),
                   "discovery": {"existence": torch.full((1, 1, 32, 48), -100.),
                                 "offset": torch.full((1, 2, 32, 48), .5),
                                 "log_box_size": torch.full((1, 2, 32, 48), -3.)},
                   "gain": torch.zeros(1, 1, 32, 48)}}
    preds = ((torch.zeros(1, 9, count), proto), raw)
    return head, preds, detections


@pytest.mark.parametrize("count", [4, 64])
def test_zero_residual_projection_keeps_original_crop_semantics(monkeypatch, count):
    import citrus_j6_inference as infer

    head, preds, detections = make_inference_scene(count)
    calls = []

    def first_nms(*args, **kwargs):
        calls.append(True)
        return [detections.clone()]

    monkeypatch.setattr(infer.nms, "non_max_suppression", first_nms)
    expected_proto = F.interpolate(preds[0][1].float(), (32, 48), mode="bilinear", align_corners=False)[0]
    expected = ops.process_mask(expected_proto, detections[:, 6:], detections[:, :4], (64, 96))
    result = postprocess_j6(head, preds, conf=.001)[0]
    assert calls == [True]  # no second NMS that silently loses touching instances
    assert torch.equal(result["bboxes"], detections[:, :4])
    assert torch.equal(result["masks"], expected)
    assert result["masks"].shape == (count, 32, 48)


def test_enabled_inference_rejects_resize_only_fallback(monkeypatch):
    import citrus_j6_inference as infer

    head, preds, detections = make_inference_scene(4)
    preds[1]["j6"]["native_high"] = False
    monkeypatch.setattr(infer.nms, "non_max_suppression", lambda *a, **kw: [detections])
    with pytest.raises(ValueError, match="original-pixel high"):
        postprocess_j6(head, preds)


def test_zero_untrained_readiness_cannot_run_roi_or_relation(monkeypatch):
    import citrus_j6_inference as infer

    head, preds, detections = make_inference_scene(4)
    head.auxiliary_updates.zero_()
    head.selection = True
    monkeypatch.setattr(infer.nms, "non_max_suppression", lambda *a, **kw: [detections])

    def forbidden(*args, **kwargs):
        raise AssertionError("untrained auxiliary heads must not be used")

    monkeypatch.setattr(head, "decode_rois", forbidden)
    monkeypatch.setattr(head, "discover", forbidden)
    result = postprocess_j6(head, preds)[0]
    assert len(result["bboxes"]) == 4


def test_untrained_relation_gate_skips_candidate_set_suppression(monkeypatch):
    import citrus_j6_inference as infer

    head, preds, detections = make_inference_scene(4)
    head.selection = True
    monkeypatch.setattr(infer.nms, "non_max_suppression", lambda *a, **kw: [detections])

    def forbidden(*args, **kwargs):
        raise AssertionError("untrained relation may not suppress candidates")

    monkeypatch.setattr(infer, "bounded_set_selection", forbidden)
    result = postprocess_j6(head, preds)[0]
    assert len(result["bboxes"]) == 4


@pytest.mark.parametrize("same_domain", [False, True])
def test_trained_relation_unknown_retention_and_known_duplicate_penalty(monkeypatch, same_domain):
    import citrus_j6_inference as infer

    head, preds, detections = make_inference_scene(2)
    head.selection = True
    head.relation.mark_trained()
    with torch.no_grad():
        for parameter in head.relation.parameters():
            parameter.zero_()
        head.relation.net[-1].bias.fill_(20.)
    if same_domain:
        detections[1, :4] = detections[0, :4]
    monkeypatch.setattr(infer.nms, "non_max_suppression", lambda *a, **kw: [detections])
    result = postprocess_j6(head, preds)[0]
    if same_domain:
        assert len(result["bboxes"]) == 1
        assert head.last_inference["known_relations"] == 1
        assert head.last_inference["removed_by_relation"] == 1
    else:
        assert len(result["bboxes"]) == 2
        assert head.last_inference["known_relations"] == 0


def test_independent_new_candidate_is_not_removed_by_second_box_nms(monkeypatch):
    import citrus_j6_inference as infer

    head, preds, detections = make_inference_scene(1)
    with torch.no_grad():
        head.roi_decoder.output.bias.fill_(.5)
    calls = []

    def first_nms(*args, **kwargs):
        calls.append(True)
        return [detections]

    monkeypatch.setattr(infer.nms, "non_max_suppression", first_nms)
    new_box = detections[:, :4] / torch.tensor([96., 64., 96., 64.])
    monkeypatch.setattr(head, "discover", lambda *a, **kw: (new_box, torch.tensor([.7]), torch.tensor([0])))
    result = postprocess_j6(head, preds)[0]
    assert calls == [True]
    assert len(result["bboxes"]) == 2
    assert torch.equal(result["bboxes"][0], result["bboxes"][1])
    assert head.last_inference["new_candidates"] == 1


def test_inference_uses_no_gt_fields_and_requires_fixed_eval_raster(monkeypatch):
    import citrus_j6_inference as infer

    head, preds, detections = make_inference_scene(4)
    monkeypatch.setattr(infer.nms, "non_max_suppression", lambda *a, **kw: [detections])
    # preds contain predictions/evidence only: no labels, GT masks or assignments.
    assert not any(key in preds[1] for key in ("bboxes", "cls", "masks", "batch_idx"))
    assert postprocess_j6(head, preds)[0]["masks"].shape[-2:] == (32, 48)
    with pytest.raises(ValueError, match="mask_ratio=2"):
        postprocess_j6(head, preds, mask_ratio=1)
