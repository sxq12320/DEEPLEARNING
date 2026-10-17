"""Scout geometry roundtrips; synthetic CPU contracts, not recall/AP evidence."""

from io import BytesIO
from pathlib import Path

import pytest
import torch
from torch.nn import functional as F

from ultralytics import YOLO
from ultralytics.nn.modules.citrus_j6 import SegmentCitrusJ6
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.citrus_j6_loss import assign_cells, visible_points


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(17102026)
    yield
    torch.set_num_threads(previous)


def _head():
    return SegmentCitrusJ6(nm=4, npr=8, ch=(8, 16, 32, 3)).eval()


def _xyxy(xywh):
    return torch.cat((xywh[:, :2] - xywh[:, 2:] / 2, xywh[:, :2] + xywh[:, 2:] / 2), -1).clamp(0, 1)


def _perfect_maps(points, xywh, grid_size, visible_offset=False):
    """Independent label oracle; deliberately can reproduce the former error."""
    height, width = grid_size
    chosen, cells = assign_cells(points, xywh[:, 2:], height, width)
    existence = torch.full((1, 1, height, width), -100.)
    existence.flatten()[cells] = 8.
    offset = torch.full((1, 2, height, width), .5)
    log_size = torch.zeros_like(offset)
    if len(chosen):
        target = points[chosen] if visible_offset else xywh[chosen, :2]
        offset[0].flatten(1)[:, cells] = (
            target * target.new_tensor([width, height]) - torch.stack((cells % width, cells // width), -1)).T
        log_size[0].flatten(1)[:, cells] = xywh[chosen, 2:].log().T
    raw = {"j6": {"discovery": {"existence": existence, "offset": offset, "log_box_size": log_size},
                   "gain": torch.zeros(1, 1, height, width)}}
    return raw, chosen, cells


def _assert_roundtrip(points, xywh, grid_size):
    raw, chosen, cells = _perfect_maps(points, xywh, grid_size)
    boxes, _, ids = _head().discover(raw, 0, threshold=.1)
    assert len(boxes) == len(chosen)
    for output, cell in zip(boxes, ids):
        location = (cells == cell).nonzero(as_tuple=False).flatten()
        assert len(location) == 1
        expected = _xyxy(xywh[chosen[location]])[0]
        torch.testing.assert_close(output, expected, rtol=2e-5, atol=8e-8)
    return raw, boxes, chosen, cells


def _c_mask():
    mask = torch.zeros(1, 64, 64, dtype=torch.bool)
    mask[:, 24:27, 24:40] = True
    mask[:, 37:40, 24:40] = True
    mask[:, 24:40, 24:27] = True
    return mask


def test_c_shaped_visible_point_and_bbox_center_remain_distinct_but_roundtrip():
    xywh = torch.tensor([[.5, .5, .25, .25]])
    points, known = visible_points(_c_mask(), xywh)
    assert known.item() and not torch.allclose(points, xywh[:, :2])
    raw, boxes, chosen, cells = _assert_roundtrip(points, xywh, (32, 32))
    # Positive assignment remains on visible material, not the centre hole.
    assigned_xy = torch.stack((cells % 32, cells // 32), -1)
    assert torch.equal(assigned_xy, (points[chosen] * 32).floor().long())
    assert raw["j6"]["discovery"]["offset"].abs().max() > 1
    old_raw, _, _ = _perfect_maps(points, xywh, (32, 32), visible_offset=True)
    wrong = _head().discover(old_raw, 0, threshold=.1)[0]
    assert not torch.allclose(wrong, boxes)
    intersection = (torch.minimum(wrong[:, 2:], boxes[:, 2:]) -
                    torch.maximum(wrong[:, :2], boxes[:, :2])).clamp_min(0).prod(-1)
    union = (wrong[:, 2:] - wrong[:, :2]).prod(-1) + (boxes[:, 2:] - boxes[:, :2]).prod(-1) - intersection
    assert (intersection / union).item() < .5


@pytest.mark.parametrize("raster_shape,grid_size", [((64, 96), (8, 12)), ((45, 91), (13, 29))])
def test_single_visible_pixel_subcell_boxes_and_non_square_frames(raster_shape, grid_size):
    height, width = raster_shape
    masks = torch.zeros(1, height, width, dtype=torch.bool)
    masks[:, 5, 7] = True
    xywh = torch.tensor([[(7.5 / width), (5.5 / height), 1 / width, 1 / height]])
    points, known = visible_points(masks, xywh)
    assert known.item()
    _, boxes, _, _ = _assert_roundtrip(points, xywh, grid_size)
    assert boxes[0, 2] - boxes[0, 0] < 1 / grid_size[1]
    assert boxes[0, 3] - boxes[0, 1] < 1 / grid_size[0]


def test_non_square_concavity_recovers_original_bbox_not_visible_point():
    masks = torch.zeros(1, 40, 96, dtype=torch.bool)
    masks[:, 5:10, 18:75] = True
    masks[:, 26:32, 18:75] = True
    masks[:, 5:32, 18:22] = True
    xywh = torch.tensor([[(18 + 75) / 192, (5 + 32) / 80, 57 / 96, 27 / 40]])
    points, known = visible_points(masks, xywh)
    assert known.item() and not torch.allclose(points, xywh[:, :2])
    _assert_roundtrip(points, xywh, (13, 29))


def test_collided_visible_cells_keep_distinct_bbox_centres_and_exact_boxes():
    points = torch.full((12, 2), .55)
    centres = torch.stack((torch.linspace(.48, .62, 12), torch.linspace(.61, .49, 12)), -1)
    sizes = torch.stack((torch.linspace(.18, .29, 12), torch.linspace(.22, .33, 12)), -1)
    xywh = torch.cat((centres, sizes), -1)
    raw, _, chosen, cells = _assert_roundtrip(points, xywh, (6, 8))
    assert len(chosen) == 9 and len(cells.unique()) == 9
    offsets = raw["j6"]["discovery"]["offset"][0].flatten(1)[:, cells]
    assert (offsets < 0).any() and (offsets > 1).any()


@pytest.mark.parametrize("xywh", [
    [[.00001, .00002, .00002, .00004]],
    [[.99999, .99998, .00002, .00004]],
    [[.25, .75, .00003, .00002]],
])
def test_extremely_small_positive_boundary_boxes_have_no_one_cell_floor(xywh):
    boxes = torch.tensor(xywh)
    # A vanished raster has a known bbox but no claimed visible-pixel evidence.
    points, nonempty = visible_points(torch.zeros(1, 64, 96, dtype=torch.bool), boxes)
    assert not nonempty.item() and torch.equal(points, boxes[:, :2])
    _, decoded, _, _ = _assert_roundtrip(points, boxes, (32, 48))
    assert (decoded[:, 2:] > decoded[:, :2]).all()
    assert ((decoded >= 0) & (decoded <= 1)).all()


def test_empty_gt_target_assignment_and_decoding():
    xywh = torch.empty(0, 4)
    points, known = visible_points(torch.empty(0, 40, 70, dtype=torch.bool), xywh)
    assert points.shape == (0, 2) and known.shape == (0,)
    _, boxes, chosen, cells = _assert_roundtrip(points, xywh, (7, 13))
    assert boxes.shape == (0, 4) and chosen.numel() == cells.numel() == 0


def test_adjacent_collision_cells_with_unequal_heat_both_survive_within_budget():
    head = _head()
    # Two different fruits may occupy adjacent scout cells. A 3x3 maximum gate
    # would drop the second despite exact box regression and spare budget.
    existence = torch.full((1, 1, 6, 8), -100.)
    existence[0, 0, 2, 3], existence[0, 0, 2, 4] = 8., 7.
    xywh = torch.tensor([[.43, .4, .08, .1], [.57, .4, .07, .09]])
    cells = torch.tensor([19, 20])
    offset = torch.full((1, 2, 6, 8), .5)
    offset[0].flatten(1)[:, cells] = (xywh[:, :2] * torch.tensor([8., 6.]) -
                                     torch.stack((cells % 8, cells // 8), -1)).T
    size = torch.zeros_like(offset)
    size[0].flatten(1)[:, cells] = xywh[:, 2:].log().T
    raw = {"j6": {"discovery": {"existence": existence, "offset": offset, "log_box_size": size},
                   "gain": torch.zeros_like(existence)}}
    boxes, _, ids = head.discover(raw, 0, threshold=.1)
    assert ids.tolist() == cells.tolist()
    torch.testing.assert_close(boxes, _xyxy(xywh), atol=1e-7, rtol=1e-6)


def test_gain_and_source_one_inputs_use_assigned_existence_cell_not_roi_center(monkeypatch):
    model = YOLO(str(ROOT / "0_orange_yaml/J_V6_series/J94_full.yaml"), task="segment", verbose=False).model.train()
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    head = model.model[-1]
    masks = torch.cat((_c_mask().float(), torch.zeros(1, 64, 64)))
    xywh = torch.tensor([[.5, .5, .25, .25]])
    batch = dict(img=torch.rand(2, 3, 128, 128), high_img=torch.rand(2, 3, 256, 256),
                 masks=masks, bboxes=xywh, cls=torch.zeros(1, 1), batch_idx=torch.zeros(1))

    def no_new(raw, image, threshold):
        return xywh.new_empty(0, 4), xywh.new_empty(0), torch.empty(0, dtype=torch.long)

    def gain_map(scout):
        height, width = scout.shape[-2:]
        raster = torch.linspace(-.8, .8, height * width).reshape(1, 1, height, width)
        return raster.expand(scout.shape[0], -1, -1, -1) + scout.sum() * 0

    original_decode, original_smooth = head.decode_rois, F.smooth_l1_loss
    calls, gain_inputs = [], []

    def record_decode(raw, domains, coefficients, has_own, scores, sources):
        result = original_decode(raw, domains, coefficients, has_own, scores, sources)
        calls.append((raw, domains.detach().clone(), coefficients.detach().clone(),
                      scores.detach().clone(), sources.detach().clone(), result))
        return result

    def record_smooth(input, target, *args, **kwargs):
        if input.ndim == 1 and kwargs.get("reduction") == "none":
            gain_inputs.append(input.detach().clone())
        return original_smooth(input, target, *args, **kwargs)

    monkeypatch.setattr(head, "discover", no_new)
    monkeypatch.setattr(head.gain_head, "forward", gain_map)
    monkeypatch.setattr(head, "decode_rois", record_decode)
    monkeypatch.setattr(F, "smooth_l1_loss", record_smooth)
    total, _ = model.loss(batch)
    assert torch.isfinite(total).all() and len(calls) == len(gain_inputs) == 1
    raw, domains, coefficients, scores, sources, decoded = calls[0]
    points, _ = visible_points(_c_mask(), xywh)
    height, width = raw["j6"]["discovery"]["existence"].shape[-2:]
    chosen, cells = assign_cells(points, xywh[:, 2:], height, width)
    new = sources == 1
    assert new.sum() == len(chosen) == 1
    assert torch.equal(coefficients[new], torch.zeros_like(coefficients[new]))
    expected_score = raw["j6"]["discovery"]["existence"].sigmoid().flatten()[cells]
    torch.testing.assert_close(scores[new], expected_score)
    assert torch.equal(decoded["reference_logits"][new], torch.zeros_like(decoded["reference_logits"][new]))
    expected_gain = raw["j6"]["gain"].flatten()[cells].tanh()
    torch.testing.assert_close(gain_inputs[0], expected_gain)
    centre = (domains[:, :, :2] + domains[:, :, 2:]) / 2
    old_roi_gain = F.grid_sample(raw["j6"]["gain"], (2 * centre - 1)[:, None], align_corners=False)[0, 0, 0].tanh()
    assert not torch.allclose(gain_inputs[0], old_roi_gain[new[0]])


def test_unassigned_source_one_collision_has_no_local_quality_gradient(monkeypatch):
    model = YOLO(str(ROOT / "0_orange_yaml/J_V6_series/J94_full.yaml"), task="segment", verbose=False).model.train()
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": False})
    head = model.model[-1]
    # Repeated masks deliberately stress collision bookkeeping; they are not
    # asserted to be a physically possible or valid Citrus annotation example.
    masks = _c_mask().float().expand(12, -1, -1).clone()
    xywh = torch.tensor([[.5, .5, .25, .25]]).expand(12, -1).clone()
    batch = dict(img=torch.rand(2, 3, 128, 128), high_img=torch.rand(2, 3, 256, 256),
                 masks=masks, bboxes=xywh, cls=torch.zeros(12, 1), batch_idx=torch.zeros(12))
    original_decode, calls = head.decode_rois, []

    def no_new(raw, image, threshold):
        return xywh.new_empty(0, 4), xywh.new_empty(0), torch.empty(0, dtype=torch.long)

    def record_decode(*args, **kwargs):
        result = original_decode(*args, **kwargs)
        result["quality_logits"].retain_grad()
        result["logits"].retain_grad()
        calls.append(result)
        return result

    monkeypatch.setattr(head, "discover", no_new)
    monkeypatch.setattr(head, "decode_rois", record_decode)
    total, _ = model.loss(batch)
    total.sum().backward()
    assert model.criterion.last["scout_collision_unresolved"] == 3
    assert model.criterion.last["unassigned_source1"] == 3
    assert len(calls) == 1
    new_slots = torch.arange(1, 24, 2)
    unknown = new_slots[-3:]
    assert torch.equal(calls[0]["quality_logits"].grad[0, unknown], torch.zeros(3))
    assert torch.equal(calls[0]["logits"].grad[0, unknown], torch.zeros(3, 1, 16, 16))
    assert calls[0]["quality_logits"].grad[0, new_slots[:-3]].abs().sum() > 0


def test_actual_loss_offset_targets_bbox_centres_not_assigned_visible_pixels(monkeypatch):
    model = YOLO(str(ROOT / "0_orange_yaml/J_V6_series/J94_full.yaml"), task="segment", verbose=False).model.train()
    model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True})
    masks = torch.cat((_c_mask().float(), torch.zeros(1, 64, 64)))
    xywh = torch.tensor([[.5, .5, .25, .25]])
    batch = dict(img=torch.rand(2, 3, 128, 128), high_img=torch.rand(2, 3, 256, 256),
                 masks=masks, bboxes=xywh, cls=torch.zeros(1, 1), batch_idx=torch.zeros(1))
    original = F.smooth_l1_loss
    captured = []

    def record(input, target, *args, **kwargs):
        if input.ndim == 2 and input.shape[-1] == 2:
            captured.append(target.detach().clone())
        return original(input, target, *args, **kwargs)

    monkeypatch.setattr(F, "smooth_l1_loss", record)
    total, _ = model.loss(batch)
    assert torch.isfinite(total).all() and len(captured) == 2
    points, _ = visible_points(_c_mask(), xywh)
    chosen, cells = assign_cells(points, xywh[:, 2:], 64, 64)
    cell_xy = torch.stack((cells % 64, cells // 64), -1)
    expected = xywh[chosen, :2] * 64 - cell_xy
    old_wrong = points[chosen] * 64 - cell_xy
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)
    assert not torch.allclose(captured[0], old_wrong)
    torch.testing.assert_close(captured[1], xywh[chosen, 2:].log())


def test_geometry_metadata_persists_without_changing_parameter_shapes():
    head = _head()
    assert head.scout_geometry == "bbox_center_offset_v2"
    assert head.discovery.output.weight.shape[0] == 5
    assert head.discovery.output.bias.shape == (5,)
    stream = BytesIO()
    torch.save(head, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)
    assert restored.scout_geometry == "bbox_center_offset_v2"
    assert restored.scout_candidate_policy == "cell_topk_no_peak_nms_v2"
    assert restored.scout_source1_policy == "zero_coeff_cell_heat_v2"
    assert restored.scout_min_box_size == 1e-5
