"""J V3 contracts: trust codes, loss equivalences, ignore/pseudo semantics, view clipping, mask paint order, tiers."""

import numpy as np
import pytest
import torch

from citrus_j3_batch import SOURCES, settings
from citrus_j3_data import PSEUDO_MODES, J3Format, clip_pseudo, inject_pseudo, instance_code, make_trainer
from citrus_j3_suite import CACHE, FACTORS, NAMES, SUITES, TEACHER_ARM, YAML_DIR, needs_teacher, select_names
from citrus_j3_teacher import assign_tiers, mask_polygon, overlap_matrix
from ultralytics import YOLO
from ultralytics.cfg import get_cfg
from ultralytics.nn.modules import Segment
from ultralytics.utils.citrus_j3_loss import (
    IGNORE_BASE,
    J3SegmentationLoss,
    decode_codes,
    ignore_code,
    pseudo_code,
)
from ultralytics.utils.instance import Instances
from ultralytics.utils.loss import v8SegmentationLoss

torch.set_num_threads(2)
H, W = 96, 128  # network input; masks at input/2 = 48x64
GT_BOX = [0.375, 0.375, 0.25, 0.25]  # xywh normalized -> x 32..64, y 24..48 px
FAR_BOX = [0.8, 0.75, 0.2, 0.3]  # x 89.6..115.2, y 57.6..86.4 px; disjoint from GT_BOX


def build(name="J50_noisy_anchor", overlap=True):
    model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model
    model.args = get_cfg(overrides={"overlap_mask": overlap, "mask_ratio": 2})
    return model.train()


def box_pixels(box):
    """xywh normalized -> mask-grid (48x64) slice."""
    x, y, w, h = box
    return slice(int((y - h / 2) * 48), int((y + h / 2) * 48)), slice(int((x - w / 2) * 64), int((x + w / 2) * 64))


def make_batch(items, overlap=True):
    """items: list of (image, code, box) in batch order; overlap ids follow within-image order (1, 2, ...)."""
    items = sorted(items, key=lambda t: t[0])
    if overlap:
        masks = torch.zeros((2, 48, 64))
        seen = [0, 0]
        for image, _, box in items:
            seen[image] += 1
            ys, xs = box_pixels(box)
            masks[image, ys, xs] = seen[image]
    else:
        masks = torch.zeros((len(items), 48, 64))
        for k, (_, _, box) in enumerate(items):
            ys, xs = box_pixels(box)
            masks[k, ys, xs] = 1
    return dict(
        img=torch.rand(2, 3, H, W, generator=torch.Generator().manual_seed(0)),
        masks=masks,
        batch_idx=torch.tensor([float(i) for i, _, _ in items]),
        cls=torch.tensor([[c] for _, c, _ in items], dtype=torch.float32).view(-1, 1),
        bboxes=torch.tensor([b for _, _, b in items], dtype=torch.float32).view(-1, 4),
    )


def losses(model, batch, criterion_cls):
    torch.manual_seed(0)
    preds = model(batch["img"])
    criterion = criterion_cls(model)
    total, items = criterion(preds, batch)
    return items, criterion


def test_codes_roundtrip_and_validation():
    codes = np.array([0.0, pseudo_code(1.0), pseudo_code(0.3), ignore_code(0.0), ignore_code(0.9)])
    kind, weight = decode_codes(codes)
    assert kind.tolist() == [0, 1, 1, 2, 2]
    assert weight == pytest.approx([1.0, 1.0, 0.3, 0.0, 0.0])
    tk, tw = decode_codes(torch.tensor(codes, dtype=torch.float32))
    assert tk.tolist() == kind.tolist() and torch.allclose(tw, torch.tensor(weight, dtype=torch.float32))
    with pytest.raises(ValueError):
        pseudo_code(0.0)
    assert ignore_code(7.0) == IGNORE_BASE + 1.0


@pytest.mark.parametrize("overlap", [True, False])
def test_gt_only_loss_identical_to_stock(overlap):
    model = build(overlap=overlap)
    batch = make_batch([(0, 0.0, GT_BOX), (1, 0.0, [0.5, 0.5, 0.0625, 0.0833]), (1, 0.0, FAR_BOX)], overlap)
    stock, _ = losses(model, batch, v8SegmentationLoss)
    ours, _ = losses(model, batch, J3SegmentationLoss)
    torch.testing.assert_close(ours, stock, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize("overlap", [True, False])
def test_ignore_region_only_removes_negatives_and_keeps_mask_ids(overlap):
    model = build(overlap=overlap)
    # Ignore region FIRST in batch order: in overlap mode it owns id 1 and the GT id 2 (remap must hold).
    with_ignore = make_batch([(0, ignore_code(0.7), FAR_BOX), (0, 0.0, GT_BOX)], overlap)
    gt_only = make_batch([(0, 0.0, GT_BOX)], overlap)
    ours, criterion = losses(model, with_ignore, J3SegmentationLoss)
    stock, _ = losses(model, gt_only, v8SegmentationLoss)
    torch.testing.assert_close(ours[[0, 1, 3]], stock[[0, 1, 3]], rtol=1e-6, atol=1e-7)  # box, seg, dfl
    assert ours[2] < stock[2]  # background BCE inside the ignore box is no longer charged
    assert criterion.stats["ignore_regions"] == 1 and criterion.stats["ignored_anchors"] > 0


def test_pseudo_weight_one_equals_ground_truth_and_soft_downweights():
    model = build()
    as_gt = make_batch([(0, 0.0, GT_BOX), (0, 0.0, FAR_BOX)])
    hard = make_batch([(0, 0.0, GT_BOX), (0, pseudo_code(1.0), FAR_BOX)])
    soft = make_batch([(0, 0.0, GT_BOX), (0, pseudo_code(0.3), FAR_BOX)])
    reference, _ = losses(model, as_gt, J3SegmentationLoss)
    hard_items, criterion = losses(model, hard, J3SegmentationLoss)
    soft_items, _ = losses(model, soft, J3SegmentationLoss)
    torch.testing.assert_close(hard_items, reference, rtol=1e-6, atol=1e-7)
    assert criterion.stats["pseudo"] == 1
    assert not torch.allclose(soft_items, reference)
    assert soft_items[1] < reference[1]  # mask term is weighted per positive anchor


def test_empty_and_ignore_only_batches_are_finite():
    model = build()
    empty = dict(img=torch.rand(2, 3, H, W), masks=torch.zeros(2, 48, 64), batch_idx=torch.zeros(0),
                 cls=torch.zeros(0, 1), bboxes=torch.zeros(0, 4))
    for batch in (empty, make_batch([(1, ignore_code(0.5), FAR_BOX)])):
        items, _ = losses(model, batch, J3SegmentationLoss)
        assert torch.isfinite(items).all()


def test_unsupported_auxiliary_terms_refused():
    model = build()
    model.args = get_cfg(overrides={"overlap_mask": True, "mask_ratio": 2})
    model.args.citrus_exclusive = 0.1
    with pytest.raises(ValueError):
        J3SegmentationLoss(model)


def test_format_paints_ignore_lowest_and_reorders_codes():
    fmt = J3Format(bbox_format="xywh", normalize=True, return_mask=True, mask_ratio=1, mask_overlap=True)
    big = np.array([[0, 0], [40, 0], [40, 40], [0, 40]], np.float32)  # ignore, largest
    small = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32)  # GT inside the ignore region
    mid = np.array([[15, 15], [35, 15], [35, 35], [15, 35]], np.float32)  # pseudo overlapping both
    segments = np.stack([big, small, mid])
    boxes = np.array([[0, 0, 40, 40], [10, 10, 20, 20], [15, 15, 35, 35]], np.float32)
    instances = Instances(boxes, segments, bbox_format="xyxy", normalized=False)
    cls = np.array([[ignore_code(0.4)], [0.0], [pseudo_code(0.8)]], np.float32)
    canvas, _, ordered = fmt._format_segments(instances, cls, 48, 48)
    kind = decode_codes(ordered.reshape(-1))[0].tolist()
    assert kind == [2, 1, 0]  # ignore first; positives by decreasing area (pseudo 400 px, GT 100 px)
    assert (canvas[0, 12:18, 12:18] == 3).all()  # GT keeps its pixels against both
    assert (canvas[0, 25:33, 25:33] == 2).all()  # pseudo beats ignore
    assert (canvas[0, 2:8, 2:8] == 1).all()


def test_clip_pseudo_global_window_and_disconnected():
    square = [[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]
    assert clip_pseudo(square, (100, 100), (0, 0, 100, 100))[0] == "polygon"
    kind, verts = clip_pseudo(square, (100, 100), (20, 0, 60, 40))
    assert kind == "polygon"
    assert verts[:, 0].min() == pytest.approx(0.0) and verts[:, 0].max() == pytest.approx(0.25)
    assert clip_pseudo(square, (100, 100), (50, 50, 90, 90)) is None
    u = [[0.0, 0.0], [0.6, 0.0], [0.6, 0.6], [0.4, 0.6], [0.4, 0.2], [0.2, 0.2], [0.2, 0.6], [0.0, 0.6]]
    kind, rect = clip_pseudo(u, (100, 100), (0, 30, 100, 70))  # cuts both U legs -> two pieces
    assert kind == "box" and len(rect) == 4


def test_instance_code_policies():
    assert instance_code("high", 0.9, "none") is None
    assert decode_codes([instance_code("high", 0.9, "ignore")])[0][0] == 2
    assert decode_codes([instance_code("mid", 0.9, "hard")])[0][0] == 2
    assert decode_codes([instance_code("high", 0.9, "hard")])[1][0] == pytest.approx(1.0)
    assert decode_codes([instance_code("high", 0.7, "soft")])[1][0] == pytest.approx(0.7)
    assert decode_codes([instance_code("mid", 0.3, "naive")])[1][0] == pytest.approx(1.0)
    assert set(PSEUDO_MODES) == {"none", "ignore", "hard", "soft", "naive"}


def test_inject_pseudo_into_views(tmp_path):
    source = str((tmp_path / "src.jpg").resolve())
    groups = [dict(source=source, original_shape=[100, 100], views=[
        dict(image="v0.png", window=[0, 0, 100, 100], global_view=True),
        dict(image="v1.png", window=[0, 0, 40, 40], global_view=False),
        dict(image="v2.png", window=[60, 60, 100, 100], global_view=False),
    ])]
    files = [str(tmp_path / f"v{i}.png") for i in range(3)]
    labels = [dict(cls=np.zeros((1, 1), np.float32), bboxes=np.array([[0.5, 0.5, 0.1, 0.1]], np.float32),
                   segments=[np.array([[0.45, 0.45], [0.55, 0.45], [0.55, 0.55]], np.float32)]) for _ in files]
    pseudo = {"images": {source: {"instances": [
        dict(tier="high", score=0.8, poly=[[0.1, 0.1], [0.3, 0.1], [0.3, 0.3], [0.1, 0.3]]),
        dict(tier="mid", score=0.4, poly=[[0.65, 0.65], [0.8, 0.65], [0.8, 0.8], [0.65, 0.8]]),
    ]}}}
    summary = inject_pseudo(labels, files, tmp_path, groups, pseudo, "soft", imgsz=640)
    assert [len(lb["cls"]) for lb in labels] == [3, 2, 2]
    assert len(labels[0]["segments"]) == len(labels[0]["cls"]) == len(labels[0]["bboxes"])
    kinds = decode_codes(labels[0]["cls"].reshape(-1))[0].tolist()
    assert kinds == [0, 1, 2]
    assert decode_codes(labels[1]["cls"].reshape(-1))[1][1] == pytest.approx(0.8)
    assert decode_codes(labels[2]["cls"].reshape(-1))[0][1] == 2
    assert summary["positive"] == 2 and summary["ignore"] == 2 and summary["views_touched"] == 3
    untouched = [dict(cls=np.zeros((0, 1), np.float32), bboxes=np.zeros((0, 4), np.float32), segments=[])]
    assert inject_pseudo(untouched, files[:1], tmp_path, groups, pseudo, "none")["views_touched"] == 0


def test_tiers_exclusion_corroboration_and_polygon():
    def disk(cy, cx, r, shape=(60, 60)):
        yy, xx = np.mgrid[: shape[0], : shape[1]]
        return (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r

    labelled, missing, lonely, weak = disk(10, 10, 6), disk(30, 30, 6), disk(48, 12, 5), disk(48, 48, 5)
    masks = np.stack([labelled, missing, lonely, weak])
    scores = np.array([0.95, 0.9, 0.9, 0.3])
    gt = np.stack([disk(10, 11, 6)])
    other_view = np.stack([disk(30, 31, 6)])  # corroborates only the "missing" fruit
    tiers = assign_tiers(scores, masks, gt, [other_view])
    assert tiers[0]["excluded_by_gt"] and tiers[0]["tier"] is None
    assert tiers[1]["tier"] == "high" and tiers[1]["votes"] == 2
    assert tiers[2]["tier"] == "mid"  # confident but uncorroborated
    assert tiers[3]["tier"] == "mid"  # >= TAU_LOW, < TAU_HIGH
    assert assign_tiers(np.array([0.1]), masks[1:2], gt, [])[0]["tier"] is None
    iou, cover_a, _ = overlap_matrix(masks[:1], gt)
    assert 0.7 < iou[0, 0] < 1 and cover_a[0, 0] > 0.8
    poly = np.asarray(mask_polygon(missing, (60, 60)))
    assert len(poly) >= 3 and abs(poly[:, 0].mean() - 30.5 / 60) < 0.03


@pytest.mark.parametrize("name", NAMES)
def test_every_arm_deploys_the_j20_graph(name):
    reference = YOLO(str(YAML_DIR.parent / "J_V2_series/J20_glance_anchor.yaml"), task="segment", verbose=False).model
    model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model
    assert type(model.model[-1]) is Segment
    ref, sd = reference.state_dict(), model.state_dict()
    assert ref.keys() == sd.keys() and all(ref[k].shape == sd[k].shape for k in ref)


def test_generated_yamls_are_current():
    from scripts.generate_citrus_j3_yaml import main

    main(check=True)


def test_suite_and_runner_contracts():
    assert NAMES[0] == TEACHER_ARM and not needs_teacher(TEACHER_ARM)
    assert set(select_names("all")) == set(FACTORS) == set(NAMES)
    assert all(FACTORS[n]["teacher"] in (None, "fine", "global") for n in NAMES)
    assert all((FACTORS[n]["pseudo"] == "none") == (FACTORS[n]["teacher"] is None) for n in NAMES)
    assert set(SUITES["ladder"]) | set(SUITES["controls"]) == set(SUITES["main"])
    assert CACHE["J56_privileged_soft_960"] is False and CACHE["J53_privileged_soft"] is True
    with pytest.raises(ValueError):
        select_names(only="J53_privileged_soft,J53_privileged_soft")
    args = settings(50, 42)
    assert args["pretrained"] is False and args["amp"] is False and args["mask_ratio"] == 2
    assert "ultralytics/utils/citrus_j3_loss.py" in SOURCES
    with pytest.raises(ValueError):
        make_trainer(FACTORS["J53_privileged_soft"], None)
    assert make_trainer(FACTORS["J50_noisy_anchor"]).__name__ == "J3Trainer_none"
