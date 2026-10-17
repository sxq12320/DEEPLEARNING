"""Diagnostic contracts, separate from model training and native validation."""

import numpy as np
import torch

from eval_citrus_e_v5 import merge_views
from scripts.diagnose_citrus_20261005 import classify, ev6_pairs, metric_rows, rho_bootstrap, traced_merge
from ultralytics.models.yolo.segment import SegmentationValidator


def test_matching_agrees_with_validator_at_all_thresholds():
    rng = np.random.default_rng(42)
    iou = rng.uniform(size=(7, 11))
    validator = SegmentationValidator(args={"plots": False})
    actual = validator.match_predictions(torch.zeros(11), torch.zeros(7), torch.from_numpy(iou)).numpy()
    for k, threshold in enumerate(validator.iouv.tolist()):
        pairs = ev6_pairs(iou, threshold)
        expected = np.zeros(11, bool)
        expected[pairs[:, 1]] = True
        np.testing.assert_array_equal(actual[:, k], expected)


def test_merge_preserves_existing_paired_masks_and_scores():
    candidates = [dict(box=np.array([0, 0, 3, 3]), score=score, cls=0, patch=np.ones((3, 3), bool),
                       offset=(0, 0), uid=str(k), view=k, internal_border=False)
                  for k, score in enumerate((0.8, 0.7))]
    existing = merge_views(candidates, (4, 4), border_weight=0.5)
    kept, masks, removed = traced_merge(candidates, (4, 4))
    np.testing.assert_array_equal(existing["masks"], masks)
    np.testing.assert_allclose(existing["conf"].numpy(), [c["score"] for c in kept])
    assert removed["1"] == ("cross_view_mask", "0")


def test_mask_tp_is_not_restricted_by_box_gate():
    gt = {"boxes": torch.tensor([[0, 0, 4, 4]], dtype=torch.float32), "masks": np.ones((1, 4, 4), bool)}
    prop = [{"gt_index": 0, "area640": 16}]
    candidate = dict(box=np.array([0, 0, 1, 1]), score=0.9, uid="0:0")
    view = dict(boxes=np.array([[0, 0, 1, 1]]), scores=np.array([0.9]), uids=["0:0"], view=0,
                survivors={0}, capped=set(), candidates=[candidate])
    records, stats, _, _, _ = classify(gt, prop, [view], [candidate], np.ones((1, 4, 4), bool), {}, 0.25)
    assert records[0]["stage"] == "S4" and records[0]["box_gate_exception"]
    assert stats["tiny_hits25"] == 1


def test_undefined_rho_is_not_filled_with_zero():
    row = dict(tiny_targets=1, tiny_hits25=1)
    modes = {m: {"image": row} for m in ("global", "fine", "upsample_linear", "upsample_area")}
    result = rho_bootstrap(modes, {"image": "group"}, repetitions=5)
    assert result["upsample_linear"]["ci95"] is None
    assert result["upsample_linear"]["invalid_denominator_replicates"] == 5


def test_zero_prediction_and_no_gt_are_explicit():
    empty = dict(targets=0, conf=[], tp_m=[], hits25=0, tiny_targets=0, tiny_hits25=0)
    assert metric_rows([empty])["ap5095"] is None
    empty["targets"] = 1
    assert metric_rows([empty])["ap5095"] == 0.0
