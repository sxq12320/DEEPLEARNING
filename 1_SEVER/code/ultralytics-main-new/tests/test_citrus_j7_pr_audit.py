"""Analytic PR-prefix tests; no trained-model accuracy assertion."""

import copy
import json
from types import SimpleNamespace

import pytest
import torch
import numpy as np

from citrus_e_v6_training import EV6Validator
from citrus_j7_pr_audit import J7FinalAuditValidator, audit_mask_matches


@pytest.fixture(autouse=True)
def threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def audit(flags, confidence, pred=None, gt=None, names=None):
    matrix = np.zeros((len(flags), 10), bool)
    matrix[:, 0] = flags
    return audit_mask_matches(matrix, np.asarray(confidence),
                              np.asarray(pred if pred is not None else [0] * len(flags)),
                              np.asarray(gt if gt is not None else [0] * sum(flags)),
                              names or {0: "fruit"})


def test_known_raw_curve_and_no_sentinel():
    report = audit([1, 1, 0, 1, 0], [.99, .95, .90, .80, .10], gt=[0] * 4)
    row, = report["classes"]
    assert row["max_reachable_recall"] == .75
    assert row["unreachable_recall_fraction"] == .25
    assert row["recall_at_precision90"]["recall"] == .5
    assert row["recall_at_precision90"]["threshold"] == .95
    assert row["score_tail_last20_groups"][-1]["precision"] == .6
    assert row["score_tail_last20_groups"][-1]["recall"] == .75
    assert report["compute_ap_sentinel_included"] is False
    assert report["interpolated_for_operating_points"] is False


def test_confidence_ties_cannot_choose_only_true_positive_prefix():
    for flags in ([1, 0], [0, 1]):
        row, = audit(flags, [.9, .9], gt=[0])["classes"]
        assert row["max_reachable_recall"] == 1
        assert row["recall_at_precision90"]["recall"] == 0
        assert row["recall_at_precision90"]["eligible_threshold_found"] is False
        assert row["threshold_group_count"] == 1
        assert row["score_tail_last20_groups"][0]["count"] == 2


def test_most_recall_at_valid_precision_uses_full_groups():
    row, = audit([1, 1, 0, 1], [.9, .8, .8, .7], gt=[0] * 3)["classes"]
    assert row["recall_at_precision95"]["recall"] == pytest.approx(1 / 3)
    assert row["recall_at_precision95"]["threshold"] == .9
    assert row["threshold_group_count"] == 3


@pytest.mark.parametrize("flags,confidence,gt,expected", [([], [], [0], 0), ([0], [.5], [0], 0),
                                                       ([], [], [], None), ([0], [.5], [], None)])
def test_empty_no_tp_and_no_gt_are_honest(flags, confidence, gt, expected):
    report = audit(flags, confidence, gt=gt)
    row, = report["classes"]
    assert row["max_reachable_recall"] == expected
    assert row["recall_at_precision95"]["recall"] == expected
    assert row["recall_at_precision95"]["threshold"] is None
    json.dumps(report, allow_nan=False)


def test_multiclass_counts_and_tail_are_per_class():
    report = audit([1, 0, 1, 0], [.9, .8, .7, .6], pred=[2, 0, 0, 2], gt=[0, 0, 2],
                   names={0: "a", 2: "b"})
    a, b = report["classes"]
    assert a["gt_count"] == 2 and a["max_reachable_recall"] == .5
    assert b["gt_count"] == 1 and b["max_reachable_recall"] == 1
    assert b["recall_at_precision95"]["threshold"] == .9
    row, = audit([0] * 25, np.linspace(.99, .01, 25), gt=[0])["classes"]
    assert row["threshold_group_count"] == 25
    assert len(row["score_tail_last20_groups"]) == 20 and row["score_tail_fp"] == 20


@pytest.mark.parametrize("mutation", ["length", "nan", "fractional_class", "unknown_pred", "unknown_gt",
                                     "score_range", "bad_flag", "more_tp_than_gt"])
def test_invalid_input_rejected(mutation):
    values = dict(tp_m=np.zeros((2, 10)), conf=np.asarray([.9, .8]), pred_cls=np.asarray([0., 0.]),
                  target_cls=np.asarray([0.]), names={0: "fruit"})
    if mutation == "length":
        values["conf"] = np.asarray([.9])
    elif mutation == "nan":
        values["conf"][0] = np.nan
    elif mutation == "fractional_class":
        values["pred_cls"][0] = .1
    elif mutation == "unknown_pred":
        values["pred_cls"][0] = 1
    elif mutation == "unknown_gt":
        values["target_cls"][0] = 1
    elif mutation == "score_range":
        values["conf"][0] = 1.1
    elif mutation == "bad_flag":
        values["tp_m"][0, 0] = .5
    else:
        values["tp_m"][:, 0] = 1
    with pytest.raises(ValueError):
        audit_mask_matches(**values)


def test_final_validator_snapshots_before_parent_clears_without_mutation(monkeypatch, tmp_path):
    validator = object.__new__(J7FinalAuditValidator)
    stats = dict(tp_m=[np.asarray([[True] + [False] * 9])], conf=[np.asarray([.9])],
                 pred_cls=[np.asarray([0.])], target_cls=[np.asarray([0., 0.])])
    original = copy.deepcopy(stats)
    validator.metrics = SimpleNamespace(stats=stats)
    validator.names, validator.iouv = {0: "fruit"}, torch.linspace(.5, .95, 10)
    validator.args = SimpleNamespace(conf=.001, iou=.7, max_det=300)
    validator.save_dir = tmp_path
    calls = []

    def parent(instance):
        calls.append(instance)
        for name in original:
            assert isinstance(stats[name], list)
            for actual, expected in zip(stats[name], original[name]):
                np.testing.assert_array_equal(actual, expected)
            stats[name].clear()
        return {"official_parent_metric": .123}

    monkeypatch.setattr(EV6Validator, "get_stats", parent)
    assert validator.get_stats() == {"official_parent_metric": .123}
    assert calls == [validator] and not any(stats.values())
    raw = np.load(tmp_path / "raw_mask_matches.npz", allow_pickle=False)
    np.testing.assert_array_equal(raw["tp_m"], original["tp_m"][0])
    np.testing.assert_array_equal(raw["target_cls"], original["target_cls"][0])
    report = json.loads((tmp_path / "mask_pr_audit.json").read_text("utf-8"))
    assert report["classes"][0]["max_reachable_recall"] == .5
    assert report["protocol"]["confidence_lower_bound"] == .001
    assert report["protocol"]["max_det"] == 300
    with pytest.raises(FileExistsError):
        validator.get_stats()
    assert calls == [validator]


def test_audit_cannot_silently_use_wrong_iou_column(tmp_path):
    validator = object.__new__(J7FinalAuditValidator)
    validator.iouv = torch.linspace(.6, .95, 10)
    with pytest.raises(ValueError, match="IoU=.5"):
        validator.get_stats()
