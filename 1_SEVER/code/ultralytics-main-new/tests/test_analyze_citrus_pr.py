"""Unit tests for citrus PR and confusion-matrix diagnostics."""

from types import SimpleNamespace

import numpy as np
import pytest

from analyze_citrus_pr import summarize_mask_curves, summarize_operating_counts, summarize_raw_box_confusion


def test_raw_single_class_confusion_is_not_misread_as_background_accuracy():
    """The right-top cell is an FP count, not a true-background accuracy denominator."""
    summary = summarize_raw_box_confusion(np.array([[1676, 410], [422, 0]]))
    assert summary["box_cm45_tp"] == 1676
    assert summary["box_cm45_fp"] == 410
    assert summary["box_cm45_fn"] == 422
    assert summary["box_cm45_precision"] == pytest.approx(1676 / (1676 + 410))
    assert summary["box_cm45_recall"] == pytest.approx(1676 / (1676 + 422))


def test_operating_counts_include_mask_false_positives_and_false_negatives():
    """Raw validator matches must expose mask errors without relying on a box-only confusion plot."""
    stats = {
        "conf": [np.array([0.9, 0.6, 0.2])],
        "target_cls": [np.array([0, 0, 0])],
        "tp": [np.array([[1, 1], [1, 0], [0, 0]], dtype=bool)],
        "tp_m": [np.array([[1, 1], [0, 0], [0, 0]], dtype=bool)],
    }
    summary = summarize_operating_counts(stats, confidence=0.25)
    assert (summary["box_op50_tp"], summary["box_op50_fp"], summary["box_op50_fn"]) == (2, 0, 1)
    assert (summary["mask_op50_tp"], summary["mask_op50_fp"], summary["mask_op50_fn"]) == (1, 1, 2)


def test_mask_curve_summary_reports_supported_recall_and_operating_point():
    """Recall support and best-F1 confidence must come from confidence curves, not padded PR zeros."""
    grid = np.linspace(0, 1, 1000)
    precision = (1.0 - 0.3 * grid)[None]
    recall = (0.88 * (1.0 - grid))[None]
    f1 = 2 * precision * recall / (precision + recall + 1e-16)
    metric = SimpleNamespace(
        p_curve=precision,
        r_curve=recall,
        f1_curve=f1,
        px=grid,
        prec_values=np.full((1, 1000), 0.8),
    )
    summary = summarize_mask_curves(metric)
    assert summary["mask_recall_ceiling_at_val_prefilter"] == pytest.approx(0.88)
    assert summary["mask_unmatched_fraction_at_val_prefilter"] == pytest.approx(0.12)
    assert 0 <= summary["mask_best_f1_conf"] <= 1
    assert summary["mask_precision_at_recall_085"] == pytest.approx(0.8)
