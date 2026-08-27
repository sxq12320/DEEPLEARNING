"""Tests for the companion PR-curve diagnostic."""

from types import SimpleNamespace

import numpy as np

from diagnose_citrus_pr import summarize_mask_curves


def test_pr_diagnostic_excludes_recall_targets_above_retained_ceiling():
    """Precision-at-recall must be absent when only the COCO sentinel reaches that recall."""
    segment = SimpleNamespace(
        px=np.array([0.0, 0.5, 1.0]),
        p_curve=np.array([[0.90, 0.80, 0.70]]),
        r_curve=np.array([[0.00, 0.60, 0.75]]),
        f1_curve=np.array([[0.00, 0.6857, 0.7241]]),
        prec_values=np.array([[1.00, 0.85, 0.00]]),
    )
    metrics = SimpleNamespace(seg=segment, results_dict={"metrics/mAP50(M)": 0.78})
    summary, rows = summarize_mask_curves(metrics, conf_floor=0.001)

    assert summary["mask_recall_ceiling"] == 0.75
    assert summary["mask_precision_at_recall_0.80"] is None
    assert summary["mask_best_f1_confidence"] == 1.0
    assert rows[-1]["within_supported_recall_range"] is False
