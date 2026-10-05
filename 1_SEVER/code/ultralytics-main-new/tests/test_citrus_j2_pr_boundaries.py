"""Keep observed precision separate from the unchanged native AP sentinel tail."""

import numpy as np

from ultralytics.utils.metrics import compute_ap


def test_native_zero_tail_is_not_an_observed_zero_precision_point():
    recall = np.array([0.2, 0.4, 0.6])
    precision = np.array([1.0, 0.9, 0.8])
    _, envelope, extended_recall = compute_ap(recall, precision)
    assert precision[-1] == 0.8
    assert extended_recall[-2] == recall[-1]
    assert envelope[-2] == 0.0
    np.testing.assert_array_equal(np.interp([0.61, 0.85, 1.0], extended_recall, envelope), [0.0, 0.0, 0.0])


def test_perfect_observed_precision_can_still_have_a_native_zero_pr_tail():
    recall = np.array([0.2, 0.4, 0.6])
    precision = np.ones(3)
    _, envelope, extended_recall = compute_ap(recall, precision)
    assert np.interp(0.59, extended_recall, envelope) == 1.0
    assert np.interp(0.61, extended_recall, envelope) == 0.0
    assert np.all(precision == 1.0)  # No observed prediction has zero precision.
