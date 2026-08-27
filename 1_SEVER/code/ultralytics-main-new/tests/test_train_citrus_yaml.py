"""Safety tests for the official-YAML single-model trainer."""

from types import SimpleNamespace

import pytest

from train_citrus_yaml import custom_loss_overrides


def arguments(**updates):
    """Build the subset of parsed arguments consumed by custom loss resolution."""
    values = {
        "citrus_quality": None,
        "citrus_boundary": None,
        "citrus_query": None,
        "citrus_contrast": None,
        "citrus_exclusive": None,
        "citrus_concavity": None,
        "citrus_vfl": None,
        "nwd_ratio": None,
    }
    values.update(updates)
    return SimpleNamespace(**values)


def test_quality_head_cannot_silently_train_without_quality_supervision():
    """B08/B09-style heads receive their documented default unless explicitly overridden."""
    assert custom_loss_overrides(arguments(), "SegmentCitrusBQuality") == {"citrus_quality": 0.20}
    assert custom_loss_overrides(arguments(citrus_quality=0.35), "SegmentCitrusBQuality") == {
        "citrus_quality": 0.35
    }


def test_topology_heads_receive_the_s_supported_boundary_query_defaults():
    """Standalone topology training must match the B-series batch protocol."""
    expected = {"citrus_boundary": 0.25, "citrus_query": 0.05}
    assert custom_loss_overrides(arguments(), "SegmentCitrusBLite") == expected
    assert custom_loss_overrides(arguments(), "SegmentCitrusLiteBQ") == expected
    assert custom_loss_overrides(arguments(citrus_query=0.02), "SegmentCitrusLiteBQ") == {
        "citrus_boundary": 0.25,
        "citrus_query": 0.02,
    }


def test_other_heads_keep_auxiliary_losses_disabled_and_negative_gains_fail():
    """Legacy models keep stock losses unless requested; invalid gains fail before training."""
    assert custom_loss_overrides(arguments(), "Segment") == {}
    with pytest.raises(ValueError, match="non-negative"):
        custom_loss_overrides(arguments(nwd_ratio=-0.1), "Segment")
