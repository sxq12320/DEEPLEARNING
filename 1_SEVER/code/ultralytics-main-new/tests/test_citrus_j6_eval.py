"""Inference-only interventions must not confound quality and pair suppression."""

from types import SimpleNamespace

import pytest

from eval_citrus_j6 import apply_mode


class Relation:
    def __init__(self, ready=True):
        self.ready = ready

    def mark_trained(self, ready):
        self.ready = bool(ready)


def head(ready=True):
    return SimpleNamespace(enabled=True, selection=True, relation=Relation(ready), gain_route=True, neighbor=True)


def test_quality_only_preserves_decode_quality_and_route():
    model = head()
    apply_mode(model, "quality_only")
    assert not model.relation.ready
    assert model.enabled and model.selection and model.gain_route and model.neighbor


def test_full_never_forces_untrained_relation_ready():
    model = head(False)
    apply_mode(model, "full")
    assert not model.relation.ready
    assert model.enabled


@pytest.mark.parametrize("mode,field", [("global", "enabled"), ("no_gain", "gain_route"),
                                        ("no_neighbor", "neighbor")])
def test_single_inference_intervention_changes_only_its_field(mode, field):
    model = head()
    apply_mode(model, mode)
    for key in ("enabled", "selection", "gain_route", "neighbor"):
        assert getattr(model, key) == (key != field)
    assert model.relation.ready


def test_invalid_mode_and_incompatible_quality_checkpoint_fail():
    with pytest.raises(ValueError):
        apply_mode(head(), "invented")
    model = head()
    model.selection = False
    with pytest.raises(ValueError):
        apply_mode(model, "quality_only")
