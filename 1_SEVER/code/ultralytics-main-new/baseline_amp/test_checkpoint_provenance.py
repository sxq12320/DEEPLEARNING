"""CPU-only final-evaluation provenance and original-result protection contracts."""

import contextlib
import json
import sys
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common  # noqa: E402
import inference_runtime  # noqa: E402
import validation  # noqa: E402
import validation_adapters as adapters  # noqa: E402
from registry import COMMON_VALIDATION  # noqa: E402


@pytest.fixture
def evaluation_probe(tmp_path, monkeypatch):
    """Exercise the real final-evaluation metadata without CUDA or framework loading."""
    prepared, run = tmp_path / "prepared", tmp_path / "run"
    prepared.mkdir()
    (prepared / "summary.json").write_text(json.dumps(dict(names=["orange_immature"])), encoding="utf-8")
    (run / "validation_common").mkdir(parents=True)
    checkpoint = run / "best.pt"
    checkpoint.touch()
    selection = dict(epoch=10, mask_ap_50_95=0.8, checkpoint=str(checkpoint), weight_policy="ema")
    (run / "validation_common/best.json").write_text(json.dumps(selection), encoding="utf-8")
    for name in ("val_common.json", "evaluation_timing.json", "complete.json"):
        (run / name).write_text(json.dumps(dict(existing_result=name)), encoding="utf-8")
    originals = {path: path.read_bytes() for path in run.rglob("*") if path.is_file()}
    probe = types.SimpleNamespace(prepared=prepared, run=run, checkpoint=checkpoint, originals=originals,
                                  score=0.8, calls=[], output=tmp_path / "reevaluation")

    class Model:
        def float(self):
            return self

        def eval(self):
            return self

        def to(self, device):
            return self

        def parameters(self):
            return iter([types.SimpleNamespace(numel=lambda: 7)])

    def collect(*args, profile_images=0):
        probe.calls.append(profile_images)
        metrics = dict(seconds=1.0, images=1, timing={}, per_image_timing=[], validation_engine="cpu_probe")
        if profile_images:
            metrics["profile_only"] = True
        else:
            metrics["mask_ap_50_95"] = probe.score
        return metrics

    monkeypatch.setitem(sys.modules, "ultralytics", types.SimpleNamespace(
        YOLO=lambda *args, **kwargs: types.SimpleNamespace(model=Model())))
    monkeypatch.setattr(common, "seed_everything", lambda seed: None)
    monkeypatch.setattr(inference_runtime, "install_inference_runtime", lambda: None)
    monkeypatch.setattr(inference_runtime, "runtime_status", lambda: {})
    monkeypatch.setattr(validation, "validation_state", lambda model: contextlib.nullcontext())
    monkeypatch.setattr(validation, "collect_metrics", collect)
    monkeypatch.setattr(adapters, "yolo_predictor", lambda *args: None)
    probe.job = dict(seed=42, validation=COMMON_VALIDATION, recipe=dict(family="yolo"))
    return probe


def evaluate(probe, checkpoint=None, output=None, profile_images=0):
    return adapters.evaluate_unified(probe.job, probe.prepared, probe.run,
                                     checkpoint or probe.checkpoint, output or probe.output,
                                     profile_images=profile_images)


def assert_originals_unchanged(probe):
    assert {path: path.read_bytes() for path in probe.run.rglob("*") if path.is_file()} == probe.originals


def test_selected_checkpoint_retains_selection_provenance(evaluation_probe):
    probe = evaluation_probe
    result = evaluate(probe)
    assert result["is_selected_checkpoint"] is True and result["checkpoint_role"] == "selected"
    assert result["selected_epoch"] == 10 and result["weight_policy"] == "ema"
    assert json.loads((probe.output / "val_common.json").read_text()) == result
    assert not (probe.output / "complete.json").exists()
    assert_originals_unchanged(probe)


def test_selected_checkpoint_ap_reproduction_guard_remains(evaluation_probe):
    probe = evaluation_probe
    probe.score = 0.2
    with pytest.raises(RuntimeError, match="reproduce selection AP"):
        evaluate(probe)
    assert not probe.output.exists()
    assert_originals_unchanged(probe)


def test_custom_checkpoint_does_not_inherit_best_provenance(evaluation_probe):
    probe = evaluation_probe
    probe.score = 0.2
    result = evaluate(probe, checkpoint=probe.run / "different_epoch_last.pt")
    assert result["mask_ap_50_95"] == 0.2
    assert result["is_selected_checkpoint"] is False and result["checkpoint_role"] == "custom"
    assert result["selected_epoch"] is None and result["weight_policy"] is None
    assert not (probe.output / "complete.json").exists()
    assert_originals_unchanged(probe)


def test_custom_checkpoint_does_not_require_a_best_marker(evaluation_probe):
    probe = evaluation_probe
    marker = probe.run / "validation_common/best.json"
    marker.unlink()
    probe.originals.pop(marker)
    result = evaluate(probe, checkpoint=probe.run / "interrupted_checkpoint.pt")
    assert result["is_selected_checkpoint"] is False and result["selected_epoch"] is None
    assert_originals_unchanged(probe)


@pytest.mark.parametrize("profile_images", [0, 3])
def test_custom_checkpoint_and_profile_cannot_overwrite_run(evaluation_probe, profile_images):
    probe = evaluation_probe
    with pytest.raises(ValueError, match="separate output directory"):
        evaluate(probe, checkpoint=probe.run / "custom.pt", output=probe.run, profile_images=profile_images)
    assert probe.calls == []
    assert_originals_unchanged(probe)


def test_profile_has_no_ap_selection_or_completed_result(evaluation_probe):
    probe = evaluation_probe
    result = evaluate(probe, checkpoint=probe.run / "custom.pt", profile_images=3)
    assert probe.calls == [3] and result["profile_only"] is True
    assert "mask_ap_50_95" not in result and "selected_epoch" not in result
    assert (probe.output / "evaluation_timing.json").is_file()
    assert not (probe.output / "val_common.json").exists()
    assert not (probe.output / "complete.json").exists()
    assert_originals_unchanged(probe)
