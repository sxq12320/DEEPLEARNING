"""Keep J's validation identical to I's native input/2 mask protocol."""

import builtins
import inspect
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import RUN_CITRUS_J as launcher
import citrus_j_batch as batch
from citrus_e_v6_training import EV6TrainingTrainer, EV6Validator
from citrus_protocol import fixed_train_args
from ultralytics.cfg import get_cfg
from ultralytics.models.yolo.segment import SegmentationTrainer


def test_launcher_and_api_default_to_native_validation():
    assert launcher.COMMON_VALIDATION is False
    assert launcher.DIAGNOSE_PR is False
    assert inspect.signature(batch.run).parameters["common_validation"].default is False
    assert inspect.signature(batch.run).parameters["diagnose_pr"].default is False
    assert "NATIVEVAL" in launcher.PROJECT
    assert "baseline_amp" not in inspect.getsource(batch)


def test_recipe_restores_plots_without_changing_training_hyperparameters():
    historical = fixed_train_args()
    historical.update(
        epochs=300, seed=43, pretrained=False, amp=False, cache=True,
        mask_ratio=2, copy_paste=0.3, patience=301, plots=False,
    )
    native = batch.settings(300, 43)
    for key, value in historical.items():
        if key != "plots":
            assert native[key] == value, key
    assert native["plots"] is True and native["val"] is True
    assert native["conf"] == 0.001 and native["iou"] == 0.7
    assert native["max_det"] == 300 and native["half"] is False
    assert native["save_json"] is False and native["save_txt"] is False


@pytest.mark.parametrize("source_balanced", [True, False])
def test_actual_validator_is_the_i_series_validator(source_balanced, tmp_path):
    trainer_type = batch.make_native_trainer(source_balanced)
    if source_balanced:
        assert trainer_type is EV6TrainingTrainer
    assert trainer_type.get_validator is EV6TrainingTrainer.get_validator
    trainer = trainer_type.__new__(trainer_type)
    trainer.args = get_cfg(overrides=batch.settings(1, 42))
    trainer.test_loader = None
    trainer.save_dir = tmp_path
    trainer.callbacks = {}
    validator = trainer.get_validator()
    assert type(validator) is EV6Validator
    assert validator.evaluation_mask_ratio == validator.args.mask_ratio == 2
    assert validator.args.conf == 0.001 and validator.args.iou == 0.7
    assert validator.args.max_det == 300 and validator.args.plots is True


def test_plain_debug_route_changes_only_training_dataset(monkeypatch):
    calls = []

    def plain(self, path, mode="train", batch=None):
        calls.append(("plain", mode))
        return "plain_train"

    def historical(self, path, mode="train", batch=None):
        calls.append(("historical", mode))
        return "historical_val"

    monkeypatch.setattr(SegmentationTrainer, "build_dataset", plain)
    monkeypatch.setattr(EV6TrainingTrainer, "build_dataset", historical)
    trainer_type = batch.make_native_trainer(False)
    trainer = trainer_type.__new__(trainer_type)
    assert trainer.build_dataset("unused", "train", 2) == "plain_train"
    assert trainer.build_dataset("unused", "val", 2) == "historical_val"
    assert calls == [("plain", "train"), ("historical", "val")]


def test_legacy_common_flag_cannot_reroute_to_baselines(tmp_path):
    with pytest.raises(ValueError, match="EV6Validator"):
        batch.run(tmp_path / "missing.yaml", "cpu", tmp_path / "out", common_validation=True)
    assert not (tmp_path / "out").exists()


def test_mask_snapshots_preserve_native_best_and_separate_selection(tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    last, best = weights / "last.pt", weights / "best.pt"
    last.write_bytes(b"epoch1")
    best.write_bytes(b"official_native_best")
    trainer = SimpleNamespace(
        save_dir=tmp_path, last=last, epoch=0,
        metrics={"metrics/mAP50-95(M)": 0.3, "metrics/mAP50(M)": 0.7},
    )
    batch.save_mask_checkpoints(trainer)
    last.write_bytes(b"epoch2")
    trainer.epoch = 1
    trainer.metrics = {"metrics/mAP50-95(M)": 0.4, "metrics/mAP50(M)": 0.6}
    batch.save_mask_checkpoints(trainer)
    assert best.read_bytes() == b"official_native_best"
    assert (weights / "best_mask.pt").read_bytes() == b"epoch2"
    assert (weights / "best_mask50.pt").read_bytes() == b"epoch1"
    assert json.loads((tmp_path / "best_mask_selection.json").read_text())["epoch"] == 2
    assert json.loads((tmp_path / "best_mask50_selection.json").read_text())["epoch"] == 1
    last.write_bytes(b"tie_must_not_replace")
    trainer.epoch = 2
    batch.save_mask_checkpoints(trainer)
    assert (weights / "best_mask.pt").read_bytes() == b"epoch2"


@pytest.mark.parametrize("source_balanced", [True, False])
def test_queue_has_no_baseline_import_or_extra_default_validation(monkeypatch, tmp_path, source_balanced):
    import citrus_e_v5_slicing
    import ultralytics

    calls = []

    class FakeYOLO:
        def __init__(self, model, task="segment"):
            self.callbacks = {}

        def add_callback(self, event, callback):
            self.callbacks.setdefault(event, []).append(callback)

        def train(self, **kwargs):
            calls.append(kwargs)
            directory = Path(kwargs["project"]) / kwargs["name"]
            weights = directory / "weights"
            weights.mkdir(parents=True)
            last = weights / "last.pt"
            last.write_bytes(b"mock_ema")
            (weights / "best.pt").write_bytes(b"mock_native_best")
            trainer = SimpleNamespace(
                last=last, save_dir=directory, epoch=0,
                metrics={"metrics/mAP50-95(M)": 0.3, "metrics/mAP50(M)": 0.7},
            )
            for callback in self.callbacks["on_model_save"]:
                callback(trainer)

        def val(self, **kwargs):
            pytest.fail("Default J queue must not add a second validation pass")

    real_import = builtins.__import__

    def no_baselines(name, *args, **kwargs):
        if name.startswith("baseline_amp") or name in {"validation_adapters", "prepare"}:
            pytest.fail("J attempted to import a baseline evaluator: " + name)
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(batch, "source_digest", lambda: "test_only_source_digest")
    monkeypatch.setattr(citrus_e_v5_slicing, "prepare_multiscale_views", lambda data, output, imgsz: data)
    monkeypatch.setattr(builtins, "__import__", no_baselines)
    data = tmp_path / "data.yaml"
    data.write_text("train: train/images\nval: val/images\nnames: [orange_immature]\n", encoding="utf-8")
    project = tmp_path / "out"
    batch.run(data, "cpu", project, epochs=1, only="J00_yolo11n_control", source_balanced=source_balanced)
    assert len(calls) == 1
    assert calls[0]["trainer"].get_validator is EV6TrainingTrainer.get_validator
    assert calls[0]["plots"] is True and calls[0]["max_det"] == 300
    assert calls[0]["exist_ok"] is False
    assert not (project / "_common_data").exists()
    directory = project / "J00_yolo11n_control_seed42"
    record = json.loads((directory / "j_completed.json").read_text())
    assert record["common_validation"] is False and record["diagnose_pr"] is False
    assert record["validation"]["protocol"] == "i_series_native_input2"
    assert record["validation"]["every_epoch"] is True
    assert not (directory / "validation_common").exists()
    batch.run(data, "cpu", project, epochs=1, only="J00_yolo11n_control", source_balanced=source_balanced)
    assert len(calls) == 1, "An identical completed run must be skipped, not overwritten"
