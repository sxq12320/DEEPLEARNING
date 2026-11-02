"""Focused J7 runner and checkpoint-selection contracts; no GPU or real training required."""

import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import citrus_j7_batch as batch_runner
import citrus_j7_training as training


@pytest.fixture
def fake_queue(tmp_path, monkeypatch):
    """Replace only training/runtime interfaces; exercise real queue bookkeeping."""
    import citrus_j2_batch
    import ultralytics
    import ultralytics.utils.torch_utils

    yaml_dir = tmp_path / "yaml"
    yaml_dir.mkdir()
    for name in ("J100_control", "J102_anchor", "J103_hub", "J104_factor"):
        (yaml_dir / (name + ".yaml")).write_text("nc: 1\n", encoding="utf-8")
    data = tmp_path / "data.yaml"
    data.write_text("nc: 1\n", encoding="utf-8")
    state = SimpleNamespace(calls=[], error=None)

    class FakeYOLO:
        def __init__(self, *args, **kwargs):
            self.path = Path(args[0])

        def add_callback(self, *args, **kwargs):
            pass

        def train(self, **kwargs):
            state.calls.append(self.path.stem)
            if state.error is not None:
                raise state.error
            result_dir = Path(kwargs["project"]) / kwargs["name"] / "selected_mask_validation"
            result_dir.mkdir(parents=True)
            (result_dir / "selected_checkpoint.json").write_text("{}", encoding="utf-8")

    monkeypatch.setattr(batch_runner, "YAML_DIR", yaml_dir)
    monkeypatch.setattr(batch_runner, "source_manifest", lambda: {"test.py": "fixed"})
    monkeypatch.setattr(citrus_j2_batch, "settings", lambda *args, **kwargs: {"amp": False, "cache": True})
    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    monkeypatch.setattr(ultralytics.utils.torch_utils, "init_seeds", lambda *args, **kwargs: None)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch, "set_num_threads", lambda value: None)
    return data, tmp_path / "results", state


def test_lightweight_import_and_help_do_not_import_torch():
    root = Path(batch_runner.__file__).parent
    check = subprocess.run(
        [sys.executable, "-c", "import sys; import citrus_j7_batch; assert 'torch' not in sys.modules"],
        cwd=str(root), capture_output=True, text=True, check=False,
    )
    assert check.returncode == 0, check.stderr
    help_run = subprocess.run(
        [sys.executable, str(root / "citrus_j7_batch.py"), "--help"],
        cwd=str(root), capture_output=True, text=True, check=False,
    )
    assert help_run.returncode == 0
    assert "--no-plots" in help_run.stdout and "--suite" in help_run.stdout


def test_model_selection_and_physical_binding(monkeypatch):
    assert [path.stem.split("_")[0] for path in batch_runner.select_models()] == list(batch_runner.PRIORITY_IDS)
    assert len(batch_runner.select_models("all")) == 10
    assert [path.stem.split("_")[0] for path in batch_runner.select_models(only="J104,J102")] == ["J104", "J102"]
    with pytest.raises(ValueError, match="Duplicate model"):
        batch_runner.select_models(only="J100,J100_official_control")
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    assert batch_runner.bind_device("1", 2) == "0"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "2"
    assert batch_runner.bind_device("cpu", 2) == "cpu"
    for invalid in ("0,1", "cuda:1", "-1"):
        with pytest.raises(ValueError, match="physical GPU"):
            batch_runner.bind_device(invalid)


def test_existing_partial_output_is_not_overwritten(fake_queue):
    data, project, state = fake_queue
    existing = project / "J100_control_seed42"
    existing.mkdir(parents=True)
    marker = existing / "keep.txt"
    marker.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Partial/conflicting"):
        batch_runner.run(data, "cpu", project, only="J100", epochs=1, plots=False)
    assert state.calls == []
    assert marker.read_text(encoding="utf-8") == "original"


def test_exact_completed_run_skips_but_changed_protocol_refuses(fake_queue):
    data, project, state = fake_queue
    batch_runner.run(data, "cpu", project, only="J100", epochs=1, plots=False)
    assert state.calls == ["J100_control"]
    assert batch_runner.run(data, "cpu", project, only="J100", epochs=1, plots=False) == []
    assert state.calls == ["J100_control"]
    with pytest.raises(FileExistsError, match="Partial/conflicting"):
        batch_runner.run(data, "cpu", project, only="J100", epochs=2, plots=False)
    assert state.calls == ["J100_control"]


@pytest.mark.parametrize("error", [RuntimeError("failed arm"), KeyboardInterrupt()])
def test_failed_or_interrupted_arm_aborts_remaining_queue(fake_queue, error):
    data, project, state = fake_queue
    state.error = error
    with pytest.raises(type(error)):
        batch_runner.run(data, "cpu", project, only="J100,J102", epochs=1, plots=False)
    assert state.calls == ["J100_control"]
    assert not (project / "J102_anchor_seed42").exists()


def test_plotting_failure_is_early_and_never_silently_disables_plots(fake_queue, monkeypatch):
    data, project, state = fake_queue

    def incompatible_plotting():
        raise RuntimeError("plotting dependency failure")

    monkeypatch.setattr(batch_runner, "plotting_preflight", incompatible_plotting)
    with pytest.raises(RuntimeError, match="plotting dependency failure"):
        batch_runner.run(data, "cpu", project, only="J100", epochs=300, plots=True)
    assert state.calls == []
    assert not project.exists()


def test_final_eval_matches_mask_checkpoint_and_preserves_native_best(tmp_path, monkeypatch):
    weights = tmp_path / "weights"
    weights.mkdir()
    selected = weights / "best_mask.pt"
    selected.write_bytes(b"mask-selected-checkpoint")
    native_best = weights / "best.pt"
    native_best.write_bytes(b"native-fitness-checkpoint")
    (tmp_path / "best_mask_selection.json").write_text(
        json.dumps({"epoch": 7, "mask_mAP50_95": 0.51}), encoding="utf-8"
    )
    observed = {}

    class FakeValidator:
        def __init__(self, dataloader, save_dir, args, _callbacks):
            observed.update(dataloader=dataloader, save_dir=save_dir, args=args)
            save_dir.mkdir(parents=True)

        def __call__(self, model):
            observed["checkpoint"] = model
            return {"metrics/mAP50-95(M)": 0.51, "fitness": 0.51}

    monkeypatch.setattr(training, "J7FinalAuditValidator", FakeValidator)
    trainer = SimpleNamespace(
        save_dir=tmp_path, best=native_best, args=SimpleNamespace(half=True, plots=True),
        test_loader=object(), callbacks={}, epoch=9, run_callbacks=lambda event: observed.update(callback=event),
    )
    training.J7Trainer.final_eval(trainer)
    assert observed["checkpoint"] == str(selected)
    assert observed["args"].half is False and observed["args"].compile is False
    assert observed["args"].plots is True
    assert observed["dataloader"] is trainer.test_loader
    assert trainer.best == selected  # Upstream Model.train() reloads this pointer.
    assert trainer.j7_native_best == native_best
    assert native_best.read_bytes() == b"native-fitness-checkpoint"
    assert selected.read_bytes() == b"mask-selected-checkpoint"
    assert trainer.epoch == 9
    record = json.loads((tmp_path / "selected_mask_validation" / "selected_checkpoint.json").read_text("utf-8"))
    assert record["selection"]["epoch"] == 7
    assert len(record["checkpoint_sha256"]) == 64
    assert "fitness" not in record["metrics"]
    with pytest.raises(FileExistsError, match="overwrite"):
        training.J7Trainer.final_eval(trainer)


def test_missing_mask_checkpoint_does_not_fall_back_to_fitness_checkpoint(tmp_path):
    trainer = SimpleNamespace(save_dir=tmp_path)
    with pytest.raises(RuntimeError, match="NOT substituted"):
        training.J7Trainer.final_eval(trainer)


def test_batch_drift_fails_before_rebuilding_pipeline():
    trainer = SimpleNamespace(args=SimpleNamespace(batch=8), _j7_requested_batch=16)
    with pytest.raises(RuntimeError, match="fixed protocol"):
        training.J7Trainer._build_train_pipeline(trainer)


@pytest.mark.parametrize(
    ("name", "shared", "fine", "criterion"),
    [("J104", False, True, "v8SegmentationLoss"), ("J105", True, True, "v8SegmentationLoss"),
     ("J108", False, False, "v8SegmentationLoss"), ("J109", False, True, "J7SegmentationLoss")],
)
def test_instantiated_model_configuration_records_actual_operators(name, shared, fine, criterion):
    from ultralytics import YOLO
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace

    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = YOLO(str(batch_runner.select_models(only=name)[0]), task="segment", verbose=False).model
        model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, "mask_ratio": 2})
        record = batch_runner.actual_model_configuration(model)
        assert record["quality_predictor_present"] is False
        assert record["quality_calibration"] is False
        assert record["shared_towers"] is shared
        assert record["fine_mask"] is fine
        assert record["detail_channels"] == 16
        assert record["persistent"] is True
        assert record["recognition_route"] == 0
        assert record["criterion_class"].endswith("." + criterion)
        assert record["gains"]["quality_gain"] == 0
        if name == "J109":
            assert record["gains"]["tiny_dice_gain"] == 0.25
            assert record["gains"]["boundary_gain"] == 0.5
    finally:
        torch.set_num_threads(previous)
