"""Queue safety, mask checkpoint selection and initialization provenance for v4r."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from citrus_protocol import fixed_train_args
from citrus_sage_v4r_suite import NAMES, ROOT, YAML_DIR
from ultralytics import YOLO


@pytest.fixture
def batch_module():
    spec = importlib.util.spec_from_file_location("sage_v4r_test_runner", ROOT / "20260903_citrus_sage_v4r_batch.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_initialization_is_measured_before_training(batch_module, tmp_path):
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        checkpoint = ROOT / "yolo11n-seg.pt"
        if not checkpoint.exists():
            pytest.skip("Local pretrained checkpoint unavailable")
        model = YOLO(str(YAML_DIR / f"{NAMES[4]}.yaml"), verbose=False).load(str(checkpoint)).model
        callback = batch_module.initialization_recorder(checkpoint)
        callback(SimpleNamespace(model=model, save_dir=tmp_path))
        report = json.loads((tmp_path / "initialization_transfer.json").read_text())
        assert 0.8 < report["equal_fraction"] < 1
        assert "model.2.cv1.conv.weight" in report["equal_keys"]
        assert any("refiner" in key for key in report["other_keys"])
    finally:
        torch.set_num_threads(previous)


def test_mask_selection_does_not_replace_official_best(batch_module, tmp_path):
    weights = tmp_path / "weights"
    weights.mkdir()
    official, last = weights / "best.pt", weights / "last.pt"
    official.write_bytes(b"official")
    last.write_bytes(b"epoch1")
    trainer = SimpleNamespace(save_dir=tmp_path, last=last, epoch=0, metrics={"metrics/mAP50-95(M)": 0.6})
    batch_module.save_best_mask(trainer)
    last.write_bytes(b"epoch2")
    trainer.epoch, trainer.metrics["metrics/mAP50-95(M)"] = 1, 0.5
    batch_module.save_best_mask(trainer)
    assert (weights / "best_mask.pt").read_bytes() == b"epoch1"
    trainer.epoch, trainer.metrics["metrics/mAP50-95(M)"] = 2, 0.7
    batch_module.save_best_mask(trainer)
    assert (weights / "best_mask.pt").read_bytes() == b"epoch2"
    assert official.read_bytes() == b"official"


@pytest.mark.parametrize("interrupt", [False, True])
def test_queue_is_sequential_locked_and_interruptible(batch_module, tmp_path, monkeypatch, interrupt):
    import ultralytics

    data = tmp_path / "data.yaml"
    data.write_text("train: train/images\nval: val/images\nnames: [fruit]\n", encoding="utf-8")
    checkpoint = tmp_path / "initialization.pt"
    checkpoint.write_bytes(b"mock checkpoint -- never used for real training")
    project = tmp_path / "mock_results"
    chosen = [NAMES[1], NAMES[4]]
    arguments = [
        "test",
        "--data",
        str(data),
        "--only",
        ",".join(chosen),
        "--pretrained",
        str(checkpoint),
        "--epochs",
        "3",
        "--project",
        str(project),
        "--seeds",
        "42,43",
        "--skip-completed",
        "--fail-fast",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    protocol = batch_module.load_protocol()
    protocol["fixed_validation"]["plots"] = False  # Mock-only: no actual trainer or plot creation.
    monkeypatch.setattr(batch_module, "load_protocol", lambda: protocol)
    monkeypatch.setattr(batch_module, "initialization_recorder", lambda _: lambda trainer: None)
    calls, active = [], []

    class FakeYOLO:
        def __init__(self, filename, **kwargs):
            self.name = Path(filename).stem
            self.callbacks = {}

        def load(self, checkpoint):
            return self

        def add_callback(self, event, callback):
            self.callbacks[event] = callback

        def train(self, **kwargs):
            assert not active
            active.append(self.name)
            calls.append((self.name, kwargs["seed"]))
            try:
                for key, value in fixed_train_args().items():
                    assert kwargs[key] == value
                if interrupt:
                    raise KeyboardInterrupt()
                directory = Path(kwargs["project"]) / kwargs["name"]
                (directory / "weights").mkdir(parents=True)
                (directory / "weights/best.pt").write_bytes(b"mock")
                (directory / "results.csv").write_text("epoch,metric\n3,0.5\n", encoding="utf-8")
            finally:
                active.clear()

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    if interrupt:
        with pytest.raises(KeyboardInterrupt):
            batch_module.main()
        assert len(calls) == 1
        events = [json.loads(line) for line in (project / "_protocol/ledger.jsonl").read_text().splitlines()]
        assert events[-1]["status"] == "interrupted"
        assert not list(project.glob("*/completed.json"))
        return
    batch_module.main()
    assert len(calls) == 4
    assert [seed for _, seed in calls] == [42, 42, 43, 43]
    assert set(calls) == {(name, seed) for name in chosen for seed in (42, 43)}
    assert len(list(project.glob("*/completed.json"))) == 4
    batch_module.main()  # Completed experiments are skipped without launching another trainer.
    assert len(calls) == 4
    (project / f"{calls[0][0]}_seed{calls[0][1]}" / "results.csv").write_text("epoch,metric\n1,0.5\n")
    with pytest.raises(FileExistsError, match="partial or existing"):
        batch_module.main()


def test_wrong_amp_is_rejected_before_training(batch_module, tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["test", "--data", str(tmp_path / "missing.yaml"), "--amp"])
    with pytest.raises(ValueError, match="locks amp=False"):
        batch_module.main()


def test_real_trainer_calls_batch_callbacks_on_explicit_smoke_fixture(batch_module, tmp_path, monkeypatch):
    """TEST-ONLY overrides exercise the actual runner, not formal paper hyperparameters."""
    data = ROOT / "1_results/_validation_sage_v4_20260903/fixture/data.yaml"
    checkpoint = ROOT / "yolo11n-seg.pt"
    if not data.exists() or not checkpoint.exists():
        pytest.skip("Optional local four-image smoke fixture/checkpoint unavailable")
    project = tmp_path / "SMOKE_NOT_FORMAL"
    original_protocol = batch_module.load_protocol()
    original_protocol["fixed_validation"]["plots"] = False
    smoke = fixed_train_args()
    smoke.update(batch=2, imgsz=256, workers=0, mosaic=0.0, close_mosaic=0, plots=False)
    monkeypatch.setattr(batch_module, "fixed_train_args", lambda: dict(smoke))
    monkeypatch.setattr(batch_module, "load_protocol", lambda: original_protocol)
    monkeypatch.setattr(batch_module, "validate_locked_runtime", lambda **kwargs: None)

    def smoke_snapshot(directory, additions):
        (directory / "_protocol").mkdir(parents=True)
        target = directory / "_protocol/SMOKE_NOT_FORMAL.json"
        target.write_text(json.dumps({"smoke_overrides": smoke, "not_formal": True}, indent=2), encoding="utf-8")
        return target, "TEST_ONLY_SMOKE_NOT_FORMAL"

    monkeypatch.setattr(batch_module, "write_protocol_snapshot", smoke_snapshot)
    name = NAMES[7]  # Exercise the two geometry losses as well as standard callbacks.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "test",
            "--data",
            str(data),
            "--device",
            "cpu",
            "--only",
            name,
            "--epochs",
            "1",
            "--project",
            str(project),
            "--pretrained",
            str(checkpoint),
            "--fail-fast",
        ],
    )
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    try:
        batch_module.main()
    finally:
        torch.set_num_threads(previous)
    run = project / f"{name}_seed42"
    assert batch_module.completed(run, 1, 42)
    for filename in (
        "weights/best_mask.pt",
        "best_mask_selection.json",
        "initialization_transfer.json",
        "train_loaded_files.txt",
        "val_loaded_files.txt",
        "loaded_data_summary.json",
    ):
        assert (run / filename).exists(), filename
    report = json.loads((run / "loaded_data_summary.json").read_text())
    assert report["train"]["images"] == 4 and report["val"]["images"] == 4
    inherited = json.loads((run / "initialization_transfer.json").read_text())
    assert 0.8 < inherited["equal_fraction"] < 1
