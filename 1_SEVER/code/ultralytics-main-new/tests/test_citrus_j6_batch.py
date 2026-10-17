"""Queue, resume, provenance and physical-device contracts; no citrus efficacy claims."""

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import citrus_j6_batch as runner
from citrus_j6_suite import NAMES, jobs, select_names
from RUN_CITRUS_J6 import configure_device, parse_args


def test_explicit_queue_and_seed_validation():
    assert len(NAMES) == 10 and len(jobs("all")) == 10
    assert jobs("screen") == [(NAMES[index], 42) for index in (0, 1, 2, 4)]
    assert jobs("all", (42, 43), NAMES[4]) == [(NAMES[4], 42), (NAMES[4], 43)]
    assert select_names("all", f"{NAMES[4]},{NAMES[1]}") == [NAMES[4], NAMES[1]]
    for seeds in ((), (42, 42), (True,), (-1,), ("42",)):
        with pytest.raises(ValueError):
            jobs(seeds=seeds)
    for only in (NAMES[4] + "," + NAMES[4], "J00_unknown", ", ,"):
        with pytest.raises(ValueError):
            select_names(only=only)


def test_yaml_switches_and_single_factor_pairs_match_the_declared_experiments():
    import yaml
    from citrus_j6_suite import YAML_DIR

    configurations = {}
    for name in NAMES:
        source = yaml.safe_load((YAML_DIR / f"{name}.yaml").read_text(encoding="utf-8"))
        assert source["nc"] == 1 and source["backbone"][0] == [-1, 1, "nn.Identity", []]
        head = source["head"][-1]
        assert head[:3] == [[17, 20, 23, 0], 1, "SegmentCitrusJ6"]
        expected = [source["citrus_j6"][key] for key in
                    ("enabled", "neighbor", "selection", "shuffle", "use_d2", "gt_balanced", "gain_route")]
        assert head[3] == ["nc", 32, 256, *expected, 8, 16, 16, 32]
        configurations[name] = source["citrus_j6"]
    for first, second, changed in ((0, 1, {"enabled"}), (1, 2, {"neighbor"}),
                                    (1, 3, {"selection"}), (2, 5, {"shuffle"}),
                                    (1, 6, {"use_d2"}), (1, 7, {"gt_balanced"}),
                                    (1, 8, {"gain_route"}), (4, 9, {"gain_route"})):
        left, right = configurations[NAMES[first]], configurations[NAMES[second]]
        assert {key for key in left if left[key] != right[key]} == changed


def test_real_yaml_registration_keeps_three_detection_strides_and_all_switches():
    import torch
    import yaml
    from citrus_j6_suite import YAML_DIR
    from ultralytics.nn.modules import SegmentCitrusJ6
    from ultralytics.nn.tasks import SegmentationModel

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        for name in NAMES:
            path = YAML_DIR / f"{name}.yaml"
            model = SegmentationModel(path, verbose=False)
            head = model.model[-1]
            assert isinstance(head, SegmentCitrusJ6) and head.nl == 3
            assert model.stride.tolist() == [8., 16., 32.] and 0 in model.save
            switches = yaml.safe_load(path.read_text(encoding="utf-8"))["citrus_j6"]
            assert all(getattr(head, key) == value for key, value in switches.items())
    finally:
        torch.set_num_threads(previous_threads)


def test_device_is_bound_before_torch_and_same_physical_id_returned(monkeypatch):
    import sys

    monkeypatch.delitem(sys.modules, "torch", raising=False)
    assert configure_device("1", 3) == "1"
    import os

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["OMP_NUM_THREADS"] == "3"
    assert configure_device("cpu", 2) == "cpu"
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "-1"
    for invalid in ("0,1", "cuda:1", "-1", "gpu"):
        with pytest.raises(ValueError):
            configure_device(invalid, 4)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace())
    with pytest.raises(RuntimeError, match="fresh Python"):
        configure_device("1", 4)


def test_smoke_is_one_epoch_in_an_isolated_project():
    args = parse_args(["--smoke", "--project", "/somewhere/formal", "--device", "cpu", "--seeds", "42,43"])
    assert args.epochs == 1 and Path(args.project).name == "formal_SMOKE_1EP"
    assert args.seeds == (42, 43)
    with pytest.raises(SystemExit):
        parse_args(["--smoke", "--resume", NAMES[4]])
    assert parse_args([]).plots is True
    assert parse_args(["--no-plots"]).plots is False


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    """Isolate filesystem/scheduler contracts from GPU and data-loader implementation."""
    yaml_dir = tmp_path / "yamls"
    yaml_dir.mkdir()
    for name in NAMES:
        (yaml_dir / f"{name}.yaml").write_text("nc: 1\n", encoding="utf-8")
    data = tmp_path / "original_data.yaml"
    data.write_text("train: original_train/images\nval: original_val/images\n", encoding="utf-8")
    calls, seeds, plot_imports = [], [], []

    class FakeYOLO:
        failure = None

        def __init__(self, source, **kwargs):
            self.source = str(source)
            self.callbacks = {}
            self.model = SimpleNamespace(parameters=lambda: [])
            self.ckpt = {"epoch": 0, "optimizer": {"state": {}, "param_groups": []}}

        def add_callback(self, name, callback):
            self.callbacks.setdefault(name, []).append(callback)

        def train(self, **kwargs):
            calls.append(dict(source=self.source, **kwargs))
            directory = Path(kwargs["project"]) / kwargs["name"]
            directory.mkdir(parents=True, exist_ok=kwargs["exist_ok"])
            (directory / "weights").mkdir(exist_ok=True)
            self.trainer = SimpleNamespace(save_dir=directory)
            for callback in self.callbacks.get("on_pretrain_routine_end", []):
                callback(self.trainer)
            if self.failure:
                raise self.failure("synthetic training interruption")
            write_finished(directory, kwargs["epochs"])

    fake_torch = SimpleNamespace(set_num_threads=lambda count: None,
                                 cuda=SimpleNamespace(is_available=lambda: False))
    monkeypatch.setattr(runner, "YAML_DIR", yaml_dir)
    monkeypatch.setattr(runner, "source_snapshot", lambda: dict(sha256="test-code", files={}))
    monkeypatch.setattr(runner, "runtime_dependencies", lambda: (
        fake_torch, FakeYOLO, lambda seed, **kwargs: seeds.append(seed),
        lambda trainer, **kwargs: None, lambda trainer: None, object))
    monkeypatch.setattr(runner, "plotting_preflight", lambda: plot_imports.append(True) or "test-version")
    monkeypatch.setattr(runner, "build_train_args", lambda epochs, seed, workers, batch, plots=True: dict(
        epochs=epochs, seed=seed, workers=workers, batch=batch, imgsz=640, mask_ratio=2,
        pretrained=False, amp=False, cache=True, optimizer="AdamW", multi_scale=0.0, mosaic=1.0,
        rect=False, plots=plots))
    return SimpleNamespace(data=data, project=tmp_path / "project", calls=calls, seeds=seeds,
                           model_type=FakeYOLO, plot_imports=plot_imports)


def write_finished(directory, epochs):
    """Write real-format CSV/provenance for synthetic queue-only tests."""
    directory = Path(directory)
    fields = ["epoch", "metrics/mAP50-95(M)", "metrics/mAP50(M)",
              "metrics/precision(M)", "metrics/recall(M)"]
    history = [[1, .8, .9, .75, .7]]
    if epochs > 1:
        history.append([epochs, .7, .99, .8, .99])
    with (directory / "results.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(fields)
        writer.writerows(history)
    (directory / "weights/last.pt").write_bytes(b"test-only-not-a-real-checkpoint")
    (directory / "weights/best_mask.pt").write_bytes(b"test-only-not-a-real-checkpoint")
    runner.write_json(directory / "best_mask_selection.json", dict(epoch=1, mask_mAP50_95=.8))


def test_foreground_queue_uses_original_data_and_skips_verified_completed(fake_runtime):
    fake = fake_runtime
    options = dict(data=fake.data, device="1", project=fake.project, epochs=3,
                   only=f"{NAMES[0]},{NAMES[4]}", seeds=(42,))
    runner.run(**options)
    assert len(fake.calls) == 2 and fake.seeds == [42, 42]
    assert all(call["data"] == str(fake.data.resolve()) and call["device"] == "1" for call in fake.calls)
    assert not list(fake.project.glob("_prepared*"))
    rows = runner.summarize(fake.project, jobs(only=options["only"]))
    assert all(row["mask_recall"] == "0.7" and row["selected_epoch"] == "1" for row in rows)
    assert all((fake.project / f"{name}_seed42" / runner.MARKER).is_file() for name in (NAMES[0], NAMES[4]))
    runner.run(**options)
    assert len(fake.calls) == 2  # No repeated training or hidden extra jobs.


@pytest.mark.parametrize("failure", [RuntimeError, KeyboardInterrupt])
def test_failure_or_ctrl_c_stops_queue_and_never_marks_completion(fake_runtime, failure):
    fake = fake_runtime
    fake.model_type.failure = failure
    with pytest.raises(failure):
        runner.run(fake.data, "cpu", fake.project, epochs=3, only=f"{NAMES[0]},{NAMES[4]}")
    assert len(fake.calls) == 1
    first = fake.project / f"{NAMES[0]}_seed42"
    assert not (first / runner.MARKER).exists()
    assert not (fake.project / f"{NAMES[4]}_seed42").exists()
    status = json.loads((first / "j6_run_state.json").read_text(encoding="utf-8"))
    assert status["status"] == ("interrupted" if failure is KeyboardInterrupt else "failed")


def test_partial_run_is_not_overwritten_and_preflight_precedes_all_training(fake_runtime):
    fake = fake_runtime
    partial = fake.project / f"{NAMES[4]}_seed42"
    partial.mkdir(parents=True)
    (partial / "keep.txt").write_text("do not change", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Partial"):
        runner.run(fake.data, "cpu", fake.project, epochs=3, only=f"{NAMES[0]},{NAMES[4]}")
    assert not fake.calls  # Later partial output is detected BEFORE earlier models train.
    assert (partial / "keep.txt").read_text(encoding="utf-8") == "do not change"


def test_explicit_resume_uses_last_checkpoint_and_same_trainer(fake_runtime):
    fake = fake_runtime
    options = dict(data=fake.data, device="cpu", project=fake.project, epochs=3, only=NAMES[4])
    runner.run(**options)
    directory = fake.project / f"{NAMES[4]}_seed42"
    (directory / runner.MARKER).unlink()  # Synthetic incomplete run with its exact protocol.
    write_finished(directory, 1)
    runner.run(**options, resume=NAMES[4])
    call = fake.calls[-1]
    assert call["source"] == str(directory / "weights/last.pt")
    assert call["resume"] == str(directory / "weights/last.pt") and call["exist_ok"] is True
    assert (directory / runner.MARKER).is_file()
    with pytest.raises(ValueError, match="exactly one seed"):
        runner.resume_names(NAMES[4], [(NAMES[4], 42), (NAMES[4], 43)])


def test_finalized_checkpoint_is_never_silently_restarted():
    runner.assert_resumable_checkpoint({"epoch": 0, "optimizer": {}}, 300)
    for checkpoint in ({"epoch": -1, "optimizer": None}, {"epoch": 10, "optimizer": None}):
        with pytest.raises(RuntimeError, match="finalized/optimizer-stripped"):
            runner.assert_resumable_checkpoint(checkpoint, 300)
    with pytest.raises(RuntimeError, match="already reached"):
        runner.assert_resumable_checkpoint({"epoch": 299, "optimizer": {}}, 300)
    with pytest.raises(RuntimeError, match="dictionary/epoch"):
        runner.assert_resumable_checkpoint(None, 300)


def test_real_mask_checkpoint_callback_loads_and_keeps_native_dispatch(tmp_path):
    """Synthetic checkpoint transport, not a trained quality/gain result."""
    import torch
    from citrus_j_batch import save_mask_checkpoints
    from citrus_j6_suite import YAML_DIR
    from citrus_j6_training import J6Predictor, J6Trainer, J6Validator
    from ultralytics import YOLO

    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        directory = tmp_path / "synthetic_checkpoint"
        (directory / "weights").mkdir(parents=True)
        model = YOLO(str(YAML_DIR / "J94_full.yaml"), task="segment", verbose=False)
        model.model.model[-1].auxiliary_updates.fill_(7)
        model.model.model[-1].gain_updates.fill_(3)
        last = directory / "weights/last.pt"
        model.save(last)
        trainer = SimpleNamespace(save_dir=directory, last=last, epoch=2, metrics={
            "metrics/mAP50-95(M)": .5, "metrics/mAP50(M)": .8})
        save_mask_checkpoints(trainer)
        saved = json.loads((directory / "best_mask_selection.json").read_text(encoding="utf-8"))
        assert saved == {"epoch": 3, "mask_mAP50_95": .5}
        restored = YOLO(str(directory / "weights/best_mask.pt"), task="segment", verbose=False)
        head = restored.model.model[-1]
        assert int(head.auxiliary_updates) == 7 and int(head.gain_updates) == 3
        for key, expected in (("trainer", J6Trainer), ("validator", J6Validator), ("predictor", J6Predictor)):
            assert restored._smart_load(key) is expected
        assert head.enabled and head.neighbor and head.selection and head.gain_route
    finally:
        torch.set_num_threads(previous_threads)


def test_incomplete_epochs_and_missing_selected_checkpoint_are_not_success(fake_runtime):
    fake = fake_runtime
    runner.run(fake.data, "cpu", fake.project, epochs=3, only=NAMES[4])
    directory = fake.project / f"{NAMES[4]}_seed42"
    with pytest.raises(RuntimeError, match="3/300"):
        runner.assert_finished(directory, 300)
    (directory / "weights/best_mask.pt").unlink()
    with pytest.raises(FileNotFoundError, match="Mask-selected"):
        runner.assert_finished(directory, 3)


def test_conflicting_completed_run_is_preserved(fake_runtime):
    fake = fake_runtime
    runner.run(fake.data, "cpu", fake.project, epochs=3, only=NAMES[4])
    with pytest.raises(FileExistsError, match="Completed protocol changed"):
        runner.run(fake.data, "cpu", fake.project, epochs=4, only=NAMES[4])
    assert len(fake.calls) == 1


def test_summary_follows_checkpoint_epoch_even_when_csv_rounds_scores_to_a_tie(fake_runtime):
    fake = fake_runtime
    runner.run(fake.data, "cpu", fake.project, epochs=3, only=NAMES[4])
    directory = fake.project / f"{NAMES[4]}_seed42"
    runner.write_json(directory / "best_mask_selection.json", dict(epoch=3, mask_mAP50_95=.7))
    rows = runner.summarize(fake.project, [(NAMES[4], 42)])
    assert rows[0]["selected_epoch"] == "3" and rows[0]["mask_recall"] == "0.99"
    assert rows[0]["selection_source"] == "best_mask_selection.json"


def test_dry_run_does_not_require_data_or_write_training_outputs(fake_runtime):
    fake = fake_runtime
    runner.run(fake.data.with_name("missing.yaml"), "cpu", fake.project,
               only=NAMES[4], dry_run=True)
    assert not fake.calls and not fake.project.exists() and not fake.plot_imports


def test_plotting_preflight_failure_stops_before_training_or_output_creation(fake_runtime, monkeypatch):
    fake = fake_runtime

    def failed_import():
        raise RuntimeError("synthetic incompatible plotting environment")

    monkeypatch.setattr(runner, "plotting_preflight", failed_import)
    with pytest.raises(RuntimeError, match="incompatible plotting"):
        runner.run(fake.data, "cpu", fake.project, only=NAMES[4])
    assert not fake.calls and not fake.project.exists()
    runner.run(fake.data, "cpu", fake.project, only=NAMES[4], plots=False)
    assert len(fake.calls) == 1 and fake.calls[0]["plots"] is False
    protocol = json.loads((fake.project / f"{NAMES[4]}_seed42" / runner.RECORD).read_text(encoding="utf-8"))
    assert protocol["train"]["plots"] is False
    assert protocol["protocol"].endswith("20261017_r2")
    assert protocol["scout_geometry"] == "bbox_center_offset_v2"
    assert protocol["candidate_policy"] == "cell_topk_no_peak_nms_v2"
    assert protocol["scout_source1_policy"] == "zero_coeff_cell_heat_v2"
    assert "assigned_scout_cell" in protocol["gain_target"]


def test_plotting_import_error_is_explained_without_install_or_silent_fallback(monkeypatch):
    import builtins

    original_import = builtins.__import__

    def incompatible_import(name, *args, **kwargs):
        if name == "matplotlib":
            raise ImportError("Matplotlib requires a newer NumPy")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", incompatible_import)
    with pytest.raises(RuntimeError, match="Training has not started") as error:
        runner.plotting_preflight()
    assert "--no-plots" in str(error.value) and "No automatic pip changes" in str(error.value)


def test_real_locked_recipe_is_not_silently_changed():
    train = runner.build_train_args(300, 42, 4, 16)
    assert train["amp"] is False and train["cache"] is True and train["pretrained"] is False
    assert train["imgsz"] == 640 and train["mask_ratio"] == 2 and train["optimizer"] == "AdamW"
    assert train["mosaic"] == 1 and train["multi_scale"] == 0 and train["rect"] is False
    assert train["copy_paste"] == .3 and train["patience"] == 301 and train["time"] is None
    assert train["plots"] is True
    assert runner.build_train_args(1, 42, 0, 2, plots=False)["plots"] is False
