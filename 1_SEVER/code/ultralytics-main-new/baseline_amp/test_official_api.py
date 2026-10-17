"""CPU contracts for the common API; no model rewriting or training in these tests."""

import copy
import json
import subprocess
import sys
from pathlib import Path

import pytest

from baseline_amp import OfficialBaseline, default_pythons, run_baselines
from baseline_amp import batch, recovery
from baseline_amp.registry import COMMON_VALIDATION, MODELS


@pytest.mark.parametrize("model", tuple(MODELS))
def test_train_keeps_official_recipe_and_uses_direct_runner(model, tmp_path, monkeypatch):
    received = {}

    def run(settings, **options):
        received.update(settings=copy.deepcopy(settings), options=options)

    monkeypatch.setattr(batch, "run", run)
    monkeypatch.setattr(sys, "argv", ["vscode_debug.py", "--an-unrelated-debugger-flag"])
    pythons = {"mmdet": "official-mm-python", "modern": "official-modern-python"}
    project = OfficialBaseline(model).train(
        tmp_path / "data.yaml", tmp_path / "results", epochs=3, device=2, workers=2,
        batch=4, seeds=(42, 43), pythons=pythons, dry_run=True,
    )
    settings = received["settings"]
    assert project == tmp_path / "results"
    assert settings["DATA"] == str(tmp_path / "data.yaml")
    assert settings["PROJECT"] == str(project)
    assert settings["DEVICE"] == 2 and settings["EPOCHS"] == 3
    assert settings["SEEDS"] == [42, 43] and settings["AMP_MODES"] == [1, 0]
    assert settings["WORKERS"] == 2 and settings["BATCHES"] == {model: 4}
    assert settings["VALIDATION"] == COMMON_VALIDATION
    assert settings["PYTHONS"] == pythons and settings["SUITE"] == "all"
    assert received["options"] == dict(
        models=(model,), dry_run=True, preflight_only=False, summarize_only=False,
    )
    assert not project.exists()


def test_single_precision_does_not_change_batch_or_other_recipe(tmp_path, monkeypatch):
    received = {}
    monkeypatch.setattr(batch, "run", lambda settings, **kw: received.update(settings))
    OfficialBaseline("mask_rcnn_r50").train("data.yaml", tmp_path, amp_modes=(0,), dry_run=True)
    assert received["AMP_MODES"] == [0]
    assert received["BATCHES"] == {}
    assert received["PYTHONS"] == default_pythons()


def test_unrecognized_yolo_only_options_cannot_silently_change_official_model(tmp_path):
    model = OfficialBaseline("rtmdet_ins_tiny")
    with pytest.raises(TypeError):
        model.train("data.yaml", tmp_path, cache=True)
    with pytest.raises(TypeError):
        model.train("data.yaml", tmp_path, mask_ratio=4)
    with pytest.raises(ValueError, match="Unknown official baseline"):
        OfficialBaseline("made_up_yolo_head")


def test_batch_api_does_not_mutate_callers_settings(tmp_path, monkeypatch):
    settings = dict(PROJECT=str(tmp_path), nested=dict(values=[1]))
    before = copy.deepcopy(settings)

    def mutate(snapshot, **options):
        snapshot["nested"]["values"].append(2)

    monkeypatch.setattr(batch, "run", mutate)
    assert run_baselines(settings, dry_run=True) == tmp_path
    assert settings == before


def make_saved_job(tmp_path, model="rtmdet_ins_tiny"):
    run_dir = tmp_path / "completed_model"
    run_dir.mkdir()
    checkpoint = run_dir / "native_weights.pth"
    checkpoint.write_bytes(b"placeholder; never loaded by the mock runner")
    job = dict(
        model=model, run_dir=str(run_dir), prepared=str(tmp_path / "prepared"),
        recipe=dict(MODELS[model]), seed=42, validation=dict(COMMON_VALIDATION),
    )
    (run_dir / "job.json").write_text(json.dumps(job), encoding="utf-8")
    (run_dir / "trained.json").write_text(json.dumps(dict(checkpoint=str(checkpoint))), encoding="utf-8")
    (run_dir / "complete.json").write_text('{"preserve": true}', encoding="utf-8")
    return run_dir, checkpoint


@pytest.mark.parametrize("mode", ("val", "profile"))
@pytest.mark.parametrize("model", ("rtmdet_ins_tiny", "mask_rcnn_r50", "solov2_light_r18", "rfdetr_seg_nano"))
def test_common_eval_uses_saved_native_checkpoint_in_isolated_process(tmp_path, monkeypatch, mode, model):
    run_dir, checkpoint = make_saved_job(tmp_path, model)
    original = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    calls = []
    monkeypatch.setattr(sys, "argv", ["debug.py", "--invalid-cli-flag"])
    monkeypatch.setattr(recovery, "run_visible", lambda command, cwd, env, log: calls.append((command, env)))
    pythons = {"mmdet": sys.executable, "modern": sys.executable}
    destination = getattr(OfficialBaseline(model), mode)(run_dir, device=3, pythons=pythons)
    assert destination.is_dir() and destination.parent == run_dir
    assert {name: (run_dir / name).read_bytes() for name in original} == original
    assert len(calls) == 1
    command, env = calls[0]
    assert command[1:3] == ["-I", "-u"]
    assert str(command[command.index("--checkpoint") + 1]) == str(checkpoint)
    assert command[command.index("--stage") + 1] == "evaluate"
    assert Path(command[command.index("--output-dir") + 1]) == destination / "output"
    assert env["CUDA_VISIBLE_DEVICES"] == "3" and env["PYTHONHASHSEED"] == "42"
    assert "WORLD_SIZE" not in env
    if mode == "profile":
        assert command[command.index("--profile-images") + 1] == "5"
    else:
        assert "--profile-images" not in command
    request = json.loads((destination / "request.json").read_text())
    assert request["original_results_preserved"] is True
    assert request["training_hyperparameters_changed"] is False


def test_val_rejects_wrong_model_before_running(tmp_path, monkeypatch):
    run_dir, _ = make_saved_job(tmp_path)
    monkeypatch.setattr(recovery, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(ValueError, match="not 'mask_rcnn_r50'"):
        OfficialBaseline("mask_rcnn_r50").val(run_dir)


def test_profile_dry_run_has_no_writes(tmp_path, monkeypatch):
    run_dir, _ = make_saved_job(tmp_path)
    before = {path.name: path.read_bytes() for path in run_dir.iterdir()}
    monkeypatch.setattr(recovery, "run_visible", lambda *a: pytest.fail("dry-run must not launch"))
    destination = OfficialBaseline("rtmdet_ins_tiny").profile(
        run_dir, images=2, pythons={"mmdet": sys.executable}, dry_run=True,
    )
    assert not destination.exists()
    assert {path.name: path.read_bytes() for path in run_dir.iterdir()} == before


def test_recovery_cli_keeps_overrides_and_calls_direct_runner(monkeypatch):
    received = {}
    settings = dict(RUN_DIR="old", MODE="profile", CHECKPOINT="old.pth", PROFILE_IMAGES=5)
    monkeypatch.setattr(recovery, "run", lambda settings, **options: received.update(settings, **options))
    monkeypatch.setattr(sys, "argv", [
        "recover", "--run-dir", "new", "--mode", "evaluate", "--checkpoint", "new.pth",
        "--profile-images", "2", "--dry-run",
    ])
    recovery.main(settings)
    assert received == dict(RUN_DIR="new", MODE="evaluate", CHECKPOINT="new.pth", PROFILE_IMAGES=2, dry_run=True)
    assert settings["RUN_DIR"] == "old"


def test_import_and_dry_run_do_not_import_ml_frameworks_or_write(tmp_path):
    root = Path(__file__).resolve().parent.parent
    program = """
import builtins
import sys
sys.path.insert(0, sys.argv[1])
original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'torch', 'numpy', 'PIL', 'ultralytics', 'mmdet', 'mmengine', 'rfdetr'}:
        raise AssertionError('Unexpected framework import: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
from baseline_amp import OfficialBaseline
OfficialBaseline('rtmdet_ins_tiny').train('not-present.yaml', sys.argv[2], dry_run=True)
"""
    project = tmp_path / "not_created"
    result = subprocess.run(
        [sys.executable, "-I", "-c", program, str(root), str(project)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "rtmdet_ins_tiny_amp" in result.stdout and "2 jobs" in result.stdout
    assert not project.exists()


@pytest.mark.parametrize("suite,jobs", (("non_yolo", 8), ("mmdet", 6), ("rfdetr", 2)))
def test_current_script_suite_override_is_usable_without_changing_settings(tmp_path, suite, jobs):
    script = Path(__file__).resolve().parent.parent / "RUN_PAPER_BASELINES.py"
    project = tmp_path / "dry_run_only"
    result = subprocess.run(
        [sys.executable, str(script), "--dry-run", "--suite", suite, "--project", str(project)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert f"{jobs} jobs" in result.stdout
    assert "COCO transfer + native optimizer" in result.stdout
    assert "official_transfer_native_citrus_commonval_v1_20261009" in result.stdout
    assert "yolo11n_seg_amp" not in result.stdout
    assert not project.exists()


def test_current_all_entry_keeps_paired_amp_and_retired_launchers_absent(tmp_path):
    root = Path(__file__).resolve().parent.parent
    for retired in ("RUN_CITRUS_BASELINES_AMP.py", "RUN_BASELINE_OTHER_MODELS.py"):
        assert not (root / retired).exists()
    project = tmp_path / "dry_run_only"
    result = subprocess.run(
        [sys.executable, str(root / "RUN_PAPER_BASELINES.py"), "--dry-run", "--project", str(project)],
        text=True, capture_output=True, check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "14 jobs" in result.stdout and "COCO transfer + native optimizer" in result.stdout
    for model in (
        "yolo11n_seg", "yolov8n_seg", "yolo26n_seg", "rtmdet_ins_tiny",
        "mask_rcnn_r50", "solov2_light_r18", "rfdetr_seg_nano",
    ):
        for amp in (0, 1):
            assert f"transfer_{model}_amp{amp}_seed42_300ep" in result.stdout
    assert str(project.resolve()) in result.stdout
    assert not project.exists()
