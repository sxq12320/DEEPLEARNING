"""Foreground entry/API contracts; all child processes are mocked, never trained."""

import copy
import json
import sys
from pathlib import Path

import pytest

from baseline_amp import batch
from baseline_amp.registry import COMMON_VALIDATION, make_queue


@pytest.fixture
def settings(tmp_path):
    return dict(
        DATA=str(tmp_path / "missing_data.yaml"),
        PROJECT=str(tmp_path / "project"),
        SUITE="all",
        SEEDS=[42, 43],
        EPOCHS=300,
        WORKERS=4,
        BATCHES={},
        DEVICE=1,
        PYTHONS={"modern": str(tmp_path / "missing_modern"), "mmdet": str(tmp_path / "missing_mmdet")},
        AMP_MODES=[1, 0],
        VALIDATION=copy.deepcopy(COMMON_VALIDATION),
    )


def forbidden(*args, **kwargs):
    pytest.fail("Dry/API entry unexpectedly parsed global CLI, checked files, or launched/wrote output")


def expected_queue(settings, models=None):
    queue = make_queue(
        settings["SUITE"], settings["SEEDS"], settings["EPOCHS"], settings["WORKERS"], settings["BATCHES"],
        settings["AMP_MODES"], validation=settings["VALIDATION"],
    )
    return [job for job in queue if models is None or job["model"] in models]


def test_api_dry_run_ignores_global_cli_checks_no_dependencies_and_writes_nothing(settings, monkeypatch, capsys):
    before = copy.deepcopy(settings)
    monkeypatch.setattr(sys, "argv", ["unrelated_vscode_file.py", "--unrelated-option"])
    monkeypatch.setattr(batch.argparse, "ArgumentParser", forbidden)
    monkeypatch.setattr(batch, "run_visible", forbidden)
    monkeypatch.setattr(batch, "save_json", forbidden)
    monkeypatch.setattr(batch, "summarize", forbidden)
    with monkeypatch.context() as checks:
        checks.setattr(Path, "is_file", forbidden)
        batch.run(settings, dry_run=True)
    output = capsys.readouterr().out
    printed = [line.split(". ", 1)[1].split(" |", 1)[0] for line in output.splitlines() if ". " in line[:5]]
    assert printed == [job["name"] for job in expected_queue(settings)]
    assert settings == before and not Path(settings["PROJECT"]).exists()


def test_selected_models_keep_suite_order_and_original_amp_pair_order(settings, monkeypatch, capsys):
    selected = ["mask_rcnn_r50", "rtmdet_ins_tiny"]  # Deliberately opposite to suite order.
    monkeypatch.setattr(batch, "run_visible", forbidden)
    batch.run(settings, models=selected, dry_run=True)
    output = capsys.readouterr().out
    printed = [line.split(". ", 1)[1].split(" |", 1)[0] for line in output.splitlines() if ". " in line[:5]]
    expected = expected_queue(settings, selected)
    assert printed == [job["name"] for job in expected]
    assert [int(job["amp"]) for job in expected[:4]] == [0, 1, 1, 0]
    assert "rfdetr_seg_nano_amp" not in output
    assert not Path(settings["PROJECT"]).exists()


@pytest.mark.parametrize(
    "models,message",
    [
        ([], "nonempty"),
        ("rtmdet_ins_tiny", "nonempty"),
        ([None], "nonempty"),
        ({"rtmdet_ins_tiny"}, "nonempty"),
        (["rtmdet_ins_tiny", "rtmdet_ins_tiny"], "duplicate"),
        (["new_unknown_model"], "Unknown original models"),
    ],
)
def test_model_selection_fails_before_queue_or_any_side_effect(settings, monkeypatch, models, message):
    monkeypatch.setattr(batch, "make_queue", forbidden)
    monkeypatch.setattr(batch, "run_visible", forbidden)
    monkeypatch.setattr(batch, "save_json", forbidden)
    with pytest.raises(ValueError, match=message):
        batch.run(settings, models=models)
    assert not Path(settings["PROJECT"]).exists()


def test_model_outside_suite_fails_fast(settings, monkeypatch):
    settings["SUITE"] = "yolo"
    monkeypatch.setattr(batch, "make_queue", forbidden)
    with pytest.raises(ValueError, match="not in suite 'yolo'"):
        batch.run(settings, models=["rtmdet_ins_tiny"])
    assert not Path(settings["PROJECT"]).exists()


def test_selected_models_require_valid_suite(settings, monkeypatch):
    settings["SUITE"] = "invalid"
    monkeypatch.setattr(batch, "make_queue", forbidden)
    with pytest.raises(ValueError, match="Invalid suite"):
        batch.run(settings, models=["rtmdet_ins_tiny"])


def test_main_keeps_existing_cli_and_does_not_mutate_settings(settings, monkeypatch, tmp_path):
    before = copy.deepcopy(settings)
    captured = []
    monkeypatch.setattr(batch, "run", lambda config, **options: captured.append((config, options)))
    project = str(tmp_path / "cli_project")
    monkeypatch.setattr(
        sys, "argv", ["existing_entry.py", "--suite", "non_yolo", "--project", project, "--dry-run",
                      "--preflight-only", "--summarize-only"],
    )
    batch.main(settings)
    expected = dict(before, SUITE="non_yolo", PROJECT=project)
    assert captured == [(expected, dict(dry_run=True, preflight_only=True, summarize_only=True, models=None))]
    assert settings == before


@pytest.mark.parametrize("cli_models", [None, ["mask_rcnn_r50", "solov2_light_r18"]])
def test_main_reads_config_models_and_cli_can_override(settings, monkeypatch, cli_models):
    settings["MODELS"] = ["rtmdet_ins_tiny"]
    captured = []
    monkeypatch.setattr(batch, "run", lambda config, **options: captured.append((config, options)))
    arguments = ["existing_entry.py", "--dry-run"]
    if cli_models is not None:
        arguments.extend(["--models", *cli_models])
    monkeypatch.setattr(sys, "argv", arguments)
    batch.main(settings)
    assert captured[0][1]["models"] == (cli_models if cli_models is not None else settings["MODELS"])
    assert settings["MODELS"] == ["rtmdet_ins_tiny"]


def test_api_reads_config_models_without_parsing_cli_or_writing(settings, monkeypatch, capsys):
    settings["MODELS"] = ["rtmdet_ins_tiny"]
    monkeypatch.setattr(batch.argparse, "ArgumentParser", forbidden)
    monkeypatch.setattr(batch, "run_visible", forbidden)
    monkeypatch.setattr(batch, "save_json", forbidden)
    batch.run(settings, dry_run=True)
    output = capsys.readouterr().out
    assert "rtmdet_ins_tiny_amp" in output and "mask_rcnn_r50_amp" not in output
    assert not Path(settings["PROJECT"]).exists()


def test_summarize_only_works_without_training_settings(tmp_path, monkeypatch):
    captured = []
    monkeypatch.setattr(batch, "summarize", lambda project: captured.append(project))
    monkeypatch.setattr(batch, "make_queue", forbidden)
    project = tmp_path / "existing"
    batch.run(dict(PROJECT=str(project)), summarize_only=True)
    assert captured == [project.resolve()]
    assert not project.exists()


def test_dry_run_prevents_summarize_writes_too(tmp_path, monkeypatch):
    monkeypatch.setattr(batch, "summarize", forbidden)
    batch.run(dict(PROJECT=str(tmp_path / "project")), dry_run=True, summarize_only=True)
    assert not (tmp_path / "project").exists()


def make_existing_input_files(settings, tmp_path):
    dataset = tmp_path / "data.yaml"
    dataset.write_text("names: [orange_immature]\n", encoding="utf-8")
    settings["DATA"] = str(dataset)
    # A real Python file is sufficient; it is never executed by these tests.
    settings["PYTHONS"]["mmdet"] = sys.executable


def test_preflight_only_checks_selected_environment_without_training(settings, tmp_path, monkeypatch):
    make_existing_input_files(settings, tmp_path)
    calls = []
    monkeypatch.setattr(batch, "run_visible", lambda command, *args: calls.append(command))
    batch.run(settings, preflight_only=True, models=["rtmdet_ins_tiny"])
    assert len(calls) == 1 and calls[0][-2:] == ["--check", "mmdet"]
    protocol = json.loads((Path(settings["PROJECT"]) / "batch_protocol.json").read_text(encoding="utf-8"))
    assert protocol["SUITE"] == "all" and protocol["VALIDATION"] == COMMON_VALIDATION
    assert protocol["AMP_MODES"] == [1, 0] and "MODELS" not in protocol
    assert not (Path(settings["PROJECT"]) / "jobs").exists()


def test_mocked_execution_preserves_original_job_and_stage_protocol(settings, tmp_path, monkeypatch):
    make_existing_input_files(settings, tmp_path)
    stages, checks = [], []

    def visible(command, cwd, env, logfile):
        arguments = [str(item) for item in command]
        assert env["CUDA_VISIBLE_DEVICES"] == "1"
        if "--check" in arguments:
            checks.append(arguments[arguments.index("--check") + 1:])
        elif "--prepare" in arguments:
            prepared = Path(arguments[arguments.index("--prepared") + 1])
            prepared.mkdir()
            (prepared / "summary.json").write_text(json.dumps(dict(splits=["train", "val", "test"])))
        else:
            job = json.loads(Path(arguments[arguments.index("--job") + 1]).read_text(encoding="utf-8"))
            stage = arguments[arguments.index("--stage") + 1]
            stages.append((job, stage))
            run_dir = Path(job["run_dir"])
            if stage == "train":
                run_dir.mkdir()
                (run_dir / "job.json").write_text(json.dumps(job))
                checkpoint = run_dir / "checkpoint.pth"
                checkpoint.write_bytes(b"mock native checkpoint")
                (run_dir / "trained.json").write_text(json.dumps(dict(checkpoint=str(checkpoint))))
            else:
                metrics = dict(
                    mask_ap_50_95=0.5, mask_ap_50=0.8, mask_ap_small=0.2,
                    mask_ar_100=0.7, mask_precision=0.8, mask_recall=0.7, params=100,
                )
                (run_dir / "complete.json").write_text(json.dumps(dict(job=job, metrics=metrics)))
                for filename, record in (
                    ("amp_actual.json", dict(dtype="float16" if job["amp"] else "float32")),
                    ("initialization.json", dict(sha256="same")), ("dataset.json", dict(signature="same")),
                ):
                    (run_dir / filename).write_text(json.dumps(record))

    monkeypatch.setattr(batch, "run_visible", visible)
    selected = ["mask_rcnn_r50", "rtmdet_ins_tiny"]
    batch.run(settings, models=selected)
    expected = expected_queue(settings, selected)
    project = Path(settings["PROJECT"]).resolve()
    for job in expected:
        job.update(run_dir=str(project / job["name"]), prepared=str(project / "_prepared"))
    assert stages == [(job, stage) for job in expected for stage in ("train", "evaluate")]
    assert checks == [["mmdet"]]
    assert (project / "baseline_summary.csv").is_file()
    state = json.loads((project / "batch_status.json").read_text())
    assert state["status"] == "completed" and len(state["completed"]) == len(expected)
