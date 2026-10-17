"""Failure injection for controller state, atomic records and project-only writer locks."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from baseline_amp import batch, common
from baseline_amp.batch_state import LOCK_NAME, inspect_job, project_lock, selected_jobs
from baseline_amp.registry import COMMON_VALIDATION, make_queue


@pytest.fixture
def settings(tmp_path):
    data = tmp_path / "data.yaml"
    data.write_text("names: [orange_immature]\n", encoding="utf-8")
    return dict(
        DATA=str(data), PROJECT=str(tmp_path / "project"), SUITE="mmdet", SEEDS=[42],
        EPOCHS=3, WORKERS=0, BATCHES={}, DEVICE=1, PYTHONS={"mmdet": sys.executable},
        AMP_MODES=[0], VALIDATION=dict(COMMON_VALIDATION),
    )


def jobs(settings):
    project = Path(settings["PROJECT"])
    queue = make_queue(
        settings["SUITE"], settings["SEEDS"], settings["EPOCHS"], settings["WORKERS"],
        settings["BATCHES"], settings["AMP_MODES"], settings["VALIDATION"],
    )
    for job in queue:
        job.update(run_dir=str(project / job["name"]), prepared=str(project / "_prepared"))
    return queue


def native_trained(job):
    run_dir = Path(job["run_dir"])
    run_dir.mkdir(parents=True)
    common.save_json(run_dir / "job.json", job)
    checkpoint = run_dir / "checkpoint.pth"
    checkpoint.write_bytes(b"mock checkpoint, never loaded")
    common.save_json(run_dir / "trained.json", dict(checkpoint=str(checkpoint)))


def native_complete(job):
    run_dir = Path(job["run_dir"])
    metrics = dict(mask_ap_50_95=0.5, mask_ap_50=0.8, mask_ap_small=-1,
                   mask_ar_100=0.7, mask_precision=0.8, mask_recall=0.7, params=100)
    common.save_json(run_dir / "complete.json", dict(job=job, metrics=metrics))
    for filename, record in (
        ("amp_actual.json", dict(dtype="float32")),
        ("initialization.json", dict(sha256="same")), ("dataset.json", dict(signature="same")),
    ):
        common.save_json(run_dir / filename, record)


def fake_prepare(command, calls):
    args = [str(item) for item in command]
    calls.append(args)
    if "--prepare" in args:
        prepared = Path(args[args.index("--prepared") + 1])
        common.save_json(prepared / "summary.json", dict(splits=["train", "val", "test"]))
    return args


@pytest.mark.parametrize("values", [dict(START_FROM="typo"), dict(SKIP_RUNS=["typo"]), dict(SKIP_RUNS="all")])
def test_invalid_skips_fail_before_imports_or_writes(settings, monkeypatch, values):
    settings.update(values)
    monkeypatch.setattr(batch, "run_visible", lambda *args: pytest.fail("must not run a child"))
    with pytest.raises(ValueError):
        batch.run(settings, dry_run=True)
    assert not Path(settings["PROJECT"]).exists()


def test_all_explicitly_skipped_launches_nothing(settings, monkeypatch):
    settings["SKIP_RUNS"] = [job["name"] for job in jobs(settings)]
    monkeypatch.setattr(batch, "run_visible", lambda *args: pytest.fail("must not run a child"))
    batch.run(settings)
    assert not Path(settings["PROJECT"]).exists()


def test_same_project_lock_is_exclusive_but_other_project_is_allowed(tmp_path):
    with project_lock(tmp_path / "one"):
        with project_lock(tmp_path / "two"):
            pass
        with pytest.raises(RuntimeError, match="Another launcher"):
            with project_lock(tmp_path / "one"):
                pytest.fail("same project cannot have two writers")
    # File may remain: kernel ownership, not mere file existence, determines liveness.
    assert (tmp_path / "one" / LOCK_NAME).is_file()
    with project_lock(tmp_path / "one"):
        pass


def test_project_lock_is_released_after_exception(tmp_path):
    with pytest.raises(KeyboardInterrupt):
        with project_lock(tmp_path):
            raise KeyboardInterrupt()
    with project_lock(tmp_path):
        pass


def test_project_lock_blocks_a_second_real_python(tmp_path):
    root = Path(__file__).resolve().parent.parent
    program = (
        "import sys; sys.path.insert(0, sys.argv[1]); "
        "from baseline_amp.batch_state import project_lock; "
        "\nwith project_lock(sys.argv[2]): pass\n"
    )
    with project_lock(tmp_path):
        result = subprocess.run(
            [sys.executable, "-I", "-c", program, str(root), str(tmp_path)],
            capture_output=True, text=True, timeout=15,
        )
    assert result.returncode != 0 and "Another launcher" in result.stderr


def test_failed_atomic_publish_keeps_original_and_cleans_only_its_temp(tmp_path, monkeypatch):
    record = tmp_path / "complete.json"
    common.save_json(record, dict(value="old"))
    before = record.read_bytes()

    def fail(*args):
        raise OSError("injected full disk / rename failure")

    monkeypatch.setattr(os, "replace", fail)
    with pytest.raises(OSError, match="injected"):
        common.save_json(record, dict(value="new"))
    assert record.read_bytes() == before
    assert sorted(path.name for path in tmp_path.iterdir()) == ["complete.json"]


def test_nan_cannot_be_written_as_an_experiment_record(tmp_path):
    record = tmp_path / "metrics.json"
    common.save_json(record, dict(value=0.5))
    with pytest.raises(ValueError):
        common.save_json(record, dict(value=float("nan")))
    assert json.loads(record.read_text()) == dict(value=0.5)


def test_job_state_requires_real_artifacts_not_just_exit_zero(settings):
    job = jobs(settings)[0]
    assert inspect_job(job) == "train"
    native_trained(job)
    assert inspect_job(job) == "evaluate"
    native_complete(job)
    assert inspect_job(job) == "complete"
    Path(job["run_dir"], "checkpoint.pth").unlink()
    with pytest.raises(FileNotFoundError, match="checkpoint is missing"):
        inspect_job(job)


def test_complete_record_with_missing_provenance_is_not_skipped(settings):
    job = jobs(settings)[0]
    native_trained(job)
    native_complete(job)
    Path(job["run_dir"], "initialization.json").unlink()
    with pytest.raises(RuntimeError, match="initialization.json"):
        inspect_job(job)


@pytest.mark.parametrize("key,value", [
    ("mask_ap_50_95", -1000), ("mask_ap_50", 1.5), ("mask_ap_small", -2),
    ("mask_precision", -0.1), ("mask_recall", 1.001),
    ("params", 0), ("params", -1), ("params", 3.5),
])
def test_invalid_metric_ranges_cannot_mark_job_complete(settings, key, value):
    job = jobs(settings)[0]
    native_trained(job)
    native_complete(job)
    path = Path(job["run_dir"]) / "complete.json"
    record = json.loads(path.read_text())
    record["metrics"][key] = value
    common.save_json(path, record)
    with pytest.raises(RuntimeError, match="Out-of-range metric"):
        inspect_job(job)


def test_evaluated_checkpoint_must_match_trained_when_recorded(settings):
    job = jobs(settings)[0]
    native_trained(job)
    native_complete(job)
    path = Path(job["run_dir"]) / "complete.json"
    record = json.loads(path.read_text())
    record["metrics"]["checkpoint"] = str(path.parent / "different_missing.pth")
    common.save_json(path, record)
    with pytest.raises(RuntimeError, match="Evaluated checkpoint does not match"):
        inspect_job(job)
    record["metrics"]["checkpoint"] = str(path.parent / "checkpoint.pth")
    common.save_json(path, record)
    assert inspect_job(job) == "complete"


def test_later_incomplete_job_is_detected_before_any_gpu_preflight(settings, monkeypatch):
    # Establish matching protocol through a mocked preflight, then create a bad later job.
    monkeypatch.setattr(batch, "run_visible", lambda *args: None)
    batch.run(settings, preflight_only=True)
    last_job = jobs(settings)[-1]
    Path(last_job["run_dir"]).mkdir()
    monkeypatch.setattr(batch, "run_visible", lambda *args: pytest.fail("must fail before any GPU work"))
    with pytest.raises(RuntimeError, match="Interrupted/incomplete"):
        batch.run(settings)


@pytest.mark.parametrize("failure", [RuntimeError("native training failed"), KeyboardInterrupt()])
def test_child_failure_records_status_and_never_starts_next_job(settings, monkeypatch, failure):
    calls = []

    def visible(command, *args):
        arguments = fake_prepare(command, calls)
        if "--stage" in arguments:
            raise failure

    monkeypatch.setattr(batch, "run_visible", visible)
    with pytest.raises(type(failure)):
        batch.run(settings)
    stages = [call for call in calls if "--stage" in call]
    assert len(stages) == 1 and stages[0][-1] == "train"
    status = json.loads((Path(settings["PROJECT"]) / "batch_status.json").read_text())
    assert status["status"] == ("interrupted" if isinstance(failure, KeyboardInterrupt) else "failed")
    assert status["job"] == jobs(settings)[0]["name"] and status["stage"] == "train"
    assert status["completed"] == []
    with project_lock(Path(settings["PROJECT"])):
        pass


def test_success_exit_without_trained_artifact_stops_queue(settings, monkeypatch):
    calls = []
    monkeypatch.setattr(batch, "run_visible", lambda command, *args: fake_prepare(command, calls))
    with pytest.raises(RuntimeError, match="without valid evaluate artifacts"):
        batch.run(settings)
    assert len([call for call in calls if "--stage" in call]) == 1


def test_existing_trained_job_only_evaluates_not_retrains(settings, monkeypatch):
    calls = []
    job = jobs(settings)[0]
    settings["SKIP_RUNS"] = [item["name"] for item in jobs(settings)[1:]]
    monkeypatch.setattr(batch, "run_visible", lambda *args: None)
    batch.run(settings, preflight_only=True)
    native_trained(job)

    def visible(command, *args):
        arguments = fake_prepare(command, calls)
        if "--stage" in arguments:
            assert arguments[-1] == "evaluate"
            native_complete(job)

    monkeypatch.setattr(batch, "run_visible", visible)
    batch.run(settings)
    assert [call[-1] for call in calls if "--stage" in call] == ["evaluate"]
    status = json.loads((Path(settings["PROJECT"]) / "batch_status.json").read_text())
    assert status["status"] == "completed" and status["completed"] == [job["name"]]
    assert len(status["user_skipped"]) == 2


def test_selection_preserves_original_order(settings):
    queue = jobs(settings)
    active, skipped = selected_jobs(queue, queue[1]["name"], [queue[2]["name"]])
    assert active == [queue[1]] and skipped == [queue[0], queue[2]]
