"""Small controller safeguards; no GPU locks, framework imports or dataset mutations."""

from __future__ import annotations

import json
import math
import os
from contextlib import contextmanager
from pathlib import Path

LOCK_NAME = ".baseline_launcher.lock"


@contextmanager
def project_lock(project):
    """Prevent concurrent writers to ONE project; kernel releases the lock on process exit.

    This does not reserve/check GPUs or stop any process. The small lock file is
    intentionally retained; its existence does not imply a lock is held.
    """
    project = Path(project)
    project.mkdir(parents=True, exist_ok=True)
    with (project / LOCK_NAME).open("a+b") as handle:
        if os.name == "nt":
            import msvcrt

            if handle.seek(0, os.SEEK_END) == 0:
                handle.write(b"\0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise RuntimeError(f"Another launcher is writing this PROJECT: {project}. Nothing was stopped.") from error
        else:
            import fcntl

            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError(f"Another launcher is writing this PROJECT: {project}. Nothing was stopped.") from error
        try:
            yield
        finally:
            if os.name == "nt":
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(handle, fcntl.LOCK_UN)


def read_record(path):
    """Read a controller/worker record with an actionable file-specific error."""
    path = Path(path)
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Missing or unreadable experiment record: {path}. Original files were preserved.") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"Experiment record must be a JSON object: {path}")
    return value


def inspect_job(job):
    """Return train/evaluate/complete, rejecting interrupted or inconsistent artifacts early."""
    run_dir = Path(job["run_dir"])
    if not run_dir.exists():
        return "train"
    if not run_dir.is_dir():
        raise RuntimeError(f"Model run path is not a directory: {run_dir}")
    trained = run_dir / "trained.json"
    if not trained.is_file():
        raise RuntimeError(
            f"Interrupted/incomplete training: {run_dir}. Explicitly add '{job['name']}' to SKIP_RUNS "
            "or use a new PROJECT. Nothing was deleted or overwritten."
        )
    if read_record(run_dir / "job.json") != job:
        raise RuntimeError(f"Saved job metadata mismatch: {run_dir}; choose a NEW PROJECT.")
    checkpoint = read_record(trained).get("checkpoint")
    if not isinstance(checkpoint, str) or not checkpoint:
        raise RuntimeError(f"trained.json has no valid checkpoint: {trained}")
    if not Path(checkpoint).expanduser().is_file():
        raise FileNotFoundError(f"Recorded checkpoint is missing: {checkpoint}. Run: {run_dir}")
    complete = run_dir / "complete.json"
    if not complete.is_file():
        return "evaluate"
    record = read_record(complete)
    if record.get("job") != job or not isinstance(record.get("metrics"), dict):
        raise RuntimeError(f"Completed job metadata mismatch: {complete}")
    for key in (
        "mask_ap_50_95", "mask_ap_50", "mask_ap_small", "mask_ar_100",
        "mask_precision", "mask_recall", "params",
    ):
        value = record["metrics"].get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
            raise RuntimeError(f"Invalid/missing metric {key} in {complete}; do not treat this job as complete.")
        if key == "params":
            valid = isinstance(value, int) and value > 0
        else:
            valid = 0 <= value <= 1 or (key == "mask_ap_small" and value == -1)
        if not valid:
            raise RuntimeError(f"Out-of-range metric {key}={value!r} in {complete}")
    if "checkpoint" in record["metrics"]:
        evaluated = record["metrics"]["checkpoint"]
        if (
            not isinstance(evaluated, str) or not evaluated
            or Path(evaluated).expanduser().resolve() != Path(checkpoint).expanduser().resolve()
        ):
            raise RuntimeError(f"Evaluated checkpoint does not match trained.json: {complete}")
    # Summary and paired-AMP attribution require these independently saved records.
    for filename, key in (
        ("amp_actual.json", "dtype"), ("initialization.json", "sha256"), ("dataset.json", "signature"),
    ):
        if not read_record(run_dir / filename).get(key):
            raise RuntimeError(f"Incomplete provenance in {run_dir / filename}: missing {key}")
    return "complete"


def selected_jobs(queue, start, skip):
    """Validate explicit skips before preflight or data preparation."""
    names = {job["name"] for job in queue}
    if not isinstance(start, str) or (start and start not in names):
        raise ValueError(f"START_FROM is not a queue job: {start!r}")
    if not isinstance(skip, (list, tuple)) or any(not isinstance(name, str) for name in skip):
        raise ValueError("SKIP_RUNS must be a list or tuple of exact job names")
    if len(set(skip)) != len(skip):
        raise ValueError("SKIP_RUNS contains duplicate job names")
    if set(skip) - names:
        raise ValueError(f"Unknown SKIP_RUNS: {sorted(set(skip) - names)}")
    active, skipped = [], []
    started = not start
    for job in queue:
        started = started or job["name"] == start
        (active if started and job["name"] not in skip else skipped).append(job)
    return active, skipped
