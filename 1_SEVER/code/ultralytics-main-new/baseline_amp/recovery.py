"""Use saved experiment provenance; no dataset re-conversion or changed training recipe."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

from .batch import run_visible
from .common import save_json


def find_checkpoint(run_dir):
    """Prefer selected weights; otherwise MMEngine's explicit latest checkpoint marker."""
    trained = run_dir / "trained.json"
    if trained.is_file():
        checkpoint = Path(json.loads(trained.read_text(encoding="utf-8"))["checkpoint"])
    else:
        marker = run_dir / "train/last_checkpoint"
        if not marker.is_file():
            raise FileNotFoundError(
                "Set CHECKPOINT or provide train/last_checkpoint; no checkpoint is chosen by guessing."
            )
        checkpoint = Path(marker.read_text(encoding="utf-8").strip())
        if not checkpoint.is_absolute():
            checkpoint = run_dir / "train" / checkpoint
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    return checkpoint.resolve()


def main(settings):
    """Keep the existing CLI while exposing run() to direct Python callers."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir")
    parser.add_argument("--mode", choices=("profile", "resume", "evaluate"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--profile-images", type=int)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    settings = dict(settings)
    for key, value in (
        ("RUN_DIR", args.run_dir),
        ("MODE", args.mode),
        ("CHECKPOINT", args.checkpoint),
        ("PROFILE_IMAGES", args.profile_images),
    ):
        if value is not None:
            settings[key] = value
    if not settings["RUN_DIR"]:
        parser.error("Set SETTINGS['RUN_DIR'] to the existing MODEL run directory, not the batch PROJECT.")
    return run(settings, dry_run=args.dry_run)


def run(settings, *, dry_run=False):
    """Run one saved job without reading or changing process command-line args."""
    run_value = settings["RUN_DIR"]
    if not run_value:
        raise ValueError("Set RUN_DIR to the existing MODEL run directory, not the batch PROJECT.")
    run_dir = Path(run_value).expanduser().resolve()
    job_path = run_dir / "job.json"
    job = json.loads(job_path.read_text(encoding="utf-8"))
    if Path(job["run_dir"]).resolve() != run_dir:
        raise ValueError("Saved run_dir differs from RUN_DIR. Use the experiment's original location.")
    mode = settings["MODE"]
    if mode not in ("profile", "resume", "evaluate"):
        raise ValueError(f"Unknown MODE: {mode}")
    count = settings["PROFILE_IMAGES"]
    if mode == "profile" and count < 1:
        raise ValueError("PROFILE_IMAGES must be positive")
    if mode == "resume" and job["recipe"]["family"] != "mmdet":
        raise ValueError("Training resume is implemented for MMDetection; profile/evaluate support all families.")
    if mode == "resume" and (run_dir / "complete.json").is_file():
        raise ValueError("This job is complete; use evaluate to preserve it, not resume.")
    python = Path(settings["PYTHONS"][job["recipe"]["environment"]]).expanduser().resolve()
    if not python.is_file():
        raise FileNotFoundError(f"Set PYTHONS to the appropriate installed environment: {python}")
    device = str(settings["DEVICE"])
    if not device.isdigit():
        raise ValueError("DEVICE must be one physical GPU index")
    checkpoint_value = settings["CHECKPOINT"]
    if mode == "resume" and checkpoint_value:
        raise ValueError("Resume uses train/last_checkpoint including optimizer state; leave CHECKPOINT blank.")
    checkpoint = Path(checkpoint_value).expanduser().resolve() if checkpoint_value else find_checkpoint(run_dir)
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    destination = run_dir / ("runtime_" + mode + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    worker = Path(__file__).resolve().parent / "worker.py"
    prefix = [python, "-I", "-u", worker, "--job", job_path]
    if mode == "resume":
        commands = [prefix + ["--stage", "train", "--resume-train"], prefix + ["--stage", "evaluate"]]
    else:
        commands = [
            prefix + ["--stage", "evaluate", "--checkpoint", checkpoint, "--output-dir", destination / "output"]
        ]
        if mode == "profile":
            commands[0] += ["--profile-images", str(count)]
    print(
        "BASELINE RECOVERY:",
        mode,
        "\nRun:",
        run_dir,
        "\nCheckpoint:",
        checkpoint,
        "\nGPU:",
        device,
        "\nOutput:",
        destination,
        flush=True,
    )
    print("Stop the old training process BEFORE using resume. No unrelated process is killed.", flush=True)
    if dry_run:
        for command in commands:
            print([str(value) for value in command])
        return destination
    destination.mkdir(parents=True, exist_ok=False)
    save_json(
        destination / "request.json",
        dict(
            mode=mode,
            checkpoint=str(checkpoint),
            job=job,
            training_hyperparameters_changed=False,
            original_results_preserved=True,
        ),
    )
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
        PYTHONHASHSEED=str(job["seed"]),
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(variable, None)
    for command in commands:
        run_visible(command, run_dir, env, destination / "recovery.log")
    print("DONE:", destination, flush=True)
    return destination
