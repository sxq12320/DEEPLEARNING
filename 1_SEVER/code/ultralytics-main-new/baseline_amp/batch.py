"""Sequential cross-environment foreground execution; this module only needs Python's standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from .batch_state import LOCK_NAME, inspect_job, project_lock, selected_jobs
from .common import save_json
from .process import run_visible
from .registry import MODELS, SUITES, make_queue

HERE = Path(__file__).resolve().parent


def summarize(project):
    rows = []
    for complete in sorted(Path(project).glob("*/complete.json")):
        result = json.loads(complete.read_text(encoding="utf-8"))
        job, metrics = result["job"], result["metrics"]
        actual = json.loads((complete.parent / "amp_actual.json").read_text(encoding="utf-8"))
        rows.append(
            dict(
                model=job["model"],
                protocol=job.get("protocol", "previous_pretrained_recipe"),
                initialization=job.get("initialization", "pretrained"),
                seed=job["seed"],
                amp=int(job["amp"]),
                epochs=job["epochs"],
                batch=job["recipe"]["batch"],
                imgsz=job["recipe"]["imgsz"],
                validation_protocol=job.get("validation", {}).get("id", "legacy_native_selection"),
                amp_dtype=actual.get("dtype"),
                mask_AP50_95=metrics["mask_ap_50_95"],
                mask_AP50=metrics["mask_ap_50"],
                mask_APS=metrics["mask_ap_small"],
                mask_AR100=metrics["mask_ar_100"],
                precision_conf025=metrics["mask_precision"],
                recall_conf025=metrics["mask_recall"],
                params=metrics["params"],
                run=str(complete.parent),
            )
        )
    if not rows:
        return
    keys = [(row["model"], row["seed"], row["epochs"], row["amp"]) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate model/seed/epoch/AMP results; summary was not overwritten.")
    paired = []
    groups = {(row["model"], row["seed"], row["epochs"]) for row in rows}
    for model, seed, epochs in sorted(groups):
        pair = {row["amp"]: row for row in rows if (row["model"], row["seed"], row["epochs"]) == (model, seed, epochs)}
        if set(pair) != {0, 1}:
            continue
        if any(pair[0][key] != pair[1][key] for key in (
            "batch", "imgsz", "protocol", "initialization", "validation_protocol"
        )):
            raise RuntimeError(f"Mismatched AMP pair: {model}")
        for filename, key in (("initialization.json", "sha256"), ("dataset.json", "signature")):
            records = [Path(pair[amp]["run"]) / filename for amp in (0, 1)]
            if not all(record.is_file() for record in records):
                raise RuntimeError(f"Missing pair provenance for {model}: {filename}")
            values = [json.loads(record.read_text(encoding="utf-8"))[key] for record in records]
            if values[0] != values[1]:
                raise RuntimeError(f"AMP pair has different {filename}: {model}; do not attribute this to AMP")
        paired.append(
            dict(
                model=model,
                seed=seed,
                epochs=epochs,
                delta_AP50_pp=100 * (pair[1]["mask_AP50"] - pair[0]["mask_AP50"]),
                delta_AP50_95_pp=100 * (pair[1]["mask_AP50_95"] - pair[0]["mask_AP50_95"]),
                delta_Recall_pp=100 * (pair[1]["recall_conf025"] - pair[0]["recall_conf025"]),
            )
        )
    # Validate all attribution before publishing either summary artifact.
    import io

    from .common import save_text

    content = io.StringIO(newline="")
    writer = csv.DictWriter(content, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    save_text(Path(project) / "baseline_summary.csv", "\ufeff" + content.getvalue())
    save_json(
        Path(project) / "amp_paired_deltas.json", dict(definition="AMP1 minus AMP0, percentage points", pairs=paired)
    )


def run(settings, *, dry_run=False, preflight_only=False, summarize_only=False, models=None):
    """Run the original foreground workflow without reading global command-line arguments.

    ``models`` optionally selects original models already in ``SUITE``. Filter
    the original queue rather than rebuilding it, preserving suite order and
    the established AMP-pair order. No job/training/evaluation protocol changes.
    A dry run only prints; it never checks framework dependencies or writes.
    """
    settings = dict(settings)
    if models is None:
        models = settings.get("MODELS")
    selected = None
    if models is not None:
        if not isinstance(models, (list, tuple)) or not models or any(not isinstance(name, str) for name in models):
            raise ValueError("models must be a nonempty list or tuple of original model names")
        if len(set(models)) != len(models):
            raise ValueError("models contains duplicate names")
        unknown = set(models) - set(MODELS)
        if unknown:
            raise ValueError(f"Unknown original models: {sorted(unknown)}")
        suite = settings.get("SUITE")
        if suite not in SUITES:
            raise ValueError(f"Invalid suite: {suite}")
        outside = set(models) - set(SUITES[suite])
        if outside:
            raise ValueError(f"Models not in suite '{suite}': {sorted(outside)}")
        selected = set(models)
    if not str(settings.get("PROJECT", "")).strip():
        raise ValueError("PROJECT must be an explicit output directory")
    project = Path(settings["PROJECT"]).expanduser().resolve()
    if summarize_only:
        if dry_run:
            print("DRY RUN: summarize existing project:", project)
        else:
            if project.is_dir():
                with project_lock(project):
                    summarize(project)
            else:
                summarize(project)
        return
    queue = make_queue(
        settings["SUITE"], settings["SEEDS"], settings["EPOCHS"], settings["WORKERS"], settings["BATCHES"],
        settings.get("AMP_MODES", [1]),
        validation=settings.get("VALIDATION"),
        recipe_policy=settings.get("RECIPE_POLICY", "legacy78_scratch"),
    )
    if selected is not None:
        queue = [job for job in queue if job["model"] in selected]
    active, skipped = selected_jobs(queue, settings.get("START_FROM", ""), settings.get("SKIP_RUNS", []))
    device = str(settings["DEVICE"])
    if not device.isascii() or not device.isdigit():
        raise ValueError("DEVICE must be ONE physical GPU index, e.g. 1 (not '0,1')")
    # No occupancy guard, no reservations, no killing unrelated processes.
    env = dict(
        os.environ,
        CUDA_VISIBLE_DEVICES=device,
        CUDA_DEVICE_ORDER="PCI_BUS_ID",
        PYTHONUNBUFFERED="1",
        PYTHONUTF8="1",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        OMP_NUM_THREADS="4",
        MKL_NUM_THREADS="4",
        OPENBLAS_NUM_THREADS="4",
    )
    for variable in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(variable, None)
    prepared = project / "_prepared"
    for job in queue:
        job.update(run_dir=str(project / job["name"]), prepared=str(prepared))
    print(f"FOREGROUND BASELINES | {len(queue)} jobs | GPU physical {device} -> logical cuda:0")
    print("Data:", settings["DATA"], "\nProject:", project)
    protocol_id = queue[0]["protocol"]
    native = settings.get("RECIPE_POLICY") == "official_transfer"
    print("Protocol:", protocol_id,
          "| COCO transfer + native optimizer" if native else "| SCRATCH initialization | AdamW lr=0.001")
    if settings.get("VALIDATION"):
        print("COMMON VALIDATION: FP32 / quick640 every epoch / original epoch1, every10, last / COCO maxDets100")
        print("Validation engine: bounded CPU prefetch + cached GT/IoU; evaluation protocol unchanged")
        print("Selection: original Mask AP50-95 only; ties keep earlier epoch; fixed budget, no early stopping")
    else:
        print("LEGACY selection: native framework validation; patience=100")
    print("Separate official-transfer track; fixed citrus resize/epochs and common validation are adaptations."
          if native else "Training uses legacy78 knobs; validation changes selection/stopping, not native losses.")
    print("Cross-framework batch/resize/AMP/EMA differences remain declared; see baseline_amp/README.")
    for i, job in enumerate(queue, 1):
        suffix = " | USER-SKIPPED" if job in skipped else ""
        print(f"{i:02d}. {job['name']} | batch={job['recipe']['batch']} | size={job['recipe']['imgsz']}{suffix}")
    if dry_run:
        return
    if not active:
        print("No jobs selected after START_FROM/SKIP_RUNS. Nothing was launched or written.")
        return
    if not isinstance(settings.get("PYTHONS"), dict):
        raise ValueError("PYTHONS must map environment names to interpreter paths")
    interpreters = {key: str(Path(value).expanduser().resolve()) for key, value in settings["PYTHONS"].items()}
    used_envs = {job["recipe"]["environment"] for job in active}
    for environment in used_envs:
        if environment not in interpreters or not Path(interpreters[environment]).is_file():
            raise FileNotFoundError(
                f"Set PYTHONS['{environment}'] to its installed Python: {interpreters.get(environment, '(missing)')}"
            )
    if not str(settings.get("DATA", "")).strip() or not Path(settings["DATA"]).expanduser().is_file():
        raise FileNotFoundError(f"Set DATA to your cleaned data.yaml: {settings['DATA']}")
    # A child uses a different cwd; relative input paths must not silently change meaning.
    settings["DATA"] = str(Path(settings["DATA"]).expanduser().resolve())
    settings["PYTHONS"] = interpreters
    _check_project(project)
    with project_lock(project):
        _execute(settings, project, queue, active, skipped, interpreters, env, preflight_only)


def _check_project(project):
    if project.exists() and not project.is_dir():
        raise FileExistsError(f"PROJECT is not a directory: {project}")
    if (
        project.exists() and any(path.name != LOCK_NAME for path in project.iterdir())
        and not (project / "batch_protocol.json").is_file()
    ):
        raise FileExistsError(f"Project already contains unrelated results: {project}; choose a new path")


def _execute(settings, project, queue, active, skipped, interpreters, env, preflight_only):
    """Execute under the project writer lock; verify every native-stage result."""
    _check_project(project)  # Recheck after acquisition, including a concurrent startup race.
    worker, prepared = HERE / "worker.py", project / "_prepared"
    used_envs = {job["recipe"]["environment"] for job in active}
    protocol_id = queue[0]["protocol"]
    protocol = {
        key: settings[key] for key in ("DATA", "SUITE", "SEEDS", "EPOCHS", "WORKERS", "BATCHES", "DEVICE", "PYTHONS")
    }
    protocol["protocol_id"] = protocol_id
    protocol["VALIDATION"] = settings.get("VALIDATION")
    protocol["AMP_MODES"] = settings.get("AMP_MODES", [1])
    if settings.get("RECIPE_POLICY") == "official_transfer":
        protocol["RECIPE_POLICY"] = "official_transfer"
    digest = hashlib.sha256()
    for source_file in sorted(HERE.glob("*.py")) + sorted(HERE.glob("requirements*.txt")):
        if source_file.name.startswith("test_"):
            continue
        digest.update(source_file.name.encode())
        digest.update(source_file.read_bytes())
    protocol["implementation_sha256"] = digest.hexdigest()
    marker = project / "batch_protocol.json"
    if marker.exists() and json.loads(marker.read_text(encoding="utf-8")) != protocol:
        raise RuntimeError("Existing PROJECT has a different protocol. Choose a NEW PROJECT rather than overwrite.")
    save_json(marker, protocol)
    weights = project / "_pretrained"
    weights.mkdir(exist_ok=True)
    status = dict(
        status="running", pid=os.getpid(), stage="startup", job=None,
        queue=[job["name"] for job in active], completed=[], already_complete=[],
        user_skipped=[job["name"] for job in skipped],
    )

    def update(**values):
        status.update(values, updated_at=datetime.now(timezone.utc).isoformat())
        save_json(project / "batch_status.json", status)

    try:
        if not preflight_only:
            update(stage="validate_saved_jobs")
            # Check the ENTIRE requested queue before spending time on backend smoke runs.
            for job in active:
                inspect_job(job)
        for environment in sorted(used_envs):
            update(stage="preflight", job=None, environment=environment)
            families = sorted({job["recipe"]["family"] for job in active if job["recipe"]["environment"] == environment})
            run_visible(
                [interpreters[environment], "-I", "-u", worker, "--check", *families],
                weights, env, project / "logs" / f"preflight_{environment}.log",
            )
        if preflight_only:
            update(status="preflight_passed", stage="done", environment=None)
            return
        update(stage="prepare", environment=None)
        prepare_python = interpreters[active[0]["recipe"]["environment"]]
        run_visible(
            [prepare_python, "-I", "-u", worker, "--prepare", settings["DATA"], "--prepared", prepared],
            weights, env, project / "logs/prepare.log",
        )
        source = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
        if any(job["recipe"]["family"] == "rfdetr" for job in active) and "test" not in source["splits"]:
            raise ValueError(
                "RF-DETR 1.4 constructs a test loader even with run_test=False. Provide the REAL held-out test "
                "split; never substitute training/validation images."
            )
        for job in skipped:
            print("USER-SKIPPED:", job["name"])
        for job in active:
            state = inspect_job(job)
            if state == "complete":
                status["already_complete"].append(job["name"])
                print("ALREADY COMPLETE:", job["name"])
                update(stage="skip_complete", job=job["name"])
                continue
            job_path = project / "jobs" / f"{job['name']}.json"
            save_json(job_path, job)
            begin = time.monotonic()
            stages = ("train", "evaluate") if state == "train" else ("evaluate",)
            for stage in stages:
                update(stage=stage, job=job["name"], log=str(project / "logs" / f"{job['name']}.log"))
                run_visible(
                    [interpreters[job["recipe"]["environment"]], "-I", "-u", worker,
                     "--job", job_path, "--stage", stage],
                    weights, dict(env, PYTHONHASHSEED=str(job["seed"])),
                    project / "logs" / f"{job['name']}.log",
                )
                expected = "evaluate" if stage == "train" else "complete"
                if inspect_job(job) != expected:
                    raise RuntimeError(f"{job['name']} {stage} exited without valid {expected} artifacts; queue stopped.")
            status["completed"].append(job["name"])
            print(f"FINISHED {job['name']} ({(time.monotonic() - begin) / 60:.1f} minutes)", flush=True)
            update(stage="summarize")
            summarize(project)
        summarize(project)
        update(status="completed", stage="done", job=None)
        print("Queue finished. Read baseline_summary.csv and amp_paired_deltas.json; skipped jobs are not results.")
    except BaseException as error:
        try:
            update(status="interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed",
                   error_type=type(error).__name__, error=str(error))
        except Exception:
            pass  # Preserve the original error if a full disk also prevents recording status.
        raise


def main(settings):
    """Preserve the existing CLI while delegating execution to the Python API."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the queue only; no imports/downloads/training")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--suite", choices=tuple(SUITES), help="Override the model queue; non_yolo skips all YOLO jobs")
    parser.add_argument("--project", help="Override output directory without changing your DATA/PYTHONS settings")
    parser.add_argument("--models", nargs="+", help="Select original models within SUITE; preserves suite/AMP order")
    parser.add_argument("--data", help="Actual cleaned data.yaml; relative paths are resolved in this terminal")
    parser.add_argument("--device", type=int, help="One physical GPU index; no occupancy protection")
    parser.add_argument("--epochs", type=int, help="Use 1-3 with a NEW project for a smoke run")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--seeds", nargs="+", type=int)
    parser.add_argument("--amp-modes", nargs="+", type=int, choices=(0, 1))
    parser.add_argument("--start-from", help="Exact queue job name; skip earlier jobs, not checkpoint resume")
    parser.add_argument("--skip-runs", nargs="*", help="Exact incomplete job names to skip explicitly")
    args = parser.parse_args()
    settings = dict(settings)
    if args.suite is not None:
        settings["SUITE"] = args.suite
    if args.project is not None:
        settings["PROJECT"] = args.project
    for key, value in (
        ("DATA", args.data), ("DEVICE", args.device), ("EPOCHS", args.epochs), ("WORKERS", args.workers),
        ("SEEDS", args.seeds), ("AMP_MODES", args.amp_modes),
        ("START_FROM", args.start_from), ("SKIP_RUNS", args.skip_runs),
    ):
        if value is not None:
            settings[key] = value
    models = args.models if args.models is not None else settings.get("MODELS")
    return run(settings, dry_run=args.dry_run, preflight_only=args.preflight_only,
               summarize_only=args.summarize_only, models=models)
