"""Sequential cross-environment foreground execution; this module only needs Python's standard library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import signal
import subprocess
import time
from pathlib import Path

from .common import save_json
from .registry import make_queue

HERE = Path(__file__).resolve().parent


def run_visible(command, cwd, env, logfile):
    """Tee one child into this terminal; Ctrl+C stops only this child group and aborts the queue."""
    logfile.parent.mkdir(parents=True, exist_ok=True)
    print("COMMAND:", subprocess.list2cmdline([str(arg) for arg in command]), flush=True)
    kwargs = dict(
        cwd=str(cwd),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    with logfile.open("a", encoding="utf-8") as log:
        process = subprocess.Popen([str(arg) for arg in command], **kwargs)
        try:
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
        except KeyboardInterrupt:
            print("\nStopping current training; no next model will start.", flush=True)
            if process.poll() is None:
                if os.name == "nt":
                    process.send_signal(signal.CTRL_BREAK_EVENT)
                else:
                    os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        process.terminate()
                    else:
                        os.killpg(process.pid, signal.SIGTERM)
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        if os.name == "nt":
                            process.kill()
                        else:
                            os.killpg(process.pid, signal.SIGKILL)
                        process.wait()
            raise
        finally:
            process.stdout.close()
    if code:
        raise RuntimeError(
            f"Child exited with code {code}. Log: {logfile}. Queue stopped; no automatic parameter changes."
        )


def summarize(project):
    rows = []
    for complete in sorted(Path(project).glob("*/complete.json")):
        result = json.loads(complete.read_text(encoding="utf-8"))
        job, metrics = result["job"], result["metrics"]
        actual = json.loads((complete.parent / "amp_actual.json").read_text(encoding="utf-8"))
        rows.append(
            dict(
                model=job["model"],
                seed=job["seed"],
                amp=int(job["amp"]),
                epochs=job["epochs"],
                batch=job["recipe"]["batch"],
                imgsz=job["recipe"]["imgsz"],
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
    with (Path(project) / "baseline_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    paired = []
    groups = {(row["model"], row["seed"], row["epochs"]) for row in rows}
    for model, seed, epochs in sorted(groups):
        pair = {row["amp"]: row for row in rows if (row["model"], row["seed"], row["epochs"]) == (model, seed, epochs)}
        if set(pair) != {0, 1}:
            continue
        if any(pair[0][key] != pair[1][key] for key in ("batch", "imgsz")):
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
    save_json(
        Path(project) / "amp_paired_deltas.json", dict(definition="AMP1 minus AMP0, percentage points", pairs=paired)
    )


def main(settings):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print the queue only; no imports/downloads/training")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    project = Path(settings["PROJECT"]).expanduser().resolve()
    if args.summarize_only:
        summarize(project)
        return
    queue = make_queue(
        settings["SUITE"], settings["SEEDS"], settings["EPOCHS"], settings["WORKERS"], settings["BATCHES"]
    )
    device = str(settings["DEVICE"])
    if not device.isdigit():
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
    worker = HERE / "worker.py"
    prepared = project / "_prepared"
    for job in queue:
        job.update(run_dir=str(project / job["name"]), prepared=str(prepared))
    print(f"FOREGROUND BASELINES | {len(queue)} jobs | GPU physical {device} -> logical cuda:0")
    print("Data:", settings["DATA"], "\nProject:", project)
    for i, job in enumerate(queue, 1):
        print(f"{i:02d}. {job['name']} | batch={job['recipe']['batch']} | size={job['recipe']['imgsz']}")
    if args.dry_run:
        return
    interpreters = {key: str(Path(value).expanduser().resolve()) for key, value in settings["PYTHONS"].items()}
    used_envs = {job["recipe"]["environment"] for job in queue}
    for environment in used_envs:
        if not Path(interpreters[environment]).is_file():
            raise FileNotFoundError(
                f"Set PYTHONS['{environment}'] to the installed environment's Python: {interpreters[environment]}"
            )
    if not Path(settings["DATA"]).expanduser().is_file():
        raise FileNotFoundError(f"Set DATA to your cleaned data.yaml: {settings['DATA']}")
    if project.exists() and any(project.iterdir()) and not (project / "batch_protocol.json").is_file():
        raise FileExistsError(f"Project already contains unrelated results: {project}; choose a new path")
    project.mkdir(parents=True, exist_ok=True)
    protocol = {
        key: settings[key] for key in ("DATA", "SUITE", "SEEDS", "EPOCHS", "WORKERS", "BATCHES", "DEVICE", "PYTHONS")
    }
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
    for environment in sorted(used_envs):
        families = sorted({job["recipe"]["family"] for job in queue if job["recipe"]["environment"] == environment})
        run_visible(
            [interpreters[environment], "-I", "-u", worker, "--check", *families],
            weights,
            env,
            project / "logs" / f"preflight_{environment}.log",
        )
    if args.preflight_only:
        return
    prepare_python = interpreters[queue[0]["recipe"]["environment"]]
    run_visible(
        [prepare_python, "-I", "-u", worker, "--prepare", settings["DATA"], "--prepared", prepared],
        weights,
        env,
        project / "logs/prepare.log",
    )
    source = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))
    if any(job["recipe"]["family"] == "rfdetr" for job in queue) and "test" not in source["splits"]:
        raise ValueError(
            "RF-DETR 1.4 constructs a test loader even with run_test=False. Provide the REAL held-out test "
            "split; never substitute training/validation images."
        )
    start = settings.get("START_FROM", "")
    if start and start not in {job["name"] for job in queue}:
        raise ValueError(f"START_FROM is not a queue job: {start}")
    skip = set(settings.get("SKIP_RUNS", []))
    if skip - {job["name"] for job in queue}:
        raise ValueError(f"Unknown SKIP_RUNS: {skip}")
    started = not start
    for job in queue:
        started = started or job["name"] == start
        if not started or job["name"] in skip:
            print("USER-SKIPPED:", job["name"])
            continue
        run_dir = Path(job["run_dir"])
        complete = run_dir / "complete.json"
        if complete.exists():
            if json.loads(complete.read_text(encoding="utf-8"))["job"] != job:
                raise RuntimeError(f"Completed job metadata mismatch: {run_dir}")
            print("ALREADY COMPLETE:", job["name"])
            continue
        if run_dir.exists() and not (run_dir / "trained.json").is_file():
            raise RuntimeError(
                f"Interrupted/incomplete training: {run_dir}. Explicitly add its name to SKIP_RUNS "
                "or use a new PROJECT. Nothing was deleted or overwritten."
            )
        job_path = project / "jobs" / f"{job['name']}.json"
        save_json(job_path, job)
        begin = time.time()
        run_visible(
            [interpreters[job["recipe"]["environment"]], "-I", "-u", worker, "--job", job_path],
            weights,
            dict(env, PYTHONHASHSEED=str(job["seed"])),
            project / "logs" / f"{job['name']}.log",
        )
        print(f"FINISHED {job['name']} ({(time.time() - begin) / 60:.1f} minutes)", flush=True)
        summarize(project)
    summarize(project)
    print("Queue finished. Read baseline_summary.csv and amp_paired_deltas.json; skipped jobs are not results.")
