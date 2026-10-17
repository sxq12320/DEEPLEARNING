"""Foreground J5 runner; I-series EV6 validation, no GPU guard or background child."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import os
import subprocess
from functools import partial
from pathlib import Path

from citrus_j5_suite import KD, KD_COMMON, NAMES, ROOT, YAML_DIR, jobs

PROTOCOL_ID = "j5_gt_supported_task_distillation_scratch_ev6_20261015"


def sha256(path):
    """Hash one implementation or weight file, not a dataset-confirmation gate."""
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def source_snapshot():
    """Record scoped source hashes and Git state without copying unrelated files."""
    from citrus_j4_batch import tracked_sources

    paths = set(tracked_sources()) | {
        "RUN_CITRUS_J5.py", "citrus_j5_batch.py", "citrus_j5_suite.py", "citrus_j5_distill.py",
    }
    files = {name: sha256(ROOT / name) for name in sorted(paths)}
    result = dict(files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    for key, arguments in (("git_head", ["rev-parse", "HEAD"]), ("git_status", ["status", "--porcelain", "--untracked-files=no"])):
        try:
            check = subprocess.run(["git", *arguments], cwd=str(ROOT), text=True, capture_output=True, timeout=20)
            result[key] = check.stdout.strip() if check.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            result[key] = "unavailable"
    return result


def completed(record, directory):
    """Only skip identical completed runs; never overwrite partial/conflicting runs."""
    marker = directory / "j5_completed.json"
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != record:
            raise FileExistsError(f"Protocol changed at {directory}; choose a NEW PROJECT")
        if not (directory / "weights/best_mask.pt").is_file():
            raise FileNotFoundError(f"Completion marker exists but best_mask.pt is absent: {directory}")
        return True
    if directory.exists():
        raise FileExistsError(f"Partial/conflicting output: {directory}. Use NEW PROJECT or ONLY other models.")
    return False


def resolve_teacher(teacher, project, required):
    """Students use the project teacher, never an old checkpoint by hidden fallback."""
    if not required:
        return None
    path = project / f"{NAMES[1]}_seed42/weights/best_mask.pt" if teacher == "auto" else Path(teacher).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Frozen teacher is missing: {path}; run J81 first or set TEACHER explicitly")
    if teacher == "auto" and not (path.parent.parent / "j5_completed.json").is_file():
        raise RuntimeError("Automatic teacher lacks J5 completion provenance; do not use partial/unknown teacher")
    return path


def summarize(project, queue):
    """Export comparable same-epoch mask metrics; do not merge different peaks."""
    rows = []
    for name, seed in queue:
        directory = project / f"{name}_seed{seed}"
        path = directory / "results.csv"
        if not path.is_file():
            continue
        with path.open(encoding="utf-8-sig", newline="") as stream:
            history = [{key.strip(): value.strip() for key, value in row.items() if key is not None}
                       for row in csv.DictReader(stream)]
        if not history or "metrics/mAP50-95(M)" not in history[-1]:
            continue
        best = max(history, key=lambda row: float(row["metrics/mAP50-95(M)"]))
        rows.append(dict(name=name, seed=seed, epochs_observed=history[-1]["epoch"],
                         best_mask_epoch=best["epoch"], mask_AP5095=best["metrics/mAP50-95(M)"],
                         mask_AP50=best["metrics/mAP50(M)"], mask_recall=best["metrics/recall(M)"],
                         mask_precision=best["metrics/precision(M)"],
                         completed=(directory / "j5_completed.json").is_file(), source=str(path)))
    if rows:
        target = project / "j5_training_summary.csv"
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)


def run(data, device, project, suite="priority", epochs=300, seeds=(42,), only="", dry_run=False,
        cpu_threads=4, workers=4, batch=16, teacher="auto"):
    """Train sequentially with scratch initialization; old training code is unchanged."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from citrus_j2_batch import save_initial_state, settings
    from citrus_j_batch import save_mask_checkpoints

    if epochs < 1 or batch < 1 or workers < 0 or cpu_threads < 1:
        raise ValueError("Positive epochs/batch/threads and nonnegative workers required")
    queue = jobs(suite, tuple(seeds), only, teacher)
    data, project = Path(data).expanduser().resolve(), Path(project).expanduser().resolve()
    torch.set_num_threads(cpu_threads)
    print("J5 FOREGROUND / SEQUENTIAL / SCRATCH / AMP=False / cache=True / I-series EV6", flush=True)
    print(f"DATA={data}\nPROJECT={project}\nQUEUE={queue}", flush=True)
    print(f"Physical device argument={device}; visible={os.environ.get('CUDA_VISIBLE_DEVICES')}; "
          "Ultralytics returns logical cuda:0 on this one-device visibility.", flush=True)
    print("No nohup, no GPU occupancy lock, no dataset-confirmation gate. Ctrl+C stops the queue.", flush=True)
    print("Dataset/labels changed at the same path? Use a NEW PROJECT. Teacher makes TRAINING slower, "
          "but is not part of the deployed student.", flush=True)
    if dry_run:
        for name in dict.fromkeys(name for name, _ in queue):
            model = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment", verbose=False)
            print(f"BUILD {name}: {sum(parameter.numel() for parameter in model.model.parameters()):,} params; "
                  f"KD={KD[name]}", flush=True)
            del model
        return
    if not data.is_file():
        raise FileNotFoundError(f"Edit DATA in RUN_CITRUS_J5.py: {data}")
    if teacher != "auto" and any(KD[name]["kd_mask"] or KD[name]["kd_cls"] for name, _ in queue):
        print("WARNING: explicit external teacher; verify it never trained on this val/test. "
              "Its path/hash are recorded, but cannot prove its training data provenance.", flush=True)
    from citrus_j2_data import prepare_resampled_views
    from citrus_j5_distill import make_trainer

    sources = source_snapshot()
    data_yaml_sha = sha256(data)
    common = settings(epochs, int(seeds[0]), imgsz=640, cache=True)
    common.update(workers=int(workers), batch=int(batch))
    project.mkdir(parents=True, exist_ok=True)
    # Preparing/caching views does not alter val/test or any original annotation.
    train_data = None
    for name, seed in queue:
        kd = dict(KD[name])
        required = bool(kd["kd_mask"] or kd["kd_cls"])
        teacher_path = resolve_teacher(teacher, project, required)
        train = dict(common, seed=int(seed))
        record = dict(protocol=PROTOCOL_ID, name=name, seed=int(seed), data=str(data), data_yaml_sha256=data_yaml_sha,
                      train=train, yaml_sha256=sha256(YAML_DIR / f"{name}.yaml"), source_sha256=sources["sha256"],
                      validator="EV6Validator (I/E/J series); mask_ratio=2", initialization="scratch",
                      teacher=None if teacher_path is None else dict(path=str(teacher_path), sha256=sha256(teacher_path)),
                      distillation=dict(kd, **KD_COMMON), evaluation_teacher=False, pseudo_labels=False)
        directory = project / f"{name}_seed{seed}"
        if completed(record, directory):
            print(f"SKIP completed {directory.name}", flush=True)
            continue
        if train_data is None:
            train_data = prepare_resampled_views(data, project / "_prepared_views_640_linear", 640, "linear")
        init_seeds(seed, deterministic=True)
        model = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment")
        model.add_callback("on_pretrain_routine_end", partial(save_initial_state, cpu_threads=cpu_threads))
        model.add_callback("on_model_save", save_mask_checkpoints)

        def record_protocol(trainer, value=record):
            target = Path(trainer.save_dir)
            (target / "j5_protocol.json").write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
            (target / "j5_sources.json").write_text(json.dumps(sources, ensure_ascii=False, indent=2), encoding="utf-8")
            manifest = Path(train_data).parent / "views.json"
            if manifest.is_file():
                views = json.loads(manifest.read_text(encoding="utf-8"))
                source_rows = [{key: group.get(key) for key in ("source", "instances", "original_shape")}
                               for group in views.get("groups", [])]
                (target / "original_train_sources.json").write_text(
                    json.dumps(source_rows, ensure_ascii=False, indent=2), encoding="utf-8")

        model.add_callback("on_pretrain_routine_end", record_protocol)
        trainer_type = make_trainer(teacher_path=None if teacher_path is None else str(teacher_path), **kd, **KD_COMMON)
        print(f"TRAIN {name} seed={seed}; teacher={teacher_path}; output={directory}", flush=True)
        model.train(trainer=trainer_type, data=str(train_data), device=device, project=str(project),
                    name=directory.name, exist_ok=False, **train)
        actual_dir = Path(model.trainer.save_dir).resolve()
        if actual_dir != directory:
            raise RuntimeError(f"Unexpected output redirect: {actual_dir} != {directory}")
        if not (directory / "weights/best_mask.pt").is_file():
            raise FileNotFoundError(f"Training returned without a Mask-selected checkpoint: {directory}")
        (directory / "j5_completed.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        summarize(project, queue)
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    summarize(project, queue)
