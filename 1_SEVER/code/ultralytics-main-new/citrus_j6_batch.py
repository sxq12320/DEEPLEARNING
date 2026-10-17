"""Safe foreground J6 queue with original-image data, locked hyperparameters and EV6-compatible evaluation."""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import subprocess
from functools import partial
from pathlib import Path

from citrus_j6_suite import DESCRIPTIONS, ROOT, YAML_DIR, jobs

PROTOCOL_ID = "j6_bbox_center_offset_cell_topk_source_aligned_scratch_ev6_20261017_r2"
SCOUT_GEOMETRY = "bbox_center_offset_v2"
CANDIDATE_POLICY = "cell_topk_no_peak_nms_v2"
SCOUT_SOURCE1_POLICY = "zero_coeff_cell_heat_v2"
MARKER = "j6_completed.json"
RECORD = "j6_protocol.json"


def sha256(path):
    """Hash code/config provenance only; no image-content fingerprint or confirmation gate."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def source_snapshot():
    """Record implementation paths without depending on old documents or deleted report folders."""
    required = {
        "citrus_j6_batch.py", "citrus_j6_suite.py", "citrus_j6_training.py", "citrus_j2_batch.py",
        "citrus_j_batch.py", "citrus_protocol.py", "citrus_e_v6_training.py", "citrus_e_v5_slicing.py",
        "citrus_slicing.py", "eval_citrus_e_v5.py", "eval_citrus_sliced.py",
        "ultralytics/nn/modules/citrus_j6.py", "ultralytics/cfg/default.yaml",
        "protocols/citrus_paper1_formal_v2_ram.yaml",
    }
    required.update(path.relative_to(ROOT).as_posix() for path in ROOT.glob("citrus_j6*.py"))
    for directory in ("ultralytics/nn", "ultralytics/engine", "ultralytics/data", "ultralytics/utils",
                      "ultralytics/models/yolo/segment", "ultralytics/models/yolo/detect"):
        required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / directory).rglob("*.py"))
    missing = [name for name in sorted(required) if not (ROOT / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Upload the whole current code folder; missing required implementation: {missing}")
    files = {name: sha256(ROOT / name) for name in sorted(required)}
    result = dict(files=files, sha256=hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest())
    # Changing DATA/DEVICE/ONLY constants in the launcher must not change a completed model's scientific identity.
    result["launcher_sha256"] = sha256(ROOT / "RUN_CITRUS_J6.py")
    for key, arguments in (("git_head", ["rev-parse", "HEAD"]),
                           ("git_status", ["status", "--porcelain", "--untracked-files=no"])):
        try:
            process = subprocess.run(["git", *arguments], cwd=str(ROOT), text=True, capture_output=True, timeout=20)
            result[key] = process.stdout.strip() if process.returncode == 0 else "unavailable"
        except (OSError, subprocess.TimeoutExpired):
            result[key] = "unavailable"
    return result


def write_json(path, value):
    """Atomically replace a small derived metadata file, never checkpoints or annotations."""
    target = Path(path)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(target)


def history_rows(directory):
    """Read Ultralytics CSV headers robustly without combining metric peaks from different epochs."""
    path = Path(directory) / "results.csv"
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return [{key.strip(): value.strip() for key, value in row.items() if key is not None}
                for row in csv.DictReader(stream)]


def assert_finished(directory, epochs):
    """A returned train() call alone is not evidence that all requested epochs completed."""
    directory = Path(directory)
    history = history_rows(directory)
    try:
        observed = int(float(history[-1]["epoch"]))
    except (KeyError, IndexError, ValueError) as error:
        raise RuntimeError(f"Missing valid final training history: {directory}") from error
    if observed != epochs:
        raise RuntimeError(f"Only {observed}/{epochs} epochs in {directory}; no completion marker written")
    for name in ("weights/last.pt", "weights/best_mask.pt", "best_mask_selection.json"):
        if not (directory / name).is_file():
            raise FileNotFoundError(f"Training lacks its Mask-selected checkpoint/provenance: {directory / name}")
    selection = json.loads((directory / "best_mask_selection.json").read_text(encoding="utf-8"))
    selected_epoch = selection.get("epoch")
    if not isinstance(selected_epoch, int) or not 1 <= selected_epoch <= epochs:
        raise RuntimeError(f"Invalid Mask checkpoint selection epoch: {selection}")
    selected = next((row for row in history if int(float(row["epoch"])) == selected_epoch), None)
    if selected is None or not math.isclose(float(selected["metrics/mAP50-95(M)"]),
                                           float(selection["mask_mAP50_95"]), rel_tol=1e-4, abs_tol=1e-5):
        raise RuntimeError(f"Mask checkpoint provenance does not match training CSV: {directory}")
    return selected


def output_state(record, directory, resume=False):
    """Refuse partial/conflicting output unless the caller explicitly asks to resume that exact run."""
    directory = Path(directory)
    marker = directory / MARKER
    if marker.is_file():
        if json.loads(marker.read_text(encoding="utf-8")) != record:
            raise FileExistsError(f"Completed protocol changed: {directory}. Choose a NEW PROJECT")
        assert_finished(directory, record["train"]["epochs"])
        return "completed"
    if not directory.exists():
        if resume:
            raise FileNotFoundError(f"Explicit resume directory does not exist: {directory}")
        return "new"
    if not resume:
        raise FileExistsError(f"Partial/conflicting output: {directory}. Use ONLY to skip it, NEW PROJECT, or RESUME.")
    protocol = directory / RECORD
    if not protocol.is_file() or json.loads(protocol.read_text(encoding="utf-8")) != record:
        raise FileExistsError(f"Resume requires the identical saved J6 protocol/code/settings: {directory}")
    if not (directory / "weights/last.pt").is_file():
        raise FileNotFoundError(f"Resume checkpoint missing: {directory / 'weights/last.pt'}")
    history = history_rows(directory)
    if history and int(float(history[-1]["epoch"])) >= record["train"]["epochs"]:
        raise RuntimeError(f"Requested epochs already exist but completion was not verified: {directory}")
    return "resume"


def resume_names(value, queue):
    """Only explicitly listed models may resume; one seed avoids an ambiguous last.pt choice."""
    if not value:
        return set()
    requested = [name.strip() for name in value.split(",") if name.strip()]
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("RESUME must be a comma-separated list of distinct selected J6 model names")
    if set(requested) - {name for name, _ in queue} or len({seed for _, seed in queue}) != 1:
        raise ValueError("RESUME names must belong to ONLY/SUITE and SEEDS must contain exactly one seed")
    return set(requested)


def summarize(project, queue):
    """Report the mask-selected epoch's AP50/AP50-95/P/R together, not four independent maxima."""
    rows = []
    for name, seed in queue:
        directory = Path(project) / f"{name}_seed{seed}"
        history = history_rows(directory)
        valid = []
        for row in history:
            try:
                if math.isfinite(float(row["metrics/mAP50-95(M)"])):
                    valid.append(row)
            except (KeyError, ValueError):
                continue
        if not valid:
            continue
        best = max(valid, key=lambda row: float(row["metrics/mAP50-95(M)"]))
        selection_file = directory / "best_mask_selection.json"
        selection_source = "CSV maximum (no checkpoint selection record)"
        if selection_file.is_file():
            checkpoint_selection = json.loads(selection_file.read_text(encoding="utf-8"))
            selected_epoch = checkpoint_selection.get("epoch")
            selected_row = next((row for row in valid if int(float(row["epoch"])) == selected_epoch), None)
            if selected_row is None:
                raise RuntimeError(f"Mask-selected epoch is absent from CSV: {directory}")
            best = selected_row
            selection_source = "best_mask_selection.json"
        rows.append(dict(name=name, seed=seed, epochs_observed=history[-1]["epoch"],
                         selected_epoch=best["epoch"], mask_AP5095=best["metrics/mAP50-95(M)"],
                         mask_AP50=best.get("metrics/mAP50(M)", ""),
                         mask_precision=best.get("metrics/precision(M)", ""),
                         mask_recall=best.get("metrics/recall(M)", ""),
                         completed=(directory / MARKER).is_file(), selection_source=selection_source,
                         source=str(directory / "results.csv")))
    if rows:
        target = Path(project) / "j6_training_summary.csv"
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(target)
    return rows


def build_train_args(epochs, seed, workers, batch, plots=True):
    """Lock the J2/I-series recipe, explicitly retaining the original images for J6's paired views."""
    from citrus_j2_batch import settings

    train = settings(epochs, seed, imgsz=640, cache=True)
    train.update(workers=int(workers), batch=int(batch), multi_scale=0.0, rect=False, mosaic=1.0,
                 save=True, time=None, plots=plots)
    return train


def plotting_preflight():
    """Fail before training on an incompatible plotting environment; never install packages or disable plots."""
    try:
        import matplotlib
        import matplotlib.pyplot  # noqa: F401  Import the plotting path, not just a metadata package.
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "J6 plots=True but Matplotlib/NumPy plotting dependencies failed to import: "
            f"{error}. Training has not started. Use the intended server Python environment and check "
            "its NumPy/Matplotlib compatibility before changing packages. No automatic pip changes were made. "
            "For an explicitly plot-free engineering smoke only, pass --no-plots (or PLOTS=False); "
            "this does not validate PR/figure generation and does not silently repair the environment."
        ) from error
    return str(matplotlib.__version__)


def runtime_dependencies():
    """Defer CUDA-sensitive imports until the launcher has bound the physical device."""
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds
    from citrus_j2_batch import save_initial_state
    from citrus_j_batch import save_mask_checkpoints
    from citrus_j6_training import J6Trainer

    return torch, YOLO, init_seeds, save_initial_state, save_mask_checkpoints, J6Trainer


def restore_mask_best(trainer):
    """Preserve the previous best mask epoch across a genuine optimizer-state resume."""
    for filename, field, attribute in (("best_mask_selection.json", "mask_mAP50_95", "j_best_mask"),
                                        ("best_mask50_selection.json", "mask_mAP50", "j_best_mask50")):
        path = Path(trainer.save_dir) / filename
        if path.is_file():
            setattr(trainer, attribute, float(json.loads(path.read_text(encoding="utf-8"))[field]))


def assert_resumable_checkpoint(checkpoint, epochs):
    """Reject finalized or incomplete checkpoint state instead of silently starting over."""
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("epoch"), int):
        raise RuntimeError("last.pt lacks a native resumable checkpoint dictionary/epoch")
    if checkpoint["epoch"] < 0 or checkpoint.get("optimizer") is None:
        raise RuntimeError(
            "last.pt is finalized/optimizer-stripped and cannot exactly resume. "
            "Normal final_eval strips last.pt only after training ends; use a NEW PROJECT for a new run, "
            "not resume=True or a silent scratch restart."
        )
    if checkpoint["epoch"] + 1 >= epochs:
        raise RuntimeError("last.pt already reached the requested epochs; no additional training is a resume")


def run(data, device, project, suite="priority", epochs=300, seeds=(42,), only="", dry_run=False,
        cpu_threads=4, workers=4, batch=16, resume="", plots=True):
    """Train in the current Python process; failures and Ctrl+C stop the entire queue immediately."""
    if epochs < 1 or batch < 1 or workers < 0 or cpu_threads < 1:
        raise ValueError("Positive epochs/batch/threads and nonnegative workers required")
    if not isinstance(plots, bool):
        raise ValueError("plots must be True or False")
    queue = jobs(suite, tuple(seeds), only)
    resuming = resume_names(resume, queue)
    data, project = Path(data).expanduser().resolve(), Path(project).expanduser().resolve()
    torch, YOLO, init_seeds, initial_callback, mask_callback, trainer_type = runtime_dependencies()
    torch.set_num_threads(cpu_threads)
    print("J6 FOREGROUND / SEQUENTIAL / SCRATCH / AMP=False / cache=True / I-series EV6 metric protocol", flush=True)
    print(f"DATA={data}\nPROJECT={project}\nQUEUE={queue}", flush=True)
    print(f"Physical device={device}; CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}; "
          "one visible GPU is internally logical cuda:0", flush=True)
    print("Original DATA is passed directly; no cached 640-only training views, no dataset confirmation, "
          "no GPU occupancy guard, no nohup. Ctrl+C stops all subsequent models.", flush=True)
    if dry_run:
        for name in dict.fromkeys(name for name, _ in queue):
            model = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment", verbose=False)
            print(f"BUILD {name}: {sum(parameter.numel() for parameter in model.model.parameters()):,} parameters; "
                  f"{DESCRIPTIONS[name]}", flush=True)
            del model
        print("DRY RUN ONLY: no image data loaded and no training started.", flush=True)
        return
    if not data.is_file():
        raise FileNotFoundError(f"Edit DATA in RUN_CITRUS_J6.py or pass --data: {data}")
    sources = source_snapshot()
    data_yaml_sha = sha256(data)
    pending = []
    for name, seed in queue:
        train = build_train_args(epochs, seed, workers, batch, plots=plots)
        record = dict(protocol=PROTOCOL_ID, name=name, seed=seed, data=str(data), data_yaml_sha256=data_yaml_sha,
                      train=train, yaml_sha256=sha256(YAML_DIR / f"{name}.yaml"),
                      implementation_sha256=sources["sha256"],
                      validator="J6Validator: EV6 matching/AP protocol with explicit J6 ROI-mask adapter",
                      initialization="scratch", original_image_data=True, pretrained_teacher=False,
                      automatic_label_changes=False, scout_geometry=SCOUT_GEOMETRY,
                      candidate_policy=CANDIDATE_POLICY,
                      scout_source1_policy=SCOUT_SOURCE1_POLICY,
                      gain_target="source1_zero_logit_brier_difference_at_assigned_scout_cell_v2")
        directory = project / f"{name}_seed{seed}"
        state = output_state(record, directory, name in resuming)
        if state == "completed":
            print(f"SKIP verified completed {directory.name}", flush=True)
        else:
            pending.append((name, seed, directory, record, state))
    if not pending:
        summarize(project, queue)
        return
    if plots:
        print(f"Plotting dependency import OK: Matplotlib {plotting_preflight()}", flush=True)
    else:
        print("EXPLICIT plots=False: metrics unchanged; PR/confusion/label plotting is NOT being validated.", flush=True)
    project.mkdir(parents=True, exist_ok=True)
    for name, seed, directory, record, state in pending:
        init_seeds(seed, deterministic=True)
        checkpoint = directory / "weights/last.pt"
        model = YOLO(str(checkpoint if state == "resume" else YAML_DIR / f"{name}.yaml"), task="segment")
        if state != "resume":
            model.add_callback("on_pretrain_routine_end", partial(initial_callback, cpu_threads=cpu_threads))
        else:
            model.add_callback("on_pretrain_routine_end", restore_mask_best)
        model.add_callback("on_model_save", mask_callback)

        def record_protocol(trainer, value=record):
            target = Path(trainer.save_dir).resolve()
            if target != directory:
                raise RuntimeError(f"Unexpected output directory: {target} != {directory}")
            write_json(target / RECORD, value)
            write_json(target / "j6_sources.json", sources)
            write_json(target / "j6_run_state.json", dict(status="running", resumed=state == "resume"))

        model.add_callback("on_pretrain_routine_end", record_protocol)
        print(f"{'RESUME' if state == 'resume' else 'TRAIN'} {name} seed={seed} -> {directory}", flush=True)
        try:
            if state == "resume":
                assert_resumable_checkpoint(model.ckpt, epochs)
            train_options = dict(record["train"])
            if state == "resume":
                train_options["resume"] = str(checkpoint)
            # Do not resize/copy train images before this call. J6Trainer builds the two
            # coordinate-registered views from the original samples in the same batch.
            model.train(trainer=trainer_type, data=str(data), device=device, project=str(project),
                        name=directory.name, exist_ok=state == "resume", **train_options)
            actual_dir = Path(model.trainer.save_dir).resolve()
            if actual_dir != directory:
                raise RuntimeError(f"Unexpected output redirect: {actual_dir} != {directory}")
            assert_finished(directory, epochs)
            write_json(directory / MARKER, record)
            write_json(directory / "j6_run_state.json", dict(status="completed", epochs=epochs))
        except BaseException as error:
            if directory.is_dir():
                write_json(directory / "j6_run_state.json", dict(
                    status="interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                    error_type=type(error).__name__, message=str(error), completion_marker_written=False))
                summarize(project, queue)
            raise
        finally:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        summary = summarize(project, queue)
    return summary
