"""Accuracy-first J7R: explicit observations, foreground sequential training, unchanged EV6."""

import argparse
import gc
import hashlib
import json
from functools import partial
from pathlib import Path

from citrus_j7_batch import bind_device, plotting_preflight, save_run_state, source_manifest as inherited_sources

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/J_V7R_series"
PROTOCOL_ID = "j7r_early_evidence_packed15_ev6_amp0_20261020_v4"
PRIORITY_IDS = ("J119", "J116", "J118", "J123", "J120")
POLICY_IDS = ("J119", "J116")
EVIDENCE_IDS = ("J118", "J123", "J120")
NATIVE_IDS = ("J116", "J117", "J118")
PHASE_IDS = ("J121", "J122")
LEGACY_IDS = tuple("J{}".format(index) for index in range(110, 116))
ALL_IDS = tuple("J{}".format(index) for index in range(110, 124))
NEW_SOURCES = (
    "citrus_j7r_batch.py", "RUN_CITRUS_J7R.py", "citrus_j7r_data.py", "citrus_j7r_training.py",
    "citrus_j7r_predict.py", "citrus_j7r_initialization.py", "citrus_j6_data.py",
    "ultralytics/nn/modules/citrus_j7r.py", "ultralytics/cfg/models/11/yolo11-seg.yaml",
    "ultralytics/models/yolo/model.py", "ultralytics/engine/validator.py",
    "ultralytics/data/base.py", "ultralytics/data/build.py", "ultralytics/data/dataset.py", "ultralytics/data/augment.py",
    "ultralytics/nn/autobackend.py", "ultralytics/nn/backends/pytorch.py",
)


def select_models(suite="priority", only=""):
    suites = {"priority": PRIORITY_IDS, "screen": PRIORITY_IDS, "policy": POLICY_IDS, "evidence": EVIDENCE_IDS,
              "native": NATIVE_IDS,
              "phase": PHASE_IDS, "legacy": LEGACY_IDS, "all": ALL_IDS}
    if suite not in suites:
        raise ValueError("suite must be priority, screen, policy, evidence, native, phase, legacy, or all")
    defaults = suites[suite]
    values = [v.strip() for v in only.split(",") if v.strip()] if only else list(
        defaults)
    ids = [value.split("_", 1)[0].upper() for value in values]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError("Select distinct J7R IDs")
    result = []
    for value, model_id in zip(values, ids):
        if model_id not in ALL_IDS:
            raise ValueError("Unknown J7R ID: {}".format(value))
        paths = list(YAML_DIR.glob(model_id + "_*.yaml"))
        if len(paths) != 1 or ("_" in value and paths[0].stem != value):
            raise ValueError("Expected one matching J7R YAML: {}".format(value))
        result.append(paths[0])
    return result


def source_manifest():
    result = inherited_sources()
    for name in NEW_SOURCES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError("Missing J7R dependency: {}".format(path))
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def profile_model(yaml_path, image_size=640):
    import torch
    from thop import profile
    from ultralytics import YOLO
    from citrus_j7r_data import j7r_model_spec

    model = YOLO(str(yaml_path), task="segment", verbose=False).model.eval()
    spec = j7r_model_spec(model)
    count = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        macs, _ = profile(model, inputs=(torch.zeros(1, 15, image_size, image_size),), verbose=False)
    row = {"model": yaml_path.stem, "parameters": count, "gflops_640": 2 * float(macs) / 1e9,
           "input": spec, "high_shape": [image_size * 2] * 2 if spec["observation"] != "global" else None,
           "scope": "complete packed forward including raw encoder; excludes raw resize/transfer/NMS/mask composition"}
    print("BUILD {}: {:,} params; {:.6f} GFLOPs; {}".format(
        row["model"], count, row["gflops_640"], spec["observation"]), flush=True)
    del model
    return row


def experiment_record(yaml_path, data, epochs, seed, batch, workers, plots, sources):
    from citrus_j2_batch import settings
    from ultralytics.utils import YAML

    config = YAML.load(yaml_path)
    recipe = settings(epochs, seed, imgsz=640, cache=True)
    recipe.update(batch=int(batch), workers=int(workers), plots=bool(plots), save=True, half=False)
    return {
        "protocol": PROTOCOL_ID, "name": yaml_path.stem, "seed": seed, "epochs": epochs,
        "data": str(data), "yaml_sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
        "sources": sources, "train": recipe, "validator": "J7RValidator / unchanged EV6 input/2",
        "training_images": ("native YOLO640 augmentation; no high image loaded"
                            if config.get("citrus_data_policy") == "native640"
                            else "one high1280 augmentation; AREA global640; paired1280 loader"),
        "data_policy": config.get("citrus_data_policy", "paired1280"),
        "input": {"sensor_channels": 3, "model_channels": 15, "observation": config["citrus_observation"]},
        "native_initialization": config.get("citrus_native_initialization", "legacy per-model seed initialization"),
        "final_checkpoint": "weights/best_mask.pt",
        "augmentation_limit": "Same recipe and seed are NOT proof of identical per-sample augmentation; first batch audited",
    }


def _save_state(trainer, expected_dir, record, cpu_threads):
    # Reuse the original effective-settings/initial-state logger, not its data loader.
    save_run_state(trainer, expected_dir, record, cpu_threads)
    (Path(trainer.save_dir) / "j7r_protocol.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    from ultralytics.utils.torch_utils import unwrap_model

    initialization = getattr(unwrap_model(trainer.model), "j7r_native_initialization", None)
    if initialization is not None:
        (Path(trainer.save_dir) / "j7r_native_initialization.json").write_text(
            json.dumps(initialization, ensure_ascii=False, indent=2), encoding="utf-8")


def record_failure(directory, error):
    """Log this process only; never inspect other jobs or change batch/AMP after OOM."""
    import torch

    destination = Path(directory) / "j7r_failure.json"
    if not Path(directory).is_dir() or destination.exists():
        return
    memory = None
    if torch.cuda.is_initialized():
        try:
            memory = {"allocated": torch.cuda.memory_allocated(), "reserved": torch.cuda.memory_reserved(),
                      "peak_allocated": torch.cuda.max_memory_allocated(),
                      "peak_reserved": torch.cuda.max_memory_reserved()}
        except RuntimeError:
            memory = {"note": "CUDA memory query unavailable after failure"}
    with destination.open("x", encoding="utf-8") as handle:
        json.dump({"status": "failed", "error_type": type(error).__name__, "message": str(error),
                   "cuda_bytes_this_process": memory,
                   "automatic_batch_or_amp_change": False}, handle, ensure_ascii=False, indent=2)


def run(data, device, project, suite="priority", epochs=300, seeds=(42,), only="", dry_run=False,
        cpu_threads=4, workers=4, batch=16, plots=True):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    if epochs < 1 or batch < 1 or workers < 0 or cpu_threads < 1:
        raise ValueError("Positive epochs/batch/threads and nonnegative workers required")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Use at least one distinct seed")
    torch.set_num_threads(cpu_threads)
    selected = select_models(suite, only)
    data, project = Path(data).expanduser().resolve(), Path(project).expanduser().resolve()
    print("J7R ACCURACY FIRST / FOREGROUND / SEQUENTIAL: {}".format([p.stem for p in selected]), flush=True)
    print("Data={} | logical device={} | AMP=False/cache=True/scratch/EV6; no occupancy/data confirmation gate".format(
        data, device), flush=True)
    if dry_run:
        print("Build/profile only. No data loaded, no AP/speed claim; high encoder cost is INCLUDED.", flush=True)
        return [profile_model(path) for path in selected]
    if not data.is_file():
        raise FileNotFoundError("Edit DATA in RUN_CITRUS_J7R.py to your actual dataset YAML: {}".format(data))
    sources = source_manifest()
    pending = []
    for seed in seeds:
        for yaml_path in selected:
            directory = project / "{}_seed{}".format(yaml_path.stem, seed)
            record = experiment_record(yaml_path, data, epochs, seed, batch, workers, plots, sources)
            completed = directory / "j7r_completed.json"
            if completed.is_file() and json.loads(completed.read_text(encoding="utf-8")) == record:
                print("SKIP completed {}".format(directory), flush=True)
                continue
            if directory.exists():
                raise FileExistsError(
                    "Partial/conflicting output exists: {}. Select ONLY or use a NEW PROJECT; no files overwritten.".format(
                        directory))
            pending.append((yaml_path, seed, directory, record))
    if not pending:
        return []
    if plots:
        print("Plotting preflight OK: {}".format(plotting_preflight()), flush=True)
    project.mkdir(parents=True, exist_ok=True)
    for yaml_path, seed, directory, record in pending:
        if directory.exists():
            raise FileExistsError("Output appeared while waiting: {}".format(directory))
        init_seeds(seed, deterministic=True)
        model = YOLO(str(yaml_path), task="segment")
        model.add_callback("on_pretrain_routine_end", partial(
            _save_state, expected_dir=directory, record=record, cpu_threads=cpu_threads))
        print("TRAIN {} -> {}".format(yaml_path.stem, directory), flush=True)
        # Native .train() automatically dispatches J7RTrainer from the YAML/head protocol.
        if torch.cuda.is_initialized():
            torch.cuda.reset_peak_memory_stats()
        try:
            model.train(data=str(data), device=device, project=str(project), name=directory.name,
                        exist_ok=False, **record["train"])
        except (Exception, KeyboardInterrupt) as error:
            try:
                record_failure(directory, error)
            except (OSError, RuntimeError):
                pass  # Preserve the original training failure, never mask it with diagnostic I/O.
            raise
        if not (directory / "selected_mask_validation/selected_checkpoint.json").is_file():
            raise RuntimeError("Training returned without mask-selected final EV6 evaluation")
        (directory / "j7r_completed.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [str(item[2]) for item in pending]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--device", default="1", help="Physical GPU index, or cpu")
    parser.add_argument("--suite", choices=("priority", "screen", "policy", "evidence", "native", "phase", "legacy", "all"),
                        default="policy")
    parser.add_argument("--only", default="")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true", help="Explicit engineering-only plot-free smoke")
    args = parser.parse_args()
    device = bind_device(args.device, args.cpu_threads)
    run(args.data, device, args.project, args.suite, args.epochs,
        tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip()), args.only,
        args.dry_run, args.cpu_threads, args.workers, args.batch, not args.no_plots)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J7R stopped by Ctrl+C; no next model starts.", flush=True)
        raise SystemExit(130)
