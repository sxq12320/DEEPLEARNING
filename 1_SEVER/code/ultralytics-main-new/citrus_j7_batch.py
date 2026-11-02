"""Foreground sequential J7 experiments; native images, fixed recipe, EV6 mask evaluation."""

import argparse
import gc
import hashlib
import json
import os
from functools import partial
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "J_V7_series"
PROTOCOL_ID = "citrus_j7_native_global_ev6_amp0_20261018_v1"
PRIORITY_IDS = ("J100", "J102", "J103", "J104")
ALL_IDS = tuple("J{}".format(index) for index in range(100, 110))
SOURCES = (
    "citrus_j7_batch.py", "citrus_j7_training.py", "citrus_j7_pr_audit.py", "RUN_CITRUS_J7.py",
    "citrus_j2_batch.py", "citrus_j_batch.py", "citrus_protocol.py", "citrus_e_v6_training.py",
    "protocols/citrus_paper1_formal_v2_ram.yaml", "ultralytics/nn/modules/citrus_j7.py",
    "ultralytics/utils/citrus_j7_loss.py", "ultralytics/nn/modules/__init__.py", "ultralytics/nn/tasks.py",
    "ultralytics/nn/modules/citrus_j4.py", "ultralytics/nn/modules/citrus_e_v11.py",
    "ultralytics/nn/modules/citrus_e_v12.py", "ultralytics/nn/modules/head.py",
    "ultralytics/utils/loss.py", "ultralytics/utils/citrus_e_v11_loss.py",
    "ultralytics/models/yolo/segment/val.py", "ultralytics/utils/metrics.py",
    "ultralytics/nn/modules/block.py", "ultralytics/nn/modules/conv.py",
    "ultralytics/nn/modules/citrus_e_v3.py", "ultralytics/nn/modules/citrus_e_v4.py",
    "ultralytics/nn/modules/citrus_e_v4r.py", "ultralytics/nn/modules/citrus_e_v5.py",
    "ultralytics/nn/modules/citrus_e_v6.py", "ultralytics/nn/modules/citrus_e_v7.py",
    "ultralytics/nn/modules/citrus_e_v9.py", "ultralytics/nn/modules/citrus_e_v10.py",
    "ultralytics/nn/modules/citrus_far.py", "ultralytics/nn/modules/citrus_sage_v4r.py",
    "ultralytics/nn/modules/citrus_sage_v5.py", "ultralytics/utils/citrus_e_v3_loss.py",
    "ultralytics/utils/citrus_e_v5_loss.py", "ultralytics/utils/citrus_e_v6_loss.py",
    "ultralytics/utils/citrus_e_v7_loss.py", "ultralytics/utils/citrus_e_v9_loss.py",
    "ultralytics/utils/sage_v4r_loss.py", "ultralytics/utils/torch_utils.py", "ultralytics/utils/ops.py",
    "ultralytics/engine/model.py", "ultralytics/engine/trainer.py", "ultralytics/engine/validator.py",
)


def bind_device(device, cpu_threads=4):
    """Bind a physical GPU before importing Torch; no occupancy or confirmation guard."""
    physical = str(device).lower()
    if physical != "cpu" and (not physical.isascii() or not physical.isdigit()):
        raise ValueError("DEVICE must be one physical GPU index, or cpu")
    if cpu_threads < 1:
        raise ValueError("cpu_threads must be positive")
    if physical != "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = physical
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(cpu_threads)
    return "cpu" if physical == "cpu" else "0"


def select_models(suite="priority", only=""):
    """Resolve stable IDs to YAML files without importing the model implementation."""
    ids = list(PRIORITY_IDS if suite in ("priority", "screen") else ALL_IDS)
    if only:
        ids = [item.strip() for item in only.split(",") if item.strip()]
    if not ids or len(set(ids)) != len(ids):
        raise ValueError("Select at least one model, without duplicate IDs")
    selected = []
    for value in ids:
        model_id = value.split("_", 1)[0].upper()
        if model_id not in ALL_IDS:
            raise ValueError("Unknown J7 model: {}".format(value))
        matches = sorted(YAML_DIR.glob(model_id + "_*.yaml"))
        if len(matches) != 1:
            raise FileNotFoundError("Expected one {} YAML in {}; found {}".format(model_id, YAML_DIR, len(matches)))
        if "_" in value and value != matches[0].stem:
            raise ValueError("Model name does not match YAML: {}".format(value))
        selected.append(matches[0])
    if len({path.stem for path in selected}) != len(selected):
        raise ValueError("Duplicate model selected via both ID and full name")
    return selected


def source_manifest():
    """Record implementation hashes; this is bookkeeping, never a dataset confirmation gate."""
    manifest = {}
    for name in SOURCES:
        path = ROOT / name
        if not path.is_file():
            raise FileNotFoundError("Required J7 source missing: {}".format(path))
        manifest[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return manifest


def plotting_preflight():
    """Exercise a tiny Agg rendering before training; do not alter packages or disable plots."""
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure

        figure = Figure(figsize=(1, 1), dpi=32)
        canvas = FigureCanvasAgg(figure)
        figure.add_subplot(111).plot([0, 1], [0, 1])
        canvas.draw()
        figure.clear()
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as error:
        raise RuntimeError(
            "J7 plots=True but the NumPy/Matplotlib plotting preflight failed: {}. Training has NOT started. "
            "Use the intended server Python and compatible plotting packages. No packages were changed and plots "
            "were not silently disabled. For an explicitly plot-free engineering smoke only, use --no-plots "
            "or PLOTS=False; that does not verify PR/figure generation.".format(error)
        ) from error
    return str(matplotlib.__version__)


def actual_model_configuration(model):
    """Record instantiated operators and the actual criterion, not merely YAML intentions."""
    from ultralytics.utils.torch_utils import unwrap_model

    model = unwrap_model(model)
    head = model.model[-1]
    if getattr(model, "criterion", None) is None:
        # The normal loss path performs this same lazy initialization on its first call.
        # Initialize it now, on the already-selected device, so this is the actual training criterion.
        model.criterion = model.init_criterion()
    detail_projection = getattr(head, "detail_to_proto", None)
    scalar_names = (
        "tiny_dice_gain", "boundary_gain", "neighbor_gain", "ring_gain", "assignment_mix",
        "quality_gain", "negative_quality_gain", "region_gain", "visibility_gain",
    )
    return {
        "head_class": "{}.{}".format(type(head).__module__, type(head).__name__),
        "criterion_class": "{}.{}".format(type(model.criterion).__module__, type(model.criterion).__name__),
        "quality_predictor_present": hasattr(head, "quality_predictor"),
        "quality_calibration": getattr(head, "quality_calibration", None),
        "shared_towers": getattr(head, "shared_towers", False),
        "tower_width": getattr(head, "tower_width", None),
        "detail_channels": getattr(detail_projection, "in_channels", None),
        "persistent": getattr(head, "persistent", False),
        "fine_mask": getattr(head, "fine_mask", False),
        "proto_stride": getattr(head, "proto_stride", 4),
        "prototype_hidden_channels": getattr(head, "npr", None),
        "recognition_route": getattr(head, "recognition_route", None),
        "gains": {name: float(getattr(head, name)) for name in scalar_names if hasattr(head, name)},
    }


def save_run_state(trainer, expected_dir, record, cpu_threads):
    """Capture actual initialization/effective settings before the first optimization step."""
    from citrus_j_batch import save_initial_state

    if Path(trainer.save_dir).resolve() != Path(expected_dir).resolve():
        raise RuntimeError("Output path changed; refusing an implicitly incremented experiment directory")
    save_initial_state(trainer, cpu_threads=cpu_threads)
    output = Path(trainer.save_dir)
    (output / "j7_protocol.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    effective = dict(vars(trainer.args))
    effective["j7_actual_model"] = actual_model_configuration(trainer.model)
    (output / "j7_effective_args.json").write_text(
        json.dumps(effective, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )


def profile_model(yaml_path, image_size=640):
    """Measure the complete single-image forward graph, not ROI/NMS/decoded-mask cost."""
    import torch
    from thop import profile
    from ultralytics import YOLO

    model = YOLO(str(yaml_path), task="segment", verbose=False).model.eval()
    parameters = sum(parameter.numel() for parameter in model.parameters())
    with torch.no_grad():
        macs, _ = profile(model, inputs=(torch.zeros(1, 3, image_size, image_size),), verbose=False)
    row = {"model": yaml_path.stem, "parameters": parameters, "gflops_640": 2.0 * float(macs) / 1e9}
    print("BUILD {}: {:,} params; {:.6f} GFLOPs @{}".format(
        row["model"], parameters, row["gflops_640"], image_size
    ), flush=True)
    del model
    return row


def run(data, device, project, suite="priority", epochs=300, seeds=(42,), only="", dry_run=False,
        cpu_threads=4, workers=4, batch=16, plots=True):
    """Train models serially, stop the whole queue on failure, never overwrite existing runs.

    ``device`` is logical (0 after binding) or cpu. Use RUN_CITRUS_J7.py, or call
    ``bind_device`` before importing Torch when invoking this function directly.
    """
    import torch
    from citrus_j2_batch import settings
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import init_seeds

    if epochs < 1 or batch < 1 or workers < 0 or cpu_threads < 1:
        raise ValueError("Positive epochs/batch/threads and nonnegative workers required")
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("At least one distinct seed is required")
    torch.set_num_threads(cpu_threads)
    selected = select_models(suite, only)
    data = Path(data).expanduser().resolve()
    project = Path(project).expanduser().resolve()
    print("J7 FOREGROUND / SEQUENTIAL | {} | {} epochs | seeds={}".format(
        [path.stem for path in selected], epochs, seeds
    ), flush=True)
    print("Data: {} | physical visible={} / logical={} | AMP=False; cache=True; scratch; EV6".format(
        data, os.environ.get("CUDA_VISIBLE_DEVICES", "unchanged"), device
    ), flush=True)
    if dry_run:
        print("Dry run: no dataset loading/training. FLOPs exclude NMS/mask decoding; no speed/accuracy claim.",
              flush=True)
        rows = [profile_model(path) for path in selected]
        print(json.dumps({"build_measurements": rows}, indent=2), flush=True)
        return rows
    if not data.is_file():
        raise FileNotFoundError("Edit DATA in RUN_CITRUS_J7.py to your actual data.yaml: {}".format(data))
    sources = source_manifest()
    pending = []
    for seed in seeds:
        for yaml_path in selected:
            name = "{}_seed{}".format(yaml_path.stem, seed)
            directory = project / name
            train_args = settings(epochs, seed, imgsz=640, cache=True)
            train_args.update(batch=int(batch), workers=int(workers), plots=bool(plots), save=True, half=False)
            record = {
                "protocol": PROTOCOL_ID, "name": yaml_path.stem, "seed": seed, "epochs": epochs,
                "data": str(data), "yaml_sha256": hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
                "sources": sources, "train": train_args, "validator": "EV6Validator (I/J input/2)",
                "training_images": "native global images; no paired/sliced/pre-resampled views",
                "final_checkpoint": "weights/best_mask.pt",
            }
            completed = directory / "j7_completed.json"
            if completed.is_file() and json.loads(completed.read_text(encoding="utf-8")) == record:
                print("SKIP completed {}".format(directory), flush=True)
                continue
            if directory.exists():
                raise FileExistsError(
                    "Partial/conflicting output exists: {}. Choose a NEW PROJECT or use ONLY to select other arms; "
                    "nothing was deleted or overwritten.".format(directory)
                )
            pending.append((yaml_path, seed, directory, record))
    if not pending:
        print("All requested runs already completed with this exact recorded protocol.", flush=True)
        return []
    if plots:
        print("Plotting preflight OK: Matplotlib {}".format(plotting_preflight()), flush=True)
    from citrus_j7_training import J7Trainer

    project.mkdir(parents=True, exist_ok=True)
    for yaml_path, seed, directory, record in pending:
        # Repeat the path check immediately before training, including changes from another process.
        if directory.exists():
            raise FileExistsError("Output appeared while queue was waiting: {}".format(directory))
        init_seeds(seed, deterministic=True)
        model = YOLO(str(yaml_path), task="segment")
        model.add_callback("on_pretrain_routine_end", partial(
            save_run_state, expected_dir=directory, record=record, cpu_threads=cpu_threads
        ))
        print("TRAIN {} -> {}".format(yaml_path.stem, directory), flush=True)
        model.train(
            trainer=J7Trainer, data=str(data), device=device, project=str(project),
            name=directory.name, exist_ok=False, **record["train"]
        )
        if not (directory / "selected_mask_validation" / "selected_checkpoint.json").is_file():
            raise RuntimeError("Training returned without the required mask-selected final evaluation")
        (directory / "j7_completed.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return [str(item[2]) for item in pending]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="Your actual dataset YAML; no fingerprint confirmation gate")
    parser.add_argument("--project", required=True, help="A new result project, not an existing partial run")
    parser.add_argument("--device", default="1", help="Physical GPU index, or cpu")
    parser.add_argument("--suite", choices=("priority", "screen", "all"), default="priority")
    parser.add_argument("--only", default="", help="Comma-separated J100...J109 IDs, overriding suite")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds; use 42,43,44 for final replication")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--dry-run", action="store_true", help="Build/profile only; no dataset is loaded")
    parser.add_argument("--no-plots", action="store_true",
                        help="Explicitly disable plots for smoke/unsupported plotting env")
    args = parser.parse_args()
    logical_device = bind_device(args.device, args.cpu_threads)
    run(args.data, logical_device, args.project, args.suite, args.epochs,
        tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip()), args.only,
        args.dry_run, args.cpu_threads, args.workers, args.batch, not args.no_plots)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("J7 stopped by Ctrl+C. The next model will NOT start.", flush=True)
        raise SystemExit(130)
