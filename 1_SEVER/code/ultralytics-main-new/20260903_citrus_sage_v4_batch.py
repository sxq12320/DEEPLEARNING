"""SAGE-v4 fixed-protocol sequential training. For VS Code use RUN_CITRUS_FOREGROUND.py.

Example: python 20260903_citrus_sage_v4_batch.py --data /data/orange/data.yaml --suite screen --epochs 300
No shell, nohup, background queue or concurrent model training is used.
"""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import platform
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from citrus_protocol import fixed_train_args, load_protocol, validate_locked_runtime, write_protocol_snapshot

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "SAGE_series"
NAMES = (
    "SAGE30_official_control",
    "SAGE31_bounded_fusion",
    "SAGE32_direct_mask_detail",
    "SAGE33_fusion_detail",
    "SAGE34_decoupled_geometry",
    "SAGE35_gated_p4",
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--suite", choices=("smoke", "screen", "all", "control", "backbone", "final"), default="screen")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--device", default="0")
    fixed = fixed_train_args()
    for key in ("batch", "imgsz", "workers"):
        parser.add_argument(f"--{key}", type=int, default=fixed[key])
    parser.add_argument("--cache", choices=("false", "ram", "disk"), default="false")
    amp = parser.add_mutually_exclusive_group()
    amp.add_argument("--amp", dest="amp", action="store_true")
    amp.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=fixed["amp"])
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--only", default="")
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def select_names(args):
    selections = {
        "smoke": NAMES,
        "screen": NAMES[:5],
        "all": NAMES,
        "control": NAMES[:1],
        "backbone": (NAMES[3], NAMES[5]),
        "final": (NAMES[3],),
    }
    if args.only:
        requested = {name.strip() for name in args.only.split(",") if name.strip()}
        if not requested or requested - set(NAMES):
            raise ValueError(f"Unknown/empty model selection; choose from {NAMES}")
        return [name for name in NAMES if name in requested]
    return list(selections[args.suite])


def completed(run_dir, epochs, seed):
    """Do not mistake a partial run with best.pt for a finished experiment."""
    marker, metrics = run_dir / "completed.json", run_dir / "results.csv"
    if not marker.is_file() or not metrics.is_file() or not (run_dir / "weights" / "best.pt").is_file():
        return False
    try:
        saved = json.loads(marker.read_text(encoding="utf-8"))
        with metrics.open(encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        final_epoch = int(float({k.strip(): v for k, v in rows[-1].items()}["epoch"]))
        return saved["epochs"] == epochs and saved["seed"] == seed and final_epoch >= epochs
    except (ValueError, KeyError, IndexError):
        return False


def save_best_mask(trainer):
    """Save a second checkpoint selected strictly by mask AP50-95; keep official best.pt unchanged."""
    score = float(trainer.metrics.get("metrics/mAP50-95(M)", -1))
    if score > getattr(trainer, "sage_best_mask", -1):
        trainer.sage_best_mask = score
        shutil.copy2(trainer.last, trainer.save_dir / "weights" / "best_mask.pt")
        (trainer.save_dir / "best_mask_selection.json").write_text(
            json.dumps({"epoch": trainer.epoch + 1, "mask_mAP50_95": score}, indent=2), encoding="utf-8"
        )


def append_event(path, event):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")


def record_loaded_samples(trainer):
    """Save actual loader membership, not an assumed identity inferred from a data.yaml path."""
    summary = {"rgb_paired_cache_dedup": True}
    for split, loader in (("train", trainer.train_loader), ("val", trainer.test_loader)):
        if loader is None:
            continue
        dataset = loader.dataset
        files = list(dataset.im_files)
        summary[split] = {
            "images": len(files),
            "instances": sum(len(label["cls"]) for label in dataset.labels),
            "npy_sources": sum(Path(p).suffix.lower() == ".npy" for p in files),
        }
        (trainer.save_dir / f"{split}_loaded_files.txt").write_text("\n".join(files) + "\n", encoding="utf-8")
    summary["device"] = str(trainer.device)
    summary["gpu"] = torch.cuda.get_device_name(trainer.device) if trainer.device.type == "cuda" else "CPU"
    (trainer.save_dir / "loaded_data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"ACTUAL DATA LOADER: {summary}", flush=True)


def main():
    args = parse_args()
    if args.epochs < 1 or (args.suite == "smoke" and args.epochs > 3):
        raise ValueError("Use positive epochs; smoke requires 1--3 epochs")
    if "," in args.device:
        raise ValueError("This suite is sequential and single-device only")
    validate_locked_runtime(
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        cache=False if args.cache == "false" else args.cache,
        amp=args.amp,
    )
    data = args.data.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(data)
    config = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Expected a dataset YAML with train/val/names: {data}")
    if int(config.get("channels", 3)) != 3:
        raise ValueError("SAGE-v4 paper-1 experiments use RGB only")
    seeds = [int(value) for value in args.seeds.split(",") if value.strip()]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("Use a nonempty list of distinct seeds")
    names = select_names(args)
    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.torch_utils import get_flops

    print(f"Python: {sys.executable}\nTorch: {torch.__version__}\nData: {data}\nQueue: {names}", flush=True)
    if args.dry_run:
        for name in names:
            model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=len(config["names"]), verbose=False).eval()
            with torch.no_grad():
                model(torch.zeros(1, 3, 128, 128))
            print(
                f"BUILD OK {name}: {sum(p.numel() for p in model.parameters()):,} params; "
                f"{get_flops(model, 640):.3f} GFLOPs @640",
                flush=True,
            )
        return
    if load_protocol()["fixed_validation"].get("plots", True):
        try:
            import matplotlib.pyplot  # noqa: F401 -- fail before training if plots cannot be generated
        except ImportError as error:
            raise RuntimeError(
                "Plotting dependencies are incompatible in this Python environment. "
                "Resolve its NumPy/Matplotlib versions before training; no hyperparameters were changed."
            ) from error
    pretrained = args.pretrained.expanduser().resolve()
    if not pretrained.is_file():
        raise FileNotFoundError(pretrained)
    project = (
        (args.project or ROOT / "1_results" / "SAGE_series" / f"CITRUS_SAGE_V4_{args.suite.upper()}_{args.epochs}EP")
        .expanduser()
        .resolve()
    )
    _, digest = write_protocol_snapshot(
        project,
        {
            "series": "SAGE-v4",
            "rgb_paired_cache_dedup": True,
            "checkpoint_selection": "best.pt=box+mask; best_mask.pt=mask-only",
        },
    )
    source_files = [
        Path(__file__).resolve(),
        ROOT / "citrus_protocol.py",
        ROOT / "protocols/citrus_paper1_formal_v1.yaml",
        ROOT / "ultralytics/nn/modules/citrus_sage_v4.py",
        ROOT / "ultralytics/utils/sage_v4_loss.py",
        ROOT / "ultralytics/nn/tasks.py",
        ROOT / "ultralytics/data/base.py",
        ROOT / "ultralytics/utils/loss.py",
        ROOT / "ultralytics/utils/metrics.py",
        *[YAML_DIR / f"{name}.yaml" for name in NAMES],
    ]
    source_hashes = {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest() for path in source_files
    }
    provenance = project / "_protocol" / "implementation_sha256.json"
    source_content = json.dumps(source_hashes, sort_keys=True, indent=2)
    if provenance.exists() and provenance.read_text(encoding="utf-8") != source_content:
        raise FileExistsError("Implementation changed within this project; use a new project directory")
    provenance.write_text(source_content, encoding="utf-8")
    snapshot = project / "_protocol" / "dataset_source_snapshot.yaml"
    content = data.read_text(encoding="utf-8")
    if snapshot.exists() and snapshot.read_text(encoding="utf-8") != content:
        raise FileExistsError("Dataset YAML differs from this project's saved snapshot; use a new project")
    snapshot.write_text(content, encoding="utf-8")
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, timeout=10).strip()
    except (OSError, subprocess.SubprocessError):
        revision = "unavailable"
    ledger = project / "_protocol" / "ledger.jsonl"
    failures = []
    for name in names:
        for seed in seeds:
            run_name = f"{name}_seed{seed}"
            directory = project / run_name
            if args.skip_completed and completed(directory, args.epochs, seed):
                print(f"SKIP completed: {run_name}", flush=True)
                continue
            if directory.exists():
                raise FileExistsError(f"Will not overwrite a partial or existing experiment: {directory}")
            event = {
                "name": name,
                "seed": seed,
                "epochs": args.epochs,
                "data": str(data),
                "git": revision,
                "protocol": digest,
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "yaml_sha256": hashlib.sha256((YAML_DIR / f"{name}.yaml").read_bytes()).hexdigest(),
            }
            append_event(ledger, {**event, "status": "started", "time": time.time()})
            model = None
            try:
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                model = YOLO(str(YAML_DIR / f"{name}.yaml"), task="segment").load(str(pretrained))
                model.add_callback("on_model_save", save_best_mask)
                model.add_callback("on_pretrain_routine_end", record_loaded_samples)
                training = fixed_train_args()
                training.update(load_protocol()["fixed_validation"])
                # Explicitly disable all legacy method losses regardless of package defaults.
                training.update(
                    {
                        key: 0.0
                        for key in (
                            "citrus_boundary",
                            "citrus_concavity",
                            "citrus_query",
                            "citrus_contrast",
                            "citrus_exclusive",
                            "citrus_quality",
                            "citrus_topology",
                            "citrus_vfl",
                            "nwd_ratio",
                        )
                    }
                )
                training.update(
                    data=str(data),
                    project=str(project),
                    name=run_name,
                    epochs=args.epochs,
                    device=args.device,
                    seed=seed,
                    exist_ok=False,
                )
                model.train(**training)
                (directory / "completed.json").write_text(json.dumps(event, indent=2), encoding="utf-8")
                append_event(ledger, {**event, "status": "completed", "time": time.time()})
            except KeyboardInterrupt:
                append_event(ledger, {**event, "status": "interrupted", "time": time.time()})
                raise  # never continue the queue after Ctrl+C
            except Exception as error:
                append_event(ledger, {**event, "status": "failed", "error": repr(error), "time": time.time()})
                failures.append(name)
                if args.fail_fast:
                    raise
                print(f"FAILED {name}: {error!r}", flush=True)
            finally:
                del model
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
    if failures:
        raise RuntimeError(f"Failed experiments: {failures}; inspect the ledger")


if __name__ == "__main__":
    main()
