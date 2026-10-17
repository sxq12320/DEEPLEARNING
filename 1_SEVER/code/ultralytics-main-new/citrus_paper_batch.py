"""Sequential, resumable paper experiments with an explicitly frozen queue."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import importlib
import json
import os
import random
import sys
import time
from pathlib import Path

import yaml

from citrus_paper_suite import NAMES, ROOT, YAML_DIR, configurations, jobs


def save_json(path, payload):
    """Atomically update this run's state, not an existing historical experiment."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(str(temporary), str(path))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--device", default="1")
    parser.add_argument("--suite", choices=("core", "paper"), default="paper")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=("true", "false"), default="true")
    parser.add_argument("--amp-audit", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--coco-secondary", action="store_true")
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def effective_training(args):
    """Apply one recipe to EVERY arm, with numbered AMP sensitivity only."""
    from citrus_protocol import fixed_train_args, load_protocol

    result = fixed_train_args()
    result.update(load_protocol()["fixed_validation"])
    result.update(
        batch=args.batch, workers=args.workers, cache=args.cache == "true", imgsz=640,
        epochs=args.epochs, mask_ratio=2, copy_paste=0.3, cos_lr=False,
        patience=max(args.epochs, 300), amp=False, pretrained=False,
    )
    for key in (
        "citrus_boundary", "citrus_concavity", "citrus_query", "citrus_contrast",
        "citrus_exclusive", "citrus_quality", "citrus_topology", "citrus_vfl", "nwd_ratio",
    ):
        result[key] = 0.0
    return result


def verify_configs():
    """Reject stale controls rather than silently regenerating YAMLs on a server."""
    for name, expected in configurations().items():
        path = YAML_DIR / (name + ".yaml")
        if not path.is_file() or yaml.safe_load(path.read_text(encoding="utf-8")) != expected:
            raise ValueError("Missing/stale control YAML: %s; upload the complete code folder." % path)


def restore_mask_selection(trainer):
    """A resumed low-score epoch must not overwrite the previous best-mask file."""
    marker = trainer.save_dir / "best_mask_selection.json"
    if marker.is_file():
        trainer.sage_best_mask = float(json.loads(marker.read_text(encoding="utf-8"))["mask_mAP50_95"])
    marker = trainer.save_dir / "best_mask50_selection.json"
    if marker.is_file():
        trainer.ev5_best_mask50 = float(json.loads(marker.read_text(encoding="utf-8"))["mask_mAP50"])


def finished_training(directory, epochs):
    """A checkpoint alone is not proof of completed training."""
    metrics = directory / "results.csv"
    if not metrics.is_file() or not (directory / "weights/best_mask.pt").is_file():
        return False
    with metrics.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return bool(rows) and int(float({k.strip(): v for k, v in rows[-1].items()}["epoch"])) >= epochs


def summarize(project, queue):
    """Summarize only jobs with completed evaluation; retain seed-level rows."""
    rows = []
    for job in queue:
        name = "%s_seed%d_amp%d" % (job["model"], job["seed"], job["amp"])
        for split in ("val", "test"):
            path = project / name / ("eval_" + split) / "paired_metrics.json"
            if not path.is_file():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            for mode, metric in payload["summary"].items():
                coco = payload.get("coco", {}).get(mode, {})
                rows.append({
                    "run": name, "model": job["model"], "role": job["role"], "seed": job["seed"],
                    "amp": job["amp"], "split": split, "mode": mode,
                    "diagnostic_mask_ap50": metric.get("metrics/mAP50(M)"),
                    "diagnostic_mask_ap5095": metric.get("metrics/mAP50-95(M)"),
                    "tiny_matched": metric.get("tiny_matched"), "tiny_n": metric.get("tiny_n"),
                    "r_at_p90": (metric.get("operating_p90") or {}).get("recall"),
                    "median_ms": metric.get("median_ms"), "coco_ap5095": coco.get("AP"),
                    "coco_ap50": coco.get("AP50"), "coco_ap_small": coco.get("AP_small"),
                })
    if rows:
        with (project / "paper_results.csv").open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return rows


def main(argv=None):
    args = parse_args(argv)
    seeds = tuple(int(v) for v in args.seeds.split(",") if v.strip())
    queue = jobs(args.suite, seeds, args.amp_audit)
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        raise ValueError("epochs/batch must be positive and workers nonnegative")
    if args.device != "cpu" and not args.device.isdigit():
        raise ValueError("Use one physical GPU index or cpu; no multi-GPU launcher is used")
    verify_configs()
    print("PAPER FOREGROUND RUN | sequential | no nohup | no GPU occupancy guard", flush=True)
    print("Data: %s\nProject: %s\nEpochs per run: %s\nSeeds: %s" %
          (args.data, args.project, args.epochs, seeds), flush=True)
    print("Training queue (%d jobs; %d total epochs):" % (len(queue), len(queue) * args.epochs), flush=True)
    for job in queue:
        print("  %s seed=%d amp=%s role=%s" % (job["model"], job["seed"], job["amp"], job["role"]), flush=True)
    if args.dry_run:
        print("PLAN ONLY: no dataset preparation, CUDA, training or output writes.", flush=True)
        return queue
    # Visibility is fixed BEFORE importing Torch. Ultralytics receives the same
    # physical ID, not an incorrect hard-coded logical zero.
    if args.device != "cpu":
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import numpy as np
    import torch
    from ultralytics import YOLO
    from citrus_e_v5_slicing import prepare_multiscale_views
    from citrus_e_v6_training import EV6TrainingTrainer
    from citrus_slicing import source_files
    from citrus_paper_eval import evaluate

    helpers = importlib.import_module("20260918_citrus_e_v12_batch")
    data, pretrained, project = args.data.resolve(), args.pretrained.resolve(), args.project.resolve()
    if not data.is_file() or not pretrained.is_file():
        raise FileNotFoundError("Check DATA and PRETRAINED; there is no fallback to the old dataset.")
    config, _ = source_files(data, "val")
    if len(config["names"]) != 1:
        raise ValueError("This preregistered citrus study requires exactly one class")
    if args.evaluate_test:
        source_files(data, "test")  # Fail before training if a requested test split is absent.
    if args.coco_secondary:
        import pycocotools.cocoeval  # noqa: F401 -- dependency check before training
    training = effective_training(args)
    protocol = {
        "queue": queue, "training": training, "data": str(data), "initialization": str(pretrained),
        "epochs": args.epochs, "evaluate_test": args.evaluate_test, "coco_secondary": args.coco_secondary,
        "data_yaml": data.read_text(encoding="utf-8"), "validation": "EV6 input/2 throughout training",
        "posthoc": "public640 diagnostics plus separately labelled native-GT COCO",
        "test_rule": "only P00/P01 main roles, fixed BEFORE outcomes; never ablation/AMP model selection",
    }
    project.mkdir(parents=True, exist_ok=True)
    frozen = project / "paper_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != protocol:
        raise FileExistsError("This project has another protocol. Choose a NEW PROJECT; do not mix experiments.")
    save_json(frozen, protocol)
    provenance = {
        "command": sys.argv, "python": sys.version, "torch": torch.__version__, "time": time.time(),
        "physical_device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pretrained_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest(),
    }
    save_json(project / "runtime_latest.json", provenance)
    prepared = prepare_multiscale_views(data, project / "_prepared_multiscale", 640)
    for job in queue:
        name = "%s_seed%d_amp%d" % (job["model"], job["seed"], job["amp"])
        directory = project / name
        model = None
        try:
            if not finished_training(directory, args.epochs):
                checkpoint = directory / "weights/last.pt"
                if directory.exists() and not (args.resume_partial and checkpoint.is_file()):
                    raise FileExistsError("Partial output without resumable last.pt: %s; use a new project." % directory)
                random.seed(job["seed"])
                np.random.seed(job["seed"])
                torch.manual_seed(job["seed"])
                resume = directory.exists()
                model = YOLO(str(checkpoint) if resume else str(YAML_DIR / (job["model"] + ".yaml")),
                             task="segment")
                if not resume:
                    model.load(str(pretrained))
                model.add_callback("on_model_save", helpers.save_best_mask)
                model.add_callback("on_pretrain_routine_end", restore_mask_selection)
                model.add_callback("on_pretrain_routine_end", helpers.record_loaded_samples)
                if not resume:
                    model.add_callback("on_pretrain_routine_end", helpers.initialization_recorder(pretrained))
                save_json(project / "current_job.json", dict(job, status="training", resumed=resume))
                kwargs = dict(training)
                kwargs.update(
                    data=str(prepared), project=str(project), name=name, device=args.device,
                    seed=job["seed"], amp=job["amp"], exist_ok=False,
                )
                if resume:
                    kwargs["resume"] = True
                model.train(trainer=EV6TrainingTrainer, **kwargs)
                if not finished_training(directory, args.epochs):
                    raise RuntimeError("Training ended before declared budget; not marked complete: %s" % name)
                save_json(directory / "training_completed.json", dict(job, epochs=args.epochs))
            model = None
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
            splits = ["val", "test"] if args.evaluate_test and job["role"] == "main" else ["val"]
            for split in splits:
                output = directory / ("eval_" + split)
                save_json(project / "current_job.json", dict(job, status="evaluation", split=split))
                evaluate(
                    directory / "weights/best_mask.pt", data, output, device=args.device, split=split,
                    mechanisms=(split == "val"), coco_secondary=args.coco_secondary,
                )
            save_json(directory / "paper_completed.json", dict(job, epochs=args.epochs))
            summarize(project, queue)
        except KeyboardInterrupt:
            save_json(project / "current_job.json", dict(job, status="interrupted"))
            print("Stopped. No next model starts; rerun the same entry to resume.", flush=True)
            raise
        except Exception as exc:
            save_json(project / "current_job.json", dict(job, status="failed", error=repr(exc)))
            raise  # Fail fast. Never hide a failed arm and continue reporting a subset.
        finally:
            del model
            gc.collect()
            if torch.cuda.is_initialized():
                torch.cuda.empty_cache()
    print("All declared jobs completed. Read: %s" % (project / "paper_results.csv"), flush=True)


if __name__ == "__main__":
    main()
