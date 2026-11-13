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
    parser.add_argument("--baseline-recipe", choices=("matched", "legacy78"), default="matched",
                        help="P00 uses the V12 recipe or the historical scratch/AMP1 recipe")
    parser.add_argument("--baseline-train-views", choices=("whole", "mixed", "both"), default="whole",
                        help="Training views for legacy78 P00; both adds a second independently trained P00")
    parser.add_argument("--only", default="", help="Comma-separated model names from the selected suite")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cache", choices=("true", "false"), default="true")
    parser.add_argument("--amp-audit", action="store_true")
    parser.add_argument("--repeat-ablations", action="store_true", help="Run the selected paper ablations at every seed")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--coco-secondary", action="store_true")
    parser.add_argument("--resume-partial", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Separate _SMOKE project, one epoch, two eval images")
    return parser.parse_args(argv)


def effective_training(args):
    """Return the established V12 best recipe; per-job dispatch handles the historical baseline."""
    from citrus_protocol import fixed_train_args, load_protocol

    result = fixed_train_args()
    result.update(load_protocol()["fixed_validation"])
    result.update(
        batch=args.batch, workers=args.workers, cache=args.cache == "true", imgsz=640,
        epochs=args.epochs, mask_ratio=2, copy_paste=0.3, cos_lr=False,
        patience=max(args.epochs, 300), amp=False, pretrained=str(args.pretrained.resolve()),
    )
    for key in (
        "citrus_boundary", "citrus_concavity", "citrus_query", "citrus_contrast",
        "citrus_exclusive", "citrus_quality", "citrus_topology", "citrus_vfl", "nwd_ratio",
    ):
        result[key] = 0.0
    return result


def legacy_baseline(args, job):
    return args.baseline_recipe == "legacy78" and job["model"] == NAMES[0]


def training_for_job(args, job):
    if legacy_baseline(args, job):
        from citrus_control78_batch import training_args
        result = training_args(args, job["seed"])
        result["cache"] = args.cache == "true"
    else:
        result = effective_training(args)
        result.update(seed=job["seed"], amp=job["amp"])
    return result


def job_name(job):
    suffix = "_mixed" if job.get("recipe") == "legacy78" and job.get("train_views") == "mixed" else ""
    return "%s%s_seed%d_amp%d" % (job["model"], suffix, job["seed"], job["amp"])


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


def finished_training(directory, epochs, legacy78=False):
    """A checkpoint alone is not proof of completed training."""
    if legacy78:
        from citrus_control78_batch import training_done
        return training_done(directory, epochs)
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
        name = job_name(job)
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
                    "training_recipe": job.get("recipe", "v12_best"),
                    "train_views": job.get("train_views", "mixed"),
                    "diagnostic_mask_ap50": metric.get("metrics/mAP50(M)"),
                    "diagnostic_mask_ap5095": metric.get("metrics/mAP50-95(M)"),
                    "diagnostic_mask_precision": metric.get("metrics/precision(M)"),
                    "diagnostic_mask_recall": metric.get("metrics/recall(M)"),
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
        if any(job.get("recipe") == "legacy78" for job in queue):
            summarize_recipe_comparison(project, rows)
    return rows


def summarize_recipe_comparison(project, rows):
    """Compare predeclared modes on the same split and seed, with percentage-point differences."""
    index = {(r["model"], r.get("train_views", "whole" if r["model"] == NAMES[0] else "mixed"),
              r["seed"], r["split"], r["mode"]): r for r in rows}
    comparisons, slicing = [], []
    measures = {"map50": "diagnostic_mask_ap50", "map5095": "diagnostic_mask_ap5095",
                "recall": "diagnostic_mask_recall", "recall_at_p90": "r_at_p90"}

    def differences(row, left, right, left_name, right_name):
        for metric, key in measures.items():
            a, b = left[key], right[key]
            row[left_name + "_" + metric + "_percent"] = None if a is None else 100*a
            row[right_name + "_" + metric + "_percent"] = None if b is None else 100*b
            row[metric + "_delta_pp"] = None if a is None or b is None else 100*(b-a)
        return row

    for seed, split in sorted({(r["seed"], r["split"]) for r in rows}):
        for views in ("whole", "mixed"):
            for label, base_mode, full_mode in (
                ("requested_full_system", "global", "trustedmask"),
                ("whole_vs_whole", "global", "global"),
                ("ten_views_vs_ten_views", "trustedmask", "trustedmask"),
            ):
                base = index.get((NAMES[0], views, seed, split, base_mode))
                full = index.get((NAMES[1], "mixed", seed, split, full_mode))
                if base is None or full is None:
                    continue
                row = dict(comparison=label, seed=seed, split=split, baseline_run=base["run"], e12_run=full["run"],
                           baseline_recipe=base["training_recipe"], e12_recipe=full["training_recipe"],
                           baseline_train_views=views, baseline_mode=base_mode, e12_mode=full_mode)
                comparisons.append(differences(row, base, full, "baseline", "e12"))
        for mode in ("global", "fixed_box", "crossmask", "trustedmask"):
            whole = index.get((NAMES[0], "whole", seed, split, mode))
            mixed = index.get((NAMES[0], "mixed", seed, split, mode))
            if whole is not None and mixed is not None:
                row = dict(seed=seed, split=split, evaluation_mode=mode, whole_run=whole["run"], sliced_run=mixed["run"],
                           training_recipe=whole["training_recipe"])
                slicing.append(differences(row, whole, mixed, "whole", "sliced"))
    for filename, records in (("P00_P01_COMPARISON.csv", comparisons), ("P00_SLICE_COMPARISON.csv", slicing)):
        if not records:
            continue
        target = project / filename
        temporary = target.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
        os.replace(str(temporary), str(target))
    return comparisons


def selected_queue(args):
    seeds = tuple(int(v) for v in args.seeds.split(",") if v.strip())
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be distinct nonnegative integers")
    if args.baseline_recipe == "legacy78" and args.amp_audit:
        raise ValueError("The legacy78 baseline already uses AMP=1. Use --baseline-recipe matched for --amp-audit.")
    queue = jobs(args.suite, seeds, args.amp_audit, args.repeat_ablations)
    if args.only:
        names = set(args.only.split(","))
        if not names.issubset({job["model"] for job in queue}):
            raise ValueError("--only contains names outside the selected suite: %s" % args.only)
        queue = [job for job in queue if job["model"] in names]
    if args.baseline_recipe == "legacy78":
        expanded = []
        for job in queue:
            historical = legacy_baseline(args, job)
            views = ("whole", "mixed") if args.baseline_train_views == "both" else (args.baseline_train_views,)
            for view in views if historical else ("mixed",):
                expanded.append(dict(job, amp=historical, recipe="legacy78" if historical else "v12_best",
                                     train_views=view))
        queue = expanded
    return seeds, queue


def main(argv=None):
    args = parse_args(argv)
    seeds, queue = selected_queue(args)
    if args.smoke:
        args.project = args.project.with_name(args.project.name + "_SMOKE")
        args.epochs = 1
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
        print("  %s role=%s" % (job_name(job), job["role"]), flush=True)
        recipe = training_for_job(args, job)
        print("    recipe=%s init=%s train_views=%s mask_ratio=%s dropout=%s copy_paste=%s patience=%s" %
              (job.get("recipe", "v12_best"), "pretrained" if recipe["pretrained"] else "scratch",
               job.get("train_views", "mixed"), recipe["mask_ratio"], recipe["dropout"],
               recipe["copy_paste"], recipe["patience"]), flush=True)
    if args.dry_run:
        print("PLAN ONLY: no dataset preparation, CUDA, training or output writes.", flush=True)
        return queue
    from citrus_control78_batch import project_lock
    args.project.mkdir(parents=True, exist_ok=True)
    with project_lock(args.project):
        return run_queue(args, queue)


def run_queue(args, queue):
    # Visibility is fixed BEFORE importing Torch. Ultralytics receives the same
    # physical ID, not an incorrect hard-coded logical zero.
    if args.device != "cpu":
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise RuntimeError("CUDA already initialized; run this file in a fresh Python process")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch
    import numpy as np
    import matplotlib.pyplot  # noqa: F401 -- fail before training on incompatible plotting packages
    from ultralytics import YOLO
    from citrus_e_v6_training import EV6TrainingTrainer
    from citrus_control78_training import Control78Trainer
    from citrus_slicing import source_files
    from citrus_paper_eval import evaluate
    from citrus_control78_batch import assert_resumable, prepare_training_views, restore_best, save_best_and_terminal

    helpers = importlib.import_module("20260918_citrus_e_v12_batch")
    data, pretrained, project = args.data.resolve(), args.pretrained.resolve(), args.project.resolve()
    needs_pretrained = any(not legacy_baseline(args, job) for job in queue)
    if not data.is_file() or (needs_pretrained and not pretrained.is_file()):
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
        "posthoc": ("public640 diagnostics plus separately labelled native-GT COCO" if args.coco_secondary
                    else "paired public640 diagnostics; no COCO evaluator"),
        "test_rule": "only P00/P01 main roles, fixed BEFORE outcomes; never ablation/AMP model selection",
        "smoke": args.smoke,
        "pretrained_sha256": hashlib.sha256(pretrained.read_bytes()).hexdigest() if needs_pretrained else None,
        "yaml_sha256": {job["model"]: hashlib.sha256(
            (YAML_DIR / (job["model"] + ".yaml")).read_bytes()).hexdigest() for job in queue},
    }
    if args.baseline_recipe == "legacy78":
        protocol.update(
            baseline_recipe="legacy78", initialization="per-job, see training",
            training={job_name(job): training_for_job(args, job) for job in queue},
            interpretation="P00 vs P01 compares the complete recipes; P01 vs P02-P10 keeps the V12 recipe",
        )
    project.mkdir(parents=True, exist_ok=True)
    frozen = project / "paper_protocol.json"
    if frozen.exists() and json.loads(frozen.read_text(encoding="utf-8")) != protocol:
        raise FileExistsError("This project has another protocol. Choose a NEW PROJECT; do not mix experiments.")
    save_json(frozen, protocol)
    provenance = {
        "command": sys.argv, "python": sys.version, "torch": torch.__version__, "time": time.time(),
        "physical_device": args.device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "pretrained_sha256": protocol["pretrained_sha256"],
    }
    save_json(project / "runtime_latest.json", provenance)
    prepared = None
    for job in queue:
        name = job_name(job)
        directory = project / name
        historical = legacy_baseline(args, job)
        whole_image = job.get("train_views", "mixed") == "whole"
        training = training_for_job(args, job)
        model = None
        try:
            if not finished_training(directory, args.epochs, historical):
                checkpoint = directory / "weights/last.pt"
                if directory.exists() and not (args.resume_partial and checkpoint.is_file()):
                    raise FileExistsError("Partial output without resumable last.pt: %s; use a new project." % directory)
                random.seed(job["seed"])
                np.random.seed(job["seed"])
                torch.manual_seed(job["seed"])
                resume = directory.exists()
                model = YOLO(str(checkpoint) if resume else str(YAML_DIR / (job["model"] + ".yaml")),
                             task="segment")
                if resume:
                    assert_resumable(model, directory)
                elif not historical:
                    model.load(str(pretrained))
                model.add_callback("on_model_save", save_best_and_terminal if historical else helpers.save_best_mask)
                model.add_callback("on_pretrain_routine_end", restore_best if historical else restore_mask_selection)
                model.add_callback("on_pretrain_routine_end", helpers.record_loaded_samples)
                if not resume and not historical:
                    model.add_callback("on_pretrain_routine_end", helpers.initialization_recorder(pretrained))
                save_json(project / "current_job.json", dict(job, status="training", resumed=resume))
                if not whole_image and prepared is None:
                    prepared = prepare_training_views(data, project / "_prepared_multiscale")
                kwargs = dict(training)
                kwargs.update(
                    data=str(data if whole_image else prepared), project=str(project), name=name, device=args.device,
                    seed=job["seed"], amp=job["amp"], exist_ok=False,
                )
                if resume:
                    kwargs["resume"] = True
                model.train(trainer=Control78Trainer if whole_image else EV6TrainingTrainer, **kwargs)
                if not finished_training(directory, args.epochs, historical):
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
                    limit=2 if args.smoke else 0,
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
    if args.baseline_recipe == "legacy78":
        for filename in ("P00_P01_COMPARISON.csv", "P00_SLICE_COMPARISON.csv"):
            if (project / filename).is_file():
                print("Comparison: %s" % (project / filename), flush=True)


if __name__ == "__main__":
    main()
