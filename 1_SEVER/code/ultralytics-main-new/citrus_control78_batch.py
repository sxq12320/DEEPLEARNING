"""Scratch training with the legacy78 recipe and paired val diagnostics.

Whole-image training is the default; --train-views selects explicit tile controls.
No GPU occupancy or split-confirmation guard, no implicit pretrained loading.
Normal validation remains EV6, not baseline COCO.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parent
MODELS = {
    "baseline": ("C00_yolo11n_78", "0_orange_yaml/T_series/00_t00_yolo11n_reference.yaml"),
    "candidate": ("C01_v12_04_78", "0_orange_yaml/E_V12_series/V12_04_shared_route_control.yaml"),
}
# Active training/augmentation parameters copied from 001_3_yolo11-seg_adamw/args.yaml.
# pretrained=False explicitly enforces the YAML-scratch behavior of that run.
RECIPE78 = dict(
    task="segment", epochs=300, patience=100, batch=16, imgsz=640,
    save=True, save_period=-1, cache=True, workers=4, pretrained=False,
    optimizer="AdamW", verbose=True, seed=42, deterministic=True,
    single_cls=False, rect=False, cos_lr=False, close_mosaic=10, amp=True,
    fraction=1.0, profile=False, freeze=None, multi_scale=0.0, compile=False,
    overlap_mask=True, mask_ratio=4, dropout=0.1, val=True, split="val",
    save_json=False, conf=None, iou=0.7, max_det=300, half=False,
    dnn=False, plots=True, augment=False, agnostic_nms=False, classes=None,
    retina_masks=False, lr0=0.001, lrf=0.01, momentum=0.937,
    weight_decay=0.0005, warmup_epochs=3.0, warmup_momentum=0.8,
    warmup_bias_lr=0.1, box=7.5, cls=0.5, dfl=1.5, nbs=64,
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
    scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
    bgr=0.0, mosaic=1.0, mixup=0.0, cutmix=0.0, copy_paste=0.0,
    copy_paste_mode="flip", auto_augment="randaugment", erasing=0.4,
)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    os.replace(str(temporary), str(path))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@contextmanager
def project_lock(project):
    """Prevent two writers to THIS output project; never inspect GPU occupancy."""
    handle = (project / ".writer.lock").open("a+b")
    locked = False
    try:
        handle.seek(0)
        if os.name == "nt":
            import msvcrt
            if os.fstat(handle.fileno()).st_size == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as exc:
                raise RuntimeError("This PROJECT is already running in another process.") from exc
        else:
            import fcntl
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as exc:
                raise RuntimeError("This PROJECT is already running in another process.") from exc
        locked = True
        yield
    finally:
        if locked:
            if os.name == "nt":
                import msvcrt
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(handle, fcntl.LOCK_UN)
        handle.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--device", default="1")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", default="42")
    parser.add_argument("--only", choices=("all", "baseline", "candidate"), default="all")
    parser.add_argument("--train-views", choices=("whole", "mixed", "both"), default="whole")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if args.epochs < 1 or args.batch < 1 or args.workers < 0:
        parser.error("epochs/batch must be positive; workers must be nonnegative")
    if args.device != "cpu" and not args.device.isdigit():
        parser.error("device must be one physical GPU index or cpu")
    try:
        args.seeds = [int(s) for s in args.seeds.split(",")]
    except ValueError:
        parser.error("seeds must be comma-separated integers")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("seeds must be distinct nonnegative integers")
    if args.smoke:
        args.project = args.project.with_name(args.project.name + "_SMOKE")
        args.epochs = 1
    return args


def training_args(args, seed):
    recipe = dict(RECIPE78)
    recipe.update(epochs=args.epochs, batch=args.batch, workers=args.workers, seed=seed)
    # Explicitly disable newer GLOBAL auxiliary switches. The V12 head's own
    # intrinsic objectives remain enabled and are part of the compared method.
    for key in (
        "citrus_boundary", "citrus_concavity", "citrus_query", "citrus_contrast",
        "citrus_exclusive", "citrus_quality", "citrus_topology", "citrus_vfl", "nwd_ratio",
    ):
        recipe[key] = 0.0
    return recipe


def queue_for(args):
    kinds = list(MODELS) if args.only == "all" else [args.only]
    views = ("whole", "mixed") if args.train_views == "both" else (args.train_views,)
    return [dict(kind=k, name="%s%s_seed%d" % (MODELS[k][0], "_mixed" if v == "mixed" else "", s),
                 seed=s, train_views=v) for s in args.seeds for k in kinds for v in views]


def csv_epoch(directory):
    path = directory / "results.csv"
    if not path.is_file():
        return 0
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]
    return int(float(rows[-1]["epoch"])) if rows else 0


def training_done(directory, epochs):
    marker = directory / "training_finished.json"
    if not marker.is_file():
        return False
    saved = json.loads(marker.read_text(encoding="utf-8"))
    actual = csv_epoch(directory)
    return (
        saved["budget"] == epochs and saved["last_epoch"] == actual and actual > 0
        and (actual >= epochs or saved["reason"] == "early_stop")
        and (directory / "weights/best_mask.pt").is_file()
    )


def restore_best(trainer):
    path = trainer.save_dir / "best_mask_selection.json"
    trainer.control78_best = -math.inf
    if path.is_file():
        trainer.control78_best = float(json.loads(path.read_text(encoding="utf-8"))["mask_mAP50_95"])
    effective_amp = bool(trainer.amp)
    runtime = dict(
        python=sys.version, device=str(trainer.device), amp_requested=True,
        amp_effective=effective_amp, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        train_dataset=type(trainer.train_loader.dataset).__name__,
        train_images=len(trainer.train_loader.dataset),
        validation="EV6 input/2", train_mask_ratio=trainer.args.mask_ratio,
        initialization="scratch" if not trainer.resume else "resume optimizer and EMA",
    )
    save_json(trainer.save_dir / "effective_runtime.json", runtime)
    if trainer.device.type == "cuda" and not effective_amp:
        raise RuntimeError("AMP safety check disabled AMP. Stopping rather than silently mixing AMP protocols.")
    if trainer.device.type == "cpu":
        print("CPU engineering check: AMP is unavailable; this is NOT a formal AMP=1 experiment.", flush=True)


def save_best_and_terminal(trainer):
    score = float(trainer.metrics.get("metrics/mAP50-95(M)", float("nan")))
    if not math.isfinite(score):
        raise RuntimeError("Missing/non-finite Mask AP50-95; refuse to select a misleading checkpoint.")
    if score > getattr(trainer, "control78_best", -math.inf):
        target = trainer.save_dir / "weights/best_mask.pt"
        temporary = target.with_suffix(".pt.tmp")
        shutil.copy2(trainer.last, temporary)
        os.replace(str(temporary), str(target))
        save_json(trainer.save_dir / "best_mask_selection.json",
                  dict(epoch=trainer.epoch + 1, mask_mAP50_95=score))
        trainer.control78_best = score
    if trainer.stop or trainer.epoch + 1 >= trainer.epochs:
        # Written after CSV/checkpoint save, before final_eval strips optimizer.
        save_json(trainer.save_dir / "training_finished.json", dict(
            budget=trainer.epochs, last_epoch=trainer.epoch + 1,
            reason="budget" if trainer.epoch + 1 >= trainer.epochs else "early_stop",
        ))


def assert_resumable(model, directory):
    checkpoint = model.ckpt
    if not checkpoint or checkpoint.get("epoch", -1) < 0 or checkpoint.get("optimizer") is None:
        raise RuntimeError(
            "Partial run has no resumable optimizer checkpoint: %s. "
            "Use a new PROJECT; this runner will not silently restart or overwrite it." % directory
        )


def prepare_training_views(data, output):
    """Expose crop fallback; do not label an all-global dataset as tile training."""
    from citrus_e_v5_slicing import prepare_multiscale_views
    prepared = prepare_multiscale_views(data, output, 640)
    saved = json.loads((Path(output) / "views.json").read_text(encoding="utf-8"))
    groups = saved["groups"]
    tiled = [sum(not view["global_view"] for view in group["views"]) for group in groups]
    audit = dict(sources=len(groups), retained_tiles=sum(tiled),
                 sources_with_tiles=sum(n > 0 for n in tiled),
                 sources_global_only=sum(n == 0 for n in tiled),
                 skipped_tiles=len(saved.get("skipped_tiles", [])))
    save_json(Path(output) / "training_view_audit.json", audit)
    print("TRAIN VIEW AUDIT: %s" % json.dumps(audit), flush=True)
    if not audit["retained_tiles"]:
        raise ValueError("No usable training tiles remain under polygon clipping. See views.json; "
                         "original labels are unchanged. This is not a valid sliced-training comparison.")
    return prepared


def summarize(project):
    rows = []
    for report in sorted(project.glob("C0*_seed*/eval_val/paired_metrics.json")):
        data = json.loads(report.read_text(encoding="utf-8"))
        for mode in ("global", "fixed_box", "crossmask", "trustedmask"):
            metric = data["summary"][mode]
            rows.append(dict(
                run=report.parent.parent.name, mode=mode,
                mask_ap50_percent=100 * metric["metrics/mAP50(M)"],
                mask_ap5095_percent=100 * metric["metrics/mAP50-95(M)"],
                recall_percent=100 * metric["metrics/recall(M)"],
                r_at_p90_percent=100 * (metric.get("operating_p90") or {}).get("recall", 0),
                tiny_matched=metric.get("tiny_matched"), tiny_n=metric.get("tiny_n"),
                median_ms=metric.get("median_ms"),
                protocol="paired public640 diagnostic; NOT official COCO",
            ))
    if rows:
        path = project / "CONTROL78_RESULTS.csv"
        temporary = path.with_suffix(".csv.tmp")
        with temporary.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(str(temporary), str(path))
    return rows


def main(argv=None):
    args = parse_args(argv)
    queue = queue_for(args)
    print("CONTROL78 | foreground sequential | scratch | train views=%s | AMP=1 | cache=True" %
          args.train_views, flush=True)
    print("Data: %s\nProject: %s\nDevice: %s\nQueue: %s" %
          (args.data, args.project, args.device, [j["name"] for j in queue]), flush=True)
    print("Epoch budget=%d; legacy patience=100 (may stop early). Train masks /4, common EV6 val /2." %
          args.epochs, flush=True)
    print("No pretrained loading, no busy-GPU or split-confirmation gate.", flush=True)
    if args.dry_run:
        print(json.dumps(training_args(args, args.seeds[0]), indent=2), flush=True)
        return queue
    data, project = args.data.resolve(), args.project.resolve()
    if not data.is_file():
        raise FileNotFoundError("Edit DATA to YOUR server data.yaml: %s. No old-data fallback." % data)
    for job in queue:
        if not (ROOT / MODELS[job["kind"]][1]).is_file():
            raise FileNotFoundError("Upload the whole code folder; missing " + MODELS[job["kind"]][1])
    # Set visibility before Torch import; every downstream select_device sees
    # the same physical index. Do not change device=1 to device=0 after masking.
    if args.device != "cpu":
        if "torch" in sys.modules and sys.modules["torch"].cuda.is_initialized():
            raise RuntimeError("CUDA already initialized. Run this file in a fresh Python process.")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch
    import matplotlib.pyplot  # Fail before training if the plotting dependencies are incompatible.
    from ultralytics import YOLO
    from citrus_control78_training import Control78Trainer
    from citrus_e_v6_training import EV6TrainingTrainer
    from citrus_paper_eval import evaluate

    project.mkdir(parents=True, exist_ok=True)
    with project_lock(project):
        try:
            git = subprocess.run(["git", "status", "--short"], cwd=str(ROOT),
                                 capture_output=True, text=True, timeout=15)
            git_status = git.stdout
        except (OSError, subprocess.TimeoutExpired):
            git_status = "unavailable"
        save_json(project / "runtime_latest.json", dict(
            command=sys.argv, python=sys.version, torch=torch.__version__, time=time.time(),
            device=args.device, git_status=git_status,
        ))
        for job in queue:
            directory = project / job["name"]
            yaml_path = ROOT / MODELS[job["kind"]][1]
            recipe = training_args(args, job["seed"])
            protocol = dict(
                training=recipe, data=str(data), data_yaml_text=data.read_text(encoding="utf-8"),
                model=MODELS[job["kind"]][1], model_sha256=digest(yaml_path),
                initialization="scratch", train_views="whole-image only",
                validation="EV6 input/2; common for both networks",
                paired_evaluation="val only, global+9 fine, four selection modes, no COCO",
                smoke=args.smoke,
                runner_sha256=digest(__file__),
                trainer_sha256=digest(ROOT / "citrus_control78_training.py"),
            )
            if job["train_views"] == "mixed":
                protocol["train_views"] = "mixed whole/coarse/fine; EV6 cached-view sampler"
                protocol["sampler_sha256"] = digest(ROOT / "citrus_e_v5_slicing.py")
            protocol_path = directory / "control78_protocol.json"
            if directory.exists():
                if not protocol_path.is_file() or json.loads(protocol_path.read_text(encoding="utf-8")) != protocol:
                    raise FileExistsError("Existing output has different/unrecorded settings; use a new PROJECT: %s" %
                                          directory)
            save_json(protocol_path, protocol)
            model = None
            try:
                if not training_done(directory, args.epochs):
                    last = directory / "weights/last.pt"
                    if not last.is_file() and csv_epoch(directory):
                        raise RuntimeError("Training CSV exists but last.pt is missing; refusing to restart.")
                    if last.is_file():
                        model = YOLO(str(last), task="segment")
                        assert_resumable(model, directory)
                        print("RESUME %s" % job["name"], flush=True)
                    else:
                        model = YOLO(str(yaml_path), task="segment")
                    model.add_callback("on_pretrain_routine_end", restore_best)
                    model.add_callback("on_model_save", save_best_and_terminal)
                    save_json(project / "current_job.json", dict(job, stage="training"))
                    train_data = (prepare_training_views(data, project / "_prepared_multiscale")
                                  if job["train_views"] == "mixed" else data)
                    trainer_type = EV6TrainingTrainer if job["train_views"] == "mixed" else Control78Trainer
                    kwargs = dict(recipe, data=str(train_data), project=str(project), name=job["name"],
                                  device=args.device, exist_ok=True, resume=str(last) if last.is_file() else False)
                    model.train(trainer=trainer_type, **kwargs)
                    if not training_done(directory, args.epochs):
                        raise RuntimeError("Training did not reach a valid terminal checkpoint.")
                else:
                    print("SKIP COMPLETED TRAINING %s (reuse best_mask.pt)" % job["name"], flush=True)
                model = None
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
                save_json(project / "current_job.json", dict(job, stage="paired_validation"))
                evaluate(
                    directory / "weights/best_mask.pt", data, directory / "eval_val",
                    device=args.device, split="val", mechanisms=False, coco_secondary=False,
                    limit=2 if args.smoke else 0,
                )
                save_json(directory / "completed.json", dict(job, status="complete", smoke=args.smoke))
                summarize(project)
            except BaseException as exc:
                save_json(project / "current_job.json",
                          dict(job, stage="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               error=str(exc)))
                print("STOPPED: no next model starts. Rerun the same file to resume.", flush=True)
                raise
            finally:
                model = None
                gc.collect()
                if torch.cuda.is_initialized():
                    torch.cuda.empty_cache()
        save_json(project / "current_job.json", dict(stage="complete", jobs=len(queue)))
        print("DONE: %s" % (project / "CONTROL78_RESULTS.csv"), flush=True)
    return queue


if __name__ == "__main__":
    main()
