"""One isolated foreground job. Run via RUN_CITRUS_BASELINES_AMP.py, not an editable Ultralytics install."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

# -I suppresses PYTHONPATH and the repository root, preventing accidental use of custom historical modules.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import save_json, seed_everything, sha256, snapshot
from registry import MODELS, YOLO_TRAIN


def official_mmdet_root():
    import mmdet

    root = Path(mmdet.__file__).resolve().parent / ".mim"
    if not (root / "configs").is_dir():
        raise FileNotFoundError(f"Official packaged MMDetection configs missing: {root}. Reinstall mmdet==3.3.0")
    return root


def model_zoo_weight(root, config):
    import yaml

    metadata = root / Path(config).parent / "metafile.yml"
    items = yaml.safe_load(metadata.read_text(encoding="utf-8"))["Models"]
    match = next((item for item in items if item.get("Config") == config), None)
    if not match or not match.get("Weights"):
        raise RuntimeError(f"No official COCO checkpoint for {config}; will NOT silently train from scratch")
    return match["Weights"]


def require_version(package, version):
    actual = importlib.metadata.version(package)
    if actual != version:
        raise RuntimeError(f"Expected {package}=={version}, found {actual}. Use the new baseline environment.")


def preflight(families, cpu=False):
    import torch
    import pycocotools.mask  # noqa: F401
    import yaml  # noqa: F401

    if not cpu and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable. Check interpreter, driver and CUDA_VISIBLE_DEVICES; AMP=1 needs CUDA.")
    if "yolo" in families:
        require_version("ultralytics", "8.4.60")
        import ultralytics

        if HERE.parent in Path(ultralytics.__file__).resolve().parents:
            raise RuntimeError("Imported the modified project Ultralytics. Install the official wheel in a NEW env.")
        print("Official YOLO:", ultralytics.__file__)
    if "rfdetr" in families:
        require_version("rfdetr", "1.4.0")
        from rfdetr import RFDETRSegNano  # noqa: F401
        from rfdetr.config import RFDETRSegNanoConfig

        assert RFDETRSegNanoConfig(amp=False).amp is False
        assert RFDETRSegNanoConfig(amp=True).amp is True
        if not cpu and not torch.cuda.is_bf16_supported():
            raise RuntimeError("RF-DETR 1.4 native AMP requires BF16-capable CUDA hardware; no silent FP16 fallback")
    if "mmdet" in families:
        require_version("mmdet", "3.3.0")
        require_version("mmcv", "2.1.0")
        from mmcv.ops import nms
        from mmengine.config import Config

        root = official_mmdet_root()
        for recipe in MODELS.values():
            if recipe["family"] == "mmdet":
                Config.fromfile(str(root / recipe["config"]))
                model_zoo_weight(root, recipe["config"])
        if not cpu:
            nms(torch.tensor([[0.0, 0.0, 4.0, 4.0]], device="cuda"), torch.ones(1, device="cuda"), 0.5)
    if not cpu:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        for enabled in (False, True):
            layer = torch.nn.Conv2d(3, 4, 3).cuda()
            with torch.autocast("cuda", dtype=torch.float16, enabled=enabled):
                loss = layer(torch.randn(2, 3, 16, 16, device="cuda")).float().square().mean()
            loss.backward()
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite CUDA preflight loss with AMP={enabled}")
    print("PREFLIGHT OK", snapshot(), flush=True)


def train_yolo(job, prepared, run_dir):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.downloads import attempt_download_asset

    checkpoint = Path(attempt_download_asset(job["recipe"]["weights"])).resolve()
    save_json(run_dir / "initialization.json", dict(weights=str(checkpoint), sha256=sha256(checkpoint)))
    seed_everything(job["seed"])
    model = YOLO(str(checkpoint))
    options = dict(YOLO_TRAIN)
    options.update(
        data=str(prepared / "yolo/data.yaml"),
        epochs=job["epochs"],
        patience=job["epochs"] + 1,
        batch=job["recipe"]["batch"],
        workers=job["workers"],
        seed=job["seed"],
        amp=job["amp"],
        device="",
        project=str(run_dir),
        name="train",
        exist_ok=False,
        plots=True,
        save=True,
    )
    save_json(run_dir / "effective_train.json", options)

    def verify_amp(trainer):
        actual = bool(trainer.amp)
        save_json(
            run_dir / "amp_actual.json",
            dict(
                requested=job["amp"],
                actual=actual,
                scaler_enabled=trainer.scaler.is_enabled(),
                dtype="float16" if actual else "float32",
            ),
        )
        if actual != job["amp"]:
            raise RuntimeError("Ultralytics AMP autocheck changed the requested mode. Pair is invalid; stopping.")
        seed_everything(job["seed"])
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    model.add_callback("on_pretrain_routine_end", verify_amp)
    model.train(**options)
    return Path(model.trainer.best)


def mmdet_config(job, prepared, run_dir, checkpoint):
    from mmdet_common import build_training_config, _walk_mappings

    names = json.loads((prepared / "summary.json").read_text(encoding="utf-8"))["names"]
    cfg = build_training_config(
        job["recipe"],
        official_mmdet_root(),
        prepared,
        run_dir / "train",
        names,
        job["epochs"],
        job["recipe"]["batch"],
        job["workers"],
        job["seed"],
        checkpoint,
        val_interval=1,
    )
    original_batch = cfg.get("auto_scale_lr", {}).get("base_batch_size", 16)
    original_lr = cfg.optim_wrapper.optimizer.lr
    cfg.optim_wrapper.optimizer.lr = original_lr * job["recipe"]["batch"] / original_batch
    cfg.optim_wrapper.type = "AmpOptimWrapper" if job["amp"] else "OptimWrapper"
    cfg.optim_wrapper.pop("loss_scale", None)
    cfg.optim_wrapper.pop("dtype", None)
    if job["amp"]:
        cfg.optim_wrapper.loss_scale = "dynamic"
        cfg.optim_wrapper.dtype = "float16"
    # Declared citrus adaptation: fixed 640 resize, no crop/filter that removes tiny instances.
    train_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(type="Resize", scale=(640, 640), keep_ratio=True),
        dict(type="RandomFlip", prob=0.5),
        dict(type="PackDetInputs"),
    ]
    val_pipeline = [
        dict(type="LoadImageFromFile"),
        dict(type="Resize", scale=(640, 640), keep_ratio=True),
        dict(type="LoadAnnotations", with_bbox=True, with_mask=True),
        dict(type="PackDetInputs", meta_keys=("img_id", "img_path", "ori_shape", "img_shape", "scale_factor")),
    ]
    for name, pipeline in (
        ("train_dataloader", train_pipeline),
        ("val_dataloader", val_pipeline),
        ("test_dataloader", val_pipeline),
    ):
        for mapping in _walk_mappings(cfg[name].dataset):
            if mapping.get("type") == "CocoDataset":
                mapping["pipeline"] = pipeline
    cfg.custom_hooks = [hook for hook in cfg.get("custom_hooks", []) if hook["type"] != "PipelineSwitchHook"]
    cfg.train_cfg.pop("dynamic_intervals", None)
    if "batch_augments" in cfg.model.data_preprocessor:
        cfg.model.data_preprocessor.batch_augments = None
    for mapping in _walk_mappings(cfg.model):
        if mapping.get("type") == "SyncBN":
            mapping["type"] = "BN"
        # Low threshold for subsequent common AP evaluation; no architecture changes.
        if "score_thr" in mapping:
            mapping["score_thr"] = 0.001
        if "max_per_img" in mapping:
            mapping["max_per_img"] = 300
    epochs = job["epochs"]
    if job["model"] == "rtmdet_ins_tiny":
        cfg.param_scheduler = [
            dict(
                type="CosineAnnealingLR",
                T_max=epochs,
                by_epoch=True,
                begin=0,
                end=epochs,
                eta_min=cfg.optim_wrapper.optimizer.lr * 0.05,
            )
        ]
    else:
        milestones = sorted({int(epochs * ratio) for ratio in (2 / 3, 8 / 9)} - {0, epochs})
        cfg.param_scheduler = [
            dict(type="MultiStepLR", by_epoch=True, begin=0, end=epochs, milestones=milestones, gamma=0.1)
        ]
    cfg.env_cfg.cudnn_benchmark = False
    cfg.launcher = "none"
    cfg.default_hooks.logger.interval = 20
    cfg.visualizer.vis_backends = [dict(type="LocalVisBackend")]
    return cfg


def train_mmdet(job, prepared, run_dir):
    import torch
    from mmengine.runner import Runner

    url = model_zoo_weight(official_mmdet_root(), job["recipe"]["config"])
    checkpoint = Path.cwd() / Path(url).name
    if not checkpoint.exists():
        torch.hub.download_url_to_file(url, str(checkpoint))
    save_json(run_dir / "initialization.json", dict(url=url, weights=str(checkpoint), sha256=sha256(checkpoint)))
    cfg = mmdet_config(job, prepared, run_dir, checkpoint)
    cfg.dump(str(run_dir / "effective_config.py"))
    seed_everything(job["seed"])
    runner = Runner.from_cfg(cfg)
    # The wrapper is lazily built in train(); inspect via a hook after train initialization.
    from mmengine.hooks import Hook

    class VerifyAMP(Hook):
        def before_train(self, runner):
            from mmengine.optim import AmpOptimWrapper

            actual = isinstance(runner.optim_wrapper, AmpOptimWrapper)
            save_json(
                run_dir / "amp_actual.json",
                dict(
                    requested=job["amp"],
                    actual=actual,
                    wrapper=type(runner.optim_wrapper).__name__,
                    dtype="float16" if actual else "float32",
                ),
            )
            if actual != job["amp"]:
                raise RuntimeError("MMDetection optimizer wrapper does not match requested AMP")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

    runner.register_hook(VerifyAMP(), priority="VERY_HIGH")
    runner.train()
    candidates = list((run_dir / "train").glob("best_coco_segm_mAP*.pth"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one mask-mAP best checkpoint, found {candidates}")
    return candidates[0]


def train_rfdetr(job, prepared, run_dir):
    from rfdetr import RFDETRSegNano

    seed_everything(job["seed"])
    model = RFDETRSegNano(resolution=job["recipe"]["imgsz"], positional_encoding_size=26, device="cuda", amp=job["amp"])
    # Model construction loads COCO weights; then train() reinitializes the task head using this same seed.
    weights = Path(model.model_config.pretrain_weights).resolve()
    save_json(run_dir / "initialization.json", dict(weights=str(weights), sha256=sha256(weights)))
    seed_everything(job["seed"])
    actual = bool(model.model_config.amp)
    if actual != job["amp"]:
        raise RuntimeError("RF-DETR model config did not accept the AMP flag")
    save_json(
        run_dir / "amp_actual.json",
        dict(
            requested=job["amp"],
            actual=actual,
            dtype="bfloat16" if actual else "float32",
            source="RFDETRSegNano.model_config.amp -> train_from_config -> model.train(args.amp)",
        ),
    )
    options = dict(
        dataset_dir=str(prepared / "rfdetr"),
        output_dir=str(run_dir / "train"),
        epochs=job["epochs"],
        batch_size=job["recipe"]["batch"],
        grad_accum_steps=job["recipe"]["grad_accum_steps"],
        num_workers=job["workers"],
        seed=job["seed"],
        lr=1e-4,
        lr_encoder=1.5e-4,
        weight_decay=1e-4,
        lr_drop=max(1, int(job["epochs"] * 0.8)),
        checkpoint_interval=10,
        early_stopping=False,
        use_ema=True,
        multi_scale=False,
        expanded_scales=False,
        run_test=False,
        tensorboard=False,
        wandb=False,
        eval_max_dets=100,
    )
    save_json(run_dir / "effective_train.json", dict(model=model.model_config.model_dump(), train=options))
    model.train(**options)
    best = run_dir / "train/checkpoint_best_total.pth"
    if not best.is_file():
        raise FileNotFoundError(f"RF-DETR did not produce its best checkpoint: {best}")
    return best


def evaluate(job, prepared, run_dir, checkpoint):
    """Re-evaluate selected checkpoint on ORIGINAL validation images, all in FP32 and common COCO metrics."""
    import numpy as np
    import torch
    from PIL import Image
    from coco_utils import evaluate_predictions, prediction_from_mask, save_predictions

    seed_everything(job["seed"])
    family = job["recipe"]["family"]
    if family == "yolo":
        from ultralytics import YOLO

        model = YOLO(str(checkpoint))
    elif family == "mmdet":
        from mmdet.apis import init_detector, inference_detector

        model = init_detector(str(run_dir / "effective_config.py"), str(checkpoint), device="cuda:0")
    else:
        from rfdetr import RFDETRSegNano

        nc = len(json.loads((prepared / "summary.json").read_text(encoding="utf-8"))["names"])
        model = RFDETRSegNano(
            pretrain_weights=str(checkpoint),
            resolution=624,
            positional_encoding_size=26,
            num_classes=nc,
            device="cuda",
            amp=False,
        )
    annotation = prepared / "coco/annotations/instances_val.json"
    records = json.loads(annotation.read_text(encoding="utf-8"))["images"]
    nc = len(json.loads((prepared / "summary.json").read_text(encoding="utf-8"))["names"])
    predictions = []
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            path = prepared / "coco/images/val" / record["file_name"]
            if family == "yolo":
                result = model.predict(
                    str(path),
                    imgsz=640,
                    conf=0.001,
                    iou=0.7,
                    max_det=300,
                    retina_masks=True,
                    half=False,
                    device="",
                    verbose=False,
                )[0]
                if result.masks is None:
                    continue
                masks = result.masks.data.cpu().numpy() > 0.5
                scores, labels = result.boxes.conf.cpu().numpy(), result.boxes.cls.cpu().numpy().astype(int) + 1
            elif family == "mmdet":
                result = inference_detector(model, str(path)).pred_instances
                masks = result.masks.cpu().numpy() > 0.5
                scores, labels = result.scores.cpu().numpy(), result.labels.cpu().numpy() + 1
            else:
                with Image.open(path) as image:
                    result = model.predict(image.convert("RGB"), threshold=0.001)
                if result.mask is None:
                    continue
                masks, scores, labels = result.mask, result.confidence, result.class_id + 1
                # RF-DETR's derived view is 0-based. Convert exactly ONCE to canonical COCO 1..N.
            for mask, score, label in zip(masks, scores, labels):
                if score < 0.001:
                    continue
                if int(label) not in range(1, nc + 1):
                    raise RuntimeError(f"Unexpected category ID {label}; refusing silently incorrect evaluation")
                if np.shape(mask) != (record["height"], record["width"]):
                    raise RuntimeError(f"Prediction mask is not in original-image coordinates: {np.shape(mask)}")
                predictions.append(prediction_from_mask(record["id"], label, score, np.asarray(mask) > 0.5))
            if index % 25 == 0:
                print(f"Common FP32 mask evaluation {index}/{len(records)}", flush=True)
    save_predictions(run_dir / "val_predictions.coco.json", predictions)
    metrics = evaluate_predictions(annotation, predictions, run_dir / "val_common.json", evaluate_bbox=False)
    metrics.update(
        evaluation="original-val / COCO mask AP maxDets=100 / FP32",
        checkpoint=str(checkpoint),
        params=sum(
            p.numel()
            for p in (
                model.model.model if family == "rfdetr" else model.model if family == "yolo" else model
            ).parameters()
        ),
    )
    save_json(run_dir / "val_common.json", metrics)
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", nargs="+")
    parser.add_argument("--cpu-check", action="store_true")
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--prepared", type=Path)
    parser.add_argument("--job", type=Path)
    args = parser.parse_args()
    if args.check:
        preflight(args.check, cpu=args.cpu_check)
        return
    if args.prepare:
        from prepare import prepare

        prepare(args.prepare, args.prepared)
        return
    if not args.job:
        parser.error("Expected --check, --prepare or --job")
    job = json.loads(args.job.read_text(encoding="utf-8"))
    preflight([job["recipe"]["family"]])
    run_dir = Path(job["run_dir"])
    trained = run_dir / "trained.json"
    if trained.is_file() and not (run_dir / "complete.json").exists():
        previous = json.loads((run_dir / "job.json").read_text(encoding="utf-8"))
        if previous != job:
            raise RuntimeError("Cannot resume evaluation with a changed job")
        checkpoint = Path(json.loads(trained.read_text(encoding="utf-8"))["checkpoint"])
        metrics = evaluate(job, Path(job["prepared"]), run_dir, checkpoint)
        save_json(run_dir / "complete.json", dict(job=job, metrics=metrics))
        return
    run_dir.mkdir(parents=True, exist_ok=False)
    save_json(run_dir / "job.json", job)
    save_json(run_dir / "environment.json", snapshot())
    prepared = Path(job["prepared"])
    save_json(run_dir / "dataset.json", json.loads((prepared / "summary.json").read_text(encoding="utf-8")))
    functions = {"yolo": train_yolo, "mmdet": train_mmdet, "rfdetr": train_rfdetr}
    checkpoint = functions[job["recipe"]["family"]](job, prepared, run_dir)
    save_json(run_dir / "trained.json", dict(checkpoint=str(checkpoint)))
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()
    metrics = evaluate(job, prepared, run_dir, checkpoint)
    save_json(run_dir / "complete.json", dict(job=job, metrics=metrics))


if __name__ == "__main__":
    main()
