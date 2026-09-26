"""One isolated foreground job. Run via RUN_CITRUS_BASELINES_AMP.py, not an editable Ultralytics install."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import subprocess
import sys
from pathlib import Path

# -I suppresses PYTHONPATH and the repository root, preventing accidental use of custom historical modules.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from common import save_json, seed_everything, snapshot, state_sha256  # noqa: E402
from registry import MODELS, YOLO_TRAIN  # noqa: E402


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
        if package == "rfdetr" and actual == "1.4.0" and version == "1.4.0.post0":
            raise RuntimeError(
                "Installed rfdetr==1.4.0 is the yanked release whose training fails after startup. "
                "Do not bypass this check. From the project root run: "
                f"{sys.executable} REPAIR_CITRUS_MODERN_ENV.py --python {sys.executable}"
            )
        raise RuntimeError(f"Expected {package}=={version}, found {actual}. Use the new baseline environment.")


def preflight(families, cpu=False, model_name=None, training_smoke=True):
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
        require_version("rfdetr", "1.4.0.post0")
        from rfdetr import RFDETRSegNano  # noqa: F401
        from rfdetr.config import RFDETRSegNanoConfig

        assert RFDETRSegNanoConfig(amp=False).amp is False
        assert RFDETRSegNanoConfig(amp=True).amp is True
        if not cpu and not torch.cuda.is_bf16_supported():
            raise RuntimeError("RF-DETR 1.4 native AMP requires BF16-capable CUDA hardware; no silent FP16 fallback")
    if "mmdet" in families:
        require_version("mmdet", "3.3.0")
        require_version("mmcv", "2.1.0")
        from environment_check import check_mmdet_environment

        print("MMDetection runtime environment:", check_mmdet_environment(), flush=True)
        from mmcv.ops import nms
        from mmengine.config import Config
        from mmengine.optim import build_optim_wrapper
        from mmdet.registry import MODELS as MMDET_MODELS
        from mmdet.utils import register_all_modules
        from mmdet_common import (audit_optimizer_parameters, make_optim_wrapper_config,
                                 configure_mask_preprocessor, remove_pretraining, set_num_classes, _walk_mappings)
        from mmdet_smoke import check_mask_padding, check_training_step, check_prediction_step, check_large_mask_decode
        from mmdet_memory import install_memory_safe_rtmdet

        register_all_modules(init_default_scope=True)
        print("MASK MEMORY ADAPTER:", install_memory_safe_rtmdet(), flush=True)
        if not cpu and training_smoke:
            print("LARGE MASK DECODE PREFLIGHT OK:", check_large_mask_decode(), flush=True)
        root = official_mmdet_root()
        for checked_name, recipe in MODELS.items():
            if recipe["family"] == "mmdet" and (model_name is None or checked_name == model_name):
                cfg = Config.fromfile(str(root / recipe["config"]))
                remove_pretraining(cfg.model)
                set_num_classes(cfg.model, 1)
                configure_mask_preprocessor(cfg.model)
                for mapping in _walk_mappings(cfg.model):
                    if mapping.get("type") == "SyncBN":
                        mapping["type"] = "BN"
                model = MMDET_MODELS.build(cfg.model)
                wrapper = build_optim_wrapper(model, make_optim_wrapper_config(False))
                print("OPTIMIZER PREFLIGHT OK:", checked_name, audit_optimizer_parameters(model, wrapper), flush=True)
                print("MASK PADDING PREFLIGHT OK:", checked_name,
                      check_mask_padding(model.data_preprocessor), flush=True)
                if not cpu and training_smoke:
                    model.init_weights()
                    model.cuda()
                    # Test the actual loss/backward path, not just import/build.
                    for amp in (False, True):
                        print("TRAIN STEP PREFLIGHT OK:", checked_name, "AMP=", int(amp),
                              check_training_step(model, amp), flush=True)
                    print("PREDICTION PREFLIGHT OK:", checked_name, check_prediction_step(model), flush=True)
                del wrapper, model
                if not cpu:
                    torch.cuda.empty_cache()
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
    seed_everything(job["seed"])
    model = YOLO(job["recipe"]["yaml"], task="segment")
    options = dict(YOLO_TRAIN)
    options.update(
        data=str(prepared / "yolo/data.yaml"),
        epochs=job["epochs"],
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
        save_json(run_dir / "initialization.json", dict(
            mode="scratch", weights=None, seed=job["seed"], sha256=state_sha256(trainer.model),
            stage="after trainer initialization, before first optimizer step",
        ))
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
    from mmdet_common import configure_mask_preprocessor, make_optim_wrapper_config, remove_pretraining

    remove_pretraining(cfg.model)
    cfg.load_from = None
    cfg.optim_wrapper = make_optim_wrapper_config(job["amp"])
    configure_mask_preprocessor(cfg.model)
    # Full-resolution instance masks can be large even after float decoding is
    # chunked. Do not hold five high-resolution images' masks simultaneously.
    cfg.val_dataloader.batch_size = 1
    cfg.test_dataloader.batch_size = 1
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
    cfg.custom_hooks.append(dict(type="EarlyStoppingHook", monitor="coco/segm_mAP", rule="greater",
                                 patience=100, min_delta=0.0, strict=True))
    cfg.train_cfg.pop("dynamic_intervals", None)
    for mapping in _walk_mappings(cfg.model):
        if mapping.get("type") == "SyncBN":
            mapping["type"] = "BN"
        # Low threshold for subsequent common AP evaluation; no architecture changes.
        if "score_thr" in mapping:
            mapping["score_thr"] = 0.001
        if "max_per_img" in mapping:
            mapping["max_per_img"] = 300
    epochs = job["epochs"]
    warmup = min(3, epochs)
    cfg.param_scheduler = [dict(type="LinearLR", start_factor=0.001, end_factor=1.0,
                               begin=0, end=warmup, by_epoch=True, convert_to_iter_based=True)]
    if epochs > warmup:
        cfg.param_scheduler.append(dict(type="LinearLR", start_factor=1.0, end_factor=0.01,
                                        begin=warmup, end=epochs, by_epoch=True))
    cfg.env_cfg.cudnn_benchmark = False
    cfg.launcher = "none"
    cfg.default_hooks.logger.interval = 20
    cfg.visualizer.vis_backends = [dict(type="LocalVisBackend")]
    return cfg


def train_mmdet(job, prepared, run_dir):
    import torch
    from mmengine.runner import Runner
    from mmdet_memory import install_memory_safe_rtmdet

    save_json(run_dir / "inference_memory.json", install_memory_safe_rtmdet())
    cfg = mmdet_config(job, prepared, run_dir, None)
    cfg.dump(str(run_dir / "effective_config.py"))
    seed_everything(job["seed"])
    runner = Runner.from_cfg(cfg)
    # The wrapper is lazily built in train(); inspect via a hook after train initialization.
    from mmengine.hooks import Hook

    class VerifyAMP(Hook):
        def __init__(self):
            self.scale_decreases = 0
            self.checked_iterations = 0

        def before_train(self, runner):
            from mmengine.optim import AmpOptimWrapper
            from mmdet_common import audit_optimizer_parameters

            actual = isinstance(runner.optim_wrapper, AmpOptimWrapper)
            save_json(run_dir / "optimizer_parameters.json", audit_optimizer_parameters(runner.model, runner.optim_wrapper))
            save_json(run_dir / "initialization.json", dict(
                mode="scratch", weights=None, seed=job["seed"], sha256=state_sha256(runner.model),
                stage="after init_weights, before first optimizer step",
            ))
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

        def before_train_iter(self, runner, batch_idx, data_batch=None):
            scaler = getattr(runner.optim_wrapper, "loss_scaler", None)
            self.scale_before = scaler.get_scale() if scaler is not None else None

        def after_train_iter(self, runner, batch_idx, data_batch=None, outputs=None):
            scaler = getattr(runner.optim_wrapper, "loss_scaler", None)
            if scaler is not None:
                self.checked_iterations += 1
                self.scale_decreases += int(scaler.get_scale() < self.scale_before)

        def after_train_epoch(self, runner):
            scaler = getattr(runner.optim_wrapper, "loss_scaler", None)
            save_json(run_dir / "amp_runtime.json", dict(
                checked_iterations=self.checked_iterations, scale_decreases=self.scale_decreases,
                current_scale=scaler.get_scale() if scaler is not None else None,
                note="Scale decreases indicate AMP overflow/backoff; not a claim that every update succeeded.",
            ))

    runner.register_hook(VerifyAMP(), priority="VERY_HIGH")
    runner.train()
    candidates = list((run_dir / "train").glob("best_coco_segm_mAP*.pth"))
    if len(candidates) != 1:
        raise RuntimeError(f"Expected one mask-mAP best checkpoint, found {candidates}")
    return candidates[0]


def train_rfdetr(job, prepared, run_dir):
    from rfdetr import RFDETRSegNano

    seed_everything(job["seed"])
    nc = len(json.loads((prepared / "summary.json").read_text(encoding="utf-8"))["names"])
    model = RFDETRSegNano(pretrain_weights=None, num_classes=nc, resolution=job["recipe"]["imgsz"],
                         positional_encoding_size=26, device="cuda", amp=job["amp"])
    # Pinned 1.4.0.post0: patch_size=12 disables DINOv2 weight loading in DinoV2.__init__.
    # force_no_pretrain is NOT forwarded by that version's build_backbone, so do not rely on it.
    if model.model_config.pretrain_weights is not None or model.model_config.patch_size != 12:
        raise RuntimeError("RF-DETR scratch contract requires no checkpoint and the verified Nano patch_size=12 path")
    save_json(run_dir / "initialization.json", dict(
        mode="scratch", weights=None, seed=job["seed"], sha256=state_sha256(model.model.model),
        stage="random Nano with dataset class count; train head resize preserves existing entries",
        encoder="patch_size=12 disables DINOv2 pretrained loading in pinned package",
    ))
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
        lr=0.001,
        lr_encoder=0.001,
        lr_vit_layer_decay=1.0,
        lr_component_decay=1.0,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        lr_drop=max(1, int(job["epochs"] * 0.8)),
        checkpoint_interval=10,
        early_stopping=True,
        early_stopping_patience=100,
        early_stopping_min_delta=0.0,
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
        from ultralytics.utils import ops

        decode_full = ops.process_mask_native

        def decode_chunked(protos, masks_in, bboxes, shape):
            """Bound the N x H x W FP32 peak of process_mask_native; per-detection math is unchanged."""
            step = 16
            if masks_in.shape[0] <= step:
                return decode_full(protos, masks_in, bboxes, shape)
            return torch.cat(
                [
                    decode_full(protos, masks_in[start : start + step], bboxes[start : start + step], shape)
                    for start in range(0, masks_in.shape[0], step)
                ]
            )

        ops.process_mask_native = decode_chunked
        model = YOLO(str(checkpoint))
    elif family == "mmdet":
        from mmdet.apis import init_detector, inference_detector
        from mmdet_memory import install_memory_safe_rtmdet

        install_memory_safe_rtmdet()
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
                masks, scores, labels = result.mask, result.confidence, result.class_id
                # post0 uses canonical foreground IDs 1..N; output 0 is unused, never remap it to fruit.
                keep = labels != 0
                masks, scores, labels = masks[keep], scores[keep], labels[keep]
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
    # Full synthetic training was checked once at batch startup. Per-job checks
    # retain environment/config/padding checks without repeating all six steps.
    preflight([job["recipe"]["family"]], model_name=job["model"], training_smoke=False)
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
    # Evaluate in a fresh interpreter: training leaves multi-GiB CUDA allocations reachable through
    # trainer internals, and original-resolution mask decode needs nearly the whole card. The child
    # hits the trained.json resume path above and writes complete.json itself.
    code = subprocess.call([sys.executable, "-I", "-u", str(Path(__file__).resolve()), "--job", str(args.job)])
    raise SystemExit(code)


if __name__ == "__main__":
    main()
