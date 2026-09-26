"""MMDetection configuration helpers for the citrus baselines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Iterable, MutableMapping, Sequence

from common import resolve_path


def configure_mask_preprocessor(model):
    """Align image and instance-mask canvases after aspect-preserving resize."""
    preprocessor = model["data_preprocessor"]
    if preprocessor.get("type") != "DetDataPreprocessor":
        raise ValueError("Expected DetDataPreprocessor for the registered MMDetection baselines")
    # The official RTMDet pipeline pads to 640; our no-crop pipeline instead
    # pads to this batch's maximum dimensions. Both images AND masks must pad.
    preprocessor.update(pad_mask=True, mask_pad_value=0, batch_augments=None)
    preprocessor["pad_size_divisor"] = 32


def make_optim_wrapper_config(amp=False):
    """Build the fixed recipe while retaining RTMDet's shared-parameter de-duplication."""
    cfg = dict(
        type="AmpOptimWrapper" if amp else "OptimWrapper",
        optimizer=dict(type="AdamW", lr=0.001, betas=(0.937, 0.999), weight_decay=0.0005),
        # RTMDet shares convolutions across feature levels. The official config
        # also enables this; otherwise MMEngine registers one tensor repeatedly.
        paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0, bypass_duplicate=True),
        clip_grad=dict(max_norm=10.0, norm_type=2),
    )
    if amp:
        cfg.update(loss_scale="dynamic", dtype="float16")
    return cfg


def audit_optimizer_parameters(model, wrapper):
    """Ensure each trainable tensor is optimized once, with no missing or foreign tensors."""
    model_ids = {id(parameter) for parameter in model.parameters()}
    trainable = {id(parameter) for parameter in model.parameters() if parameter.requires_grad}
    entries = [parameter for group in wrapper.optimizer.param_groups for parameter in group["params"]]
    ids = [id(parameter) for parameter in entries]
    actual = set(ids)
    if len(ids) != len(actual) or trainable - actual or actual - model_ids:
        raise RuntimeError("Optimizer parameter groups contain duplicate, missing, or foreign model parameters")
    return dict(parameter_tensors=len(ids), trainable_tensors=len(trainable),
                parameter_elements=sum(parameter.numel() for parameter in entries),
                duplicate_tensors=0, missing_trainable_tensors=0)


def require_mmdet() -> Any:
    """Import MMEngine after the caller activates the MMDetection environment."""
    try:
        from mmengine.config import Config
    except ImportError as exc:
        raise RuntimeError(
            "MMDetection dependencies are unavailable. Activate citrus_mmdet and install MMDetection v3.3.0."
        ) from exc
    return Config


def official_config_path(mmdet_root: Path, relative_path: str) -> Path:
    """Resolve and validate an official MMDetection config."""
    path = resolve_path(mmdet_root) / relative_path
    if not path.is_file():
        raise FileNotFoundError(f"Official config not found: {path}. Run the setup script or clone MMDetection v3.3.0.")
    return path


def _walk_mappings(value: Any) -> Iterable[MutableMapping[str, Any]]:
    """Yield nested mutable mappings and mappings inside lists."""
    if isinstance(value, MutableMapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def set_num_classes(model: MutableMapping[str, Any], num_classes: int) -> None:
    """Update every model head that declares a class count."""
    updated = 0
    for mapping in _walk_mappings(model):
        if "num_classes" in mapping:
            mapping["num_classes"] = num_classes
            updated += 1
    if not updated:
        raise ValueError("The MMDetection model config does not contain a num_classes field.")


def remove_pretraining(model: MutableMapping[str, Any]) -> None:
    """Remove checkpoint initializers at every level, retaining normal random initializers."""
    for mapping in _walk_mappings(model):
        if "pretrained" in mapping:
            mapping["pretrained"] = None
        init = mapping.get("init_cfg")
        if isinstance(init, MutableMapping) and init.get("type") == "Pretrained":
            mapping["init_cfg"] = None
        elif isinstance(init, list):
            mapping["init_cfg"] = [x for x in init if not (
                isinstance(x, MutableMapping) and x.get("type") == "Pretrained")]
        if "frozen_stages" in mapping:
            mapping["frozen_stages"] = -1
        if "norm_eval" in mapping:
            mapping["norm_eval"] = False
        if mapping.get("type") in ("BN", "SyncBN", "GN") and "requires_grad" in mapping:
            mapping["requires_grad"] = True


def configure_dataset(
    dataset_cfg: MutableMapping[str, Any],
    annotation_path: Path,
    image_dir: Path,
    class_names: Sequence[str],
    test_mode: bool,
) -> None:
    """Rewrite nested dataset wrappers to use the prepared citrus COCO layout."""
    if "dataset" in dataset_cfg and isinstance(dataset_cfg["dataset"], MutableMapping):
        configure_dataset(dataset_cfg["dataset"], annotation_path, image_dir, class_names, test_mode)
        return
    if "datasets" in dataset_cfg:
        for child in dataset_cfg["datasets"]:
            configure_dataset(child, annotation_path, image_dir, class_names, test_mode)
        return

    dataset_cfg["type"] = "CocoDataset"
    dataset_cfg["data_root"] = ""
    dataset_cfg["ann_file"] = str(annotation_path)
    dataset_cfg["data_prefix"] = {"img": str(image_dir) + os.sep}
    dataset_cfg["metainfo"] = {"classes": tuple(class_names)}
    dataset_cfg["test_mode"] = test_mode
    if test_mode:
        dataset_cfg.pop("filter_cfg", None)
    else:
        dataset_cfg["filter_cfg"] = {"filter_empty_gt": False, "min_size": 1}


def configure_evaluator(evaluator_cfg: Any, annotation_path: Path) -> None:
    """Point one or more COCO evaluators at a citrus annotation file."""
    evaluators = evaluator_cfg if isinstance(evaluator_cfg, list) else [evaluator_cfg]
    for evaluator in evaluators:
        if isinstance(evaluator, MutableMapping):
            evaluator["ann_file"] = str(annotation_path)
            evaluator["metric"] = ["bbox", "segm"]
            evaluator["format_only"] = False


def scale_epoch_schedulers(cfg: Any, old_epochs: int, new_epochs: int) -> None:
    """Scale epoch-based scheduler boundaries when the official schedule length changes."""
    if old_epochs <= 0 or old_epochs == new_epochs:
        return
    ratio = new_epochs / old_epochs
    schedulers = cfg.param_scheduler if isinstance(cfg.param_scheduler, list) else [cfg.param_scheduler]
    for scheduler in schedulers:
        if not isinstance(scheduler, MutableMapping) or scheduler.get("by_epoch", True) is False:
            continue
        if isinstance(scheduler.get("milestones"), (list, tuple)):
            scheduler["milestones"] = [
                max(1, min(new_epochs - 1, round(float(value) * ratio))) for value in scheduler["milestones"]
            ]
        for key in ("begin", "end", "T_max"):
            value = scheduler.get(key)
            if isinstance(value, (int, float)) and value > 0:
                scheduler[key] = max(1, round(float(value) * ratio))


def build_training_config(
    baseline: Dict[str, Any],
    mmdet_root: Path,
    dataset_root: Path,
    run_dir: Path,
    class_names: Sequence[str],
    epochs: int,
    batch_size: int,
    workers: int,
    seed: int,
    checkpoint: Path | None,
    val_interval: int,
) -> Any:
    """Create a self-contained MMDetection training config."""
    Config = require_mmdet()
    config_path = official_config_path(mmdet_root, str(baseline["config"]))
    cfg = Config.fromfile(str(config_path))

    train_ann = dataset_root / "coco" / "annotations" / "instances_train.json"
    val_ann = dataset_root / "coco" / "annotations" / "instances_val.json"
    train_images = dataset_root / "coco" / "images" / "train"
    val_images = dataset_root / "coco" / "images" / "val"
    configure_dataset(cfg.train_dataloader.dataset, train_ann, train_images, class_names, test_mode=False)
    configure_dataset(cfg.val_dataloader.dataset, val_ann, val_images, class_names, test_mode=True)
    configure_dataset(cfg.test_dataloader.dataset, val_ann, val_images, class_names, test_mode=True)
    configure_evaluator(cfg.val_evaluator, val_ann)
    configure_evaluator(cfg.test_evaluator, val_ann)

    set_num_classes(cfg.model, len(class_names))
    old_epochs = int(cfg.train_cfg.get("max_epochs", epochs))
    scale_epoch_schedulers(cfg, old_epochs, epochs)
    cfg.train_cfg.max_epochs = epochs
    cfg.train_cfg.val_interval = max(1, val_interval)
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = workers
    cfg.train_dataloader.persistent_workers = workers > 0
    cfg.val_dataloader.num_workers = workers
    cfg.val_dataloader.persistent_workers = workers > 0
    cfg.test_dataloader.num_workers = workers
    cfg.test_dataloader.persistent_workers = workers > 0

    cfg.work_dir = str(run_dir)
    cfg.randomness = {"seed": seed, "deterministic": True}
    cfg.resume = False
    cfg.load_from = str(checkpoint) if checkpoint else None
    cfg.default_hooks.checkpoint.interval = max(1, val_interval)
    cfg.default_hooks.checkpoint.save_best = "coco/segm_mAP"
    cfg.default_hooks.checkpoint.rule = "greater"
    cfg.default_hooks.checkpoint.max_keep_ckpts = 3
    if "auto_scale_lr" in cfg:
        cfg.auto_scale_lr.enable = False
    return cfg
