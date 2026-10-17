"""Pinned transfer recipes; no framework imports, downloads, or training side effects.

The official-transfer track uses COCO weights and backend optimizer conventions.
Dataset, epoch budget, fixed citrus resolution, and common validation remain the
caller's explicit adaptations. Legacy scratch recipes never call these helpers.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, MutableMapping
from numbers import Integral
from pathlib import Path
from types import MappingProxyType

RFDETR_VERSION = "1.4.0.post0"
RFDETR_CONFIG_SOURCE = "https://github.com/roboflow/rf-detr/blob/1.4.0.post0/rfdetr/config.py"
MMDET_CONFIG_SOURCE = "https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs"

# Ultralytics 8.4.60 defaults. `auto` resolves the actual optimizer/LR at startup.
YOLO_NATIVE_OVERRIDES = MappingProxyType(dict(pretrained=True, optimizer="auto", lr0=0.01, dropout=0.0))

# Verified SegmentationTrainConfig/TrainConfig defaults, frozen to the pinned API.
# Fixed resolution is a declared citrus adaptation: native multi_scale=True with
# expanded_scales=True resizes each batch over the backend's patch-aligned scales.
RFDETR_NATIVE_TRAIN = MappingProxyType(dict(
    lr=1e-4,
    lr_encoder=1.5e-4,
    weight_decay=1e-4,
    lr_vit_layer_decay=0.8,
    lr_component_decay=0.7,
    warmup_epochs=0.0,
    lr_scheduler="step",
    lr_min_factor=0.0,
    use_ema=True,
    ema_decay=0.993,
    ema_tau=100,
    multi_scale=False,
    expanded_scales=False,
    do_random_resize_via_padding=False,
))

# Official one-step global batch/LR. This runner is one GPU, no accumulation.
MMDET_REFERENCES = MappingProxyType({
    "rtmdet_ins_tiny": (256, "AdamW", 0.004),
    "mask_rcnn_r50": (16, "SGD", 0.02),
    "solov2_light_r18": (16, "SGD", 0.01),
})


def is_native(job):
    """Return whether the job explicitly selects the independent transfer policy."""
    return job.get("recipe_policy") == "official_transfer"


def yolo_options(current):
    """Copy caller controls and enable checkpoint reuse with native optimization."""
    options = copy.deepcopy(dict(current))
    options.update(YOLO_NATIVE_OVERRIDES)
    return options


def yolo_checkpoint(recipe):
    """Return the matching official checkpoint name for a registered YOLO YAML."""
    path = Path(recipe["yaml"])
    if path.suffix.lower() != ".yaml":
        raise ValueError("Official YOLO transfer requires a registered .yaml architecture")
    return str(path.with_suffix(".pt"))


def rfdetr_options(current):
    """Copy caller controls and apply pinned optimizer/EMA defaults at fixed size.

    Epochs, batch size, accumulation, lr_drop, selection/stopping, logging, and
    eval_max_dets retain the explicit values selected by the caller.
    """
    options = copy.deepcopy(dict(current))
    options.update(RFDETR_NATIVE_TRAIN)
    return options


def _walk_mappings(value):
    if isinstance(value, MutableMapping):
        yield value
        for child in value.values():
            yield from _walk_mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _walk_mappings(child)


def _remove_checkpoint_initializers(model):
    """Avoid a redundant ImageNet download without altering freeze/norm settings."""
    for mapping in _walk_mappings(model):
        if "pretrained" in mapping:
            mapping["pretrained"] = None
        init = mapping.get("init_cfg")
        if isinstance(init, Mapping) and init.get("type") == "Pretrained":
            mapping["init_cfg"] = None
        elif isinstance(init, list):
            mapping["init_cfg"] = [item for item in init if not (
                isinstance(item, Mapping) and item.get("type") == "Pretrained")]


def mmdet_apply_native(cfg, job, checkpoint):
    """Adapt a resolved official MMDetection config and return LR provenance.

    The caller must skip scratch remove_pretraining/optimizer/scheduler overrides.
    Its build_training_config already scales epoch scheduler boundaries. We keep
    that scheduler and official optimizer/paramwise/clip/freeze settings, scale
    absolute LR values once for the actual single-GPU microbatch, and load full
    COCO weights without resuming the COCO optimizer. Class-head shape mismatches
    are handled and logged by MMEngine's non-strict transfer loader.
    """
    if not is_native(job):
        raise ValueError("Native MMDetection helper requires recipe_policy='official_transfer'")
    model_name = job["model"]
    if model_name not in MMDET_REFERENCES:
        raise ValueError(f"No verified official MMDetection recipe for {model_name!r}")
    if checkpoint is None or not str(checkpoint).strip() or "://" in str(checkpoint):
        raise ValueError("Native MMDetection requires an explicit local COCO checkpoint")
    batch = job["recipe"]["batch"]
    if isinstance(batch, bool) or not isinstance(batch, Integral) or batch < 1:
        raise ValueError("Native MMDetection batch must be a positive integer")
    wrapper = copy.deepcopy(dict(cfg["optim_wrapper"]))
    if wrapper.get("type", "OptimWrapper") not in ("OptimWrapper", "AmpOptimWrapper"):
        raise ValueError("Unsupported official optimizer wrapper")
    if wrapper.get("accumulative_counts", 1) != 1:
        raise ValueError("Native MMDetection LR policy requires no gradient accumulation")
    reference_batch, optimizer_type, expected_lr = MMDET_REFERENCES[model_name]
    reference_lr = float(wrapper["optimizer"]["lr"])
    if wrapper["optimizer"]["type"] != optimizer_type or not math.isclose(reference_lr, expected_lr):
        raise ValueError("Native MMDetection requires the unmodified official optimizer before LR scaling")

    reference_optimizer = copy.deepcopy(wrapper["optimizer"])
    reference_paramwise = copy.deepcopy(wrapper.get("paramwise_cfg", {}))
    scale = int(batch) / reference_batch
    wrapper["optimizer"]["lr"] = reference_lr * scale
    # Amp-only constructor kwargs must never reach OptimWrapper on AMP=0.
    for key in ("loss_scale", "dtype", "use_fsdp"):
        wrapper.pop(key, None)
    wrapper["type"] = "AmpOptimWrapper" if job["amp"] else "OptimWrapper"
    if job["amp"]:
        wrapper.update(loss_scale="dynamic", dtype="float16")

    schedulers = copy.deepcopy(cfg["param_scheduler"])
    minimums = []
    for scheduler in _walk_mappings(schedulers):
        if "eta_min" in scheduler:
            original = float(scheduler["eta_min"])
            scheduler["eta_min"] = original * scale
            minimums.append(dict(reference=original, effective=scheduler["eta_min"]))
    _remove_checkpoint_initializers(cfg["model"])
    cfg["optim_wrapper"] = wrapper
    cfg["param_scheduler"] = schedulers
    cfg["load_from"] = str(checkpoint)
    cfg["resume"] = False
    if "auto_scale_lr" in cfg:
        cfg["auto_scale_lr"]["enable"] = False
    return dict(
        policy="official_transfer",
        source=f"{MMDET_CONFIG_SOURCE}/{job['recipe']['config'].removeprefix('configs/')}",
        reference_batch=reference_batch,
        actual_batch=int(batch),
        world_size=1,
        accumulation=1,
        reference_lr=reference_lr,
        effective_lr=wrapper["optimizer"]["lr"],
        lr_scale=scale,
        reference_optimizer=reference_optimizer,
        reference_paramwise_cfg=reference_paramwise,
        cosine_minimums=minimums,
        checkpoint=str(checkpoint),
    )
