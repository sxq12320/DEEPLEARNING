"""Explicit baseline identities and paired-AMP recipes (not architecture ablations)."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from numbers import Integral

PROTOCOL_ID = "legacy78_scratch_v1_20260920"
CONTROLLED_PROTOCOL_ID = "legacy78_scratch_fixedbudget_commonval_v1_20260930"
TRANSFER_PROTOCOL_ID = "official_transfer_native_citrus_commonval_v1_20261009"
COMMON_VALIDATION = dict(
    id="common_mask_coco_v1_20260930", quick_size=640, original_interval=10,
    first_and_last_original=True, dtype="float32", score_min=0.001, max_dets=100,
    pr_score=0.25, pr_iou=0.5, selection="original_mask_ap_50_95_strict_greater", early_stopping=False,
)
MODELS = {
    "yolo11n_seg": dict(family="yolo", environment="modern", yaml="yolo11n-seg.yaml", batch=16, imgsz=640),
    "yolov8n_seg": dict(family="yolo", environment="modern", yaml="yolov8n-seg.yaml", batch=16, imgsz=640),
    "yolo26n_seg": dict(family="yolo", environment="modern", yaml="yolo26n-seg.yaml", batch=16, imgsz=640),
    "rtmdet_ins_tiny": dict(
        family="mmdet",
        environment="mmdet",
        batch=8,
        imgsz=640,
        config="configs/rtmdet/rtmdet-ins_tiny_8xb32-300e_coco.py",
    ),
    "mask_rcnn_r50": dict(
        family="mmdet", environment="mmdet", batch=2, imgsz=640, config="configs/mask_rcnn/mask-rcnn_r50_fpn_1x_coco.py"
    ),
    "solov2_light_r18": dict(
        family="mmdet",
        environment="mmdet",
        batch=2,
        imgsz=640,
        config="configs/solov2/solov2-light_r18_fpn_ms-3x_coco.py",
    ),
    # Real Nano, NOT RFDETRSegPreview. Native Nano is 312; 624 is a declared high-resolution adaptation.
    "rfdetr_seg_nano": dict(family="rfdetr", environment="modern", batch=2, imgsz=624, grad_accum_steps=8),
}

SUITES = {
    "all": list(MODELS),
    "non_yolo": [name for name, cfg in MODELS.items() if cfg["family"] != "yolo"],
    "yolo": list(MODELS)[:3],
    "mmdet": [name for name, cfg in MODELS.items() if cfg["family"] == "mmdet"],
    "rfdetr": ["rfdetr_seg_nano"],
    "anchor": ["yolo11n_seg", "rtmdet_ins_tiny"],
}

# Explicit exception to the old fixed-AMP protocol: amp is the experimental factor, not a tuned parameter.
YOLO_TRAIN = dict(
    imgsz=640,
    optimizer="AdamW",
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_momentum=0.8,
    warmup_bias_lr=0.1,
    box=7.5,
    cls=0.5,
    dfl=1.5,
    nbs=64,
    close_mosaic=10,
    overlap_mask=True,
    mask_ratio=4,
    dropout=0.1,
    patience=100,
    deterministic=True,
    rect=False,
    cos_lr=False,
    multi_scale=0.0,
    fraction=1.0,
    freeze=None,
    compile=False,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    degrees=0.0,
    translate=0.1,
    scale=0.5,
    shear=0.0,
    perspective=0.0,
    flipud=0.0,
    fliplr=0.5,
    bgr=0.0,
    mosaic=1.0,
    mixup=0.0,
    cutmix=0.0,
    copy_paste=0.0,
    copy_paste_mode="flip",
    cache=True,
    # Boolean True in the historical YAML run did not load a checkpoint. Make scratch explicit.
    pretrained=False,
    max_det=300,
)


def _integer(value, name, minimum, maximum=None):
    """Validate without truncating floats, parsing strings, or accepting bool as int."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer, not a boolean, float, or string")
    value = int(value)
    if value < minimum or (maximum is not None and value > maximum):
        bounds = f"{minimum}..{maximum}" if maximum is not None else f">= {minimum}"
        raise ValueError(f"{name} must be {bounds}")
    return value


def _ordered_values(values, name):
    """Reject scalar/string/unordered input instead of silently changing run order."""
    if isinstance(values, (str, bytes, bytearray)) or not isinstance(values, Sequence) or not values:
        raise ValueError(f"{name} must be a nonempty ordered sequence")
    return list(values)


def make_queue(suite, seeds, epochs, workers, batches=None, amp_modes=(1, 0), validation=None,
               recipe_policy="legacy78_scratch"):
    """Keep pairs adjacent; reject ambiguous parameters before constructing any jobs."""
    if not isinstance(suite, str) or suite not in SUITES:
        raise ValueError(f"Invalid suite: {suite!r}. Available: {', '.join(SUITES)}")
    epochs = _integer(epochs, "epochs", 1)
    workers = _integer(workers, "workers", 0)
    seeds = [_integer(seed, "seed", 0, 2**32 - 1) for seed in _ordered_values(seeds, "seeds")]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    amp_modes = _ordered_values(amp_modes, "AMP_MODES")
    if any(not isinstance(value, Integral) or value not in (0, 1) for value in amp_modes):
        raise ValueError("AMP_MODES must be [1], [0], [1, 0] or [0, 1] (integer or boolean values only)")
    amp_modes = [int(value) for value in amp_modes]
    if len(set(amp_modes)) != len(amp_modes):
        raise ValueError("AMP_MODES must be [1], [0], [1, 0] or [0, 1]")
    if validation is not None and validation != COMMON_VALIDATION:
        raise ValueError("Unsupported common validation protocol")
    if recipe_policy not in ("legacy78_scratch", "official_transfer"):
        raise ValueError("recipe_policy must be legacy78_scratch or official_transfer")
    if recipe_policy == "official_transfer" and validation is None:
        raise ValueError("The paper transfer track requires COMMON_VALIDATION")
    if batches is None:
        batches = {}
    if not isinstance(batches, Mapping):
        raise ValueError("batches must be a mapping of known model IDs to positive integer batch sizes")
    unknown = [model for model in batches if not isinstance(model, str) or model not in MODELS]
    if unknown:
        raise ValueError(f"Unknown batch override(s): {unknown!r}")
    batches = {model: _integer(batch, f"batch for {model}", 1) for model, batch in batches.items()}
    queue = []
    for seed_index, seed in enumerate(seeds):
        for model_index, model in enumerate(SUITES[suite]):
            order = tuple(amp_modes)
            if (seed_index + model_index) % 2:
                order = order[::-1]
            for amp in order:
                recipe = dict(MODELS[model])
                recipe["batch"] = batches.get(model, recipe["batch"])
                queue.append(
                    dict(
                        model=model,
                        protocol=CONTROLLED_PROTOCOL_ID if validation is not None else PROTOCOL_ID,
                        initialization="scratch",
                        seed=int(seed),
                        amp=bool(amp),
                        epochs=epochs,
                        workers=workers,
                        recipe=recipe,
                        name=f"{model}_amp{amp}_seed{seed}_{epochs}ep",
                    )
                )
                if validation is not None:
                    queue[-1]["validation"] = copy.deepcopy(validation)
                if recipe_policy == "official_transfer":
                    queue[-1].update(recipe_policy=recipe_policy, protocol=TRANSFER_PROTOCOL_ID,
                                     initialization="official_coco_pretrained",
                                     name="transfer_" + queue[-1]["name"])
    return queue
