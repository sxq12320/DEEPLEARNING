"""Explicit baseline identities and paired-AMP recipes (not architecture ablations)."""

from __future__ import annotations

PROTOCOL_ID = "legacy78_scratch_v1_20260920"
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


def make_queue(suite, seeds, epochs, workers, batches=None, amp_modes=(1, 0)):
    """Keep pairs adjacent, alternate AMP order by model/seed to reduce order confounding."""
    if suite not in SUITES or not seeds or epochs < 1 or workers < 0:
        raise ValueError("Invalid suite, seeds, epochs or workers")
    if len(set(seeds)) != len(seeds):
        raise ValueError("Seeds must be unique")
    if not amp_modes or len(set(amp_modes)) != len(amp_modes) or any(x not in (0, 1) for x in amp_modes):
        raise ValueError("AMP_MODES must be [1], [0], [1, 0] or [0, 1]")
    batches = batches or {}
    unknown = set(batches) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown batch override(s): {sorted(unknown)}")
    queue = []
    for seed_index, seed in enumerate(seeds):
        for model_index, model in enumerate(SUITES[suite]):
            order = tuple(amp_modes)
            if (seed_index + model_index) % 2:
                order = order[::-1]
            for amp in order:
                recipe = dict(MODELS[model])
                recipe["batch"] = int(batches.get(model, recipe["batch"]))
                if recipe["batch"] < 1:
                    raise ValueError("Batch must be a positive fixed integer; no AutoBatch")
                queue.append(
                    dict(
                        model=model,
                        protocol=PROTOCOL_ID,
                        initialization="scratch",
                        seed=int(seed),
                        amp=bool(amp),
                        epochs=epochs,
                        workers=workers,
                        recipe=recipe,
                        name=f"{model}_amp{amp}_seed{seed}_{epochs}ep",
                    )
                )
    return queue
