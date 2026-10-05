"""Matched FP32 GPU microbenchmark; synthetic resident inputs, not deployment FPS."""

import csv
import gc
import json
import math
import os
import platform
import statistics
import sys
import time
from copy import deepcopy
from pathlib import Path


def synthetic_batch(torch, batch_size, size, device, instances=8):
    """Create identical visible-mask workload across architectures, including small supports."""
    if size < 64 or size % 32 or batch_size < 1 or instances < 1 or instances > 16:
        raise ValueError("Use size >=64 divisible by32, positive batch, and 1..16 instances")
    generator = torch.Generator().manual_seed(42)
    images = torch.rand(batch_size, 3, size, size, generator=generator).to(device)
    masks = torch.zeros(batch_size, size // 2, size // 2, device=device)
    boxes, image_indices = [], []
    for image_index in range(batch_size):
        for index in range(instances):
            cx, cy = 0.125 + (index % 4) * 0.25, 0.125 + (index // 4) * 0.25
            width = (0.025, 0.05, 0.10, 0.18)[index % 4]
            x1, x2 = max(0, int((cx - width / 2) * size / 2)), int((cx + width / 2) * size / 2)
            y1, y2 = max(0, int((cy - width / 2) * size / 2)), int((cy + width / 2) * size / 2)
            masks[image_index, y1 : max(y1 + 1, y2), x1 : max(x1 + 1, x2)] = index + 1
            boxes.append([cx, cy, width, width])
            image_indices.append(image_index)
    return dict(
        img=images,
        masks=masks,
        bboxes=torch.tensor(boxes, device=device),
        batch_idx=torch.tensor(image_indices, device=device),
        cls=torch.zeros(batch_size * instances, 1, device=device),
    )


def measure(torch, action, device, warmup, repeats):
    """Synchronized wall timing; include only the supplied action."""
    sync = lambda: torch.cuda.synchronize(device) if device.type == "cuda" else None
    for _ in range(warmup):
        action()
    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    baseline_bytes = torch.cuda.memory_allocated(device) if device.type == "cuda" else None
    samples = []
    for _ in range(repeats):
        sync()
        start = time.perf_counter()
        action()
        sync()
        samples.append(1000 * (time.perf_counter() - start))
    ordered = sorted(samples)
    return dict(
        median_ms=statistics.median(samples),
        p90_ms=ordered[math.ceil(0.9 * len(ordered)) - 1],
        peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else None,
        baseline_allocated_mib=baseline_bytes / 2**20 if baseline_bytes is not None else None,
        incremental_peak_mib=(torch.cuda.max_memory_allocated(device) - baseline_bytes) / 2**20
        if baseline_bytes is not None
        else None,
        samples_ms=samples,
    )


def benchmark_one(torch, name, device, size, batch_size, warmup, repeats):
    from citrus_j_suite import YAML_DIR
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg

    torch.manual_seed(42)
    model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model.float().to(device)
    model.args = get_cfg(overrides={"overlap_mask": True, "mask_ratio": 2, "amp": False})
    inputs = synthetic_batch(torch, batch_size, size, device)
    image = inputs["img"][:1]
    model.eval()
    with torch.inference_mode():
        unfused = measure(torch, lambda: model(image), device, warmup, repeats)
        fused_model = deepcopy(model).fuse(verbose=False)
        reference, actual = model(image)[0], fused_model(image)[0]
        torch.testing.assert_close(actual[0], reference[0], rtol=2e-4, atol=2e-4)
        torch.testing.assert_close(actual[1], reference[1], rtol=2e-4, atol=2e-4)
        del reference, actual
        fused = measure(torch, lambda net=fused_model: net(image), device, warmup, repeats)
    del fused_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001, weight_decay=0.0005)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        loss, _ = model.loss(inputs)
        total = loss.sum()
        if not torch.isfinite(total):
            raise RuntimeError("Nonfinite benchmark loss; cannot report timing as success")
        total.backward()
        optimizer.step()

    training = measure(torch, train_step, device, warmup, repeats)
    return dict(
        model=name,
        status="ok",
        params=sum(p.numel() for p in model.parameters()),
        forward_unfused=unfused,
        forward_fused=fused,
        training_step=training,
    )


def run(output, device="0", names=None, size=640, batch_size=16, warmup=5, repeats=20, threads=4):
    """Compare selected graphs without silently reducing a failing batch or input size."""
    import torch
    from citrus_j_suite import NAMES

    names = names or [NAMES[0], NAMES[11], NAMES[17]]
    if any(name not in NAMES for name in names) or warmup < 1 or repeats < 2:
        raise ValueError("Known J models, warmup >=1 and repeats >=2 required")
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Use a new benchmark output directory: {}".format(output))
    runtime_device = torch.device("cpu" if str(device) == "cpu" else "cuda:" + str(device))
    if runtime_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; fix the selected environment or explicitly choose CPU")
    torch.set_num_threads(threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True)
    report = dict(
        python=sys.version,
        torch=torch.__version__,
        cuda=torch.version.cuda,
        platform=platform.platform(),
        device=str(runtime_device),
        visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        cpu_threads=torch.get_num_threads(),
        seed=42,
        gpu=torch.cuda.get_device_name(runtime_device) if runtime_device.type == "cuda" else None,
        input_size=size,
        training_batch=batch_size,
        inference_batch=1,
        instances_per_training_image=8,
        warmup=warmup,
        repeats=repeats,
        dtype="FP32; AMP=False; TF32=False",
        includes="resident synthetic forward; or forward + real criterion + backward + one-group AdamW step",
        excludes="I/O, caching, augmentation, NMS, mask crop, stitching, EMA, validation, transfer and export runtimes",
        limitations="Kernel microbenchmark, not whole-epoch training speed, deployment FPS or accuracy evidence",
        results=[],
    )
    for name in names:
        try:
            row = benchmark_one(torch, name, runtime_device, size, batch_size, warmup, repeats)
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            row = dict(model=name, status="out_of_memory", error=str(error))
        report["results"].append(row)
        (output / "benchmark.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(row), flush=True)
        gc.collect()
        if runtime_device.type == "cuda":
            torch.cuda.empty_cache()
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["model", "status", "params", "forward_unfused_ms", "forward_fused_ms", "training_step_ms"],
        )
        writer.writeheader()
        for row in report["results"]:
            writer.writerow(
                dict(
                    model=row["model"],
                    status=row["status"],
                    params=row.get("params"),
                    **{
                        key + "_ms": row.get(key, {}).get("median_ms")
                        for key in ("forward_unfused", "forward_fused", "training_step")
                    },
                )
            )
    return report
