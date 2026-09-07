"""Interleaved V7R compute benchmark. Synthetic timings are not dataset accuracy or deployment FPS."""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from citrus_protocol import fixed_train_args
from citrus_sage_v7_suite import SUITES, YAML_DIR
from scripts.benchmark_sage_v7 import make_batch
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--only", default=",".join(SUITES["refusion"]))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    names = args.only.split(",")
    if set(names) - set(SUITES["refusion"]) or len(names) != len(set(names)) or not names:
        raise ValueError("Choose distinct refusion models")
    if min(args.batch, args.steps, args.threads) < 1:
        raise ValueError("Positive batch/steps/threads required")
    torch.set_num_threads(args.threads)
    device = torch.device("cuda:" + args.device if args.device.isdigit() else args.device)
    fixed = fixed_train_args()
    batch = make_batch(args.batch, 640, device)
    models, costs, optimizers = {}, {}, {}
    for name in names:
        torch.manual_seed(42)
        model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
        costs[name] = {"parameters": sum(p.numel() for p in model.parameters()), "gflops640": get_flops(model, 640)}
        model = model.to(device)
        model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **fixed})
        models[name] = model
        # Deliberately simple single-group AdamW for compute timing, NOT a formal
        # training optimizer configuration (no Ultralytics parameter groups).
        optimizers[name] = torch.optim.AdamW(model.parameters(), lr=fixed["lr0"])
    raw = {n: {p: [] for p in ("forward", "train_step")} for n in names}
    rng = random.Random(20260906)

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for phase in ("forward", "train_step"):
        for model in models.values():
            model.train(phase == "train_step")
        for step in range(args.steps + 3):
            order = list(names)
            rng.shuffle(order)
            for name in order:
                model = models[name]
                if device.type == "cuda":
                    torch.cuda.reset_peak_memory_stats(device)
                sync()
                start = time.perf_counter()
                if phase == "forward":
                    with torch.inference_mode():
                        output = model(batch["img"])
                    del output
                else:
                    optimizers[name].zero_grad(set_to_none=True)
                    loss, items = model.loss(batch)
                    loss.sum().backward()
                    optimizers[name].step()
                    del loss, items
                sync()
                elapsed = 1000 * (time.perf_counter() - start)
                if step >= 3:
                    raw[name][phase].append(elapsed)
            if step % 5 == 0:
                print(f"{phase}: {step}/{args.steps + 2}", flush=True)
    summary = {n: {**costs[n], **{p + "_ms": statistics.median(v) for p, v in raw[n].items()}} for n in names}
    payload = dict(
        device=str(device),
        torch=torch.__version__,
        batch=args.batch,
        imgsz=640,
        amp=False,
        threads=args.threads,
        summary=summary,
        raw_ms=raw,
        note="Interleaved 3 warmup + measured steps; synthetic 12 instances/image. Train step includes simple "
        "AdamW, loss and backward; excludes loader, real augmentation, NMS, validation, callbacks. "
        "Not formal optimizer groups or dataset accuracy. GFLOPs estimator omits some functional operations.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
