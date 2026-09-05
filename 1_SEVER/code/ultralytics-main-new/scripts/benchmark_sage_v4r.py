"""Same-device synthetic timing for reconstructed v4; not dataset accuracy or end-to-end FPS."""

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

from citrus_sage_v4r_suite import NAMES, YAML_DIR
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


def make_batch(batch_size, imgsz, device):
    """Twelve visible instances including concavities; a stress fixture, not citrus statistics."""
    size = imgsz // 4
    ids = torch.zeros(batch_size, size, size, device=device)
    boxes = []
    for row in range(3):
        for col in range(4):
            number = row * 4 + col + 1
            y0, x0 = (row + 1) * size // 5, col * size // 4 + 1
            y1, x1 = y0 + max(3, size // 8), min(size, x0 + max(3, size // 4 - 2))
            ids[:, y0:y1, x0:x1] = number
            if number % 3 == 0:
                ids[:, (y0 + y1) // 2 : y1 - 1, (x0 + x1) // 2 : x1] = 0
            boxes.append([(x0 + x1) / 2 / size, (y0 + y1) / 2 / size, (x1 - x0) / size, (y1 - y0) / size])
    return {
        "img": torch.rand(batch_size, 3, imgsz, imgsz, device=device),
        "masks": ids,
        "cls": torch.zeros(batch_size * 12, 1, device=device),
        "batch_idx": torch.arange(batch_size, device=device).repeat_interleave(12),
        "bboxes": torch.tensor(boxes, device=device).repeat(batch_size, 1),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--only", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batch, args.steps, args.threads) < 1 or args.imgsz < 128 or args.imgsz % 32:
        raise ValueError("Positive counts and imgsz>=128 divisible by32 required")
    names = args.only.split(",") if args.only else list(NAMES)
    if set(names) - set(NAMES):
        raise ValueError(f"Choose exact model names from {NAMES}")
    random.Random(20260903).shuffle(names)
    torch.set_num_threads(args.threads)
    device = torch.device(f"cuda:{args.device}" if args.device.isdigit() else args.device)
    torch.manual_seed(42)
    batch = make_batch(args.batch, args.imgsz, device)

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for name in names:
        torch.manual_seed(42)
        model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False).to(device)
        model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        row = {"model": name, "params": sum(p.numel() for p in model.parameters()), "gflops640": get_flops(model, 640)}
        for phase in ("forward", "forward_loss_backward"):
            model.train(phase != "forward")
            durations = []
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            for step in range(args.steps + 5):
                synchronize()
                start = time.perf_counter()
                if phase == "forward":
                    with torch.inference_mode():
                        model(batch["img"])
                else:
                    model.zero_grad(set_to_none=True)
                    total, _ = model.loss(batch)
                    total.sum().backward()
                synchronize()
                if step >= 5:
                    durations.append(1000 * (time.perf_counter() - start))
            row[f"{phase}_median_ms"] = statistics.median(durations)
            row[f"{phase}_p90_ms"] = sorted(durations)[min(len(durations) - 1, int(0.9 * len(durations)))]
            row[f"{phase}_peak_allocated_mb"] = (
                torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else None
            )
            if phase == "forward_loss_backward":
                del total  # Do not retain the previous model through its last loss graph.
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result = {
        "device": str(device),
        "torch": torch.__version__,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "threads": args.threads,
        "steps": args.steps,
        "amp": False,
        "instances_per_image": 12,
        "excludes": ["data_loading", "optimizer_step", "NMS", "mask_postprocessing", "validation"],
        "warning": "Synthetic microbenchmark, not final training speed, end-to-end latency or accuracy.",
        "results": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
