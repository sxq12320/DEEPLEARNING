"""Synthetic same-device forward/backward timing; not end-to-end training speed or accuracy."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
from ultralytics.utils.torch_utils import get_flops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.batch, args.steps, args.threads) < 1 or args.imgsz % 32:
        raise ValueError("Positive batch/steps/threads and an image size divisible by 32 are required")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    size = args.imgsz // 4
    masks = torch.zeros(args.batch, size, size, device=device)
    masks[:, size // 4 : size // 2, size // 4 : size // 2] = 1
    batch = {
        "img": torch.rand(args.batch, 3, args.imgsz, args.imgsz, device=device),
        "masks": masks,
        "cls": torch.zeros(args.batch, 1, device=device),
        "batch_idx": torch.arange(args.batch, device=device),
        "bboxes": torch.tensor([[0.375, 0.375, 0.25, 0.25]], device=device).repeat(args.batch, 1),
    }
    folder = ROOT / "0_orange_yaml" / "SAGE_series"
    paths = [
        folder / "SAGE21_innovation_pyramid.yaml",
        folder / "SAGE23_joint_core_v3.yaml",
        *sorted(folder.glob("SAGE3*.yaml")),
    ]
    results = []
    for path in paths:
        torch.manual_seed(42)
        model = SegmentationModel(path, nc=1, verbose=False).to(device).train()
        model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        if path.stem.startswith("SAGE23"):
            model.args.citrus_topology, model.args.citrus_boundary, model.args.citrus_query = 0.1, 0.1, 0.03
        times = []
        for step in range(args.steps + 5):
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            model.zero_grad(set_to_none=True)
            loss, _ = model.loss(batch)
            loss.sum().backward()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            if step >= 5:
                times.append((time.perf_counter() - start) * 1000)
        item = {
            "model": path.stem,
            "median_forward_backward_ms": statistics.median(times),
            "params": sum(p.numel() for p in model.parameters()),
            "gflops640": get_flops(model, 640),
        }
        results.append(item)
        print(item, flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
    base = next(r["median_forward_backward_ms"] for r in results if r["model"].startswith("SAGE30"))
    for result in results:
        result["relative_to_control"] = result["median_forward_backward_ms"] / base
    output = {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch": torch.__version__,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "steps": args.steps,
        "threads": args.threads,
        "amp": False,
        "io_optimizer_validation_included": False,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
