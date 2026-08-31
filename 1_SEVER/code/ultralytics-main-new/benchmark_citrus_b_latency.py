"""Benchmark fused CitrusB inference latency with an explicit hardware manifest.

This is a structural speed benchmark, not an accuracy evaluation. Run it on the
same deployment GPU for every model included in a paper table.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "B_series"
DEFAULT_OUTPUT = ROOT / "1_results" / "_compatibility"


def parse_args() -> argparse.Namespace:
    """Parse reproducible latency settings."""
    parser = argparse.ArgumentParser(description="Benchmark batch-1 fused CitrusB forward latency.")
    parser.add_argument("--device", default="cpu", help="cpu, 0, or cuda:0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads; 0 keeps the current PyTorch setting.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    """Normalize Ultralytics-style numeric device strings."""
    if value.isdigit():
        value = f"cuda:{value}"
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return device


def synchronize(device: torch.device) -> None:
    """Synchronize asynchronous CUDA work when needed."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    """Build, fuse, warm up, and time every CitrusB YAML."""
    args = parse_args()
    if args.imgsz <= 0 or args.warmup < 0 or args.iterations < 1:
        raise ValueError("imgsz and iterations must be positive; warmup must be non-negative.")
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    device = resolve_device(args.device)
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment = {
        "platform": platform.platform(),
        "processor": platform.processor() or "unreported",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "torch_threads": torch.get_num_threads(),
        "imgsz": args.imgsz,
        "batch": 1,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "fused": True,
    }

    sample = torch.zeros(1, 3, args.imgsz, args.imgsz, device=device)
    rows = []
    for yaml_path in sorted(YAML_DIR.glob("*.yaml")):
        train_model = SegmentationModel(yaml_path, ch=3, nc=1, verbose=False)
        params = sum(parameter.numel() for parameter in train_model.parameters())
        gflops = get_flops(train_model, imgsz=args.imgsz)
        model = train_model.eval().fuse(verbose=False).to(device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(sample)
            synchronize(device)
            timings = []
            for _ in range(args.iterations):
                start = time.perf_counter()
                model(sample)
                synchronize(device)
                timings.append((time.perf_counter() - start) * 1000.0)
        row = {
            "model": yaml_path.stem,
            "params": params,
            "gflops": gflops,
            "latency_median_ms": float(np.median(timings)),
            "latency_mean_ms": float(np.mean(timings)),
            "latency_p90_ms": float(np.percentile(timings, 90)),
            "fps_from_median": 1000.0 / float(np.median(timings)),
        }
        rows.append(row)
        print(
            f"{row['model']}: median={row['latency_median_ms']:.2f} ms, "
            f"p90={row['latency_p90_ms']:.2f} ms, {row['fps_from_median']:.1f} FPS"
        )
        del model, train_model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    suffix = "cuda" if device.type == "cuda" else "cpu"
    csv_path = output / f"citrus_b_latency_{suffix}.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / f"citrus_b_latency_{suffix}_environment.json").write_text(
        json.dumps(environment, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
