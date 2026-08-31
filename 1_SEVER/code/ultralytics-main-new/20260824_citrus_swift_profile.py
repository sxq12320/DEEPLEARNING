"""Measure CitrusSwift complexity, transfer coverage, and real operator latency.

FLOPs are useful but insufficient: memory traffic, interpolation, concatenation,
and kernel implementation change real latency. This script therefore benchmarks
the same one-class 640-pixel models before and after Ultralytics fusion and saves
the complete protocol next to the table.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops, intersect_dicts


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "S_series"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile all CitrusSwift architecture candidates.")
    parser.add_argument("--device", default="cpu", help="cpu, 0, cuda:0, etc.")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1, help="CPU intra-op threads; ignored on CUDA.")
    parser.add_argument("--weights", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "citrus_swift_complexity_latency.csv")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    if value.isdigit():
        return torch.device(f"cuda:{value}")
    return torch.device(value)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(model: torch.nn.Module, sample: torch.Tensor, warmup: int, iterations: int) -> tuple[float, float]:
    with torch.inference_mode():
        for _ in range(warmup):
            model(sample)
        synchronize(sample.device)
        elapsed = []
        for _ in range(iterations):
            start = time.perf_counter()
            model(sample)
            synchronize(sample.device)
            elapsed.append((time.perf_counter() - start) * 1000.0)
    elapsed.sort()
    p90_index = min(len(elapsed) - 1, max(0, int(0.9 * len(elapsed)) - 1))
    return statistics.median(elapsed), elapsed[p90_index]


def load_source_state(weights: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(weights, map_location="cpu", weights_only=False)
    source = checkpoint.get("ema") or checkpoint["model"]
    return source.float().state_dict()


def profile_one(
    yaml_path: Path,
    source_state: dict[str, torch.Tensor],
    device: torch.device,
    imgsz: int,
    warmup: int,
    iterations: int,
) -> dict[str, float | int | str]:
    model = SegmentationModel(yaml_path, ch=3, nc=1, verbose=False).eval()
    target = model.state_dict()
    matched = intersect_dicts(source_state, target, exclude=())
    coverage = sum(target[key].numel() for key in matched) / sum(value.numel() for value in target.values())
    model.load_state_dict(matched, strict=False)
    params = sum(parameter.numel() for parameter in model.parameters())
    flops = get_flops(model, imgsz=imgsz)
    model.to(device)
    sample = torch.randn(1, 3, imgsz, imgsz, device=device)
    unfused_median, unfused_p90 = benchmark(model, sample, warmup, iterations)
    model.fuse(verbose=False)
    fused_median, fused_p90 = benchmark(model, sample, warmup, iterations)
    return {
        "model": yaml_path.stem,
        "params": params,
        "gflops": flops,
        "pretrained_coverage": coverage,
        "unfused_median_ms": unfused_median,
        "unfused_p90_ms": unfused_p90,
        "fused_median_ms": fused_median,
        "fused_p90_ms": fused_p90,
    }


def main() -> None:
    args = parse_args()
    if args.iterations < 3 or args.warmup < 1:
        raise ValueError("Use at least one warm-up and three measured iterations.")
    if not args.weights.is_file():
        raise FileNotFoundError(args.weights)
    device = resolve_device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    if device.type == "cpu":
        torch.set_num_threads(args.threads)

    source_state = load_source_state(args.weights)
    rows = [
        profile_one(path, source_state, device, args.imgsz, args.warmup, args.iterations)
        for path in sorted(YAML_DIR.glob("*.yaml"))
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    reference = rows[0]
    markdown = args.output.with_suffix(".md")
    lines = [
        "# CitrusSwift complexity and measured latency",
        "",
        f"Device: `{device}`; input: `{args.imgsz}`; batch: `1`; warm-up: `{args.warmup}`; "
        f"iterations: `{args.iterations}`; CPU threads: `{args.threads if device.type == 'cpu' else 'n/a'}`.",
        "",
        "| Model | Params | GFLOPs | Pretrain coverage | Fused median ms | vs ref | Fused p90 ms |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        speed_delta = (row["fused_median_ms"] / reference["fused_median_ms"] - 1.0) * 100.0
        lines.append(
            f"| {row['model']} | {row['params']:,} | {row['gflops']:.3f} | "
            f"{100.0 * row['pretrained_coverage']:.2f}% | {row['fused_median_ms']:.2f} | "
            f"{speed_delta:+.1f}% | {row['fused_p90_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "These timings are device-specific engineering measurements, not accuracy results. Re-run this script on the "
            "deployment GPU and exported TensorRT engine before making a speed claim.",
        ]
    )
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}\nWrote {markdown}")


if __name__ == "__main__":
    main()
