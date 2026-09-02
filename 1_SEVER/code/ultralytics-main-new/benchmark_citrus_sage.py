"""Measure whole-model forward/backward overhead before committing to SAGE training."""

from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "SAGE_series"


def parse_args() -> argparse.Namespace:
    """Parse a portable speed-gate interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="SAGE14_joint_core.yaml")
    parser.add_argument("--device", default="0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--max-ratio", type=float, default=1.20)
    parser.add_argument("--amp", action="store_true", help="Use only for an explicit paired AMP audit.")
    return parser.parse_args()


def tensor_objective(value) -> torch.Tensor:
    """Reduce nested training output to a differentiable scalar."""
    tensors: list[torch.Tensor] = []

    def collect(item) -> None:
        if isinstance(item, torch.Tensor) and item.is_floating_point():
            tensors.append(item)
        elif isinstance(item, dict):
            for child in item.values():
                collect(child)
        elif isinstance(item, (list, tuple)):
            for child in item:
                collect(child)

    collect(value)
    if not tensors:
        raise RuntimeError("Model produced no floating-point training tensors")
    return sum(tensor.float().square().mean() for tensor in tensors)


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA only when required for honest timing."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(yaml_path: Path, device: torch.device, image: torch.Tensor, warmup: int, iterations: int, amp: bool) -> float:
    """Return median milliseconds for a whole-model forward/backward step."""
    model = YOLO(str(yaml_path), task="segment", verbose=False).model.to(device).train()
    timings: list[float] = []
    total = warmup + iterations
    for index in range(total):
        model.zero_grad(set_to_none=True)
        synchronize(device)
        started = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp and device.type == "cuda"):
            objective = tensor_objective(model(image))
        objective.backward()
        synchronize(device)
        if index >= warmup:
            timings.append((time.perf_counter() - started) * 1000.0)
    return statistics.median(timings)


def main() -> None:
    """Compare one SAGE model against the exact control and enforce a ratio gate."""
    args = parse_args()
    device = torch.device(f"cuda:{args.device}" if str(args.device).isdigit() else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but torch.cuda.is_available() is false")
    if args.batch < 1 or args.imgsz < 64 or args.warmup < 1 or args.iterations < 3:
        raise ValueError("Use batch>=1, imgsz>=64, warmup>=1 and iterations>=3")

    candidate = YAML_DIR / args.model
    control = YAML_DIR / "SAGE10_official_control.yaml"
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    image = torch.randn(args.batch, 3, args.imgsz, args.imgsz, device=device)
    control_ms = benchmark(control, device, image, args.warmup, args.iterations, args.amp)
    candidate_ms = benchmark(candidate, device, image, args.warmup, args.iterations, args.amp)
    ratio = candidate_ms / control_ms
    print(f"control={control_ms:.2f} ms/step")
    print(f"candidate={candidate_ms:.2f} ms/step")
    print(f"ratio={ratio:.3f}x, limit={args.max_ratio:.3f}x")
    if ratio > args.max_ratio:
        raise SystemExit("SPEED GATE FAILED: do not start the long SAGE run on this GPU.")
    print("SPEED GATE PASSED")


if __name__ == "__main__":
    main()
