"""Measure parameters, GFLOPs and wall-clock forward latency for Light models."""

from __future__ import annotations

import argparse
import csv
import statistics
import time
from pathlib import Path

import torch

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


ROOT = Path(__file__).resolve().parent
MODELS = (
    ("YOLO11n_control", ROOT / "0_orange_yaml" / "G_0830_series" / "00_g00_official_control.yaml"),
    ("Light00_backbone_only", ROOT / "0_orange_yaml" / "Light_series" / "Light00_backbone_only.yaml"),
    ("Light01_neck_only", ROOT / "0_orange_yaml" / "Light_series" / "Light01_neck_only.yaml"),
    ("Light02_joint_core", ROOT / "0_orange_yaml" / "Light_series" / "Light02_joint_core.yaml"),
    ("Light03_deploy_lite", ROOT / "0_orange_yaml" / "Light_series" / "Light03_deploy_lite.yaml"),
    ("Light04_quality_rank", ROOT / "0_orange_yaml" / "Light_series" / "Light04_quality_rank.yaml"),
    (
        "Light05_repmixer_backbone_only",
        ROOT / "0_orange_yaml" / "Light_series" / "Light05_repmixer_backbone_only.yaml",
    ),
    ("Light06_repmixer_afpn", ROOT / "0_orange_yaml" / "Light_series" / "Light06_repmixer_afpn.yaml"),
    ("Light07_repmixer_pan_lite", ROOT / "0_orange_yaml" / "Light_series" / "Light07_repmixer_pan_lite.yaml"),
)


def parse_args() -> argparse.Namespace:
    """Parse deterministic latency benchmark options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "figures" / "citrus_light_profile_v3.csv")
    return parser.parse_args()


def synchronize(device: torch.device) -> None:
    """Synchronize CUDA when necessary."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    """Run the same eager-mode benchmark for the control and every Light YAML."""
    args = parse_args()
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    rows = []
    for name, yaml_path in MODELS:
        model = YOLO(str(yaml_path), task="segment", verbose=False).model.to(device).eval()
        sample = torch.randn(1, 3, args.imgsz, args.imgsz, device=device)
        with torch.inference_mode():
            for _ in range(args.warmup):
                model(sample)
            trials = []
            for _ in range(args.trials):
                synchronize(device)
                start = time.perf_counter()
                for _ in range(args.repeats):
                    model(sample)
                synchronize(device)
                trials.append((time.perf_counter() - start) * 1000 / args.repeats)
        latency_ms = statistics.median(trials)
        rows.append(
            {
                "model": name,
                "params_m": round(sum(parameter.numel() for parameter in model.parameters()) / 1e6, 4),
                "gflops_640": round(get_flops(model, imgsz=640), 3),
                f"latency_ms_{args.imgsz}": round(latency_ms, 3),
                "latency_min_ms": round(min(trials), 3),
                "latency_max_ms": round(max(trials), 3),
                "device": str(device),
                "threads": args.threads,
            }
        )
        print(rows[-1])
        del model, sample

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
