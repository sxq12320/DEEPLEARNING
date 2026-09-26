"""Profile independent I_V4 models with real-size inputs, avoiding YOLO stride-scaled FLOP estimates."""

import argparse
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--only", default="I52_native_review")
    args = parser.parse_args()
    torch.set_num_threads(min(torch.get_num_threads(), 4))
    for name in args.only.split(","):
        name = name.strip()
        yaml_path = ROOT / "0_orange_yaml/I_V4_series" / (name + ".yaml")
        model = YOLO(str(yaml_path), task="segment").model.eval().to(args.device)
        image = torch.zeros(1, 3, args.imgsz, args.imgsz, device=args.device)
        with torch.no_grad():
            model(image)
            if args.device != "cpu":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            model(image)
            if args.device != "cpu":
                torch.cuda.synchronize()
            elapsed = (time.perf_counter() - t0) * 1000
            activities = [torch.profiler.ProfilerActivity.CPU]
            if args.device != "cpu":
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            with torch.profiler.profile(activities=activities, with_flops=True) as profiler:
                model(image)
                if args.device != "cpu":
                    torch.cuda.synchronize()
        counted = sum(event.flops for event in profiler.key_averages()) / 1e9
        params = sum(parameter.numel() for parameter in model.parameters())
        print("{}: params={} profiler-counted GFLOPs@{}={:.3f}; single-forward ms={:.1f}".format(
            name, params, args.imgsz, counted, elapsed
        ))
        print("  FLOPs exclude unsupported operators; compare measured GPU latency on one machine.")


if __name__ == "__main__":
    main()
