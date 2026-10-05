"""Real-size J FLOP estimates and forward timings, not training-speed promises."""

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import thop

from citrus_j_suite import NAMES, YAML_DIR
from ultralytics import YOLO
from ultralytics.nn.modules.citrus_j import JContext


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    rows = []
    for name in NAMES:
        model = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model.to(args.device).eval()
        image = torch.rand(1, 3, 640, 640, device=args.device)
        with torch.no_grad():
            macs, _ = thop.profile(model, inputs=(image,), verbose=False)
            functional = sum(
                6 * 40 * 40 * m.grid**2 * m.dim for m in model.modules() if isinstance(m, JContext) and m.mode
            )
            for _ in range(2):
                model(image)
            if image.is_cuda:
                torch.cuda.synchronize()
            times = []
            for _ in range(args.repeats):
                start = time.perf_counter()
                model(image)
                if image.is_cuda:
                    torch.cuda.synchronize()
                times.append(1000 * (time.perf_counter() - start))
        row = dict(
            model=name,
            params=sum(p.numel() for p in model.parameters()),
            thop_gflops=2 * macs / 1e9,
            context_matmul_gflops=functional / 1e9,
            estimated_gflops=(2 * macs + functional) / 1e9,
            median_forward_ms=statistics.median(times),
        )
        print(json.dumps(row), flush=True)
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "model_costs.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "profile_environment.json").write_text(
        json.dumps(
            dict(
                torch=torch.__version__,
                device=args.device,
                cpu_threads=args.threads,
                input=[1, 3, 640, 640],
                repeats=args.repeats,
                dtype="float32",
                fused=False,
                includes="forward only; no I/O, pre/postprocessing, NMS, crop stitching, losses or backward",
                limitations="THOP Conv/BN estimate + explicit J context matmuls; softmax/pooling/elementwise ops omitted.",
                models=rows,
            ),
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
