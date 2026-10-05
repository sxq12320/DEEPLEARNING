"""J2 unfused/fused cost and forward timing; excludes I/O, NMS, mask reconstruction and stitching."""

import argparse
import copy
import csv
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import thop  # noqa: E402

from citrus_j2_suite import YAML_DIR  # noqa: E402
from citrus_j2_structure_suite import STRUCTURE_NAMES  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--only", default="")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    names = args.only.split(",") if args.only else ["J20_glance_anchor"] + list(STRUCTURE_NAMES)
    rows = []
    x = torch.rand(1, 3, 640, 640, device=args.device)
    for name in names:
        base = YOLO(str(YAML_DIR / (name + ".yaml")), task="segment", verbose=False).model.to(args.device).eval()
        for fused in (False, True):
            model = copy.deepcopy(base)
            if fused:
                model.fuse(verbose=False)
            with torch.inference_mode():
                macs, _ = thop.profile(model, inputs=(x,), verbose=False)
                for _ in range(2):
                    model(x)
                times = []
                for _ in range(args.repeats):
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    start = time.perf_counter()
                    model(x)
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    times.append((time.perf_counter() - start) * 1000)
            row = dict(model=name, fused=fused, params=sum(p.numel() for p in model.parameters()),
                       thop_gflops=2 * macs / 1e9, median_forward_ms=statistics.median(times))
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "costs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / "environment.json").write_text(json.dumps(dict(torch=torch.__version__, device=args.device,
        threads=args.threads, repeats=args.repeats, input=[1, 3, 640, 640], dtype="FP32",
        scope="Forward only. THOP excludes elementwise/pooling/interpolation; not training or end-to-end timing."),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
