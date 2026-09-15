"""Bounded E V3 microbenchmark; excludes decoding/NMS, data loading and optimizer time."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu or one physical CUDA index, in a fresh Python process")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.iterations < 3:
        raise ValueError("At least three measurements required")
    if args.device != "cpu":
        if not args.device.isdigit():
            raise ValueError("Use one physical CUDA index")
        existing = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if existing and existing != args.device:
            raise ValueError("Inherited CUDA visibility conflicts; use a fresh terminal")
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device

    import torch
    import torch.nn.functional as F

    from citrus_e_v3_suite import NAMES, YAML_DIR
    from tests.test_citrus_e_v3 import batch as example_batch
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    from ultralytics.utils.torch_utils import get_flops

    torch.set_num_threads(2)
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")
    torch.manual_seed(42)
    batch = example_batch()
    batch["img"] = torch.rand(2, 3, 640, 640)
    batch["masks"] = F.interpolate(batch["masks"][None], (160, 160), mode="nearest")[0]
    batch = {k: v.to(device) for k, v in batch.items()}
    records = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    for name in NAMES:
        model = SegmentationModel(YAML_DIR / f"{name}.yaml", nc=1, verbose=False)
        row = dict(name=name, parameters=sum(p.numel() for p in model.parameters()), gflops640=get_flops(model, 640))
        model = model.to(device).train()
        model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        measurements = []
        for i in range(args.iterations + 3):
            model.zero_grad(set_to_none=True)
            sync()
            start = time.perf_counter()
            loss, _ = model.loss(batch)
            loss.sum().backward()
            sync()
            if i >= 3:
                measurements.append(1000 * (time.perf_counter() - start))
        row["forward_loss_backward_median_ms"] = statistics.median(measurements)
        model.eval().fuse(verbose=False)
        measurements = []
        with torch.inference_mode():
            for i in range(args.iterations + 3):
                sync()
                start = time.perf_counter()
                model(batch["img"])
                sync()
                if i >= 3:
                    measurements.append(1000 * (time.perf_counter() - start))
        row["fused_forward_median_ms"] = statistics.median(measurements)
        records.append(row)
        print(json.dumps(row), flush=True)
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    report = dict(
        records=records,
        module_sha256=hashlib.sha256((ROOT / "ultralytics/nn/modules/citrus_e_v3.py").read_bytes()).hexdigest(),
        protocol=dict(
            batch=2,
            imgsz=640,
            threads=2,
            warmup=3,
            iterations=args.iterations,
            amp=False,
            dtype="float32",
            device=str(device),
            physical_device=args.device,
            platform=platform.platform(),
            torch=torch.__version__,
        ),
        limits="Synthetic batch, no optimizer/data-loader/augmentation/NMS. CPU is not a GPU speed guarantee. "
        "Measure whole-image plus crop passes and fusion separately; repeated identical architectures expose timing noise.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
