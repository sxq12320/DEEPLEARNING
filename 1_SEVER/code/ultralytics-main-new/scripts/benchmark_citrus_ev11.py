"""Compare actual operators on YOUR GPU, float32/AMP-off; no training queue or occupancy guard.

Example: python scripts/benchmark_citrus_ev11.py --device 1 --output /data/sxq/results/E/V11_speed.json --train
Synthetic training steps are a diagnostic, NOT epoch/end-to-end dataset speed or formal accuracy.
"""
# ruff: noqa: E402

import argparse
import gc
import json
import os
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="1", help="Physical GPU index; CPU allowed for inference diagnostic only")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--train", action="store_true")
    args = p.parse_args()
    if args.batch < 1 or args.steps < 1:
        raise ValueError("Positive batch and timing steps required")
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.device != "cpu":
        if not args.device.isdigit():
            raise ValueError("Use one physical GPU index")
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    import torch
    from citrus_e_v11_suite import NAMES, YAML_DIR
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace
    from ultralytics.utils.torch_utils import get_flops

    torch.set_num_threads(2)
    if args.train and args.device == "cpu":
        raise ValueError("640/batch16 train benchmark is intentionally GPU-only")
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; select the server's matching PyTorch environment")
    device = torch.device("cpu" if args.device == "cpu" else "cuda:0")

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    rows = []
    for index in (0, 1, 2, 3, 4, 7, 6, 8, 9, 10, 11):
        torch.manual_seed(42)
        m = SegmentationModel(YAML_DIR / f"{NAMES[index]}.yaml", nc=1, verbose=False).float().to(device).eval()
        row = dict(name=NAMES[index], params=sum(v.numel() for v in m.parameters()), gflops640=get_flops(m, 640))
        x = torch.rand(1, 3, 640, 640, device=device)
        times = []
        with torch.inference_mode():
            for i in range(3 + args.steps):
                sync()
                start = time.perf_counter()
                m(x)
                sync()
                if i >= 3:
                    times.append((time.perf_counter() - start) * 1000)
        row["unfused_forward_b1_ms"] = statistics.median(times)
        if args.train:
            m.train()
            m.args = IterableSimpleNamespace(
                **{**DEFAULT_CFG_DICT, "mask_ratio": 2, "overlap_mask": True, "nwd_ratio": 0.0}
            )
            masks = torch.zeros(args.batch, 320, 320, device=device)
            masks[:, 100:220, 100:220] = 1
            masks[:, 12:15, 12:15] = 2
            batch = dict(
                img=torch.rand(args.batch, 3, 640, 640, device=device),
                cls=torch.zeros(2 * args.batch, 1, device=device),
                batch_idx=torch.arange(args.batch, device=device).repeat_interleave(2).float(),
                masks=masks,
                bboxes=torch.tensor(
                    [[0.5, 0.5, 0.375, 0.375], [0.0421875, 0.0421875, 0.009375, 0.009375]], device=device
                ).repeat(args.batch, 1),
            )
            opt = torch.optim.AdamW(m.parameters(), lr=0.001)
            times = []
            torch.cuda.reset_peak_memory_stats(device)
            for i in range(3 + args.steps):
                opt.zero_grad(set_to_none=True)
                sync()
                start = time.perf_counter()
                loss, _ = m.loss(batch)
                loss.sum().backward()
                opt.step()
                sync()
                if i >= 3:
                    times.append((time.perf_counter() - start) * 1000)
            row["synthetic_step_batch_ms"] = statistics.median(times)
            row["peak_memory_gib"] = torch.cuda.max_memory_allocated(device) / 1024**3
            del opt, batch, masks, loss
        rows.append(row)
        print(json.dumps(row), flush=True)
        del m, x
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
    payload = dict(
        device=str(device),
        physical_request=args.device,
        torch=torch.__version__,
        amp=False,
        batch=args.batch,
        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        models=rows,
        limits="Unfused float32; fixed synthetic GT; excludes dataloader, NMS, slicing, validation. THOP omits functional ops.",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
