"""Bounded CPU diagnostics: full training step and dense mask decoding, not GPU benchmarks."""
# ruff: noqa: E402
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from citrus_e_v6_suite import RUN_OVERRIDES, YAML_DIR
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace, ops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(2)
    rows = []
    for name in ("V6_00_control", "V6_01_mr2", "V6_02_fine", "V6_08_phase"):
        torch.manual_seed(42)
        model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False)
        model.args = IterableSimpleNamespace(**{**DEFAULT_CFG_DICT, **RUN_OVERRIDES[name], "amp": False})
        ratio = model.args.mask_ratio
        masks = torch.zeros(2, 640 // ratio, 640 // ratio)
        boxes = []
        for i in range(12):
            x, y, side = 64 + 128 * (i % 4), 64 + 160 * (i // 4), (12, 24, 48)[i % 3]
            boxes.append([(x + side / 2) / 640, (y + side / 2) / 640, side / 640, side / 640])
            masks[:, y // ratio:(y + side) // ratio, x // ratio:(x + side) // ratio] = i + 1
        batch = dict(img=torch.rand(2, 3, 640, 640), masks=masks, cls=torch.zeros(24, 1),
                     bboxes=torch.tensor(boxes * 2), batch_idx=torch.arange(2).repeat_interleave(12).float())
        steps = []
        for i in range(3):
            model.zero_grad(set_to_none=True)
            start = time.perf_counter()
            loss, _ = model.loss(batch)
            loss.sum().backward()
            assert torch.isfinite(loss).all()
            if i:
                steps.append((time.perf_counter() - start) * 1000)
        size = 640 // model.model[-1].proto_stride
        proto, coefficient = torch.rand(32, size, size), torch.rand(300, 32)
        bb = torch.tensor([[0., 0., 640., 640.]]).repeat(300, 1)
        decode = []
        with torch.inference_mode():
            for i in range(3):
                start = time.perf_counter()
                masks_out = ops.process_mask(proto, coefficient, bb, (640, 640))
                if i:
                    decode.append((time.perf_counter() - start) * 1000)
                del masks_out
        row = dict(name=name, step_batch2_ms=statistics.median(steps),
                   decode_300_masks_ms=statistics.median(decode),
                   decoded_logits_fp32_MiB=300 * size * size * 4 / 2**20)
        rows.append(row)
        print(json.dumps(row), flush=True)
        del model
    args.output.write_text(json.dumps(dict(device="cpu", torch=torch.__version__, threads=2,
        imgsz=640, batch=2, instances_per_image=12, rows=rows,
        limit="Synthetic CPU probe, two timed repetitions; not formal GPU training or end-to-end sliced latency."),
        indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
