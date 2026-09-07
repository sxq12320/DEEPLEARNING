"""Interleave models to reduce order/thermal confounding in CPU microtiming."""

# ruff: noqa: E402
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import torch
from citrus_sage_v7_suite import NAMES, YAML_DIR
from scripts.benchmark_sage_v7 import make_batch
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils import DEFAULT_CFG_DICT, IterableSimpleNamespace


def main():
    output = ROOT / "reports/sage_v7_20260906/cpu_interleaved640.json"
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    torch.manual_seed(42)
    models = {n: SegmentationModel(YAML_DIR / f"{n}.yaml", nc=1, verbose=False) for n in NAMES}
    for model in models.values():
        model.args = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
    batch = make_batch(1, 640, torch.device("cpu"))
    durations = {n: {phase: [] for phase in ("forward", "train")} for n in NAMES}
    rng = random.Random(20260906)
    for phase in ("forward", "train"):
        for model in models.values():
            model.train(phase == "train")
        for step in range(23):
            names = list(NAMES)
            rng.shuffle(names)
            for name in names:
                model = models[name]
                start = time.perf_counter()
                if phase == "forward":
                    with torch.inference_mode():
                        pred = model(batch["img"])
                    del pred
                else:
                    model.zero_grad(set_to_none=True)
                    loss, _ = model.loss(batch)
                    loss.sum().backward()
                    del loss
                elapsed = 1000 * (time.perf_counter() - start)
                if step >= 3:
                    durations[name][phase].append(elapsed)
    payload = dict(
        device="CPU",
        batch=1,
        imgsz=640,
        torch=torch.__version__,
        threads=2,
        amp=False,
        protocol="Randomized model order each iteration; 3 warmup+20 measurements per phase. "
        "No data loading/optimizer/NMS. Not GPU latency or accuracy.",
        medians={n: {p: statistics.median(v) for p, v in d.items()} for n, d in durations.items()},
        raw_ms=durations,
    )
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["medians"], indent=2))


if __name__ == "__main__":
    main()
