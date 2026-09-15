"""Build/profile eight E V5 arms, optionally exercise a tiny real sliced-data training loop."""
# ruff: noqa: E402 -- CLI establishes the scoped project import root before project imports.

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from citrus_e_v5_suite import NAMES, YAML_DIR
from citrus_e_v5_slicing import MultiScaleTrainingTrainer, prepare_multiscale_views
from citrus_slicing import SlicedTrainingTrainer, prepare_views
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-data", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model.float()
    original = source.state_dict()
    results = []
    for name in NAMES:
        model = SegmentationModel(YAML_DIR / (name + ".yaml"), nc=1, verbose=False)
        model.load(source, verbose=False)
        params = sum(p.numel() for p in model.parameters())
        equal = sum(
            p.numel()
            for key, p in model.named_parameters()
            if key in original and p.shape == original[key].shape and torch.equal(p, original[key])
        )
        gflops = get_flops(model, 640)
        model.eval()
        x = torch.rand(1, 3, 640, 640)
        times = []
        with torch.inference_mode():
            for i in range(7):
                start = time.perf_counter()
                model(x)
                if i >= 2:
                    times.append((time.perf_counter() - start) * 1000)
        row = dict(
            name=name,
            params=params,
            thop_gflops640=gflops,
            initialization_equal_fraction=equal / params,
            cpu_batch1_forward_ms=statistics.median(times),
        )
        results.append(row)
        print(json.dumps(row), flush=True)
    payload = dict(
        torch=torch.__version__,
        threads=2,
        device="cpu",
        models=results,
        warning="THOP omits some functional operators. CPU diagnostic timing is not GPU speed evidence.",
        smoke=[],
    )
    if args.smoke_data:
        prepared = prepare_views(args.smoke_data, args.output / "SMOKE_VIEWS_NOT_FORMAL", 128, 0.6)
        fine = prepare_multiscale_views(args.smoke_data, args.output / "SMOKE_FINE_VIEWS_NOT_FORMAL", 128)
        for name in (NAMES[0], NAMES[-1]):
            model = YOLO(str(YAML_DIR / (name + ".yaml")), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
            import importlib

            runner = importlib.import_module("20260911_citrus_e_v5_batch")
            model.add_callback("on_model_save", runner.save_best_mask)
            model.add_callback("on_pretrain_routine_end", runner.record_loaded_samples)
            metrics = model.train(
                data=str(fine if name == NAMES[-1] else prepared),
                trainer=MultiScaleTrainingTrainer if name == NAMES[-1] else SlicedTrainingTrainer,
                epochs=1,
                imgsz=128,
                batch=2,
                workers=0,
                device="cpu",
                cache=True,
                amp=False,
                optimizer="AdamW",
                lr0=0.001,
                momentum=0.937,
                weight_decay=0.0005,
                warmup_epochs=0,
                mosaic=0,
                close_mosaic=0,
                nbs=2,
                seed=42,
                plots=False,
                project=str(args.output / "SMOKE_RUNS_NOT_FORMAL"),
                name=name,
                exist_ok=False,
            )
            path = Path(model.trainer.save_dir) / "weights/best.pt"
            assert path.is_file()
            assert (path.parent / "best_mask50.pt").is_file()
            assert (path.parent / "best_mask.pt").is_file()
            loaded = YOLO(str(path), verbose=False)
            assert type(loaded.model.model[-1]) is type(model.model.model[-1])
            payload["smoke"].append(
                dict(name=name, checkpoint=str(path), success=True, metrics=metrics.results_dict, formal_accuracy=False)
            )
    (args.output / "audit.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
