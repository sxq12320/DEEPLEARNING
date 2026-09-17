"""Build/profile E V11 arms, optionally exercise a tiny real sliced-data training loop."""
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

from citrus_e_v5_slicing import prepare_multiscale_views
from citrus_e_v6_training import EV6TrainingTrainer, EV6Validator
from citrus_e_v11_suite import NAMES, RUN_OVERRIDES, YAML_DIR
from ultralytics import YOLO
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-data", type=Path)
    parser.add_argument("--smoke-only", default="", help="Comma-separated exact model names; default 00/06/08/11")
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
        equal = 0
        mapping = model.yaml.get("pretrained_layer_map", {})
        for key, p in model.named_parameters():
            prefix, index, tail = key.split(".", 2)
            src_index = mapping.get(int(index), int(index))
            src_key = f"{prefix}.{src_index}.{tail}"
            if src_index >= 0 and src_key in original and p.shape == original[src_key].shape:
                equal += p.numel() if torch.equal(p, original[src_key]) else 0
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
            proto_stride=int(getattr(model.model[-1], "proto_stride", 4)),
            initialization_equal_fraction=equal / params,
            cpu_batch1_forward_ms=statistics.median(times),
            candidates640=34000 if model.model[-1].p2_head else 8400,
            # THOP sees the offset conv, but not functional deform_conv2d.
            # This is the dense multiply-add component only, excluding sampling.
            untraced_dcn_conv_gflops640=(
                2 * model.model[-1].deform.weight.weight.numel() * 160**2 / 1e9
                if model.model[-1].deform_detail
                else 0.0
            ),
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
    record = args.output / "audit.json"
    record.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.smoke_data:
        selected = (
            [n.strip() for n in args.smoke_only.split(",") if n.strip()]
            if args.smoke_only
            else [NAMES[i] for i in (0, 6, 8, 11)]
        )
        if not selected or set(selected) - set(NAMES):
            raise ValueError("Choose smoke model names from the declared suite")
        fine = prepare_multiscale_views(args.smoke_data, args.output / "SMOKE_FINE_VIEWS_NOT_FORMAL", 128)
        for name in selected:
            model = YOLO(str(YAML_DIR / (name + ".yaml")), verbose=False).load(str(ROOT / "yolo11n-seg.pt"))
            import importlib

            runner = importlib.import_module("20260917_citrus_e_v11_batch")
            model.add_callback("on_model_save", runner.save_best_mask)
            model.add_callback("on_pretrain_routine_end", runner.record_loaded_samples)
            metrics = model.train(
                data=str(fine),
                trainer=EV6TrainingTrainer,
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
                **RUN_OVERRIDES[name],
            )
            path = Path(model.trainer.save_dir) / "weights/best.pt"
            assert path.is_file()
            assert (path.parent / "best_mask50.pt").is_file()
            assert (path.parent / "best_mask.pt").is_file()
            loaded = YOLO(str(path), verbose=False)
            assert type(loaded.model.model[-1]) is type(model.model.model[-1])
            sample = model.trainer.test_loader.dataset[0]
            assert tuple(sample["masks"].shape[-2:]) == tuple(s // 2 for s in sample["img"].shape[-2:])
            loaded.val(
                validator=EV6Validator,
                data=str(fine),
                imgsz=128,
                batch=2,
                workers=0,
                device="cpu",
                plots=False,
                project=str(args.output / "STANDALONE_NOT_FORMAL"),
                name=name,
            )
            payload["smoke"].append(
                dict(
                    name=name,
                    checkpoint=str(path),
                    success=True,
                    metrics=metrics.results_dict,
                    formal_accuracy=False,
                )
            )
            record.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
