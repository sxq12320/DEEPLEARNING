"""Audit EXISTING E V4 models. CPU diagnostics, not accuracy/speed benchmark claims."""
# ruff: noqa: E402 -- direct script entry must bind this checkout before local imports.

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.nn.modules.citrus_e_v4 import EV4ChromaFront, EV4IntegralContext
from ultralytics.nn.modules.citrus_far import CARAFE
from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops


def profile(module, x, reference):
    def timing(forward):
        times = []
        for i in range(7):
            module.zero_grad(set_to_none=True)
            x.grad = None
            start = time.perf_counter()
            forward(x).square().mean().backward()
            if i >= 2:
                times.append((time.perf_counter() - start) * 1000)
        return round(statistics.median(times), 3)

    return {
        "shape": list(x.shape),
        "before_cpu_forward_backward_ms": timing(reference),
        "after_cpu_forward_backward_ms": timing(module),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke-data", type=Path, help="Optional TEMPORARY tiny dataset for real one-epoch tests.")
    args = parser.parse_args()
    torch.set_num_threads(2)
    args.output.mkdir(parents=True, exist_ok=True)
    source = YOLO(str(ROOT / "yolo11n-seg.pt"), verbose=False).model
    csd = source.state_dict()
    rows = []
    for path in sorted((ROOT / "0_orange_yaml/E_V4_series").glob("*.yaml")):
        model = SegmentationModel(path, nc=1, verbose=False)
        params = dict(model.named_parameters())
        count = sum(p.numel() for p in params.values())
        before = sum(p.numel() for k, p in params.items() if k in csd and csd[k].shape == p.shape)
        after = 0
        for k, p in params.items():
            _, index, suffix = k.split(".", 2)
            origin = model.yaml["pretrained_layer_map"][int(index)]
            key = f"model.{origin}.{suffix}"
            if origin >= 0 and key in csd and csd[key].shape == p.shape:
                after += p.numel()
        row = dict(
            model=path.stem,
            nc=1,
            params=count,
            thop_gflops_estimate_640=round(get_flops(model, 640), 3),
            old_same_index_shape_match_pct=round(before / count * 100, 2),
            explicit_semantic_mapping_pct=round(after / count * 100, 2),
        )
        rows.append(row)
        print(json.dumps(row), flush=True)
    chroma = EV4ChromaFront(3)

    def old_chroma(x):
        mixed = x + torch.einsum("oc,bchw->bohw", chroma.ccm, x)
        basis = (1 - (mixed.clamp(0, 1).unsqueeze(2) - chroma.knots).abs() * 7).clamp_min(0)
        return mixed + torch.einsum("cp,bcphw->bchw", chroma.tone, basis)

    carafe = CARAFE(128)

    def old_carafe(x):
        b, c, h, w = x.shape
        weights = carafe.pix_shf(carafe.enc(carafe.comp(x))).softmax(1)
        patches = carafe.unfold(carafe.upsmp(x)).view(b, c, -1, 2 * h, 2 * w)
        return torch.einsum("bkhw,bckhw->bchw", weights, patches)

    context = EV4IntegralContext(256)

    def old_context(x):
        cat = torch.cat([F.interpolate(p(x), x.shape[-2:], mode="nearest") for p in context.pools], 1)
        return x + context.gain * context.mix(cat)

    timing = {
        "chroma": profile(chroma, torch.rand(2, 3, 640, 640, requires_grad=True), old_chroma),
        "carafe": profile(carafe, torch.rand(2, 128, 40, 40, requires_grad=True), old_carafe),
        "integral_context": profile(context, torch.rand(2, 256, 20, 20, requires_grad=True), old_context),
    }
    result = {
        "torch": torch.__version__,
        "device": "cpu",
        "threads": 2,
        "note": "Old shape matches can include WRONG layers. THOP omits some functional operations; "
        "CPU microbenchmarks do not predict CUDA epoch time. No new trained accuracy results.",
        "models": rows,
        "microbenchmarks": timing,
        "smoke": [],
    }
    if args.smoke_data:
        for stem, optimizer, extra in [
            ("E40_control", "SMC", {"nwd_ratio": 0.5, "citrus_vfl": 0.5}),
            ("E52_topo_head", "SMCAO", {"citrus_boundary": 0.25, "citrus_query": 0.1}),
        ]:
            checkpoint_states = []

            def inspect_checkpoint(trainer):
                ckpt = torch.load(trainer.last, map_location="cpu", weights_only=False)
                state = ckpt["smc_scheduler"]
                assert state and state["step_count"] > 0
                checkpoint_states.append({"steps": state["step_count"], "mode": state["mode"]})

            model = YOLO(str(ROOT / f"0_orange_yaml/E_V4_series/{stem}.yaml"), verbose=False)
            model.load(str(ROOT / "yolo11n-seg.pt"))
            model.add_callback("on_model_save", inspect_checkpoint)
            model.train(
                data=str(args.smoke_data),
                epochs=1,
                imgsz=128,
                batch=2,
                workers=0,
                device="cpu",
                optimizer=optimizer,
                lr0=0.001,
                momentum=0.937,
                amp=False,
                cache=True,
                warmup_epochs=0,
                project=str(args.output / "SMOKE_NOT_FORMAL"),
                name=stem,
                exist_ok=False,
                plots=False,
                mosaic=0,
                close_mosaic=0,
                nbs=2,
                seed=42,
                **extra,
            )
            assert checkpoint_states
            result["smoke"].append({"model": stem, "optimizer": optimizer, "checkpoint_state": checkpoint_states})
    (args.output / "audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"microbenchmarks": timing, "smoke": result["smoke"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
