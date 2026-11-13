"""Unified Whole-Image (640x640) Sequential Runner & Profiler for APEX-8 Citrus Networks.

Designed for one-click VS Code Play or CLI execution on the server:
    python RUN_CITRUS_APEX.py --check-only --device cpu
    python RUN_CITRUS_APEX.py --smoke --device cpu
    python RUN_CITRUS_APEX.py --suite apex8 --device 0 --epochs 250
"""

from __future__ import annotations

import argparse
import csv
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _ensure_optional_vision_stubs() -> None:
    """Allow CPU architectural profiling & unit verification when cv2/torchvision are absent locally."""
    if "cv2" not in sys.modules and importlib.util.find_spec("cv2") is None:
        cv2_stub = types.ModuleType("cv2")
        cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", None)
        cv2_stub.__version__ = "0.0.0-stub"
        for attr, val in {
            "IMREAD_COLOR": 1,
            "IMREAD_UNCHANGED": -1,
            "INTER_LINEAR": 1,
            "INTER_NEAREST": 0,
            "BORDER_CONSTANT": 0,
            "COLOR_BGR2RGB": 4,
            "COLOR_RGB2BGR": 4,
            "CAP_PROP_FRAME_COUNT": 7,
            "CAP_PROP_FPS": 5,
            "CAP_PROP_FRAME_WIDTH": 3,
            "CAP_PROP_FRAME_HEIGHT": 4,
        }.items():
            setattr(cv2_stub, attr, val)
        for fn in ("setNumThreads", "imshow", "imread", "imwrite", "waitKey", "destroyAllWindows", "resize", "cvtColor"):
            setattr(cv2_stub, fn, lambda *a, **kw: None)
        sys.modules["cv2"] = cv2_stub

    if "torchvision" not in sys.modules and importlib.util.find_spec("torchvision") is None:
        import importlib.metadata as _im

        tv = types.ModuleType("torchvision")
        tv.__spec__ = importlib.machinery.ModuleSpec("torchvision", None)
        tv.__version__ = "0.0.0"
        ops = types.ModuleType("torchvision.ops")
        ops.__spec__ = importlib.machinery.ModuleSpec("torchvision.ops", None)
        transforms = types.ModuleType("torchvision.transforms")
        transforms.__spec__ = importlib.machinery.ModuleSpec("torchvision.transforms", None)
        tv.ops = ops
        tv.transforms = transforms
        sys.modules["torchvision"] = tv
        sys.modules["torchvision.ops"] = ops
        sys.modules["torchvision.transforms"] = transforms

        _orig_version = _im.version

        def _patched_version(pkg: str) -> str:
            try:
                return _orig_version(pkg)
            except _im.PackageNotFoundError:
                if pkg == "torchvision":
                    return "0.0.0"
                raise

        _im.version = _patched_version


_ensure_optional_vision_stubs()

import torch
import torch.nn as nn
import yaml

from ultralytics.nn.tasks import SegmentationModel
from ultralytics.utils.torch_utils import get_flops

# ===================== USER EDITABLE SERVER CONFIG =====================
SUITE = "apex8"  # "apex8", "top4", or "all_with_refs"
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
PROJECT = "/data/sxq/results/CITRUS_APEX_20261113"
PRETRAINED = ROOT / "yolo11n-seg.pt"
DEVICE = "0"
EPOCHS = 250
IMGSZ = 640
BATCH = 16
WORKERS = 4
AMP = False
CACHE = True
SEED = 0
# =====================================================================

MODEL_CATALOG = {
    "APEX00_baseline": "0_orange_yaml/APEX_series/APEX00_baseline.yaml",
    "APEX01_bsd_flagship": "0_orange_yaml/APEX_series/APEX01_bsd_flagship.yaml",
    "APEX02_g10_ultra_light": "0_orange_yaml/APEX_series/APEX02_g10_ultra_light.yaml",
    "APEX03_full_pan_bsd": "0_orange_yaml/APEX_series/APEX03_full_pan_bsd.yaml",
    "APEX04_g10_asym_bsd": "0_orange_yaml/APEX_series/APEX04_g10_asym_bsd.yaml",
    "APEX05_pds_fullrank_bsd": "0_orange_yaml/APEX_series/APEX05_pds_fullrank_bsd.yaml",
    "APEX06_m3_bsd_only": "0_orange_yaml/APEX_series/APEX06_m3_bsd_only.yaml",
    "APEX07_m2_m3_bifreq_bsd": "0_orange_yaml/APEX_series/APEX07_m2_m3_bifreq_bsd.yaml",
    "APEX08_ultimate_max_capacity": "0_orange_yaml/APEX_series/APEX08_ultimate_max_capacity.yaml",
    # Reference baselines in repository
    "REF_G10": "0_orange_yaml/G_series/10_yolo11n-seg-hybrid-lska-carafe-bifpn-p2b.yaml",
    "REF_SAGE42": "0_orange_yaml/SAGE_series/SAGE42_asym_semantic_detail.yaml",
    "REF_V12_06": "0_orange_yaml/E_V12_series/V12_06_recognition_contrast.yaml",
    "REF_RE07": "0_orange_yaml/R_EVIDENCE_series/RE07_full.yaml",
}

SUITES = {
    "apex8": [
        "APEX00_baseline",
        "APEX01_bsd_flagship",
        "APEX02_g10_ultra_light",
        "APEX03_full_pan_bsd",
        "APEX04_g10_asym_bsd",
        "APEX05_pds_fullrank_bsd",
        "APEX06_m3_bsd_only",
        "APEX07_m2_m3_bifreq_bsd",
        "APEX08_ultimate_max_capacity",
    ],
    "ablation": [
        "APEX00_baseline",
        "APEX06_m3_bsd_only",
        "APEX07_m2_m3_bifreq_bsd",
        "APEX03_full_pan_bsd",
        "APEX01_bsd_flagship",
    ],
    "top4": [
        "APEX00_baseline",
        "APEX02_g10_ultra_light",
        "APEX03_full_pan_bsd",
        "APEX06_m3_bsd_only",
        "APEX08_ultimate_max_capacity",
    ],
    "all_with_refs": list(MODEL_CATALOG.keys()),
}


def measure_exact_gflops(model: nn.Module, imgsz: int = 640) -> float:
    """Measure 640x640 GFLOPs (2 * MACs / 1e9) via THOP when available or exact Conv/Linear hooks."""
    thop_flops = get_flops(model, imgsz=imgsz)
    if thop_flops and thop_flops > 0.0:
        return float(thop_flops)

    total_macs = [0.0]
    hooks = []

    def _conv_hook(mod: nn.Conv2d, inp, out):
        out_h, out_w = out.shape[-2:]
        kernel_ops = mod.kernel_size[0] * mod.kernel_size[1] * (mod.in_channels // mod.groups)
        total_macs[0] += float(out_h * out_w * mod.out_channels * kernel_ops)

    def _linear_hook(mod: nn.Linear, inp, out):
        total_macs[0] += float(mod.in_features * mod.out_features)

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            hooks.append(m.register_forward_hook(_conv_hook))
        elif isinstance(m, nn.Linear):
            hooks.append(m.register_forward_hook(_linear_hook))

    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            x = torch.zeros(1, 3, imgsz, imgsz, dtype=torch.float32)
            model(x)
    finally:
        for h in hooks:
            h.remove()
        if was_training:
            model.train()

    return float(2.0 * total_macs[0] / 1e9)


def compute_transfer_stats(model: SegmentationModel, pretrained_path: Path) -> dict[str, float | int]:
    """Compute exact pretrained parameter count & element transfer rate from `yolo11n-seg.pt`."""
    target_sd = model.state_dict()
    total_tensors = len(target_sd)
    total_elems = sum(int(v.numel()) for v in target_sd.values())
    if not pretrained_path.is_file():
        return {
            "transferred_tensors": 0,
            "total_tensors": total_tensors,
            "tensor_ratio": 0.0,
            "transferred_elems": 0,
            "total_elems": total_elems,
            "elem_ratio": 0.0,
        }

    ckpt = torch.load(pretrained_path, map_location="cpu", weights_only=False)
    src_model = ckpt["model"] if isinstance(ckpt, dict) else ckpt
    src_sd = src_model.float().state_dict()

    before = {k: v.detach().clone() for k, v in target_sd.items()}
    model.load(ckpt, verbose=False)
    after = model.state_dict()

    # Also verify exact match count by comparing pretrained remapped state dict.
    matched_tensors = 0
    matched_elems = 0
    layer_map = getattr(model, "yaml", {}).get("pretrained_layer_map", {}) or {}
    for k, v in after.items():
        if k in src_sd and src_sd[k].shape == v.shape and torch.equal(v, src_sd[k]):
            matched_tensors += 1
            matched_elems += int(v.numel())
            continue
        # Check explicit remapped layer source
        parts = k.split(".", 2)
        if len(parts) == 3 and parts[0] == "model" and parts[1].isdigit():
            t_idx = int(parts[1])
            s_idx = layer_map.get(t_idx, layer_map.get(str(t_idx)))
            if s_idx is not None and int(s_idx) >= 0:
                src_k = f"model.{int(s_idx)}.{parts[2]}"
                if src_k in src_sd and src_sd[src_k].shape == v.shape and torch.equal(v, src_sd[src_k]):
                    matched_tensors += 1
                    matched_elems += int(v.numel())

    return {
        "transferred_tensors": matched_tensors,
        "total_tensors": total_tensors,
        "tensor_ratio": round(100.0 * matched_tensors / max(total_tensors, 1), 2),
        "transferred_elems": matched_elems,
        "total_elems": total_elems,
        "elem_ratio": round(100.0 * matched_elems / max(total_elems, 1), 2),
        "src_weights_count": len(src_sd),
    }


def profile_models(model_names: list[str], imgsz: int = 640, pretrained_path: Path = PRETRAINED) -> list[dict]:
    """Build, check forward/backward/fuse, measure Params, GFLOPs, and Pretrained Transfer %."""
    records = []
    for name in model_names:
        yaml_rel = MODEL_CATALOG[name]
        yaml_path = ROOT / yaml_rel
        model = SegmentationModel(str(yaml_path), ch=3, nc=1, verbose=False)
        transfer = compute_transfer_stats(model, pretrained_path)
        n_layers = len(model.model)
        n_params = sum(p.numel() for p in model.parameters())
        gflops = measure_exact_gflops(model, imgsz=imgsz)

        # Quick train/eval forward smoke test at 128x128
        model.train()
        x_s = torch.randn(1, 3, 128, 128)
        out_train = model(x_s)
        assert isinstance(out_train, dict) and "proto" in out_train
        model.eval()
        with torch.no_grad():
            out_eval = model(x_s)
            assert isinstance(out_eval, tuple) and len(out_eval) == 2

        rec = {
            "name": name,
            "yaml": yaml_rel,
            "layers": n_layers,
            "params_M": round(n_params / 1e6, 3),
            "params_raw": int(n_params),
            "gflops_640": round(gflops, 3),
            "transferred_tensors": transfer["transferred_tensors"],
            "total_tensors": transfer["total_tensors"],
            "tensor_transfer_pct": transfer["tensor_ratio"],
            "elem_transfer_pct": transfer["elem_ratio"],
            "head_type": type(model.model[-1]).__name__,
        }
        records.append(rec)
        print(
            f"[{name:<28}] layers={n_layers:>2} | params={rec['params_M']:.3f}M | "
            f"GFLOPs@640={rec['gflops_640']:.3f} | PretrainElem={rec['elem_transfer_pct']:5.1f}% "
            f"({rec['transferred_tensors']}/{rec['total_tensors']}) | head={rec['head_type']}"
        )
    return records


def train_suite(args: argparse.Namespace, model_names: list[str]) -> None:
    """Sequentially train models on server using unified whole-image 640 protocol."""
    from ultralytics import YOLO

    project_dir = Path(args.project)
    project_dir.mkdir(parents=True, exist_ok=True)
    for name in model_names:
        yaml_path = ROOT / MODEL_CATALOG[name]
        run_dir = project_dir / name
        if (run_dir / "weights" / "last.pt").is_file() and (run_dir / "results.csv").is_file():
            print(f"[SKIP] {name} already exists in {run_dir}")
            continue
        model = YOLO(str(yaml_path))
        if Path(args.pretrained).is_file():
            model.load(str(args.pretrained))
        model.train(
            data=str(args.data),
            epochs=1 if args.smoke else int(args.epochs),
            imgsz=int(args.imgsz),
            batch=2 if args.smoke else int(args.batch),
            workers=0 if args.smoke else int(args.workers),
            device=args.device,
            amp=AMP,
            cache=False if args.smoke else CACHE,
            seed=SEED,
            project=str(project_dir),
            name=name,
            exist_ok=True,
        )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=SUITE, choices=list(SUITES.keys()))
    parser.add_argument("--models", nargs="+", choices=list(MODEL_CATALOG.keys()))
    parser.add_argument("--data", default=DATA)
    parser.add_argument("--project", default=PROJECT)
    parser.add_argument("--pretrained", default=str(PRETRAINED))
    parser.add_argument("--device", default=DEVICE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--imgsz", type=int, default=IMGSZ)
    parser.add_argument("--batch", type=int, default=BATCH)
    parser.add_argument("--workers", type=int, default=WORKERS)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--ai-advise", action="store_true", help="Call DeepSeek & Intern AI advisor after profiling/training")
    parser.add_argument("--api-base", default=os.environ.get("CITRUS_AI_API_BASE", "https://discovery-api.intern-ai.org.cn/v1"))
    parser.add_argument("--api-key", default=os.environ.get("CITRUS_AI_API_KEY", ""))
    return parser.parse_args(argv)


def _call_openai_chat(api_base: str, api_key: str, model_id: str, system_prompt: str, user_prompt: str) -> tuple[str, str]:
    """Query OpenAI-compatible chat completions endpoint with safe fallback handling."""
    url = f"{api_base.rstrip('/')}/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "CitrusAPEX-Advisor/1.0",
    }
    candidates = [model_id, "intern-s2"] if model_id == "intern-decision-4b" else [model_id]
    for candidate in candidates:
        payload = {
            "model": candidate,
            "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
            "temperature": 0.2,
            "max_tokens": 2000,
        }
        req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                msg = data["choices"][0]["message"]
                return candidate, (msg.get("content") or msg.get("reasoning_content") or "").strip()
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="replace")
            if e.code == 404 and "does not support openai_chat protocol" in body and candidate != candidates[-1]:
                continue
            return candidate, f"HTTPError {e.code}: {body[:300]}"
        except Exception as e:
            return candidate, f"Error: {e}"
    return model_id, "No available chat model responded."


def _collect_training_summaries(project_dir: Path, model_names: list[str]) -> list[dict]:
    """Collect best validation metrics from existing results.csv files without fabricating missing runs."""
    summaries = []
    for name in model_names:
        csv_file = project_dir / name / "results.csv"
        if not csv_file.is_file():
            continue
        with csv_file.open("r", encoding="utf-8-sig") as f:
            rows = list(csv.DictReader(f))
        if not rows:
            continue
        best_row = max(rows, key=lambda r: float(r.get("metrics/mAP50-95(M)", 0.0) or 0.0))
        summaries.append(
            {
                "model": name,
                "epochs_completed": len(rows),
                "best_mask_mAP50": best_row.get("metrics/mAP50(M)", ""),
                "best_mask_mAP50_95": best_row.get("metrics/mAP50-95(M)", ""),
                "best_box_mAP50_95": best_row.get("metrics/mAP50-95(B)", ""),
            }
        )
    return summaries


def run_ai_advisor(args: argparse.Namespace, records: list[dict], model_names: list[str], out_dir: Path) -> None:
    """Invoke DeepSeek-V4.1-Flash & Intern Decision/S2 to critique profiles and finished training results."""
    api_key = (args.api_key or "").strip()
    if not api_key:
        print("[AI-ADVISE] Skipped: set CITRUS_AI_API_KEY env var or pass --api-key to enable live AI advisory.")
        return

    train_summaries = _collect_training_summaries(Path(args.project), model_names)
    context_payload = {
        "static_profiles": records,
        "training_results": train_summaries or "No completed results.csv on this machine yet.",
    }
    prompt = (
        "请基于以下 APEX 绿柑橘全图(640x640)实例分割模型静态/训练实测数据，给出下一步消融验证与超参决策建议：\n"
        + json.dumps(context_payload, indent=2, ensure_ascii=False)
    )
    ds_model, ds_reply = _call_openai_chat(
        args.api_base,
        api_key,
        "deepseek-v4.1-flash-1017",
        "你是深度学习计算机视觉实例分割架构专家，请基于实测指标给出精炼决策。",
        prompt,
    )
    in_model, in_reply = _call_openai_chat(
        args.api_base,
        api_key,
        "intern-decision-4b",
        "你是定量实验决策引擎，请给出训练排序、消融优先级与超参建议。",
        prompt,
    )
    report_path = out_dir / "AI_DECISION_EMPOWERMENT_REPORT.md"
    report_text = (
        "# APEX Live AI Decision & Architecture Advisory\n\n"
        f"## 1. {ds_model} Output\n\n{ds_reply}\n\n"
        f"## 2. {in_model} Output\n\n{in_reply}\n"
    )
    report_path.write_text(report_text, encoding="utf-8")
    print(f"[AI-ADVISE] Saved live dual-model report to {report_path}")


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    model_names = args.models if args.models else SUITES[args.suite]

    print(f"=== Profiling {len(model_names)} models @ imgsz={args.imgsz} ===")
    records = profile_models(model_names, imgsz=args.imgsz, pretrained_path=Path(args.pretrained))

    out_dir = ROOT / "docs" / "APEX_8_NETWORK_DESIGN_20261113"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "MODEL_PROFILE_SUMMARY.json").write_text(
        json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    csv_path = out_dir / "MODEL_PROFILE_SUMMARY.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    print(f"Saved profile summary to {csv_path}")

    if not args.check_only:
        train_suite(args, model_names)
    if args.ai_advise:
        run_ai_advisor(args, records, model_names, out_dir)


if __name__ == "__main__":
    main()
