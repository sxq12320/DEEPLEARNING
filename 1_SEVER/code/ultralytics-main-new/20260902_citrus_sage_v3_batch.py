"""Sequential fixed-protocol runner for the evidence-grounded SAGE-v3 series.

Examples:
    python 20260902_citrus_sage_v3_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite smoke --epochs 3
    python 20260902_citrus_sage_v3_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite screen --epochs 50
    python 20260902_citrus_sage_v3_batch.py --data /data/sxq/datasets/orange_yolo/data.yaml --suite all --epochs 50
"""

from __future__ import annotations

import argparse
import gc
import json
import random
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

from citrus_protocol import fixed_train_args, validate_locked_runtime, write_protocol_snapshot


ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "SAGE_series"
FIXED_TRAIN = fixed_train_args()


@dataclass(frozen=True)
class Experiment:
    """One causally interpretable SAGE-v3 experiment."""

    name: str
    yaml: str
    hypothesis: str
    losses: tuple[tuple[str, float], ...] = ()


SHARED_TOPOLOGY = (("citrus_topology", 0.10), ("citrus_boundary", 0.10), ("citrus_query", 0.03))
FULL_OCCLUSION = (*SHARED_TOPOLOGY, ("citrus_concavity", 0.03), ("citrus_exclusive", 0.02))
CONTROL = Experiment("SAGE10_official_control", "SAGE10_official_control.yaml", "exact fixed-protocol control")
EXPERIMENTS = (
    Experiment(
        "SAGE20_shape_context_backbone",
        "SAGE20_shape_context_backbone.yaml",
        "P4/P5 axial shape-context backbone only",
    ),
    Experiment(
        "SAGE21_innovation_pyramid",
        "SAGE21_innovation_pyramid.yaml",
        "measurement-prediction innovation pyramid under stock loss",
    ),
    Experiment(
        "SAGE22_contrast_topology",
        "SAGE22_contrast_topology.yaml",
        "innovation pyramid plus explicit context/interior/boundary/separator supervision",
        SHARED_TOPOLOGY,
    ),
    Experiment(
        "SAGE23_joint_core_v3",
        "SAGE23_joint_core_v3.yaml",
        "primary shape-context backbone and supervised innovation pyramid",
        SHARED_TOPOLOGY,
    ),
    Experiment(
        "SAGE24_style_robust",
        "SAGE24_style_robust.yaml",
        "joint core plus training-only feature-statistics exchange",
        SHARED_TOPOLOGY,
    ),
    Experiment(
        "SAGE25_quality_aligned",
        "SAGE25_quality_aligned.yaml",
        "joint core plus full Varifocal quality-aware classification",
        (*SHARED_TOPOLOGY, ("citrus_vfl", 1.0)),
    ),
    Experiment(
        "SAGE26_occlusion_topology",
        "SAGE26_occlusion_topology.yaml",
        "joint core plus concavity preservation and adjacent-instance exclusivity",
        FULL_OCCLUSION,
    ),
    Experiment(
        "SAGE27_joint_lite",
        "SAGE27_joint_lite.yaml",
        "narrow joint core for the accuracy-latency Pareto frontier",
        SHARED_TOPOLOGY,
    ),
)
ALL_EXPERIMENTS = (CONTROL, *EXPERIMENTS)


def parse_args() -> argparse.Namespace:
    """Parse the server-safe interface without Python 3.9-only argparse features."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Server-side dataset YAML selected by the user.")
    parser.add_argument(
        "--suite",
        choices=("smoke", "screen", "all", "control", "backbone", "fusion", "final"),
        default="screen",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch", type=int, default=FIXED_TRAIN["batch"])
    parser.add_argument("--imgsz", type=int, default=FIXED_TRAIN["imgsz"])
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=FIXED_TRAIN["workers"])
    parser.add_argument("--project", type=Path, default=None)
    parser.add_argument("--pretrained", type=Path, default=ROOT / "yolo11n-seg.pt")
    parser.add_argument("--seeds", default="42", help="Comma-separated seeds; formal finals use 42,43,44.")
    parser.add_argument("--only", default="", help="Comma-separated exact experiment names.")
    parser.add_argument("--cache", choices=("false", "disk", "ram"), default="false")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", help="Explicit paired AMP audit only.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false")
    parser.set_defaults(amp=FIXED_TRAIN["amp"])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def choose_experiments(args: argparse.Namespace) -> list[Experiment]:
    """Select a deterministic queue without silently rerunning the control."""
    if args.suite == "smoke":
        selected = [EXPERIMENTS[index] for index in (0, 1, 3)]
    elif args.suite == "screen":
        selected = [EXPERIMENTS[index] for index in (0, 1, 2, 3)]
    elif args.suite == "all":
        selected = list(EXPERIMENTS)
    elif args.suite == "control":
        selected = [CONTROL]
    elif args.suite == "backbone":
        selected = [EXPERIMENTS[0], EXPERIMENTS[4]]
    elif args.suite == "fusion":
        selected = [EXPERIMENTS[index] for index in (1, 2, 3)]
    else:
        selected = [EXPERIMENTS[3]]

    if args.only:
        requested = {item.strip() for item in args.only.split(",") if item.strip()}
        known = {experiment.name for experiment in ALL_EXPERIMENTS}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown experiments: {sorted(unknown)}; choices: {sorted(known)}")
        selected = [experiment for experiment in ALL_EXPERIMENTS if experiment.name in requested]
    return selected


def set_seed(seed: int) -> None:
    """Lock random sources before constructing each model."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def snapshot_data_yaml(source: Path, project: Path) -> Path:
    """Record the chosen data YAML without changing user paths or split membership."""
    content = source.read_text(encoding="utf-8")
    config = yaml.safe_load(content)
    if not isinstance(config, dict) or not {"train", "val", "names"}.issubset(config):
        raise ValueError(f"Invalid segmentation dataset YAML: {source}")
    target = project / "_protocol" / "dataset_source_snapshot.yaml"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def repository_state() -> dict[str, str | bool]:
    """Return the exact source revision and dirty-worktree state."""

    def run(*command: str) -> str:
        try:
            return subprocess.check_output(command, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"

    return {"commit": run("git", "rev-parse", "HEAD"), "dirty": bool(run("git", "status", "--porcelain"))}


def append_event(path: Path, event: dict) -> None:
    """Append a durable experiment event to the JSON-lines ledger."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def transfer_compatible_head(target, source_state: dict[str, torch.Tensor]) -> int:
    """Copy all shape-compatible official Segment tensors into a SAGE head."""
    target_head = target.model.model[-1]
    target_state = target_head.state_dict()
    compatible = {
        key: value for key, value in source_state.items() if key in target_state and target_state[key].shape == value.shape
    }
    target_head.load_state_dict(compatible, strict=False)
    return len(compatible)


def main() -> None:
    """Build or train SAGE-v3 models sequentially on one device."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    pretrained = args.pretrained.expanduser().resolve()
    default_project = ROOT / "1_results" / "SAGE_series" / f"CITRUS_SAGE_V3_{args.suite.upper()}_{args.epochs}EP"
    project = (args.project or default_project).expanduser().resolve()
    cache: bool | str = False if args.cache == "false" else args.cache
    deviations = validate_locked_runtime(
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        cache=cache,
        amp=args.amp,
        allow_amp_audit=args.amp,
    )
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not pretrained.is_file():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained}")
    if args.suite == "smoke" and not 1 <= args.epochs <= 3:
        raise ValueError("Smoke runs must use 1--3 epochs.")

    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]
    if not seeds:
        raise ValueError("At least one seed is required.")
    experiments = choose_experiments(args)

    from ultralytics import YOLO
    from ultralytics.nn.tasks import SegmentationModel
    from ultralytics.utils.torch_utils import get_flops

    if args.dry_run:
        for experiment in experiments:
            model = SegmentationModel(YAML_DIR / experiment.yaml, ch=3, nc=1, verbose=False).eval()
            parameters = sum(parameter.numel() for parameter in model.parameters())
            print(f"BUILD OK {experiment.name}: params={parameters:,}, GFLOPs@640={get_flops(model, 640):.3f}")
        return

    source = YOLO(str(pretrained), task="segment", verbose=False)
    source_head_state = {key: value.detach().cpu() for key, value in source.model.model[-1].state_dict().items()}
    del source
    data_snapshot = snapshot_data_yaml(data, project)
    ledger = project / "_protocol" / "ledger.jsonl"
    state = repository_state()
    protocol_path, protocol_hash = write_protocol_snapshot(
        project,
        {
            "series": "SAGE-v3",
            "architecture_goal": "shape-context backbone and topology-supervised cross-scale innovation correction",
            "operator_contract": "Conv/BN/SiLU, low-resolution axial DW conv, nearest resize, PixelUnshuffle, pooling and elementwise gates",
            "declared_protocol_deviations": deviations,
        },
    )

    for experiment in experiments:
        for seed in seeds:
            run_name = experiment.name if len(seeds) == 1 else f"{experiment.name}_seed{seed}"
            run_dir = project / run_name
            complete = (run_dir / "weights" / "best.pt").is_file() and (run_dir / "results.csv").is_file()
            if complete and args.skip_completed:
                print(f"SKIP completed: {run_name}")
                continue
            if run_dir.exists():
                raise FileExistsError(f"Refusing to overwrite existing run: {run_dir}")

            event = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "started",
                "run_name": run_name,
                "experiment": asdict(experiment),
                "seed": seed,
                "epochs": args.epochs,
                "data": str(data),
                "dataset_yaml_snapshot": str(data_snapshot),
                "pretrained": str(pretrained),
                "repository": state,
                "formal_protocol": str(protocol_path),
                "protocol_sha256": protocol_hash,
                "protocol_deviations": deviations,
            }
            append_event(ledger, event)
            model = None
            try:
                set_seed(seed)
                model = YOLO(str(YAML_DIR / experiment.yaml), task="segment", verbose=False).load(str(pretrained))
                event["compatible_head_tensors_transferred"] = transfer_compatible_head(model, source_head_state)
                training = fixed_train_args()
                training.update(
                    data=str(data),
                    project=str(project),
                    name=run_name,
                    epochs=args.epochs,
                    device=args.device,
                    seed=seed,
                    exist_ok=False,
                )
                training.update(dict(experiment.losses))
                if deviations:
                    training["amp"] = args.amp
                model.train(**training)
                append_event(ledger, {**event, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "completed"})
            except Exception as error:
                append_event(
                    ledger,
                    {**event, "time": time.strftime("%Y-%m-%d %H:%M:%S"), "status": "failed", "error": repr(error)},
                )
                if args.fail_fast:
                    raise
                print(f"FAILED {run_name}: {error!r}")
            finally:
                del model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
