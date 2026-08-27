"""Re-evaluate citrus checkpoints and separate measured PR behavior from COCO sentinels.

The official Ultralytics plots and metrics are left unchanged. This companion
diagnostic marks the maximum recall supported by the predictions retained by the
validator, reports precision at selected recalls only inside that range, and
records the best-F1 confidence threshold.

Example:
    python diagnose_citrus_pr.py \
        --weights 1_results/RUN_A/weights/best.pt 1_results/RUN_B/weights/best.pt \
        --data /data/orange_grouped/data.yaml --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path(r"E:\mastercode\data\orange_yolo_grouped_dedup_20260820\data.yaml")
DEFAULT_OUTPUT = ROOT / "1_results" / "_pr_diagnostic" / "manual"


def parse_args() -> argparse.Namespace:
    """Parse validation and output arguments."""
    parser = argparse.ArgumentParser(description="Diagnose the non-sentinel portion of citrus mask PR curves.")
    parser.add_argument("--weights", type=Path, nargs="+", required=True)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--conf-floor",
        type=float,
        default=0.001,
        help="Lowest confidence retained by validation; Rmax is conditional on this floor.",
    )
    return parser.parse_args()


def safe_name(path: Path) -> str:
    """Derive a stable output name from a conventional run/weights/checkpoint path."""
    candidate = path.parent.parent.name if path.parent.name == "weights" else path.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", candidate).strip("_") or "checkpoint"


def scalar_dict(values: dict[str, Any]) -> dict[str, float]:
    """Convert an Ultralytics results dictionary to JSON-safe floats."""
    converted = {}
    for key, value in values.items():
        array = np.asarray(value)
        if array.size == 1:
            converted[key] = float(array.reshape(-1)[0])
    return converted


def summarize_mask_curves(metrics: Any, conf_floor: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Summarize mask curves without treating the COCO endpoint sentinel as an observation."""
    segment = metrics.seg
    confidence = np.asarray(segment.px, dtype=float)
    precision_conf = np.asarray(segment.p_curve, dtype=float).mean(axis=0)
    recall_conf = np.asarray(segment.r_curve, dtype=float).mean(axis=0)
    f1_conf = np.asarray(segment.f1_curve, dtype=float).mean(axis=0)
    recall_grid = np.asarray(segment.px, dtype=float)
    precision_envelope = np.asarray(segment.prec_values, dtype=float).mean(axis=0)

    best_index = int(np.nanargmax(f1_conf))
    recall_max = float(np.nanmax(recall_conf))
    summary: dict[str, Any] = {
        **scalar_dict(metrics.results_dict),
        "validator_conf_floor": float(conf_floor),
        "mask_recall_ceiling": recall_max,
        "mask_best_f1": float(f1_conf[best_index]),
        "mask_best_f1_confidence": float(confidence[best_index]),
        "mask_precision_at_best_f1": float(precision_conf[best_index]),
        "mask_recall_at_best_f1": float(recall_conf[best_index]),
    }
    for target in (0.80, 0.82, 0.84):
        key = f"mask_precision_at_recall_{target:.2f}"
        summary[key] = float(np.interp(target, recall_grid, precision_envelope)) if target <= recall_max else None

    curve_rows = []
    for recall, precision in zip(recall_grid, precision_envelope):
        curve_rows.append(
            {
                "recall": float(recall),
                "precision_envelope": float(precision),
                "within_supported_recall_range": bool(recall <= recall_max + 1e-12),
            }
        )
    return summary, curve_rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write dictionaries to a UTF-8 CSV with deterministic columns."""
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_diagnostic(path: Path, curve_rows: list[dict[str, Any]], recall_max: float, title: str) -> None:
    """Plot the official envelope while visually separating its unsupported sentinel extension."""
    import matplotlib.pyplot as plt

    recall = np.array([row["recall"] for row in curve_rows])
    precision = np.array([row["precision_envelope"] for row in curve_rows])
    supported = recall <= recall_max + 1e-12
    figure, axis = plt.subplots(figsize=(7.2, 5.4), constrained_layout=True)
    axis.plot(recall, precision, color="#B8B8B8", linewidth=1.6, label="Official COCO envelope (incl. sentinel)")
    axis.plot(recall[supported], precision[supported], color="#0072B2", linewidth=2.6, label="Supported recall range")
    axis.axvline(recall_max, color="#D55E00", linestyle="--", linewidth=1.7, label=f"Recall ceiling={recall_max:.3f}")
    axis.scatter([recall_max], [np.interp(recall_max, recall, precision)], color="#D55E00", s=35, zorder=3)
    axis.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="Recall", ylabel="Precision", title=title)
    axis.grid(alpha=0.2)
    axis.legend(loc="lower left", frameon=True)
    figure.savefig(path, dpi=220)
    plt.close(figure)


def main() -> None:
    """Run official validation, then emit companion non-sentinel diagnostics."""
    args = parse_args()
    data = args.data.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not data.is_file():
        raise FileNotFoundError(f"Dataset YAML not found: {data}")
    if not 0.0 <= args.conf_floor < 1.0:
        raise ValueError("--conf-floor must be in [0, 1).")
    output.mkdir(parents=True, exist_ok=True)

    from ultralytics import YOLO

    summaries = []
    used_names: set[str] = set()
    for weight_arg in args.weights:
        weights = weight_arg.expanduser().resolve()
        if not weights.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {weights}")
        name = safe_name(weights)
        if name in used_names:
            name = f"{name}_{len(used_names):02d}"
        used_names.add(name)
        run_dir = output / name
        metrics = YOLO(str(weights)).val(
            data=str(data),
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            workers=args.workers,
            conf=args.conf_floor,
            plots=False,
            save_json=False,
            project=str(run_dir),
            name="validator",
            exist_ok=True,
        )
        summary, curve_rows = summarize_mask_curves(metrics, args.conf_floor)
        summary.update({"name": name, "weights": str(weights), "data": str(data), "split": args.split})
        summaries.append(summary)
        run_dir.mkdir(parents=True, exist_ok=True)
        write_csv(run_dir / "mask_pr_envelope.csv", curve_rows)
        try:
            plot_diagnostic(
                run_dir / "MaskPR_supported_range.png",
                curve_rows,
                summary["mask_recall_ceiling"],
                f"{name}: mask PR diagnostic",
            )
        except ImportError as error:
            summary["plot_warning"] = str(error)
            print(f"WARNING: diagnostic CSV/JSON saved, but Matplotlib plot was skipped: {error}")
        (run_dir / "pr_diagnostic.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))

    write_csv(output / "pr_summary.csv", summaries)
    (output / "pr_summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
