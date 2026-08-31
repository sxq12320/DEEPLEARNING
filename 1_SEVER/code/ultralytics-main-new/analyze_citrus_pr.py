"""Diagnose the real citrus PR bottleneck without misreading Ultralytics plot sentinels.

The standard Ultralytics/COCO PR plot pads the curve with zero precision after
the largest achieved recall.  This script leaves that standard evaluation
untouched and reports the supported recall range, operating threshold, and raw
box-confusion counts separately.

Example:
    python analyze_citrus_pr.py --weights /data/sxq/results/Light/run/weights/best.pt \
        --data /data/sxq/datasets/orange_yolo/data.yaml --device 0
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

import numpy as np

from ultralytics import YOLO
from ultralytics.models.yolo.segment import SegmentationValidator
from ultralytics.utils.metrics import smooth


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT / "1_results" / "PR_DIAGNOSTICS"


class DiagnosticSegmentationValidator(SegmentationValidator):
    """Retain matched prediction arrays that the stock validator clears after metric aggregation."""

    def get_stats(self) -> dict:
        """Process standard metrics and attach a copy of raw match arrays for operating-point analysis."""
        self.metrics.process(save_dir=self.save_dir, plot=self.args.plots, on_plot=self.on_plot)
        self.metrics.diagnostic_stats = {
            key: [np.array(value, copy=True) for value in values] for key, values in self.metrics.stats.items()
        }
        self.metrics.clear_stats()
        return self.metrics.results_dict


def parse_args() -> argparse.Namespace:
    """Parse validation and output options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", nargs="+", required=True, help="One or more best.pt paths.")
    parser.add_argument("--data", required=True, help="Exact dataset YAML used by the compared runs.")
    parser.add_argument("--split", default="val", choices=("train", "val", "test"))
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.001,
        help="Validation prefilter. Keep 0.001 for standard AP and the standard conf=0.25 box confusion matrix.",
    )
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU, not the confusion-matrix matching IoU.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Also regenerate Ultralytics plots and the conventional box CM (requires a working Matplotlib install).",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _curve_value_at_recall(precision_envelope: np.ndarray, recall: float) -> float:
    """Read mean interpolated precision at one recall grid point."""
    curve = np.asarray(precision_envelope, dtype=float)
    if curve.ndim == 1:
        curve = curve[None]
    index = int(np.clip(round(recall * (curve.shape[1] - 1)), 0, curve.shape[1] - 1))
    return float(curve[:, index].mean())


def summarize_mask_curves(seg) -> dict[str, float]:
    """Summarize mask recall support and the max-F1 operating point."""
    p_curve = np.asarray(seg.p_curve, dtype=float)
    r_curve = np.asarray(seg.r_curve, dtype=float)
    f1_curve = np.asarray(seg.f1_curve, dtype=float)
    confidence_grid = np.asarray(seg.px, dtype=float)
    precision_envelope = np.asarray(seg.prec_values, dtype=float)
    if p_curve.ndim == 1:
        p_curve, r_curve, f1_curve = p_curve[None], r_curve[None], f1_curve[None]

    mean_f1 = smooth(f1_curve.mean(0), 0.1)
    best_index = int(mean_f1.argmax())
    recall_ceiling = float(r_curve[:, 0].mean())
    supported_recall = max(recall_ceiling - 0.01, 0.0)
    return {
        "mask_recall_ceiling_at_val_prefilter": recall_ceiling,
        "mask_unmatched_fraction_at_val_prefilter": 1.0 - recall_ceiling,
        "mask_best_f1_conf": float(confidence_grid[best_index]),
        "mask_precision_at_best_f1": float(p_curve[:, best_index].mean()),
        "mask_recall_at_best_f1": float(r_curve[:, best_index].mean()),
        "mask_f1_at_best_f1": float(f1_curve[:, best_index].mean()),
        "mask_precision_at_recall_080": _curve_value_at_recall(precision_envelope, 0.80),
        "mask_precision_at_recall_085": _curve_value_at_recall(precision_envelope, 0.85),
        "mask_precision_at_recall_090": _curve_value_at_recall(precision_envelope, 0.90),
        "mask_precision_near_supported_recall": _curve_value_at_recall(precision_envelope, supported_recall),
    }


def summarize_raw_box_confusion(matrix: np.ndarray) -> dict[str, float | int]:
    """Convert an unnormalized detection confusion matrix into aggregate counts."""
    matrix = np.asarray(matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 2:
        raise ValueError(f"Expected square detection confusion matrix, got {matrix.shape}")
    class_count = matrix.shape[0] - 1
    true_positive = float(np.diag(matrix[:class_count, :class_count]).sum())
    false_positive = float(matrix[:class_count, :].sum() - true_positive)
    false_negative = float(matrix[:, :class_count].sum() - true_positive)
    precision = true_positive / max(true_positive + false_positive, 1.0)
    recall = true_positive / max(true_positive + false_negative, 1.0)
    return {
        "box_cm45_tp": int(round(true_positive)),
        "box_cm45_fp": int(round(false_positive)),
        "box_cm45_fn": int(round(false_negative)),
        "box_cm45_precision": precision,
        "box_cm45_recall": recall,
    }


def summarize_operating_counts(stats: dict[str, list[np.ndarray]], confidence: float = 0.25) -> dict[str, float | int]:
    """Compute raw box and mask TP/FP/FN at IoU 0.5 directly from validator matches."""
    arrays = {key: np.concatenate(value, axis=0) for key, value in stats.items() if len(value)}
    conf = np.asarray(arrays["conf"], dtype=float)
    keep = conf > confidence
    target_count = int(len(arrays["target_cls"]))
    row: dict[str, float | int] = {}
    for prefix, key in (("box_op50", "tp"), ("mask_op50", "tp_m")):
        matches = np.asarray(arrays[key], dtype=bool)
        true_positive = int(matches[keep, 0].sum())
        false_positive = int(keep.sum() - true_positive)
        false_negative = int(target_count - true_positive)
        row.update(
            {
                f"{prefix}_tp": true_positive,
                f"{prefix}_fp": false_positive,
                f"{prefix}_fn": false_negative,
                f"{prefix}_precision": true_positive / max(true_positive + false_positive, 1),
                f"{prefix}_recall": true_positive / max(true_positive + false_negative, 1),
            }
        )
    return row


def plot_supported_pr(seg, save_path: Path, name: str) -> None:
    """Plot only the recall region supported by detections; this is diagnostic, not an AP replacement."""
    import matplotlib.pyplot as plt

    recall_grid = np.asarray(seg.px, dtype=float)
    precision = np.asarray(seg.prec_values, dtype=float)
    if precision.ndim == 1:
        precision = precision[None]
    recall_ceiling = float(np.asarray(seg.r_curve, dtype=float)[:, 0].mean())
    supported = recall_grid < recall_ceiling
    if not supported.any():
        return
    fig, axis = plt.subplots(figsize=(7.2, 5.4))
    axis.plot(recall_grid[supported], precision.mean(0)[supported], color="#1756b3", linewidth=2.5)
    axis.axvline(recall_ceiling, color="#c7422f", linestyle="--", label=f"supported R max={recall_ceiling:.3f}")
    axis.set(xlabel="Recall", ylabel="Precision", xlim=(0, 1), ylim=(0, 1), title=f"{name}: supported Mask PR")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower left")
    axis.text(
        0.02,
        0.03,
        "Diagnostic view only; standard COCO AP/PR remains unchanged.",
        transform=axis.transAxes,
        fontsize=8,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=220)
    plt.close(fig)


def _safe_name(weights: Path) -> str:
    """Build a stable filesystem label from a run path."""
    run_name = weights.parents[1].name if len(weights.parents) > 1 and weights.parent.name == "weights" else weights.stem
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", run_name).strip("_") or "model"


def write_summary(rows: list[dict], output_dir: Path) -> None:
    """Write machine-readable CSV and JSON summaries."""
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "pr_diagnostics.csv"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "pr_diagnostics.json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, ensure_ascii=False, indent=2)


def main() -> None:
    """Validate each checkpoint and report PR support plus raw box-confusion counts."""
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for weights_text in args.weights:
        weights = Path(weights_text).expanduser().resolve()
        name = _safe_name(weights)
        validation_dir = args.output_dir / name
        model = YOLO(str(weights), task="segment")
        metrics = model.val(
            validator=DiagnosticSegmentationValidator,
            data=args.data,
            split=args.split,
            imgsz=args.imgsz,
            batch=args.batch,
            workers=args.workers,
            device=args.device,
            conf=args.conf,
            iou=args.iou,
            max_det=args.max_det,
            plots=args.plots,
            project=str(args.output_dir),
            name=name,
            exist_ok=True,
            verbose=False,
        )
        operating = summarize_operating_counts(metrics.diagnostic_stats, confidence=0.25)
        confusion = (
            summarize_raw_box_confusion(metrics.confusion_matrix.matrix)
            if args.plots
            else {
                "box_cm45_tp": None,
                "box_cm45_fp": None,
                "box_cm45_fn": None,
                "box_cm45_precision": None,
                "box_cm45_recall": None,
            }
        )
        curves = summarize_mask_curves(metrics.seg)
        row = {
            "name": name,
            "weights": str(weights),
            "data": str(Path(args.data)),
            "split": args.split,
            "imgsz": args.imgsz,
            "val_prefilter_conf": args.conf,
            "operating_conf": 0.25,
            "operating_match_iou": 0.50,
            "box_cm_conf": 0.25 if args.conf == 0.001 else args.conf,
            "box_cm_match_iou": 0.45,
            "mask_map50": float(metrics.seg.map50),
            "mask_map50_95": float(metrics.seg.map),
            **curves,
            **operating,
            **confusion,
        }
        rows.append(row)
        if args.plots:
            validation_dir.mkdir(parents=True, exist_ok=True)
            plot_supported_pr(metrics.seg, validation_dir / "MaskPR_supported_diagnostic.png", name)
        print(
            f"[{name}] Mask AP50-95={row['mask_map50_95']:.4f}, "
            f"R_ceiling={row['mask_recall_ceiling_at_val_prefilter']:.4f}, "
            f"bestF1@conf={row['mask_best_f1_conf']:.3f}; "
            f"Mask@conf.25/IoU.50: TP={row['mask_op50_tp']} FP={row['mask_op50_fp']} FN={row['mask_op50_fn']}"
        )

    write_summary(rows, args.output_dir)
    print(f"Wrote diagnostics to {args.output_dir}")
    print("The standard zero PR tail is an AP sentinel; judge improvement by R_ceiling, FP/FN, and supported precision.")


if __name__ == "__main__":
    main()
