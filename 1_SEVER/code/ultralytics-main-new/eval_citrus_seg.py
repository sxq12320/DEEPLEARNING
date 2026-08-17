"""Standardised evaluation for the immature-citrus seg experiments — one口径 for every run.

Runs model.val() on the requested split(s) and records the paper-table metrics:
mask mAP50 / mAP50-95 (primary), box mAP50-95, precision, recall, plus Params / GFLOPs / FPS.
Every call appends one row per split to results_summary.csv so E0..E4 line up in a single table.

Examples:
    python eval_citrus_seg.py --weights 1_results/ORANGE_WUXI_SEG/E0_yolo11n_seg_baseline_941/weights/best.pt
    python eval_citrus_seg.py --weights .../best.pt --splits test        # final reporting only
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

from ultralytics import YOLO

try:  # stable ultralytics utils for size/compute
    from ultralytics.utils.torch_utils import get_flops, get_num_params
except Exception:  # pragma: no cover - fork API drift fallback
    get_num_params = get_flops = None

DATA = r"/data/sxq/datasets/orange_yolo/data.yaml"
SUMMARY = r"/data/sxq/results/000_anyothers/results_summary.csv"
COLUMNS = [
    "name", "split", "mask_mAP50", "mask_mAP50_95", "box_mAP50", "box_mAP50_95",
    "precision", "recall", "params_M", "GFLOPs", "FPS", "infer_ms",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate a citrus YOLO-seg run with a fixed metric protocol.")
    p.add_argument("--weights", required=True, help="Path to a run's weights/best.pt")
    p.add_argument("--data", default=DATA)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--splits", default="val,test", help="Comma list: val,test")
    p.add_argument("--name", default=None, help="Row label; defaults to the run dir name.")
    return p.parse_args()


def model_size(model: YOLO, imgsz: int) -> tuple[float, float]:
    """(Params in millions, GFLOPs). Robust to minor fork API differences."""
    params = gflops = 0.0
    if get_num_params is not None:
        params = get_num_params(model.model) / 1e6
        try:
            gflops = get_flops(model.model, imgsz)
        except Exception:
            gflops = 0.0
    return round(params, 3), round(gflops, 2)


def grab_metrics(m) -> dict:
    seg, box = m.seg, m.box  # SegmentMetrics: mask + box sub-metrics
    infer_ms = float(getattr(m, "speed", {}).get("inference", 0.0) or 0.0)
    return {
        "mask_mAP50": round(float(seg.map50), 4),
        "mask_mAP50_95": round(float(seg.map), 4),
        "box_mAP50": round(float(box.map50), 4),
        "box_mAP50_95": round(float(box.map), 4),
        "precision": round(float(seg.mp), 4),
        "recall": round(float(seg.mr), 4),
        "infer_ms": round(infer_ms, 3),
        "FPS": round(1000.0 / infer_ms, 1) if infer_ms > 0 else 0.0,
    }


def append_row(row: dict) -> None:
    path = Path(SUMMARY)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in COLUMNS})


def main() -> None:
    args = parse_args()
    name = args.name or Path(args.weights).parents[1].name  # <run>/weights/best.pt -> <run>
    model = YOLO(args.weights)
    params_m, gflops = model_size(model, args.imgsz)

    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        m = model.val(data=args.data, split=split, imgsz=args.imgsz, device=args.device, verbose=False)
        row = {"name": name, "split": split, "params_M": params_m, "GFLOPs": gflops, **grab_metrics(m)}
        append_row(row)
        print(f"[{name}/{split}] maskAP50={row['mask_mAP50']} maskAP50-95={row['mask_mAP50_95']} "
              f"boxAP50-95={row['box_mAP50_95']} P={row['precision']} R={row['recall']} "
              f"| {params_m}M {gflops}GFLOPs {row['FPS']}FPS")

    print(f"\n-> appended to {SUMMARY}")


if __name__ == "__main__":
    main()
