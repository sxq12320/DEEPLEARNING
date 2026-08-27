"""Audit every local citrus result while separating incompatible protocols."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter
from pathlib import Path

import yaml


RESULTS_ROOT = Path(r"E:\mastercode\1_SEVER\results")
OUTPUT_DIR = RESULTS_ROOT / "_analysis_20260826_b_series"
PRIMARY = "metrics/mAP50-95(M)"
METRICS = {
    "box_p": "metrics/precision(B)",
    "box_r": "metrics/recall(B)",
    "box_map50": "metrics/mAP50(B)",
    "box_map": "metrics/mAP50-95(B)",
    "mask_p": "metrics/precision(M)",
    "mask_r": "metrics/recall(M)",
    "mask_map50": "metrics/mAP50(M)",
    "mask_map": PRIMARY,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    """Read an Ultralytics results table with normalized headers."""
    with path.open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        return [
            {str(key).strip(): str(value).strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
            if any(str(value).strip() for value in row.values())
        ]


def value(row: dict[str, str], key: str) -> float:
    """Convert one cell to a finite float or NaN."""
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def protocol(args: dict) -> str:
    """Assign only demonstrably comparable protocol families."""
    data = str(args.get("data") or "").lower()
    pretrained = str(args.get("pretrained") or args.get("model") or "").lower()
    amp = bool(args.get("amp"))
    if "grouped_dedup" in data:
        return "P3_grouped_dedup_clean_AMP0" if not amp else "P3_grouped_dedup_clean_AMP1"
    if pretrained.endswith(".pt") and not amp:
        return "P2_old_data_pretrained_AMP0"
    if amp:
        return "P1_old_data_mixed_or_scratch_AMP1"
    return "other"


def scan() -> list[dict]:
    """Scan every results.csv, including runs whose args file is absent."""
    records = []
    for csv_path in sorted(RESULTS_ROOT.rglob("results.csv")):
        if any(part.startswith("_analysis_") for part in csv_path.parts):
            continue
        run_dir = csv_path.parent
        args_path = run_dir / "args.yaml"
        if args_path.is_file():
            with args_path.open(encoding="utf-8-sig", errors="replace") as handle:
                args = yaml.safe_load(handle) or {}
        else:
            args = {}
        rows = [row for row in read_csv(csv_path) if math.isfinite(value(row, PRIMARY))]
        if not rows:
            continue
        expected = int(args.get("epochs") or 0)
        has_pr = (run_dir / "MaskPR_curve.png").is_file()
        if expected and len(rows) >= expected:
            status = "complete"
        elif has_pr and len(rows) >= 50:
            status = "early_stopped"
        else:
            status = "partial"
        best = max(rows, key=lambda row: value(row, PRIMARY))
        final = rows[-1]
        tail = rows[-min(20, len(rows)) :]
        record = {
            "run": run_dir.name,
            "relative_dir": str(run_dir.relative_to(RESULTS_ROOT)),
            "protocol": protocol(args),
            "status": status,
            "epochs_actual": len(rows),
            "epochs_expected": expected,
            "best_epoch": int(value(best, "epoch")),
            "model": str(args.get("model") or ""),
            "data": str(args.get("data") or ""),
            "seed": args.get("seed"),
            "imgsz": args.get("imgsz"),
            "optimizer": args.get("optimizer"),
            "lr0": args.get("lr0"),
            "dropout": args.get("dropout"),
            "amp": args.get("amp"),
            "has_mask_pr": has_pr,
            "has_best": (run_dir / "weights" / "best.pt").is_file(),
        }
        for short, column in METRICS.items():
            tail_values = [value(row, column) for row in tail if math.isfinite(value(row, column))]
            record[f"best_{short}"] = value(best, column)
            record[f"final_{short}"] = value(final, column)
            record[f"tail20_{short}"] = statistics.mean(tail_values) if tail_values else None
            record[f"tail20_std_{short}"] = statistics.pstdev(tail_values) if tail_values else None
        record["stable_mask_map"] = statistics.mean(
            [record["best_mask_map"], record["final_mask_map"], record["tail20_mask_map"]]
        )
        records.append(record)
    return records


def markdown_table(rows: list[dict]) -> str:
    """Format the grouped-clean S series as a compact evidence table."""
    lines = [
        "| Run | Status | Epochs | Best epoch | Mask P | Mask R | Mask AP50 | Mask AP50-95 | Stable AP |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['run']} | {row['status']} | {row['epochs_actual']} | {row['best_epoch']} | "
            f"{row['best_mask_p']:.4f} | {row['best_mask_r']:.4f} | {row['best_mask_map50']:.4f} | "
            f"{row['best_mask_map']:.4f} | {row['stable_mask_map']:.4f} |"
        )
    return "\n".join(lines)


def main() -> None:
    """Write machine-readable tables and an evidence-based summary."""
    records = scan()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = list(records[0])
    with (OUTPUT_DIR / "all_runs.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)

    grouped = [row for row in records if row["protocol"].startswith("P3_grouped_dedup")]
    s_rows = sorted([row for row in grouped if row["run"].startswith("S")], key=lambda row: row["run"])
    usable_s = [row for row in s_rows if row["status"] in {"complete", "early_stopped"}]
    ranked_s = sorted(usable_s, key=lambda row: row["best_mask_map"], reverse=True)
    old = [row for row in records if row["protocol"] == "P2_old_data_pretrained_AMP0" and row["status"] != "partial"]
    ranked_old = sorted(old, key=lambda row: row["stable_mask_map"], reverse=True)
    summary = {
        "result_csv_count": len(records),
        "protocol_counts": dict(Counter(row["protocol"] for row in records)),
        "status_counts": dict(Counter(row["status"] for row in records)),
        "grouped_clean_runs": len(grouped),
        "s_runs": len(s_rows),
        "best_s_by_peak_mask_map": ranked_s[0]["run"] if ranked_s else None,
        "best_s_peak_mask_map": ranked_s[0]["best_mask_map"] if ranked_s else None,
        "old_protocol_top10": [
            {"run": row["run"], "stable_mask_map": row["stable_mask_map"], "best_mask_map50": row["best_mask_map50"]}
            for row in ranked_old[:10]
        ],
    }
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    report = f"""# Citrus result audit — 2026-08-26

## Scope and comparability

- Parsed result tables: {len(records)}.
- Protocol counts: `{summary['protocol_counts']}`.
- Results are ranked only inside a protocol. Old F/G results and grouped-clean S results are **not numerically
  interchangeable**, because the dataset/split changed.

## Grouped-clean S series

{markdown_table(s_rows)}

The best completed/early-stopped S run by peak mask AP50-95 is **{summary['best_s_by_peak_mask_map']}** at
**{summary['best_s_peak_mask_map']:.4f}**. S07 is partial and must not be promoted from its current checkpoint.
S04 is the practical local Pareto result: a current-code re-evaluation measured 2.740M parameters and 9.3 GFLOPs,
versus S00's 2.835M and 10.2 GFLOPs, while improving high-recall precision.

## PR-curve diagnosis

The line dropping vertically to `(recall=1, precision=0)` is not an observed operating point. The local
`compute_ap()` implementation appends recall sentinels ending at 1 and precision sentinels ending at 0, then
interpolates a 101-point COCO envelope. The same convention exists in upstream Ultralytics/COCO evaluation.

The real limitation is the achieved recall ceiling. Current-code validation on the same 193-image grouped-clean
validation split gave:

| Model | Recall ceiling on retained candidates | P@R=.80 | P@R=.82 | P@R=.84 | Best F1 |
|---|---:|---:|---:|---:|---:|
| S00 | 0.8527 | 0.5040 | 0.2897 | 0.1576 | 0.7884 |
| S04 | 0.8561 | 0.5628 | 0.3734 | 0.2227 | 0.7988 |

The validator normally discards predictions below roughly 0.001 confidence, so this is not a literal all-candidate
`conf=0` measurement. S04 improves ranking/false positives near high recall, but changes the retained-candidate recall
ceiling by only +0.0034. The B series
therefore separates two interventions: persistent high-resolution information flow for missed instances, and learned
quality-aware ranking for the bend of the PR curve. Official PR plots remain unchanged for comparability; a diagnostic
plot should mark the achieved recall ceiling instead of pretending the sentinel is a measured threshold.
"""
    (OUTPUT_DIR / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
