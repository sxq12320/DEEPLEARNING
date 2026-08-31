"""Summarize CitrusTopo-Seg batch results with noise-aware deltas."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


METRICS = {
    "mask_map50": ("metrics/mAP50(M)", "metrics/mAP50(Mask)"),
    "mask_map": ("metrics/mAP50-95(M)", "metrics/mAP50-95(Mask)"),
    "mask_precision": ("metrics/precision(M)", "metrics/precision(Mask)"),
    "mask_recall": ("metrics/recall(M)", "metrics/recall(Mask)"),
    "box_map50": ("metrics/mAP50(B)", "metrics/mAP50(Box)"),
    "box_map": ("metrics/mAP50-95(B)", "metrics/mAP50-95(Box)"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a provisional CitrusTopo result table.")
    parser.add_argument("--project", type=Path, default=Path("1_results/L_series/grouped_clean_300ep"))
    parser.add_argument("--tail", type=int, default=10, help="Epochs used for the stable-tail mean.")
    return parser.parse_args()


def pick(row: dict[str, str], candidates: tuple[str, ...]) -> float:
    for key in candidates:
        value = row.get(key, "")
        if value not in ("", None):
            try:
                return float(value)
            except ValueError:
                continue
    return float("nan")


def read_run(run_dir: Path, tail: int) -> dict | None:
    result_path = run_dir / "results.csv"
    if not result_path.exists():
        return None
    with result_path.open(encoding="utf-8-sig", errors="ignore") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return None
    enriched = []
    for row in rows:
        values = {name: pick(row, candidates) for name, candidates in METRICS.items()}
        if math.isfinite(values["mask_map"]):
            enriched.append((row, values))
    if not enriched:
        return None
    best_row, best = max(enriched, key=lambda item: item[1]["mask_map"])
    tail_values = [values["mask_map"] for _, values in enriched[-tail:] if math.isfinite(values["mask_map"])]
    return {
        "name": run_dir.name,
        "epochs": len(rows),
        "best_epoch": int(float(best_row.get("epoch", 0))) + 1,
        **best,
        "stable_tail_mask_map": sum(tail_values) / len(tail_values),
    }


def classify(delta_map: float, delta_map50: float) -> str:
    """Screening label; 0.003 mAP50-95 is the observed old-run repeatability floor."""
    if delta_map >= 0.003 and delta_map50 > 0:
        return "provisionally useful"
    if delta_map <= -0.003:
        return "provisionally harmful"
    return "within single-run noise"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    rows = []
    for run_dir in sorted(project.iterdir()):
        result = read_run(run_dir, args.tail) if run_dir.is_dir() else None
        if result:
            rows.append(result)
    if not rows:
        raise RuntimeError(f"No usable results.csv found below {project}")
    references = {"A": next((row for row in rows if row["name"] == "A00_reference"), None),
                  "L": next((row for row in rows if row["name"] == "L00_no_aux"), None)}
    for row in rows:
        reference = references["L"] if row["name"].startswith("L") else references["A"]
        if reference:
            row["delta_mask_map"] = row["mask_map"] - reference["mask_map"]
            row["delta_mask_map50"] = row["mask_map50"] - reference["mask_map50"]
            row["screening_judgement"] = classify(row["delta_mask_map"], row["delta_mask_map50"])
        else:
            row["delta_mask_map"] = float("nan")
            row["delta_mask_map50"] = float("nan")
            row["screening_judgement"] = "reference missing"
    output_csv = project / "citrus_topo_summary.csv"
    write_csv(output_csv, rows)

    lines = [
        "# CitrusTopo-Seg provisional result table",
        "",
        "| Run | Epochs | Best epoch | Mask mAP50 | Mask mAP50-95 | Stable tail | ΔmAP50 | ΔmAP50-95 | Judgment |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['epochs']} | {row['best_epoch']} | {row['mask_map50']:.4f} | "
            f"{row['mask_map']:.4f} | {row['stable_tail_mask_map']:.4f} | {row['delta_mask_map50']:+.4f} | "
            f"{row['delta_mask_map']:+.4f} | {row['screening_judgement']} |"
        )
    lines.extend(
        [
            "",
            "Judgments are provisional single-seed screening labels. A change below 0.003 mAP50-95 is treated as "
            "indistinguishable from the repeatability noise observed in the historical runs. Final claims require "
            "three seeds on the same grouped split.",
        ]
    )
    output_md = project / "CITRUS_TOPO_SUMMARY.md"
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {output_csv}\nWrote {output_md}")


if __name__ == "__main__":
    main()
