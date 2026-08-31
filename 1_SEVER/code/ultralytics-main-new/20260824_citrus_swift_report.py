"""Create noise-aware CitrusSwift screening and multi-seed result tables."""

from __future__ import annotations

import argparse
import csv
import math
import re
import statistics
from pathlib import Path


METRICS = {
    "mask_map50": ("metrics/mAP50(M)", "metrics/mAP50(Mask)"),
    "mask_map": ("metrics/mAP50-95(M)", "metrics/mAP50-95(Mask)"),
    "mask_precision": ("metrics/precision(M)", "metrics/precision(Mask)"),
    "mask_recall": ("metrics/recall(M)", "metrics/recall(Mask)"),
    "box_map50": ("metrics/mAP50(B)", "metrics/mAP50(Box)"),
    "box_map": ("metrics/mAP50-95(B)", "metrics/mAP50-95(Box)"),
}
SEED_SUFFIX = re.compile(r"_seed(?P<seed>\d+)$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize CitrusSwift runs without hiding run-to-run variance.")
    parser.add_argument("--project", type=Path, default=Path("1_results/S_series/grouped_clean_screen50"))
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--noise-floor", type=float, default=0.003)
    return parser.parse_args()


def pick(row: dict[str, str], candidates: tuple[str, ...]) -> float:
    for key in candidates:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
    return float("nan")


def read_run(run_dir: Path, tail: int) -> dict | None:
    result_path = run_dir / "results.csv"
    if not result_path.is_file():
        return None
    with result_path.open(encoding="utf-8-sig", errors="ignore") as handle:
        rows = list(csv.DictReader(handle))
    enriched = [(row, {name: pick(row, keys) for name, keys in METRICS.items()}) for row in rows]
    enriched = [(row, values) for row, values in enriched if math.isfinite(values["mask_map"])]
    if not enriched:
        return None
    best_row, best = max(enriched, key=lambda pair: pair[1]["mask_map"])
    stable = [values["mask_map"] for _, values in enriched[-tail:]]
    match = SEED_SUFFIX.search(run_dir.name)
    family = SEED_SUFFIX.sub("", run_dir.name)
    return {
        "run": run_dir.name,
        "family": family,
        "seed": int(match.group("seed")) if match else "unspecified",
        "epochs": len(rows),
        "best_epoch": int(float(best_row.get("epoch", 0))) + 1,
        **best,
        "stable_tail_mask_map": statistics.fmean(stable),
    }


def reference_family(family: str) -> str:
    return "L00_standard" if family.startswith("L") else "S00_reference"


def judgment(delta_map: float, delta_map50: float, floor: float) -> str:
    if delta_map >= floor and delta_map50 > 0:
        return "promote provisionally"
    if delta_map <= -floor:
        return "reject provisionally"
    return "within one-run noise"


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float, digits: int = 4) -> str:
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)
    runs = [read_run(path, args.tail) for path in sorted(project.iterdir()) if path.is_dir()]
    runs = [run for run in runs if run]
    if not runs:
        raise RuntimeError(f"No usable results.csv found below {project}")

    family_rows: list[dict] = []
    for family in sorted({run["family"] for run in runs}):
        members = [run for run in runs if run["family"] == family]
        values = [run["stable_tail_mask_map"] for run in members]
        family_rows.append(
            {
                "family": family,
                "runs": len(members),
                "epochs_min": min(run["epochs"] for run in members),
                "mask_map50_mean": statistics.fmean(run["mask_map50"] for run in members),
                "mask_map_mean": statistics.fmean(run["mask_map"] for run in members),
                "mask_map_std": statistics.stdev(run["mask_map"] for run in members) if len(members) > 1 else float("nan"),
                "stable_tail_mean": statistics.fmean(values),
                "stable_tail_std": statistics.stdev(values) if len(values) > 1 else float("nan"),
            }
        )
    by_family = {row["family"]: row for row in family_rows}
    for row in family_rows:
        reference = by_family.get(reference_family(row["family"]))
        if reference:
            row["delta_mask_map"] = row["mask_map_mean"] - reference["mask_map_mean"]
            row["delta_mask_map50"] = row["mask_map50_mean"] - reference["mask_map50_mean"]
            row["judgment"] = judgment(row["delta_mask_map"], row["delta_mask_map50"], args.noise_floor)
        else:
            row["delta_mask_map"] = float("nan")
            row["delta_mask_map50"] = float("nan")
            row["judgment"] = "reference missing"

    individual_csv = project / "citrus_swift_individual_runs.csv"
    family_csv = project / "citrus_swift_family_summary.csv"
    write_csv(individual_csv, runs)
    write_csv(family_csv, family_rows)

    lines = [
        "# CitrusSwift result summary",
        "",
        "| Family | Runs | Epochs min | Mask mAP50 | Mask mAP50-95 mean ± SD | Stable-tail mean ± SD | ΔmAP50 | ΔmAP50-95 | Decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in family_rows:
        map_sd = fmt(row["mask_map_std"])
        tail_sd = fmt(row["stable_tail_std"])
        lines.append(
            f"| {row['family']} | {row['runs']} | {row['epochs_min']} | {row['mask_map50_mean']:.4f} | "
            f"{row['mask_map_mean']:.4f} ± {map_sd} | {row['stable_tail_mean']:.4f} ± {tail_sd} | "
            f"{row['delta_mask_map50']:+.4f} | {row['delta_mask_map']:+.4f} | {row['judgment']} |"
        )
    lines.extend(
        [
            "",
            f"A single-run change smaller than `{args.noise_floor:.3f}` mask mAP50-95 is not treated as evidence. "
            "Final claims require the same grouped split and at least three seeds for the reference and finalist.",
            "",
            "This summary cannot recover AP by size, Boundary AP, topology split/merge errors, or challenge-subset scores "
            "from the default training CSV. Run the dedicated evaluation pipeline before thesis reporting.",
        ]
    )
    markdown = project / "CITRUS_SWIFT_SUMMARY.md"
    markdown.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {individual_csv}\nWrote {family_csv}\nWrote {markdown}")


if __name__ == "__main__":
    main()
