"""Create protocol-aware CitrusB result tables after server training."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
PROFILE_CSV = ROOT / "1_results" / "_compatibility" / "citrus_b_profiles.csv"
LATENCY_CSV = ROOT / "1_results" / "_compatibility" / "citrus_b_latency_cpu.csv"
SEED_SUFFIX = re.compile(r"_seed(?P<seed>\d+)$")
METRICS = {
    "mask_map50": ("metrics/mAP50(M)", "metrics/mAP50(Mask)"),
    "mask_map": ("metrics/mAP50-95(M)", "metrics/mAP50-95(Mask)"),
    "mask_precision": ("metrics/precision(M)", "metrics/precision(Mask)"),
    "mask_recall": ("metrics/recall(M)", "metrics/recall(Mask)"),
    "box_map50": ("metrics/mAP50(B)", "metrics/mAP50(Box)"),
    "box_map": ("metrics/mAP50-95(B)", "metrics/mAP50-95(Box)"),
}


def parse_args() -> argparse.Namespace:
    """Parse report settings."""
    parser = argparse.ArgumentParser(description="Summarize CitrusB runs and judge changes within one protocol.")
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--tail", type=int, default=10)
    parser.add_argument("--noise-floor", type=float, default=0.003)
    return parser.parse_args()


def pick(row: dict[str, str], candidates: tuple[str, ...]) -> float:
    """Read one metric across known Ultralytics column aliases."""
    for key in candidates:
        try:
            return float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
    return float("nan")


def read_table(path: Path, key: str) -> dict[str, dict[str, str]]:
    """Index a CSV if it exists."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8-sig", errors="ignore") as handle:
        return {row[key]: row for row in csv.DictReader(handle)}


def read_ledger(project: Path) -> dict[str, dict[str, Any]]:
    """Load the latest durable event for every run name."""
    path = project / "experiment_ledger.jsonl"
    latest: dict[str, dict[str, Any]] = {}
    if not path.is_file():
        return latest
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        try:
            event = json.loads(line)
            run_name = event["protocol"]["name"]
        except (json.JSONDecodeError, KeyError, TypeError):
            continue
        latest[run_name] = event
    return latest


def read_run(
    run_dir: Path,
    tail: int,
    ledger: dict[str, dict[str, Any]],
    profiles: dict[str, dict[str, str]],
    latencies: dict[str, dict[str, str]],
) -> dict[str, Any] | None:
    """Extract the best and stable-tail metrics for one run."""
    path = run_dir / "results.csv"
    if not path.is_file():
        return None
    with path.open(encoding="utf-8-sig", errors="ignore") as handle:
        raw_rows = list(csv.DictReader(handle))
    enriched = [(row, {name: pick(row, aliases) for name, aliases in METRICS.items()}) for row in raw_rows]
    enriched = [(row, values) for row, values in enriched if math.isfinite(values["mask_map"])]
    if not enriched:
        return None
    best_row, best = max(enriched, key=lambda pair: pair[1]["mask_map"])
    stable = [values["mask_map"] for _, values in enriched[-tail:]]
    match = SEED_SUFFIX.search(run_dir.name)
    family = SEED_SUFFIX.sub("", run_dir.name)
    event = ledger.get(run_dir.name, {})
    experiment = event.get("experiment", {})
    protocol = event.get("protocol", {})
    yaml_stem = Path(experiment.get("yaml", "")).stem
    profile = profiles.get(yaml_stem, {})
    latency = latencies.get(yaml_stem, {})
    desired_epochs = int(protocol.get("epochs", 0) or 0)
    epochs = len(raw_rows)
    weights_exist = (run_dir / "weights" / "best.pt").is_file()
    if desired_epochs and epochs >= desired_epochs:
        status = "complete"
    elif event.get("status") == "completed" and weights_exist:
        status = "early_stop"
    else:
        status = "partial"
    return {
        "run": run_dir.name,
        "family": family,
        "seed": int(match.group("seed")) if match else protocol.get("seed", "unspecified"),
        "status": status,
        "epochs": epochs,
        "desired_epochs": desired_epochs or "unknown",
        "best_epoch": int(float(best_row.get("epoch", epochs))),
        **best,
        "stable_tail_mask_map": statistics.fmean(stable),
        "yaml": experiment.get("yaml", "unknown"),
        "params": int(float(profile["params"])) if profile.get("params") else "NA",
        "gflops_640": float(profile["gflops_640"]) if profile.get("gflops_640") else float("nan"),
        "local_cpu_median_ms": (
            float(latency["latency_median_ms"]) if latency.get("latency_median_ms") else float("nan")
        ),
    }


def reference_family(family: str) -> str:
    """Return the controlled reference for an architecture or loss experiment."""
    if family.startswith("BL"):
        return "BL00_none"
    if family.startswith("B"):
        return "B00_reference"
    return ""


def decision(delta_map: float, delta_map50: float, floor: float) -> str:
    """Make a deliberately conservative single-protocol judgment."""
    if not math.isfinite(delta_map):
        return "reference missing"
    if delta_map >= floor and delta_map50 > 0:
        return "provisionally useful"
    if delta_map <= -floor:
        return "provisionally harmful"
    return "within one-run noise"


def fmt(value: float, digits: int = 4) -> str:
    """Format finite values for Markdown."""
    return "NA" if not math.isfinite(value) else f"{value:.{digits}f}"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a list of homogeneous dictionaries."""
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build individual and seed-aggregated CitrusB tables."""
    args = parse_args()
    project = args.project.expanduser().resolve()
    if not project.is_dir():
        raise FileNotFoundError(project)
    ledger = read_ledger(project)
    profiles = read_table(PROFILE_CSV, "model")
    latencies = read_table(LATENCY_CSV, "model")
    runs = [
        read_run(path, args.tail, ledger, profiles, latencies)
        for path in sorted(project.iterdir())
        if path.is_dir()
    ]
    runs = [run for run in runs if run]
    if not runs:
        raise RuntimeError(f"No usable results.csv found below {project}")

    families = []
    for family in sorted({run["family"] for run in runs}):
        members = [run for run in runs if run["family"] == family]
        mask_maps = [run["mask_map"] for run in members]
        row = {
            "family": family,
            "runs": len(members),
            "status": ",".join(sorted({run["status"] for run in members})),
            "epochs_min": min(run["epochs"] for run in members),
            "mask_map50_mean": statistics.fmean(run["mask_map50"] for run in members),
            "mask_map_mean": statistics.fmean(mask_maps),
            "mask_map_std": statistics.stdev(mask_maps) if len(mask_maps) > 1 else float("nan"),
            "mask_precision_mean": statistics.fmean(run["mask_precision"] for run in members),
            "mask_recall_mean": statistics.fmean(run["mask_recall"] for run in members),
            "stable_tail_mean": statistics.fmean(run["stable_tail_mask_map"] for run in members),
            "params": members[0]["params"],
            "gflops_640": members[0]["gflops_640"],
            "local_cpu_median_ms": members[0]["local_cpu_median_ms"],
        }
        families.append(row)
    indexed = {row["family"]: row for row in families}
    for row in families:
        reference = indexed.get(reference_family(row["family"]))
        row["delta_mask_map50"] = row["mask_map50_mean"] - reference["mask_map50_mean"] if reference else float("nan")
        row["delta_mask_map"] = row["mask_map_mean"] - reference["mask_map_mean"] if reference else float("nan")
        row["decision"] = decision(row["delta_mask_map"], row["delta_mask_map50"], args.noise_floor)

    write_csv(project / "citrus_b_individual_runs.csv", runs)
    write_csv(project / "citrus_b_family_summary.csv", families)
    lines = [
        "# CitrusB result summary",
        "",
        "Only rows inside this project/protocol are compared. One-run changes below the declared noise floor are not",
        "treated as evidence; final reference and finalist require three seeds.",
        "",
        "| Family | Runs | Status | Mask AP50 | Mask AP50-95 mean ± SD | P | R | ΔAP50 | ΔAP | "
        "Params | GFLOPs | Decision |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in families:
        lines.append(
            f"| {row['family']} | {row['runs']} | {row['status']} | {row['mask_map50_mean']:.4f} | "
            f"{row['mask_map_mean']:.4f} ± {fmt(row['mask_map_std'])} | {row['mask_precision_mean']:.4f} | "
            f"{row['mask_recall_mean']:.4f} | {row['delta_mask_map50']:+.4f} | {row['delta_mask_map']:+.4f} | "
            f"{row['params']} | {fmt(row['gflops_640'], 2)} | {row['decision']} |"
        )
    lines.extend(
        [
            "",
            f"Noise floor used for provisional decisions: `{args.noise_floor:.3f}` Mask AP50-95.",
            "Local CPU latency is retained in the CSV only; use the same server GPU benchmark for the paper table.",
            "Default training CSV does not contain AP-small, Boundary F1, split/merge errors, or challenge subsets;",
            "run dedicated evaluation before thesis claims. The official PR `(1,0)` sentinel is not an observed "
            "threshold.",
        ]
    )
    (project / "CITRUS_B_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote CitrusB tables to {project}")


if __name__ == "__main__":
    main()
