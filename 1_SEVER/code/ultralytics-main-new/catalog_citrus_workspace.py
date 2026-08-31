"""Build model and result catalogs for the series-organized citrus workspace."""

from __future__ import annotations

import csv
import math
import re
from collections import Counter
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parent
YAML_ROOT = ROOT / "0_orange_yaml"
RESULTS_ROOT = ROOT.parents[1] / "results"
MODEL_INDEX = YAML_ROOT / "MODEL_INDEX.csv"
RESULT_INDEX = RESULTS_ROOT / "RESULTS_INDEX.csv"
SERIES_STATUS = {
    "A_baselines": "baseline/legacy comparison",
    "B_series": "current redesigned series; pending training",
    "C_series": "task-specific semantic/detail topology series; pending training",
    "D_series": "shape-semantic bilateral backbone series; pending training",
    "F_series": "legacy exploratory series; old-data results",
    "G_series": "legacy G series; old-data results",
    "G_0839_series": "dual-resolution preserve-search-discriminate-refine series; pending screening",
    "H_series": "hybrid redesign series; pending/compatibility",
    "L_series": "topology series; superseded as primary by B v2",
    "N_series": "legacy evidence-combination series; old-data results",
    "S_series": "completed grouped-clean ablation series",
    "SXQ_series": "legacy SXQ series; old-data results",
    "T_series": "unified same-dataset finalist rerun; pending training",
}
METRICS = {
    "mask_map50": ("metrics/mAP50(M)", "metrics/mAP50(Mask)"),
    "mask_map": ("metrics/mAP50-95(M)", "metrics/mAP50-95(Mask)"),
    "mask_precision": ("metrics/precision(M)", "metrics/precision(Mask)"),
    "mask_recall": ("metrics/recall(M)", "metrics/recall(Mask)"),
}


def pick(row: dict[str, str], candidates: tuple[str, ...]) -> float:
    """Read a finite metric from known Ultralytics column aliases."""
    for key in candidates:
        try:
            value = float(row[key])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return float("nan")


def model_rows() -> list[dict[str, object]]:
    """Return one catalog row for every active series YAML."""
    rows = []
    for path in sorted(YAML_ROOT.rglob("*.yaml")):
        relative = path.relative_to(YAML_ROOT)
        if relative.parts[0].startswith("_archive"):
            continue
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            head = data["head"][-1][2]
            nc = data.get("nc", "unknown")
            error = ""
        except Exception as exc:  # noqa: BLE001 - catalog malformed YAML instead of hiding it.
            head, nc, error = "unreadable", "unknown", f"{type(exc).__name__}: {exc}"
        series = relative.parts[0]
        rows.append(
            {
                "series": series,
                "model": path.stem,
                "yaml": relative.as_posix(),
                "head": head,
                "nc": nc,
                "series_status": SERIES_STATUS.get(series, "uncategorized"),
                "parse_error": error,
            }
        )
    return rows


def match_yaml(series: str, run: str, models: list[dict[str, object]]) -> str:
    """Map a historical run name to the best matching current series YAML."""
    candidates = [row for row in models if row["series"] == series]
    token = ""
    if series in {"S_series", "B_series", "C_series", "D_series", "G_0839_series", "T_series"}:
        match = re.search(rf"{series[0]}(\d{{2}})", run)
        token = match.group(1) if match else ""
    elif series == "N_series":
        match = re.search(r"(?:^|_)(\d{2})_", run)
        token = match.group(1) if match else ""
    elif series == "G_series":
        match = re.match(r"(\d{2})_", run)
        token = match.group(1) if match else ""
    elif series == "F_series":
        match = re.match(r"(F\d{2})", run)
        token = match.group(1) if match else ""
    elif series == "A_baselines":
        aliases = {
            "001_1_": "001_1_yolov8-seg",
            "001_2_": "001_2_yolov9c-seg",
            "001_3_": "001_3_yolo11-seg",
            "001_4_": "001_4_yolo12-seg",
            "001_5_": "001_5_yolo26-seg",
            "002_1_": "002_yolo11-seg-starnet-official-s1",
            "002_2_": "002_yolo11-seg-starnet-official-s2",
            "003_": "003_yolo11-seg-mobilenetv4",
            "004_": "004_yolo11-seg-mano",
        }
        for prefix, stem in aliases.items():
            if run.startswith(prefix):
                exact = [row for row in candidates if row["model"] == stem]
                return str(exact[0]["yaml"]) if exact else "unresolved"
    if token:
        for row in candidates:
            stem = str(row["model"])
            if stem.lower().startswith(token.lower()):
                return str(row["yaml"])
    if series == "SXQ_series":
        normalized = re.sub(r"_300ep(?:_.*)?$", "", run)
        exact = [row for row in candidates if str(row["model"]).lower() == normalized.lower()]
        if exact:
            return str(exact[0]["yaml"])
    return "unresolved"


def result_rows(models: list[dict[str, object]]) -> list[dict[str, object]]:
    """Return one row for every training run containing a results.csv."""
    rows = []
    for path in sorted(RESULTS_ROOT.rglob("results.csv")):
        relative = path.relative_to(RESULTS_ROOT)
        if relative.parts[0].startswith("_analysis"):
            continue
        with path.open(encoding="utf-8-sig", errors="ignore") as handle:
            table = list(csv.DictReader(handle))
        enriched = [(row, {name: pick(row, keys) for name, keys in METRICS.items()}) for row in table]
        enriched = [(row, values) for row, values in enriched if math.isfinite(values["mask_map"])]
        if enriched:
            best_raw, best = max(enriched, key=lambda item: item[1]["mask_map"])
            best_epoch = int(float(best_raw.get("epoch", len(table))))
        else:
            best = {name: float("nan") for name in METRICS}
            best_epoch = 0
        run_dir = path.parent
        series = relative.parts[0]
        protocol = relative.parts[1] if len(relative.parts) > 2 else "unspecified"
        rows.append(
            {
                "series": series,
                "protocol": protocol,
                "run": run_dir.name,
                "epochs": len(table),
                "best_epoch": best_epoch,
                **best,
                "yaml": match_yaml(series, run_dir.name, models),
                "results_csv": path.relative_to(RESULTS_ROOT).as_posix(),
                "best_weight": (
                    (run_dir / "weights" / "best.pt").relative_to(RESULTS_ROOT).as_posix()
                    if (run_dir / "weights" / "best.pt").is_file()
                    else "missing"
                ),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write homogeneous catalog rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Refresh both catalogs and print category counts."""
    models = model_rows()
    results = result_rows(models)
    write_csv(MODEL_INDEX, models)
    write_csv(RESULT_INDEX, results)
    print(f"Wrote {len(models)} models to {MODEL_INDEX}")
    print(f"Wrote {len(results)} runs to {RESULT_INDEX}")
    print("Models by series:", dict(sorted(Counter(row["series"] for row in models).items())))
    print("Results by series:", dict(sorted(Counter(row["series"] for row in results).items())))


if __name__ == "__main__":
    main()
