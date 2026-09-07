"""Read all historical CSVs and audit V6 without changing uploaded artifacts."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT.parents[1] / "results"
V6 = RESULTS / "SAGE/CITRUS_SAGE_V6_STRUCTURE_300EP"
OUT = ROOT / "reports/sage_v7_20260906"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    records, errors = [], []
    for path in sorted(RESULTS.rglob("results.csv")):
        try:
            with path.open(encoding="utf-8-sig") as f:
                rows = [{k.strip(): float(v) for k, v in row.items() if k and v.strip()} for row in csv.DictReader(f)]
            peak = max(rows, key=lambda r: r["metrics/mAP50-95(M)"])
            args_path = path.parent / "args.yaml"
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.exists() else {}
            records.append(
                dict(
                    path=str(path),
                    name=path.parent.name,
                    rows=len(rows),
                    peak=peak,
                    csv_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    args=args,
                    tail20=statistics.mean(r["metrics/mAP50-95(M)"] for r in rows[-20:]),
                    seconds=statistics.median(b["time"] - a["time"] for a, b in zip(rows, rows[1:]))
                    if len(rows) > 1 and "time" in peak
                    else None,
                )
            )
        except (ValueError, KeyError, TypeError) as error:
            errors.append(dict(path=str(path), error=str(error)))
    configs = {}
    for path in (ROOT / "0_orange_yaml").rglob("*.yaml"):
        try:
            data = yaml.safe_load(path.read_text(encoding="utf-8"))
            configs[str(path)] = dict(
                modules=[layer[2] for layer in data.get("backbone", []) + data.get("head", [])],
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            )
        except (TypeError, AttributeError, yaml.YAMLError):
            continue
    v6 = [r for r in records if Path(r["path"]).parent.parent == V6]
    reference = v6[0]["args"]
    ignore = {"model", "name", "project", "save_dir"}
    for r in v6:
        run = Path(r["path"]).parent
        r["arg_differences"] = {
            k: [reference.get(k), r["args"].get(k)]
            for k in reference.keys() | r["args"].keys()
            if k not in ignore and reference.get(k) != r["args"].get(k)
        }
        r["initialization"] = json.loads((run / "initialization_transfer.json").read_text())
        r["loaded"] = json.loads((run / "loaded_data_summary.json").read_text())
        r["completed_marker"] = (run / "completed.json").is_file()
    recorded = json.loads((V6 / "_protocol/implementation_sha256.json").read_text())
    sources = {}
    for name, digest in recorded.items():
        p = ROOT / ("yolo11n-seg.pt" if name == "initialization_checkpoint" else name)
        sources[name] = p.exists() and hashlib.sha256(p.read_bytes()).hexdigest() == digest
    payload = dict(history=records, errors=errors, yaml_index=configs, v6=v6, source_matches=sources)
    (OUT / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    fields = ["name", "rows", "mask_ap", "mask_ap50", "tail20", "seconds", "path", "csv_sha256"]
    with (OUT / "history.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow(
                {
                    **{k: r[k] for k in fields if k in r},
                    "mask_ap": 100 * r["peak"]["metrics/mAP50-95(M)"],
                    "mask_ap50": 100 * r["peak"]["metrics/mAP50(M)"],
                    "tail20": 100 * r["tail20"],
                }
            )
    print(
        json.dumps(
            dict(
                total_csv=len(records),
                unique_csv=len({r["csv_sha256"] for r in records}),
                errors=errors,
                yaml_count=len(configs),
                unmatched_sources=[k for k, v in sources.items() if not v],
                v6=[{k: v for k, v in r.items() if k not in {"args", "initialization", "loaded"}} for r in v6],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
