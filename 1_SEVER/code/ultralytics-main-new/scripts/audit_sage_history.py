"""Read historical results and current YAMLs without changing experiments or datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import yaml

AP = "metrics/mAP50-95(M)"


def digest(content):
    return hashlib.sha256(content).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--code", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    configs, by_name, errors = [], defaultdict(list), []
    for path in sorted((args.code / "0_orange_yaml").rglob("*.yaml")):
        try:
            config = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
            if not isinstance(config, dict) or "backbone" not in config:
                continue
            graph = {k: config.get(k) for k in ("backbone", "head", "scales", "scale", "nc")}
            item = {
                "path": str(path),
                "series": path.relative_to(args.code / "0_orange_yaml").parts[0],
                "graph_hash": digest(json.dumps(graph, sort_keys=True).encode()),
                "backbone": config["backbone"],
                "head": config.get("head", []),
            }
            configs.append(item)
            by_name[path.name].append(item)
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
    runs, csv_groups = [], defaultdict(list)
    for path in sorted(args.results.rglob("results.csv")):
        try:
            raw = path.read_bytes()
            rows = [
                {k.strip(): float(v) for k, v in row.items() if k and v and v.strip()}
                for row in csv.DictReader(raw.decode("utf-8-sig").splitlines())
            ]
            if not rows or AP not in rows[0]:
                continue
            config_path = path.parent / "args.yaml"
            train_args = yaml.safe_load(config_path.read_text(encoding="utf-8-sig")) if config_path.exists() else {}
            peak = max(rows, key=lambda r: r[AP])
            model_name = str(train_args.get("model", "")).replace("\\", "/").rsplit("/", 1)[-1]
            matches = by_name.get(model_name, [])
            csv_hash = digest(raw)
            csv_groups[csv_hash].append(str(path))
            item = {
                "name": path.parent.name,
                "path": str(path),
                "series": path.relative_to(args.results).parts[0],
                "csv_hash": csv_hash,
                "epochs_observed": len(rows),
                "epoch_max": max(r.get("epoch", 0) for r in rows),
                "peak": peak,
                "tail20_mean": statistics.mean(r[AP] for r in rows[-20:]),
                "budget_peaks": {
                    str(n): max((r[AP] for r in rows if r.get("epoch", 0) <= n), default=None)
                    for n in (50, 100, 150, 200, 300)
                },
                "args": train_args,
                "current_yaml_candidates": [m["path"] for m in matches],
                "mapping_status": "basename_candidate_only" if matches else "unresolved_renamed_or_absent",
            }
            # Basename correspondence does NOT establish the exact historical source revision.
            runs.append(item)
        except Exception as error:
            errors.append({"path": str(path), "error": repr(error)})
    payload = {
        "caveat": "Identical CSVs are copies, not independent repeats. Current YAMLs are not historical source proof.",
        "runs": runs,
        "configs": configs,
        "duplicate_csv_groups": [v for v in csv_groups.values() if len(v) > 1],
        "errors": errors,
    }
    (args.output / "history_architecture_inventory.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    fields = ["series", "name", "epochs_observed", "mask_ap", "mask_ap50", "peak_epoch", "path", "yaml_candidates"]
    with (args.output / "history_summary.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for r in runs:
            writer.writerow(
                {
                    "series": r["series"],
                    "name": r["name"],
                    "epochs_observed": r["epochs_observed"],
                    "mask_ap": r["peak"][AP],
                    "mask_ap50": r["peak"]["metrics/mAP50(M)"],
                    "peak_epoch": r["peak"]["epoch"],
                    "path": r["path"],
                    "yaml_candidates": " | ".join(r["current_yaml_candidates"]),
                }
            )
    print(
        json.dumps(
            {
                "csv_files": len(runs),
                "unique_csv_contents": len(csv_groups),
                "model_yamls": len(configs),
                "results_by_series": dict(Counter(r["series"] for r in runs)),
                "yaml_by_series": dict(Counter(c["series"] for c in configs)),
                "unresolved_results": sum(not r["current_yaml_candidates"] for r in runs),
                "errors": errors,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
