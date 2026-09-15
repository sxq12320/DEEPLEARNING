"""Audit received V6 runs without modifying results or choosing independent metric peaks."""

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path

import yaml


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(source, code):
    runs = []
    for path in sorted(source.glob("V6_*_seed*/results.csv")):
        directory = path.parent
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(handle)]
        ap = "metrics/mAP50-95(M)"
        args = yaml.safe_load((directory / "args.yaml").read_text(encoding="utf-8"))
        marker = read_json(directory / "completed.json")
        local_yaml = code / "0_orange_yaml/E_V6_series" / (marker["name"] + ".yaml")
        paired = {}
        for budget in ("coarse", "fine"):
            report = directory / f"paired_{budget}_eval/paired_metrics.json"
            data = read_json(report)
            paired[budget] = {
                "path": str(report), "summary": data["summary"], "protocol": data["protocol"],
                "images": len({record["image"] for record in data["records"]}),
                "weights": data["weights"],
            }
        times = [b["time"] - a["time"] for a, b in zip(rows, rows[1:]) if b["time"] > a["time"]]
        runs.append({
            "name": marker["name"], "path": str(directory), "epochs": len(rows),
            "best": max(rows, key=lambda r: r[ap]), "last": rows[-1],
            "last20_ap": statistics.mean(r[ap] for r in rows[-20:]),
            "median_epoch_s": statistics.median(times), "args": args, "marker": marker,
            "yaml_matches": digest(local_yaml) == marker["yaml_sha256"],
            "val_list_sha256": digest(directory / "val_loaded_files.txt"),
            "train_list_sha256": digest(directory / "train_loaded_files.txt"),
            "loaded": read_json(directory / "loaded_data_summary.json"),
            "paired": paired,
        })
    variable = {k: {r["name"]: r["args"].get(k) for r in runs}
                for k in set().union(*(r["args"] for r in runs))
                if len({json.dumps(r["args"].get(k), sort_keys=True) for r in runs}) > 1}
    by_name = {r["name"]: r for r in runs}
    pairs = []
    for before, after in [(0, 1), (1, 2), (1, 8), (2, 8), (0, 3), (2, 5), (8, 9), (0, 4), (2, 6), (5, 7)]:
        a, b = runs[before], runs[after]
        row = {"before": a["name"], "after": b["name"],
               "epoch_best_delta_pp": 100 * (b["best"][ap] - a["best"][ap]),
               "last20_delta_pp": 100 * (b["last20_ap"] - a["last20_ap"])}
        row["paired"] = {budget: {mode: {
            "ap_delta_pp": 100 * (b["paired"][budget]["summary"][mode][ap]
                                  - a["paired"][budget]["summary"][mode][ap]),
            "tiny_matches_delta": b["paired"][budget]["summary"][mode]["tiny_matched"]
                                  - a["paired"][budget]["summary"][mode]["tiny_matched"],
            "background_delta": b["paired"][budget]["summary"][mode]["errors25"]["background"]
                                - a["paired"][budget]["summary"][mode]["errors25"]["background"],
        } for mode in ("global", "trustedmask")} for budget in ("coarse", "fine")}
        pairs.append(row)
    return {"runs": runs, "pairs": pairs, "variable_args": variable,
            "val_list_identical": len({r["val_list_sha256"] for r in runs}) == 1,
            "train_list_identical": len({r["train_list_sha256"] for r in runs}) == 1,
            "all_yaml_match": all(r["yaml_matches"] for r in by_name.values()),
            "source_limitation": "Server Python snapshots/weights absent: YAML match does not prove full source identity."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.source, Path(__file__).resolve().parents[1])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print("same train/val lists:", result["train_list_identical"], result["val_list_identical"])
    print("all YAML match:", result["all_yaml_match"], "variable args:", sorted(result["variable_args"]))
    for r in result["runs"]:
        print(r["name"], "epoch", round(r["median_epoch_s"], 1), "seconds",
              "AP", round(r["best"]["metrics/mAP50-95(M)"] * 100, 3))
    for pair in result["pairs"]:
        print(json.dumps(pair))


if __name__ == "__main__":
    main()
