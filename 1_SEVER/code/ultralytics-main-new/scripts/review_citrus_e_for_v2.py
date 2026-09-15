"""Read-only audit of uploaded E results, including paired per-image recall differences."""

from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT.parents[1] / "results"
SOURCE = RESULTS / "E/CITRUS_E9_GUIDED_DEVICEBOUND_ALL_300EP"
OUT = ROOT / "reports/citrus_e_v2"
AP = "metrics/mAP50-95(M)"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None


def paired_recall(a, b, tiny=True):
    """Image-cluster bootstrap: uncertainty across validation images, not training seeds."""
    aa = {(r["image"], i): r for i, r in enumerate(a)}
    # Per-image arrays preserve GT order; positions are checked by areas.
    groups = {}
    for rows, key in ((list(aa.values()), "a"), (b, "b")):
        for r in rows:
            if not tiny or r["area640"] < 256:
                groups.setdefault(r["image"], {"a": [], "b": []})[key].append(r)
    values = []
    for name, g in sorted(groups.items()):
        if [r["area640"] for r in g["a"]] != [r["area640"] for r in g["b"]]:
            raise ValueError(f"GT identity/area mismatch: {name}")
        values.append([len(g["a"]), sum(r["matched25"] for r in g["a"]), sum(r["matched25"] for r in g["b"])])
    v = np.array(values)
    rng = np.random.default_rng(20260908)
    # Include images with zero tiny instances in the cluster population.
    n_images = len({r["image"] for r in a})
    v = np.concatenate((v, np.zeros((n_images - len(v), 3))))
    boot = v[rng.integers(0, len(v), (4000, len(v)))].sum(1)
    delta = (boot[:, 2] - boot[:, 1]) / np.maximum(boot[:, 0], 1) * 100
    totals = v.sum(0)
    return dict(
        n=int(totals[0]),
        before=int(totals[1]),
        after=int(totals[2]),
        delta_pp=float((totals[2] - totals[1]) / totals[0] * 100),
        image_bootstrap_ci95_pp=np.quantile(delta, [0.025, 0.975]).tolist(),
    )


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    runs, history, errors = [], [], []
    for path in sorted(RESULTS.rglob("results.csv")):
        try:
            with path.open(encoding="utf-8-sig", newline="") as f:
                rows = [{k.strip(): float(v) for k, v in row.items() if k and v.strip()} for row in csv.DictReader(f)]
            peak = max(rows, key=lambda x: x[AP])
            args_path = path.parent / "args.yaml"
            args = yaml.safe_load(args_path.read_text(encoding="utf-8")) if args_path.is_file() else {}
            h = dict(
                name=path.parent.name,
                path=str(path),
                rows=len(rows),
                peak=peak,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                args=args,
            )
            history.append(h)
            if path.parent.parent != SOURCE:
                continue
            run = dict(
                h,
                last=rows[-1],
                tail20=statistics.mean(x[AP] for x in rows[-20:]),
                completed=read_json(path.parent / "completed.json"),
                selection=read_json(path.parent / "best_mask_selection.json"),
                loaded=read_json(path.parent / "loaded_data_summary.json"),
                init=read_json(path.parent / "initialization_transfer.json"),
                weight_files=[str(p) for p in (path.parent / "weights").glob("*.pt")],
                epoch_seconds_median=statistics.median(b["time"] - a["time"] for a, b in zip(rows, rows[1:])),
            )
            report = read_json(path.parent / "paired_sliced_eval/paired_metrics.json")
            run["paired"] = report
            run["yaml"] = yaml.safe_load(
                (ROOT / "0_orange_yaml/E_series" / (path.parent.name.removesuffix("_seed42") + ".yaml")).read_text()
            )
            run["yaml_sha_matches_completed"] = hashlib.sha256(
                (ROOT / "0_orange_yaml/E_series" / (path.parent.name.removesuffix("_seed42") + ".yaml")).read_bytes()
            ).hexdigest() == run["completed"].get("yaml_sha256")
            runs.append(run)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            errors.append(dict(path=str(path), error=repr(exc)))
    reference = runs[0]["args"]
    ignore = {"name", "model", "data", "save_dir", "project"}
    for run in runs:
        run["argument_differences"] = {
            k: [reference.get(k), run["args"].get(k)]
            for k in reference.keys() | run["args"].keys()
            if k not in ignore and reference.get(k) != run["args"].get(k)
        }
        run["validation_list_sha"] = hashlib.sha256(
            (Path(run["path"]).parent / "val_loaded_files.txt").read_bytes()
        ).hexdigest()

    def records(i, mode):
        return [r for r in runs[i]["paired"]["records"] if r["mode"] == mode]

    comparisons = {}
    for a, b in ((0, 1), (2, 3), (5, 4), (3, 6), (3, 7), (3, 8)):
        for mode in ("global", "sliced"):
            comparisons[f"E{b:02}-E{a:02}/{mode}"] = dict(
                tiny=paired_recall(records(a, mode), records(b, mode)),
                all=paired_recall(records(a, mode), records(b, mode), False),
            )
    for i in range(len(runs)):
        comparisons[f"E{i:02}/sliced-global"] = dict(
            tiny=paired_recall(records(i, "global"), records(i, "sliced")),
            all=paired_recall(records(i, "global"), records(i, "sliced"), False),
        )
    if "guided" in runs[8]["paired"]["summary"]:
        comparisons["E08/guided-sliced"] = dict(
            tiny=paired_recall(records(8, "sliced"), records(8, "guided")),
            all=paired_recall(records(8, "sliced"), records(8, "guided"), False),
        )
    payload = dict(
        runs=runs,
        history=history,
        errors=errors,
        comparisons=comparisons,
        source_protocol_present=(SOURCE / "_protocol").exists(),
        guide_uploaded=(SOURCE / "_crop_guide").exists(),
    )
    (OUT / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    table = [
        "# E series uploaded-result audit",
        "",
        "Percent units; same best strict-mask-AP epoch for CSV columns.",
        "",
        "| Model | Epoch | AP50 | AP50-95 | P | R | Last20 AP | s/epoch |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in runs:
        p = r["peak"]
        table.append(
            f"| {r['name']} | {int(p['epoch'])} | {p['metrics/mAP50(M)'] * 100:.3f} | {p[AP] * 100:.3f} | "
            f"{p['metrics/precision(M)'] * 100:.3f} | {p['metrics/recall(M)'] * 100:.3f} | {r['tail20'] * 100:.3f} | {r['epoch_seconds_median']:.2f} |"
        )
    table += [
        "",
        "Paired common 640 mask raster; NOT historical stride4 AP. Tiny: area640<256, conf>=.25, mask IoU>=.5.",
        "",
        "| Model/mode | AP50 | AP50-95 | tiny R | all R@.25 | median ms |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for i, r in enumerate(runs):
        for mode, s in r["paired"]["summary"].items():
            rr = records(i, mode)
            table.append(
                f"| E{i:02}/{mode} | {s['metrics/mAP50(M)'] * 100:.3f} | {s[AP] * 100:.3f} | {s['tiny_recall25'] * 100:.3f} | "
                f"{sum(x['matched25'] for x in rr) / len(rr) * 100:.3f} | {s['inference_fusion_median_ms']:.2f} |"
            )
    table += ["", "```json", json.dumps(comparisons, indent=2), "```"]
    (OUT / "RESULTS.md").write_text("\n".join(table), encoding="utf-8")
    print("\n".join(table[: table.index("```json")]))
    print(
        json.dumps(
            dict(
                history=len(history),
                unique=len({r["sha256"] for r in history}),
                errors=errors,
                protocol_present=payload["source_protocol_present"],
                guide_uploaded=payload["guide_uploaded"],
                checks=[
                    dict(
                        name=r["name"],
                        rows=r["rows"],
                        arg_diff=r["argument_differences"],
                        loaded=r["loaded"],
                        yaml_match=r["yaml_sha_matches_completed"],
                        weights=len(r["weight_files"]),
                    )
                    for r in runs
                ],
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
