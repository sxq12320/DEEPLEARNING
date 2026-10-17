"""D5 paired image/cluster bootstrap, recomputing globally ranked EV6-style AP.

Inputs are prediction_stats.jsonl files from the read-only diagnostic exporter.
Matching was done within each image with EV6's rule, before resampling. Replicas
remain separate copies of their image's GT/predictions. This is finite-val
sampling uncertainty, NOT training-seed variation. Use matching raster/protocol.
"""

# ruff: noqa: E402
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.diagnose_citrus_20261005 import metric_rows


def paired_bootstrap(left, right, repetitions=2000, seed=42, unit="group"):
    """Resample shared units, preserving paired members and all predictions."""
    if set(left) != set(right):
        raise ValueError("Paired image IDs must match exactly")
    if any(left[k]["targets"] != right[k]["targets"] or left[k]["tiny_targets"] != right[k]["tiny_targets"]
           for k in left):
        raise ValueError("GT/raster identity differs; these statistics are not paired")
    mapping = {name: row["group"] if unit == "group" else name for name, row in left.items()}
    if unit == "group" and any(right[k]["group"] != left[k]["group"] for k in left):
        raise ValueError("Group membership mismatch")
    units = sorted(set(mapping.values()))
    members = {u: [name for name in left if mapping[name] == u] for u in units}
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in ("ap50", "ap5095", "tiny_recall25")}
    for _ in range(repetitions):
        names = sum((members[u] for u in rng.choice(units, len(units), replace=True)), [])
        a, b = metric_rows([left[n] for n in names]), metric_rows([right[n] for n in names])
        for k in samples:
            if a[k] is not None and b[k] is not None:
                samples[k].append(b[k] - a[k])
    a, b = metric_rows(list(left.values())), metric_rows(list(right.values()))
    return dict(unit=unit, units=len(units), images=len(left), repetitions=repetitions, seed=seed,
                difference="right minus left", left=a, right=b,
                delta={k: dict(estimate=b[k] - a[k] if a[k] is not None and b[k] is not None else None,
                               ci95=np.quantile(v, [0.025, 0.975]).tolist() if v else None,
                               valid_replicates=len(v)) for k, v in samples.items()},
                limitations="Conditional finite-validation sampling CI; does not estimate seed variance, "
                "remove old-split train/val leakage, correct multi-arm selection, or verify training CSV metrics.")


def load(path, mode):
    """Read a single mode and reject duplicate image identities."""
    rows = [json.loads(s) for s in path.read_text(encoding="utf-8").splitlines() if s.strip()]
    selected = [r for r in rows if r["mode"] == mode]
    result = {r["image"]: r for r in selected}
    if not result or len(result) != len(selected):
        raise ValueError("Missing mode or duplicate image IDs")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--left-mode", default="global")
    parser.add_argument("--right-mode", default="fine")
    parser.add_argument("--unit", choices=("image", "group"), default="group")
    parser.add_argument("--repetitions", type=int, default=2000)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    payload = paired_bootstrap(load(args.left, args.left_mode), load(args.right, args.right_mode),
                               args.repetitions, unit=args.unit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
