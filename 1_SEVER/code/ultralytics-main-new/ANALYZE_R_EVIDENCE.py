"""Report predeclared factorial contrasts in percentage points; never fill missing arms.

python ANALYZE_R_EVIDENCE.py /data/sxq/results/R_EVIDENCE_20261112_factorial
All metrics must come from this runner's fixed full-label protocol. One seed is
descriptive only. Multiple models on one seed are not independent replicates.
"""
import argparse
import csv
from pathlib import Path
import statistics

ARMS = {
    (0, 0, 0): "RE00_baseline", (1, 0, 0): "RE01_asym", (0, 1, 0): "RE02_contrast",
    (0, 0, 1): "RE03_semantic", (1, 1, 0): "RE04_asym_contrast",
    (1, 0, 1): "RE05_asym_semantic", (0, 1, 1): "RE06_contrast_semantic", (1, 1, 1): "RE07_full",
}


def contrasts(rows):
    result = []
    for seed in sorted({r["seed"] for r in rows}):
        selected = [r for r in rows if r["seed"] == seed and r["model"] in ARMS.values()]
        if len({r["model"] for r in selected}) != len(selected):
            raise ValueError(f"Duplicate model/seed rows: {seed}")
        by_name = {r["model"]: r for r in selected}
        missing = sorted(set(ARMS.values()) - set(by_name))
        if missing:
            print(f"Seed {seed}: incomplete factorial; missing {missing}. No contrast estimated.")
            continue
        if len({r["targets"] for r in selected}) != 1:
            raise ValueError("Different validation target counts within the same seed")
        for metric in ("mask_ap50", "mask_ap5095", "recall", "r_at_p90"):
            y = {f: float(by_name[name][metric]) for f, name in ARMS.items()}
            effects = {"ABC_minus_baseline": y[1, 1, 1] - y[0, 0, 0],
                       "A_alone": y[1, 0, 0] - y[0, 0, 0],
                       "B_alone": y[0, 1, 0] - y[0, 0, 0],
                       "C_alone": y[0, 0, 1] - y[0, 0, 0],
                       "AB_interaction_C0": y[1, 1, 0]-y[1, 0, 0]-y[0, 1, 0]+y[0, 0, 0],
                       "AC_interaction_B0": y[1, 0, 1]-y[1, 0, 0]-y[0, 0, 1]+y[0, 0, 0],
                       "BC_interaction_A0": y[0, 1, 1]-y[0, 1, 0]-y[0, 0, 1]+y[0, 0, 0]}
            for i, label in enumerate("ABC"):
                effects[f"{label}_marginal_mean"] = statistics.mean(v for f, v in y.items() if f[i]) - statistics.mean(
                    v for f, v in y.items() if not f[i])
            for effect, value in effects.items():
                result.append(dict(seed=seed, metric=metric, contrast=effect, delta_pp=value))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path)
    args = parser.parse_args()
    with (args.project / "FULL_LABEL_RESULTS.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    output = contrasts(rows)
    if not output:
        return
    path = args.project / "FACTOR_CONTRASTS.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    print(f"Saved {path}. These are within-seed differences, not a significance test.")


if __name__ == "__main__":
    main()
