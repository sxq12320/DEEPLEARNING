"""Stratified summary of the 2026-10-05 stage-01 diagnosis (read-only; consumes diagnose_citrus_20261005 output).

Outputs ``stratified.json`` and ``stratified.md`` next to the run: stage funnel by size/solidity/gap/contrast,
paired per-GT rescue/loss between views, per-view s50, and group-bootstrap CIs for tiny-recall deltas.
"""

import argparse
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eval_citrus_j2_gaze import fit_s50  # noqa: E402

MODES = ("global", "fine", "upsample_linear", "upsample_area")
SIZE_BINS = ((0, 8), (8, 16), (16, 32), (32, 64), (64, 1e9))
STAGE_ORDER = ("S4", "collision", "S3_mask", "S2_nms", "capacity_pre_nms", "capacity_post_nms", "merge_suppression",
               "S1_score", "S0_box", "unresolved")


def num(value, default=math.nan):
    return float(value) if value not in ("", None) else default


def label(lo, hi):
    return "{}-{}".format(int(lo), "inf" if hi > 1e8 else int(hi))


def load(run, threshold):
    rows = list(csv.DictReader((run / "records.csv").open(encoding="utf-8-sig")))
    rows = [r for r in rows if abs(float(r["threshold"]) - threshold) < 1e-9]
    by = defaultdict(dict)
    for r in rows:
        by[(r["image"], int(r["gt_index"]))][r["mode"]] = r
    return rows, by


def funnel(rows, key):
    table = defaultdict(Counter)
    for r in rows:
        table[(r["mode"], key(r))][r["stage"]] += 1
    return table


def size_key(r):
    s = float(r["size640"])
    return next(label(lo, hi) for lo, hi in SIZE_BINS if lo <= s < hi)


def bootstrap_delta(by, groups, a, b, tiny, reps=2000, seed=42):
    """Group-level paired bootstrap of recall(b) - recall(a) over GTs selected by ``tiny``."""
    per_group = defaultdict(lambda: np.zeros(3))
    for (image, _), modes in by.items():
        if a not in modes or b not in modes or not tiny(modes[a]):
            continue
        g = modes[a]["group"]
        per_group[g] += [modes[a]["stage"] == "S4", modes[b]["stage"] == "S4", 1]
    if not per_group:
        return None
    m = np.stack(list(per_group.values()))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(m), size=(reps, len(m)))
    s = m[idx].sum(1)
    d = (s[:, 1] - s[:, 0]) / s[:, 2]
    point = (m[:, 1].sum() - m[:, 0].sum()) / m[:, 2].sum()
    return dict(delta=float(point), ci95=[float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))],
                groups=int(len(m)), gts=int(m[:, 2].sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.25)
    args = parser.parse_args()
    rows, by = load(args.run, args.threshold)
    out = dict(evidence="旧划分探索", threshold=args.threshold, gts=len(by))
    strata = dict(
        size=size_key,
        solidity=lambda r: "n/a" if math.isnan(num(r["solidity"])) else
        "solidity<0.85" if num(r["solidity"]) < 0.85 else "solidity>=0.85",
        gap=lambda r: "no_neighbor" if math.isnan(num(r["gap640"])) else
        "gap<=2px" if num(r["gap640"]) <= 2 else "gap>2px",
        cnr=lambda r: "n/a" if math.isnan(num(r["cnr"])) else "cnr<0.5" if num(r["cnr"]) < 0.5 else "cnr>=0.5",
    )
    out["funnels"] = {}
    for name, key in strata.items():
        table = funnel(rows, key)
        out["funnels"][name] = {"{}|{}".format(m, k): dict(v) for (m, k), v in sorted(table.items())}
    # Paired per-GT transitions between views (same GT, same matching contract).
    pairs = {}
    for a, b in (("global", "fine"), ("global", "upsample_linear"), ("global", "upsample_area"),
                 ("fine", "upsample_linear")):
        t = Counter()
        for modes in by.values():
            if a in modes and b in modes:
                t[(modes[a]["stage"] == "S4", modes[b]["stage"] == "S4")] += 1
        pairs["{}->{}".format(a, b)] = dict(both_hit=t[(True, True)], rescued=t[(False, True)],
                                            lost=t[(True, False)], both_miss=t[(False, False)])
    out["paired_transitions"] = pairs
    # Where do fine-view rescues come from (global stage of GTs that fine rescues)?
    origin = Counter(m["global"]["stage"] for m in by.values()
                     if "global" in m and "fine" in m and m["global"]["stage"] != "S4" and m["fine"]["stage"] == "S4")
    out["fine_rescue_origin_global_stage"] = dict(origin)
    up_origin = Counter(m["global"]["stage"] for m in by.values()
                        if "global" in m and "upsample_linear" in m and m["global"]["stage"] != "S4"
                        and m["upsample_linear"]["stage"] == "S4")
    out["upsample_rescue_origin_global_stage"] = dict(up_origin)
    # s50 per view.
    out["s50"] = {}
    for mode in MODES:
        sel = [m[mode] for m in by.values() if mode in m]
        if sel:
            sizes = np.array([float(r["size640"]) for r in sel])
            hits = np.array([r["stage"] == "S4" for r in sel], float)
            s50, tau = fit_s50(sizes, hits)
            out["s50"][mode] = dict(s50=s50, tau=tau, n=len(sel))
    # Group bootstrap of tiny/small recall deltas.
    tiny = lambda r: float(r["area640"]) < 256  # noqa: E731
    small16 = lambda r: float(r["size640"]) < 16  # noqa: E731
    out["bootstrap"] = {}
    for name, sel in (("tiny_area<256", tiny), ("size<16", small16), ("all", lambda r: True)):
        for b in ("fine", "upsample_linear", "upsample_area"):
            out["bootstrap"]["{}|global->{}".format(name, b)] = bootstrap_delta(by, None, "global", b, sel)
    # rho with group bootstrap using the same resampling for numerator and denominator.
    per_group = defaultdict(lambda: np.zeros(5))
    for modes in by.values():
        if all(m in modes for m in MODES) and tiny(modes["global"]):
            per_group[modes["global"]["group"]] += [modes[m]["stage"] == "S4" for m in MODES] + [1]
    m = np.stack(list(per_group.values()))
    rng = np.random.default_rng(7)
    idx = rng.integers(0, len(m), size=(2000, len(m)))
    s = m[idx].sum(1)
    out["rho_tiny"] = {}
    for j, mode in ((2, "upsample_linear"), (3, "upsample_area")):
        den = s[:, 1] - s[:, 0]
        gain = s[:, j] - s[:, 0]
        valid = den > 0
        rho = gain[valid] / den[valid]
        point_den = m[:, 1].sum() - m[:, 0].sum()
        out["rho_tiny"][mode] = dict(
            point=float((m[:, j].sum() - m[:, 0].sum()) / point_den) if point_den > 0 else None,
            ci95=[float(np.quantile(rho, 0.025)), float(np.quantile(rho, 0.975))] if valid.any() else None,
            invalid_denominator=int((~valid).sum()), replicates=2000, groups=int(len(m)), gts=int(m[:, 4].sum()))
    (args.run / "stratified.json").write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
