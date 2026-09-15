"""Eight controlled V7 arms using the V6 phase-fine decoder as a candidate base.

V6 suggests better strict-IoU masks with fine decoding, not a large AP50 gain.
Phase trades some dense-decoder accuracy/tiny recall for promising cost and
precision. Single-seed timing differences are not proof of GPU acceleration.
V7_00 matches V6_08's graph; all arms share phase mode and mask_ratio=2.

NWD is disabled because its observed effects depend on the decoder and budget,
not because a statistical equivalence/noise band has been established.
V7_05_cp alone tests copy_paste=0.3; copy-paste improved some V6 recall metrics
but can increase background false positives and training time.

Factors: P2 detection scale, DCNv2 detail sampling, PMCE pooled-frequency
residual, paired boundary/neighbor supervision. V7_06 combines P2 and PMCE.
The full arm excludes copy-paste. These eight runs are not a full factorial.

Small-GT candidate expansion exists in TAL, but cannot rule out assignment
competition. EV7SegmentationLoss fixes its scale metadata for appended P2:
[8,16,32,4] tower order must not masquerade as sorted spatial stride order.
No feature/tower reordering, learned crop guide, new optimizer, convex prior,
or claim that this fix guarantees tiny-object gains is made.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V7_series"
NAMES = [
    "V7_00_control",
    "V7_01_p2",
    "V7_02_deform",
    "V7_03_pmce",
    "V7_04_boundary",
    "V7_05_cp",
    "V7_06_p2_pmce",
    "V7_07_full",
]
# S = stride-4 detection scale, D = deformable detail, C = channel enhancement,
# G = geometry (boundary+neighbor) supervision. V7_05_cp shares the control
# graph -- its difference is the copy_paste override below, not the head args.
FACTORS = [(0, 0, 0, 0), (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1), (0, 0, 0, 0), (1, 0, 1, 0), (1, 1, 1, 1)]
P2_MODELS = {n for n, f in zip(NAMES, FACTORS) if f[0]}
# Geometry is an explicit paired loss intervention, not an architecture change.
# mask_ratio=2 belongs to the adopted phase-fine base (stride-2 GT supervision),
# not to any V7 factor. nwd stays at its V6-measured neutral setting. V7_05_cp is
# the second recall lever: copy_paste=0.3 matched V6's best tiny recall (.224)
# at zero inference cost; other arms keep it off so its effect is attributable.
RUN_OVERRIDES = {
    name: {"mask_ratio": 2, "nwd_ratio": 0.0, "copy_paste": 0.3 if name == "V7_05_cp" else 0.0}
    for name in NAMES
}
# Predeclared (not fitted) geometry weights: the boundary band term enters at
# half strength and the neighbor repulsion at quarter strength relative to the
# per-instance mask BCE, matching the magnitude convention documented in
# sage_v4r_loss.py (gains are effective weights, not box-gain scaled).
BOUNDARY_GAIN = 0.5
NEIGHBOR_GAIN = 0.25
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(control=(NAMES[0],), priority=tuple(NAMES[:5]), combined=tuple(NAMES[5:]))
TILE_PROBABILITY = {n: 0.5 for n in NAMES}
TILE_FRACTION = 0.6


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
