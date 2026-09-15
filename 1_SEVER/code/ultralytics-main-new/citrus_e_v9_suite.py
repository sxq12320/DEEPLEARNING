"""Ten E V9 arms: preserve the detector, separate mask fusion and loss factors."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V9_series"
NAMES = [
    "V9_00_phase_control",
    "V9_01_geometry_cp_anchor",
    "V9_02_mask_neck",
    "V9_03_compact_basis",
    "V9_04_dual_route",
    "V9_05_tiny_overlap",
    "V9_06_dual_tiny",
    "V9_07_background_quality",
    "V9_08_complete",
    "V9_09_complete_cosine",
]
# mask neck, compact basis, tiny Dice, negative-quality supervision.
# All except 00 use the V8_03 geometry+CP recipe. 09 changes ONLY the LR schedule vs 08.
FACTORS = [
    (0, 0, 0, 0),
    (0, 0, 0, 0),
    (1, 0, 0, 0),
    (0, 1, 0, 0),
    (1, 1, 0, 0),
    (0, 0, 1, 0),
    (1, 1, 1, 0),
    (0, 0, 0, 1),
    (1, 1, 1, 1),
    (1, 1, 1, 1),
]
RUN_OVERRIDES = {
    n: {"mask_ratio": 2, "nwd_ratio": 0.0, "copy_paste": 0.0 if i == 0 else 0.3, "cos_lr": i == 9}
    for i, n in enumerate(NAMES)
}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(
    control=(NAMES[0], NAMES[1]),
    priority=tuple(NAMES[i] for i in (1, 2, 3, 4, 8)),
    losses=tuple(NAMES[i] for i in (1, 5, 7, 8, 9)),
)


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
