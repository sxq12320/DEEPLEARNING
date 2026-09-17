"""V11 controlled arms: retain V10 references; test architecture before compound losses."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V11_series"
NAMES = [
    "V11_00_v10_structure_anchor",
    "V11_01_v10_tiny_reference",
    "V11_02_persistent_detail",
    "V11_03_repmix_backbone",
    "V11_04_native_neck",
    "V11_05_tiny_assignment",
    "V11_06_integrated",
    "V11_07_factorized_box_tower",
    "V11_08_local_mask_contrast",
    "V11_09_contrast_complete",
    "V11_10_gap_context_control",
    "V11_11_selective_context",
]
# contrast stem, persistent detail, rep C3/C4/C5, native neck, NWD/TAL mix, ring, tiny Dice, box factorization
FACTORS = [
    (1, 0, 0, 0, 0.0, 0.0, 0.25, 0),
    (0, 0, 0, 0, 0.0, 0.0, 0.25, 0),
    (0, 1, 0, 0, 0.0, 0.0, 0.25, 0),
    (0, 1, 1, 0, 0.0, 0.0, 0.25, 0),
    (0, 1, 1, 1, 0.0, 0.0, 0.25, 0),
    (0, 0, 0, 0, 0.2, 0.0, 0.25, 0),
    (0, 1, 1, 1, 0.2, 0.0, 0.25, 1),
    (0, 1, 1, 1, 0.0, 0.0, 0.25, 1),
    (0, 1, 1, 1, 0.2, 0.05, 0.25, 1),
    (1, 1, 1, 1, 0.2, 0.0, 0.25, 1),
    (0, 1, 1, 1, 0.2, 0.0, 0.25, 1),
    (0, 1, 1, 1, 0.2, 0.0, 0.25, 1),
]
# A paired hypothesis test, NOT an untested change to the integrated main arm.
CONTEXT = {NAMES[10]: False, NAMES[11]: True}
RUN_OVERRIDES = {n: dict(mask_ratio=2, nwd_ratio=0.0, copy_paste=0.3, cos_lr=False) for n in NAMES}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(
    control=tuple(NAMES[:2]),
    priority=tuple(NAMES[i] for i in (0, 2, 3, 4, 5, 6, 7)),
    losses=tuple(NAMES[i] for i in (0, 5, 6, 7, 8, 9)),
    paper=tuple(NAMES[i] for i in (6, 10, 11)),
)


def select_names(suite, only=""):
    selected = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return selected
