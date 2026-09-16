"""Ten evidence-led V10 arms; 00/01 exactly replay V9_01/V9_05 architectures."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V10_series"
NAMES = [
    "V10_00_geometry_anchor",
    "V10_01_tiny_anchor",
    "V10_02_balanced_decoder",
    "V10_03_contrast_stem",
    "V10_04_detail_transport",
    "V10_05_structure_pair",
    "V10_06_structure_balanced",
    "V10_07_visibility_supervision",
    "V10_08_complete",
    "V10_09_complete_cosine",
]
# contrast stem, detail transport, moderate decoder, tiny overlap, visibility.
FACTORS = [
    (0, 0, 0, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 1, 1, 0),
    (1, 0, 0, 1, 0),
    (0, 1, 0, 1, 0),
    (1, 1, 0, 1, 0),
    (1, 1, 1, 1, 0),
    (0, 0, 0, 1, 1),
    (1, 1, 1, 1, 1),
    (1, 1, 1, 1, 1),
]
RUN_OVERRIDES = {n: dict(mask_ratio=2, nwd_ratio=0.0, copy_paste=0.3, cos_lr=i == 9) for i, n in enumerate(NAMES)}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(
    control=tuple(NAMES[:2]),
    priority=tuple(NAMES[i] for i in (1, 2, 3, 4, 7, 8)),
    losses=tuple(NAMES[i] for i in (0, 1, 7, 8, 9)),
)


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
