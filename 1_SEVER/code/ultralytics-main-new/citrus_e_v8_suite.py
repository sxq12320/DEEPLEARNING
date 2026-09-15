"""Ten evidence-led E V8 ablations. Same optimizer, data recipe and mask evaluation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V8_series"
NAMES = [
    "V8_00_phase_control",
    "V8_01_geometry",
    "V8_02_copy_paste",
    "V8_03_geometry_cp",
    "V8_04_context_backbone",
    "V8_05_native_neck",
    "V8_06_structure_pair",
    "V8_07_structure_geometry",
    "V8_08_structure_complete",
    "V8_09_dense_complete",
]
# backbone, neck, geometry, copy-paste, dense decoder. Not a full factorial.
FACTORS = [
    (0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 0, 0, 1, 0),
    (0, 0, 1, 1, 0),
    (1, 0, 0, 0, 0),
    (0, 1, 0, 0, 0),
    (1, 1, 0, 0, 0),
    (1, 1, 1, 0, 0),
    (1, 1, 1, 1, 0),
    (1, 1, 1, 1, 1),
]
RUN_OVERRIDES = {
    n: {"mask_ratio": 2, "nwd_ratio": 0.0, "copy_paste": 0.3 if f[3] else 0.0} for n, f in zip(NAMES, FACTORS)
}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(
    control=(NAMES[0],), priority=(NAMES[0], NAMES[3], NAMES[4], NAMES[5], NAMES[8]), combined=tuple(NAMES[6:])
)


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
