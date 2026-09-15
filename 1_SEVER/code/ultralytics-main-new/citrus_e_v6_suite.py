"""Ten tests of fine mask decoding, tiny-object regression and instance diversity.

V6 arms all train on the V5_01 input recipe (multi-scale view sampling). The
single-variable factors are F = stride-2 prototype refinement, N = size-gated
NWD/CIoU blend, P = flip-mode instance copy-paste. V6_01 additionally isolates
the mask_ratio 4->2 supervision granularity that F requires, so the fine-mask
path is never confounded with its raster setting.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V6_series"
NAMES = [
    "V6_00_control",
    "V6_01_mr2",
    "V6_02_fine",
    "V6_03_nwd",
    "V6_04_cp",
    "V6_05_fine_nwd",
    "V6_06_fine_cp",
    "V6_07_full",
    "V6_08_phase",
    "V6_09_phase_nwd",
]
# F = stride-2 fine mask path (implies mask_ratio=2), N = nwd_ratio .5, P = copy_paste .3.
FACTORS = [
    (0, 0, 0), (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1),
    (1, 1, 0), (1, 0, 1), (1, 1, 1), (1, 0, 0), (1, 1, 0),
]
PHASE_MODELS = {"V6_08_phase", "V6_09_phase_nwd"}
FINE_MASK_MODELS = {n for n, f in zip(NAMES, FACTORS) if f[0]}
# Per-run hyperparameter deltas over the locked protocol. mask_ratio=2 is a
# supervision-raster change, not an architecture switch; V6_01 tests it alone.
RUN_OVERRIDES = {
    name: {
        "mask_ratio": 2 if factors[0] or name == "V6_01_mr2" else 4,
        "nwd_ratio": 0.5 if factors[1] else 0.0,
        "copy_paste": 0.3 if factors[2] else 0.0,
    }
    for name, factors in zip(NAMES, FACTORS)
}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(
    control=(NAMES[0],),
    priority=tuple(NAMES[i] for i in (0, 1, 2, 3, 8)),
    combined=tuple(NAMES[4:]),
)
TILE_PROBABILITY = {n: 0.5 for n in NAMES}
TILE_FRACTION = 0.6


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
