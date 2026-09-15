"""Eight controlled tests of scale exposure, mask routing, regional supervision."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V5_series"
NAMES = [
    "V5_00_control",
    "V5_01_fine",
    "V5_02_mask_route",
    "V5_03_region",
    "V5_04_fine_route",
    "V5_05_fine_region",
    "V5_06_route_region",
    "V5_07_full",
]
# T = multi-scale input, R = mask-specific routing, C = training-only region loss.
FACTORS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
FINE_MODELS = {n for n, f in zip(NAMES, FACTORS) if f[0]}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(control=(NAMES[0],), priority=tuple(NAMES[:4]), combined=tuple(NAMES[4:]))
GUIDED = set()
TILE_PROBABILITY = {n: 0.5 for n in NAMES}
TILE_FRACTION = 0.6


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
