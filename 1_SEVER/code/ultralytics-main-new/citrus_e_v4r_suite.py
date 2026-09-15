"""Eight E V4R factorial tests. The prior E40--E77 designs remain untouched."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V4_series/reconstruction_20260910"
NAMES = [
    "V4R00_control",
    "V4R01_deep8",
    "V4R02_detail",
    "V4R03_quality",
    "V4R04_deep_detail",
    "V4R05_deep_quality",
    "V4R06_detail_quality",
    "V4R07_full",
]
# ONLY deepest stage 8; narrow semantic-conditioned P2 correction; bounded ranking.
FACTORS = [(0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1), (1, 1, 1)]
SUITES = {key: tuple(NAMES) for key in ("all", "screen", "structure", "smoke")}
SUITES.update(control=(NAMES[0],), priority=tuple(NAMES[:4]), combined=tuple(NAMES[4:]))
GUIDED = set()
TILE_PROBABILITY = {name: 0.5 for name in NAMES}
TILE_FRACTION = 0.6


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
