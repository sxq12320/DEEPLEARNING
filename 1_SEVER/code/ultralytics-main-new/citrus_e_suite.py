"""E00--E07 preserved; E08 adds learned RGB-guided crop placement, not a new segmenter."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_series"
NAMES = [
    "E00_global_control",
    "E01_sliced_control",
    "E02_phase_global",
    "E03_phase_sliced",
    "E04_hybrid_sliced",
    "E05_hybrid_global",
    "E06_phase_context_sliced",
    "E07_phase_topdown_sliced",
    "E08_phase_guided_sliced",
]
SUITES = {key: tuple(NAMES) for key in ("all", "screen", "structure", "smoke")}
SUITES.update(control=(NAMES[0],), priority=tuple(NAMES[:4]), guided=(NAMES[3], NAMES[8]))
GUIDED = {NAMES[8]}
TILE_PROBABILITY = {name: (0.5 if i in (1, 3, 4, 7, 8) else 0.25 if i == 6 else 0.0)
                    for i, name in enumerate(NAMES)}
SLICED = {name for name, probability in TILE_PROBABILITY.items() if probability > 0}
TILE_FRACTION = 0.6


def select_names(suite, only=""):
    names = [x.strip() for x in only.split(",") if x.strip()] if only else list(SUITES[suite])
    if not names or len(set(names)) != len(names) or set(names) - set(NAMES):
        raise ValueError(f"Choose distinct names from {NAMES}")
    return names
