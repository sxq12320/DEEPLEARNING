"""E V2 controlled structure comparisons; ordinary YOLO YAML entry points."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V2_series"
NAMES = [
    "E20_fixed_control",
    "E21_contrast_detail",
    "E22_context_hub",
    "E23_rep_backbone",
    "E24_rep_hub",
    "E25_rep_hub_contrast",
    "E26_full_global",
    "E27_full_guided",
]
SUITES = {key: tuple(NAMES) for key in ("all", "screen", "structure", "smoke")}
SUITES.update(priority=(NAMES[0], NAMES[1], NAMES[2], NAMES[5]), control=(NAMES[0],), guided=(NAMES[5], NAMES[7]))
GUIDED = {NAMES[7]}
TILE_PROBABILITY = {n: (0.0 if n == NAMES[6] else 0.5) for n in NAMES}
TILE_FRACTION = 0.6


def select_names(suite, only=""):
    chosen = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return chosen
