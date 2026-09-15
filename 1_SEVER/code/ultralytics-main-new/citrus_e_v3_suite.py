"""Eight explicit 2x2x2 architecture factors; identical source-balanced input."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V3_series"
NAMES = ["E30_control", "E31_native_neck", "E32_deep_partial", "E33_mask_quality",
         "E34_deep_neck", "E35_neck_quality", "E36_deep_quality", "E37_full"]
# Factors: deep backbone, native/phase neck, mask-quality ranking.
FACTORS = [(0,0,0),(0,1,0),(1,0,0),(0,0,1),(1,1,0),(0,1,1),(1,0,1),(1,1,1)]
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
