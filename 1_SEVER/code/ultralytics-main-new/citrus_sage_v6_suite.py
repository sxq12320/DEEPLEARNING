"""SAGE V6: protected prototype, then neck/backbone/persistent-detail ablations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "SAGE_V6_series"
NAMES = (
    "SAGE60_relay_control",
    "SAGE61_singlepass_neck",
    "SAGE62_residual_backbone",
    "SAGE63_persistent_detail",
    "SAGE64_selective_exchange",
    "SAGE65_visible_geometry",
)
SUITES = {
    "all": NAMES,
    "screen": NAMES[:5],
    "structure": NAMES[:5],
    "priority": (NAMES[0], NAMES[2], NAMES[4]),
    "geometry": NAMES[4:],
    "backbone": NAMES[1:4],
    "control": NAMES[:1],
    "smoke": NAMES,
}


def select_names(suite, only=""):
    if not only:
        return list(SUITES[suite])
    names = [v.strip() for v in only.split(",") if v.strip()]
    if not names or len(names) != len(set(names)) or set(names) - set(NAMES):
        raise ValueError(f"Use distinct exact names from {NAMES}")
    return names
