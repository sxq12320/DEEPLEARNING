"""SAGE-v5 ablations. screen is a complete relay x late-prototype 2x2 plus YOLO control."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "SAGE_series"
NAMES = (
    "SAGE30_official_control",
    "SAGE42_asym_semantic_detail",
    "SAGE50_late_proto",
    "SAGE51_detail_relay",
    "SAGE52_dual_route",
    "SAGE53_dual_boundary",
    "SAGE54_dual_neighbor",
    "SAGE55_dual_geometry",
    "SAGE56_dual_wt_p5",
)
SUITES = {
    "screen": NAMES[:5],
    "structure": NAMES[:5],
    "geometry": NAMES[4:8],
    "backbone": (NAMES[4], NAMES[8]),
    "control": NAMES[:2],
    "all": NAMES,
    "smoke": NAMES,
}


def select_names(suite, only=""):
    if not only:
        return list(SUITES[suite])
    names = [value.strip() for value in only.split(",") if value.strip()]
    if not names or len(names) != len(set(names)) or set(names) - set(NAMES):
        raise ValueError(f"Use distinct exact names from {NAMES}")
    return names
