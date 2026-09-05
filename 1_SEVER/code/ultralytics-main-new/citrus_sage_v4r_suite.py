"""Explicit reconstructed-v4 ablation matrix; old SAGE30--35 stay unchanged."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml" / "SAGE_series"
CONTROL = "SAGE30_official_control"
NAMES = (
    CONTROL,
    "SAGE40_asym_control",
    "SAGE41_asym_direct_detail",
    "SAGE42_asym_semantic_detail",
    "SAGE43_asym_reprojection",
    "SAGE44_asym_boundary",
    "SAGE45_asym_neighbor",
    "SAGE46_asym_geometry",
    "SAGE47_asym_faster_p4",
    "SAGE48_asym_gated_p4",
)
SUITES = {
    "screen": NAMES[:5],
    "structure": NAMES[:5],
    "geometry": NAMES[4:8],  # Complete 2x2 boundary x neighbor design at ONE fixed architecture.
    "backbone": (NAMES[4], NAMES[8], NAMES[9]),
    "control": NAMES[:1],
    "all": NAMES,
    "smoke": NAMES,
}


def select_names(suite, only=""):
    """Select exact names; a final winner is deliberately NOT predeclared."""
    if not only:
        return list(SUITES[suite])
    names = [value.strip() for value in only.split(",") if value.strip()]
    if not names or len(names) != len(set(names)) or set(names) - set(NAMES):
        raise ValueError(f"Use distinct exact names from {NAMES}")
    return names
