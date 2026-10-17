"""Explicit, paired J6 experiment ladder; no hidden teacher or dataset dependency."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/J_V6_series"
NAMES = (
    "J90_global_control",
    "J91_shared_unary",
    "J92_neighbor",
    "J93_unknown_aware_selection",
    "J94_full",
    "J95_shuffled_neighbor",
    "J96_no_d2",
    "J97_tiny_balanced",
    "J98_unary_gain",
    "J99_full_no_gain",
)
SUITES = {
    "all": NAMES,
    "screen": (NAMES[0], NAMES[1], NAMES[2], NAMES[4]),
    "priority": (NAMES[0], NAMES[1], NAMES[2], NAMES[4]),
    "neighbors": (NAMES[1], NAMES[2], NAMES[5]),
    "selection": (NAMES[1], NAMES[3], NAMES[9]),
    "detail": (NAMES[1], NAMES[6]),
    "loss": (NAMES[1], NAMES[7]),
    "gain": (NAMES[1], NAMES[8], NAMES[9], NAMES[4]),
}
DESCRIPTIONS = {
    NAMES[0]: "Global control; auxiliary high-resolution head disabled",
    NAMES[1]: "Shared high-resolution detail and unary instance refinement",
    NAMES[2]: "Unary refinement plus actual neighboring-instance evidence",
    NAMES[3]: "Unary refinement plus unknown-aware candidate selection",
    NAMES[4]: "Full detail, neighbor conditioning and candidate selection",
    NAMES[5]: "Shuffled-neighbor negative control for J92",
    NAMES[6]: "Unary refinement without the stride-2 detail path; paired with J91",
    NAMES[7]: "Unary refinement with per-instance tiny-balanced auxiliary supervision; paired with J91",
    NAMES[8]: "Unary refinement with signed marginal-gain routing; paired with J91",
    NAMES[9]: "Full neighbor/selection model without gain routing; paired with J94",
}


def select_names(suite="priority", only=""):
    """Resolve precisely the requested distinct model names, preserving their order."""
    if suite not in SUITES:
        raise ValueError(f"Unknown J6 suite {suite!r}; choose from {tuple(SUITES)}")
    selected = [name.strip() for name in only.split(",") if name.strip()] if only else list(SUITES[suite])
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(NAMES):
        raise ValueError(f"ONLY must contain distinct J6 names from {NAMES}")
    return selected


def jobs(suite="priority", seeds=(42,), only=""):
    """Generate a deterministic seed-major queue with no unrequested dependency jobs."""
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("SEEDS must contain distinct nonnegative integers")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in seeds):
        raise ValueError("SEEDS must contain distinct nonnegative integers")
    return [(name, seed) for seed in seeds for name in select_names(suite, only)]
