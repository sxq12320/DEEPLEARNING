"""Evidence-led V8 ablations; all use the same fixed RAM-cache protocol."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/SAGE_V8_series"
NAMES = [
    "SAGE80_relay_control",
    "SAGE81_decoupled_p2",
    "SAGE82_scale_budget",
    "SAGE83_phase_scale",
    "SAGE84_phase_control",
]
SUITES = {key: tuple(NAMES) for key in ("all", "smoke", "screen", "structure")}
SUITES.update(control=(NAMES[0],), priority=(NAMES[0], NAMES[2], NAMES[3]))


def select_names(suite, only=""):
    names = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not names or len(names) != len(set(names)) or set(names) - set(NAMES):
        raise ValueError(f"Choose distinct names from {NAMES}")
    return names
