"""V7 candidate-resolution ablations. No data or optimizer changes."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/SAGE_V7_series"
NAMES = [
    "SAGE70_relay_control",
    "SAGE71_compact_candidates",
    "SAGE72_p2_candidates",
    "SAGE73_p2_no_relay",
    "SAGE74_p2_local_context",
    "SAGE75_detail_bypass",
    "SAGE76_shared_context",
    "SAGE77_task_routed_context",
    "SAGE78_single_scale_route",
]
SUITES = {key: tuple(NAMES) for key in ("all", "smoke")}
# Existing suites retain their original meaning; new results get a new project.
SUITES.update(screen=tuple(NAMES[:5]), structure=tuple(NAMES[:5]))
SUITES.update(refusion=(NAMES[0], NAMES[2], *NAMES[5:]), refusion_new=tuple(NAMES[5:]))
SUITES.update(control=(NAMES[0],), priority=tuple(NAMES[:3]))


def select_names(suite, only=""):
    names = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not names or len(names) != len(set(names)) or set(names) - set(NAMES):
        raise ValueError(f"Choose distinct names from {NAMES}")
    return names
