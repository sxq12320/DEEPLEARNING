"""I V2: correctness-first tiny-candidate recovery ablations."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/I_V2_series"
NAMES = [
    "I20_corrected_control",
    "I21_corrected_no_recog",
    "I22_p2_raw",
    "I23_p2_semantic",
    "I24_p2_largesmall",
    "I25_ls_c3",
    "I26_ls_c34",
    "I27_ls_c34_p2_semantic",
    "I28_p2_semantic_no_recog",
    "I29_p2_semantic_tinyassign",
]

# p2_mode: -1=no P2, 0=raw detail, 1=semantic filter, 2=large-small filter.
# ls_depth: 0=none, 1=C3 (stride 8), 2=C3+C4 (strides 8/16).
# assignment_mix is a training-only tiny-GT NWD blend inherited from V11.
FACTORS = {
    "I20_corrected_control": dict(p2_mode=-1, ls_depth=0, route=2, assignment_mix=0.0),
    "I21_corrected_no_recog": dict(p2_mode=-1, ls_depth=0, route=0, assignment_mix=0.0),
    "I22_p2_raw": dict(p2_mode=0, ls_depth=0, route=2, assignment_mix=0.0),
    "I23_p2_semantic": dict(p2_mode=1, ls_depth=0, route=2, assignment_mix=0.0),
    "I24_p2_largesmall": dict(p2_mode=2, ls_depth=0, route=2, assignment_mix=0.0),
    "I25_ls_c3": dict(p2_mode=-1, ls_depth=1, route=2, assignment_mix=0.0),
    "I26_ls_c34": dict(p2_mode=-1, ls_depth=2, route=2, assignment_mix=0.0),
    "I27_ls_c34_p2_semantic": dict(p2_mode=1, ls_depth=2, route=2, assignment_mix=0.0),
    "I28_p2_semantic_no_recog": dict(p2_mode=1, ls_depth=0, route=0, assignment_mix=0.0),
    "I29_p2_semantic_tinyassign": dict(p2_mode=1, ls_depth=0, route=2, assignment_mix=0.2),
}
RUN_OVERRIDES = {name: dict(mask_ratio=2, nwd_ratio=0.0, copy_paste=0.3, cos_lr=False) for name in NAMES}
SUITES = {
    "all": tuple(NAMES),
    "screen": tuple(NAMES),
    "smoke": tuple(NAMES),
    "priority": tuple(NAMES[i] for i in (0, 1, 2, 3, 4)),
    "control": tuple(NAMES[:2]),
    "p2": tuple(NAMES[i] for i in (0, 2, 3, 4)),
    "backbone": tuple(NAMES[i] for i in (0, 5, 6)),
    "combined": tuple(NAMES[i] for i in (0, 3, 6, 7)),
    "assignment": tuple(NAMES[i] for i in (3, 9)),
}


def select_names(suite, only=""):
    """Resolve one deterministic queue and reject duplicates/unknown stems."""
    selected = [name.strip() for name in only.split(",") if name.strip()] if only else list(SUITES[suite])
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return selected
