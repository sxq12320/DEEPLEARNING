"""I V4: auditable scene-layer/neck ablations, not an asserted best model."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/I_V4_series"
FACTORS = {
    "I40_replay": dict(mode=-1, steps=1, aux=0.0, gray=False),
    "I41_parallel_neck": dict(mode=0, steps=1, aux=0.0, gray=False),
    "I42_discovery_aux": dict(mode=1, steps=1, aux=0.2, gray=False),
    "I43_region_feedback": dict(mode=2, steps=1, aux=0.2, gray=False),
    "I44_boundary_protected": dict(mode=3, steps=1, aux=0.2, gray=False),
    "I45_two_step": dict(mode=3, steps=2, aux=0.2, gray=False),
    "I46_achromatic": dict(mode=3, steps=1, aux=0.2, gray=True),
    "I47_no_layer_supervision": dict(mode=3, steps=1, aux=0.0, gray=False),
}
NAMES = tuple(FACTORS)
RUN_OVERRIDES = {name: dict(mask_ratio=2, nwd_ratio=0.0, copy_paste=0.3, cos_lr=False) for name in NAMES}
SUITES = dict(all=NAMES, priority=NAMES[:5], screen=NAMES[:5], control=NAMES[:2],
              mechanism=(NAMES[4], *NAMES[5:]), smoke=(NAMES[0], NAMES[4]))


def select_names(suite, only=""):
    selected = [s.strip() for s in only.split(",") if s.strip()] if only else list(SUITES[suite])
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(NAMES):
        raise ValueError(f"Choose distinct model stems from {NAMES}")
    return selected
