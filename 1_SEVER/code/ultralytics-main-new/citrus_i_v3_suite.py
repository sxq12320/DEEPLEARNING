"""I V3 controlled ablations: narrow achromatic stream, then isolated loss/head tests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/I_V3_series"
FACTORS = {
    "I30_rgb_control": dict(view=-1, p3=False, gate=1, nwd=0.0, tiny=0.25, assign=0.0, route=2),
    "I31_luma_p2": dict(view=0, p3=False, gate=1, nwd=0.0, tiny=0.25, assign=0.0, route=2),
    "I32_contrast_p2": dict(view=1, p3=False, gate=1, nwd=0.0, tiny=0.25, assign=0.0, route=2),
    "I33_contrast_p23": dict(view=1, p3=True, gate=1, nwd=0.0, tiny=0.25, assign=0.0, route=2),
    "I34_gradient_p23": dict(view=2, p3=True, gate=1, nwd=0.0, tiny=0.25, assign=0.0, route=2),
    "I35_direct_p23": dict(view=1, p3=True, gate=0, nwd=0.0, tiny=0.25, assign=0.0, route=2),
    "I36_nwd_p23": dict(view=1, p3=True, gate=1, nwd=0.1, tiny=0.25, assign=0.0, route=2),
    "I37_tinydice_p23": dict(view=1, p3=True, gate=1, nwd=0.0, tiny=0.40, assign=0.0, route=2),
    "I38_assign_p23": dict(view=1, p3=True, gate=1, nwd=0.0, tiny=0.25, assign=0.2, route=2),
    "I39_norecog_p23": dict(view=1, p3=True, gate=1, nwd=0.0, tiny=0.25, assign=0.0, route=0),
}
NAMES = tuple(FACTORS)
RUN_OVERRIDES = {
    name: dict(mask_ratio=2, nwd_ratio=factor["nwd"], copy_paste=0.3, cos_lr=False)
    for name, factor in FACTORS.items()
}
SUITES = {
    "all": NAMES,
    "screen": NAMES[:6],
    "priority": NAMES[:5],
    "control": NAMES[:3],
    "losses": (NAMES[3], NAMES[6], NAMES[7], NAMES[8]),
    "recognition": (NAMES[3], NAMES[9]),
    "smoke": NAMES[:3],
}


def select_names(suite, only=""):
    """Return exact distinct model stems for one sequential run."""
    selected = [s.strip() for s in only.split(",") if s.strip()] if only else list(SUITES[suite])
    if not selected or len(selected) != len(set(selected)) or set(selected) - set(NAMES):
        raise ValueError(f"Choose distinct names from {NAMES}")
    return selected
