"""V12 convergent ablations: three replay anchors and a task-routed recognition hypothesis."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/E_V12_series"
NAMES = [
    "V12_00_contrast_anchor",
    "V12_01_detail_anchor",
    "V12_02_persistent_anchor",
    "V12_03_recognition_route",
    "V12_04_shared_route_control",
    "V12_05_recognition_lite_box",
    "V12_06_recognition_contrast",
    "V12_07_head_detail_route",
    "V12_08_sum_evidence_control",
    "V12_09_without_tiny_dice",
]
# V11 parent index; routing 0/off 1/cls 2/shared; difference; box; stem; tiny Dice
FACTORS = [
    (0, 0, True, False, True, 0.25),
    (1, 0, True, False, False, 0.25),
    (2, 0, True, False, False, 0.25),
    (2, 1, True, False, False, 0.25),
    (2, 2, True, False, False, 0.25),
    (2, 1, True, True, False, 0.25),
    (2, 1, True, False, True, 0.25),
    (1, 1, True, False, False, 0.25),
    (2, 1, False, False, False, 0.25),
    (2, 1, True, False, False, 0.0),
]
RUN_OVERRIDES = {n: dict(mask_ratio=2, nwd_ratio=0.0, copy_paste=0.3, cos_lr=False) for n in NAMES}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(
    priority=tuple(NAMES[i] for i in (0, 1, 2, 3, 4, 5)),
    control=tuple(NAMES[:3]),
    losses=tuple(NAMES[i] for i in (3, 9)),
)


def select_names(suite, only=""):
    selected = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return selected
