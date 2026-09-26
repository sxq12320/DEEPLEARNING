"""I V1 convergent ablations: two V12 replay anchors and a synchronized dual-prototype hypothesis."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/I_V1_series"
V12_YAML_DIR = ROOT / "0_orange_yaml/E_V12_series"
V12_PARENTS = ["V12_03_recognition_route", "V12_04_shared_route_control"]
NAMES = [
    "I00_replay_recognition",
    "I01_replay_shared",
    "I02_dual_plain",
    "I03_dual_msca",
    "I04_sync_msca",
    "I05_sync_caa",
    "I06_sync_feedback",
    "I07_sync_semlead",
    "I08_semantic_only",
    "I09_sync_no_recog",
]
# V12 parent index; recognition route 0/off 1/cls 2/shared; dual proto 0..3;
# context 0/plain 1/strip 2/anchor; discrepancy feedback; initial scalar lead
FACTORS = [
    (0, 1, 0, 1, 0, 0.0),  # exact V12_03 replay under the new head class
    (1, 2, 0, 1, 0, 0.0),  # exact V12_04 replay
    (0, 1, 1, 0, 0, 0.0),  # second prototype, plain projection, scalar mix
    (0, 1, 1, 1, 0, 0.0),  # + strip (MSCA-adapted) context, still scalar mix
    (0, 1, 2, 1, 0, 0.0),  # + per-position arbitration gate  (main hypothesis)
    (0, 1, 2, 2, 0, 0.0),  # anchor (CAA-adapted) context instead of strip
    (0, 1, 2, 1, 1, 0.0),  # + bounded prototype-discrepancy feedback
    (0, 1, 2, 1, 0, 1.0),  # mixture initialized toward the semantic side
    (0, 1, 3, 1, 0, 0.0),  # semantic prototype replaces the detail prototype
    (0, 0, 2, 1, 0, 0.0),  # gated sync without the classification route
]
RUN_OVERRIDES = {n: dict(mask_ratio=2, nwd_ratio=0.0, copy_paste=0.3, cos_lr=False) for n in NAMES}
SUITES = {s: tuple(NAMES) for s in ("all", "screen", "structure", "smoke")}
SUITES.update(
    priority=tuple(NAMES[i] for i in (0, 1, 4, 5)),
    control=tuple(NAMES[:2]),
    mechanism=tuple(NAMES[i] for i in (0, 1, 2, 3, 4)),  # replay -> plain -> context -> gate
    feedback=tuple(NAMES[i] for i in (4, 6)),
    losses=tuple(NAMES[i] for i in (4, 6)),  # historical alias; NOT a loss-function ablation
)


def select_names(suite, only=""):
    selected = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(NAMES):
        raise ValueError(f"Choose distinct model names from {NAMES}")
    return selected
