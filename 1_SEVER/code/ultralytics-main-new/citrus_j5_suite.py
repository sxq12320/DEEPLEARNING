"""J5: evidence-preserving compact students and GT-supported task distillation."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/J_V5_series"
NAMES = (
    "J80_yolo_control", "J81_v12_teacher", "J82_compact_control",
    "J83_compact_mask_kd", "J84_compact_joint_kd", "J85_thin_joint_kd",
)
GRAPH_SOURCES = dict(zip(NAMES, (
    "J60_yolo_control", "J61_v12_replay", "J65_shared_towers",
    "J65_shared_towers", "J65_shared_towers", "J67_thin_detail",
)))
KD = {
    NAMES[0]: dict(kd_mask=0.0, kd_cls=0.0),
    NAMES[1]: dict(kd_mask=0.0, kd_cls=0.0),
    NAMES[2]: dict(kd_mask=0.0, kd_cls=0.0),
    NAMES[3]: dict(kd_mask=0.5, kd_cls=0.0),
    NAMES[4]: dict(kd_mask=0.5, kd_cls=0.1),
    NAMES[5]: dict(kd_mask=0.5, kd_cls=0.1),
}
KD_COMMON = dict(warmup_epochs=10, ramp_epochs=20, point_grid=12, max_instances=32,
                 teacher_min_iou=0.5, teacher_min_score=0.25)
SUITES = {
    "all": NAMES, "screen": (NAMES[0], NAMES[1], NAMES[2], NAMES[4]),
    "priority": (NAMES[0], NAMES[1], NAMES[2], NAMES[4]),
    "controls": NAMES[:3], "distill": NAMES[2:5], "thin": (NAMES[4], NAMES[5]),
}


def select_names(suite="priority", only=""):
    """Resolve an explicit queue without silently adding unselected experiments."""
    chosen = [value.strip() for value in only.split(",") if value.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError(f"Choose distinct J5 names from {NAMES}")
    return chosen


def jobs(suite="priority", seeds=(42,), only="", teacher="auto"):
    """Put the frozen teacher first; add its seed42 dependency for automatic KD."""
    selected = select_names(suite, only)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("SEEDS must contain distinct integer seeds")
    if any(not isinstance(seed, int) or isinstance(seed, bool) or seed < 0 for seed in seeds):
        raise ValueError("SEEDS must be nonnegative integers")
    queue = [(name, seed) for seed in seeds for name in selected]
    needs_teacher = any(KD[name]["kd_mask"] or KD[name]["kd_cls"] for name in selected)
    dependency = (NAMES[1], 42)
    if needs_teacher and teacher == "auto" and dependency not in queue:
        queue.insert(0, dependency)
    # All teacher seeds precede students; all students use only the fixed seed42 teacher.
    return [item for item in queue if item[0] == NAMES[1]] + [item for item in queue if item[0] != NAMES[1]]
