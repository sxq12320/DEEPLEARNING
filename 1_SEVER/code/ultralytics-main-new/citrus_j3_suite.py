"""J V3 factors: privileged high-resolution teacher + learning under incomplete annotation.

Every arm deploys the SAME graph (YOLO11n-seg, identical to J20/J00): parameters, FLOPs and latency are unchanged.
Arms differ only in how teacher-mined instances enter training (``pseudo``) and which teacher produced them
(``teacher``: ``fine`` = original-pixel 3x3 views, ``global`` = one 640 pass, no extra pixels). The teacher is the
J50 anchor of the same project (seed TEACHER_SEED), frozen for all student seeds. Evidence behind the design is
exploratory (old split, one historical model); see docs/J_V3_DESIGN_20261007.md.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/J_V3_series"
TEACHER_ARM = "J50_noisy_anchor"
TEACHER_SEED = 42
NAMES = [
    "J50_noisy_anchor",
    "J51_ignore_only",
    "J52_privileged_hard",
    "J53_privileged_soft",
    "J54_global_teacher_soft",
    "J55_naive_selftrain",
    "J56_privileged_soft_960",
    "J59_clean_label_oracle",
]
_BASE = dict(pseudo="none", teacher=None, oracle=False, imgsz=640)
FACTORS = {
    "J50_noisy_anchor": dict(_BASE),
    "J51_ignore_only": dict(_BASE, pseudo="ignore", teacher="fine"),
    "J52_privileged_hard": dict(_BASE, pseudo="hard", teacher="fine"),
    "J53_privileged_soft": dict(_BASE, pseudo="soft", teacher="fine"),
    "J54_global_teacher_soft": dict(_BASE, pseudo="soft", teacher="global"),
    "J55_naive_selftrain": dict(_BASE, pseudo="naive", teacher="fine"),
    "J56_privileged_soft_960": dict(_BASE, pseudo="soft", teacher="fine", imgsz=960),
    "J59_clean_label_oracle": dict(_BASE, oracle=True),
}
DESCRIPTIONS = {
    "J50_noisy_anchor": "YOLO11n-seg on the released (incomplete) labels; anchor AND frozen teacher source",
    "J51_ignore_only": "every teacher candidate becomes an ignore region: stop punishing likely-real fruit only",
    "J52_privileged_hard": "fine-teacher corroborated high tier = full fruit (w=1); mid tier ignored",
    "J53_privileged_soft": "MAIN: fine-teacher high tier weighted by teacher confidence; mid tier ignored",
    "J54_global_teacher_soft": "control: J53 policy, teacher sees one 640 pass (no privileged pixels)",
    "J55_naive_selftrain": "control: every fine-teacher candidate >= .25 as full fruit, no ignore band",
    "J56_privileged_soft_960": "optional cost arm: J53 at imgsz 960 (resolution, not architecture)",
    "J59_clean_label_oracle": "upper bound: anchor graph on human-cleaned TRAIN labels (needs ORACLE_DATA)",
}
SUITES = {
    "all": tuple(NAMES),
    "main": ("J50_noisy_anchor", "J51_ignore_only", "J52_privileged_hard", "J53_privileged_soft",
             "J54_global_teacher_soft", "J55_naive_selftrain"),
    "ladder": ("J50_noisy_anchor", "J51_ignore_only", "J52_privileged_hard", "J53_privileged_soft"),
    "controls": ("J54_global_teacher_soft", "J55_naive_selftrain"),
    "final": ("J50_noisy_anchor", "J53_privileged_soft"),
    "oracle": ("J59_clean_label_oracle",),
    "resolution": ("J56_privileged_soft_960",),
    "smoke": ("J50_noisy_anchor", "J53_privileged_soft"),
}
# RAM: prepared views x imgsz^2 x 3 bytes. Cache is speed-only; 960 arms read from disk.
CACHE = {name: FACTORS[name]["imgsz"] <= 640 for name in NAMES}


def select_names(suite="main", only=""):
    chosen = [s.strip() for s in only.split(",") if s.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError("Choose distinct model names from citrus_j3_suite.NAMES")
    return chosen


def needs_teacher(name):
    return FACTORS[name]["pseudo"] != "none"
