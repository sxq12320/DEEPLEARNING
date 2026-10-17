"""J4 evidence-preserving compression ladder; no measured accuracy claim."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/J_V4_series"
NAMES = (
    "J60_yolo_control", "J61_v12_replay", "J62_no_pan_return", "J63_projected_direct",
    "J64_narrow_semantics", "J65_shared_towers", "J66_without_auxiliary", "J67_thin_detail",
)
DESCRIPTIONS = dict(zip(NAMES, (
    "Official YOLO11n graph, controlled views and EV6; no custom geometry objectives",
    "V12_04 graph replay, scratch; historical gains are not copied as new results",
    "J61 minus P3-down/N4 bottom-up return; P4=N4, C5 directly reused",
    "J62 with two source-resolution projected residual fusions instead of concat/reconstruction",
    "J63 deep semantic width 256->192; other paths, decoder and losses unchanged",
    "J64 plus three task-specific narrow spatial towers shared across scales",
    "J65 with tiny/boundary/neighbor/quality auxiliaries off (objective+ranking joint control)",
    "J65 with persistent detail width 16->12; discovery candidate grids unchanged",
)))
SUITES = {"screen": NAMES, "priority": (NAMES[0], NAMES[1], NAMES[4], NAMES[5]), "all": NAMES,
          "cost": NAMES[1:6], "loss": (NAMES[5], NAMES[6])}


def select_names(suite="priority", only=""):
    selected = [n.strip() for n in only.split(",") if n.strip()] if only else list(SUITES[suite])
    if not selected or len(set(selected)) != len(selected) or any(n not in NAMES for n in selected):
        raise ValueError("Choose distinct J4 model IDs")
    return selected
