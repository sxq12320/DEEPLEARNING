"""J V2 factors: retained gaze controls, structural controls and decoder repairs.

YAMLs define graphs; runner factors are explicit and recorded per arm. Historical
miss distributions and contrast measurements came from a different diagnostic
protocol; they are not causal guarantees or current formal-data measurements.
Resolution, resampling, density guidance and structural repairs stay separate.
"""

from pathlib import Path
from citrus_j2_structure_suite import STRUCTURE_DESCRIPTIONS, STRUCTURE_FACTORS, STRUCTURE_NAMES

ROOT = Path(__file__).resolve().parent
YAML_DIR = ROOT / "0_orange_yaml/J_V2_series"
LEGACY_NAMES = [
    "J20_glance_anchor",
    "J21_glance_density",
    "J22_density_readout",
    "J23_gaze_training",
    "J24_antialias",
    "J25_glance_gaze",
    "J26_highres_960",
    "J27_highres_1280",
]
NAMES = list(LEGACY_NAMES) + list(STRUCTURE_NAMES)
_BASE = dict(density=False, detach=False, gaze_train=False, antialias=False, imgsz=640)
FACTORS = {
    "J20_glance_anchor": dict(_BASE),
    "J21_glance_density": dict(_BASE, density=True),
    "J22_density_readout": dict(_BASE, density=True, detach=True),
    "J23_gaze_training": dict(_BASE, density=True, gaze_train=True),
    "J24_antialias": dict(_BASE, antialias=True),
    "J25_glance_gaze": dict(_BASE, density=True, gaze_train=True, antialias=True),
    "J26_highres_960": dict(_BASE, imgsz=960),
    "J27_highres_1280": dict(_BASE, imgsz=1280),
}
DESCRIPTIONS = {
    "J20_glance_anchor": "YOLO11n scratch glance anchor (graph identical to J00); gaze evaluated self-guided",
    "J21_glance_density": "+ P3 small-fruit density head, shared gradient (multi-task)",
    "J22_density_readout": "+ density head on detached P3 (pure readout; isolates multi-task effect)",
    "J23_gaze_training": "density + fine views sampled 1/2 uniform, 1/2 by small-fruit count (train what is gazed)",
    "J24_antialias": "anchor + INTER_AREA resampling for prepared views and validation",
    "J25_glance_gaze": "full Glance-Gaze: density + gaze training + antialias",
    "J26_highres_960": "single-pass resolution control: anchor graph at imgsz 960",
    "J27_highres_1280": "single-pass resolution control: anchor graph at imgsz 1280",
}
FACTORS.update(STRUCTURE_FACTORS)
DESCRIPTIONS.update(STRUCTURE_DESCRIPTIONS)
SUITES = {
    "all": tuple(NAMES),
    "priority": ("J20_glance_anchor", "J29_i20_structure_anchor", "J34_shared_prediction",
                 "J36_structure_complete", "J38_compact_complete"),
    "mechanism": ("J20_glance_anchor", "J29_i20_structure_anchor", "J36_structure_complete",
                  "J41_subpixel_p3", "J42_subpixel_p2", "J43_compact_subpixel_p2"),
    "decoder_repairs": ("J41_subpixel_p3", "J42_subpixel_p2", "J43_compact_subpixel_p2", "J44_compact_subpixel_p3"),
    "cost_controls": ("J30_rep_pan", "J31_rep_direct", "J32_p2_discovery", "J33_structure_residual",
                      "J39_compact_decoder", "J40_shared_highres_proto"),
    "structure": tuple(STRUCTURE_NAMES),
    "gaze": ("J20_glance_anchor", "J21_glance_density", "J23_gaze_training", "J25_glance_gaze"),
    "ablation": ("J22_density_readout", "J24_antialias"),
    "resolution": ("J26_highres_960", "J27_highres_1280"),
    "smoke": ("J21_glance_density",),
}
# RAM: 8545 cached views x imgsz^2 x 3 bytes = 9.8/22/39 GiB at 640/960/1280. Cache is speed-only.
CACHE = {name: (True if FACTORS[name]["imgsz"] <= 640 else False) for name in NAMES}


def select_names(suite="priority", only=""):
    chosen = [s.strip() for s in only.split(",") if s.strip()] if only else list(SUITES[suite])
    if not chosen or len(set(chosen)) != len(chosen) or set(chosen) - set(NAMES):
        raise ValueError("Choose distinct model names from citrus_j2_suite.NAMES")
    return chosen
