"""Four-arm ablation relative to original YOLO11n-seg; existing full labels, whole images.

    python RUN_CITRUS_MINIMAL_ABLATION.py
    python RUN_CITRUS_MINIMAL_ABLATION.py --check-only --device cpu
    python RUN_CITRUS_MINIMAL_ABLATION.py --smoke --device cpu --workers 0 --batch 2
    python RUN_CITRUS_MINIMAL_ABLATION.py --only R00_baseline R03_asym_relay

A: remove the last P4-to-P5 neck reconstruction (reuse SAGE40's known graph reduction).
B: one narrow, gated C2-to-P3 relay, retaining the native head and native loss.
No accuracy gain or publication-level novelty is established before real experiments.
Shared engine fixes pretrained initialization, 640 inputs, mask_ratio=2, all-label
checkpoint selection, seed 42, 300 epochs and no tiles/min8. --seeds 42,43,44 is
available for the final selected comparison. Test images are not evaluated.
"""

import os
from pathlib import Path

import RUN_WHOLE_PRETRAIN_COMPARE as engine

ROOT = Path(__file__).resolve().parent
MODELS = {
    "R00_baseline": "0_orange_yaml/R_MINIMAL_series/R00_baseline.yaml",
    "R01_asym": "0_orange_yaml/R_MINIMAL_series/R01_asym.yaml",
    "R02_relay": "0_orange_yaml/R_MINIMAL_series/R02_relay.yaml",
    "R03_asym_relay": "0_orange_yaml/R_MINIMAL_series/R03_asym_relay.yaml",
}
# Edit these server defaults if needed; CLI options override them.
SERVER_DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
SERVER_PROJECT = "/data/sxq/results/CITRUS_MINIMAL_ABLATION_20261111"
DEVICE = "1"


def main(argv=None):
    engine.DATA = SERVER_DATA
    engine.DEVICE = DEVICE
    project = ROOT / "1_results/CITRUS_MINIMAL_ABLATION_20261111" if os.name == "nt" else Path(SERVER_PROJECT)
    engine.main(argv, model_catalog=MODELS, default_models=list(MODELS), project_default=project, entrypoint=__file__)


if __name__ == "__main__":
    main()
