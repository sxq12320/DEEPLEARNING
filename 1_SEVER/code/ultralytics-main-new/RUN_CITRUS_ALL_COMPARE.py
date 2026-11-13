"""Seven-model whole-image, full-label pretrained comparison with ONE shared baseline.

Default queue (300 epochs each, seed 42):
    R00_baseline          Original YOLO11n-seg, shared by both comparisons
    S42_sage42            Historical SAGE42 architecture
    V06_v12               Historical V12_06 architecture
    C01_sage42_contrast   SAGE42 + local-contrast stem (included by default)
    R01_asym              Original + asymmetric neck A
    R02_relay             Original + shallow relay B
    R03_asym_relay        Original + A + B

Main ablation: R00/R01/R02/R03. The SAGE/V12 arms are additional comparisons,
not extra components in R03. C01 is only a one-change ablation relative to S42,
NOT a one-change model relative to original YOLO11n-seg. All gains are unverified.

Examples:
    python RUN_CITRUS_ALL_COMPARE.py
    python RUN_CITRUS_ALL_COMPARE.py --check-only --device cpu
    python RUN_CITRUS_ALL_COMPARE.py --smoke --device cpu --workers 0 --batch 2
    python RUN_CITRUS_ALL_COMPARE.py --epochs 50 --project /data/sxq/results/CITRUS_ALL_SCREEN_01
    python RUN_CITRUS_ALL_COMPARE.py --only R00_baseline S42_sage42 V06_v12 C01_sage42_contrast

Use a new --project if changing the model queue, epoch budget, seed list or code.
All arms use existing full labels, whole 640 images, pretrained weights and
all-label validation checkpoint selection. No tiling or min8. This does not
certify annotation completeness. Test data are never used for model selection.
"""

import os
from pathlib import Path

import RUN_WHOLE_PRETRAIN_COMPARE as engine

ROOT = Path(__file__).resolve().parent

# Server defaults; command-line options override these settings.
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
PROJECT = "/data/sxq/results/CITRUS_ALL_COMPARE_20261111"
DEVICE = "1"
EPOCHS = 300
BATCH = 16
WORKERS = 4
SEEDS = [42]

MODELS = {
    "R00_baseline": "0_orange_yaml/R_MINIMAL_series/R00_baseline.yaml",
    "S42_sage42": "0_orange_yaml/SAGE_series/SAGE42_asym_semantic_detail.yaml",
    "V06_v12": "0_orange_yaml/E_V12_series/V12_06_recognition_contrast.yaml",
    "C01_sage42_contrast": "0_orange_yaml/SAGE_series/SAGE42_asym_semantic_detail.yaml",
    "R01_asym": "0_orange_yaml/R_MINIMAL_series/R01_asym.yaml",
    "R02_relay": "0_orange_yaml/R_MINIMAL_series/R02_relay.yaml",
    "R03_asym_relay": "0_orange_yaml/R_MINIMAL_series/R03_asym_relay.yaml",
}


def main(argv=None):
    """Reuse the validated training engine without duplicating the original baseline."""
    engine.__doc__ = __doc__  # Show this suite's queue and usage in the shared CLI help.
    engine.DATA, engine.DEVICE = DATA, DEVICE
    engine.EPOCHS, engine.BATCH, engine.WORKERS = EPOCHS, BATCH, WORKERS
    engine.SEEDS = list(SEEDS)
    project = ROOT / "1_results/CITRUS_ALL_COMPARE_20261111" if os.name == "nt" else Path(PROJECT)
    engine.main(argv, model_catalog=MODELS, default_models=list(MODELS), project_default=project, entrypoint=__file__)


if __name__ == "__main__":
    main()
