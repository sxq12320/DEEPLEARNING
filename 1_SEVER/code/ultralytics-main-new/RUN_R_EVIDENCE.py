"""Whole-image three-factor study, foreground sequential training from VS Code.

No new operator is claimed. A=asymmetric neck; B=local contrast stem;
C=semantic correction before native prediction towers. All eight combinations
are declared before training. V12_06 and SAGE42 are historical reference arms
retrained on the SAME full-label dataset and recipe, not imported scores.

python RUN_R_EVIDENCE.py --check-only --device cpu
python RUN_R_EVIDENCE.py --smoke --device cpu --workers 0 --batch 2
python RUN_R_EVIDENCE.py

Default RUN_MODE="reference" compares three existing models first. Set it to
"factorial" for the eight new arms, or "all" for all ten, in a new output folder.

Only the eight RE models have identical native losses. Reference models retain
their intrinsic losses; they are complete-method comparisons, not factorial arms.
"""
import os
from pathlib import Path

import RUN_WHOLE_PRETRAIN_COMPARE as engine

ROOT = Path(__file__).resolve().parent
DATA = "/data/sxq/datasets/orange_yolo_grouped_dedup_20260820/data.yaml"
PROJECT = "/data/sxq/results/R_EVIDENCE_20261112"
DEVICE = "1"
EPOCHS = 300
BATCH = 16
WORKERS = 4
CACHE = True
AMP = False
SEEDS = [42]
RUN_MODE = "reference"  # reference (3), factorial (8), or all (10)
MODELS = {
    name: f"0_orange_yaml/R_EVIDENCE_series/{name}.yaml"
    for name in (
        "RE00_baseline", "RE01_asym", "RE02_contrast", "RE03_semantic",
        "RE04_asym_contrast", "RE05_asym_semantic", "RE06_contrast_semantic", "RE07_full",
    )
}
MODELS.update({
    "REF_V12_06": "0_orange_yaml/E_V12_series/V12_06_recognition_contrast.yaml",
    "REF_SAGE42": "0_orange_yaml/SAGE_series/SAGE42_asym_semantic_detail.yaml",
})


def main(argv=None):
    queues = {"reference": ["RE00_baseline", "REF_V12_06", "REF_SAGE42"],
              "factorial": list(MODELS)[:8], "all": list(MODELS)}
    if RUN_MODE not in queues:
        raise ValueError("RUN_MODE must be reference, factorial or all")
    engine.DATA, engine.DEVICE = DATA, DEVICE
    engine.EPOCHS, engine.BATCH, engine.WORKERS = EPOCHS, BATCH, WORKERS
    engine.AMP, engine.CACHE, engine.SEEDS = AMP, CACHE, SEEDS
    project = ROOT / "1_results/R_EVIDENCE_20261112" if os.name == "nt" else Path(PROJECT)
    project = project.with_name(project.name + "_" + RUN_MODE)
    engine.main(argv, model_catalog=MODELS, default_models=queues[RUN_MODE],
                project_default=project, entrypoint=__file__)


if __name__ == "__main__":
    main()
