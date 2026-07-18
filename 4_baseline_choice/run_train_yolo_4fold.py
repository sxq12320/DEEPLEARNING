"""One-click sequential training for the four citrus cross-validation folds.

Edit only the configuration block below, then run this Python file directly.
By default it trains four folds and does not inspect the fixed test set.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


# ----------------------------- user configuration -----------------------------
MODEL = "yolo11n-seg.pt"
PRETRAINED = None
RUN_PREFIX = "CV4_yolo11n_seg"
EPOCHS = 300
BATCH = 4
IMGSZ = 640
DEVICE = "0"
FOLDS = (1, 2, 3, 4)
EVALUATE_FIXED_TEST = False
# -----------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ULTRALYTICS_ROOT = PROJECT_ROOT / "ultralytics-main-new"
DATASET_ROOT = PROJECT_ROOT / "data" / "orange_yolo_4fold"
OUTPUT_ROOT = ULTRALYTICS_ROOT / "1_results" / "ORANGE_WUXI_SEG_CV4"

if str(ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))

from train_citrus_seg import FIXED, SEED, build_model, set_seed  # noqa: E402
from ultralytics import YOLO  # noqa: E402


def scalar_results(metrics: object) -> dict[str, float]:
    """Convert Ultralytics result values into JSON-safe floats."""
    results = getattr(metrics, "results_dict", {})
    return {str(key): float(value) for key, value in results.items()}


def main() -> int:
    """Train each requested fold and optionally evaluate its fixed test split."""
    os.chdir(ULTRALYTICS_ROOT)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    summaries = []

    print("Citrus YOLO four-fold training")
    print(f"Model   : {MODEL}")
    print(f"Dataset : {DATASET_ROOT}")
    print(f"Output  : {OUTPUT_ROOT}")
    print(f"Folds   : {FOLDS}")
    print(f"Test    : {'enabled' if EVALUATE_FIXED_TEST else 'disabled'}")
    print()

    for fold in FOLDS:
        data_yaml = DATASET_ROOT / f"fold_{fold}" / "data.yaml"
        if not data_yaml.is_file():
            raise FileNotFoundError(f"Missing fold YAML: {data_yaml}")
        run_name = f"{RUN_PREFIX}_fold{fold}"
        run_dir = OUTPUT_ROOT / run_name
        if run_dir.exists():
            raise FileExistsError(
                f"Output already exists: {run_dir}. Change RUN_PREFIX or move the old run."
            )

        print(f"\n{'=' * 24} fold {fold}/4 {'=' * 24}")
        set_seed(SEED)
        model = build_model(MODEL, PRETRAINED)
        model.train(
            data=str(data_yaml),
            project=str(OUTPUT_ROOT),
            name=run_name,
            epochs=EPOCHS,
            batch=BATCH,
            imgsz=IMGSZ,
            device=DEVICE,
            **FIXED,
        )

        best_weights = Path(model.trainer.best).resolve()
        summary = {
            "fold": fold,
            "data": str(data_yaml),
            "run": str(run_dir),
            "best_weights": str(best_weights),
        }
        if EVALUATE_FIXED_TEST:
            test_model = YOLO(str(best_weights))
            metrics = test_model.val(
                data=str(data_yaml),
                split="test",
                imgsz=IMGSZ,
                batch=BATCH,
                device=DEVICE,
                project=str(OUTPUT_ROOT),
                name=f"{run_name}_fixed_test",
                plots=True,
            )
            summary["fixed_test_metrics"] = scalar_results(metrics)
        summaries.append(summary)
        (OUTPUT_ROOT / f"{RUN_PREFIX}_summary.json").write_text(
            json.dumps(summaries, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    print(f"\nAll requested folds finished. Summary: {OUTPUT_ROOT / f'{RUN_PREFIX}_summary.json'}")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"\nFour-fold training failed: {exc}")
        exit_code = 1
    print()
    input("Finished. Press Enter to close...")
    raise SystemExit(exit_code)
