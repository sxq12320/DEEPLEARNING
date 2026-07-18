"""One-click rebuild for the current citrus baseline datasets."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from labelme_to_yolo_dataset import (  # noqa: E402
    DEFAULT_CLASS_NAME,
    DEFAULT_OUTPUT,
    DEFAULT_RATIOS,
    DEFAULT_SEED,
    DEFAULT_SOURCE,
    convert_dataset,
)
from prepare_dataset import prepare_dataset  # noqa: E402


PREPARED_DATASET = ROOT / "datasets" / "citrus_prepared"


def clean_directory(path: Path) -> None:
    """Remove and recreate a generated dataset directory."""
    resolved = path.resolve()
    expected = PREPARED_DATASET.resolve()
    if resolved != expected:
        raise ValueError(f"Refusing to clean unexpected path: {resolved}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    """Rebuild orange_yolo and every derived baseline format."""
    yolo_report = convert_dataset(
        source_root=DEFAULT_SOURCE,
        output_root=DEFAULT_OUTPUT,
        class_name=DEFAULT_CLASS_NAME,
        seed=DEFAULT_SEED,
        final_ratios=DEFAULT_RATIOS,
        augment_splits=[],
        augmentations=[],
    )

    clean_directory(PREPARED_DATASET)
    prepared_report = prepare_dataset(
        source_root=DEFAULT_OUTPUT,
        output_root=PREPARED_DATASET,
        class_names=[DEFAULT_CLASS_NAME],
        mode="auto",
    )

    summary = {
        "orange_yolo": {
            "source_images": yolo_report["source_images"],
            "split_counts": yolo_report["final_split_counts"],
            "instances_by_split": yolo_report["instances_by_split"],
            "missing_json": len(yolo_report["missing_json"]),
            "ratio_vs_test": yolo_report["actual_final_ratio_vs_test"],
        },
        "prepared_dataset": prepared_report["totals"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
