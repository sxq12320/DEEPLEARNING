"""One-click launcher for leakage-controlled citrus dataset generation."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from build_grouped_citrus_cv import build_datasets  # noqa: E402


def main() -> int:
    """Build the group-aware 7:2:1 and four-fold datasets."""
    print("Citrus grouped dataset builder")
    print("Source      : E:\\mastercode\\data\\orange_standardized")
    print("7:2:1 output: E:\\mastercode\\data\\orange_yolo (overwrite)")
    print("4-fold output: E:\\mastercode\\data\\orange_yolo_4fold (overwrite)")
    print()
    report = build_datasets()
    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"\nDataset build failed: {exc}")
        exit_code = 1
    print()
    input("Finished. Press Enter to close...")
    raise SystemExit(exit_code)
