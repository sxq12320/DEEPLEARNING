"""One-click integrity audit for the grouped citrus datasets."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from validate_grouped_citrus_cv import validate_datasets  # noqa: E402


def main() -> int:
    """Validate both the 7:2:1 and four-fold datasets."""
    print("Citrus dataset integrity audit")
    print("7:2:1 dataset: E:\\mastercode\\data\\orange_yolo")
    print("4-fold dataset: E:\\mastercode\\data\\orange_yolo_4fold")
    print()
    result = validate_datasets()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\nAudit passed.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"\nAudit failed: {exc}")
        exit_code = 1
    print()
    input("Finished. Press Enter to close...")
    raise SystemExit(exit_code)
