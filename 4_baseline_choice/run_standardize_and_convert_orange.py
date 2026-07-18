"""Double-click launcher for standardizing orange_wuxi and rebuilding orange_yolo."""

from __future__ import annotations

import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from standardize_orange_wuxi import rebuild_all  # noqa: E402


def main() -> int:
    """Run the complete dataset rebuild and keep the console open."""
    print("Citrus dataset rebuild")
    print("Source       : E:\\mastercode\\data\\orange_wuxi")
    print("Standardized : E:\\mastercode\\data\\orange_standardized")
    print("YOLO output  : E:\\mastercode\\data\\orange_yolo (overwrite)")
    print()
    report = rebuild_all()
    print()
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except Exception as exc:
        print(f"\nDataset rebuild failed: {exc}")
        exit_code = 1
    print()
    input("Finished. Press Enter to close...")
    raise SystemExit(exit_code)
