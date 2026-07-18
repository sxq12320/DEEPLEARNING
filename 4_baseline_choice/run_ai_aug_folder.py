"""Double-click launcher for Qwen citrus hard-sample batch generation.

Edit INPUT_DIR before running. Double-click this file or run it in PyCharm.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Change this to the folder you want to process.
INPUT_DIR = Path(r"E:\mastercode\data\orange_wuxi\img")

# Generated images and the CSV log will be saved here.
OUTPUT_DIR = Path(r"E:\mastercode\4_baseline_choice\ai_aug_batch")

# Generate only the first N images for a quick test. Set to 0 to process all images.
LIMIT = 1

# Keep the first successful version settings.
SIZE = "1152*2048"
MODEL = "qwen-image-2.0"


def main() -> int:
    script = Path(__file__).resolve().parent / "scripts" / "generate_qwen_citrus_hard_batch.py"
    command = [
        sys.executable,
        str(script),
        "--input-dir",
        str(INPUT_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
        "--model",
        MODEL,
        "--size",
        SIZE,
    ]
    if LIMIT > 0:
        command.extend(["--limit", str(LIMIT)])

    print("Qwen citrus hard-sample batch generation")
    print(f"Input folder : {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Limit        : {LIMIT if LIMIT > 0 else 'all'}")
    print()

    if not INPUT_DIR.exists():
        print(f"Input folder does not exist: {INPUT_DIR}")
        return 1

    result = subprocess.run(command)
    return result.returncode


if __name__ == "__main__":
    code = main()
    print()
    input("Finished. Press Enter to close...")
    raise SystemExit(code)
