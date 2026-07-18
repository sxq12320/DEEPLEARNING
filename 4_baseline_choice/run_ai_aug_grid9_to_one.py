"""Double-click launcher: use multiple citrus images to generate 1 harder image.

Edit the variables below, then double-click this file or run it in PyCharm.
The script sends multiple original reference images directly to the image model.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path


# Folder containing the simple reference images.
INPUT_DIR = Path(r"E:\mastercode\data\orange_wuxi\img")

# Generated images, metadata, and logs go here.
OUTPUT_DIR = Path(r"E:\mastercode\4_baseline_choice\ai_aug_multi")

# How many reference images are used to generate one new image.
# wan2.7-image-pro supports up to 9 images in one request.
# qwen-image-2.0 / qwen-image-2.0-pro support up to 6 images in one request.
GROUP_SIZE = 9

# Generate only the first N groups for testing. Set to 0 to process all full groups.
LIMIT_GROUPS = 1

# Standard multi-image input is recommended with wan2.7-image-pro.
MODEL = "wan2.7-image-pro"
SIZE = "2048*2048"


def main() -> int:
    script = Path(__file__).resolve().parent / "scripts" / "generate_qwen_citrus_grid9_to_one.py"
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
        "--group-size",
        str(GROUP_SIZE),
        "--limit-groups",
        str(LIMIT_GROUPS),
    ]

    print("Citrus AI augmentation: multiple reference images -> 1 harder image")
    print(f"Input folder : {INPUT_DIR}")
    print(f"Output folder: {OUTPUT_DIR}")
    print(f"Model        : {MODEL}")
    print(f"Size         : {SIZE}")
    print(f"Group size   : {GROUP_SIZE}")
    print(f"Limit groups : {LIMIT_GROUPS if LIMIT_GROUPS > 0 else 'all'}")
    print()

    if not INPUT_DIR.exists():
        print(f"Input folder does not exist: {INPUT_DIR}")
        return 1

    return subprocess.run(command).returncode


if __name__ == "__main__":
    code = main()
    print()
    input("Finished. Press Enter to close...")
    raise SystemExit(code)
