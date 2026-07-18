"""Double-click launcher for extracting visually distinct video frames."""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog


# True checks every original frame. False samples according to INTERVAL_SECONDS.
SCAN_EVERY_FRAME = True

# Used only when SCAN_EVERY_FRAME is False.
INTERVAL_SECONDS = 1.0

# Frames below this pHash distance are treated as near-duplicates.
# Larger values remove more similar frames. Recommended range: 10-16.
MIN_PHASH_DISTANCE = 12

# A frame is rejected when its SSIM to a retained frame reaches this value.
# Lower values remove more similar frames. Recommended starting range: 0.90-0.95.
MAX_SSIM = 0.92

# Compare each frame with this many pHash-nearest retained frames.
SSIM_CANDIDATES = 3

# Smaller analysis images make comparison faster without changing saved resolution.
COMPARISON_SIZE = 320

# JPEG output quality.
JPEG_QUALITY = 95


def select_videos() -> list[Path]:
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    selected = filedialog.askopenfilenames(
        title="Select one or more videos",
        initialdir=str(Path.home() / "Desktop"),
        filetypes=[
            ("Video files", "*.mp4 *.mov *.avi *.mkv *.m4v"),
            ("All files", "*.*"),
        ],
    )
    root.destroy()
    return [Path(path) for path in selected]


def main() -> int:
    videos = select_videos()
    if not videos:
        print("No videos selected.")
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path.home() / "Desktop" / f"video_unique_frames_{timestamp}"
    script = Path(__file__).resolve().parent / "scripts" / "extract_unique_video_frames.py"

    command = [
        sys.executable,
        str(script),
        *[str(video) for video in videos],
        "--output-dir",
        str(output_dir),
        "--interval",
        str(INTERVAL_SECONDS),
        "--min-phash-distance",
        str(MIN_PHASH_DISTANCE),
        "--max-ssim",
        str(MAX_SSIM),
        "--ssim-candidates",
        str(SSIM_CANDIDATES),
        "--comparison-size",
        str(COMPARISON_SIZE),
        "--jpeg-quality",
        str(JPEG_QUALITY),
    ]
    if SCAN_EVERY_FRAME:
        command.append("--every-frame")

    print("Unique video frame extraction")
    print(f"Selected videos : {len(videos)}")
    print(f"Scan mode       : {'every original frame' if SCAN_EVERY_FRAME else 'fixed interval'}")
    if not SCAN_EVERY_FRAME:
        print(f"Interval        : {INTERVAL_SECONDS} seconds")
    print(f"pHash ranking   : {MIN_PHASH_DISTANCE}")
    print(f"SSIM threshold  : {MAX_SSIM}")
    print(f"Output directory: {output_dir}")
    print()

    result = subprocess.run(command)
    if result.returncode == 0:
        print()
        print(f"Finished. Frames are in: {output_dir}")
    return result.returncode


if __name__ == "__main__":
    try:
        code = main()
    except Exception as exc:
        print(f"\nExtraction failed: {exc}")
        code = 1
    print()
    input("Press Enter to close...")
    raise SystemExit(code)
