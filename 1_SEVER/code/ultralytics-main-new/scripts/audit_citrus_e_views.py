"""Read-only train geometry audit; no source pixels, annotations or splits are changed."""

# ruff: noqa: E402
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PIL import Image

from citrus_slicing import clip_instances, read_polygons, source_files, view_windows
from ultralytics.data.utils import img2label_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    _, files = source_files(args.data)
    reasons, counts = Counter(), Counter()
    records = []
    for file in files:
        with Image.open(file) as image:
            w, h = image.size
        rows = read_polygons(img2label_paths([str(file)])[0])
        valid = 0
        for window in view_windows(h, w)[1:]:
            clipped, reason = clip_instances(rows, (h, w), window)
            if reason:
                reasons[reason] += 1
            else:
                valid += 1
                counts["tile_instances"] += len(clipped)
                counts["background_tiles"] += int(not clipped)
        counts["sources"] += 1
        counts["original_instances"] += len(rows)
        counts["accepted_tiles"] += valid
        counts["sources_without_tiles"] += int(valid == 0)
        records.append(dict(source=str(file), instances=len(rows), accepted_tiles=valid))
    payload = dict(summary=dict(counts), rejected_tile_reasons=dict(reasons), records=records,
                   note="All source/global views retained, no min-area filtering; tile rejection is not dataset cleaning.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(json.dumps(payload["rejected_tile_reasons"], indent=2))


if __name__ == "__main__":
    main()
