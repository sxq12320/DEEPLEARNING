"""Read-only local annotation feasibility and uploaded V4R size-stratified recall."""
# ruff: noqa: E402 -- direct CLI script.

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from shapely.geometry import Polygon
from shapely.validation import explain_validity

from citrus_slicing import read_polygons, source_files
from ultralytics.data.utils import img2label_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    _, files = source_files(args.data)
    total, invalid, issues = 0, 0, []
    reasons = Counter()
    for path in img2label_paths([str(p) for p in files]):
        bad = []
        for index, (_, vertices) in enumerate(read_polygons(path)):
            total += 1
            poly = Polygon(vertices)
            if not poly.is_valid:
                invalid += 1
                reason = explain_validity(poly)
                reasons[reason.split("[")[0]] += 1
                bad.append(dict(instance_index=index, reason=reason))
        if bad:
            issues.append(dict(label=path, instances=bad))
    uploaded = ROOT.parents[1] / "results/E/E_V4R/CITRUS_EV4R_ALL_300EP"
    report = dict(
        data=str(args.data),
        local_train_images=len(files),
        local_instances=total,
        images_with_invalid_polygons=len(issues),
        invalid_instances=invalid,
        reasons=dict(reasons),
        issues=issues,
        original_data_modified=False,
        caveat="Local files only: equality to server label contents has not been established.",
        sizes={},
    )
    for path in sorted(uploaded.glob("*/paired_sliced_eval/paired_metrics.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        rows = []
        for mode in ("global", "trustedmask"):
            for lo, hi in ((0, 16), (16, 64), (64, 256), (256, 1024), (1024, 100000000)):
                records = [r for r in data["records"] if r["mode"] == mode and lo <= r["area640"] < hi]
                rows.append(
                    dict(mode=mode, area_range=[lo, hi], n=len(records), matched=sum(r["matched25"] for r in records))
                )
        report["sizes"][path.parents[1].name] = rows
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"Local: {len(files)} images/{total} instances; invalid in {len(issues)} images/{invalid} instances. "
        f"No source changes. Saved {args.output}"
    )


if __name__ == "__main__":
    main()
