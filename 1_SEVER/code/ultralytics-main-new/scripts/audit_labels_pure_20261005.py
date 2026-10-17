"""Pure-label self-audit over all X-AnyLabeling JSONs (no model, read-only).

Checks every annotation polygon in the orange_wuxi source dataset for
structural label errors that do not need any model: duplicate labels on the
same fruit, degenerate polygons, out-of-bounds points, and implausible sizes.

Writes a CSV checklist; never modifies images or JSONs.
"""

import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np


def polygon_mask(points, shape, scale):
    """Rasterize one polygon onto the common long-side-640 canvas."""
    canvas = np.zeros(shape, np.uint8)
    pts = np.asarray(points, dtype=np.float64) * scale
    if len(pts) >= 3:
        cv2.fillPoly(canvas, [np.round(pts).astype(np.int32)], 1)
    return canvas.astype(bool)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wuxi-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    manifest = {r["stem"]: r for r in csv.DictReader(args.manifest.open(encoding="utf-8-sig"))}
    jsons = {p.stem: p for d in ("annotions_x", "annotion_x_2") for p in (args.wuxi_root / d).glob("*.json")}

    rows, stats = [], dict(images=0, shapes=0)
    for stem, r in sorted(manifest.items()):
        source_json = jsons.get(Path(r["source_name"]).stem)
        if source_json is None:
            rows.append(dict(image=r["source_name"], json="", gt_index="", flag="no_json", detail="manifest row has no source JSON"))
            continue
        data = json.loads(source_json.read_text(encoding="utf-8"))
        w, h = data["imageWidth"], data["imageHeight"]
        scale = 640 / max(h, w)
        eh, ew = (round(h * scale), round(w * scale)) if h >= w else (640, round(w * scale))
        masks = []
        stats["images"] += 1
        for i, s in enumerate(data["shapes"]):
            stats["shapes"] += 1
            pts = np.asarray(s.get("points", []), float)
            detail = []
            if s.get("shape_type") != "polygon":
                detail.append(f"shape_type={s.get('shape_type')}")
            if len(pts) < 3:
                detail.append(f"points={len(pts)}")
            mask = polygon_mask(pts, (eh, ew), scale)
            area = int(mask.sum())
            if area == 0:
                detail.append("zero_area_on_raster")
            elif area < 25:
                detail.append(f"tiny_area={area}px(raster)")
            elif area > eh * ew * 0.5:
                detail.append(f"huge_area={area}px(raster)")
            oob = int(((pts[:, 0] < 0) | (pts[:, 0] > w) | (pts[:, 1] < 0) | (pts[:, 1] > h)).sum()) if len(pts) else 0
            if oob:
                detail.append(f"out_of_bounds_points={oob}")
            if len(pts) >= 3:
                x0, y0 = pts.min(0)
                x1, y1 = pts.max(0)
                short = max(1e-6, min(x1 - x0, y1 - y0))
                if max(x1 - x0, y1 - y0) / short > 8:
                    detail.append(f"extreme_aspect={max(x1 - x0, y1 - y0) / short:.1f}")
            masks.append(mask)
            if detail:
                rows.append(dict(image=r["source_name"], json=str(source_json), gt_index=i,
                                 flag="geometry", detail=";".join(detail)))
        for i in range(len(masks)):
            for j in range(i + 1, len(masks)):
                inter = int((masks[i] & masks[j]).sum())
                if inter:
                    denom = min(int(masks[i].sum()), int(masks[j].sum()))
                    if inter / max(denom, 1) > 0.5:
                        rows.append(dict(image=r["source_name"], json=str(source_json), gt_index=f"{i}&{j}",
                                         flag="duplicate_pair", detail=f"overlap_of_smaller={inter / denom:.2f}"))

    with (args.output / "label_selfcheck.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["image", "json", "gt_index", "flag", "detail"])
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({**stats, "flags": len(rows),
                      "by_flag": dict(__import__("collections").Counter(x["flag"] for x in rows))},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
