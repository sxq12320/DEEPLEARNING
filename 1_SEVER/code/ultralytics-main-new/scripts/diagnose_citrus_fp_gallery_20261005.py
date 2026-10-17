"""Side-by-side FP gallery for human labelling (read-only): raw crop + box outline | existing overlay asset.

The overlay assets from ``diagnose_citrus_20261005`` hide texture under the red mask, so a labeller cannot tell an
unlabelled fruit from a curled leaf. This writes ``fp_<mode>_labeling.html`` plus raw crops; labels still go into the
existing ``fp_<mode>_human_labels.csv`` (column ``human_label``). No images, labels or model outputs are modified.
"""

import argparse
import csv
import html
import json
from pathlib import Path

import cv2
import numpy as np

LABELS = "leaf / branch / unlabeled_fruit / highlight / duplicate / partial_fruit / other"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True, help="Directory with val images, e.g. .../val/images")
    parser.add_argument("--labels-root", type=Path, help="YOLO-seg label dir; draws existing GT outlines in green")
    parser.add_argument("--wuxi-root", type=Path, help="Original X-AnyLabeling dataset (images + JSON); overrides "
                        "--data-root/--labels-root for drawing and records the source file per FP")
    parser.add_argument("--manifest", type=Path, help="split_manifest.csv mapping stem -> source_name")
    args = parser.parse_args()
    out_dir = args.run / "assets_raw"
    out_dir.mkdir(exist_ok=True)
    sources = {}
    if args.wuxi_root:
        manifest = {r["stem"]: r for r in csv.DictReader(args.manifest.open(encoding="utf-8-sig"))}
        images = {p.name: p for d in ("img", "img_2") for p in (args.wuxi_root / d).iterdir() if p.is_file()}
        jsons = {p.stem: p for d in ("annotions_x", "annotion_x_2") for p in (args.wuxi_root / d).glob("*.json")}
        for stem, r in manifest.items():
            if r["source_name"] in images:
                sources[stem + ".jpg"] = (r, images[r["source_name"]], jsons.get(Path(r["source_name"]).stem))
    for mode in ("global", "fine"):
        rows = list(csv.DictReader((args.run / f"fp_{mode}_human_labels.csv").open(encoding="utf-8-sig")))
        cells, mapping = [], []
        for n, r in enumerate(rows):
            source = sources.get(r["image"])
            image = cv2.imread(str(source[1] if source else args.data_root / r["image"]))
            if image is None:
                continue
            h, w = image.shape[:2]
            polygons = []
            if source and source[2] is not None:
                shapes = json.loads(source[2].read_text(encoding="utf-8"))["shapes"]
                polygons = [np.asarray(s["points"], float) for s in shapes if s.get("shape_type") == "polygon"]
            elif args.labels_root and (args.labels_root / (Path(r["image"]).stem + ".txt")).exists():
                for line in (args.labels_root / (Path(r["image"]).stem + ".txt")).read_text().splitlines():
                    values = line.split()[1:]
                    if len(values) >= 6:
                        polygons.append(np.asarray(values, float).reshape(-1, 2) * [w, h])
            # Existing annotation outlines (green) so duplicates / partial boxes are distinguishable.
            for poly in polygons:
                cv2.polylines(image, [poly.astype(np.int32)], True, (0, 255, 0), max(2, w // 900))
            if source:
                mapping.append(dict(row=n, image=r["image"], source_image=str(source[1]),
                                    source_json=str(source[2]) if source[2] else "", formal_split=source[0]["new_split"],
                                    box=r["box"], score=r["score"]))
            x0, y0, x1, y1 = json.loads(r["box"])
            margin = max(24, int(max(x1 - x0, y1 - y0) * 0.6))
            left, top = max(0, int(x0) - margin), max(0, int(y0) - margin)
            right, bottom = min(w, int(x1) + margin), min(h, int(y1) + margin)
            view = image[top:bottom, left:right].copy()
            cv2.rectangle(view, (int(x0) - left, int(y0) - top), (int(x1) - left, int(y1) - top), (0, 255, 255),
                          max(2, (right - left) // 150))
            raw = out_dir / f"raw_{mode}_{n:03d}.jpg"
            cv2.imwrite(str(raw), cv2.resize(view, (320, 320), interpolation=cv2.INTER_AREA))
            cells.append(
                f"<tr><td>{n}</td><td>{html.escape(r['image'])}<br>score {float(r['score']):.3f}<br>"
                f"maxIoU {float(r['max_mask_iou']):.2f}</td><td><img src='assets_raw/{raw.name}' width='320'></td>"
                f"<td><img src='{html.escape(r['asset'])}' width='256'></td><td></td></tr>")
        (args.run / f"fp_{mode}_labeling.html").write_text(
            "<meta charset='utf-8'><h1>FP labelling: {}</h1><p>Left: raw crop with yellow box (texture visible). "
            "Right: predicted mask overlay. Write one label per row into fp_{}_human_labels.csv column "
            "human_label: {}</p><table border='1'><tr><th>#</th><th>image</th><th>raw</th><th>overlay</th>"
            "<th>label</th></tr>{}</table>".format(mode, mode, LABELS, "".join(cells)), encoding="utf-8")
        if mapping:
            with (args.run / f"fp_{mode}_source_map.csv").open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
                writer.writeheader()
                writer.writerows(mapping)
        print(mode, len(cells), "mapped to source:", len(mapping))


if __name__ == "__main__":
    main()
