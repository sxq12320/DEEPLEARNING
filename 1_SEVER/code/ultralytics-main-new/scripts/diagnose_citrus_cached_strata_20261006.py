"""Export both size definitions from saved diagnostic records; no inference or training."""

import argparse
import csv
import hashlib
import json
import math
from collections import Counter
from pathlib import Path


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--d3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve prior artifacts; choose a new output directory")
    path = args.run / "records.csv"
    d3_path = args.d3 / "retained_gt.csv"
    records = read_csv(path)
    retained = read_csv(d3_path)
    bounds = (0, 8, 16, 32, 64, float("inf"))
    bins = ("<8", "8-16", "16-32", "32-64", ">=64")
    output = []
    for threshold in (0.001, 0.25):
        for mode in ("global", "fine", "upsample_linear", "upsample_area"):
            rows = [row for row in records if row["mode"] == mode and float(row["threshold"]) == threshold]
            for field, definition in (("short640", "box_short_side"), ("size640", "sqrt_visible_mask_area")):
                for index, label in enumerate(bins):
                    selected = [row for row in rows if bounds[index] <= float(row[field]) < bounds[index + 1]]
                    counts = Counter(row["stage"] for row in selected)
                    output.append(dict(mode=mode, threshold=threshold, definition=definition, size_bin=label,
                                       gt=len(selected), groups=len(set(row["group"] for row in selected)),
                                       hits=counts["S4"],
                                       recall=counts["S4"] / len(selected) if selected else None,
                                       **{stage: counts[stage] for stage in ("S0", "S1", "S2", "S3", "S4")}))
    supervised = []
    for definition in ("box_short_side", "sqrt_visible_mask_area_input2_times4"):
        for index, label in enumerate(bins):
            def size(row):
                return (float(row["short"]) if definition == "box_short_side"
                        else math.sqrt(float(row["visible_area_input"])))

            selected = [row for row in retained if bounds[index] <= size(row) < bounds[index + 1]]
            zero = sum(int(row["positive_anchors"]) == 0 for row in selected)
            supervised.append(dict(definition=definition, size_bin=label, occurrences=len(selected),
                                   zero_positive=zero, zero_fraction=zero / len(selected) if selected else None,
                                   mean_positive=sum(int(row["positive_anchors"]) for row in selected)
                                   / len(selected) if selected else None))
    # A definition changes bin membership but must not change the total GT/TP counts.
    for threshold in (0.001, 0.25):
        for mode in ("global", "fine", "upsample_linear", "upsample_area"):
            totals = []
            for definition in ("box_short_side", "sqrt_visible_mask_area"):
                rows = [row for row in output if row["threshold"] == threshold and row["mode"] == mode
                        and row["definition"] == definition]
                totals.append((sum(row["gt"] for row in rows), sum(row["hits"] for row in rows)))
            if totals[0] != totals[1] or totals[0][0] != 1049:
                raise ValueError("Unexpected GT identity/count or invalid size")
    if any(sum(row["occurrences"] for row in supervised if row["definition"] == definition) != len(retained)
           for definition in ("box_short_side", "sqrt_visible_mask_area_input2_times4")):
        raise ValueError("Invalid retained-instance size")
    args.output.mkdir(parents=True)
    write_csv(args.output / "d1_size_definitions.csv", output)
    write_csv(args.output / "d3_size_definitions.csv", supervised)
    payload = dict(evidence="旧划分探索", operation="cached records only", d1=output, d3=supervised,
                   source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (path, d3_path)},
                   limits=["D1 uses GTs on common640; D3 uses augmented surviving occurrences and input/2 masks.",
                           "Do not treat the two populations as paired augmentation outcomes.",
                           "Stage counts remain box-based explanatory proxies except actual S4 mask TP."])
    (args.output / "size_definitions.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False),
                                                       encoding="utf-8")
    print(json.dumps(dict(d1_rows=len(output), d3_rows=len(supervised), d3=supervised), ensure_ascii=False))


if __name__ == "__main__":
    main()
