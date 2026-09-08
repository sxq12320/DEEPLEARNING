"""Compare current split membership with the existing stricter capture groups."""

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


WORKSPACE = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DATA = WORKSPACE / "data"
RESULTS = WORKSPACE / "1_SEVER/results/SAGE/CITRUS_SAGE_V8_ALL_300EP"


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def main():
    manifest = read_csv(DATA / "orange_yolo_grouped_dedup_20260820/audit/split_manifest.csv")
    edges = read_csv(DATA / "orange_yolo_grouped_dedup_20260820/audit/group_edges.csv")
    source = {r["stem"]: r for r in manifest}
    payload = {}
    memberships = {}
    for name in ("orange_yolo", "orange_yolo_grouped_dedup_20260820"):
        membership, counts = {}, {}
        for split in ("train", "val", "test"):
            images = [p for p in (DATA / name / split / "images").iterdir()
                      if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
            labels = list((DATA / name / split / "labels").glob("*.txt"))
            counts[split] = dict(images=len(images), instances=sum(
                sum(bool(line.strip()) for line in p.read_text(encoding="utf-8").splitlines()) for p in labels
            ))
            membership.update({p.stem: split for p in images})
        memberships[name] = membership
        grouped = defaultdict(list)
        for stem, split in membership.items():
            grouped[source[stem]["group_id"]].append((stem, split))
        crosses = {g: members for g, members in grouped.items() if len({s for _, s in members}) > 1}
        tv = {g: members for g, members in crosses.items() if {"train", "val"} <= {s for _, s in members}}
        tt = {g: members for g, members in crosses.items() if {"train", "test"} <= {s for _, s in members}}
        val_with_train = [stem for members in tv.values() for stem, split in members if split == "val"]
        test_with_train = [stem for members in tt.values() for stem, split in members if split == "test"]
        cross_edges = [{**e, "left_split": membership[e["left"]], "right_split": membership[e["right"]],
                        "left_source": source[e["left"]]["source_name"],
                        "right_source": source[e["right"]]["source_name"]}
                       for e in edges if membership[e["left"]] != membership[e["right"]]]
        payload[name] = dict(
            counts=counts, groups=len(grouped), cross_split_groups=len(crosses),
            train_val_groups=len(tv), val_images_in_train_related_groups=len(val_with_train),
            train_test_groups=len(tt), test_images_in_train_related_groups=len(test_with_train),
            crossing_edge_reason_counts=dict(Counter(e["reasons"] for e in cross_edges)),
            cross_edges=cross_edges, cross_groups=crosses,
        )
    runs = {}
    for run in sorted(RESULTS.glob("SAGE8*")):
        rows = {}
        for split in ("train", "val"):
            listed = {Path(line.strip()).stem for line in
                      (run / f"{split}_loaded_files.txt").read_text().splitlines() if line.strip()}
            matches = {name: listed == {stem for stem, part in member.items() if part == split}
                       for name, member in memberships.items()}
            rows[split] = dict(count=len(listed), matches_local_membership=matches)
        runs[run.name] = rows
    payload["v8_saved_membership_comparison"] = runs
    payload["limitation"] = (
        "Filename membership checked against actual local image directories; correlated-group definitions are "
        "the saved 2026-08-20 capture-group manifest. No assumption of byte-identical server images, no rerun "
        "of perceptual similarity, and no performance evaluation on test."
    )
    (OUT / "data_membership_audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, row in payload.items():
        if isinstance(row, dict) and "counts" in row:
            print(name, json.dumps({k: v for k, v in row.items() if k not in {"cross_edges", "cross_groups"}},
                                  ensure_ascii=False))
            print("EXAMPLES", json.dumps(row["cross_edges"][:8], ensure_ascii=False))
    print("V8", json.dumps(runs, ensure_ascii=False))


if __name__ == "__main__":
    main()
