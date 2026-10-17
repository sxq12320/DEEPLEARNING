"""Verify cached diagnostic artifacts without inference, training, or source-data edits."""

# ruff: noqa: E402
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np

from scripts.diagnose_citrus_20261005 import MODES, metric_rows


def sha256(path):
    """Return a file checksum for artifact provenance."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def audit(run):
    """Recompute metrics and audit stage proxies and label provenance."""
    stats_path = run / "prediction_stats.jsonl"
    stats = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    source_records = read_csv(run / "records.csv")
    summary = json.loads((run / "summary.json").read_text(encoding="utf-8"))
    by_mode = {mode: [row for row in stats if row["mode"] == mode] for mode in MODES}
    names = [row["image"] for row in by_mode["global"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image statistics")
    reference = {row["image"]: (row["targets"], row["tiny_targets"], row["group"])
                 for row in by_mode["global"]}
    metrics, pr = {}, {}
    for mode, rows in by_mode.items():
        identity = {row["image"]: (row["targets"], row["tiny_targets"], row["group"]) for row in rows}
        if identity != reference or len(rows) != len(names):
            raise ValueError("Unpaired modes")
        metrics[mode] = metric_rows(rows)
        for key, value in metrics[mode].items():
            if not np.isclose(value, summary["metrics"][mode][key], atol=1e-12, rtol=0):
                raise ValueError(f"Stored metric mismatch {mode} {key}")
        confidence = np.concatenate([np.asarray(row["conf"], dtype=float) for row in rows])
        tp = np.concatenate([np.asarray(row["tp_m"], dtype=bool).reshape(-1, 10) for row in rows])[:, 0]
        order = np.argsort(-confidence)
        cumulative = np.cumsum(tp[order])
        precision = cumulative / np.arange(1, len(order) + 1)
        recall = cumulative / sum(row["targets"] for row in rows)
        high_precision = precision >= 0.9
        selected = [row for row in source_records if row["mode"] == mode and float(row["threshold"]) == 0.25]
        tp25 = sum(row["stage"] == "S4" for row in selected)
        predictions25 = int((confidence >= 0.25).sum())
        pr[mode] = dict(
            raster="additional_common640",
            iou=0.5,
            predictions001=len(confidence),
            hits001=int(tp.sum()),
            maximum_observed_recall=float(recall[-1]) if len(recall) else 0.0,
            recall_at_raw_prefix_precision90=float(recall[high_precision].max()) if high_precision.any() else 0.0,
            predictions25=predictions25,
            hits25=tp25,
            fp25=predictions25 - tp25,
            precision25=tp25 / predictions25 if predictions25 else None,
        )
    corrected, changes, counts = [], [], defaultdict(Counter)
    for row in source_records:
        new = dict(row)
        # A mask-eligible unmatched GT must be separated from the box-only funnel.
        # This correction changes explanatory labels, never the TP/AP statistics.
        if row["stage"] != "S4" and float(row["best_post_mask_iou"]) >= 0.5:
            new["stage"] = "matching_competition"
            new["subtype"] = "mask_eligible_unmatched"
            changes.append(dict(image=row["image"], gt_index=row["gt_index"], mode=row["mode"],
                                threshold=row["threshold"], before=row["stage"], after=new["stage"]))
        corrected.append(new)
        counts[f"{row['mode']}@{float(row['threshold']):g}"][new["stage"]] += 1
    label_path = run / "fp_fine_human_labels_filled.csv"
    labels = read_csv(label_path) if label_path.exists() else []
    alleged = [row for row in labels if row.get("human_label") == "unlabeled_fruit"]
    splits = sorted(set(row.get("formal_split", "unknown") for row in alleged))
    provenance = dict(
        records=len(labels),
        unique_images=len(set(row["image"] for row in labels)),
        categories=dict(Counter(row.get("human_label", "") for row in labels)),
        status_claims=dict(Counter(row.get("semantic_status", "") for row in labels)),
        reviewer_or_timestamp_columns=[key for key in (labels[0] if labels else {})
                                       if any(word in key.lower() for word in ("reviewer", "annotator", "timestamp"))],
        independently_confirmed_human_review=False,
        alleged_unlabeled_records=len(alleged),
        alleged_unlabeled_unique_images=len(set(row["image"] for row in alleged)),
        alleged_by_formal_split={split: dict(
            records=sum(row.get("formal_split", "unknown") == split for row in alleged),
            unique_images=len(set(row["image"] for row in alleged if row.get("formal_split", "unknown") == split)),
        ) for split in splits},
        limitations="A human_verified field is a provenance claim, not independently verified human annotation. "
                    "The FP sample was capped per image and stratified by score. No prevalence extrapolation.",
    )
    payload = dict(
        evidence="旧划分探索",
        operation="cached artifact audit, no inference or parameter updates",
        source_checksums={p.name: sha256(p) for p in (stats_path, run / "records.csv", run / "summary.json")},
        images=len(names), groups=len(set(row["group"] for row in by_mode["global"])),
        targets=sum(row["targets"] for row in by_mode["global"]),
        tiny_targets=sum(row["tiny_targets"] for row in by_mode["global"]),
        metric_reproduction=metrics, pr_raw_prefix=pr,
        corrected_stage_counts={key: dict(value) for key, value in counts.items()},
        stage_corrections=changes, semantic_label_provenance=provenance,
        limitations=["Box S0/S1/S2/S3 remain explanatory proxies, not a complete raw-mask counterfactual funnel.",
                     "PR prefix uses stored ten-IoU TP flags at conf 0.001. Work-threshold recall is re-matched.",
                     "Not native EV6 training CSV AP or COCO AP. Old split does not establish formal performance."])
    return payload, corrected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Use a new output directory; completed artifacts are preserved")
    payload, corrected = audit(args.run)
    args.output.mkdir(parents=True)
    (args.output / "audit.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with (args.output / "records_stage_checked.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(corrected[0]))
        writer.writeheader()
        writer.writerows(corrected)
    print(json.dumps({key: payload[key] for key in ("images", "groups", "targets", "tiny_targets",
                     "stage_corrections", "pr_raw_prefix", "semantic_label_provenance")},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
