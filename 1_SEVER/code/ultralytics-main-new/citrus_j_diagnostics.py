"""Single-class observed mask PR: no fictitious recall=1 or precision=0 padding."""

import csv
from pathlib import Path

import numpy as np


def observed_mask_pr(stats, output=None):
    """Summarize fixed validator matches at complete confidence-tie thresholds.

    This is an IoU=.50, native-mask-grid diagnostic, not a COCO AP replacement.
    Matches are the original validator's matches; threshold-specific rematching
    is not performed. Only the actual predicted confidence range is reported.
    """
    arrays = {key: np.concatenate(values, 0) for key, values in stats.items() if len(values)}
    targets = np.asarray(arrays.get("target_cls", []))
    if len(targets) and np.any(targets != 0):
        raise ValueError("J observed PR diagnostic requires single-class citrus")
    count = int(len(targets))
    conf = np.asarray(arrays.get("conf", []), dtype=float)
    matches = np.asarray(arrays.get("tp_m", np.zeros((0, 10))), dtype=bool)
    if len(conf) != len(matches) or not np.isfinite(conf).all():
        raise ValueError("Invalid confidence/mask-match arrays")
    points = []
    if len(conf):
        order = np.argsort(-conf, kind="stable")
        conf = conf[order]
        tp = matches[order, 0].astype(int).cumsum()
        ends = np.r_[np.flatnonzero(conf[:-1] != conf[1:]), len(conf) - 1]
        for index in ends:
            positive = int(tp[index])
            points.append(
                dict(
                    confidence=float(conf[index]),
                    tp=positive,
                    fp=int(index + 1 - positive),
                    fn=count - positive,
                    precision=positive / (index + 1),
                    recall=positive / count if count else 0.0,
                )
            )
    rmax = points[-1]["recall"] if points else 0.0
    reliable = [row["recall"] for row in points if row["precision"] >= 0.90]
    summary = dict(
        protocol="native_grid_mask_iou50_observed_fixed_matches",
        targets=count,
        predictions=len(conf),
        observed_thresholds=len(points),
        rmax=rmax,
        unmatched_fraction=1 - rmax if count else None,
        recall_at_precision_ge_090=max(reliable, default=0.0),
        endpoint=points[-1] if points else None,
        note="No extrapolated points. This diagnostic never changes checkpoint selection or COCO AP.",
    )
    if output is not None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("confidence", "tp", "fp", "fn", "precision", "recall"))
            writer.writeheader()
            writer.writerows(points)
    return summary
