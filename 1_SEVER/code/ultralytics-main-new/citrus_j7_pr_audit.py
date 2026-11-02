"""Final-only PR audit from existing EV6 mask-match flags; evaluation is unchanged.

This records descriptive validation operating points, not new AP, a new matcher,
or a test-set threshold-selection procedure. Equal confidence values are one
threshold group, so a prefix cannot choose only the favourable side of a tie.
"""

import json
from pathlib import Path

import numpy as np

from citrus_e_v6_training import EV6Validator


def _class_names(names):
    if isinstance(names, (tuple, list)):
        names = dict(enumerate(names))
    if not isinstance(names, dict):
        raise ValueError("Class names must be an integer-keyed mapping or ordered sequence")
    result = {}
    for key, value in names.items():
        if isinstance(key, bool) or not isinstance(key, (int, np.integer)) or key < 0:
            raise ValueError("Class IDs must be nonnegative integers")
        result[int(key)] = str(value)
    return result


def _class_vector(value, known, name):
    value = np.asarray(value)
    if value.ndim != 1 or not np.issubdtype(value.dtype, np.number):
        raise ValueError("{} must be a one-dimensional numeric array".format(name))
    if not np.isfinite(value).all() or (value < 0).any() or (value != np.floor(value)).any():
        raise ValueError("{} contains invalid class IDs".format(name))
    result = value.astype(np.int64, copy=True)
    if not set(result.tolist()).issubset(known):
        raise ValueError("{} contains classes absent from names".format(name))
    return result


def audit_mask_matches(tp_m, conf, pred_cls, target_cls, names, confidence_lower_bound=.001,
                       nms_iou=.7, max_det=300):
    """Summarize postprocess-retained matches at mask IoU=.5 without interpolation.

    tp_m is the existing NxK boolean/0-1 match matrix; column zero is IoU=.5.
    The caller verifies the validator's IoU grid. No masks are decoded/rematched.
    Recall is None without GT, and zero (not undefined) with GT but no TP.
    Precision is None when no predictions are retained at a threshold.
    """
    names = _class_names(names)
    tp_m = np.asarray(tp_m)
    conf = np.asarray(conf)
    if tp_m.ndim != 2 or tp_m.shape[1] < 1:
        raise ValueError("tp_m must be an NxK matrix with IoU=.5 in column zero")
    if not (np.issubdtype(tp_m.dtype, np.number) or tp_m.dtype == np.bool_):
        raise ValueError("tp_m must contain boolean/0-1 match flags")
    if not np.isfinite(tp_m).all() or not np.isin(tp_m, [0, 1]).all():
        raise ValueError("tp_m contains invalid match flags")
    if conf.ndim != 1 or not np.issubdtype(conf.dtype, np.number):
        raise ValueError("conf must be a one-dimensional numeric array")
    if not np.isfinite(conf).all() or ((conf < 0) | (conf > 1)).any():
        raise ValueError("conf must contain finite scores in [0,1]")
    pred_cls = _class_vector(pred_cls, set(names), "pred_cls")
    target_cls = _class_vector(target_cls, set(names), "target_cls")
    if len(tp_m) != len(conf) or len(conf) != len(pred_cls):
        raise ValueError("Prediction arrays have different lengths")
    if (not np.isfinite(confidence_lower_bound) or not 0 <= confidence_lower_bound <= 1 or
            not np.isfinite(nms_iou) or not 0 <= nms_iou <= 1 or
            isinstance(max_det, bool) or int(max_det) != max_det or max_det < 1):
        raise ValueError("Invalid confidence/NMS/max_det protocol")
    classes = []
    for class_id, name in sorted(names.items()):
        selected = pred_cls == class_id
        scores = conf[selected]
        flags = tp_m[selected, 0].astype(np.int64)
        gt_count = int((target_cls == class_id).sum())
        if flags.sum() > gt_count:
            raise ValueError("Mask TP count exceeds GT count for class {}".format(class_id))
        order = np.argsort(-scores, kind="mergesort")
        scores, flags = scores[order], flags[order]
        ends = np.flatnonzero(np.r_[scores[:-1] != scores[1:], True]) if len(scores) else np.zeros(0, int)
        cumulative_tp = np.cumsum(flags)[ends]
        cumulative_fp = ends + 1 - cumulative_tp
        precision = cumulative_tp / (ends + 1) if len(ends) else np.zeros(0)
        recall = cumulative_tp / gt_count if gt_count else None
        groups = []
        previous_end = -1
        previous_tp = 0
        for position, end in enumerate(ends):
            total_tp = int(cumulative_tp[position])
            count = int(end - previous_end)
            added_tp = total_tp - previous_tp
            groups.append(dict(threshold=float(scores[end]), count=count, tp=added_tp, fp=count - added_tp,
                               cumulative_tp=total_tp, cumulative_fp=int(cumulative_fp[position]),
                               precision=float(precision[position]),
                               recall=float(recall[position]) if gt_count else None))
            previous_end, previous_tp = end, total_tp

        def at_precision(target):
            eligible = np.flatnonzero(precision >= target)
            if not gt_count or not len(eligible):
                return dict(recall=None if not gt_count else 0.0, threshold=None, precision=None,
                            cumulative_tp=0, cumulative_fp=0, eligible_threshold_found=False)
            point = groups[int(eligible[-1])]
            return dict(recall=point["recall"], threshold=point["threshold"], precision=point["precision"],
                        cumulative_tp=point["cumulative_tp"], cumulative_fp=point["cumulative_fp"],
                        eligible_threshold_found=True)

        total_tp = int(flags.sum())
        maximum_recall = total_tp / gt_count if gt_count else None
        tail = groups[-20:]
        tail_tp = sum(group["tp"] for group in tail)
        tail_fp = sum(group["fp"] for group in tail)
        classes.append(dict(class_id=class_id, name=name, gt_count=gt_count, prediction_count=len(scores),
                            tp_at_mask_iou50=total_tp, fp_at_mask_iou50=int(len(scores) - total_tp),
                            max_reachable_recall=maximum_recall,
                            unreachable_recall_fraction=1.0 - maximum_recall if gt_count else None,
                            precision_at_all_retained_predictions=total_tp / len(scores) if len(scores) else None,
                            recall_at_precision90=at_precision(.90), recall_at_precision95=at_precision(.95),
                            threshold_group_count=len(groups), score_tail_last20_groups=tail,
                            score_tail_tp=tail_tp, score_tail_fp=tail_fp,
                            score_tail_precision=tail_tp / (tail_tp + tail_fp) if tail_tp + tail_fp else None))
    return dict(
        schema_version=1, metric="mask", match_iou=.5,
        source="Existing EV6 metrics.stats tp_m[:,0]; no rematching or AP modification",
        threshold_rule="confidence >= threshold, equal-confidence predictions grouped before prefix metrics",
        interpolated_for_operating_points=False, compute_ap_sentinel_included=False,
        protocol=dict(confidence_lower_bound=float(confidence_lower_bound), nms_iou=float(nms_iou),
                      max_det=int(max_det), mask_raster_ratio=2),
        limitations=[
            "PR horizontal axis is recall, not confidence. These are mask matches, not the box confusion matrix.",
            "Maximum reachable recall is conditional on the retained scores, postprocess/NMS and max_det budget.",
            "R@P90/95 are descriptive validation-set attainable points, not confidence intervals "
            "or certified test thresholds.",
            "Prefixes reuse fixed EV6 matches; they do not rerun NMS/rematching at each confidence threshold.",
            "Select a threshold on validation, freeze it, then report its achieved test precision "
            "and recall separately.",
            "No image-level matching identities are stored; this audit cannot localize missing annotations or objects."
        ], classes=classes)


def _snapshot_stats(stats, niou):
    """Copy concatenated statistics, never mutate the parent's lists or arrays."""
    result = {}
    for name in ("tp_m", "conf", "pred_cls", "target_cls"):
        values = stats.get(name)
        if not isinstance(values, (tuple, list)):
            raise ValueError("Expected list-of-arrays metric statistic: {}".format(name))
        if values:
            result[name] = np.concatenate(values, axis=0).copy()
        else:
            result[name] = np.zeros((0, niou), bool) if name == "tp_m" else np.zeros(0, float)
    return result


class J7FinalAuditValidator(EV6Validator):
    """Use only for final selected-checkpoint validation, never the epoch validator."""

    def get_stats(self):
        iouv = self.iouv.detach().cpu().numpy()
        if not len(iouv) or not np.isclose(iouv[0], .5):
            raise ValueError("J7 PR audit requires mask IoU=.5 in the first match column")
        snapshot = _snapshot_stats(self.metrics.stats, len(iouv))
        report = audit_mask_matches(**snapshot, names=self.names, confidence_lower_bound=self.args.conf,
                                    nms_iou=self.args.iou, max_det=self.args.max_det)
        destination = Path(self.save_dir)
        raw_path, report_path = destination / "raw_mask_matches.npz", destination / "mask_pr_audit.json"
        if raw_path.exists() or report_path.exists():
            raise FileExistsError("PR audit already exists; use a new final-validation output directory")
        results = super().get_stats()  # The unchanged EV6/AP computation clears the original stats.
        with raw_path.open("xb") as handle:
            np.savez_compressed(handle, **snapshot, mask_iou_thresholds=iouv,
                                class_ids=np.asarray(sorted(_class_names(self.names)), dtype=np.int64))
        with report_path.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2, allow_nan=False)
        return results
