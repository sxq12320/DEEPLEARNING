"""Exact CPU parity against the pre-cache COCO/P-R evaluation implementation."""

import copy
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from coco_utils import PreparedCocoEvaluation, encode_binary_mask, evaluate_predictions, prediction_from_mask


def legacy_evaluate(annotation_path, predictions, evaluate_bbox=True, pr_iou=0.5, pr_score=0.25):
    """Frozen previous evaluator: separate GT load/rasterization and P/R IoU computation."""
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    names = ("ap_50_95", "ap_50", "ap_75", "ap_small", "ap_medium", "ap_large", "ar_1", "ar_10", "ar_100",
             "ar_small", "ar_medium", "ar_large")
    metrics = {"prediction_count": len(predictions)}
    coco_gt = COCO(str(annotation_path))
    for iou_type, prefix in [("segm", "mask")] + ([("bbox", "box")] if evaluate_bbox else []):
        if predictions:
            inputs = [{k: v for k, v in p.items() if k != "bbox"} for p in predictions] if iou_type == "segm" else list(
                predictions)
            evaluator = COCOeval(coco_gt, coco_gt.loadRes(inputs), iou_type)
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
            metrics.update({f"{prefix}_{name}": float(value) for name, value in zip(names, evaluator.stats)})
        else:
            metrics.update({f"{prefix}_{name}": 0.0 for name in names})

    coco_gt = COCO(str(annotation_path))
    gt, dt = defaultdict(list), defaultdict(list)
    for annotation in coco_gt.dataset.get("annotations", []):
        gt[(int(annotation["image_id"]), int(annotation["category_id"]))].append(annotation)
    for prediction in predictions:
        if float(prediction["score"]) >= pr_score:
            dt[(int(prediction["image_id"]), int(prediction["category_id"]))].append(prediction)
    tp = fp = fn = 0
    for key in set(gt) | set(dt):
        ground_truth = gt.get(key, [])
        detected = sorted(dt.get(key, []), key=lambda item: float(item["score"]), reverse=True)
        if not ground_truth:
            fp += len(detected)
            continue
        if not detected:
            fn += len(ground_truth)
            continue
        gt_rles = [coco_gt.annToRLE(ann) for ann in ground_truth]
        dt_rles = [dict(ann["segmentation"]) for ann in detected]
        for rle in dt_rles:
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("ascii")
        ious = mask_utils.iou(dt_rles, gt_rles, [int(ann.get("iscrowd", 0)) for ann in ground_truth])
        matched = set()
        for row in ious:
            match = next((int(index) for index in np.argsort(row)[::-1]
                          if row[index] >= pr_iou and int(index) not in matched), None)
            if match is None:
                fp += 1
            else:
                matched.add(match)
                tp += 1
        fn += len(ground_truth) - len(matched)
    precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    metrics.update(mask_precision=precision, mask_recall=recall,
                   mask_f1=2 * precision * recall / max(precision + recall, 1e-12), pr_iou_threshold=pr_iou,
                   pr_score_threshold=pr_score, true_positive=tp, false_positive=fp, false_negative=fn)
    return metrics


def rectangle(x, y, width=10, height=10):
    result = np.zeros((64, 80), dtype=bool)
    result[y:y + height, x:x + width] = True
    return result


@pytest.fixture
def annotations(tmp_path):
    records = []
    for ann_id, image_id, category, x, y, width, height, crowd in [
            (21, 7, 1, 5, 5, 10, 10, 0), (22, 7, 1, 15, 5, 10, 10, 0),
            (23, 7, 3, 30, 30, 10, 10, 0), (24, 8, 1, 5, 5, 25, 25, 1)]:
        records.append(dict(id=ann_id, image_id=image_id, category_id=category, iscrowd=crowd, area=width * height,
                            bbox=[x, y, width, height], segmentation=[
                                [x, y, x + width, y, x + width, y + height, x, y + height]]))
    data = dict(info={}, images=[dict(id=i, file_name=f"{i}.png", height=64, width=80) for i in (7, 8, 9)],
                categories=[dict(id=1, name="fruit"), dict(id=3, name="other")], annotations=records)
    path = tmp_path / "instances.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def predictions_for(case):
    def pred(mask, score=0.9, image=7, category=1):
        return prediction_from_mask(image, category, score, mask)

    perfect = [pred(rectangle(5, 5)), pred(rectangle(15, 5), 0.8),
               pred(rectangle(30, 30), category=3), pred(rectangle(5, 5, 25, 25), image=8)]
    if case == "perfect":
        return perfect
    if case == "crowd":
        return perfect + [pred(rectangle(5, 5, 25, 25), 0.85, image=8), pred(rectangle(10, 10), image=8)]
    if case == "score_boundary":
        return [pred(rectangle(5, 5), 0.25), pred(rectangle(15, 5), np.nextafter(0.25, 0.0)),
                pred(rectangle(30, 30), np.nextafter(0.25, 1.0), category=3)]
    if case == "iou_boundary":
        return [pred(rectangle(5, 5, 5, 10)), pred(rectangle(15, 5, 4, 10), 0.8)]
    if case == "ties":
        # The first detection has equal IoU with both GTs; legacy argsort picks the later GT.
        return [pred(rectangle(5, 5, 20, 10), 0.9), pred(rectangle(15, 5), 0.9)]
    if case == "over_max_dets":
        return [pred(rectangle(50, 45), 0.9) for _ in range(105)] + [pred(rectangle(5, 5), 0.9)]
    if case == "empty_masks":
        return [pred(np.zeros((64, 80), dtype=bool)), *perfect]
    if case == "no_predictions":
        return []
    if case == "unmatched":
        return perfect + [pred(rectangle(50, 45), image=9), pred(rectangle(50, 45), image=8, category=3)]
    raise AssertionError(case)


@pytest.mark.parametrize("case", ["perfect", "crowd", "score_boundary", "iou_boundary", "ties", "over_max_dets",
                                 "empty_masks", "no_predictions", "unmatched"])
@pytest.mark.parametrize("evaluate_bbox", [False, True])
def test_prepared_metrics_exactly_match_legacy(annotations, case, evaluate_bbox):
    predictions = predictions_for(case)
    expected = legacy_evaluate(annotations, copy.deepcopy(predictions), evaluate_bbox)
    prepared = PreparedCocoEvaluation(annotations)
    actual = prepared.evaluate(copy.deepcopy(predictions), evaluate_bbox)
    assert actual == expected
    assert prepared.evaluate(copy.deepcopy(predictions), evaluate_bbox) == expected
    if case == "perfect":
        assert actual["mask_ap_50_95"] > 0.99 and actual["mask_precision"] == 1
    if case == "ties":
        assert actual["true_positive"] == 1 and actual["false_positive"] == 1
    if case == "over_max_dets":
        assert actual["true_positive"] == 1 and actual["false_positive"] == 105


@pytest.mark.parametrize("pr_iou,pr_score", [(0.0, 0.0), (0.4, 0.25), (0.5, 0.9), (1.0, 0.25), (0.5, 1.0)])
def test_custom_thresholds_reuse_only_matching_iou_rows(annotations, pr_iou, pr_score):
    predictions = predictions_for("score_boundary") + predictions_for("iou_boundary")
    expected = legacy_evaluate(annotations, copy.deepcopy(predictions), False, pr_iou, pr_score)
    assert PreparedCocoEvaluation(annotations).evaluate(predictions, False, pr_iou, pr_score) == expected


def test_empty_ground_truth_and_no_predictions_are_unchanged(annotations):
    data = json.loads(annotations.read_text())
    data["annotations"] = []
    annotations.write_text(json.dumps(data))
    prepared = PreparedCocoEvaluation(annotations)
    for predictions in ([], predictions_for("perfect")):
        assert prepared.evaluate(copy.deepcopy(predictions)) == legacy_evaluate(annotations, copy.deepcopy(predictions))


def test_gt_snapshot_and_source_are_not_mutated(annotations):
    source = annotations.read_bytes()
    prepared = PreparedCocoEvaluation(annotations)
    snapshot = copy.deepcopy(prepared._dataset)
    for case in ("perfect", "crowd", "no_predictions", "ties", "perfect"):
        prepared.evaluate(predictions_for(case))
        assert prepared._dataset == snapshot and annotations.read_bytes() == source
    assert [record["id"] for record in prepared.images] == [7, 8, 9]
    assert prepared.category_ids == frozenset({1, 3})


def test_cached_gt_rasterizes_once_and_pr_reuses_coco_iou(annotations, monkeypatch):
    from pycocotools import mask as mask_utils
    from pycocotools.coco import COCO

    loads, polygons, iou_calls = [], [], []
    original_init, original_rle, original_iou = COCO.__init__, COCO.annToRLE, mask_utils.iou

    def init(self, annotation_file=None):
        if annotation_file is not None:
            loads.append(annotation_file)
        original_init(self, annotation_file)

    def ann_to_rle(self, ann):
        if isinstance(ann["segmentation"], list):
            polygons.append(ann["id"])
        return original_rle(self, ann)

    def iou(*args):
        iou_calls.append(1)
        return original_iou(*args)

    monkeypatch.setattr(COCO, "__init__", init)
    monkeypatch.setattr(COCO, "annToRLE", ann_to_rle)
    monkeypatch.setattr(mask_utils, "iou", iou)
    prepared = PreparedCocoEvaluation(annotations)
    assert len(loads) == 1 and len(polygons) == 4
    prepared.evaluate(predictions_for("perfect"), evaluate_bbox=False)
    calls_per_coco_eval = len(iou_calls)
    assert calls_per_coco_eval > 0
    prepared.evaluate(predictions_for("perfect"), evaluate_bbox=False)
    assert len(iou_calls) == 2 * calls_per_coco_eval
    assert len(loads) == 1 and len(polygons) == 4


def test_evaluate_predictions_accepts_prepared_snapshot_and_checks_path(annotations, tmp_path):
    prepared = PreparedCocoEvaluation(annotations)
    predictions = predictions_for("perfect")
    output = tmp_path / "metrics.json"
    actual = evaluate_predictions(annotations, predictions, output, False, prepared_evaluator=prepared)
    assert actual == legacy_evaluate(annotations, copy.deepcopy(predictions), False)
    assert json.loads(output.read_text()) == actual
    with pytest.raises(ValueError, match="different annotation"):
        evaluate_predictions(tmp_path / "different.json", [], prepared_evaluator=prepared)


@pytest.mark.parametrize("dtype", [bool, np.uint8, np.float32])
@pytest.mark.parametrize("layout", ["c", "f", "strided", "reversed"])
def test_single_copy_fortran_encoding_matches_previous_conversion(dtype, layout):
    from pycocotools import mask as mask_utils

    array = rectangle(5, 5).astype(dtype)
    if layout == "f":
        array = np.asfortranarray(array)
    elif layout == "strided":
        array = array[::2, ::2]
    elif layout == "reversed":
        array = array[::-1, ::-1]
    expected = mask_utils.encode(np.asfortranarray(np.asarray(array, dtype=np.uint8)))
    expected["counts"] = expected["counts"].decode("ascii")
    assert encode_binary_mask(array) == expected


def test_prediction_mask_threshold_stays_strictly_above_half():
    from pycocotools import mask as mask_utils

    mask = np.array([[0.49, 0.5, np.nextafter(0.5, 1.0), 1.0]])
    prediction = prediction_from_mask(1, 1, 0.9, mask)
    assert np.array_equal(mask_utils.decode(prediction["segmentation"]), [[0, 0, 1, 1]])
