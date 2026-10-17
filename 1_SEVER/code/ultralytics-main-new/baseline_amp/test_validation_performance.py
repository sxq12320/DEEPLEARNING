"""Same-protocol equivalence and observability of the optimized validation loop."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import validation  # noqa: E402
from coco_utils import evaluate_predictions, prediction_from_mask  # noqa: E402
from registry import COMMON_VALIDATION  # noqa: E402
from validation import cap_predictions, collect_metrics, prepared_evaluator  # noqa: E402


@pytest.fixture
def annotation(tmp_path):
    path = tmp_path / "instances_val.json"
    records = dict(info={}, categories=[dict(id=1, name="fruit"), dict(id=2, name="other")],
                   images=[dict(id=3, file_name="a.png", width=32, height=24),
                           dict(id=4, file_name="b.png", width=64, height=48)],
                   annotations=[dict(id=1, image_id=3, category_id=1, iscrowd=0, area=64,
                                     bbox=[4, 4, 8, 8], segmentation=[[4, 4, 12, 4, 12, 12, 4, 12]])])
    path.write_text(json.dumps(records), encoding="utf-8")
    return path


def candidates(shape, amount=130):
    mask = np.zeros(shape, dtype=bool)
    mask[4:12, 4:12] = True
    masks = [mask] * amount
    scores = [0.9 if index % 3 else 0.5 for index in range(amount)]
    labels = [1 if index % 4 else 2 for index in range(amount)]
    return masks, scores, labels


def test_encoding_cap_is_equivalent_to_previous_post_encoding_cap(annotation, monkeypatch):
    records = json.loads(annotation.read_text())["images"]
    raw = []
    for record in records:
        masks, scores, labels = candidates((record["height"], record["width"]), 500)
        raw.extend(prediction_from_mask(record["id"], label, score, mask)
                   for mask, score, label in zip(masks, scores, labels))
    legacy = evaluate_predictions(annotation, cap_predictions(raw), evaluate_bbox=False)
    encoded = []
    original = validation.prediction_from_mask

    def count(*args):
        encoded.append(args[:3])
        return original(*args)

    monkeypatch.setattr(validation, "prediction_from_mask", count)
    result = collect_metrics(annotation, annotation.parent,
                             lambda path: candidates((24, 32) if path.name == "a.png" else (48, 64), 500),
                             COMMON_VALIDATION)
    assert len(encoded) == result["prediction_count"] == 400
    for key, value in legacy.items():
        assert result[key] == value, key


def test_collect_uses_prefetch_interface_and_closes_on_mask_error(annotation):
    class Predictor:
        closed = False

        def __call__(self, path):
            pytest.fail("collect_metrics must use the ordered iterator")

        def iter_predictions(self, paths):
            try:
                for _ in paths:
                    yield [np.zeros((1, 1))], [0.9], [1]
            finally:
                self.closed = True

    predictor = Predictor()
    with pytest.raises(ValueError, match="contract"):
        collect_metrics(annotation, annotation.parent, predictor, COMMON_VALIDATION)
    assert predictor.closed


def test_cached_ground_truth_is_reused_and_invalidated(annotation):
    before = annotation.read_bytes()
    first = prepared_evaluator(annotation)
    assert prepared_evaluator(annotation) is first
    first.evaluate([], evaluate_bbox=False)
    assert annotation.read_bytes() == before
    record = json.loads(annotation.read_text())
    record["info"]["updated"] = True
    annotation.write_text(json.dumps(record), encoding="utf-8")
    assert prepared_evaluator(annotation) is not first


def test_profile_uses_largest_images_same_cap_but_no_subset_metrics(annotation, monkeypatch):
    called = []

    def predict(path):
        called.append(path.name)
        return candidates((48, 64), 500)

    def unexpected(*args, **kwargs):
        pytest.fail("A speed profile must not calculate partial AP")

    monkeypatch.setattr(validation, "evaluate_predictions", unexpected)
    monkeypatch.setattr(validation, "prepared_evaluator", unexpected)
    result = collect_metrics(annotation, annotation.parent, predict, COMMON_VALIDATION, profile_images=1)
    assert called == ["b.png"] and result["profile_only"] is True
    assert result["images"] == 1 and result["prediction_count"] == 200
    assert not any(key.startswith("mask_ap") for key in result)
    assert result["per_image_timing"][0]["image_id"] == 4
    assert all(result["timing"][key] >= 0 for key in (
        "setup_seconds", "predict_seconds", "encode_seconds", "evaluation_seconds", "input_wait_seconds"))
    assert result["evaluation_protocol"] == COMMON_VALIDATION["id"]


def test_invalid_candidate_outside_cap_is_still_rejected(annotation):
    masks, scores, labels = candidates((24, 32), 500)
    masks[-1] = np.zeros((3, 3))
    with pytest.raises(ValueError, match="contract"):
        collect_metrics(annotation, annotation.parent, lambda _: (masks, scores, labels), COMMON_VALIDATION)


@pytest.mark.parametrize("count", [1, 3])
def test_wrong_iterator_length_is_rejected(annotation, count):
    class Predictor:
        def iter_predictions(self, paths):
            for _ in range(count):
                yield [], [], []

    with pytest.raises(ValueError, match="images than requested"):
        collect_metrics(annotation, annotation.parent, Predictor(), COMMON_VALIDATION)


def test_worker_profiles_common_validation_not_legacy_predictor(monkeypatch, tmp_path):
    import validation_adapters
    import worker

    called = {}

    def unified(job, prepared, run_dir, checkpoint, output_dir, profile_images=0):
        called.update(profile_images=profile_images, output_dir=output_dir)
        return dict(profile_only=True)

    monkeypatch.setattr(validation_adapters, "evaluate_unified", unified)
    result = worker.evaluate(dict(validation=COMMON_VALIDATION), tmp_path, tmp_path, tmp_path / "best.pt",
                             profile_images=5, output_dir=tmp_path / "profile")
    assert result == dict(profile_only=True)
    assert called == dict(profile_images=5, output_dir=tmp_path / "profile")


@pytest.mark.parametrize("output", [None, "same"])
def test_direct_profile_cannot_overwrite_model_run(tmp_path, output):
    import worker

    with pytest.raises(ValueError, match="separate output"):
        worker.evaluate(dict(validation=COMMON_VALIDATION), tmp_path, tmp_path, tmp_path / "best.pt",
                        profile_images=5, output_dir=tmp_path if output == "same" else None)
