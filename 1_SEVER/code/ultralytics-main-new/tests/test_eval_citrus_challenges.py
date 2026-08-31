"""Synthetic tests for citrus challenge metrics."""

from pathlib import Path

import cv2
import numpy as np
import yaml

from eval_citrus_challenges import (
    boundary_f1,
    greedy_iou_match,
    image_paths_from_data,
    mask_iou_matrix,
    polygon_to_mask,
    scaled_shape,
    summarize_rows,
    topology_errors,
)


def rectangle(y1: int, y2: int, x1: int, x2: int, size: int = 32) -> np.ndarray:
    """Create a binary rectangular mask."""
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def test_mask_iou_and_greedy_matching_are_one_to_one():
    gt = [rectangle(2, 10, 2, 10), rectangle(18, 28, 18, 28)]
    pred = [gt[1].copy(), gt[0].copy(), rectangle(0, 2, 0, 2)]
    iou = mask_iou_matrix(pred, gt)
    matches = greedy_iou_match(iou, threshold=0.5)
    assert {(match.pred, match.gt, match.iou) for match in matches} == {(0, 1, 1.0), (1, 0, 1.0)}


def test_boundary_f1_rewards_exact_boundary_and_penalizes_shift():
    gt = rectangle(6, 24, 6, 24)
    shifted = rectangle(9, 27, 9, 27)
    assert boundary_f1(gt, gt, tolerance=1) == 1.0
    assert 0.0 < boundary_f1(shifted, gt, tolerance=1) < 1.0


def test_topology_errors_detect_split_and_merge():
    one_gt = [rectangle(4, 28, 4, 28)]
    split_pred = [rectangle(4, 28, 4, 15), rectangle(4, 28, 17, 28)]
    split_gt, merge_pred = topology_errors(split_pred, one_gt)
    assert split_gt == [0]
    assert merge_pred == []

    two_gt = [rectangle(4, 14, 4, 14), rectangle(4, 14, 17, 27)]
    merged = np.logical_or(two_gt[0], two_gt[1])
    split_gt, merge_pred = topology_errors([merged], two_gt)
    assert split_gt == []
    assert merge_pred == [0]


def test_polygon_to_mask_accepts_normalized_coordinates():
    points = np.array([[0.25, 0.25], [0.75, 0.25], [0.75, 0.75], [0.25, 0.75]], dtype=np.float32)
    mask = polygon_to_mask(points, (20, 40))
    assert mask.shape == (20, 40)
    assert int(mask.sum()) > 0


def test_scaled_shape_preserves_aspect_and_bounds_memory():
    assert scaled_shape((3000, 4000), 640) == (480, 640)
    assert scaled_shape((4000, 3000), 640) == (640, 480)


def test_image_paths_from_data_resolves_directory(tmp_path: Path):
    root = tmp_path / "dataset"
    image_dir = root / "images" / "val"
    image_dir.mkdir(parents=True)
    cv2.imwrite(str(image_dir / "a.jpg"), np.zeros((8, 8, 3), dtype=np.uint8))
    data = tmp_path / "data.yaml"
    data.write_text(yaml.safe_dump({"path": str(root), "val": "images/val", "names": {0: "orange"}}))
    assert image_paths_from_data(data, "val") == [(image_dir / "a.jpg").resolve()]


def test_summarize_rows_keeps_challenge_subsets_separate():
    rows = [
        {"tags": {"all", "tiny_lt16"}, "matched": True, "iou": 0.8, "boundary_f1": 0.7},
        {"tags": {"all", "concave"}, "matched": False, "iou": 0.0, "boundary_f1": 0.0},
    ]
    summary = summarize_rows(rows, {"split_gt": 0, "merge_pred": 0})
    assert summary["subsets"]["all"]["recall_iou50"] == 0.5
    assert summary["subsets"]["tiny_lt16"]["recall_iou50"] == 1.0
    assert summary["subsets"]["concave"]["recall_iou50"] == 0.0
