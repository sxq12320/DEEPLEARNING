"""Evaluate citrus-specific mask quality and topology without replacing standard COCO AP.

The standard ``eval_citrus_seg.py`` remains the source for Mask AP. This script adds diagnostics that are otherwise
absent from Ultralytics CSVs: low-solidity/tiny/near-neighbour recalls, boundary F1, and split/merge errors. Predictions
are matched to ground truth at a deliberately low confidence threshold so candidate-recall failures remain visible.

Examples:
    python eval_citrus_challenges.py --weights runs/model/weights/best.pt --data /data/orange/data.yaml --split val
    python eval_citrus_challenges.py --weights best.pt --data data.yaml --difficulty-csv instances.csv --split test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".bmp", ".dng", ".jpeg", ".jpg", ".mpo", ".png", ".tif", ".tiff", ".webp"}


@dataclass(frozen=True)
class Match:
    """One one-to-one prediction/ground-truth association."""

    pred: int
    gt: int
    iou: float


def flattened_masks(masks: list[np.ndarray]) -> np.ndarray:
    """Stack masks as a compact boolean ``[instances, pixels]`` matrix."""
    return np.stack(masks).astype(bool).reshape(len(masks), -1)


def intersection_matrix(pred: np.ndarray, gt: np.ndarray, chunk_size: int = 16) -> np.ndarray:
    """Compute intersections in chunks to avoid materializing all masks as float32 at once."""
    output = np.empty((len(pred), len(gt)), dtype=np.float32)
    gt_float = gt.astype(np.float32)
    for start in range(0, len(pred), chunk_size):
        stop = min(start + chunk_size, len(pred))
        output[start:stop] = pred[start:stop].astype(np.float32) @ gt_float.T
    return output


def mask_iou_matrix(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> np.ndarray:
    """Return pairwise binary-mask IoU with shape ``[n_pred, n_gt]``."""
    if not pred_masks or not gt_masks:
        return np.zeros((len(pred_masks), len(gt_masks)), dtype=np.float32)
    pred = flattened_masks(pred_masks)
    gt = flattened_masks(gt_masks)
    intersection = intersection_matrix(pred, gt)
    pred_area = pred.sum(axis=1, dtype=np.float64)[:, None]
    gt_area = gt.sum(axis=1, dtype=np.float64)[None, :]
    union = pred_area + gt_area - intersection
    return np.divide(intersection, union, out=np.zeros_like(union, dtype=np.float64), where=union > 0).astype(np.float32)


def greedy_iou_match(iou: np.ndarray, threshold: float = 0.5) -> list[Match]:
    """Greedily select non-conflicting maximum-IoU pairs above ``threshold``."""
    work = np.asarray(iou, dtype=np.float32).copy()
    matches: list[Match] = []
    while work.size and float(work.max(initial=0.0)) >= threshold:
        pred, gt = np.unravel_index(int(work.argmax()), work.shape)
        value = float(work[pred, gt])
        matches.append(Match(pred=pred, gt=gt, iou=value))
        work[pred, :] = -1.0
        work[:, gt] = -1.0
    return matches


def mask_boundary(mask: np.ndarray) -> np.ndarray:
    """Extract a one-pixel inner boundary from a binary mask."""
    binary = np.asarray(mask, dtype=np.uint8)
    if not binary.any():
        return np.zeros_like(binary, dtype=bool)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8), iterations=1)
    return (binary > 0) & (eroded == 0)


def boundary_f1(pred: np.ndarray, gt: np.ndarray, tolerance: int = 2) -> float:
    """Compute symmetric boundary F1 using a pixel tolerance band."""
    pred_boundary = mask_boundary(pred)
    gt_boundary = mask_boundary(gt)
    if not pred_boundary.any() and not gt_boundary.any():
        return 1.0
    if not pred_boundary.any() or not gt_boundary.any():
        return 0.0
    kernel_size = 2 * max(int(tolerance), 0) + 1
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    pred_band = cv2.dilate(pred_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    gt_band = cv2.dilate(gt_boundary.astype(np.uint8), kernel, iterations=1).astype(bool)
    precision = float(np.logical_and(pred_boundary, gt_band).sum()) / float(pred_boundary.sum())
    recall = float(np.logical_and(gt_boundary, pred_band).sum()) / float(gt_boundary.sum())
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def overlap_fractions(pred_masks: list[np.ndarray], gt_masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return intersection divided by prediction area and by ground-truth area."""
    if not pred_masks or not gt_masks:
        shape = (len(pred_masks), len(gt_masks))
        return np.zeros(shape, dtype=np.float32), np.zeros(shape, dtype=np.float32)
    pred = flattened_masks(pred_masks)
    gt = flattened_masks(gt_masks)
    intersection = intersection_matrix(pred, gt)
    pred_area = pred.sum(axis=1, dtype=np.float64)[:, None]
    gt_area = gt.sum(axis=1, dtype=np.float64)[None, :]
    pred_fraction = np.divide(
        intersection, pred_area, out=np.zeros_like(intersection, dtype=np.float64), where=pred_area > 0
    )
    gt_fraction = np.divide(
        intersection, gt_area, out=np.zeros_like(intersection, dtype=np.float64), where=gt_area > 0
    )
    return pred_fraction.astype(np.float32), gt_fraction.astype(np.float32)


def topology_errors(
    pred_masks: list[np.ndarray],
    gt_masks: list[np.ndarray],
    split_pred_purity: float = 0.5,
    split_gt_coverage: float = 0.15,
    merge_gt_coverage: float = 0.15,
) -> tuple[list[int], list[int]]:
    """Return GT indices split by multiple predictions and prediction indices merging multiple GTs.

    A split fragment must lie mostly inside one GT and cover a non-trivial part of it. A merge prediction must cover a
    non-trivial part of at least two GT instances. Thresholds are diagnostic and must be reported in sensitivity tests.
    """
    pred_fraction, gt_fraction = overlap_fractions(pred_masks, gt_masks)
    split_gt = [
        gt
        for gt in range(len(gt_masks))
        if int(np.logical_and(pred_fraction[:, gt] >= split_pred_purity, gt_fraction[:, gt] >= split_gt_coverage).sum())
        > 1
    ]
    merge_pred = [
        pred for pred in range(len(pred_masks)) if int((gt_fraction[pred, :] >= merge_gt_coverage).sum()) > 1
    ]
    return split_gt, merge_pred


def resolve_source(path_value: str, root: Path, yaml_dir: Path) -> Path:
    """Resolve one Ultralytics dataset path entry."""
    source = Path(path_value).expanduser()
    if source.is_absolute():
        return source
    root_candidate = (root / source).resolve()
    return root_candidate if root_candidate.exists() else (yaml_dir / source).resolve()


def image_paths_from_data(data_yaml: Path, split: str) -> list[Path]:
    """Resolve a directory, text manifest, image path, or list from a dataset YAML split."""
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or split not in config:
        raise ValueError(f"Dataset YAML has no split {split!r}: {data_yaml}")
    root_value = Path(str(config.get("path", data_yaml.parent))).expanduser()
    root = root_value if root_value.is_absolute() else (data_yaml.parent / root_value).resolve()
    values = config[split] if isinstance(config[split], list) else [config[split]]
    images: list[Path] = []
    for value in values:
        source = resolve_source(str(value), root, data_yaml.parent)
        if source.is_dir():
            images.extend(path for path in source.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)
        elif source.suffix.lower() == ".txt":
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    images.append(resolve_source(line.strip(), root, source.parent))
        elif source.suffix.lower() in IMAGE_SUFFIXES:
            images.append(source)
        else:
            raise FileNotFoundError(f"Unsupported or missing split source: {source}")
    unique = sorted({path.resolve() for path in images})
    if not unique:
        raise FileNotFoundError(f"No images resolved for {split!r} from {data_yaml}")
    return unique


def label_path_for_image(image_path: Path) -> Path:
    """Map a standard ``images/...`` path to ``labels/...txt``."""
    parts = list(image_path.parts)
    for index in range(len(parts) - 1, -1, -1):
        if parts[index].lower() == "images":
            parts[index] = "labels"
            return Path(*parts).with_suffix(".txt")
    return image_path.parent.parent / "labels" / image_path.with_suffix(".txt").name


def polygon_to_mask(points: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Rasterize normalized or pixel-space polygon coordinates."""
    height, width = shape
    polygon = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if polygon.size and float(np.nanmax(polygon)) <= 1.5:
        polygon = polygon * np.array([width, height], dtype=np.float32)
    polygon[:, 0] = np.clip(polygon[:, 0], 0, max(width - 1, 0))
    polygon[:, 1] = np.clip(polygon[:, 1], 0, max(height - 1, 0))
    mask = np.zeros((height, width), dtype=np.uint8)
    if len(polygon) >= 3:
        cv2.fillPoly(mask, [np.rint(polygon).astype(np.int32)], 1)
    return mask.astype(bool)


def scaled_shape(shape: tuple[int, int], long_side: int) -> tuple[int, int]:
    """Scale an image shape to a fixed long side while preserving aspect ratio."""
    height, width = shape
    scale = float(long_side) / float(max(height, width))
    return max(1, int(round(height * scale))), max(1, int(round(width * scale)))


def load_yolo_masks(label_path: Path, shape: tuple[int, int]) -> list[np.ndarray]:
    """Load one-class YOLO polygon labels in file order."""
    if not label_path.is_file():
        return []
    masks: list[np.ndarray] = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        coordinates = np.asarray([float(value) for value in tokens[1:]], dtype=np.float32)
        if coordinates.size < 6 or coordinates.size % 2:
            raise ValueError(f"Invalid polygon at {label_path}:{line_number}")
        masks.append(polygon_to_mask(coordinates.reshape(-1, 2), shape))
    return masks


def load_difficulty_tags(csv_path: Path | None, split: str) -> dict[tuple[str, int], set[str]]:
    """Load canonical per-instance challenge tags keyed by image filename and label-row index."""
    if csv_path is None:
        return {}
    tags: dict[tuple[str, int], set[str]] = {}
    with csv_path.open(newline="", encoding="utf-8-sig") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") != split:
                continue
            current = {"all"}
            if int(row["coco_small_lt32sq"]):
                current.add("small")
            if int(row["ultra_tiny_min_side_lt8"]):
                current.add("ultra_tiny")
            if int(row["tiny_min_side_lt16"]):
                current.add("tiny_lt16")
            if float(row["solidity"]) < 0.85:
                current.add("concave")
            if int(row["neighbor_gap_le2_count"]) > 0:
                current.add("near_gap2")
            if int(row["neighbor_gap_le4_count"]) > 0:
                current.add("near_gap4")
            if float(row["lab_delta_e_to_context"]) < 10.0:
                current.add("low_contrast")
            if float(row["boundary_to_context_gradient_ratio"]) < 1.0:
                current.add("weak_boundary")
            if "tiny_lt16" in current and "low_contrast" in current:
                current.add("tiny_low_contrast")
            if "concave" in current and "near_gap2" in current:
                current.add("concave_near")
            tags[(row["image"], int(row["instance"]))] = current
    return tags


def prediction_masks(
    result, evaluation_shape: tuple[int, int], original_shape: tuple[int, int]
) -> tuple[list[np.ndarray], list[float]]:
    """Convert an Ultralytics result into fixed-size diagnostic masks and scores."""
    if result.masks is None or result.boxes is None:
        return [], []
    original_height, original_width = original_shape
    scale = np.array([original_width, original_height], dtype=np.float32)
    polygons = [np.asarray(points, dtype=np.float32) / scale for points in result.masks.xy]
    scores = result.boxes.conf.detach().cpu().numpy().astype(float).tolist()
    masks = [polygon_to_mask(points, evaluation_shape) for points in polygons]
    count = min(len(masks), len(scores))
    return masks[:count], scores[:count]


def summarize_rows(rows: list[dict], topology: dict) -> dict:
    """Aggregate per-instance rows by challenge tag."""
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        for tag in row["tags"]:
            groups[tag].append(row)
    summary = {"topology": topology, "subsets": {}}
    for tag, items in sorted(groups.items()):
        matched = [item for item in items if item["matched"]]
        summary["subsets"][tag] = {
            "gt_instances": len(items),
            "matched_iou50": len(matched),
            "recall_iou50": round(len(matched) / len(items), 6) if items else None,
            "mean_iou_matched": round(float(np.mean([item["iou"] for item in matched])), 6) if matched else None,
            "mean_boundary_f1": round(float(np.mean([item["boundary_f1"] for item in matched])), 6)
            if matched
            else None,
        }
    return summary


def write_outputs(output: Path, rows: list[dict], summary: dict) -> None:
    """Write auditable per-instance CSV and aggregate JSON."""
    output.mkdir(parents=True, exist_ok=False)
    csv_path = output / "per_instance.csv"
    fieldnames = ("image", "gt_instance", "tags", "matched", "pred_instance", "score", "iou", "boundary_f1")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            serialized = dict(row)
            serialized["tags"] = ";".join(sorted(row["tags"]))
            writer.writerow(serialized)
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--difficulty-csv", type=Path, default=None)
    parser.add_argument("--split", choices=("val", "test"), default="val")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", type=float, default=0.001)
    parser.add_argument("--nms-iou", type=float, default=0.7)
    parser.add_argument("--match-iou", type=float, default=0.5)
    parser.add_argument("--boundary-tolerance", type=int, default=2)
    parser.add_argument("--mask-eval-size", type=int, default=640, help="Long side used to rasterize diagnostic masks.")
    parser.add_argument("--max-det", type=int, default=300)
    parser.add_argument("--limit", type=int, default=0, help="Smoke-test image limit; use 0 for formal evaluation.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> tuple[list[dict], dict]:
    """Run inference and return per-instance and aggregate challenge diagnostics."""
    from ultralytics import YOLO

    image_paths = image_paths_from_data(args.data.resolve(), args.split)
    if args.limit > 0:
        image_paths = image_paths[: args.limit]
    tag_map = load_difficulty_tags(args.difficulty_csv.resolve() if args.difficulty_csv else None, args.split)
    model = YOLO(str(args.weights.resolve()), task="segment")
    rows: list[dict] = []
    topology_totals = {"images": 0, "gt_instances": 0, "pred_instances": 0, "split_gt": 0, "merge_pred": 0}
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        original_shape = image.shape[:2]
        evaluation_shape = scaled_shape(original_shape, args.mask_eval_size)
        gt_masks = load_yolo_masks(label_path_for_image(image_path), evaluation_shape)
        result = model.predict(
            source=str(image_path),
            imgsz=args.imgsz,
            conf=args.conf,
            iou=args.nms_iou,
            max_det=args.max_det,
            device=args.device,
            retina_masks=True,
            verbose=False,
        )[0]
        pred_masks, scores = prediction_masks(result, evaluation_shape, original_shape)
        iou = mask_iou_matrix(pred_masks, gt_masks)
        matches = greedy_iou_match(iou, args.match_iou)
        by_gt = {match.gt: match for match in matches}
        split_gt, merge_pred = topology_errors(pred_masks, gt_masks)
        topology_totals["images"] += 1
        topology_totals["gt_instances"] += len(gt_masks)
        topology_totals["pred_instances"] += len(pred_masks)
        topology_totals["split_gt"] += len(split_gt)
        topology_totals["merge_pred"] += len(merge_pred)
        for gt_index, gt_mask in enumerate(gt_masks):
            match = by_gt.get(gt_index)
            tags = tag_map.get((image_path.name, gt_index), {"all"})
            rows.append(
                {
                    "image": image_path.name,
                    "gt_instance": gt_index,
                    "tags": tags,
                    "matched": match is not None,
                    "pred_instance": match.pred if match else "",
                    "score": round(scores[match.pred], 6) if match else "",
                    "iou": round(match.iou, 6) if match else 0.0,
                    "boundary_f1": round(
                        boundary_f1(pred_masks[match.pred], gt_mask, args.boundary_tolerance), 6
                    )
                    if match
                    else 0.0,
                }
            )
    topology_totals["split_rate_per_gt"] = round(
        topology_totals["split_gt"] / max(topology_totals["gt_instances"], 1), 6
    )
    topology_totals["merge_rate_per_prediction"] = round(
        topology_totals["merge_pred"] / max(topology_totals["pred_instances"], 1), 6
    )
    summary = summarize_rows(rows, topology_totals)
    summary["protocol"] = {
        "weights": str(args.weights.resolve()),
        "data": str(args.data.resolve()),
        "difficulty_csv": str(args.difficulty_csv.resolve()) if args.difficulty_csv else None,
        "split": args.split,
        "imgsz": args.imgsz,
        "conf": args.conf,
        "nms_iou": args.nms_iou,
        "match_iou": args.match_iou,
        "boundary_tolerance": args.boundary_tolerance,
        "mask_eval_size": args.mask_eval_size,
        "max_det": args.max_det,
        "limit": args.limit,
        "warning": "Diagnostic recall/BF1/topology only; standard COCO Mask AP must be reported separately.",
    }
    return rows, summary


def main() -> None:
    """Run challenge evaluation and write versioned outputs."""
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite completed evaluation: {args.output}")
    for path in (args.weights, args.data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.difficulty_csv is not None and not args.difficulty_csv.is_file():
        raise FileNotFoundError(args.difficulty_csv)
    if args.mask_eval_size <= 0:
        raise ValueError("--mask-eval-size must be positive")
    if args.limit < 0:
        raise ValueError("--limit cannot be negative")
    rows, summary = evaluate(args)
    write_outputs(args.output.resolve(), rows, summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Saved challenge evaluation: {args.output.resolve()}")


if __name__ == "__main__":
    main()
