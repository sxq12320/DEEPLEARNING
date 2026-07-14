"""Dataset, model, training, and evaluation helpers for Torchvision Mask R-CNN."""

from __future__ import annotations

import json
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np

from baseline_common import resolve_path
from coco_utils import evaluate_predictions, prediction_from_mask


class CocoMaskRCNNDataset:
    """Load a prepared COCO instance-segmentation split for Torchvision."""

    def __init__(self, annotation_path: Path, image_dir: Path, train: bool = False) -> None:
        try:
            from pycocotools.coco import COCO
        except ImportError as exc:
            raise RuntimeError("pycocotools is required: pip install pycocotools") from exc

        self.annotation_path = resolve_path(annotation_path)
        self.image_dir = resolve_path(image_dir)
        if not self.annotation_path.is_file():
            raise FileNotFoundError(f"COCO annotation file not found: {self.annotation_path}")
        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"COCO image directory not found: {self.image_dir}")
        self.coco = COCO(str(self.annotation_path))
        self.image_ids = sorted(self.coco.getImgIds())
        self.train = train

    def __len__(self) -> int:
        return len(self.image_ids)

    def get_height_and_width(self, index: int) -> Tuple[int, int]:
        """Return image dimensions without decoding image pixels."""
        record = self.coco.loadImgs([self.image_ids[index]])[0]
        return int(record["height"]), int(record["width"])

    def __getitem__(self, index: int):
        import torch
        from PIL import Image
        from torchvision.transforms.functional import pil_to_tensor

        image_id = int(self.image_ids[index])
        image_record = self.coco.loadImgs([image_id])[0]
        image_path = self.image_dir / str(image_record["file_name"])
        if not image_path.is_file():
            raise FileNotFoundError(f"Image referenced by COCO JSON is missing: {image_path}")
        image = pil_to_tensor(Image.open(image_path).convert("RGB")).float().div(255.0)
        height, width = int(image.shape[-2]), int(image.shape[-1])

        annotation_ids = self.coco.getAnnIds(imgIds=[image_id])
        annotations = self.coco.loadAnns(annotation_ids)
        boxes: List[List[float]] = []
        labels: List[int] = []
        masks: List[np.ndarray] = []
        areas: List[float] = []
        crowds: List[int] = []
        for annotation in annotations:
            x, y, box_width, box_height = (float(value) for value in annotation["bbox"])
            if box_width <= 0 or box_height <= 0:
                continue
            mask = self.coco.annToMask(annotation).astype(np.uint8)
            if mask.shape != (height, width) or not np.any(mask):
                continue
            boxes.append([x, y, x + box_width, y + box_height])
            labels.append(int(annotation["category_id"]))
            masks.append(mask)
            areas.append(float(annotation.get("area", float(mask.sum()))))
            crowds.append(int(annotation.get("iscrowd", 0)))

        target: Dict[str, Any] = {
            "boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
            "labels": torch.as_tensor(labels, dtype=torch.int64),
            "masks": (
                torch.as_tensor(np.stack(masks), dtype=torch.uint8)
                if masks
                else torch.zeros((0, height, width), dtype=torch.uint8)
            ),
            "image_id": torch.tensor(image_id, dtype=torch.int64),
            "area": torch.as_tensor(areas, dtype=torch.float32),
            "iscrowd": torch.as_tensor(crowds, dtype=torch.uint8),
        }

        if self.train and random.random() < 0.5:
            image = image.flip(-1)
            target["masks"] = target["masks"].flip(-1)
            if len(target["boxes"]):
                boxes_tensor = target["boxes"].clone()
                boxes_tensor[:, 0] = width - target["boxes"][:, 2]
                boxes_tensor[:, 2] = width - target["boxes"][:, 0]
                target["boxes"] = boxes_tensor
        return image, target


def collate_detection_batch(batch):
    """Keep variable-size images and targets as tuples."""
    return tuple(zip(*batch))


def prepared_split_paths(dataset_root: Path, split: str) -> Tuple[Path, Path]:
    """Return annotation and image paths for one converted split."""
    root = resolve_path(dataset_root)
    return (
        root / "coco" / "annotations" / f"instances_{split}.json",
        root / "coco" / "images" / split,
    )


def validate_prepared_dataset(
    dataset_root: Path,
    splits: Iterable[str],
    class_names: Sequence[str],
) -> Dict[str, Dict[str, Any]]:
    """Validate files, IDs, categories, polygons, boxes, and areas."""
    reports: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        annotation_path, image_dir = prepared_split_paths(dataset_root, split)
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing annotation file: {annotation_path}")
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing image directory: {image_dir}")
        data = json.loads(annotation_path.read_text(encoding="utf-8"))
        images = data.get("images", [])
        annotations = data.get("annotations", [])
        categories = data.get("categories", [])
        category_ids = [int(category["id"]) for category in categories]
        category_names = [str(category["name"]) for category in categories]
        if category_ids != list(range(1, len(categories) + 1)):
            raise ValueError(f"{annotation_path}: category IDs must be consecutive and start from 1")
        if category_names != list(class_names):
            raise ValueError(f"{annotation_path}: categories {category_names} != expected {list(class_names)}")

        image_ids = [int(record["id"]) for record in images]
        if len(image_ids) != len(set(image_ids)):
            raise ValueError(f"{annotation_path}: duplicate image IDs")
        image_id_set = set(image_ids)
        missing = [
            str(image_dir / str(record["file_name"]))
            for record in images
            if not (image_dir / str(record["file_name"])).is_file()
        ]
        if missing:
            raise FileNotFoundError(f"{len(missing)} referenced images are missing; first: {missing[0]}")

        annotation_ids = [int(annotation["id"]) for annotation in annotations]
        if len(annotation_ids) != len(set(annotation_ids)):
            raise ValueError(f"{annotation_path}: duplicate annotation IDs")
        for annotation in annotations:
            annotation_id = annotation["id"]
            if int(annotation["image_id"]) not in image_id_set:
                raise ValueError(f"{annotation_path}: annotation {annotation_id} has an unknown image_id")
            if int(annotation["category_id"]) not in category_ids:
                raise ValueError(f"{annotation_path}: annotation {annotation_id} has an unknown category_id")
            bbox = annotation.get("bbox", [])
            if len(bbox) != 4 or float(bbox[2]) <= 0 or float(bbox[3]) <= 0:
                raise ValueError(f"{annotation_path}: annotation {annotation_id} has an invalid bbox")
            if float(annotation.get("area", 0)) <= 0:
                raise ValueError(f"{annotation_path}: annotation {annotation_id} has a non-positive area")
            segmentation = annotation.get("segmentation")
            if not isinstance(segmentation, list) or not segmentation:
                raise ValueError(f"{annotation_path}: annotation {annotation_id} has no polygon")
            for polygon in segmentation:
                if len(polygon) < 6 or len(polygon) % 2:
                    raise ValueError(f"{annotation_path}: annotation {annotation_id} has an invalid polygon")

        reports[split] = {
            "annotation": str(annotation_path),
            "image_dir": str(image_dir),
            "images": len(images),
            "instances": len(annotations),
            "categories": category_names,
        }
    return reports


def build_maskrcnn_model(
    num_foreground_classes: int,
    imgsz: int,
    initialization: str = "coco",
    detections_per_image: int = 300,
):
    """Build Mask R-CNN R50-FPN and replace its COCO prediction heads."""
    try:
        from torchvision.models import ResNet50_Weights
        from torchvision.models.detection import MaskRCNN_ResNet50_FPN_Weights, maskrcnn_resnet50_fpn
        from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
        from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    except ImportError as exc:
        raise RuntimeError("Torchvision detection models are unavailable. Reinstall matching torch/torchvision builds.") from exc

    mode = initialization.lower()
    if mode == "coco":
        weights = MaskRCNN_ResNet50_FPN_Weights.DEFAULT
        backbone_weights = None
    elif mode == "imagenet":
        weights = None
        backbone_weights = ResNet50_Weights.DEFAULT
    elif mode in {"none", "scratch"}:
        weights = None
        backbone_weights = None
    else:
        raise ValueError("initialization must be one of: coco, imagenet, none")

    model = maskrcnn_resnet50_fpn(
        weights=weights,
        weights_backbone=backbone_weights,
        min_size=imgsz,
        max_size=imgsz,
        box_detections_per_img=detections_per_image,
    )
    num_classes = num_foreground_classes + 1
    box_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(box_features, num_classes)
    mask_features = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_features, 256, num_classes)
    return model


def move_targets_to_device(targets: Sequence[Mapping[str, Any]], device) -> List[Dict[str, Any]]:
    """Move tensor fields in detection targets to the selected device."""
    return [
        {key: value.to(device) if hasattr(value, "to") else value for key, value in target.items()}
        for target in targets
    ]


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    epoch: int,
    log_interval: int,
) -> Dict[str, float]:
    """Train for one epoch and return mean loss components."""
    import torch

    model.train()
    totals: Dict[str, float] = defaultdict(float)
    sample_count = 0
    start = time.perf_counter()
    amp_enabled = scaler is not None and scaler.is_enabled()
    for step, (images, targets) in enumerate(loader, start=1):
        images = [image.to(device, non_blocking=True) for image in images]
        targets = move_targets_to_device(targets, device)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            loss_dict = model(images, targets)
            loss = sum(loss_dict.values())
        if not torch.isfinite(loss):
            details = {name: float(value.detach().cpu()) for name, value in loss_dict.items()}
            raise FloatingPointError(f"Non-finite training loss: {details}")
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_images = len(images)
        sample_count += batch_images
        totals["loss"] += float(loss.detach().cpu()) * batch_images
        for name, value in loss_dict.items():
            totals[name] += float(value.detach().cpu()) * batch_images
        if step == 1 or step % max(1, log_interval) == 0 or step == len(loader):
            elapsed = time.perf_counter() - start
            print(
                f"epoch={epoch} step={step}/{len(loader)} "
                f"loss={float(loss.detach().cpu()):.4f} lr={optimizer.param_groups[0]['lr']:.6g} "
                f"time={elapsed:.1f}s"
            )
    return {name: value / max(sample_count, 1) for name, value in totals.items()}


def evaluate_model(
    model,
    loader,
    device,
    annotation_path: Path,
    score_threshold: float = 0.001,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Run inference and evaluate predictions with the shared COCO protocol."""
    import torch

    model.eval()
    predictions: List[Dict[str, Any]] = []
    image_count = 0
    model_seconds = 0.0
    with torch.inference_mode():
        for images, targets in loader:
            device_images = [image.to(device, non_blocking=True) for image in images]
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            outputs = model(device_images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_seconds += time.perf_counter() - start

            for output, target in zip(outputs, targets):
                image_count += 1
                image_id = int(target["image_id"].item())
                scores = output["scores"].detach().cpu()
                labels = output["labels"].detach().cpu()
                masks = output["masks"].detach().cpu()
                for score, label, mask in zip(scores, labels, masks):
                    if float(score) < score_threshold:
                        continue
                    binary = mask[0].numpy() >= 0.5
                    if np.any(binary):
                        predictions.append(
                            prediction_from_mask(
                                image_id=image_id,
                                category_id=int(label),
                                score=float(score),
                                mask=binary,
                            )
                        )
    metrics = evaluate_predictions(resolve_path(annotation_path), predictions)
    metrics.update(
        {
            "images": image_count,
            "score_threshold_for_export": score_threshold,
            "model_latency_ms_per_image": 1000.0 * model_seconds / max(image_count, 1),
        }
    )
    return metrics, predictions
