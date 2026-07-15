"""Dataset, loss, watershed, training, and evaluation helpers for U-Net."""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
from tqdm.auto import tqdm

from baseline_common import resolve_path
from coco_utils import evaluate_predictions, prediction_from_mask


@dataclass(frozen=True)
class LetterboxInfo:
    """Geometry needed to remove padding and restore an original image."""

    original_height: int
    original_width: int
    resized_height: int
    resized_width: int
    pad_top: int
    pad_left: int
    output_size: int


def letterbox_array(
    array: np.ndarray,
    output_size: int,
    *,
    is_mask: bool = False,
    fill: int | Tuple[int, int, int] = 0,
) -> Tuple[np.ndarray, LetterboxInfo]:
    """Resize an image with preserved aspect ratio and center padding."""
    if output_size <= 0:
        raise ValueError("output_size must be positive")
    source = np.asarray(array)
    if source.ndim not in (2, 3):
        raise ValueError(f"Expected a 2D or 3D array, got shape {source.shape}")
    original_height, original_width = source.shape[:2]
    scale = min(output_size / original_width, output_size / original_height)
    resized_width = max(1, min(output_size, round(original_width * scale)))
    resized_height = max(1, min(output_size, round(original_height * scale)))
    resampling = Image.Resampling.NEAREST if is_mask else Image.Resampling.BILINEAR
    resized = np.asarray(
        Image.fromarray(source).resize((resized_width, resized_height), resampling)
    )
    pad_left = (output_size - resized_width) // 2
    pad_top = (output_size - resized_height) // 2
    if source.ndim == 2:
        canvas = np.full((output_size, output_size), fill, dtype=resized.dtype)
    else:
        canvas = np.empty(
            (output_size, output_size, source.shape[2]), dtype=resized.dtype
        )
        canvas[...] = fill
    canvas[
        pad_top : pad_top + resized_height,
        pad_left : pad_left + resized_width,
    ] = resized
    return canvas, LetterboxInfo(
        original_height=original_height,
        original_width=original_width,
        resized_height=resized_height,
        resized_width=resized_width,
        pad_top=pad_top,
        pad_left=pad_left,
        output_size=output_size,
    )


def crop_letterbox(array: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Remove padding from a letterboxed 2D array."""
    return np.asarray(array)[
        info.pad_top : info.pad_top + info.resized_height,
        info.pad_left : info.pad_left + info.resized_width,
    ]


def restore_binary_mask(mask: np.ndarray, info: LetterboxInfo) -> np.ndarray:
    """Remove letterbox padding and resize a binary mask to original resolution."""
    content = crop_letterbox(mask, info)
    restored = Image.fromarray((content > 0).astype(np.uint8) * 255).resize(
        (info.original_width, info.original_height),
        Image.Resampling.NEAREST,
    )
    return np.asarray(restored) > 0


def semantic_split_paths(dataset_root: Path, split: str) -> Tuple[Path, Path, Path]:
    """Return semantic image, mask, and COCO annotation paths."""
    root = resolve_path(dataset_root)
    return (
        root / "semantic" / "images" / split,
        root / "semantic" / "masks" / split,
        root / "coco" / "annotations" / f"instances_{split}.json",
    )


def validate_semantic_dataset(
    dataset_root: Path,
    splits: Iterable[str],
) -> Dict[str, Dict[str, Any]]:
    """Validate paired semantic files and their shared COCO image records."""
    reports: Dict[str, Dict[str, Any]] = {}
    for split in splits:
        image_dir, mask_dir, annotation_path = semantic_split_paths(dataset_root, split)
        if not image_dir.is_dir():
            raise FileNotFoundError(f"Missing semantic image directory: {image_dir}")
        if not mask_dir.is_dir():
            raise FileNotFoundError(f"Missing semantic mask directory: {mask_dir}")
        if not annotation_path.is_file():
            raise FileNotFoundError(f"Missing COCO annotation: {annotation_path}")
        coco = json.loads(annotation_path.read_text(encoding="utf-8"))
        records = coco.get("images", [])
        missing: List[str] = []
        for record in records:
            image_path = image_dir / str(record["file_name"])
            mask_path = mask_dir / f"{Path(str(record['file_name'])).stem}.png"
            if not image_path.is_file():
                missing.append(str(image_path))
            if not mask_path.is_file():
                missing.append(str(mask_path))
        if missing:
            raise FileNotFoundError(
                f"{len(missing)} semantic files are missing; first: {missing[0]}"
            )
        reports[split] = {
            "images": len(records),
            "instances": len(coco.get("annotations", [])),
            "image_dir": str(image_dir),
            "mask_dir": str(mask_dir),
            "annotation": str(annotation_path),
        }
    return reports


class CitrusSemanticDataset:
    """Load binary citrus masks while retaining COCO image IDs for evaluation."""

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        imgsz: int,
        train: bool = False,
    ) -> None:
        self.image_dir, self.mask_dir, self.annotation_path = semantic_split_paths(
            dataset_root, split
        )
        if not self.annotation_path.is_file():
            raise FileNotFoundError(
                f"COCO annotation not found: {self.annotation_path}"
            )
        coco = json.loads(self.annotation_path.read_text(encoding="utf-8"))
        self.records = sorted(coco.get("images", []), key=lambda item: int(item["id"]))
        self.imgsz = int(imgsz)
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        import torch

        record = self.records[index]
        file_name = str(record["file_name"])
        image_path = self.image_dir / file_name
        mask_path = self.mask_dir / f"{Path(file_name).stem}.png"
        image = np.asarray(Image.open(image_path).convert("RGB"))
        mask = np.asarray(Image.open(mask_path).convert("L"))
        if image.shape[:2] != mask.shape:
            raise ValueError(
                f"Image/mask size mismatch: {image_path} {image.shape[:2]} vs "
                f"{mask_path} {mask.shape}"
            )
        if self.train and random.random() < 0.5:
            image = np.ascontiguousarray(image[:, ::-1])
            mask = np.ascontiguousarray(mask[:, ::-1])
        boxed_image, info = letterbox_array(image, self.imgsz, fill=(114, 114, 114))
        boxed_mask, _ = letterbox_array(mask, self.imgsz, is_mask=True, fill=0)
        image_tensor = (
            torch.from_numpy(np.ascontiguousarray(boxed_image))
            .permute(2, 0, 1)
            .float()
            .div(255.0)
        )
        mask_tensor = (
            torch.from_numpy(np.ascontiguousarray(boxed_mask > 0)).unsqueeze(0).float()
        )
        metadata = {
            "image_id": int(record["id"]),
            "file_name": file_name,
            "original_height": info.original_height,
            "original_width": info.original_width,
            "resized_height": info.resized_height,
            "resized_width": info.resized_width,
            "pad_top": info.pad_top,
            "pad_left": info.pad_left,
            "output_size": info.output_size,
        }
        return image_tensor, mask_tensor, metadata


def collate_semantic_batch(batch):
    """Stack images and masks while retaining per-image geometry dictionaries."""
    import torch

    images, masks, metadata = zip(*batch)
    return torch.stack(images), torch.stack(masks), list(metadata)


def build_unet(
    encoder_name: str = "resnet18",
    encoder_weights: str | None = "imagenet",
):
    """Build a one-channel segmentation_models_pytorch U-Net."""
    try:
        import segmentation_models_pytorch as smp
    except ImportError as exc:
        raise RuntimeError(
            "segmentation_models_pytorch is required. Install requirements-unet.txt."
        ) from exc
    return smp.Unet(
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=1,
        activation=None,
    )


def bce_dice_loss(logits, targets, dice_weight: float = 1.0):
    """Return combined BCE-with-logits and soft Dice loss components."""
    import torch
    import torch.nn.functional as functional

    bce = functional.binary_cross_entropy_with_logits(logits, targets)
    probabilities = torch.sigmoid(logits)
    intersection = (probabilities * targets).sum(dim=(1, 2, 3))
    denominator = probabilities.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()
    total = bce + float(dice_weight) * dice_loss
    return total, bce, dice_loss


def binary_statistics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> Tuple[int, int, int]:
    """Return intersection, predicted pixels, and target pixels."""
    predicted = np.asarray(probabilities) >= threshold
    target = np.asarray(targets) > 0
    return (
        int(np.logical_and(predicted, target).sum()),
        int(predicted.sum()),
        int(target.sum()),
    )


def watershed_instances(
    probability: np.ndarray,
    info: LetterboxInfo,
    *,
    probability_threshold: float = 0.5,
    min_distance: int = 8,
    min_area: int = 20,
    max_instances: int = 50,
) -> List[Tuple[np.ndarray, float]]:
    """Split foreground with distance-transform watershed and restore instances."""
    if not 0.0 < probability_threshold < 1.0:
        raise ValueError("probability_threshold must be in (0, 1)")
    if min_distance <= 0 or min_area <= 0 or max_instances <= 0:
        raise ValueError("Watershed integer parameters must be positive")
    try:
        from scipy import ndimage as ndi
        from skimage.feature import peak_local_max
        from skimage.segmentation import watershed
    except ImportError as exc:
        raise RuntimeError(
            "scipy and scikit-image are required for watershed instance separation."
        ) from exc

    content_probability = crop_letterbox(np.asarray(probability), info)
    foreground = content_probability >= probability_threshold
    if not np.any(foreground):
        return []
    foreground = ndi.binary_fill_holes(foreground)
    component_labels, _ = ndi.label(foreground)
    component_sizes = np.bincount(component_labels.ravel())
    keep = component_sizes >= min_area
    keep[0] = False
    foreground = keep[component_labels]
    if not np.any(foreground):
        return []

    distance = ndi.distance_transform_edt(foreground)
    coordinates = peak_local_max(
        distance,
        labels=foreground,
        min_distance=min_distance,
        exclude_border=False,
    )
    markers = np.zeros(foreground.shape, dtype=np.int32)
    for marker_id, (row, column) in enumerate(coordinates, start=1):
        markers[row, column] = marker_id
    if not np.any(markers):
        markers, _ = ndi.label(foreground)
    labels = watershed(-distance, markers, mask=foreground)

    candidates: List[Tuple[np.ndarray, float, int]] = []
    for label_id in range(1, int(labels.max()) + 1):
        content_mask = labels == label_id
        area = int(content_mask.sum())
        if area < min_area:
            continue
        score = float(content_probability[content_mask].mean())
        boxed_mask = np.zeros((info.output_size, info.output_size), dtype=np.uint8)
        boxed_mask[
            info.pad_top : info.pad_top + info.resized_height,
            info.pad_left : info.pad_left + info.resized_width,
        ] = content_mask
        candidates.append((restore_binary_mask(boxed_mask, info), score, area))
    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    return [(mask, score) for mask, score, _ in candidates[:max_instances]]


def metadata_to_info(metadata: Dict[str, Any]) -> LetterboxInfo:
    """Convert a collated geometry dictionary into LetterboxInfo."""
    return LetterboxInfo(
        original_height=int(metadata["original_height"]),
        original_width=int(metadata["original_width"]),
        resized_height=int(metadata["resized_height"]),
        resized_width=int(metadata["resized_width"]),
        pad_top=int(metadata["pad_top"]),
        pad_left=int(metadata["pad_left"]),
        output_size=int(metadata["output_size"]),
    )


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    scaler,
    epoch: int,
    probability_threshold: float,
    dice_weight: float,
) -> Dict[str, float]:
    """Train one epoch with a batch progress bar and running metrics."""
    import torch

    model.train()
    totals = {"loss": 0.0, "bce_loss": 0.0, "dice_loss": 0.0}
    intersection = predicted_pixels = target_pixels = sample_count = 0
    amp_enabled = scaler is not None and scaler.is_enabled()
    progress = tqdm(
        loader,
        total=len(loader),
        desc=f"Train epoch {epoch}",
        unit="batch",
        dynamic_ncols=True,
    )
    for images, masks, _ in progress:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            logits = model(images)
            loss, bce, dice_loss = bce_dice_loss(logits, masks, dice_weight)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite U-Net loss: {float(loss)}")
        if amp_enabled:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        batch_size = int(images.shape[0])
        sample_count += batch_size
        totals["loss"] += float(loss.detach().cpu()) * batch_size
        totals["bce_loss"] += float(bce.detach().cpu()) * batch_size
        totals["dice_loss"] += float(dice_loss.detach().cpu()) * batch_size
        batch_intersection, batch_predicted, batch_target = binary_statistics(
            torch.sigmoid(logits).detach().cpu().numpy(),
            masks.detach().cpu().numpy(),
            probability_threshold,
        )
        intersection += batch_intersection
        predicted_pixels += batch_predicted
        target_pixels += batch_target
        dice = (2 * intersection + 1) / (predicted_pixels + target_pixels + 1)
        progress.set_postfix(
            loss=f"{float(loss.detach().cpu()):.4f}",
            dice=f"{dice:.4f}",
            lr=f"{optimizer.param_groups[0]['lr']:.3g}",
        )
    union = predicted_pixels + target_pixels - intersection
    metrics = {name: value / max(sample_count, 1) for name, value in totals.items()}
    metrics.update(
        {
            "dice": (2 * intersection + 1) / (predicted_pixels + target_pixels + 1),
            "iou": (intersection + 1) / (union + 1),
        }
    )
    return metrics


def evaluate_model(
    model,
    loader,
    device,
    annotation_path: Path,
    *,
    probability_threshold: float,
    min_distance: int,
    min_area: int,
    max_instances: int,
    description: str = "Validate",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate semantic quality and watershed-derived COCO instance masks."""
    import torch

    model.eval()
    predictions: List[Dict[str, Any]] = []
    intersection = predicted_pixels = target_pixels = image_count = 0
    model_seconds = 0.0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with torch.inference_mode():
        progress = tqdm(
            loader,
            total=len(loader),
            desc=description,
            unit="batch",
            dynamic_ncols=True,
        )
        for images, masks, metadata_list in progress:
            images = images.to(device, non_blocking=True)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            logits = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            model_seconds += time.perf_counter() - started
            probabilities = torch.sigmoid(logits).cpu().numpy()[:, 0]
            targets = masks.numpy()[:, 0]

            for probability, target, metadata in zip(
                probabilities, targets, metadata_list
            ):
                image_count += 1
                (
                    current_intersection,
                    current_predicted,
                    current_target,
                ) = binary_statistics(probability, target, probability_threshold)
                intersection += current_intersection
                predicted_pixels += current_predicted
                target_pixels += current_target
                instances = watershed_instances(
                    probability,
                    metadata_to_info(metadata),
                    probability_threshold=probability_threshold,
                    min_distance=min_distance,
                    min_area=min_area,
                    max_instances=max_instances,
                )
                for instance_mask, score in instances:
                    predictions.append(
                        prediction_from_mask(
                            image_id=int(metadata["image_id"]),
                            category_id=1,
                            score=score,
                            mask=instance_mask,
                        )
                    )
            running_dice = (2 * intersection + 1) / (
                predicted_pixels + target_pixels + 1
            )
            running_iou = (intersection + 1) / (
                predicted_pixels + target_pixels - intersection + 1
            )
            progress.set_postfix(
                dice=f"{running_dice:.4f}",
                iou=f"{running_iou:.4f}",
                instances=len(predictions),
            )

    metrics = evaluate_predictions(
        resolve_path(annotation_path),
        predictions,
        evaluate_bbox=True,
    )
    union = predicted_pixels + target_pixels - intersection
    metrics.update(
        {
            "semantic_dice": (2 * intersection + 1)
            / (predicted_pixels + target_pixels + 1),
            "semantic_iou": (intersection + 1) / (union + 1),
            "images": image_count,
            "probability_threshold": probability_threshold,
            "watershed_min_distance": min_distance,
            "watershed_min_area": min_area,
            "max_instances": max_instances,
            "model_latency_ms_per_image": 1000.0 * model_seconds / max(image_count, 1),
            "peak_vram_mb": (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            ),
        }
    )
    return metrics, predictions
