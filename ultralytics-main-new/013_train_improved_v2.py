"""
Train ImprovedContourNetV2 and report pixel error plus keypoint mAP.
"""

import importlib.util
import json
import os
import random

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ultralytics import YOLO

from keypoint_map_utils import compute_batch_oks, compute_mask_area, summarize_single_keypoint_map


RESULTS_DIR = r"E:\mastercode\ultralytics-main-new\results"
SEG_MODEL_PATH = os.path.join(RESULTS_DIR, "09_watermelon_seg_2", "weights", "best.pt")
NUM_BOUNDARY_POINTS = 64
SAVE_DIR = os.path.join(RESULTS_DIR, "13_improved_net_v2")
MAX_GT_MATCH_DISTANCE_PX = 160


def load_improved_net_v2_class():
    module_path = os.path.join(os.path.dirname(__file__), "013_improved_net_v2.py")
    spec = importlib.util.spec_from_file_location("improved_net_v2_module", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.ImprovedContourNetV2


ImprovedContourNetV2 = load_improved_net_v2_class()


def extract_boundary_points(mask, num_points=64):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest_contour = max(contours, key=cv2.contourArea)
    contour_points = largest_contour.reshape(-1, 2)
    indices = np.linspace(0, len(contour_points) - 1, num_points).astype(int)
    sampled_points = contour_points[indices]
    h, w = mask.shape
    normalized = sampled_points.astype(np.float32)
    normalized[:, 0] /= w
    normalized[:, 1] /= h
    return normalized.flatten()


def extract_hsv_features(image, mask):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    flower_pixels = hsv[mask > 0]
    if len(flower_pixels) == 0:
        return np.zeros(3, dtype=np.float32)
    return np.array(
        [
            np.mean(flower_pixels[:, 0]) / 180.0,
            np.mean(flower_pixels[:, 1]) / 255.0,
            np.mean(flower_pixels[:, 2]) / 255.0,
        ],
        dtype=np.float32,
    )


def compute_flower_center(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flower_center = np.array([0.5, 0.5], dtype=np.float32)
    if not contours:
        return flower_center

    largest = max(contours, key=cv2.contourArea)
    moments = cv2.moments(largest)
    if moments["m00"] <= 0:
        return flower_center

    h, w = mask.shape
    return np.array(
        [moments["m10"] / moments["m00"] / w, moments["m01"] / moments["m00"] / h],
        dtype=np.float32,
    )


def select_gt_center(label_path, w, h, flower_center):
    if not os.path.exists(label_path):
        return None

    with open(label_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    candidates = []
    img_wh = np.array([w, h], dtype=np.float32)
    for shape in data.get("shapes", []):
        if shape.get("shape_type") != "point":
            continue
        label = shape.get("label", "")
        if label not in ("fully_visible", "partially_visible", "invisible"):
            continue
        points = shape.get("points") or []
        if not points:
            continue

        point = points[0]
        norm_pt = np.array([point[0] / w, point[1] / h], dtype=np.float32)
        distance_px = float(np.linalg.norm((norm_pt - flower_center) * img_wh))
        candidates.append((distance_px, label, norm_pt))

    if not candidates:
        return None

    distance_px, label, norm_pt = min(candidates, key=lambda item: item[0])
    if label == "invisible" or distance_px > MAX_GT_MATCH_DISTANCE_PX:
        return None
    return norm_pt


class YOLOSegPollinationDataset(Dataset):
    def __init__(self, img_dir, label_dir, seg_model, num_boundary_points=64):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.seg_model = seg_model
        self.num_boundary_points = num_boundary_points
        self.img_files = sorted(
            [f for f in os.listdir(img_dir) if f.endswith(".jpg") and not f.startswith("annotations")]
        )

    def __len__(self):
        return len(self.img_files)

    def __getitem__(self, idx):
        img_file = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_file)

        image = cv2.imread(img_path)
        if image is None:
            return self._dummy()
        h, w = image.shape[:2]

        results = self.seg_model.predict(img_path, conf=0.25, verbose=False)
        mask = np.zeros((h, w), dtype=np.uint8)
        if results[0].masks is not None:
            for result_mask in results[0].masks:
                mask_data = result_mask.data.cpu().numpy()[0]
                mask_resized = cv2.resize(mask_data, (w, h))
                mask[mask_resized > 0.5] = 255

        mask_area = compute_mask_area(mask)
        boundary = extract_boundary_points(mask, self.num_boundary_points)
        hsv_feat = extract_hsv_features(image, mask)
        flower_center = compute_flower_center(mask)

        label_path = os.path.join(self.label_dir, img_file.replace(".jpg", ".json"))
        gt_center = select_gt_center(label_path, w, h, flower_center)
        gt_valid = gt_center is not None
        if not gt_valid:
            gt_center = np.array([0.0, 0.0], dtype=np.float32)

        return {
            "boundary": torch.tensor(boundary, dtype=torch.float32)
            if boundary is not None
            else torch.zeros(self.num_boundary_points * 2, dtype=torch.float32),
            "hsv": torch.tensor(hsv_feat, dtype=torch.float32),
            "flower_center": torch.tensor(flower_center, dtype=torch.float32),
            "gt_center": torch.tensor(gt_center, dtype=torch.float32),
            "img_wh": torch.tensor([w, h], dtype=torch.float32),
            "mask_area": torch.tensor(mask_area, dtype=torch.float32),
            "valid": bool(boundary is not None and gt_valid),
        }

    def _dummy(self):
        return {
            "boundary": torch.zeros(self.num_boundary_points * 2, dtype=torch.float32),
            "hsv": torch.zeros(3, dtype=torch.float32),
            "flower_center": torch.tensor([0.5, 0.5], dtype=torch.float32),
            "gt_center": torch.tensor([0.0, 0.0], dtype=torch.float32),
            "img_wh": torch.tensor([1.0, 1.0], dtype=torch.float32),
            "mask_area": torch.tensor(0.0, dtype=torch.float32),
            "valid": False,
        }


def main():
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("Train ImprovedContourNetV2")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    print(f"Loading segmentation model: {SEG_MODEL_PATH}")
    seg_model = YOLO(SEG_MODEL_PATH)

    train_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\train",
        r"E:\mastercode\data\shr_watermelon\pose\labels\train",
        seg_model,
        NUM_BOUNDARY_POINTS,
    )
    val_dataset = YOLOSegPollinationDataset(
        r"E:\mastercode\data\shr_watermelon\segmentation\images\val",
        r"E:\mastercode\data\shr_watermelon\pose\labels\val",
        seg_model,
        NUM_BOUNDARY_POINTS,
    )

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=0)

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    model = ImprovedContourNetV2(num_boundary_points=NUM_BOUNDARY_POINTS, base_channels=64, num_blocks=3).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100, eta_min=1e-5)
    criterion = nn.SmoothL1Loss()

    os.makedirs(SAVE_DIR, exist_ok=True)

    best_loss = float("inf")
    best_epoch = 0
    train_losses = []
    val_losses = []

    epoch_pbar = tqdm(range(100), desc="Training", ncols=100)

    for epoch in epoch_pbar:
        model.train()
        train_loss = 0.0
        train_count = 0

        for batch in train_loader:
            valid = batch["valid"]
            if not valid.any():
                continue

            boundary = batch["boundary"][valid].to(device)
            hsv = batch["hsv"][valid].to(device)
            flower_center = batch["flower_center"][valid].to(device)
            gt_center = batch["gt_center"][valid].to(device)

            offset = model(boundary, hsv)
            pred_center = flower_center + offset
            loss = criterion(pred_center, gt_center)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item()
            train_count += 1

        scheduler.step()

        model.eval()
        val_loss = 0.0
        val_count = 0
        errors = []
        oks_scores = []

        with torch.no_grad():
            for batch in val_loader:
                valid = batch["valid"]
                if not valid.any():
                    continue

                boundary = batch["boundary"][valid].to(device)
                hsv = batch["hsv"][valid].to(device)
                flower_center = batch["flower_center"][valid].to(device)
                gt_center = batch["gt_center"][valid].to(device)
                img_wh = batch["img_wh"][valid].to(device)
                mask_area = batch["mask_area"][valid].cpu().numpy()

                offset = model(boundary, hsv)
                pred_center = flower_center + offset

                loss = criterion(pred_center, gt_center)
                val_loss += loss.item()
                val_count += 1

                pixel_errors = torch.norm((pred_center - gt_center) * img_wh, dim=1)
                errors.extend(pixel_errors.cpu().tolist())
                oks_scores.extend(
                    compute_batch_oks(
                        pred_center.detach().cpu().numpy(),
                        gt_center.detach().cpu().numpy(),
                        img_wh.detach().cpu().numpy(),
                        mask_area,
                    ).tolist()
                )

        train_loss /= max(train_count, 1)
        val_loss /= max(val_count, 1)
        mean_error = float(np.mean(errors)) if errors else 0.0
        map_metrics = summarize_single_keypoint_map(oks_scores)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        epoch_pbar.set_postfix(
            {
                "train": f"{train_loss:.6f}",
                "val": f"{val_loss:.6f}",
                "err": f"{mean_error:.1f}px",
                "mAP50": f"{map_metrics['mAP50']:.3f}",
            }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best.pth"))

    epoch_pbar.close()

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(train_losses, label="Train Loss", alpha=0.8)
    axes[0].plot(val_losses, label="Val Loss", alpha=0.8)
    axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best Epoch {best_epoch + 1}")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Curve")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(val_losses, label="Val Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].set_title("Validation Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "training_curve.png"), dpi=150)
    plt.close(fig)

    print("\n" + "=" * 60)
    print(f"Training finished. Best epoch: {best_epoch + 1}")
    print(f"Best val loss: {best_loss:.6f}")
    print(f"Checkpoint: {os.path.join(SAVE_DIR, 'best.pth')}")
    print("=" * 60)

    model.load_state_dict(torch.load(os.path.join(SAVE_DIR, "best.pth"), map_location=device))
    model.eval()

    all_errors = []
    all_oks = []
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating", ncols=80):
            valid = batch["valid"]
            if not valid.any():
                continue

            boundary = batch["boundary"][valid].to(device)
            hsv = batch["hsv"][valid].to(device)
            flower_center = batch["flower_center"][valid].to(device)
            gt_center = batch["gt_center"][valid].to(device)
            img_wh = batch["img_wh"][valid].to(device)
            mask_area = batch["mask_area"][valid].cpu().numpy()

            offset = model(boundary, hsv)
            pred_center = flower_center + offset

            pixel_errors = torch.norm((pred_center - gt_center) * img_wh, dim=1)
            all_errors.extend(pixel_errors.cpu().tolist())
            all_oks.extend(
                compute_batch_oks(
                    pred_center.detach().cpu().numpy(),
                    gt_center.detach().cpu().numpy(),
                    img_wh.detach().cpu().numpy(),
                    mask_area,
                ).tolist()
            )

    all_errors = np.asarray(all_errors, dtype=np.float32)
    map_metrics = summarize_single_keypoint_map(all_oks)

    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)
    print(f"Samples: {len(all_errors)}")
    if len(all_errors) > 0:
        print(f"Mean error: {np.mean(all_errors):.2f} px")
        print(f"Median error: {np.median(all_errors):.2f} px")
        print(f"<10px: {np.sum(all_errors < 10)} ({np.mean(all_errors < 10) * 100:.1f}%)")
        print(f"<20px: {np.sum(all_errors < 20)} ({np.mean(all_errors < 20) * 100:.1f}%)")
        print(f"<30px: {np.sum(all_errors < 30)} ({np.mean(all_errors < 30) * 100:.1f}%)")
    else:
        print("Mean error: N/A")
        print("Median error: N/A")
        print("<10px: 0 (0.0%)")
        print("<20px: 0 (0.0%)")
        print("<30px: 0 (0.0%)")
    print(f"mAP50: {map_metrics['mAP50']:.4f}")
    print(f"mAP50-95: {map_metrics['mAP50-95']:.4f}")

    with open(os.path.join(SAVE_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_epoch": best_epoch + 1,
                "best_val_loss": best_loss,
                "num_samples": int(len(all_errors)),
                "mean_error_px": float(np.mean(all_errors)) if len(all_errors) > 0 else 0.0,
                "median_error_px": float(np.median(all_errors)) if len(all_errors) > 0 else 0.0,
                "mAP50": map_metrics["mAP50"],
                "mAP50-95": map_metrics["mAP50-95"],
                "oks_mean": map_metrics["oks_mean"],
                "oks_median": map_metrics["oks_median"],
                "ap_by_threshold": map_metrics["ap_by_threshold"],
                "params": total_params,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )


if __name__ == "__main__":
    main()
