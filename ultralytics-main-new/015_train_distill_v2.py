"""
Train the 015 mobile student with teacher distillation from the 014 ROI model.

Pipeline:
YOLO segmentation -> flower ROI image + mask -> student heatmap.
The stage-1 segmentation model and ROI/GT matching logic are reused from 014.
"""

import importlib.util
import json
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from ultralytics import YOLO


def load_module(module_name, filename):
    module_path = os.path.join(os.path.dirname(__file__), filename)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


stage14 = load_module("stage14_train_module", "014_train_improved_v2.py")
teacher_net_module = load_module("stage14_net_module", "014_improved_net_v2.py")
student_net_module = load_module("stage15_net_module", "015_improved_net_v2.py")

TeacherROIHeatmapNet = teacher_net_module.ROIHeatmapNet
StudentROIHeatmapNet = student_net_module.ROIHeatmapNet

SAVE_DIR = os.path.join(stage14.RESULTS_DIR, "15_roi_heatmap_distill")
DEFAULT_TEACHER_WEIGHTS = os.path.join(stage14.RESULTS_DIR, "14_roi_heatmap_lite", "best.pth")

TRAIN_CONFIG = {
    "seed": 42,
    "epochs": 100,
    "batch_size": 16,
    "roi_size": 128,
    "heatmap_size": 64,
    "student_base_channels": 8,
    "teacher_base_channels": 16,
    "teacher_weights": DEFAULT_TEACHER_WEIGHTS,
    "seg_model_path": stage14.SEG_MODEL_PATH,
    "seg_conf": 0.25,
    "lr": 1e-3,
    "weight_decay": 1e-4,
    "gt_weight": 1.0,
    "distill_weight": 0.7,
    "distill_temperature": 2.0,
    "distill_coord_weight": 0.25,
    "max_train_samples": 0,
    "max_val_samples": 0,
    "save_dir": SAVE_DIR,
    "cache": True,
    "max_visualizations": 0,
    "candidate_class_ids": [0, 3],
}


def spatial_kl_distill(student_logits, teacher_logits, temperature=2.0):
    b = student_logits.shape[0]
    student_flat = student_logits.reshape(b, -1) / temperature
    teacher_flat = teacher_logits.reshape(b, -1) / temperature
    return F.kl_div(
        F.log_softmax(student_flat, dim=1),
        F.softmax(teacher_flat, dim=1),
        reduction="batchmean",
    ) * (temperature * temperature)


def distillation_loss(student_logits, teacher_logits, temperature=2.0, coord_weight=0.25):
    kl_loss = spatial_kl_distill(student_logits, teacher_logits, temperature)
    student_xy = stage14.soft_argmax_2d(student_logits)
    teacher_xy = stage14.soft_argmax_2d(teacher_logits)
    coord_loss = F.smooth_l1_loss(student_xy, teacher_xy)
    return kl_loss + coord_weight * coord_loss, kl_loss.detach(), coord_loss.detach()


def load_teacher(config, device):
    teacher = TeacherROIHeatmapNet(
        in_channels=4,
        base_channels=config["teacher_base_channels"],
        output_size=config["heatmap_size"],
    ).to(device)

    if not os.path.exists(config["teacher_weights"]):
        raise FileNotFoundError(
            f"Teacher weights not found: {config['teacher_weights']}. Train 014 first or update TRAIN_CONFIG."
        )

    checkpoint = torch.load(config["teacher_weights"], map_location=device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


def main():
    config = TRAIN_CONFIG.copy()

    random.seed(config["seed"])
    np.random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])

    print("=" * 60)
    print("Train 015 distilled ROIHeatmapNet student")
    print("=" * 60)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = config["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    print(f"Device: {device}")
    print(f"Segmentation model: {config['seg_model_path']}")
    print(f"Segmentation conf: {config['seg_conf']}")
    print(f"Teacher weights: {config['teacher_weights']}")
    print(f"Save dir: {save_dir}")

    candidate_class_ids = stage14.parse_candidate_class_ids(config["candidate_class_ids"])
    print(f"Stage-2 candidate classes: {stage14.format_candidate_class_ids(candidate_class_ids)}")

    seg_model = YOLO(config["seg_model_path"])
    teacher = load_teacher(config, device)

    train_dataset = stage14.YOLOROIHeatmapDataset(
        stage14.TRAIN_IMG_DIR,
        stage14.TRAIN_LABEL_DIR,
        seg_model,
        roi_size=config["roi_size"],
        heatmap_size=config["heatmap_size"],
        max_samples=config["max_train_samples"],
        cache=config["cache"],
        candidate_class_ids=candidate_class_ids,
        seg_conf=config["seg_conf"],
    )
    val_dataset = stage14.YOLOROIHeatmapDataset(
        stage14.VAL_IMG_DIR,
        stage14.VAL_LABEL_DIR,
        seg_model,
        roi_size=config["roi_size"],
        heatmap_size=config["heatmap_size"],
        max_samples=config["max_val_samples"],
        cache=config["cache"],
        candidate_class_ids=candidate_class_ids,
        seg_conf=config["seg_conf"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    print(f"Train source images: {len(train_dataset.img_files)}")
    print(f"Val source images: {len(val_dataset.img_files)}")
    print(f"Train matched ROI samples: {len(train_dataset)}")
    print(f"Val matched ROI samples: {len(val_dataset)}")
    print(f"Train index stats: {train_dataset.index_stats}")
    print(f"Val index stats: {val_dataset.index_stats}")
    print(f"ROI size: {config['roi_size']}")
    print(f"Heatmap size: {config['heatmap_size']}")

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise RuntimeError("No matched ROI samples were found. Check YOLO masks, labels, and GT matching distance.")

    student = StudentROIHeatmapNet(
        in_channels=4,
        base_channels=config["student_base_channels"],
        output_size=config["heatmap_size"],
    ).to(device)

    teacher_params = sum(param.numel() for param in teacher.parameters())
    student_params = sum(param.numel() for param in student.parameters() if param.requires_grad)
    print(f"Teacher params: {teacher_params:,}")
    print(f"Student params: {student_params:,}")
    print(f"Student/teacher params: {student_params / max(teacher_params, 1):.3f}")

    optimizer = torch.optim.AdamW(student.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(config["epochs"], 1),
        eta_min=config["lr"] * 0.01,
    )

    best_map = -1.0
    best_loss = float("inf")
    best_epoch = 0
    train_losses = []
    train_gt_losses = []
    train_distill_losses = []
    val_losses = []
    val_maps = []

    epoch_pbar = tqdm(range(config["epochs"]), desc="Distill", ncols=120)

    for epoch in epoch_pbar:
        student.train()
        train_loss = 0.0
        train_gt_loss = 0.0
        train_distill_loss = 0.0
        train_kl_loss = 0.0
        train_teacher_coord_loss = 0.0
        train_batches = 0

        for batch in train_loader:
            valid = batch["valid"].bool()
            if not valid.any():
                continue

            roi = batch["roi"][valid].to(device)
            target_heatmap = batch["heatmap"][valid].to(device)
            target_roi_xy = batch["gt_roi_xy"][valid].to(device)

            student_logits = student(roi)
            with torch.no_grad():
                teacher_logits = teacher(roi)

            gt_loss, _, _ = stage14.heatmap_coord_loss(student_logits, target_heatmap, target_roi_xy)
            distill, kl_loss, teacher_coord_loss = distillation_loss(
                student_logits,
                teacher_logits,
                temperature=config["distill_temperature"],
                coord_weight=config["distill_coord_weight"],
            )
            loss = config["gt_weight"] * gt_loss + config["distill_weight"] * distill

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += float(loss.item())
            train_gt_loss += float(gt_loss.item())
            train_distill_loss += float(distill.item())
            train_kl_loss += float(kl_loss.item())
            train_teacher_coord_loss += float(teacher_coord_loss.item())
            train_batches += 1

        scheduler.step()

        train_loss /= max(train_batches, 1)
        train_gt_loss /= max(train_batches, 1)
        train_distill_loss /= max(train_batches, 1)
        train_kl_loss /= max(train_batches, 1)
        train_teacher_coord_loss /= max(train_batches, 1)
        val_loss, mean_error, median_error, _, _, map_metrics = stage14.evaluate(student, val_loader, device)

        train_losses.append(train_loss)
        train_gt_losses.append(train_gt_loss)
        train_distill_losses.append(train_distill_loss)
        val_losses.append(val_loss)
        val_maps.append(map_metrics["mAP50-95"])

        is_better = (
            map_metrics["mAP50-95"] > best_map
            or (map_metrics["mAP50-95"] == best_map and val_loss < best_loss)
        )
        if is_better:
            best_map = map_metrics["mAP50-95"]
            best_loss = val_loss
            best_epoch = epoch
            torch.save(
                {
                    "model": student.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch + 1,
                    "best_mAP50-95": best_map,
                    "best_val_loss": best_loss,
                    "student_params": student_params,
                    "teacher_params": teacher_params,
                    "teacher_weights": config["teacher_weights"],
                    "train_index_stats": train_dataset.index_stats,
                    "val_index_stats": val_dataset.index_stats,
                },
                os.path.join(save_dir, "best.pth"),
            )

        epoch_pbar.set_postfix(
            {
                "loss": f"{train_loss:.4f}",
                "gt": f"{train_gt_loss:.4f}",
                "kd": f"{train_distill_loss:.4f}",
                "err": f"{mean_error:.1f}px",
                "mAP95": f"{map_metrics['mAP50-95']:.3f}",
            }
        )

    epoch_pbar.close()

    checkpoint_path = os.path.join(save_dir, "best.pth")
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        student.load_state_dict(checkpoint["model"])

    final_loss, mean_error, median_error, all_errors, all_oks, map_metrics = stage14.evaluate(
        student,
        val_loader,
        device,
    )
    all_errors_arr = np.asarray(all_errors, dtype=np.float32)

    if config["epochs"] <= 0:
        best_loss = final_loss
        best_map = map_metrics["mAP50-95"]
        best_epoch = -1
        if not os.path.exists(checkpoint_path):
            torch.save(
                {
                    "model": student.state_dict(),
                    "config": config,
                    "best_epoch": 0,
                    "best_mAP50-95": best_map,
                    "best_val_loss": best_loss,
                    "student_params": student_params,
                    "teacher_params": teacher_params,
                    "teacher_weights": config["teacher_weights"],
                    "train_index_stats": train_dataset.index_stats,
                    "val_index_stats": val_dataset.index_stats,
                },
                checkpoint_path,
            )

    print("\n" + "=" * 60)
    print("Final evaluation")
    print("=" * 60)
    print(f"Best epoch: {best_epoch + 1 if best_epoch >= 0 else 0}")
    print(f"Best mAP50-95: {best_map:.4f}")
    print(f"Final val loss: {final_loss:.6f}")
    print(f"Samples: {len(all_errors_arr)}")
    if len(all_errors_arr) > 0:
        print(f"Mean error: {mean_error:.2f} px")
        print(f"Median error: {median_error:.2f} px")
        print(f"<10px: {np.sum(all_errors_arr < 10)} ({np.mean(all_errors_arr < 10) * 100:.1f}%)")
        print(f"<20px: {np.sum(all_errors_arr < 20)} ({np.mean(all_errors_arr < 20) * 100:.1f}%)")
        print(f"<30px: {np.sum(all_errors_arr < 30)} ({np.mean(all_errors_arr < 30) * 100:.1f}%)")
    else:
        print("Mean error: N/A")
        print("Median error: N/A")
    print(f"mAP50: {map_metrics['mAP50']:.4f}")
    print(f"mAP50-95: {map_metrics['mAP50-95']:.4f}")

    visualization_dir, visualization_records = stage14.save_prediction_visualizations(
        student,
        val_dataset,
        device,
        save_dir,
        max_visualizations=config["max_visualizations"],
    )
    print(f"Visualizations: {visualization_dir} ({len(visualization_records)} images)")

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        axes[0].plot(train_losses, label="Train Total")
        axes[0].plot(train_gt_losses, label="GT Loss")
        axes[0].plot(train_distill_losses, label="Distill Loss")
        axes[0].plot(val_losses, label="Val GT Loss")
        if best_epoch >= 0:
            axes[0].axvline(best_epoch, color="red", linestyle="--", alpha=0.5, label=f"Best {best_epoch + 1}")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Distillation Curve")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(val_maps, label="Val mAP50-95")
        if best_epoch >= 0:
            axes[1].axvline(best_epoch, color="red", linestyle="--", alpha=0.5)
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("mAP50-95")
        axes[1].set_title("Validation mAP")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, "training_curve.png"), dpi=150)
        plt.close(fig)
    except Exception as exc:
        print(f"Skipped training curve: {exc}")

    with open(os.path.join(save_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "model_type": "DistilledMobileROIHeatmapNet",
                "direction": "YOLO segmentation + ROI RGB/mask + distilled mobile keypoint heatmap",
                "teacher_model": "014 ROIHeatmapNet",
                "student_model": "015 ROIHeatmapNet",
                "teacher_weights": config["teacher_weights"],
                "seg_model_path": config["seg_model_path"],
                "candidate_class_ids": None if candidate_class_ids is None else sorted(candidate_class_ids),
                "candidate_classes": stage14.format_candidate_class_ids(candidate_class_ids),
                "best_epoch": int(best_epoch + 1 if best_epoch >= 0 else 0),
                "best_val_loss": float(best_loss),
                "best_mAP50-95": float(best_map),
                "final_val_loss": float(final_loss),
                "num_matched_val_samples": int(len(all_errors_arr)),
                "train_index_stats": train_dataset.index_stats,
                "val_index_stats": val_dataset.index_stats,
                "mean_error_px": float(mean_error) if len(all_errors_arr) > 0 else 0.0,
                "median_error_px": float(median_error) if len(all_errors_arr) > 0 else 0.0,
                "mAP50": map_metrics["mAP50"],
                "mAP50-95": map_metrics["mAP50-95"],
                "oks_mean": map_metrics["oks_mean"],
                "oks_median": map_metrics["oks_median"],
                "ap_by_threshold": map_metrics["ap_by_threshold"],
                "visualization_dir": visualization_dir,
                "num_visualizations": int(len(visualization_records)),
                "teacher_params": int(teacher_params),
                "student_params": int(student_params),
                "student_teacher_param_ratio": float(student_params / max(teacher_params, 1)),
                "train_losses": train_losses,
                "train_gt_losses": train_gt_losses,
                "train_distill_losses": train_distill_losses,
                "val_losses": val_losses,
                "val_mAP50-95": val_maps,
                "config": config,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"Checkpoint: {checkpoint_path}")
    print(f"Results: {os.path.join(save_dir, 'results.json')}")
    print(f"Visualization index: {os.path.join(visualization_dir, 'index.json')}")


if __name__ == "__main__":
    main()
