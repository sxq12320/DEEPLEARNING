"""
One-Click Training Script for Amodal Tri-Branch Segmentation.

Usage:
    python train.py \
        --model_yaml ../ultralytics-main-new/mine_yaml/amodal_tribranch.yaml \
        --data_root /path/to/dataset \
        --data_format npy \
        --epochs 100 \
        --batch_size 4 \
        --lr 0.001 \
        --img_size 640
"""

import os
import sys
import argparse
import time
import json
import logging

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

from model import AmodalYOLO
from dataset import AmodalRGBDDataset
from loss import TriBranchAmodalLoss


def setup_logging(save_dir):
    """Setup logging to both console and file."""
    log_file = os.path.join(save_dir, 'train.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(log_file, mode='a'),
        ],
    )
    return logging.getLogger(__name__)


def build_dataloader(dataset, batch_size, num_workers, shuffle=True):
    """Build DataLoader with proper settings."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=shuffle,  # drop last incomplete batch for training
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch.

    Returns:
        avg_loss: average total loss over the epoch
        avg_loss_dict: average of each sub-loss over the epoch
    """
    model.train()
    total_loss = 0.0
    loss_accum = {}
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Train]', ncols=120)
    for batch_idx, batch in enumerate(pbar):
        # ---- Move data to device ----
        rgbd = batch['rgbd'].to(device, non_blocking=True)
        amodal_gt = batch['amodal_gt'].to(device, non_blocking=True)
        pseudo_vis = batch['pseudo_vis'].to(device, non_blocking=True)
        pseudo_occ = batch['pseudo_occ'].to(device, non_blocking=True)
        rgb_edges = batch['rgb_edges'].to(device, non_blocking=True)

        # ---- Forward ----
        H, W = amodal_gt.shape[2], amodal_gt.shape[3]
        pred_vis, pred_occ, pred_full = model(rgbd, target_size=(H, W))

        # ---- Loss ----
        total_l, loss_dict = criterion(
            pred_vis, pred_occ, pred_full,
            amodal_gt, pseudo_vis, pseudo_occ, rgb_edges,
        )

        # ---- Backward ----
        optimizer.zero_grad()
        total_l.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        # ---- Accumulate ----
        total_loss += total_l.item()
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        # ---- Progress bar ----
        pbar.set_postfix({
            'loss': f'{total_l.item():.4f}',
            'sup': f'{loss_dict["L_sup_full"]:.4f}',
            'excl': f'{loss_dict["L_excl"]:.4f}',
            'union': f'{loss_dict["L_union"]:.4f}',
        })

    # ---- Epoch averages ----
    avg_loss = total_loss / max(num_batches, 1)
    avg_loss_dict = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    return avg_loss, avg_loss_dict


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch, logger):
    """Validate the model.

    Returns:
        avg_loss: average total loss over the validation set
        avg_loss_dict: average of each sub-loss
    """
    model.eval()
    total_loss = 0.0
    loss_accum = {}
    num_batches = len(dataloader)

    pbar = tqdm(dataloader, desc=f'Epoch {epoch + 1} [Val]', ncols=120)
    for batch in pbar:
        rgbd = batch['rgbd'].to(device, non_blocking=True)
        amodal_gt = batch['amodal_gt'].to(device, non_blocking=True)
        pseudo_vis = batch['pseudo_vis'].to(device, non_blocking=True)
        pseudo_occ = batch['pseudo_occ'].to(device, non_blocking=True)
        rgb_edges = batch['rgb_edges'].to(device, non_blocking=True)

        H, W = amodal_gt.shape[2], amodal_gt.shape[3]
        pred_vis, pred_occ, pred_full = model(rgbd, target_size=(H, W))

        total_l, loss_dict = criterion(
            pred_vis, pred_occ, pred_full,
            amodal_gt, pseudo_vis, pseudo_occ, rgb_edges,
        )

        total_loss += total_l.item()
        for k, v in loss_dict.items():
            loss_accum[k] = loss_accum.get(k, 0.0) + v

        pbar.set_postfix({'val_loss': f'{total_l.item():.4f}'})

    avg_loss = total_loss / max(num_batches, 1)
    avg_loss_dict = {k: v / max(num_batches, 1) for k, v in loss_accum.items()}

    return avg_loss, avg_loss_dict


def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, save_path, is_best=False):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'val_loss': val_loss,
    }
    torch.save(checkpoint, save_path)
    if is_best:
        best_path = os.path.join(os.path.dirname(save_path), 'best_model.pt')
        torch.save(checkpoint, best_path)


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load training checkpoint for resuming."""
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint.get('epoch', 0), checkpoint.get('val_loss', float('inf'))


def parse_args():
    parser = argparse.ArgumentParser(description='Amodal Tri-Branch Segmentation Training')

    # Model
    parser.add_argument('--model_yaml', type=str, required=True,
                        help='Path to YOLO11-seg YAML config (backbone+neck definition)')
    parser.add_argument('--p3_layer_idx', type=int, default=16,
                        help='Layer index for P3 features in the YOLO model (default: 16 for YOLO11-seg)')
    parser.add_argument('--head_mid_channels', type=int, default=128,
                        help='Intermediate channels in the tri-branch head')

    # Data
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root directory of the dataset')
    parser.add_argument('--data_format', type=str, default='npy', choices=['npy', 'separate'],
                        help='Dataset format: "npy" (4-channel numpy + polygon labels) or "separate" (separate dirs)')
    parser.add_argument('--img_size', type=int, default=640,
                        help='Input image size (square)')

    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay for AdamW')
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')

    # Loss weights
    parser.add_argument('--w_sup', type=float, default=1.0,
                        help='Weight for strong supervision loss')
    parser.add_argument('--w_weak', type=float, default=0.2,
                        help='Weight for weak supervision loss')
    parser.add_argument('--w_excl', type=float, default=0.5,
                        help='Weight for exclusivity constraint')
    parser.add_argument('--w_subset', type=float, default=0.5,
                        help='Weight for subset constraint')
    parser.add_argument('--w_union', type=float, default=0.3,
                        help='Weight for union consistency')
    parser.add_argument('--w_edge', type=float, default=0.2,
                        help='Weight for edge alignment')

    # Misc
    parser.add_argument('--save_dir', type=str, default='./runs/amodal_tribranch',
                        help='Directory to save checkpoints and logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint for resuming training')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (e.g., "cuda:0", "cpu"). Auto-detect if not specified')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility')

    return parser.parse_args()


def main():
    args = parse_args()

    # ---- Setup ----
    os.makedirs(args.save_dir, exist_ok=True)

    # Device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Device] {device}")

    # Seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Logging
    logger = setup_logging(args.save_dir)
    logger.info(f"Arguments: {vars(args)}")

    # ---- Dataset & Dataloader ----
    train_dataset = AmodalRGBDDataset(
        root=args.data_root,
        split='train',
        img_size=args.img_size,
        data_format=args.data_format,
        augment=True,
    )
    val_dataset = AmodalRGBDDataset(
        root=args.data_root,
        split='val',
        img_size=args.img_size,
        data_format=args.data_format,
        augment=False,
    )

    train_loader = build_dataloader(train_dataset, args.batch_size, args.workers, shuffle=True)
    val_loader = build_dataloader(val_dataset, args.batch_size, args.workers, shuffle=False)

    # ---- Model ----
    model = AmodalYOLO(
        cfg=args.model_yaml,
        ch=4,
        nc=1,
        p3_layer_idx=args.p3_layer_idx,
        head_mid_channels=args.head_mid_channels,
    )
    model.to(device)

    # ---- Loss ----
    criterion = TriBranchAmodalLoss(
        w_sup=args.w_sup,
        w_weak=args.w_weak,
        w_excl=args.w_excl,
        w_subset=args.w_subset,
        w_union=args.w_union,
        w_edge=args.w_edge,
    )

    # ---- Optimizer ----
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # ---- Scheduler ----
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ---- Resume ----
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume and os.path.isfile(args.resume):
        start_epoch, best_val_loss = load_checkpoint(args.resume, model, optimizer, scheduler)
        start_epoch += 1  # resume from next epoch
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ---- Training Loop ----
    logger.info("=" * 80)
    logger.info("Training started")
    logger.info("=" * 80)

    history = []

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss, train_loss_dict = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, logger
        )

        # Validate
        val_loss, val_loss_dict = validate(
            model, val_loader, criterion, device, epoch, logger
        )

        # Scheduler step
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]['lr']

        # ---- Logging ----
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        log_line = (
            f"Epoch {epoch + 1}/{args.epochs} | "
            f"Time: {epoch_time:.1f}s | LR: {current_lr:.6f} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Best: {best_val_loss:.4f} | {'***BEST***' if is_best else ''}"
        )
        logger.info(log_line)

        # Detailed sub-loss logging
        sub_log = "  ".join(
            f"{k}: {v:.4f}" for k, v in train_loss_dict.items() if k != 'L_total'
        )
        logger.info(f"  Train sub-losses: {sub_log}")
        val_sub_log = "  ".join(
            f"{k}: {v:.4f}" for k, v in val_loss_dict.items() if k != 'L_total'
        )
        logger.info(f"  Val   sub-losses: {val_sub_log}")

        # ---- Save checkpoint ----
        ckpt_path = os.path.join(args.save_dir, f'epoch_{epoch + 1:03d}.pt')
        save_checkpoint(model, optimizer, scheduler, epoch, val_loss, ckpt_path, is_best=is_best)

        # ---- Save history ----
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'train_sub_losses': {k: v for k, v in train_loss_dict.items() if k != 'L_total'},
            'val_sub_losses': {k: v for k, v in val_loss_dict.items() if k != 'L_total'},
        })
        history_path = os.path.join(args.save_dir, 'history.json')
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    logger.info("=" * 80)
    logger.info(f"Training completed! Best val loss: {best_val_loss:.4f}")
    logger.info(f"Best model saved at: {os.path.join(args.save_dir, 'best_model.pt')}")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
