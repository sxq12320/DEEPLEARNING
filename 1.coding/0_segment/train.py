"""分割模型的一键训练入口。

支持命令行参数、JSON/YAML配置文件和API调用。所有的参数都可以被外部参数覆盖。

样例：
    python train.py
    python train.py --epochs 50 --batch 16 --lr 5e-4
    python train.py --cfg configs/train.json
    python train.py --cfg configs/train.yaml --imgsz 256
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

# ---------------------------------------------------------------------------
# 0. 路径初始化
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# 1. 默认配置
# ---------------------------------------------------------------------------
DEFAULT_CFG = {
    # ---- 模型 ----
    "model_type": "fpnseg",  # miniseg  fpnseg  ts_dual
    # ---- 数据 ----
    "image_dir": "",
    "mask_dir": "",
    "label_dir": "",
    "label_type": "mask",  # mask  txt  json  npy
    "depth_dir": "",
    "prompt_dir": "",
    "imgsz": 128,
    "num_classes": 1,
    "mask_threshold": 0.5,
    # ---- 训练 ----
    "epochs": 20,
    "batch": 8,
    "lr": 1e-3,
    "workers": 0,
    "synthetic_length": 32,
    "augment": True,
    # ---- 损失 ----
    "mask_loss": "bce",
    "mask_loss_weight": 1.0,
    "fourier_loss_weight": 0.2,
    "bbox_loss_weight": 1.0,
    "nwd_constant": 20.0,
    # ---- 设备 ----
    "cpu": False,
    # ---- 保存 ----
    "project": "checkpoints/results",
    "name": "train",
    # ---- 可复现性 ----
    "seed": 22,
}


# ---------------------------------------------------------------------------
# 2. 参数解析
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    """解析 CLI 参数。

    Args:
        argv (list, optional): 命令行参数列表，默认 None。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="Segmentation Training (one-click)")

    # 外部配置文件
    parser.add_argument(
        "--cfg", type=str, default="", help="Path to JSON/YAML config file"
    )
    parser.add_argument(
        "--save-cfg", type=str, default="", help="Save merged config to JSON file"
    )
    parser.add_argument(
        "--print-cfg", action="store_true", help="Print merged config and exit"
    )

    # 模型选择
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["miniseg", "fpnseg", "ts_dual"],
        help="Model architecture",
    )

    # 数据
    parser.add_argument("--image-dir", type=str, default=None)
    parser.add_argument("--mask-dir", type=str, default=None)
    parser.add_argument("--label-dir", type=str, default=None)
    parser.add_argument("--depth-dir", type=str, default=None)
    parser.add_argument("--prompt-dir", type=str, default=None)
    parser.add_argument(
        "--label-type", type=str, default=None, choices=["mask", "txt", "json", "npy"]
    )
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--mask-threshold", type=float, default=None)

    # 训练超参数
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--synthetic-length", type=int, default=None)
    parser.add_argument(
        "--mask-loss",
        type=str,
        default=None,
        choices=["bce", "cross_entropy"],
    )
    parser.add_argument("--mask-loss-weight", type=float, default=None)
    parser.add_argument("--fourier-loss-weight", type=float, default=None)
    parser.add_argument("--bbox-loss-weight", type=float, default=None)
    parser.add_argument("--nwd-constant", type=float, default=None)

    # 数据增强
    parser.add_argument("--augment", dest="augment", action="store_true", default=None)
    parser.add_argument(
        "--no-augment", dest="augment", action="store_false", default=None
    )
    parser.set_defaults(augment=None)

    # 设备
    parser.add_argument("--cpu", action="store_true", default=None)

    # 保存
    parser.add_argument("--project", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)

    # 可复现
    parser.add_argument("--seed", type=int, default=None)

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 3. 配置加载与合并
# ---------------------------------------------------------------------------
def load_cfg(path):
    """加载 JSON 或 YAML 配置文件。

    Args:
        path (str): 配置文件路径。

    Returns:
        dict: 配置信息字典。
    """
    if not path:
        return {}
    cfg_path = Path(path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")

    suffix = cfg_path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError:
            raise ImportError(
                "PyYAML is required for YAML configs.  pip install pyyaml"
            )
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    elif suffix == ".json":
        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config format: {suffix}  (use .json or .yaml)")

    if not isinstance(data, dict):
        raise ValueError("Config file must be a JSON/YAML object")
    return data


def merge_cfg(cli_args):
    """三层合并：默认值 → 配置文件 → CLI 覆盖。

    Args:
        cli_args (argparse.Namespace): 包含所有CLI参数的对象。

    Returns:
        dict: 合并后的最终配置字典。
    """
    cfg = DEFAULT_CFG.copy()
    file_cfg = load_cfg(cli_args.cfg)

    unknown = [k for k in file_cfg if k not in cfg]
    if unknown:
        print(f"[WARNING] unknown keys in config file (ignored): {unknown}")

    for k, v in file_cfg.items():
        if k in cfg:
            cfg[k] = v

    for k in cfg:
        v = getattr(cli_args, k, None)
        if v is not None:
            cfg[k] = v

    return cfg


def save_cfg(cfg, path):
    """保存合并后的配置为 JSON。

    Args:
        cfg (dict): 需要保存的配置字典。
        path (str): 保存配置文件的路径。

    Returns:
        None
    """
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)


# ---------------------------------------------------------------------------
# 4. 工具函数
# ---------------------------------------------------------------------------
def set_seed(seed):
    """固定随机种子以保证可复现性。

    Args:
        seed (int): 随机种子，如果为 None 则不固定。

    Returns:
        None
    """
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(cpu_flag):
    """选择设备并返回设备对象与设备名称。

    Args:
        cpu_flag (bool): 是否强制使用 CPU。

    Returns:
        tuple[torch.device, str]: 返回指定的设备对象以及设备的名称。
    """
    if cpu_flag:
        return torch.device("cpu"), "CPU"
    if torch.cuda.is_available():
        device = torch.device("cuda")
        gpu_name = torch.cuda.get_device_name(0)
        return device, gpu_name
    return torch.device("cpu"), "CPU"


def count_params(model):
    """统计模型可训练 / 总参数量。

    Args:
        model (nn.Module): 待统计数量的 PyTorch 模型。

    Returns:
        tuple[int, int]: (可训练参数数量, 总参数数量)。
    """
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return trainable, total


# ---------------------------------------------------------------------------
# 5. YOLO 风格日志格式化
# ---------------------------------------------------------------------------
def print_training_header(cfg, device_name, model):
    """输出 YOLO 风格的训练头部信息。

    Args:
        cfg (dict): 包含训练参数的字典。
        device_name (str): 设别名称。
        model (nn.Module): 用于训练的模型。

    Returns:
        None
    """
    trainable, total = count_params(model)
    data_source = "synthetic" if not cfg.get("image_dir") else cfg.get("image_dir")

    header = f"""
{"=" * 70}
{"Segmentation Training":^70}
{"=" * 70}
{"image_size":<20} {cfg["imgsz"]}
{"batch_size":<20} {cfg["batch"]}
{"epochs":<20} {cfg["epochs"]}
{"learning_rate":<20} {cfg["lr"]}
{"optimizer":<20} Adam
{"device":<20} {device_name}
{"data_source":<20} {data_source}
{"augment":<20} {cfg.get("augment", False)}
{"seed":<20} {cfg.get("seed", "None")}
{"project":<20} {cfg["project"]}/{cfg["name"]}
{"model_params":<20} {trainable:,} trainable / {total:,} total
{"=" * 70}
"""
    print(header)


def print_epoch_progress(epoch, epochs, avg_loss, avg_iou, avg_dice, elapsed):
    """输出单 epoch 训练进度（YOLO 表格风格）。

    Args:
        epoch (int): 当前回合数。
        epochs (int): 总回合数。
        avg_loss (float): 当前回合的平均损失。
        avg_iou (float): 当前回合的平均 IoU。
        avg_dice (float): 当前回合的平均 Dice。
        elapsed (float): 当前回合的执行耗时。

    Returns:
        None
    """
    eta = elapsed * (epochs - epoch)
    print(
        f"{' ' * 4}{epoch:>5}/{epochs:<5} "
        f"{avg_loss:>10.6f} "
        f"{avg_iou:>8.4f} "
        f"{avg_dice:>8.4f} "
        f"{time.strftime('%H:%M:%S', time.gmtime(elapsed)):<10} "
        f"{time.strftime('%H:%M:%S', time.gmtime(eta)):<10}"
    )


def print_training_footer(epochs, total_time, best_loss, save_dir):
    """输出训练结束汇总信息。

    Args:
        epochs (int): 实际训练完成的总轮数。
        total_time (float): 训练过程总耗时。
        best_loss (float): 训练中记录的最佳损失值。
        save_dir (str): 模型权重的保存路径。

    Returns:
        None
    """
    footer = f"""
{"=" * 70}
Training Complete  |  Total epochs: {epochs}  |  Total time: {time.strftime("%H:%M:%S", time.gmtime(total_time))}
Best loss: {best_loss:.6f}
Results saved to: {save_dir}
{"=" * 70}
"""
    print(footer)


# ---------------------------------------------------------------------------
# 6. 训练主流程
# ---------------------------------------------------------------------------
def train(cfg):
    """执行训练并返回训练结果。

    参数可由外部 dict 直接传入，也可通过 CLI / config 文件间接构造。

    Args:
        cfg (dict): 训练配置字典，结构见 DEFAULT_CFG。

    Returns:
        dict: 包含 save_dir, best_loss, epoch_losses 的结果字典。
    """
    from engine.losses import SegDetLoss, SegmentationLoss
    from engine.metrics import compute_dice, compute_iou
    from models import FPNSegNet, MiniSegNet, build_ts_dual_model
    from utils.visualize import plot_loss_curve

    # ---- 设备 ----
    device, device_name = select_device(cfg.get("cpu", False))

    # ---- 模型 ----
    model_type = cfg.get("model_type", "miniseg")
    is_ts_dual = model_type == "ts_dual"
    if is_ts_dual:
        from configs.config import TS_DUAL_MODEL_CFG

        model = build_ts_dual_model(
            TS_DUAL_MODEL_CFG, num_classes=cfg.get("num_classes", 1)
        ).to(device)
    elif model_type == "fpnseg":
        model = FPNSegNet().to(device)
    else:
        model = MiniSegNet().to(device)
    print_training_header(cfg, device_name, model)

    # ---- 数据 ----
    from datasets import MultiModalSegmentationDataset, SegmentationDataset

    if is_ts_dual:
        dataset = MultiModalSegmentationDataset(
            image_dir=cfg.get("image_dir") or None,
            label_dir=cfg.get("mask_dir") or cfg.get("label_dir") or None,
            label_type=cfg.get("label_type", "mask"),
            depth_dir=cfg.get("depth_dir") or None,
            prompt_dir=cfg.get("prompt_dir") or None,
            target_size=(cfg["imgsz"], cfg["imgsz"]),
            synthetic_length=cfg.get("synthetic_length", 32),
            augment=cfg.get("augment", False),
        )
    else:
        dataset = SegmentationDataset(
            image_dir=cfg.get("image_dir") or None,
            label_dir=cfg.get("mask_dir") or cfg.get("label_dir") or None,
            label_type=cfg.get("label_type", "mask"),
            target_size=(cfg["imgsz"], cfg["imgsz"]),
            synthetic_length=cfg.get("synthetic_length", 32),
            augment=cfg.get("augment", False),
        )
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch"],
        shuffle=True,
        num_workers=cfg.get("workers", 0),
    )

    # ---- 优化器 & 损失 ----
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    if is_ts_dual:
        criterion = SegDetLoss(
            mask_loss=cfg.get("mask_loss", "bce"),
            mask_weight=cfg.get("mask_loss_weight", 1.0),
            fourier_weight=cfg.get("fourier_loss_weight", 0.2),
            bbox_weight=cfg.get("bbox_loss_weight", 1.0),
            nwd_constant=cfg.get("nwd_constant", 20.0),
        )
    else:
        criterion = SegmentationLoss(loss_type=cfg.get("mask_loss", "bce"))

    # ---- 保存路径 ----
    save_dir = Path(cfg["project"]) / cfg["name"]
    save_dir.mkdir(parents=True, exist_ok=True)
    weight_dir = save_dir / "weights"
    weight_dir.mkdir(parents=True, exist_ok=True)

    # ---- 日志 ----
    log_file = save_dir / "logs.txt"
    _log_hyperparams(log_file, cfg, device_name, model=model)

    # ---- 打印表格头 ----
    print(
        f"\n{'Epoch':>10} {'loss':>10} {'IoU':>8} {'Dice':>8} {'elapsed':<10} {'ETA':<10}"
    )
    print(f"{'─' * 72}")

    epoch_losses = []
    best_loss = float("inf")
    t0 = time.time()

    model.train()
    for epoch in range(cfg["epochs"]):
        epoch_loss = 0.0
        epoch_iou = 0.0
        epoch_dice = 0.0
        loss_items = None
        epoch_items = {"bbox": 0.0, "mask": 0.0, "fourier": 0.0} if is_ts_dual else None
        pbar = tqdm(
            loader, desc=f"Epoch {epoch + 1}/{cfg['epochs']}", unit="batch", leave=False
        )

        for batch in pbar:
            if is_ts_dual:
                imgs, prompts, depths, masks = batch
                imgs = imgs.to(device)
                prompts = prompts.to(device)
                depths = depths.to(device)
            else:
                imgs, masks = batch
                imgs = imgs.to(device)

            masks = _normalize_mask(masks).to(device)

            if is_ts_dual:
                outputs = model(imgs, prompts, depths)
                mask_logits = outputs["mask"]
                bbox_pred = outputs["bbox"]
                bbox_targets, bbox_valid = _masks_to_boxes(masks)
                loss, loss_items = criterion(
                    mask_logits, masks, bbox_pred, bbox_targets, bbox_valid
                )
                for key in epoch_items:
                    epoch_items[key] += loss_items[key]
            else:
                mask_logits = model(imgs)
                if cfg.get("mask_loss") == "cross_entropy":
                    mask_targets = masks.squeeze(1).long()
                else:
                    mask_targets = masks
                loss = criterion(mask_logits, mask_targets)
                loss_items = None

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_loss = loss.item()
            epoch_loss += batch_loss

            mask_bin = (masks > 0.5).float()
            with torch.no_grad():
                prob = torch.sigmoid(mask_logits)
                preds = (prob >= cfg.get("mask_threshold", 0.5)).float()
                batch_iou = compute_iou(preds, mask_bin)
                batch_dice = compute_dice(preds, mask_bin)
                epoch_iou += batch_iou
                epoch_dice += batch_dice

            postfix = {
                "loss": f"{batch_loss:.4f}",
                "iou": f"{batch_iou:.4f}",
                "dice": f"{batch_dice:.4f}",
            }
            if loss_items is not None:
                postfix["bbox"] = f"{loss_items['bbox']:.3f}"
                postfix["mask"] = f"{loss_items['mask']:.3f}"
                postfix["fft"] = f"{loss_items['fourier']:.3f}"
            pbar.set_postfix(postfix)

        avg_loss = epoch_loss / len(loader)
        avg_iou = epoch_iou / len(loader)
        avg_dice = epoch_dice / len(loader)
        avg_items = None
        if epoch_items is not None:
            avg_items = {k: v / len(loader) for k, v in epoch_items.items()}
        epoch_losses.append(avg_loss)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), weight_dir / "best.pt")

        elapsed = time.time() - t0
        print_epoch_progress(epoch + 1, cfg["epochs"], avg_loss, avg_iou, avg_dice, elapsed)

        with open(log_file, "a", encoding="utf-8") as f:
            line = (
                f"{' ' * 4}{epoch + 1:>5}/{cfg['epochs']:<5}  "
                f"loss={avg_loss:.6f}  iou={avg_iou:.6f}  dice={avg_dice:.6f}"
            )
            if avg_items is not None:
                line += (
                    f"  bbox={avg_items['bbox']:.6f}  "
                    f"mask={avg_items['mask']:.6f}  fft={avg_items['fourier']:.6f}"
                )
            f.write(line + "\n")

    # ---- 保存最终权重 ----
    torch.save(model.state_dict(), weight_dir / "last.pt")

    total_time = time.time() - t0
    print_training_footer(cfg["epochs"], total_time, best_loss, save_dir)

    # ---- 损失曲线 ----
    plot_loss_curve(epoch_losses, str(save_dir / "loss_curve.png"))

    return {
        "save_dir": save_dir,
        "best_loss": best_loss,
        "epoch_losses": epoch_losses,
    }


def _normalize_mask(mask):
    """规范化掩码形状为 (N,1,H,W)。

    Args:
        mask (torch.Tensor): 输入的掩码张量。

    Returns:
        torch.Tensor: 规范化后的掩码张量，形状为 (N,1,H,W)。
    """
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2)
    return mask.float()


def _masks_to_boxes(masks: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """根据二值掩码生成归一化边界框。

    Args:
        masks (torch.Tensor): 掩码张量 (B, 1, H, W)。

    Returns:
        Tuple[torch.Tensor, torch.Tensor]:
            boxes — (B, 4) 的归一化 xyxy 边界框；
            valid — (B,) 的有效性掩码。
    """
    if masks.ndim != 4:
        raise ValueError("masks 必须为 4 维张量 (B, 1, H, W)")

    b, _, h, w = masks.shape
    boxes = torch.zeros((b, 4), device=masks.device, dtype=masks.dtype)
    valid = torch.zeros((b,), device=masks.device, dtype=torch.bool)

    for i in range(b):
        pos = masks[i, 0] > 0.5
        if pos.any():
            ys, xs = torch.where(pos)
            x1 = xs.min().float() / w
            x2 = xs.max().float() / w
            y1 = ys.min().float() / h
            y2 = ys.max().float() / h
            boxes[i] = torch.stack([x1, y1, x2, y2])
            valid[i] = True

    return boxes, valid


def _log_hyperparams(log_file, cfg, device_name, model=None):
    """记录超参数到日志文件。

    Args:
        log_file (Path): 日志文件路径。
        cfg (dict): 配置字典。
        device_name (str): 设备名称。
        model (nn.Module | None): 已构建的模型实例。

    Returns:
        None
    """
    if model is None:
        from models import FPNSegNet, MiniSegNet

        model_type = cfg.get("model_type", "miniseg")
        _m = FPNSegNet() if model_type == "fpnseg" else MiniSegNet()
    else:
        _m = model
    trainable, total = count_params(_m)

    params = {
        "image_size": cfg["imgsz"],
        "batch_size": cfg["batch"],
        "epochs": cfg["epochs"],
        "learning_rate": cfg["lr"],
        "optimizer": "Adam",
        "augment": cfg.get("augment", False),
        "device": device_name,
        "seed": cfg.get("seed", "None"),
        "trainable_params": trainable,
        "total_params": total,
        "train_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(log_file, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("  Hyperparameters\n")
        f.write("=" * 70 + "\n")
        for k, v in params.items():
            f.write(f"  {k:<24}: {v}\n")
        f.write("=" * 70 + "\n")
        f.write("\n  Training Log\n")
        f.write("─" * 70 + "\n")


# ---------------------------------------------------------------------------
# 7. CLI 入口
# ---------------------------------------------------------------------------
def main(argv=None):
    """CLI 主入口。

    Args:
        argv (list, optional): 命令行参数列表，默认 None。

    Returns:
        None

    用法:
        python train.py
        python train.py --epochs 50 --batch 16
        python train.py --cfg configs/train.json
        python train.py --cfg configs/train.yaml --imgsz 256
    """
    cli_args = parse_args(argv)
    cfg = merge_cfg(cli_args)

    if cli_args.print_cfg:
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return

    if cli_args.save_cfg:
        save_cfg(cfg, cli_args.save_cfg)

    set_seed(cfg.get("seed"))
    train(cfg)


if __name__ == "__main__":
    main()
