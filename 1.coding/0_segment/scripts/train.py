"""分割模型训练 CLI 入口。"""
import argparse
from ast import Name
from email.mime import image
import sys
from pathlib import Path

from numpy import argsort, imag

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from engine.trainer import Trainer


def parse_args():
    """解析训练命令行参数。
    
    Returns:
        argparse.Namespace: 解析后的参数对象。
    """
    parser = argparse.ArgumentParser(description="Segmentation model training")
    parser.add_argument("--image-dir", type=str, default="", help="Training image directory")
    parser.add_argument("--mask-dir", type=str, default="", help="Training mask directory")
    parser.add_argument("--label-dir", type=str, default="", help="Training label directory (alias)")
    parser.add_argument("--label-type", type=str, default="mask", choices=["mask", "txt", "json", "npy"])
    parser.add_argument("--imgsz", type=int, default=128, help="Image size (square)")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--synthetic-length", type=int, default=32)
    parser.add_argument("--augment", action="store_true", default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--project", type=str, default="runs/train")
    parser.add_argument("--name", type=str, default="exp")
    return parser.parse_args()


if __name__ == "__main__":
    from argparse import Namespace
    args = Namespace(
        imgsz = 640,
        lr = 1e-3,
        epochs = 10,
        batch = 4,
        cpu = False,
    )
    trainer = Trainer(args)
    trainer.train()