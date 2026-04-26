import argparse
import os
from typing import Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from models.blocks import Basic_Conv_Block
from models.builders import make_layers
from configs import RESNET_18_CFG
import torch.nn.functional as F

class SyntheticSegDataset(Dataset):
    '''
    用于训练链路自检的合成分割数据集。

    Args:
        length (int): 数据集样本数量。
        image_size (Tuple[int, int]): 图像尺寸, 格式为 (H, W)。
    '''

    def __init__(self, length: int = 32, image_size: Tuple[int, int] = (128, 128)):
        '''
        初始化合成数据集。

        Args:
            length (int): 数据集样本数量。
            image_size (Tuple[int, int]): 图像尺寸, 格式为 (H, W)。
        '''
        self.length = length
        self.image_size = image_size

    def __len__(self) -> int:
        '''
        返回数据集长度。

        Returns:
            int: 样本数量。
        '''
        return self.length

    def __getitem__(self, index: int):
        '''
        获取单个样本。

        Args:
            index (int): 样本索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - image: 形状为 (3, H, W) 的随机图像。
                - mask: 形状为 (1, H, W) 的随机二值掩码。
        '''
        h, w = self.image_size
        image = torch.rand(3, h, w)
        mask = (torch.rand(1, h, w) > 0.5).float()
        return image, mask


class MiniSegNet(nn.Module):
    '''
    最小化分割网络, 用于训练流程验证。

    Notes:
        网络结构为两层 Basic_Conv_Block 加一层 1x1 卷积输出。
    '''

    def __init__(self):
        '''
        初始化最小分割网络结构。
        '''
        super().__init__()
        self.backbone = make_layers(RESNET_18_CFG)
        self.head = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量, 形状通常为 (N, 3, H, W)。

        Returns:
            torch.Tensor: 输出 logits, 形状通常为 (N, 1, H, W)。
        '''
        feat = self.backbone(x)
        logits = self.head(feat)
        logits = F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
        return logits


def build_dataloader(args) -> DataLoader:
    '''
    根据参数构建 DataLoader。

    Args:
        args (argparse.Namespace): 运行参数。

    Returns:
        DataLoader: 可用于训练或评估的数据加载器。

    Notes:
        当提供真实数据目录且加载成功时返回真实数据加载器,
        否则自动回退到合成数据集。
    '''
    if args.image_dir and args.label_dir and os.path.isdir(args.image_dir) and os.path.isdir(args.label_dir):
        try:
            from data.datasets import get_dataset_rgb

            dataset = get_dataset_rgb(
                image_dir=args.image_dir,
                label_dir=args.label_dir,
                label_type=args.label_type,
                target_size=(args.image_size, args.image_size),
            )
            return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
        except Exception as err:
            print(f"Failed to build real dataset ({err}), fallback to synthetic data.")

    dataset = SyntheticSegDataset(length=args.synthetic_length, image_size=(args.image_size, args.image_size))
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)


def normalize_mask(mask: torch.Tensor) -> torch.Tensor:
    '''
    统一标签维度到 (B, 1, H, W)。

    Args:
        mask (torch.Tensor): 原始标签张量。

    Returns:
        torch.Tensor: 处理后的 float 标签张量。
    '''
    # Support both [B, 1, H, W] and [B, H, W, 1] labels.
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    elif mask.ndim == 4 and mask.shape[-1] == 1:
        mask = mask.permute(0, 3, 1, 2)
    return mask.float()


def train(args):
    '''
    执行最小化训练流程并保存 checkpoint。

    Args:
        args (argparse.Namespace): 训练参数。

    Returns:
        None
    '''
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    dataloader = build_dataloader(args)

    model = MiniSegNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for image, mask in dataloader:
            image = image.to(device).float()
            mask = normalize_mask(mask).to(device)

            pred = model(image)
            loss = criterion(pred, mask)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / max(len(dataloader), 1)
        print(f"Epoch [{epoch + 1}/{args.epochs}] loss={avg_loss:.6f}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, args.checkpoint_name)
    torch.save({"model": model.state_dict(), "args": vars(args)}, ckpt_path)
    print(f"Checkpoint saved to: {ckpt_path}")


def parse_args():
    '''
    解析命令行参数。

    Returns:
        argparse.Namespace: 解析后的参数对象。
    '''
    parser = argparse.ArgumentParser(description="Minimal training entry for 0_segment.")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--image-dir", type=str, default="")
    parser.add_argument("--label-dir", type=str, default="")
    parser.add_argument("--label-type", type=str, default="mask", choices=["mask", "txt", "json", "npy"])

    parser.add_argument("--synthetic-length", type=int, default=32)
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    parser.add_argument("--checkpoint-name", type=str, default="minimal_last.pt")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
