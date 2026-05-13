import os
from datetime import datetime
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from datasets import SegmentationDataset
from models import MiniSegNet
from .losses import SegmentationLoss
class Trainer:
    """分割模型训练器，支持真实数据和合成数据自动切换。"""

    def __init__(self, args):
        """初始化训练器。

        Args:
            args (argparse.Namespace): 训练参数配置。
        """
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
        self.save_dir = Path(args.project) / args.name
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def build_dataloader(self):
        """构建训练数据加载器。

        Returns:
            torch.utils.data.DataLoader: 训练数据加载器。
        """
        dataset = SegmentationDataset(
            image_dir=self.args.image_dir or None,
            label_dir=self.args.mask_dir or self.args.label_dir or None,
            label_type=getattr(self.args, 'label_type', 'mask'),
            target_size=(self.args.imgsz, self.args.imgsz),
            synthetic_length=getattr(self.args, 'synthetic_length', 32),
            augment=getattr(self.args, 'augment', False),
        )
        return DataLoader(dataset, batch_size=self.args.batch, shuffle=True, num_workers=getattr(self.args, 'workers', 0))

    def train(self):
        """执行训练流程并保存模型与日志。

        Returns:
            None: 本函数无显式返回值。
        """
        loader = self.build_dataloader()
        print(f"Device: {self.device}")
        print(f"Dataset size: {len(loader.dataset)}, Batches: {len(loader)}")

        model = MiniSegNet().to(self.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)
        criterion = SegmentationLoss()

        log_file = self.save_dir / "logs.txt"
        self._log_hyperparams(log_file)

        epoch_losses = []
        model.train()
        for epoch in range(self.args.epochs):
            epoch_loss = 0.0
            pbar = tqdm(loader, desc=f"Epoch {epoch + 1}/{self.args.epochs}", unit="batch")
            for imgs, masks in pbar:
                imgs = imgs.to(self.device)
                masks = self._normalize_mask(masks).to(self.device)

                preds = model(imgs)
                loss = criterion(preds, masks)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = epoch_loss / len(loader)
            epoch_losses.append(avg_loss)
            print(f"Epoch {epoch + 1}/{self.args.epochs} - Loss: {avg_loss:.6f}")

            with open(log_file, 'a', encoding='utf-8') as f:
                f.write(f"Epoch {epoch + 1}/{self.args.epochs} - Average Loss: {avg_loss:.6f}\n")

        self._save_model(model)
        from utils.visualize import plot_loss_curve
        plot_loss_curve(epoch_losses, str(self.save_dir / "loss_curve.png"))
        print(f"Training log saved to: {log_file}")

    def _normalize_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """将掩码转换为 (N, 1, H, W) 形状。

        Args:
            mask (torch.Tensor): 输入掩码张量。

        Returns:
            torch.Tensor: 规范化后的掩码张量。
        """
        if mask.ndim == 3:
            mask = mask.unsqueeze(1)
        elif mask.ndim == 4 and mask.shape[-1] == 1:
            mask = mask.permute(0, 3, 1, 2)
        return mask.float()

    def _save_model(self, model):
        """保存模型权重。

        Args:
            model (torch.nn.Module): 待保存的模型。
        """
        weight_path = self.save_dir / "weights" / "best.pt"
        weight_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), weight_path)
        print(f"Model saved to: {weight_path}")

    def _log_hyperparams(self, log_file):
        """记录超参数与训练日志头部。

        Args:
            log_file (Path): 日志文件路径。
        """
        params = {
            "image_size": self.args.imgsz,
            "batch_size": self.args.batch,
            "epochs": self.args.epochs,
            "learning_rate": self.args.lr,
            "optimizer": "Adam",
            "augment": getattr(self.args, 'augment', False),
            "device": str(self.device),
            "train_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("====== Hyperparameters ======\n")
            for k, v in params.items():
                f.write(f"{k}: {v}\n")
            f.write("\n====== Training Log ======\n")
