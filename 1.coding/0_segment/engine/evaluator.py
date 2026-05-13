import os

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import SegmentationDataset
from models import MiniSegNet

from .metrics import compute_iou


class Evaluator:
    """分割模型评估器。"""

    def __init__(self, args):
        """初始化评估器。

        Args:
            args (argparse.Namespace): 评估参数配置。
        """
        self.args = args
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
        )

    def build_dataloader(self):
        """构建评估数据加载器。

        Returns:
            torch.utils.data.DataLoader: 评估数据加载器。
        """
        dataset = SegmentationDataset(
            image_dir=self.args.image_dir or None,
            label_dir=self.args.mask_dir or self.args.label_dir or None,
            label_type=getattr(self.args, "label_type", "mask"),
            target_size=(self.args.imgsz, self.args.imgsz),
            synthetic_length=getattr(self.args, "synthetic_length", 32),
        )
        return DataLoader(
            dataset, batch_size=self.args.batch, shuffle=False, num_workers=0
        )

    @torch.no_grad()
    def evaluate(self):
        """执行评估并返回平均损失与 IoU。

        Returns:
            Tuple[float, float]: 平均损失与平均 IoU。

        Raises:
            FileNotFoundError: 权重文件不存在时抛出。
        """
        if not os.path.isfile(self.args.weights):
            raise FileNotFoundError(f"Checkpoint not found: {self.args.weights}")

        loader = self.build_dataloader()

        checkpoint = torch.load(self.args.weights, map_location=self.device)
        model_state = (
            checkpoint["model"]
            if isinstance(checkpoint, dict) and "model" in checkpoint
            else checkpoint
        )

        model = MiniSegNet().to(self.device)
        model.load_state_dict(model_state, strict=True)
        model.eval()

        criterion = nn.BCEWithLogitsLoss()

        total_loss = 0.0
        total_iou = 0.0
        total_batches = 0

        for image, mask in loader:
            image = image.to(self.device).float()
            mask = self._normalize_mask(mask).to(self.device)
            mask_bin = (mask > 0).float()

            logits = model(image)
            loss = criterion(logits, mask_bin)

            prob = torch.sigmoid(logits)
            pred = (prob >= self.args.threshold).float()

            batch_iou = compute_iou(pred, mask_bin)
            total_loss += loss.item()
            total_iou += batch_iou
            total_batches += 1

        avg_loss = total_loss / max(total_batches, 1)
        avg_iou = total_iou / max(total_batches, 1)

        print(
            f"Evaluate done | loss={avg_loss:.6f} | iou={avg_iou:.6f} | batches={total_batches}"
        )
        return avg_loss, avg_iou

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
