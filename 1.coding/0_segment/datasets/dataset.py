"""分割数据集加载模块。

提供对于不同格式标签的分割数据集的数据获取及处理。
"""

import os
from pathlib import Path
from typing import Tuple, Optional, List

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .transforms import image_transform, TXT2MASK, JSON2MASK, NPY2MASK


class SegmentationDataset(Dataset):
    """RGB 图像分割数据集，支持多种标签格式与合成数据回退。

    负责加载图像及对应的分割标签，并执行相应的数据增强。

    Attributes:
        image_dir (str | Path | None): 图像目录路径。
        label_dir (str | None): 标签目录或标签文件路径。
        label_type (str): 标签类型（mask/txt/json/npy）。
        target_size (Tuple[int, int]): 目标尺寸 (W, H)。
        file_list (List[str] | None): 指定样本列表（不含后缀）。
        synthetic_length (int): 合成数据长度。
        augment (bool): 是否启用增强。
    """

    def __init__(
        self,
        image_dir=None,
        label_dir=None,
        label_type="mask",
        target_size=(640, 640),
        file_list: Optional[List[str]] = None,
        synthetic_length: int = 32,
        augment: bool = False,
    ):
        """初始化 RGB 图像分割数据集。"""
        self.image_dir = Path(image_dir) if image_dir else None
        self.label_dir = label_dir
        self.label_type = label_type
        self.target_size = target_size
        self.augment = augment

        if image_dir and label_dir and os.path.isdir(image_dir) and os.path.isdir(label_dir):
            if file_list is None:
                self.ids = sorted([
                    p.stem for p in self.image_dir.glob("*.*")
                    if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'}
                ])
            else:
                self.ids = file_list
            self.synthetic = False
        else:
            self.synthetic = True
            self.synthetic_length = synthetic_length

    def __len__(self):
        """返回数据集长度。

        Args:
            无

        Returns:
            int: 样本数量。
        """
        return self.synthetic_length if self.synthetic else len(self.ids)

    def __getitem__(self, idx):
        """根据索引读取样本。

        Args:
            idx (int): 样本索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: 图像张量与掩码张量。

        Raises:
            FileNotFoundError: 图像文件不存在时抛出。
            IOError: 图像读取失败时抛出。
        """
        if self.synthetic:
            h, w = self.target_size[1], self.target_size[0]
            image = torch.rand(3, h, w)
            mask = (torch.rand(1, h, w) > 0.5).float()
            return image, mask

        stem = self.ids[idx]
        img_path = self._find_image(stem)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {stem}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Cannot read {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        label = self._get_mask(stem)
        label = cv2.resize(label, self.target_size, interpolation=cv2.INTER_NEAREST) if label.shape[:2] != (self.target_size[1], self.target_size[0]) else label

        if self.augment:
            img_rgb, label = self._apply_augmentation(img_rgb, label)

        img_rgb = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)
        img_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        mask_tensor = torch.from_numpy((label > 127).astype(np.float32)).unsqueeze(0) if label.ndim == 2 else torch.from_numpy(label).float()

        return img_tensor, mask_tensor

    def _find_image(self, stem):
        """根据文件名主体查找图像文件。

        Args:
            stem (str): 文件名主体（不含后缀）。

        Returns:
            Path | None: 命中的图像路径，未找到则返回 None。
        """
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            candidate = self.image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _get_mask(self, image_stem):
        """根据标签类型读取并生成掩码。

        Args:
            image_stem (str): 图像文件名主体。

        Returns:
            np.ndarray: 掩码数组。
        """
        if self.label_type == "txt":
            return TXT2MASK(self.label_dir, image_stem, self.target_size)
        elif self.label_type == "json":
            return JSON2MASK(self.label_dir, image_stem, self.target_size)
        elif self.label_type == "npy":
            return NPY2MASK(self.label_dir, image_stem, self.target_size)
        else:
            return self._read_mask_file(image_stem)

    def _read_mask_file(self, stem):
        """从常见掩码文件中读取标签。

        Args:
            stem (str): 文件名主体。

        Returns:
            np.ndarray: 掩码数组。
        """
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy']:
            candidate = os.path.join(self.label_dir, stem + ext)
            if os.path.exists(candidate):
                if candidate.lower().endswith('.npy'):
                    return np.load(candidate)
                label = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                return label if label is not None else np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        return np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)

    def _apply_augmentation(self, image, mask):
        """对图像与掩码应用简单的翻转增强。

        Args:
            image (np.ndarray): 输入图像。
            mask (np.ndarray): 输入掩码。

        Returns:
            Tuple[np.ndarray, np.ndarray]: 增强后的图像与掩码。
        """
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if np.random.random() > 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask


# 保持向后兼容的别名
get_dataset_rgb = SegmentationDataset
