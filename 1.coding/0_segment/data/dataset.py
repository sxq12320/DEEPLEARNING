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
    """RGB 图像分割数据集，支持多种标签格式和合成数据回退。"""

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
        return self.synthetic_length if self.synthetic else len(self.ids)

    def __getitem__(self, idx):
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
        for ext in ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']:
            candidate = self.image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _get_mask(self, image_stem):
        if self.label_type == "txt":
            return TXT2MASK(self.label_dir, image_stem, self.target_size)
        elif self.label_type == "json":
            return JSON2MASK(self.label_dir, image_stem, self.target_size)
        elif self.label_type == "npy":
            return NPY2MASK(self.label_dir, image_stem, self.target_size)
        else:
            return self._read_mask_file(image_stem)

    def _read_mask_file(self, stem):
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy']:
            candidate = os.path.join(self.label_dir, stem + ext)
            if os.path.exists(candidate):
                if candidate.lower().endswith('.npy'):
                    return np.load(candidate)
                label = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                return label if label is not None else np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        return np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)

    def _apply_augmentation(self, image, mask):
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
        if np.random.random() > 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
        return image, mask


# 保持向后兼容的别名
get_dataset_rgb = SegmentationDataset
