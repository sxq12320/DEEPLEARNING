"""多模态分割数据集。

支持 RGB + Mask 先验 + Depth 的联合输入，并兼容多种标签格式。
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from .transforms import JSON2MASK, NPY2MASK, TXT2MASK


class MultiModalSegmentationDataset(Dataset):
    """多模态分割数据集。

    Args:
        image_dir (str | Path | None): RGB 图像目录。
        label_dir (str | Path | None): 标签目录或 JSON 文件。
        label_type (str): 标签类型（mask/txt/json/npy）。
        depth_dir (str | Path | None): Depth 图像目录。
        prompt_dir (str | Path | None): Mask 先验提示目录。
        target_size (Tuple[int, int]): 目标尺寸 (W, H)。
        file_list (List[str] | None): 指定样本列表。
        synthetic_length (int): 合成数据长度。
        augment (bool): 是否启用增强。
    """

    def __init__(
        self,
        image_dir: Union[str, Path, None] = None,
        label_dir: Union[str, Path, None] = None,
        label_type: str = "mask",
        depth_dir: Union[str, Path, None] = None,
        prompt_dir: Union[str, Path, None] = None,
        target_size: Tuple[int, int] = (640, 640),
        file_list: Optional[List[str]] = None,
        synthetic_length: int = 32,
        augment: bool = False,
    ):
        """初始化多模态分割数据集。"""
        self.image_dir = Path(image_dir) if image_dir else None
        self.label_dir = str(label_dir) if label_dir else None
        self.depth_dir = Path(depth_dir) if depth_dir else None
        self.prompt_dir = Path(prompt_dir) if prompt_dir else None
        self.label_type = label_type
        self.target_size = target_size
        self.augment = augment

        image_ok = self.image_dir is not None and self.image_dir.is_dir()
        label_ok = bool(self.label_dir) and Path(self.label_dir).exists()

        if image_ok and label_ok:
            if file_list is None:
                self.ids = sorted(
                    [
                        p.stem
                        for p in self.image_dir.glob("*.*")
                        if p.suffix.lower()
                        in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
                    ]
                )
            else:
                self.ids = file_list
            self.synthetic = False
        else:
            self.synthetic = True
            self.synthetic_length = synthetic_length

    def __len__(self) -> int:
        """返回数据集长度。

        Returns:
            int: 数据集样本数。
        """
        return self.synthetic_length if self.synthetic else len(self.ids)

    def __getitem__(self, idx: int):
        """根据索引读取样本。

        Args:
            idx (int): 样本索引。

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                RGB、Prompt、Depth 与 Mask 张量。
        """
        if self.synthetic:
            h, w = self.target_size[1], self.target_size[0]
            rgb = torch.rand(3, h, w)
            prompt = (torch.rand(1, h, w) > 0.5).float()
            depth = torch.rand(1, h, w)
            mask = (torch.rand(1, h, w) > 0.5).float()
            return rgb, prompt, depth, mask

        stem = self.ids[idx]
        img_path = self._find_image(self.image_dir, stem)
        if img_path is None:
            raise FileNotFoundError(f"Image not found for {stem}")

        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            raise IOError(f"Cannot read {img_path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        mask = self._get_mask(stem)
        depth = self._read_gray(self.depth_dir, stem)
        prompt = self._read_gray(self.prompt_dir, stem)

        mask = self._resize_if_needed(mask)
        depth = self._resize_if_needed(depth) if depth is not None else None
        prompt = self._resize_if_needed(prompt) if prompt is not None else None

        if self.augment:
            img_rgb, mask, depth, prompt = self._apply_augmentation(
                img_rgb, mask, depth, prompt
            )

        img_rgb = cv2.resize(img_rgb, self.target_size, interpolation=cv2.INTER_LINEAR)

        rgb_tensor = torch.from_numpy(img_rgb).permute(2, 0, 1).float() / 255.0
        prompt_tensor = self._to_single_channel_tensor(prompt, img_rgb.shape[:2])
        depth_tensor = self._to_single_channel_tensor(depth, img_rgb.shape[:2])
        mask_tensor = self._to_mask_tensor(mask)

        return rgb_tensor, prompt_tensor, depth_tensor, mask_tensor

    def _find_image(self, image_dir: Optional[Path], stem: str) -> Optional[Path]:
        """查找图像文件。

        Args:
            image_dir (Path | None): 图像目录。
            stem (str): 文件名主体。

        Returns:
            Path | None: 命中的图像路径。
        """
        if image_dir is None:
            return None
        for ext in [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]:
            candidate = image_dir / f"{stem}{ext}"
            if candidate.exists():
                return candidate
        return None

    def _get_mask(self, image_stem: str) -> np.ndarray:
        """读取并生成掩码。

        Args:
            image_stem (str): 图像文件名主体。

        Returns:
            np.ndarray: 掩码数组。
        """
        if self.label_type == "txt":
            return TXT2MASK(self.label_dir, image_stem, self.target_size)
        if self.label_type == "json":
            return JSON2MASK(self.label_dir, image_stem, self.target_size)
        if self.label_type == "npy":
            return NPY2MASK(self.label_dir, image_stem, self.target_size)
        return self._read_mask_file(image_stem)

    def _read_mask_file(self, stem: str) -> np.ndarray:
        """读取常见掩码文件。

        Args:
            stem (str): 文件名主体。

        Returns:
            np.ndarray: 掩码数组。
        """
        if self.label_dir is None:
            return np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"]:
            candidate = os.path.join(self.label_dir, stem + ext)
            if os.path.exists(candidate):
                if candidate.lower().endswith(".npy"):
                    return np.load(candidate)
                label = cv2.imread(candidate, cv2.IMREAD_GRAYSCALE)
                return (
                    label
                    if label is not None
                    else np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)
                )
        return np.zeros((self.target_size[1], self.target_size[0]), dtype=np.uint8)

    def _read_gray(self, data_dir: Optional[Path], stem: str) -> Optional[np.ndarray]:
        """读取单通道图像或 NPY 数据。

        Args:
            data_dir (Path | None): 数据目录。
            stem (str): 文件名主体。

        Returns:
            np.ndarray | None: 读取到的数组或 None。
        """
        if data_dir is None:
            return None
        for ext in [".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".npy"]:
            candidate = data_dir / f"{stem}{ext}"
            if candidate.exists():
                if candidate.suffix.lower() == ".npy":
                    data = np.load(candidate)
                    if data.ndim == 3:
                        data = data[..., 0]
                    return data
                return cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)
        return None

    def _resize_if_needed(self, data: np.ndarray) -> np.ndarray:
        """按需缩放到目标尺寸。

        Args:
            data (np.ndarray): 输入数组。

        Returns:
            np.ndarray: 缩放后的数组。
        """
        if data.shape[:2] == (self.target_size[1], self.target_size[0]):
            return data
        return cv2.resize(data, self.target_size, interpolation=cv2.INTER_NEAREST)

    def _apply_augmentation(
        self,
        image: np.ndarray,
        mask: np.ndarray,
        depth: Optional[np.ndarray],
        prompt: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """应用简单翻转增强。

        Args:
            image (np.ndarray): RGB 图像。
            mask (np.ndarray): 掩码。
            depth (np.ndarray | None): 深度图。
            prompt (np.ndarray | None): 先验提示。

        Returns:
            tuple: 增强后的 (image, mask, depth, prompt)。
        """
        if np.random.random() > 0.5:
            image = cv2.flip(image, 1)
            mask = cv2.flip(mask, 1)
            depth = cv2.flip(depth, 1) if depth is not None else None
            prompt = cv2.flip(prompt, 1) if prompt is not None else None
        if np.random.random() > 0.2:
            image = cv2.flip(image, 0)
            mask = cv2.flip(mask, 0)
            depth = cv2.flip(depth, 0) if depth is not None else None
            prompt = cv2.flip(prompt, 0) if prompt is not None else None
        return image, mask, depth, prompt

    def _to_single_channel_tensor(
        self, data: Optional[np.ndarray], hw: Tuple[int, int]
    ) -> torch.Tensor:
        """将单通道数据转为 Tensor。

        Args:
            data (np.ndarray | None): 输入数组。
            hw (tuple[int, int]): 参考尺寸 (H, W)。

        Returns:
            torch.Tensor: 单通道张量 (1, H, W)。
        """
        h, w = hw
        if data is None:
            return torch.zeros(1, h, w)
        if data.ndim == 3:
            data = data[..., 0]
        tensor = torch.from_numpy(data).unsqueeze(0).float()
        if tensor.max() > 1.0:
            tensor = tensor / 255.0
        return tensor

    def _to_mask_tensor(self, mask: np.ndarray) -> torch.Tensor:
        """将掩码转为二值张量。

        Args:
            mask (np.ndarray): 输入掩码。

        Returns:
            torch.Tensor: 二值掩码张量 (1, H, W)。
        """
        if mask.ndim == 3:
            mask = mask[..., 0]
        mask_bin = (mask > 127).astype(np.float32)
        return torch.from_numpy(mask_bin).unsqueeze(0)
