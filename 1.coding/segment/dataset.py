import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

"""
	默认目录结构：
	root/
	├── splits/
	│   ├── train.txt
	│   ├── val.txt
	│   └── test.txt
	├── labels/
	│   └── xxx.png
	├── images/
	│   └── xxx.png
	└── depth/
		└── xxx.png

	split 文件每行支持两种形式：
	1) sample_id
	2) image_rel_path label_rel_path
	   - 第一列作为主模态路径
	   - 第二列作为标签路径
"""


@dataclass
class ModalityConfig:
	"""
	单个模态配置。

	name: 模态名称，例如 rgb/depth/nir
	folder: 模态图像所在子目录（相对于 root）
	mode: PIL 读取模式，例如 RGB/L/I;16
	suffix: 文件后缀（含点），例如 .png/.jpg
	mean/std: 可选归一化参数；若不提供则不归一化
	"""

	name: str
	folder: str
	mode: str = "RGB"
	suffix: str = ".png"
	mean: Optional[Sequence[float]] = None
	std: Optional[Sequence[float]] = None


class SegmentationMultiModalDataset(Dataset):


	def __init__(
		self,
		root: str,
		split: str = "train",
		modalities: Optional[List[ModalityConfig]] = None,
		label_folder: str = "labels",
		label_suffix: str = ".png",
		split_folder: str = "splits",
		output_sizes: Sequence[Tuple[int, int]] = ((512, 512),),
		ignore_index: int = 255,
	):
		self.root = root
		self.split = split
		self.label_folder = label_folder
		self.label_suffix = label_suffix
		self.split_folder = split_folder
		self.output_sizes = list(output_sizes)
		self.ignore_index = ignore_index

		if modalities is None:
			modalities = [
				ModalityConfig(
					name="rgb",
					folder="images",
					mode="RGB",
					suffix=".png",
					mean=[0.485, 0.456, 0.406],
					std=[0.229, 0.224, 0.225],
				)
			]
		self.modalities = modalities

		split_path = os.path.join(self.root, self.split_folder, f"{self.split}.txt")
		if not os.path.isfile(split_path):
			raise FileNotFoundError(f"未找到划分文件: {split_path}")

		self.samples = self._build_samples(split_path)
		if len(self.samples) == 0:
			raise RuntimeError(f"{self.split} 集样本为空，请检查 {split_path}")

		self._to_tensor = transforms.ToTensor()

	def _build_samples(self, split_path: str) -> List[Dict[str, str]]:
		samples = []
		with open(split_path, "r", encoding="utf-8") as f:
			lines = [line.strip() for line in f if line.strip()]

		for line in lines:
			parts = line.split()
			if len(parts) == 1:
				sample_id = parts[0]
				entry = {
					"id": sample_id,
					"label_path": os.path.join(
						self.root, self.label_folder, sample_id + self.label_suffix
					),
				}
				for mod in self.modalities:
					entry[f"mod_{mod.name}"] = os.path.join(
						self.root, mod.folder, sample_id + mod.suffix
					)
			else:
				# 指定图像与标签相对路径；其他模态按同名ID自动拼接
				image_rel, label_rel = parts[0], parts[1]
				image_abs = os.path.join(self.root, image_rel)
				label_abs = os.path.join(self.root, label_rel)

				sample_id = os.path.splitext(os.path.basename(image_abs))[0]
				entry = {"id": sample_id, "label_path": label_abs}

				for mod in self.modalities:
					if mod == self.modalities[0]:
						entry[f"mod_{mod.name}"] = image_abs
					else:
						entry[f"mod_{mod.name}"] = os.path.join(
							self.root, mod.folder, sample_id + mod.suffix
						)

			if not os.path.isfile(entry["label_path"]):
				continue

			all_modal_exists = True
			for mod in self.modalities:
				if not os.path.isfile(entry[f"mod_{mod.name}"]):
					all_modal_exists = False
					break
			if all_modal_exists:
				samples.append(entry)

		return samples

	def __len__(self) -> int:
		return len(self.samples)

	def _read_modality(self, image_path: str, mod_cfg: ModalityConfig) -> torch.Tensor:
		img = Image.open(image_path)
		img = img.convert(mod_cfg.mode)

		if mod_cfg.mode in ("I;16", "I"):
			# 16位深度等数据使用 numpy 保留数值范围，再归一化到 [0, 1]
			arr = np.array(img, dtype=np.float32)
			if arr.max() > 0:
				arr = arr / arr.max()
			tensor = torch.from_numpy(arr).unsqueeze(0)
		else:
			tensor = self._to_tensor(img)

		if mod_cfg.mean is not None and mod_cfg.std is not None:
			mean = torch.tensor(mod_cfg.mean, dtype=tensor.dtype).view(-1, 1, 1)
			std = torch.tensor(mod_cfg.std, dtype=tensor.dtype).view(-1, 1, 1)
			tensor = (tensor - mean) / std

		return tensor

	def _resize_tensor(self, tensor: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
		resized = torch.nn.functional.interpolate(
			tensor.unsqueeze(0),
			size=size_hw,
			mode="bilinear",
			align_corners=False,
		)
		return resized.squeeze(0)

	def _resize_mask(self, mask: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
		resized = torch.nn.functional.interpolate(
			mask.unsqueeze(0).unsqueeze(0).float(),
			size=size_hw,
			mode="nearest",
		)
		return resized.squeeze(0).squeeze(0).long()

	def __getitem__(self, idx: int) -> Dict[str, object]:
		sample = self.samples[idx]

		modality_tensors = []
		for mod in self.modalities:
			t = self._read_modality(sample[f"mod_{mod.name}"], mod)
			modality_tensors.append(t)

		# 在通道维拼接多模态数据
		merged = torch.cat(modality_tensors, dim=0)

		mask = Image.open(sample["label_path"])
		mask = torch.from_numpy(np.array(mask, dtype=np.int64))
		mask[mask < 0] = self.ignore_index

		images_by_scale: Dict[str, torch.Tensor] = {}
		masks_by_scale: Dict[str, torch.Tensor] = {}

		for h, w in self.output_sizes:
			key = f"{h}x{w}"
			images_by_scale[key] = self._resize_tensor(merged, (h, w))
			masks_by_scale[key] = self._resize_mask(mask, (h, w))

		return {
			"id": sample["id"],
			"images": images_by_scale,
			"masks": masks_by_scale,
		}


def build_segmentation_dataloaders(
	root: str,
	modalities: Optional[List[ModalityConfig]] = None,
	output_sizes: Sequence[Tuple[int, int]] = ((512, 512),),
	batch_size: int = 4,
	num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
	train_set = SegmentationMultiModalDataset(
		root=root,
		split="train",
		modalities=modalities,
		output_sizes=output_sizes,
	)
	val_set = SegmentationMultiModalDataset(
		root=root,
		split="val",
		modalities=modalities,
		output_sizes=output_sizes,
	)
	test_set = SegmentationMultiModalDataset(
		root=root,
		split="test",
		modalities=modalities,
		output_sizes=output_sizes,
	)

	train_loader = DataLoader(
		train_set,
		batch_size=batch_size,
		shuffle=True,
		num_workers=num_workers,    
		pin_memory=True,
	)
	val_loader = DataLoader(
		val_set,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)
	test_loader = DataLoader(
		test_set,
		batch_size=batch_size,
		shuffle=False,
		num_workers=num_workers,
		pin_memory=True,
	)

	return train_loader, val_loader, test_loader


if __name__ == "__main__":
	# 示例：RGB + 深度（L）+ NIR（L）
	custom_modalities = [
		ModalityConfig(
			name="rgb",
			folder="images",
			mode="RGB",
			suffix=".png",
			mean=[0.485, 0.456, 0.406],
			std=[0.229, 0.224, 0.225],
		),
		ModalityConfig(name="depth", folder="depth", mode="L", suffix=".png"),
		ModalityConfig(name="nir", folder="nir", mode="L", suffix=".png"),
	]

	root_dir = r"E:\mastercode\1.coding\0_segment\your_dataset_root"
	# 仅用于演示，实际使用时替换为真实路径
	if os.path.isdir(root_dir):
		train_loader, val_loader, test_loader = build_segmentation_dataloaders(
			root=root_dir,
			modalities=custom_modalities,
			output_sizes=[(256, 256), (512, 512)],
			batch_size=2,
			num_workers=0,
		)

		batch = next(iter(train_loader))
		print("sample ids:", batch["id"])
		print("scales:", list(batch["images"].keys()))
		print("512x512 image shape:", batch["images"]["512x512"].shape)
		print("512x512 mask shape:", batch["masks"]["512x512"].shape)
