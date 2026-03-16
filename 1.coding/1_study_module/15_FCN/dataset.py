"""
PASCAL VOC 2007 语义分割数据集加载器
适配目录结构：
    VOC2007/
    ├── JPEGImages/          原始图像
    ├── SegmentationClass/   语义分割掩码（像素值=类别ID，255=忽略）
    └── ImageSets/
        └── Segmentation/
            ├── train.txt    训练集文件名列表
            └── val.txt      验证集文件名列表
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

VOC_CLASSES = [
    'background', 'aeroplane', 'bicycle', 'bird', 'boat',
    'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
NUM_CLASSES = 21  # 20类目标 + 1类背景

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]


class VOCSegDataset(Dataset):
    def __init__(self, voc_root, split='train', img_size=512, augment=False):
        """
        voc_root: VOC2007 根目录（含 JPEGImages、SegmentationClass 的那层）
        split:    'train' 或 'val'
        """
        self.img_size = img_size
        self.augment  = augment

        img_dir    = os.path.join(voc_root, 'JPEGImages')
        mask_dir   = os.path.join(voc_root, 'SegmentationClass')
        split_file = os.path.join(voc_root, 'ImageSets', 'Segmentation', f'{split}.txt')

        if not os.path.exists(split_file):
            raise FileNotFoundError(
                f"找不到划分文件：{split_file}\n"
                f"请确认路径是否正确，文件应在 ImageSets/Segmentation/ 下"
            )

        with open(split_file) as f:
            names = [line.strip() for line in f if line.strip()]

        self.samples = []
        for name in names:
            img_path  = os.path.join(img_dir,  name + '.jpg')
            mask_path = os.path.join(mask_dir, name + '.png')
            if os.path.exists(img_path) and os.path.exists(mask_path):
                self.samples.append((img_path, mask_path))
            else:
                print(f"  警告：找不到文件，跳过 {name}")

        print(f"VOC2007 {split} 集：共 {len(self.samples)} 个样本")

        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path = self.samples[idx]

        img  = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path)

        img  = img.resize( (self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            img, mask = self._augment(img, mask)

        img  = self.img_transform(img)
        mask = torch.from_numpy(np.array(mask, dtype=np.int64))

        # VOC掩码中255=边界，训练时忽略
        return img, mask

    def _augment(self, img, mask):
        if random.random() > 0.5:
            img  = img.transpose(Image.FLIP_LEFT_RIGHT)
            mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return img, mask


def build_dataloaders(voc_root, img_size=512, batch_size=4, num_workers=4):
    train_dataset = VOCSegDataset(voc_root, split='train',
                                  img_size=img_size, augment=True)
    val_dataset   = VOCSegDataset(voc_root, split='val',
                                  img_size=img_size, augment=False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              shuffle=True,  num_workers=num_workers,
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=1,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader


if __name__ == '__main__':
    # 改成你自己的路径验证
    voc_root = r'E:\mastercode\data\VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007'
    train_loader, val_loader = build_dataloaders(voc_root, img_size=512, batch_size=2)
    imgs, masks = next(iter(train_loader))
    print(f"图像 shape: {imgs.shape}")
    print(f"掩码 shape: {masks.shape}")
    print(f"掩码中的类别ID: {masks.unique().tolist()}")
