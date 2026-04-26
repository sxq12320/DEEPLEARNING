import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset ,DataLoader
import os 
import numpy as np
import cv2
from utils.image_transform import image_transform
from utils.enhance_image import enhance_image
from torchvision import transforms
from utils import TXT2MASK, JSON2MASK, NPY2MASK


class get_dataset_rgb(Dataset):
    '''
    using : 读取三通道图片,也就是单单RGB的图片
    __len__ : 返回当前文件之中图片的数量
    __getitem__: 返回当前文件之中的图片转变为固定大小后的tensor图片

    Args:
        image_dir : 图片地址
        label_dir : 标签地址
    Returns:
    '''
    def __init__(
        self,
        image_dir,
        label_dir,
        label_type="mask",
        target_size=(640, 640),
        ):
        super().__init__()
        self.image_dir = image_dir
        self.label_dir = label_dir
        self.label_type = label_type
        self.target_size = target_size

        self.images_name = sorted([
            name for name in os.listdir(self.image_dir)
            if name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff'))
        ])

    def get_mask(self, image_name):
        """
        根据 label_type 调用对应的转换函数读取标签
        """
        if self.label_type == "mask":
            label = self._read_mask_file(image_name)
        elif self.label_type == "txt":
            label = TXT2MASK(self.label_dir, image_name, self.target_size)
        elif self.label_type == "json":
            label = JSON2MASK(self.label_dir, image_name, self.target_size)
        elif self.label_type == "npy":
            label = NPY2MASK(self.label_dir, image_name, self.target_size)
        else:
            label = np.zeros((*self.target_size[::-1], 1), dtype=np.uint8)
        return label
    
    def _read_mask_file(self, image_name):
        """
        直接读取 mask 格式的标签文件
        """
        stem = os.path.splitext(image_name)[0]
        mask_path = None
        
        for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff', '.npy']:
            candidate = os.path.join(self.label_dir, stem + ext)
            if os.path.exists(candidate):
                mask_path = candidate
                break
        
        if mask_path is None:
            return np.zeros((*self.target_size[::-1], 1), dtype=np.uint8)
        
        if mask_path.lower().endswith('.npy'):
            label = np.load(mask_path)
        else:
            label = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if label is None:
                label = np.zeros((*self.target_size[::-1], 1), dtype=np.uint8)
            else:
                label = label[..., np.newaxis]
        
        return label



    def __len__(self):
        return len(self.images_name)
    
    def __getitem__(
        self,
        index: int,
        ):
        image_name = self.images_name[index]
        image_path = os.path.join(self.image_dir, image_name)
        
        # 读取并处理图像
        image = image_transform(image_path=image_path, target_size=self.target_size)
        image = enhance_image(image)
        transform = transforms.Compose([transforms.ToTensor()])
        image = transform(image)
        
        # 读取标签
        label = self.get_mask(image_name)
        label = torch.from_numpy(label).float()
        
        return image, label


        

