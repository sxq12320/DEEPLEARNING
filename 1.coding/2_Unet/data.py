import numpy as np
from pandas.core.aggregation import transform
from torch.utils.data import Dataset
import os
import torch
from utils import keep_image_size_open , keep_seg_size_open

from torchvision import transforms

from utils import keep_image_size_open, enhance_image
import matplotlib.pyplot as plt


transform = transforms.Compose([transforms.ToTensor()]) # 转换成pytorch看的懂得格式

class MyDataset(Dataset):
    def __init__(self, path):
        '''
        using:
            读取数据集的地址；
            将数据集下的地址拼接，并生成名称列表
        Args:
            path (str): 数据集的地址
        Returns:
            None
        '''
        self.path = path  # reading files' path
        self.name = os.listdir(os.path.join(path , 'SegmentationClass')) # 返回当前地的文件变成列表

    def __len__(self):
        '''
        using:
            之前的列表长度，也就是文件夹下的文件个数
        Args:
            None
        Returns:
            len (int) : 列表的长度
        '''
        return len(self.name) # get filename length

    def __getitem__(self, index : int , size = (256,256)):
        '''
        using:
            得到tensor类型的图片，包括了原图和分割图
        Args:
            index (int) : 图片的索引，0-len
        Returns:
            列表包括了tensor的两种图片
            第一个参数是索引数
            第二个参数是类型，原图和分割后的掩码图片
        '''
        segmentation_name = self.name[index]
        segmentation_path = os.path.join(self.path , 'SegmentationClass' , segmentation_name) # 掩码图片地址
        image_path = os.path.join(self.path , 'JPEGImages' , segmentation_name.replace('png', 'jpg')) # 原图地址
        # segmentation_image = keep_image_size_open(segmentation_path , size)
        image = keep_image_size_open(image_path , size)
        image = enhance_image(image)

        seg = keep_seg_size_open(segmentation_path, size)  # 直接拿到正确的 P 模式图
        seg_np = np.array(seg, dtype=np.int64)
        seg_np[seg_np == 255] = 0
        seg_tensor = torch.from_numpy(seg_np)  # shape: (H, W)

        return transform(image) , seg_tensor


# if __name__ == '__main__':
#     data = MyDataset(r"E:\mastercode\data\VOC\VOCtrainval_06-Nov-2007\VOCdevkit\VOC2007")
#     print(data[0][1].shape)
#     image, segementation = data[6]
#     image_display = image.permute(1 , 2 , 0)
#     segementation_display = segementation.permute(1,2, 0)
#
#     plt.subplot(1,2,1)
#     plt.imshow(image_display)
#     plt.subplot(1,2,2)
#     plt.imshow(segementation_display)
#     plt.show()


