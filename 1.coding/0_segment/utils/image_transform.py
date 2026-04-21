import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import os 
from PIL import Image

def image_transform(
        image_path, 
        target_size = (640 , 640)
        ): 
    '''
    using : 修改图片的大小让他的大小同目标大小一致，方法略
    
    Args:
        image_path : 图片的地址路径，需要精确到图片的单位位置
        target_size : 目标图片的大小，格式为 (width, height)

    Returns:
        image : 调整大小后的图片
    '''

    image = Image.open(image_path)
    w , h = image.size
    max_len_image = max(w ,h)
    # max_len_target = max(target_size[0] , target_size[1])
    # scaling_factor = max_len_target / max_len_image
    # h , w = h * scaling_factor , w * scaling_factor
    black_mask = Image.new("RGB" , (max_len_image , max_len_image) , (0, 0, 0))
    black_mask.paste(image , (0, 0))
    image = black_mask.resize(target_size)
    return image




            