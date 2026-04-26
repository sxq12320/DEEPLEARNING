import torch
import torch.nn as nn
import torch.nn.functional as F
from block import (
                  Conv,
                  Basic_Conv_Block,
                  Conv_Block_NONB,
                  DepthWise_Conv,
                  PointWise_Conv,
                  DepthWiseSeparable_Conv,
                  ResNetBlock_34,
                  CBAM_Channel_Attention,
                  CBAM_Spatial_Attention,
                  CBAM,
                  ResNetBlock_34,
                )
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import json
import os
import cv2

############################################################
#                                                          #
#                                                          #
#               下面是基于模块的一些常用函数                    #
#                                                          #
#                                                          #
############################################################
def get_activation(act_name:str , activation_map:dict):
    '''
    根据名称从映射表中获取激活函数模块。

    Args:
        act_name (str): 激活函数名称,函数内部会转为小写并去除首尾空格。
        activation_map (dict): 激活函数映射表,键为名称,值为激活模块实例。

    Returns:
        nn.Module: 对应的激活函数模块实例。

    Raises:
        ValueError: act_name 不在 activation_map 中时抛出。
    '''
    act_name = act_name.strip().lower()
    if act_name not in activation_map:
        supported = ",".join(sorted(activation_map.keys()))
        raise ValueError(f"Unsupported activation: {act_name}. Supported activations: {supported}")
    return activation_map[act_name]


def autopad(k, p=None, d=1):
    """
    返回p使得在当前的条件之下让卷积前后的图片尺寸不发生任何的变化

    Args:
        k (int): 卷积核大小。
        p (int, optional): 填充大小。默认为 None。
        d (int): 膨胀率。默认为 1。

    Returns:
        p (int): 适当的填充大小。
    """
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k] 
    return p


def make_layers(cfg):
    '''
    简化模块构建的繁琐方式，具体的模块包括:

    "conv" :in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , bias
    "basic_conv_block" :in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation
    "conv_block_nonb" :in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation
    "depthwise_conv":in_channels , kernel_size , stride_size , padding_size , dilation_size 
    "pointwise_conv":in_channels , out_channels , stride_size 
    "depthwise_separable_conv":in_channels , out_channels , kernel_size , padding_size , stride_size_D , stride_size_P , dilation_size_D , activation
    "resnet_block_34":in_channels , out_channels , stride_size , activation_1  , activation_2
    "cbam_channel_attention":in_channels , reductiaon_ratio , activation
    "cbam_spatial_attention":kernel_size
    "cbam":in_channels , reduction_ratio , activation , kernel_size

    Args:
        各种类型的参数，需要后续自行进行配置即可
    
    Returns:
        nn.Sequential: 构建好的网络层
  
    '''
    layers = []     
    for idx, item in enumerate(cfg):
        if not isinstance(item, (list, tuple)) or len(item) == 0:
            raise ValueError(f"cfg[{idx}] must be a non-empty list/tuple, got: {item}")

        block_type = str(item[0]).strip().lower()
        repeat = int(item[-1]) if len(item) > 1 and isinstance(item[-1], int) else 1
        if repeat < 1:
            raise ValueError(f"cfg[{idx}] repeat must be >= 1, got: {repeat}")
        

        if block_type == 'conv':
            '''
            构建不存在激活函数和批量归一化的卷积块
            '''
            if len(item) != 10:
                raise ValueError(f"cfg[{idx}] conv 期望的参数个数为10, 现在输入的参数是 {len(item)}: {item}")
            bloack_name , in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , bias , repeat = item
            layers.append(
                Conv(
                    in_ch=in_channels,
                    out_ch=out_channels,
                    k=kernel_size,
                    s=stride_size,
                    p=padding_size,
                    d=dilation_size,
                    g=groups_size,
                    b=bias,
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    Conv(
                        in_ch=out_channels,
                        out_ch=out_channels,
                        k=kernel_size,
                        s=1,
                        p=padding_size,
                        d=dilation_size,
                        g=groups_size,
                        b=bias
                    )
                )

        elif block_type == 'basic_conv_block':
            '''
            构建基本卷积块 
            '''
            if len(item) != 10:
                raise ValueError(f"cfg[{idx}] basic_conv_block 期望的参数个数为10, 现在输入的参数是 {len(item)}: {item}")
            bloack_name , in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation , repeat = item
            layers.append(
                Basic_Conv_Block(
                in_ch=in_channels,
                out_ch=out_channels,
                k=kernel_size,
                s=stride_size,
                p=padding_size,
                d=dilation_size,
                g=groups_size,
                activation=activation
                )
            )
            for _ in range(1 , repeat):
                layers.append(Basic_Conv_Block(
                    in_ch=out_channels,
                    out_ch=out_channels,
                    k=kernel_size,
                    s=1, # 设置为1防止出现奇奇怪怪的问题
                    p=padding_size,
                    d=dilation_size,
                    g=groups_size,
                    activation=activation
                ))

        elif block_type == 'conv_basic_nonb':
            '''
            构建没有批归一化的基本的卷积块
            '''
            if len(item) != 10:
                raise ValueError(f"cfg[{idx}] conv_basic_nonb 期望的参数个数为10, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , out_channels , kernel_size , stride_size , padding_size , dilation_size , groups_size , activation ,repeat = item
            layers.append(
                Conv_Block_NONB(
                    in_ch=in_channels,
                    out_ch = out_channels,
                    k=kernel_size,
                    s=stride_size,
                    p=padding_size,
                    d=dilation_size,
                    g=groups_size,
                    activation=activation
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    Conv_Block_NONB(
                        in_ch=out_channels,
                        out_ch=out_channels,
                        k=kernel_size,
                        s=1,
                        p=padding_size,
                        d=dilation_size,
                        g=groups_size,
                        activation=activation
                    )
                )

        elif block_type == 'depthwise_conv':
            '''
            构建基本的深度分离卷积模块
            '''
            if len(item) != 7:
                raise ValueError(f"cfg[{idx}] depthwise_conv 期望的参数个数为7, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , kernel_size , stride_size , padding_size , dilation_size , repeat = item
            layers.append(
                DepthWise_Conv(
                    in_ch= in_channels,
                    k = kernel_size,
                    s = stride_size,
                    p = padding_size,
                    d = dilation_size,
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    DepthWise_Conv(
                    in_ch= in_channels,
                    k = 1,
                    s = stride_size,
                    p = padding_size,
                    d = dilation_size,
                    )
                )
        
        elif block_type == "pointwise_conv":
            '''
            逐点卷积的基本模块
            '''
            if len(item) != 5:
                raise ValueError(f"cfg[{idx}] pointwise_conv 期望的参数个数为5, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , out_channels , stride_size , repeat = item
            layers.append(
                PointWise_Conv(
                    in_ch=in_channels , 
                    out_ch= out_channels,
                    s = stride_size,
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    PointWise_Conv(
                    in_ch=in_channels , 
                    out_ch= out_channels,
                    s = 1,
                    )
                )
        
        elif block_type == 'depthwise_separable_conv':
            '''
            深度可分离卷积的基本模块
            '''
            if len(item) != 10:
                raise ValueError(f"cfg[{idx}] depthwise_separable_conv 期望的参数个数为10, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , out_channels , kernel_size , padding_size , stride_size_D , stride_size_P , dilation_size_D , activation , repeat = item
            layers.append(
                DepthWiseSeparable_Conv(
                    in_ch=in_channels,
                    out_ch=out_channels,
                    k=kernel_size,
                    p=padding_size,
                    s_D=stride_size_D,
                    s_P=stride_size_P,
                    d_D=dilation_size_D,
                    activation=activation
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    DepthWiseSeparable_Conv(
                        in_ch=in_channels,
                        out_ch=out_channels,
                        k=kernel_size,
                        p=padding_size,
                        s_D=stride_size_D,
                        s_P=stride_size_P,
                        d_D=dilation_size_D,
                        activation=activation
                    )
                )

        elif block_type == "resnet_block_34":
            '''
            ResNet-34 基本模块
            '''
            if len(item) != 7:
                raise ValueError(f"cfg[{idx}] resnet_block_34 期望的参数个数为7, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , out_channels , stride_size , activation_1  , activation_2 , repeat = item
            layers.append(
                ResNetBlock_34(
                    in_ch = in_channels,
                    out_ch=out_channels,
                    s = stride_size,
                    activation_1 = activation_1,
                    activation_2 = activation_2
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    ResNetBlock_34(
                    in_ch = in_channels,
                    out_ch=out_channels,
                    s = 1,
                    activation_1 = activation_1,
                    activation_2 = activation_2
                    )
                )

        elif block_type == "cbam_channel_attention":
            '''
            CBAM通道注意力机制
            '''
            if len(item) != 5:
                raise ValueError(f"cfg[{idx}] cbam_channel_attention 期望的参数个数为4, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , reductiaon_ratio , activation ,  repeat = item
            layers.append(
                CBAM_Channel_Attention(
                    in_ch=in_channels,
                    reduction_ratio=reductiaon_ratio,
                    activation= activation
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    CBAM_Channel_Attention(
                        in_ch=in_channels,
                        reduction_ratio=reductiaon_ratio,
                        activation= activation
                    )
                )

        elif block_type == "cbam_spatial_attention":
            '''
            CBAM空间注意力机制
            '''
            if len(item) != 3:
                raise ValueError(f"cfg[{idx}] cbam_spatial_attention 期望的参数个数为4, 现在输入的参数是{len(item)}: {item}")
            name_block  , kernel_size ,  repeat = item
            layers.append(
                CBAM_Spatial_Attention(
                    k=kernel_size
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    CBAM_Spatial_Attention(
                        k=kernel_size
                    )
                )

        elif block_type == "cbam":
            '''
            CBAM注意力机制
            '''
            if len(item) != 6:
                raise ValueError(f"cfg[{idx}] cbam 期望的参数个数为6, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , reduction_ratio , activation , kernel_size ,  repeat = item
            layers.append(
                CBAM(
                    in_ch=in_channels,
                    reduction_ratio=reduction_ratio,
                    activation=activation,
                    k=kernel_size
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    CBAM(
                        in_ch=in_channels,
                        reduction_ratio=reduction_ratio,
                        activation=activation,
                        k=kernel_size
                    )
                )
        
        else:
            raise ValueError(
                f"Unsupported block type at cfg[{idx}]: {item[0]}. "
                "Supported: basic_conv_block, conv_block_nonb, depthwise_conv, pointwise_conv, "
                "depthwise_separable_conv, res34, cbam_channel_attention, cbam_spatial_attention, cbam"
            )

    return nn.Sequential(*layers)

############################################################
#                                                          #
#                                                          #
#               下面是图像增强相关的一些函数                    #
#                                                          #
#                                                          #
############################################################
def enhance_image(image):
    pass


############################################################
#                                                          #
#                                                          #
#               下面是图像变换的一些函数                       #
#                                                          #
#                                                          #
############################################################
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


############################################################
#                                                          #
#                                                          #
#               下面是标签转换的相关函数                       #
#                                                          #
#                                                          #
############################################################
def TXT2MASK(label_dir, image_name, target_size):
    """
    读取 TXT 格式标签文件（支持 YOLO 格式）
    
    Args:
        label_dir: 标签目录
        image_name: 图像文件名
        target_size: 目标大小 (width, height)
    
    Returns:
        label: 标签数组，形状为 (height, width, 1)
    """
    stem = Path(image_name).stem
    txt_path = os.path.join(label_dir, f"{stem}.txt")
    
    h, w = target_size[1], target_size[0]
    label = np.zeros((h, w, 1), dtype=np.uint8)
    
    if not os.path.exists(txt_path):
        return label
    
    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        for line in lines:
            vals = [float(v) for v in line.split()]
            if len(vals) < 5:
                continue
            
            cls_id = int(vals[0])
            
            if len(vals) == 5:
                cx, cy, bw, bh = vals[1:5]
                if max(cx, cy, bw, bh) <= 1.0:
                    cx, cy, bw, bh = cx * w, cy * h, bw * w, bh * h
                x1 = int(max(0, cx - bw / 2))
                y1 = int(max(0, cy - bh / 2))
                x2 = int(min(w - 1, cx + bw / 2))
                y2 = int(min(h - 1, cy + bh / 2))
                cv2.rectangle(label, (x1, y1), (x2, y2), color=cls_id, thickness=-1)
            else:
                # YOLO polygon: class x1 y1 x2 y2 ...
                pts = np.array(vals[1:], dtype=np.float32).reshape(-1, 2)
                if pts.max() <= 1.0:
                    pts[:, 0] *= w
                    pts[:, 1] *= h
                pts = np.round(pts).astype(np.int32)
                cv2.fillPoly(label, [pts], color=cls_id)
    
    except Exception as e:
        print(f"读取 TXT 标签失败: {txt_path}, 错误: {e}")
    
    return label


def JSON2MASK(label_dir, image_name, target_size):
    """
    读取 JSON 格式标签文件（支持 COCO 格式）
    
    Args:
        label_dir: 标签目录或 COCO json 文件路径
        image_name: 图像文件名
        target_size: 目标大小 (width, height)
    
    Returns:
        label: 标签数组，形状为 (height, width, 1)
    """
    h, w = target_size[1], target_size[0]
    label = np.zeros((h, w, 1), dtype=np.uint8)
    
    # 确定 JSON 文件路径
    json_path = None
    if os.path.isfile(label_dir) and label_dir.lower().endswith('.json'):
        json_path = label_dir
    else:
        json_files = [f for f in os.listdir(label_dir) if f.lower().endswith('.json')]
        if json_files:
            json_path = os.path.join(label_dir, json_files[0])
    
    if json_path is None or not os.path.exists(json_path):
        return label
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            coco_data = json.load(f)
        
        # 构建图像名到 ID 的映射
        img_id_map = {}
        img_info_map = {}
        for img_info in coco_data.get('images', []):
            img_id_map[img_info.get('file_name')] = img_info.get('id')
            img_info_map[img_info.get('id')] = img_info
        
        image_id = img_id_map.get(image_name)
        if image_id is None:
            return label
        
        img_w = img_info_map[image_id].get('width', w)
        img_h = img_info_map[image_id].get('height', h)
        
        # 读取该图像的所有标注
        for ann in coco_data.get('annotations', []):
            if ann.get('image_id') != image_id:
                continue
            
            cls_id = int(ann.get('category_id', 1))
            segmentation = ann.get('segmentation', [])
            
            if isinstance(segmentation, list) and segmentation:
                # Polygon segmentation
                for poly in segmentation:
                    if len(poly) < 6:
                        continue
                    pts = np.array(poly, dtype=np.float32).reshape(-1, 2)
                    # 缩放到目标大小
                    max_len = max(img_w, img_h)
                    pts[:, 0] = pts[:, 0] / max_len * w
                    pts[:, 1] = pts[:, 1] / max_len * h
                    pts = np.round(pts).astype(np.int32)
                    cv2.fillPoly(label, [pts], color=cls_id)
            elif isinstance(segmentation, dict) and 'counts' in segmentation:
                # RLE segmentation (不直接处理，可选)
                pass
            else:
                # 回退到 bbox
                bbox = ann.get('bbox', None)
                if bbox and len(bbox) == 4:
                    x, y, bw, bh = bbox
                    max_len = max(img_w, img_h)
                    x1 = int(max(0, x / max_len * w))
                    y1 = int(max(0, y / max_len * h))
                    x2 = int(min(w - 1, (x + bw) / max_len * w))
                    y2 = int(min(h - 1, (y + bh) / max_len * h))
                    cv2.rectangle(label, (x1, y1), (x2, y2), color=cls_id, thickness=-1)
    
    except Exception as e:
        print(f"读取 JSON 标签失败: {json_path}, 错误: {e}")
    
    return label


def NPY2MASK(label_dir, image_name, target_size):
    """
    读取 NPY 格式标签文件
    
    Args:
        label_dir: 标签目录
        image_name: 图像文件名
        target_size: 目标大小 (width, height)
    
    Returns:
        label: 标签数组，形状为 (height, width, 1)
    """
    stem = Path(image_name).stem
    npy_path = os.path.join(label_dir, f"{stem}.npy")
    
    h, w = target_size[1], target_size[0]
    label = np.zeros((h, w, 1), dtype=np.uint8)
    
    if not os.path.exists(npy_path):
        return label
    
    try:
        data = np.load(npy_path)
        
        # 如果 npy 文件是二维的，添加通道维度
        if data.ndim == 2:
            data = data[..., np.newaxis]
        
        # 如果大小不匹配，进行缩放
        if data.shape != (h, w, 1):
            # 简单的最近邻缩放
            data_2d = data[..., 0] if data.ndim == 3 else data
            resized = cv2.resize(data_2d, target_size, interpolation=cv2.INTER_NEAREST)
            label = resized[..., np.newaxis]
        else:
            label = data.astype(np.uint8)
    
    except Exception as e:
        print(f"读取 NPY 标签失败: {npy_path}, 错误: {e}")
    
    return label



            


# if __name__ == "__main__":

#     cfg = [
#     ]
#     model = make_layers(cfg)
#     print(model)