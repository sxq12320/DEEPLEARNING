import torch
import torch.nn as nn
import torch.nn.functional as F
# from models.Block import (Basic_Conv_Block,
#                           Conv_Block_NONB,
#                           DepthWise_Conv,
#                           PointWise_Conv,
#                           DepthWiseSeparable_Conv,
#                           ResNetBlock_34,
#                           )
# from models.Block import (CBAM_Channel_Attention,
#                           CBAM_Spatial_Attention,
#                           CBAM,
#                           )
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import models

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
    简化模块构建的繁琐方式，后续在进行构建的时候只需要做下面的例子即可
    
    Example: 
    cfg = [
    ['res34', 64, 128, 2, 'relu', 'relu', 2],
    ['basic_conv_block', 128, 256, 2, 3, 1, 1, 1, 'relu', 2],
    ]
    即可构建整个架构

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
                models.Conv(
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
                    models.Conv(
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
                models.Basic_Conv_Block(
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
                layers.append(models.Basic_Conv_Block(
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
                models.Conv_Block_NONB(
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
                    models.Conv_Block_NONB(
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
                models.DepthWise_Conv(
                    in_ch= in_channels,
                    k = kernel_size,
                    s = stride_size,
                    p = padding_size,
                    d = dilation_size,
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    models.DepthWise_Conv(
                    in_ch= in_channels,
                    k = 1,
                    s = stride_size,
                    p = padding_size,
                    d = dilation_size,
                    )
                )
        
        elif block_type == "pointwise_conv":
            if len(item) != 5:
                raise ValueError(f"cfg[{idx}] pointwise_conv 期望的参数个数为5, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , out_channels , stride_size , repeat = item
            layers.append(
                models.PointWise_Conv(
                    in_ch=in_channels , 
                    out_ch= out_channels,
                    s = stride_size,
                )
            )
            for _ in range(1 , repeat):
                layers.append(
                    models.PointWise_Conv(
                    in_ch=in_channels , 
                    out_ch= out_channels,
                    s = 1,
                    )
                )
        
        elif block_type == 'depthwise_separable_conv':
            if len(item) != 10:
                raise ValueError(f"cfg[{idx}] depthwise_separable_conv 期望的参数个数为10, 现在输入的参数是{len(item)}: {item}")
            name_block , in_channels , out_channels , kernel_size , padding_size , stride_size_D , stride_size_P , dilation_size_D , activation , repeat = item
            layers.append(
                models.DepthWiseSeparable_Conv(
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
                    models.DepthWiseSeparable_Conv(
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


        else:
            raise ValueError(
                f"Unsupported block type at cfg[{idx}]: {item[0]}. "
                "Supported: basic_conv_block, conv_block_nonb, depthwise_conv, pointwise_conv, "
                "depthwise_separable_conv, res34, cbam_channel_attention, cbam_spatial_attention, cbam"
            )

    return nn.Sequential(*layers)


# if __name__ == "__main__":

#     cfg = [
#     ]
#     model = make_layers(cfg)
#     print(model)