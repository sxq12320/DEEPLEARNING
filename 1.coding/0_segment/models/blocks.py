import torch.nn as nn
import torch
import torch.nn.functional as F
from configs.config import ACTIVATION_MAP

class Basic_Conv_Block(nn.Module):
    '''
        基本卷积块：二维卷积 + 批量归一化 + 激活函数。

        Args:
            in_channels (int): 输入通道数。
            out_channels (int): 输出通道数。
            kernel_size (int): 卷积核大小。
            stride (int): 卷积步长。
            padding (int): 填充大小。
            dilated (int): 膨胀率(dilation)。
            groups (int): 分组卷积组数。
            activation (str): 激活函数名称,大小写不敏感,取值由 ACTIVATION_MAP 决定。

        Notes:
            前向传播输出说明见 forward 方法。
    '''
    def __init__(
            self,
            in_ch:int,
            out_ch:int, 
            k:int,
            s:int,
            p:int,
            d:int,
            g:int, 
            activation:str
    ):
        super(Basic_Conv_Block, self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride = s,
                padding= p,
                dilation= d,
                groups=g,
            ),
            nn.BatchNorm2d(out_ch),
        )
        self.act = get_activation(activation , ACTIVATION_MAP)

    def forward(self , x):
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过卷积、归一化和激活后的输出张量。
        '''
        x = self.forward_basic(x)
        x = self.act(x)
        return x


class Conv_Block_NONB(nn.Module):
    '''
        无 BN 层的基本卷积块：二维卷积 + 激活函数。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 卷积步长。
            p (int): 填充大小。
            d (int): 膨胀率(dilation)。
            g (int): 分组卷积组数。
            activation (str): 激活函数名称,大小写不敏感。

        Notes:
             前向传播输出说明见 forward 方法。
    '''
    def __init__(
            self,
            in_ch:int,
            out_ch:int, 
            k:int,
            s:int,
            p:int,
            d:int,
            g:int, 
            activation:str
    ):
        super(Conv_Block_NONB, self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride = s,
                padding= p,
                dilation= d,
                groups=g,
            )
        )
        self.act = get_activation(activation , ACTIVATION_MAP)

    def forward(self , x):
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过卷积和激活后的输出张量。
        '''
        x = self.forward_basic(x)
        x = self.act(x)
        return x


class DepthWise_Conv(nn.Module):
    '''
        深度卷积模块骨架。

        说明:
            深度卷积一般满足 groups == in_channels,且每个通道单独卷积。

        Args:
            in_ch (int): 输入通道数。
            k (int): 卷积核大小。
            s (int): 卷积步长。
            p (int): 填充大小。
            d (int): 膨胀率(dilation), 最小肯定是1
    '''
    def __init__(
            self,
            in_ch:int,
            k:int,
            s:int,
            p:int,
            d:int,
            ):
        super(DepthWise_Conv , self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=in_ch
            )
        )

    def forward(self , x):
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过深度卷积后生成的张量。
        '''
        return self.forward_basic(x)
        


class PointWise_Conv(nn.Module):
    '''
        逐点卷积(1x1 卷积)模块骨架。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            s (int): 卷积步长。

    '''
    def __init__(
            self,
            in_ch:int,
            out_ch:int, 
            s:int,
    ):
        super(PointWise_Conv , self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels = in_ch,
                out_channels = out_ch,
                kernel_size= 1,
                stride = s,
                padding= 0,
                # dilation= d,
                # groups= g
            )
        )
    def forward(self , x):
        '''
        前向传播。

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过经过逐点卷积后生成的张量。
        '''

        return self.forward_basic(x)


class DepthWiseSeparable_Conv(nn.Module):
    '''
    深度可分离卷积，综合大块

    Args:
        in_ch 
    '''
    def __init__(
            self,
            in_ch:int,
            out_ch:int,
            k:int,
            s_D:int,
            s_P:int,
            d_D:int,


    )










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
    if act_name not in activation_map:
        supported = ",".join(sorted(activation_map.keys()))
        raise ValueError(f"Unsupported activation: {act_name}. Supported activations: {supported}")
    return activation_map[act_name]





# if __name__ == "__main__":
#     x = torch.randn(1 , 3 , 640 , 640)
#     model = Basic_Conv_Block(
#         in_channels=3,
#         out_channels=64,
#         kernel_size=3,
#         stride=1,
#         padding=1,
#         dilated=1,
#         groups=1,
#         activation="relu"
#     )

#     output = model(x)
#     print(output.size())