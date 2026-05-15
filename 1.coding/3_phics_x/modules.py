############################################################
#                                                          #
#                                                          #
#                     基础模块的封装库                        #
#                                                          #
#                                                          #
############################################################
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import ACTIVATION_MAP, get_activation


class MaxPool(nn.Module):
    """
    最大池化层
    ---
    Args:
        k(int): 池化核大小
        s(int): 步长
        p(int): 填充
        d(int): 膨胀
    ---
    Returns:
        池化后的特征图
    """

    def __init__(self, k: int, s: int, p: int, d: int):
        super().__init__()
        self.k = k
        self.s = s
        self.p = p
        self.d = d

    def forward(self, x):
        return F.max_pool2d(x, self.k, self.s, self.p, self.d)


class cba(nn.Module):
    """
    卷积-批归一化-激活层
    ---
    Args:
        in_channel(int): 输入通道数
        out_channel(int): 输出通道数
        kernel_size(int): 卷积核大小
        stride(int): 步长
        padding(int): 填充
        dilation(int): 膨胀
        group(int): 分组卷积
        bias(bool): 是否使用偏置
        act_name(str): 激活函数名称
    ---
    Returns:
        激活后的特征图
    """

    def __init__(
        self,
        in_channel: int,
        out_channel: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        group: int = 1,
        bias: bool = False,
        act_name: str = "relu",
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=group,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channel)
        self.act = get_activation(act_name=act_name, activation_map=ACTIVATION_MAP)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
