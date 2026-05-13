import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ACTIVATION_MAP
from utils import autopad, get_activation

"""模型基础卷积模块封装库。

这里包含了常用的各种基础构建块,如基础卷积操作,以及不同形式的池化层封装。
所有模块注册在内部的注册表中以便于统一调用。
"""
BLOCK_REGISTRY = {}


def register_block(name):
    """装饰器工厂：将构建函数注册到 BLOCK_REGISTRY。

    Args:
        name (str): 注册名称，内部会转为小写。

    Returns:
        Callable: 装饰器函数。
    """

    def decorator(func):
        """将构建函数写入注册表。

        Args:
            func (Callable): 待注册的构建函数。

        Returns:
            Callable: 原函数本身。
        """
        BLOCK_REGISTRY[name.lower()] = func
        return func

    return decorator


############################################################
#               基本卷积相关的模块                            #
############################################################
@register_block("conv")
class Conv(nn.Module):
    """基础卷积层封装。

    继承 nn.Module 的二维卷积封装模块。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
        g (int): 分组数。
        b (bool): 是否使用偏置。
    """

    def __init__(
        self, in_ch: int, out_ch: int, k: int, s: int, p: int, d: int, g: int, b: bool
    ):
        """初始化卷积层封装类。"""
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=g,
                bias=b,
            )
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        """
        return self.forward_basic(x)


@register_block("basic_conv_block")
class Basic_Conv_Block(nn.Module):
    """卷积 + BN + 激活的基础模块。

    包含二维卷积层、批量归一化层和可指定的激活函数。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
        g (int): 分组数。
        activation (str): 激活函数名称。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        s: int,
        p: int,
        d: int,
        g: int,
        activation: str,
    ):
        """初始化基础卷积模块。"""
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=g,
            ),
            nn.BatchNorm2d(out_ch),
        )
        self.act = get_activation(activation, ACTIVATION_MAP)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 激活后的输出。
        """
        return self.act(self.forward_basic(x))


@register_block("conv_block_nonb")
class Conv_Block_NONB(nn.Module):
    """不含 BN 的卷积 + 激活模块。

    包含二维卷积层和可指定的激活函数。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
        g (int): 分组数。
        activation (str): 激活函数名称。
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        k: int,
        s: int,
        p: int,
        d: int,
        g: int,
        activation: str,
    ):
        """初始化不含 BN 的卷积模块。

        Returns:
            None
        """
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=g,
            )
        )
        self.act = get_activation(activation, ACTIVATION_MAP)

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 激活后的输出。
        """
        return self.act(self.forward_basic(x))


############################################################
#               基本池化相关的模块                            #
############################################################
@register_block("maxpool")
class MaxPool(nn.Module):
    """最大池化层封装。

    提供了一个简单的最大池化层封装,继承自 nn.Module。

    Attributes:
        k (int): 卷积核大小。
        s (int): 步幅。
        p (int): 填充。
        d (int): 空洞率。
    """

    def __init__(self, k: int, s: int, p: int, d: int):
        """初始化最大池化层封装类。"""
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.MaxPool2d(kernel_size=k, stride=s, padding=p, dilation=d)
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 池化后的输出张量。
        """
        return self.forward_basic(x)


@register_block("adaptive_max_pool")
class AdaptiveMaxPool(nn.Module):
    """自适应最大池化层封装。

    提供自适应大小的最大池化功能,可动态输出指定尺寸特征图。

    Attributes:
        output_size (int | Tuple[int, int]): 输出尺寸。
    """

    def __init__(self, output_size: int):
        """初始化自适应最大池化层。"""
        super(AdaptiveMaxPool, self).__init__()
        self.forward_basic = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=output_size)
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 池化后的输出张量。
        """
        return self.forward_basic(x)
