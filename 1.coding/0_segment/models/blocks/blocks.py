from turtle import forward

import torch.nn as nn
import torch
import torch.nn.functional as F
from configs.config import ACTIVATION_MAP
from utils.common import get_activation,autopad
from ..registries.registry import register_block   # 导入注册器
import numpy as np


############################################################
#                                                          #
#                                                          #
#               下面全都是基本卷积相关的模块                    #
#                                                          #
#                                                          #
############################################################
@register_block('maxpool')
class MaxPool(nn.Module):
    '''
    最大池化函数

    Args:
        k (int): 池化核大小。
        s (int): 池化步长。
        p (int): 填充大小。
        d (int): 膨胀率(dilation)。
    
    Notes:
        具体运算过程请看forward函数的说明
    '''
    def __init__(
            self,
            k:int,
            s:int,
            p:int,
            d:int,
            ):
        '''
        初始化最大池化模块。

        Args:
            k (int): 池化核大小。
            s (int): 池化步长。
            p (int): 填充大小。
            d (int): 膨胀率。
        '''
        super(MaxPool , self).__init__()  
        self.forward_basic = nn.Sequential(
            nn.MaxPool2d(
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d
            )
        )  
    def forward(self , x):
        '''
        最大池化的前向传播函数

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过最大池化后的输出张量。
        '''
        return self.forward_basic(x)


@register_block('adaptiveavgpool')
class AdaptiveAvgPool(nn.Module):
    '''
    平均池化函数

    Agrs:
        output_size (tuple): 输出尺寸。

    Notes:
        具体运算过程请看forward函数的说明
    '''
    def __init__(
            self,
            output_size,
            ):
        '''
        初始化自适应平均池化模块。

        Args:
            output_size (tuple): 输出尺寸。
        '''
        super(AdaptiveAvgPool , self).__init__()
        self.forward_basic = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=output_size)
        )
    def forward(self , x):
        '''
        前馈函数

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过自适应平均池化后的输出张量。
        '''
        return self.forward_basic(x)


@register_block('conv')
class Conv(nn.Module):
    '''
    最最普通的二维卷积操作

    Args:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 卷积核大小。
        s (int): 卷积步长。
        p (int): 填充大小。
        d (int): 膨胀率(dilation)。
        g (int): 分组卷积组数。
        b (bool): 是否使用偏置。
    
    Notes:
        具体运算过程请看forward函数的说明

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
            b:bool,
            ):
        '''
        初始化普通二维卷积模块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 卷积步长。
            p (int): 填充大小。
            d (int): 膨胀率。
            g (int): 分组数。
            b (bool): 是否使用偏置。
        '''
        super(Conv , self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups = g,
                bias = b,
            )
        )

    def forward(self , x):
        '''
        普通卷积的前向传播函数

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过卷积后的输出张量。
        '''
        return self.forward_basic(x)


@register_block('basic_conv_block')
class Basic_Conv_Block(nn.Module):
    '''
        基本卷积块：二维卷积 + 批量归一化 + 激活函数。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 卷积步长。
            p (int): 填充大小。
            d (int): 膨胀率(dilation)。
            g (int): 分组卷积组数。
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
        '''
        初始化基本卷积块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 卷积步长。
            p (int): 填充大小。
            d (int): 膨胀率。
            g (int): 分组数。
            activation (str): 激活函数名称。
        '''
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


@register_block('conv_block_nonb')
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
        '''
        初始化无 BN 的卷积块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): 卷积核大小。
            s (int): 卷积步长。
            p (int): 填充大小。
            d (int): 膨胀率。
            g (int): 分组数。
            activation (str): 激活函数名称。
        '''
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


@register_block('depthwise_conv')
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
        '''
        初始化深度卷积模块。

        Args:
            in_ch (int): 输入通道数。
            k (int): 卷积核大小。
            s (int): 卷积步长。
            p (int): 填充大小。
            d (int): 膨胀率。
        '''
        super(DepthWise_Conv , self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch,
                kernel_size=k,
                stride=s,
                padding=p,
                dilation=d,
                groups=in_ch,
                bias=False
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
        


@register_block('pointwise_conv')
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
        '''
        初始化逐点卷积模块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            s (int): 卷积步长。
        '''
        super(PointWise_Conv , self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels = in_ch,
                out_channels = out_ch,
                kernel_size= 1,
                stride = s,
                padding= 0,
                bias=False
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


@register_block('depthwise_separable_conv')
class DepthWiseSeparable_Conv(nn.Module):
    '''
    深度可分离卷积,综合大块

    Args:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        k (int): 深度卷积的卷积核大小。
        p (int):深度卷积的填充大小。
        s_D (int): 深度卷积的步长。
        s_P (int): 逐点卷积的步长。
        d_D (int): 深度卷积的膨胀率。
        activation (str): 激活函数名称,大小写不敏感

    Notes:
            前向传播输出说明见 forward 方法。
    '''
    def __init__(
            self,
            in_ch:int,
            out_ch:int,
            k:int,
            p:int,
            s_D:int,
            s_P:int,
            d_D:int,
            activation:str
    ):
        '''
        初始化深度可分离卷积模块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            k (int): 深度卷积核大小。
            p (int): 填充大小。
            s_D (int): 深度卷积步长。
            s_P (int): 逐点卷积步长。
            d_D (int): 深度卷积膨胀率。
            activation (str): 激活函数名称。
        '''
        super(DepthWiseSeparable_Conv , self).__init__()
        self.forward_basic = nn.Sequential(
            DepthWise_Conv(
                in_ch=in_ch,
                k = k,
                s = s_D,
                p = p,
                d = d_D
            ),
            nn.BatchNorm2d(in_ch),
            get_activation(activation , ACTIVATION_MAP),
            PointWise_Conv(
                in_ch=in_ch,
                out_ch=out_ch,
                s=s_P
            ),
            nn.BatchNorm2d(out_ch),
            get_activation(activation , ACTIVATION_MAP) 
        )        
    def forward(self , x):
        '''
        深度可分离卷积的前向传播函数

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过深度可分离卷积后生成的张量。
        '''
        return self.forward_basic(x)


@register_block('resnet_block_34')
class ResNetBlock_34(nn.Module):
    '''
    ResNet网络的基本模块组,在其34层以及18层使用的基本块儿

    Args:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        s (int): 卷积步长。
        activation_1 (str): 第一个激活函数名称。
        activation_2 (str): 第二个激活函数名称。

    Returns:
        torch.Tensor: 经过ResNet块后生成的张量。
    '''
    def __init__(
            self,
            in_ch:int,
            out_ch:int,
            s:int,
            activation_1='relu',
            activation_2='relu',
            ):
        '''
        初始化 ResNetBlock_34 模块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            s (int): 步长。
            activation_1 (str): 第一处激活函数名称。
            activation_2 (str): 残差相加后的激活函数名称。
        '''
        super(ResNetBlock_34, self).__init__()
        self.forward_basic_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                stride=s,
                padding=1,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_1 , ACTIVATION_MAP),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                stride=1,
                padding=1,
                kernel_size=3,
                bias=False
            ),
            nn.BatchNorm2d(out_ch),
        )
        self.downsample = None

        if s != 1 or in_ch != out_ch:
            # 如果输入输出的通道不匹配，或者是步长不唯1，那么就需要进行下采样的操作
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch,
                    stride= s,
                    kernel_size=1,
                    bias=False
                ),
                nn.BatchNorm2d(out_ch),
            )
        self.act = get_activation(activation_2 , ACTIVATION_MAP)
    def forward(self , x):
        '''
        ResNet_34的基本块儿前向传播的基本函数。

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过ResNet块后生成的张量。
        '''
        identity = x
        n , c , h , w = x.size()
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.forward_basic_1(x)
        out = out + identity
        out = self.act(out)
        return out


@register_block('resnet_block_50')
class ResNetBlock_50(nn.Module):
    '''
    ResNet网络的基本模块组,在其较大层使用的基本块儿

    Args:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        s (int): 卷积步长。
        activation_1 (str): 第一个激活函数名称。
        activation_2 (str): 第二个激活函数名称。
        activation_3 (str): 第三个激活函数名称。
        expansion_size (int): 扩展尺寸。

    Notes:
        其他具体运行过程请看Forward函数的说明

    '''
    def __init__(
            self,
            in_ch:int,
            out_ch:int,
            s:int,
            activation_1='relu',
            activation_2='relu',
            activation_3='relu',
            expansion_size = 4,
            ):
        '''
        初始化 ResNetBlock_50 模块。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 中间输出通道数。
            s (int): 主分支步长。
            activation_1 (str): 第一处激活函数名称。
            activation_2 (str): 第二处激活函数名称。
            activation_3 (str): 第三处激活函数名称。
            expansion_size (int): 输出通道扩张倍数。
        '''
        super().__init__()
        self.forward_basic = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=out_ch,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_1 , ACTIVATION_MAP),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch,
                kernel_size=3,
                stride = s,
                padding=1,
                bias = False
            ),
            nn.BatchNorm2d(out_ch),
            get_activation(activation_2 , ACTIVATION_MAP),
            nn.Conv2d(
                in_channels=out_ch,
                out_channels=out_ch * expansion_size,
                kernel_size=1,
                stride=1,
                padding=0,
                bias=False
            ),
            nn.BatchNorm2d(out_ch * expansion_size),
            get_activation(activation_3 , ACTIVATION_MAP)
        )
        self.downsample = None
        if s != 1 or in_ch != out_ch * expansion_size:
            self.downsample = nn.Sequential(
                nn.Conv2d(
                    in_channels=in_ch,
                    out_channels=out_ch * expansion_size,
                    kernel_size=1,
                    stride=s,
                    bias = False
                ),
                nn.BatchNorm2d(out_ch * expansion_size)
            )
    def forward(self , x):
        '''
        ResNet_50基本块的前向传播函数。

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过ResNet基本块后生成的张量。
        '''
        identity = x
        b , c , h , w = x.size()
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.forward_basic(x)
        x = x + identity
        return x












############################################################
#                                                          #
#                                                          #
#               下面全都是注意力机制相关的模块                  #
#                                                          #
#                                                          #
############################################################
@register_block('cbam_channel_attention')
class CBAM_Channel_Attention(nn.Module):
    '''
    CBAM注意力机制的子模块,通道注意力机制模块

    Args:
        in_ch (int): 输入通道数。
        reduction_ratio (int): 通道缩减比例。
        activation (str): 激活函数名称。

    Notes:
        其他的具体运行过程请看Forward函数的说明

    '''
    def __init__(
            self,
            in_ch:int,
            reduction_ratio:int,
            activation:str
        ):
        '''
        初始化 CBAM 通道注意力模块。

        Args:
            in_ch (int): 输入通道数。
            reduction_ratio (int): 通道压缩比例。
            activation (str): 激活函数名称。
        '''
        super(CBAM_Channel_Attention , self).__init__()
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.MLP_shared = nn.Sequential(
            nn.Linear(
                in_features = in_ch,
                out_features = in_ch//reduction_ratio,
                bias = False
            ),
            get_activation(activation, ACTIVATION_MAP),
            nn.Linear(
                in_features = in_ch//reduction_ratio,
                out_features = in_ch,
                bias = False
            ),
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self , x):
        '''
        CBAM注意力机制子模块,也就是通道注意力机制模块的前向传播函数
                  |-->平均池化->|
        input x --|            |->共享全连接层->相加->和x相乘
                  |-->最大池化->|

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过通道注意力机制后生成的张量。

        '''
        avg_weight = self.MLP_shared(self.avgpool(x).view(x.size(0) , x.size(1)))
        max_weight = self.MLP_shared(self.maxpool(x).view(x.size(0) , x.size(1)))
        attn = self.sigmoid(avg_weight + max_weight).view(x.size(0) , x.size(1) , 1 , 1)
        return attn * x


@register_block('cbam_spatial_attention')
class CBAM_Spatial_Attention(nn.Module):
    '''
    CBAM注意力机制的子模块,空间注意力机制模块

    Args:
        k(int) : 卷积核的大小

    Notes:  
        其他的请参见forward函数的具体说明。

    '''
    def __init__(
            self,
            k:int, 
            ):
        '''
        初始化 CBAM 空间注意力模块。

        Args:
            k (int): 卷积核大小。
        '''
        super(CBAM_Spatial_Attention , self).__init__()
        self.conv = nn.Conv2d(
            in_channels=2, 
            out_channels=1 , 
            kernel_size=k,
            padding=k//2,
            bias=False
        )
        self.sigmoid = nn.Sigmoid()
    
    def forward(self , x):
        '''
        CBAM空间注意力机制的前向传播函数

        Args:
            x (torch.Tensor): 输入张量,形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过空间注意力机制后生成的张量。
        '''
        avg_weight = torch.mean(x , dim= 1 , keepdim=True)
        max_weight , _ = torch.max(x , dim= 1 , keepdim=True)
        out = torch.cat([avg_weight , max_weight ] ,dim=1)
        out = self.conv(out)
        out = self.sigmoid(out)
        return out * x


@register_block('cbam')
class CBAM(nn.Module):
    '''
    CBAM注意力机制模块

    Args:
        in_ch (int) : 输入通道数量
        reduction_ratio (int) : 通道缩减比例
        activation (str) : 激活函数名称
        k (int) : 卷积核大小

    Returns:
        torch.Tensor: 经过CBAM注意力机制后生成的张量。
    '''
    def __init__(
            self,
            in_ch:int,
            reduction_ratio:int,
            activation:str,
            k:int,
            ):
        '''
        初始化 CBAM 模块。

        Args:
            in_ch (int): 输入通道数。
            reduction_ratio (int): 通道压缩比例。
            activation (str): 激活函数名称。
            k (int): 空间注意力卷积核大小。
        '''
        super(CBAM , self).__init__()
        self.channel_attention = CBAM_Channel_Attention(
            in_ch=in_ch,
            reduction_ratio=reduction_ratio,
            activation=activation
        )
        self.spatial_attention = CBAM_Spatial_Attention(
            k=k
        )
        
    def forward(self , x):
        '''
        CBAM 模块前向传播函数。

        Args:
            x (torch.Tensor): 输入张量, 形状通常为 (N, C, H, W)。

        Returns:
            torch.Tensor: 经过通道与空间注意力后的输出张量。
        '''
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


@register_block('flatten')
class Flatten(nn.Module):
    '''
    展平的模块

    Args:
        None
    
    Notes:
        具体请看Forward函数的说明和定义
    
    '''
    def __init__(
            self,
            ):
        '''
        初始化展平模块。
        '''
        super(Flatten , self).__init__()
        self.forward_basic = nn.Sequential(
            nn.Flatten()
        )

    def forward(self, x):
        '''
        展平模块的前向传播函数

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 展平后的张量。
        '''
        return self.forward_basic(x)


@register_block('linear')
class Linear(nn.Module):
    '''
    线性层的神经网络函数框架

    Args:
        in_feature (int): 输入特征数量
        out_feature (int): 输出特征数量
        bias (bool): 是否使用偏置

    Notes:
        具体请参见forward函数的具体说明。

    '''
    def __init__(
            self,
            in_feature:int,
            out_feature:int,
            bias:bool,
            ):
        '''
        初始化线性层模块。

        Args:
            in_feature (int): 输入特征维度。
            out_feature (int): 输出特征维度。
            bias (bool): 是否使用偏置。
        '''
        super(Linear, self).__init__()
        self.forwar_basic = nn.Sequential(
            nn.Linear(
                in_features=in_feature,
                out_features=out_feature,
                bias = bias
            )
        )
    def forward(self , x):
        '''
        线性层的前向传播函数

        Args:
            x (torch.Tensor): 输入张量。

        Returns:
            torch.Tensor: 输出张量。
        '''
        return self.forwar_basic(x)
        