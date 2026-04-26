import torch.nn as nn
import torch
import torch.nn.functional as F
from config import ACTIVATION_MAP
from utils.Block_function import get_activation,autopad
import numpy as np


############################################################
#                                                          #
#                                                          #
#               下面全都是基本卷积相关的模块                    #
#                                                          #
#                                                          #
############################################################
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













############################################################
#                                                          #
#                                                          #
#               下面全都是注意力机制相关的模块                  #
#                                                          #
#                                                          #
############################################################
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
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
    



