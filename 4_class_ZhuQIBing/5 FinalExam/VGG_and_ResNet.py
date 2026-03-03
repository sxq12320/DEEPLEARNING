import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

'''
以resnet为主干，并使用densenet进行连接，同时增加了注意力机制模块RESDENSEBLOCK
'''

class SEBlock(nn.Module):
    '''
    SE注意力机制模块
    '''
    def __init__(self, in_channels, out_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels // reduction , bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels // reduction, out_channels, bias = False),
            nn.Sigmoid()
        )
    def forward(self , x):
        b,c,_,_ = x.size()
        y = self.avg_pool(x)
        y = self.fc(y).view(b,c,1 , 1)
        return y*x.expand_as(x)


class CBAMBlock(nn.Module):
    """CBAM注意力模块（通道+空间注意力）"""

    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)

        return x * sa


class MultiScaleAttention(nn.Module):
    """多尺度注意力模块"""

    def __init__(self, channels):
        super(MultiScaleAttention, self).__init__()

        self.branch1 = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True)
        )

        self.attention = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)

        combined = torch.cat([b1, b2, b3], dim=1)
        attention = self.attention(combined)

        return x * attention







