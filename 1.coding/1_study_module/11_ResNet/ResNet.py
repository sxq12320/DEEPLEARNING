import torch
import torch.nn as nn

class ResNetBlock_34(nn.Module):
    '''ResNet在34层以及18层使用的基本块
        Args:
            in_channels:    输入特征图的通道数
            out_channels:   输出特征图的通道数
            stride:         卷积步幅,默认值为1
            padding:        卷积填充,默认值为1
        Returns:
            输出特征图
    '''
    def __init__(self , in_channels , out_channels , stride = 1 , padding = 1):
        super(ResNetBlock_34 ,self ).__init__()
        self.conv1 = nn.Conv2d(         
            in_channels=in_channels , 
            out_channels=out_channels , 
            stride=stride , 
            padding = padding , 
            kernel_size = 3 , 
            bias = False
            )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels , out_channels=out_channels , stride=1 , padding = 1 , kernel_size = 3 , bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            # 如果输入输出通道数不匹配，或者步幅不为1，则需要下采样
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels , out_channels=out_channels , stride=stride , kernel_size=1 , bias=False),#1x1卷积改变通道数和尺寸
                nn.BatchNorm2d(out_channels)
            )
    def forward(self , x):
        identity = x 
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        out = x + identity 
        out =  self.relu2(out)
        return out 

class ResNetBlock_50(nn.Module):
    '''ResNet在50层及以上使用的瓶颈块
        Args:
            in_channels:    输入特征图的通道数
            out_channels:   输出特征图的通道数
            stride:         卷积步幅
            padding:        卷积填充
        Returns:
            输出特征图
    '''
    def __init__(self , in_channels , out_channels , stride = 1 , padding = 0):
        super(ResNetBlock_50 ,self ).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels , out_channels=out_channels , stride=1 , padding = 0 , kernel_size = 1 , bias = False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=out_channels , out_channels=out_channels , stride=stride , padding = 1 , kernel_size = 3 , bias = False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_channels=out_channels , out_channels=out_channels * 4 , stride=1 , padding = 0 , kernel_size = 1 , bias = False)
        self.bn3 = nn.BatchNorm2d(out_channels *4)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_channels != out_channels * 4:
            # 如果输入输出通道数不匹配，或者步幅不为1，则需要下采样
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_channels , out_channels=out_channels * 4 , stride=stride , kernel_size=1 , bias=False),
                nn.BatchNorm2d(out_channels * 4)
            )
    def forward(self , x):
        identity = x 
        if self.downsample is not None:
            identity = self.downsample(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.bn3(x)
        out = x + identity 
        out =  self.relu3(out)
        return out
    


class ResNet_18(nn.Module):
    '''ResNet-18网络结构'''
    def __init__(self):
        super(ResNet_18, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock_34(64 , 64),
            ResNetBlock_34(64 , 64)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock_34(64 , 128 , stride=2 , padding=1),
            ResNetBlock_34(128 , 128)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock_34(128 , 256 , stride=2 , padding=1),
            ResNetBlock_34(256 , 256)
        )

        self.layer5 = nn.Sequential(
            ResNetBlock_34(256 , 512 , stride=2 , padding=1),
            ResNetBlock_34(512 , 512)
        )

        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512 , 1000 , bias=True)
        )
    def forward(self , x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    


class ResNet_34(nn.Module):
    '''ResNet-34网络结构'''
    def __init__(self):
        super(ResNet_34, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock_34(64 , 64),
            ResNetBlock_34(64 , 64),
            ResNetBlock_34(64 , 64)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock_34(64 , 128 , stride=2 , padding=1),
            ResNetBlock_34(128 , 128),
            ResNetBlock_34(128 , 128),
            ResNetBlock_34(128 , 128)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock_34(128 , 256 , stride=2 , padding=1),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256)
        )

        self.layer5 = nn.Sequential(
            ResNetBlock_34(256 , 512 , stride=2 , padding=1),
            ResNetBlock_34(512 , 512),
            ResNetBlock_34(512 , 512)
        )

        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(512 , 1000 , bias=True)
        )
    def forward(self , x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    

class ResNet_CIFAR10_34(nn.Module):
    '''ResNet-34网络结构 for CIFAR-10'''
    def __init__(self):
        super(ResNet_CIFAR10_34, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=3 , stride=1 , padding=1 , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock_34(64 , 64),
            ResNetBlock_34(64 , 64),
            ResNetBlock_34(64 , 64)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock_34(64 , 128 , stride=2 , padding=1),
            ResNetBlock_34(128 , 128),
            ResNetBlock_34(128 , 128),
            ResNetBlock_34(128 , 128)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock_34(128 , 256 , stride=2 , padding=1),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256),
            ResNetBlock_34(256 , 256)
        )
        self.layer5 = nn.Sequential(
            ResNetBlock_34(256 , 512 , stride=2 , padding=1),
            ResNetBlock_34(512 , 512),
            ResNetBlock_34(512 , 512)
        )   
        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),    
            nn.Flatten(),
            nn.Linear(512 , 10 , bias=True)
        )

    def forward(self , x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x

class ResNet_50(nn.Module):
    '''ResNet-50网络结构'''
    def __init__(self):
        super(ResNet_50, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock_50(64 , 64),
            ResNetBlock_50(256 , 64),
            ResNetBlock_50(256 , 64)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock_50(256 , 128 , stride=2 , padding=1),
            ResNetBlock_50(512 , 128),
            ResNetBlock_50(512 , 128),
            ResNetBlock_50(512 , 128)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock_50(512 , 256 , stride=2 , padding=1),
            ResNetBlock_50(1024 , 256),
            ResNetBlock_50(1024 , 256),
            ResNetBlock_50(1024 , 256),
            ResNetBlock_50(1024 , 256),
            ResNetBlock_50(1024 , 256)
        )

        self.layer5 = nn.Sequential(
            ResNetBlock_50(1024 , 512 , stride=2 , padding=1),
            ResNetBlock_50(2048 , 512),
            ResNetBlock_50(2048 , 512)
        )

        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048 , 1000 , bias=True)
        )
    def forward(self , x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    


class ResNet_101(nn.Module):
    '''ResNet-101网络结构'''
    def __init__(self):
        super(ResNet_101, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock_50(64 , 64),
            ResNetBlock_50(256 , 64),
            ResNetBlock_50(256 , 64)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock_50(256 , 128 , stride=2 , padding=1),
            ResNetBlock_50(512 , 128),
            ResNetBlock_50(512 , 128),
            ResNetBlock_50(512 , 128)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock_50(512 , 256 , stride=2 , padding=1),
            *[ResNetBlock_50(1024 , 256) for _ in range(22)]
        )

        self.layer5 = nn.Sequential(
            ResNetBlock_50(1024 , 512 , stride=2 , padding=1),
            ResNetBlock_50(2048 , 512),
            ResNetBlock_50(2048 , 512)
        )

        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048 , 1000 , bias=True)
        )
    def forward(self , x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x
    


class ResNet_152(nn.Module):
    '''ResNet-152网络结构'''
    def __init__(self):
        super(ResNet_152, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        )

        self.layer2 = nn.Sequential(
            ResNetBlock_50(64 , 64),
            ResNetBlock_50(256 , 64),
            ResNetBlock_50(256 , 64)
        )

        self.layer3 = nn.Sequential(
            ResNetBlock_50(256 , 128 , stride=2 , padding=1),
            ResNetBlock_50(512 , 128),
            ResNetBlock_50(512 , 128),
            ResNetBlock_50(512 , 128)
        )

        self.layer4 = nn.Sequential(
            ResNetBlock_50(512 , 256 , stride=2 , padding=1),
            *[ResNetBlock_50(1024 , 256) for _ in range(35)]
        )

        self.layer5 = nn.Sequential(
            ResNetBlock_50(1024 , 512 , stride=2 , padding=1),
            ResNetBlock_50(2048 , 512),
            ResNetBlock_50(2048 , 512)
        )

        self.layer6 = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(2048 , 1000 , bias=True)
        )
    def forward(self , x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        return x