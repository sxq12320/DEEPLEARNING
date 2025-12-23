import torch.nn as nn
import torch

class BasicBlock(nn.Module):#定义一个block类继承至nn.Module，使用resnet18以及resnet34的残差结构
    expansion = 1 #指定扩张因子为1 ， 主分支的卷积核的个数不改变
    
    def __init__(self , in_channel , out_channel , stride =1 , downsample = None):
        super(BasicBlock , self).__init__()
        #传入输入，输出通道数，卷积核大小为3
        self.conv1 = nn.Conv2d(in_channels=in_channel , out_channels=out_channel , kernel_size=3 , padding=1 , stride=stride , bias=False)
        self,bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()

        self.conv2 = nn.Conv2d(in_channels=out_channel , out_channels=out_channel , kernel_size=3 , padding=1 , stride=stride , bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.downsample = downsample

    def forward(self , x):
        identity = x # 保存输入数据便于后续进行残差连接
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)

        return out

class bottleneck(nn.Module):
    expansion = 4

    def __init__(self , in_channel , out_channel , stride = 1  , downsample = None):
        super(bottleneck , self).__init__()
        self.conv1 =nn.Conv2d(in_channels=in_channel , out_channels=out_channel , kernel_size=1 , stride=1 , padding=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(in_channels=out_channel , out_channels=out_channel , kernel_size=3 , stride=1 , padding=1,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.conv3 = nn.Conv2d(in_channels=out_channel , out_channels=out_channel*self.expansion , kernel_size=1 , stride=1 , padding=1 , bias=False)
        self.bn3 =nn.BatchNorm2d(out_channel * self.expansio1n) 
        self.relu = nn.ReLU()
        self.downsample = downsample # 下采样层

    def forward(self,x):
        identity = x 
        if self.downsample is None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        out = out+identity
        out = self.relu(out)
        return out
