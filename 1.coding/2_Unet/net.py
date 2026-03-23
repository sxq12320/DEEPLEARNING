import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ConvBlock(nn.Module):
    '''
    using:
        基础卷积模块实现卷积操作，3×3的卷积操作
    Args:
        in_channel (int) : 输入通道数量
        out_channel (int) : 输出通道数量
    Returns:
        layer
    '''
    def __init__(self ,  in_channel , out_channel):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channel , out_channel , kernel_size = 3 , stride = 1 , padding = 1 , padding_mode='reflect' , bias = False ),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, padding_mode='reflect',bias=False),
            nn.BatchNorm2d(out_channel),
            nn.Dropout(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampleBlock(nn.Module):
    '''
    using:
        下采样操作，之前使用的是最大池化，不合适这里使用3×3卷积来替换掉原本的最大池化操作
    Args:
        channel (int) : 通道数量
    Returns:
        layer
    '''
    def __init__(self , channel):
        super(DownSampleBlock , self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel , channel , kernel_size = 3 , stride = 2 , padding = 1 , padding_mode='reflect' , bias = False ),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.layer(x)


class UpSampleBlock(nn.Module):
    '''
    using:
        上采样,同时进行之前的拼接操作
    Args:
        channel (int) : 上采样的输入通道数
        feature_map (tensor) : 需要拼接的特征图
    Returns:
        layer
    '''

    def __init__(self ,  channel):
        super(UpSampleBlock , self).__init__()
        self.layer = nn.Conv2d(channel , channel //2 , 1 , 1 )
    def forward(self , x , feature_map):
        up = F.interpolate(x , scale_factor = 2 , mode = 'nearest') # 直接变大两倍结束
        out = self.layer(up)
        return torch.cat((out , feature_map),dim = 1)

class unet(nn.Module):
    def __init__(self):
        super(unet, self).__init__()
        self.c1 = ConvBlock(3,64)
        self.d1 = DownSampleBlock(64)
        self.c2 = ConvBlock(64,128)
        self.d2 = DownSampleBlock(128)
        self.c3 = ConvBlock(128, 256)
        self.d3 = DownSampleBlock(256)
        self.c4 = ConvBlock(256, 512)
        self.d4 = DownSampleBlock(512)
        self.c5 = ConvBlock(512, 1024)

        self.u1 = UpSampleBlock(1024)
        self.c6 = ConvBlock(1024, 512)
        self.u2 = UpSampleBlock(512)
        self.c7 = ConvBlock(512, 256)
        self.u3 = UpSampleBlock(256)
        self.c8 = ConvBlock(256, 128)
        self.u4 = UpSampleBlock(128)
        self.c9 = ConvBlock(128, 64)
        self.out = nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0)

    def forward(self , x):
        c1 = self.c1(x)
        d1 = self.d1(c1)
        c2 = self.c2(d1)
        d2 = self.d2(c2)
        c3 = self.c3(d2)
        d3 = self.d3(c3)
        c4 = self.c4(d3)
        d4 = self.d4(c4)
        c5 = self.c5(d4)
        u1 = self.u1(c5,c4)
        c6 = self.c6(u1)
        u2 = self.u2(c6,c3)
        c7 = self.c7(u2)
        u3 = self.u3(c7,c2)
        c8 = self.c8(u3)
        u4 = self.u4(c8,c1)
        c9 = self.c9(u4)
        out  = self.out(c9)
        return out

if __name__ == '__main__':
    net = unet()
    x = torch.randn(2, 3 , 512 , 512)
    out = net(x)
    print(out.shape)
