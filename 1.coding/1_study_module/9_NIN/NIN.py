import torch
import torch.nn as nn

class NIN(nn.Module):
    def __init__(self):
        super(NIN , self).__init__()

        self.NIN_net = nn.Sequential(
            self.NINBlock(3 , 96 , 11 , 4 , 0),
            nn.MaxPool2d(kernel_size=3 , stride=2),
            self.NINBlock(96 , 256 , 5 , 1 , 2),
            nn.MaxPool2d(kernel_size=3 , stride=2),
            self.NINBlock(256 , 384 , 3 , 1 , 1),
            nn.MaxPool2d(kernel_size=3 , stride=2),
            nn.Dropout(0.5),
            self.NINBlock(384 , 10 , 3 , 1 , 1),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten()
        )
        
    def NINBlock(self, in_channels , out_channels , kernel_size , stride , padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels , out_channels=out_channels , kernel_size=kernel_size , stride=stride , padding=padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels , out_channels=out_channels , kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=out_channels , out_channels=out_channels , kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self , x):
        return self.NIN_net(x)
    

class NIN_CIFAR10(nn.Module):
    def __init__(self):
        super(NIN_CIFAR10, self).__init__()
        
        self.net = nn.Sequential(
            self.NINBlock(3, 192, 5, 1, 2),  # 32x32 -> 32x32
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 32x32 -> 16x16
            
            self.NINBlock(192, 160, 1, 1, 0),  # 16x16 -> 16x16
            self.NINBlock(160, 96, 1, 1, 0),  # 16x16 -> 16x16
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 16x16 -> 8x8
            
            self.NINBlock(96, 192, 5, 1, 2),  # 8x8 -> 8x8
            nn.Dropout(0.5),
            
            self.NINBlock(192, 192, 1, 1, 0),  # 8x8 -> 8x8
            self.NINBlock(192, 10, 1, 1, 0),  # 8x8 -> 8x8
            
            nn.AdaptiveAvgPool2d((1,1)),  # 8x8 -> 1x1
            nn.Flatten()
        )
    
    def NINBlock(self , in_channels, out_channels, kernel_size, stride, padding):
        return nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                     kernel_size=kernel_size, stride=stride, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.net(x)