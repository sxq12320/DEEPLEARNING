import torch.nn as nn
import torch 
import torch.nn.functional as F
from models.Block import ResNetBlock_34,DepthWise_Conv,DepthWiseSeparable_Conv
from configs.config import ACTIVATION_MAP



class ResNet_18(nn.Module):
    def __init__(self):
        super(ResNet_18 , self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 64 , kernel_size=7 , stride=2 , padding=3 , bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1)
        )
        self.layer2 = nn.Sequential(
            ResNetBlock_34(64 , 64 , 1 , 'relu' , 'relu'),
            ResNetBlock_34(64 , 64 , 1 , 'relu' , 'relu')
        )

        