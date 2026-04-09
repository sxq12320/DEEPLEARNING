import torch 
import torch.nn as nn
import torch.nn.functional as F




############################################################
#                                                          #
#                                                          #
#   首先先写一下MobileNetV3的网络架构,用来充当门控的网络GRUNet    #
#                                                          #
#                                                          #
############################################################
class HSwish(nn.Module):
    '''
    using:
        定义HSwish激活函数
    Args:
        x : 输入的矩阵图片
    Returns:
        经过HSwish激活函数处理后的结果
    '''
    def __init__(self):
        super(HSwish , self).__init__()
    def forward(self , x):
        return x * F.relu6(x + 3) / 6


class SEModule(nn.Module):
    '''
    using :
        定义SE注意力机制模块
    Args :
        x : 输入的矩阵图片
        in_channel : 输入通道数量
        reduction  : 缩减率
    Returns :
        经过SE注意力机制处理后的结果
    '''
    def __init__(self , in_channel , reduction=4):
        super(SEModule , self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channel , in_channel // reduction , bias = False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channel // reduction , in_channel , bias = False),
            nn.Hardsigmoid()
        )

    def forward(self , x):
        b , c , w , h = x.size()

        y = self.avg_pool(x).view(b , c)
        y = self.fc(y).view(b , c , 1 , 1 )
        
        return x * y







class MobileNetV3(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV3, self).__init__()
        

    def forward(self, x):
