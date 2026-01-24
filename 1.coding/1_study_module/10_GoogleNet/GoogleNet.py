import torch 
import torch.nn as nn

class InceptionBlock(nn.Module):
    '''Googlenet 的 Inception 模块
    Args:
        in_channels: 输入的通道数
    Returns: 
    '''
    def __init__(self , in_channels , out_1x1 ,  out_3x3_reduce , out_3x3 , out_5x5_reduce , out_5x5 , out_pool_proj):
        super(InceptionBlock, self).__init__()
        '''inception 模块包含四个并行的分支：
            Args:
                in_channels: 输入的通道数
                out_1x1: 1x1 卷积层输出的通道数
                out_3x3_reduce: 3x3 卷积层前的降维 1x1 卷积层输出的通道数
                out_3x3: 3x3 卷积层输出的通道数
                out_5x5_reduce: 5x5 卷积层前的降维 1x1 卷积层输出的通道数
                out_5x5: 5x5 卷积层输出的通道数
                out_pool_proj: 池化层后 1x1 卷积层输出的通道数 
        '''

        # 1x1 卷积层
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels , out_channels = out_1x1 , kernel_size = 1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # 3x3 卷积层
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels , out_channels = out_3x3_reduce , kernel_size=1), # 用1x1 卷积进行降维
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_3x3_reduce , out_channels = out_3x3 , kernel_size=3 , padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 5x5 卷积层
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels = in_channels , out_channels = out_5x5_reduce , kernel_size=1), # 用1x1 卷积进行降维
            nn.BatchNorm2d(out_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels = out_5x5_reduce , out_channels = out_5x5 , kernel_size=5 , padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )

        # 最大池化层
        self.poolbranch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3 , stride=1 , padding=1),
            nn.Conv2d(in_channels = in_channels , out_channels = out_pool_proj , kernel_size=1),
            nn.BatchNorm2d(out_pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self , x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branchpool = self.poolbranch(x)

        out = torch.cat([branch1x1 , branch3x3 , branch5x5 , branchpool], dim=1) # (batch_size, channels, height, width) ,channels维度拼接

        return out

class GoogleNet(nn.Module):
    def __init__(self , num_classes=1000):
        super(GoogleNet , self).__init__()
        '''Googlenet 网络结构
        '''
        self.googleNet = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = 64 , kernel_size = 7 , stride = 2 , padding = 3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            nn.Conv2d(in_channels = 64 , out_channels = 192 , kernel_size = 3 , stride = 1 , padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            InceptionBlock(192 , 64 , 96 , 128 , 16 , 32 , 32),#3a
            InceptionBlock(256 , 128 , 128 , 192 , 32 , 96 , 64),#3b
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            InceptionBlock(480 , 192 , 96 , 208 , 16 , 48 , 64),#4a
            InceptionBlock(512 , 160 , 112 , 224 , 24 , 64 , 64),#4b
            InceptionBlock(512 , 128 , 128 , 256 , 24 , 64 , 64),#4c
            InceptionBlock(512 , 112 , 144 , 288 , 32 , 64 , 64),#4d
            InceptionBlock(528 , 256 , 160 , 320 , 32 , 128 , 128),#4e
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            InceptionBlock(832 , 256 , 160 , 320 , 32 , 128 , 128),#5a
            InceptionBlock(832 , 384 , 192 , 384 , 48 , 128 , 128),#5b

            nn.AdaptiveAvgPool2d((1,1)),    
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(1024 , num_classes)
        )
    def forward(self , x):
        return self.googleNet(x)
    

class GoogleNet_CIFAR10(nn.Module):
    def __init__(self , num_classes=10):
        super(GoogleNet_CIFAR10 , self).__init__()
        '''Googlenet 网络结构 for CIFAR10
        '''
        self.googleNet = nn.Sequential(
            nn.Conv2d(in_channels = 3 , out_channels = 64 , kernel_size = 3 , stride = 1 , padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            nn.Conv2d(in_channels = 64 , out_channels = 192 , kernel_size = 3 , stride = 1 , padding = 1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            InceptionBlock(192 , 64 , 96 , 128 , 16 , 32 , 32),#3a
            InceptionBlock(256 , 128 , 128 , 192 , 32 , 96 , 64),#3b
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            InceptionBlock(480 , 192 , 96 , 208 , 16 , 48 , 64),#4a
            InceptionBlock(512 , 160 , 112 , 224 , 24 , 64 , 64),#4b
            InceptionBlock(512 , 128 , 128 , 256 , 24 , 64 , 64),#4c
            InceptionBlock(512 , 112 , 144 , 288 , 32 , 64 , 64),#4d
            InceptionBlock(528 , 256 , 160 , 320 , 32 , 128 , 128),#4e
            nn.MaxPool2d(kernel_size=3 , stride=2 , padding=1),

            InceptionBlock(832 , 256 , 160 , 320 , 32 , 128 , 128),#5a
            InceptionBlock(832 , 384 , 192 , 384 , 48 , 128 , 128),#5b

            nn.AdaptiveAvgPool2d((1 , 1)),
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(1024 , num_classes)
        )
    def forward(self , x):
        return self.googleNet(x)

# # 测试代码
# if __name__ == "__main__":
#     # 创建模型
#     model = GoogleNet(num_classes=1000)
    
#     # 测试输入
#     x = torch.randn(4, 3, 32, 32)  # 4张32x32的RGB图像
#     output = model(x)
    
#     print(f"输入形状: {x.shape}")
#     print(f"输出形状: {output.shape}")
#     print(f"预期输出: (4, 1000)")
    
#     # 参数统计
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"\n总参数数量: {total_params:,}")
#     print(f"可训练参数数量: {trainable_params:,}")
        
        
