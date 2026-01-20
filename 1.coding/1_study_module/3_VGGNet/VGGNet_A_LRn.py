import torch
import torch.nn as nn

class VGGNetALRn(nn.Module):
    '''VGGNetA模型定义，添加LRN层
    '''
    def __init__(self , num_classes = 10):
        super(VGGNetALRn , self).__init__()
        # input size = [3 , 224 , 224]
        self.conv1 = nn.Conv2d(3 , 64 , kernel_size = 3 , padding= 1)
        self.relu1 = nn.ReLU()
        self.lrn1 = nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2 , stride=2)

        # input size = [64 , 112 , 112]
        self.conv2 = nn.Conv2d(64 , 128 , kernel_size = 3 , padding= 1)
        self.relu2 = nn.ReLU()
        self.maxpool2 = nn.MaxPool2d(kernel_size=2 , stride=2)

        # input size = [128 , 56 , 56]
        self.conv3 = nn.Conv2d(128 , 256 , kernel_size = 3 , padding= 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(256 , 256 , kernel_size = 3 , padding= 1)
        self.relu4 = nn.ReLU()
        self.maxpool3 = nn.MaxPool2d(kernel_size=2 , stride=2)

        # input size = [256 , 28 , 28]
        self.conv5 = nn.Conv2d(256 , 512 , kernel_size = 3 , padding= 1)
        self.relu5 = nn.ReLU()
        self.conv6 = nn.Conv2d(512 , 512 , kernel_size = 3 , padding= 1)
        self.relu6 = nn.ReLU()
        self.maxpool4 = nn.MaxPool2d(kernel_size=2 , stride=2)  

        # input size = [512 , 14 , 14]
        self.conv7 = nn.Conv2d(512 , 512 , kernel_size = 3 , padding= 1)
        self.relu7 = nn.ReLU()
        self.conv8 = nn.Conv2d(512 , 512 , kernel_size = 3 , padding= 1)
        self.relu8 = nn.ReLU()
        self.maxpool5 = nn.MaxPool2d(kernel_size=2 , stride=2)

        # input size = [512 , 7 , 7]
        self.fc1 = nn.Linear(in_features=25088 ,out_features=4096 , bias=True)
        self.relu9 = nn.ReLU()
        self.fc2 = nn.Linear(in_features=4096 , out_features=4096 , bias=True)
        self.relu10 = nn.ReLU()
        self.fc3 = nn.Linear(in_features=4096 , out_features=1000 , bias= True)

    def forward(self , x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.lrn1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.maxpool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.maxpool3(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.maxpool4(x)
        x = self.conv7(x)
        x = self.relu7(x)
        x = self.conv8(x)
        x = self.relu8(x)
        x = self.maxpool5(x)
        x = x.view(x.size(0) , -1)
        x = self.fc1(x)
        x = self.relu9(x)
        x = self.fc2(x)
        x = self.relu10(x)
        x = self.fc3(x)
        return x
    
class VGGNetALRn_CIFAR10(nn.Module):
    '''VGGNetA模型定义，适用于CIFAR-10'''
    def __init__(self, num_classes=10):
        super(VGGNetALRn_CIFAR10, self).__init__()
        
        # 输入尺寸 [3, 32, 32]
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2.0),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 512),  # CIFAR-10图像经过卷积层后，尺寸变为[batch_size, 512, 1, 1]
            nn.ReLU(inplace=True),
            nn.Linear(512, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),  # CIFAR-10有10个类别
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.classifier(x)
        return x
    

