import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

class CBAMBlock(nn.Module):
    '''CBAM注意力机制模块
        包含通道注意力机制模块以及空间注意力机制模块
        Args:
            outchannels: 输入特征图的通道数
            reduction: 通道注意力机制模块中全连接层的缩减比例
    '''
    def __init__(self , outchannels , reduction = 4):
        super(CBAMBlock , self).__init__()
        # 通道注意力机制模块 channel attention module
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(outchannels , outchannels // reduction , bias = False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(outchannels // reduction , outchannels , bias = False)
        self.sigmoid = nn.Sigmoid()

        # 空间注意力机制模块 spatial attention module
        self.conv = nn.Conv2d(in_channels=2 , out_channels = 1 , kernel_size = 7 , stride = 1 , padding = 3 , bias = False)

    def forward(self , x):
        '''
            前向传播函数
            1. 通道注意力机制模块
                a. 分别进行最大池化以及平均池化
                b. 通过全连接层进行特征提取(分别)
                c. 将两个特征直接相加，然后sigmoid归一化
            2. 空间注意力机制模块
                a. 对通道注意力机制模块的输出进行最大池化以及平均池化
                b. 将两个池化结果在通道维度上进行拼接
                c. 通过卷积层进行特征提取，然后sigmoid归一化
        '''
        # 通道注意力机制模块
        maxpool_out = self.maxpool(x)
        avgpool_out = self.avgpool(x)

        maxpool_out = maxpool_out.view(-1 , x.size(1)) 
        avgpool_out = avgpool_out.view(-1 , x.size(1))

        F_max = self.fc2(self.relu(self.fc1(maxpool_out)))       
        F_avg = self.fc2(self.relu(self.fc1(avgpool_out)))

        channel_attention_out = self.sigmoid(F_max+F_avg).unsqueeze(2).unsqueeze(3)  # [B , C , 1 , 1] , add操作
        F_1 = channel_attention_out * x

        # 空间注意力机制模块
        spatial_maxpool , _ = torch.max(F_1 , dim=1 , keepdim = True)                       # [B , 1 , H , W]
        spatial_avgpool = torch.mean(F_1 , dim=1 , keepdim = True)                          # [B , 1 , H , W]
        spatial_attention_out = torch.cat([spatial_maxpool , spatial_avgpool] , dim = 1)    # [B , 2 , H , W] , concat操作
        spatial_attention_out = self.sigmoid(self.conv(spatial_attention_out))              # [B , 1 , H , W]
        F_2 = spatial_attention_out * F_1

        return F_2
    

class LeNet5(nn.Module):
    def __init__(self , num_classes = 10):
        """
        LeNet-5 网络结构的实现
        
        Args:
            num_classes (int): 分类的类别数,默认值为10
        
        """
        super(LeNet5 , self).__init__()

        self.conv1 = nn.Conv2d(1 , 6 , kernel_size=5 ,stride = 1 , padding = 0)
        # 卷积层1
        self.pool1 = nn.AvgPool2d(kernel_size = 2 , stride=2)
        # 平均池化层1
        self.conv2 = nn.Conv2d(6, 16 ,kernel_size=5 , stride=1 , padding=0)
        # 卷积层2
        self.pool2 = nn.AvgPool2d(kernel_size=2 , stride=2)
        # 平均池化层2
        self.fc1 = nn.Linear(16*4*4 , 120)
        # 全连接层1
        self.fc2 = nn.Linear(120 , 84)
        # 全连接层2
        self.fc3 = nn.Linear(84 , num_classes)
        # 全连接层3
        self.relu = nn.ReLU()
        # 激活函数

    def forward(self , x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.pool2(x)
        x = x.view(-1 , 16*4*4)# 展平操作,变成全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    

class CBAMlenet(nn.Module):
    def __init__(self , num_classes = 10):
        super(CBAMlenet , self).__init__()
        self.lenet = LeNet5()
        self.cbam1 = CBAMBlock(outchannels = 6)
        self.cbam2 = CBAMBlock(outchannels = 16)

    def forward(self , x):
        x = self.lenet.conv1(x)
        x = self.cbam1(x)
        x = self.lenet.relu(x)
        x = self.lenet.pool1(x)

        x = self.lenet.conv2(x)
        x = self.cbam2(x)
        x = self.lenet.relu(x)
        x = self.lenet.pool2(x)

        x = x.view(-1 , 16*4*4)# 展平操作,变成全连接层
        x = self.lenet.fc1(x)
        x = self.lenet.relu(x)
        x = self.lenet.fc2(x)
        x = self.lenet.relu(x)
        x = self.lenet.fc3(x)
        return x
    
if __name__ == "__main__":
        # 加载MNIST数据集
    train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
    test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

    # 定义数据加载器
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

    # 定义模型、损失函数和优化器
    model = CBAMlenet()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters() , lr = 0.001)

    TEST_LOSS = []
    TRAIN_LOSS = []
    TEST_ACC = []
    TRAIN_ACC = []

    # 训练模型
    for epoch in range(10):
        model.train()
        for i, (images, labels) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

        # 记录训练损失和准确度
        TRAIN_LOSS.append(loss.item())
        TRAIN_ACC.append(100 * (outputs.argmax(1) == labels).sum().item() / labels.size(0))
        
        # 测试模式
        model.eval()
        with torch.no_grad():
            correct = 0
            total = 0
            test_loss = 0
            for images, labels in test_loader:
                outputs = model(images)
                test_loss += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            TEST_LOSS.append(test_loss / len(test_loader))
            TEST_ACC.append(100 * correct / total)

            print('Epoch [{}/{}], Test Accuracy: {:.2f}%, Test Loss: {:.4f}'
                .format(epoch+1, 10, TEST_ACC[-1], TEST_LOSS[-1]))


    plt.figure(figsize=(12,5))
    plt.title('SE_LENET on MNIST')
    plt.subplot(1,2,1)
    plt.plot(TRAIN_LOSS , label = 'Train Loss')
    plt.plot(TEST_LOSS , label = 'Test Loss')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(TRAIN_ACC , label = 'Train Accuracy')
    plt.plot(TEST_ACC , label = 'Test Accuracy')
    plt.xlabel('Iterations')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
