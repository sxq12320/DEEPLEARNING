import torch 
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import gzip
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class LeNet5(nn.Module):
    def __init__(self , num_classes = 10):
        """LeNet-5 网络结构的实现
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
        x = x.view(-1 , 16*4*4)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x


class SEBlock(nn.Module):
    '''SEBlock: Squeeze-and-Excitation Block 注意力机制模块
        Args:
            out_channels:   输入特征图的通道数
            reduction:      压缩比例,默认值为4
    '''
    def __init__(self ,out_channels , reduction = 4):
        super(SEBlock, self).__init__()
        self.global_avg_pool  = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(out_channels, out_channels // reduction , bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(out_channels // reduction , out_channels , bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self , x):
        b , c , h , w = x.size()                    # 获取输入特征的尺寸，b:batch_size,c:channels,h:height,w:width

        y  = self.global_avg_pool(x).view(b , c)    # 第一步：全局平均池化，得到每个通道的全局特征Sequence，[b, c, 1, 1] -> [b, c]

        y = self.fc1(y)
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(b , c , 1 , 1)     # 第二步：通过两个全连接层和激活函数，得到每个通道的权重,先降低维度后上升维度[b,c] -> [b,c,1,1]

        out = x * y                                 # 第三步：将权重与原始特征图相乘，进行通道的重新校准        

        return out

class SE_LENET(nn.Module):
    def __init__(self):
        super(SE_LENET , self).__init__()
        self.conv1 = nn.Conv2d(1 , 6 , kernel_size=5 ,stride = 1 , padding = 0)
        self.Se1 = SEBlock(out_channels=6 , reduction=2)
        self.pool1 = nn.MaxPool2d(kernel_size = 2 , stride=2)

        self.conv2 = nn.Conv2d(6, 16 ,kernel_size=5 , stride=1 , padding=0)
        self.Se2 = SEBlock(out_channels=16 , reduction=4)
        self.pool2 = nn.MaxPool2d(kernel_size=2 , stride=2)

        self.fc1 = nn.Linear(16*4*4 , 120)
        # 全连接层1
        self.fc2 = nn.Linear(120 , 84)
        # 全连接层2
        self.fc3 = nn.Linear(84 , 10)
        # 全连接层3
        self.relu = nn.ReLU()
        
    def forward(self , x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.Se1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.Se2(x)
        x = self.pool2(x)

        x = x.view(-1 , 16*4*4)# 展平操作,变成全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x
    


# 加载MNIST数据集
train_dataset = datasets.MNIST(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.MNIST(root='./data', train=False, transform=transforms.ToTensor())

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)

# 定义模型、损失函数和优化器
model = LeNet5()
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