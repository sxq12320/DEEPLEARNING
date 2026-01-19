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


class LeNet5(nn.Module):
    def __init__(self , num_classes = 10):
        """
        LeNet-5 网络结构的实现
        
        Args:
            num_classes (int): 分类的类别数,默认值为10
        
        """
        super(LeNet5 , self).__init__()

        self.conv1 = nn.Conv2d(3 , 6 , kernel_size=5 ,stride = 1 , padding = 0)
        # 卷积层1
        self.pool1 = nn.AvgPool2d(kernel_size = 2 , stride=2)
        # 平均池化层1
        self.conv2 = nn.Conv2d(6, 16 ,kernel_size=5 , stride=1 , padding=0)
        # 卷积层2
        self.pool2 = nn.AvgPool2d(kernel_size=2 , stride=2)
        # 平均池化层2
        self.fc1 = nn.Linear(16*5*5 , 120)
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
        x = x.view(-1 , 16*5*5)# 展平操作,变成全连接层
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

# 加载CIFAR10数据集
train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transforms.ToTensor(), download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transforms.ToTensor())

# 定义数据加载器
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=64, shuffle=False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 定义模型、损失函数和优化器
model = LeNet5().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

TEST_LOSS = []
TRAIN_LOSS = []
TEST_ACC = []
TRAIN_ACC = []

# 训练模型
for epoch in range(10):
    model.train()
    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)
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
            images = images.to(device)
            labels = labels.to(device)
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
plt.title('LeNet-5 on MNIST')
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

 