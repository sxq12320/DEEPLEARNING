import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from fontTools.varLib.instancer import setMacOverlapFlags
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import gzip
import sys
from PIL import Image
import warnings
import time

from torchvision.models import ResNet


class LeNet_5(nn.Module):
    '''LeNet-5 网路架构的实现
        Args：
            num_classer : 分类的类别数量，36个
    '''
    def __init__(self , num_classes =36):
        super(LeNet_5, self).__init__()
        self.conv1 = nn.Conv2d(3 , 6 , kernel_size=5 , stride = 1 , padding = 0)        # I:(3,112,112) -> O:(6,108,108)
        self.pool1 = nn.MaxPool2d(kernel_size = 2 , stride = 2 )                                              # I:(6,108,108) -> O:(6,54,54)
        self.conv2 = nn.Conv2d(6 , 16 , kernel_size=5 , stride = 1 , padding = 0)       # I:(6,54,54) -> O:(16,50,50)
        self.pool2 = nn.MaxPool2d(kernel_size = 2 , stride = 2 )                                             # I:(16,50,50) -> O:(16,25,25)
        self.fc1 = nn.Linear(16*25*25 , 120)
        self.fc2 = nn.Linear(120 , 84)
        self.fc3 = nn.Linear(84 , num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)           # 第一层卷积
        x = self.relu(x)
        x = self.pool1(x)

        x = self.conv2(x)           # 第二层卷积
        x = self.relu(x)
        x = self.pool2(x)

        x = x.view(-1 , 16*25*25)   # 展平操作，将特征图拉成一维的向量，引入全连接层
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x


def safe_loader(path):
    '''安全地加载图像文件，特别是处理一些可能损坏的图像文件或者确保图像被正确转换为RGB格式。
        Args:
            path 文件的位置
        Returns:
            PIL.Image对象，格式为RGB
    '''
    with open(path, 'rb') as f:                     # 以二进制方式打开图像
        img = Image.open(f)                         # 用PIL来打开图像
        return img.convert('RGBA').convert('RGB')   # 双重转换


def get_stats(dataset):
    ''' 计算数据集的均值和标准差（每个通道单独计算）
        Args:
            dataset PyTorch的Dataset对象
        Returns:
            mean 每个通道的均值
            std 每个通道的标准差
    '''
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    mean = torch.zeros(3)               # 累加每个通道的总和
    mean_square = torch.zeros(3)        # 累加每个通道的平方和
    total_pixels = 0                    # 记录总像素数据

    for images, _ in dataloader:            # 忽略标签同时遍历图像
        batch_size = images.size(0)         # 当前的批次大小
        height = images.size(2)             # 图像的高度
        width = images.size(3)              # 图像的宽度
        pixels_per_image = height * width   # 图像的总像素大小

        for i in range(3):                                  # 对RGB三个通道分别处理
            channel_data = images[:, i, :, :]               # 提取当前通道
            mean[i] += channel_data.sum()                   # 累加像素值综合
            mean_square[i] += (channel_data ** 2).sum()     # 累加像素值的平方和

        total_pixels += batch_size * pixels_per_image

    mean = mean / total_pixels                                  # 均值 = 总和 / 总像素数
    variance = (mean_square / total_pixels) - (mean ** 2)       # 方差 = E(X²) - [E(X)]²
    std = torch.sqrt(torch.clamp(variance, min=1e-6))           # 标准差 = sqrt(方差)，clamp防止数值问题

    return mean, std

def init_dataset():
    '''数据集的预处理，包括了数据加载，统计量的求解，标准化，数据加载器的创建等
    '''
    warnings.filterwarnings("ignore", category=UserWarning)                                                 #忽略警告
    train_data_dir = r'E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\train'    # 索引训练集的位置
    test_data_dir = r'E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\val'       # 索引测试集的位置

    data_transforms_for_stats = transforms.Compose([
        transforms.Resize(256),         # 统一缩放到256x256
        transforms.CenterCrop(224),     # 沿着中心裁剪到224x224
        transforms.ToTensor()           # 转换成张量[0,1]的范围
    ])

    train_dataset_for_stats = datasets.ImageFolder(
        root=train_data_dir,
        transform=data_transforms_for_stats,
        loader=safe_loader                      # 使用自定义的安全加载器
    )

    # 计算训练集的均值和标准差
    print("开始计算训练集的均值和标准差...")
    mean_train, std_train = get_stats(train_dataset_for_stats)

    # 定义训练和测试的转换，训练集使用计算得到的统计量进行标准化
    train_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train.tolist(), std=std_train.tolist())
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train.tolist(), std=std_train.tolist())  # 使用训练集的统计量
    ])

    # 使用标准化转换重新加载数据集
    full_train_dataset = datasets.ImageFolder(
        root=train_data_dir,
        transform=train_transforms,
        loader=safe_loader
    )
    full_test_dataset = datasets.ImageFolder(
        root=test_data_dir,
        transform=test_transforms,
        loader=safe_loader
    )

    # 创建数据加载器
    train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = DataLoader(full_test_dataset, batch_size=64, shuffle=True, num_workers=4)

    print(f"\n预处理完成！")
    print(f"训练集均值: {mean_train.tolist()}")
    print(f"训练集标准差: {std_train.tolist()}")
    print(f"训练批次数量: {len(train_loader)}")
    print(f"测试批次数量: {len(test_loader)}")

    return train_loader, test_loader


if __name__ == '__main__':
    train_loader, test_loader = init_dataset()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    model = LeNet_5(num_classes=36).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    TRAIN_LOSS = []
    TEST_LOSS = []
    TRAIN_ACC = []
    TEST_ACC = []

    epochs = 50
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        correct_test = 0
        total_train = 0
        total_test = 0

        start_time = time.time()
        for i, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            print(f'Epoch [{epoch + 1}/{epochs}], Step [{i + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        end_time = time.time()
        epoch_loss = running_loss / len(train_loader)
        print(f"Epoch {epoch+1} 结束 | 平均Loss: {epoch_loss:.4f} | 耗时: {end_time - start_time:.2f}秒")
        train_loss_epoch = running_loss / len(train_loader)
        train_acc_epoch = 100 * correct_train / total_train

        TRAIN_LOSS.append(train_loss_epoch)
        TRAIN_ACC.append(train_acc_epoch)

        model.eval()  # 切换到评估模式 (关闭Dropout等)
        running_test_loss = 0.0
        correct_test = 0
        total_test = 0

        with torch.no_grad():  # 不计算梯度，节省显存
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)

                running_test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()

        # 计算本 Epoch 的平均测试损失和准确率
        test_loss_epoch = running_test_loss / len(test_loader)
        test_acc_epoch = 100 * correct_test / total_test

        TEST_LOSS.append(test_loss_epoch)
        TEST_ACC.append(test_acc_epoch)

        end_time = time.time()

        print(f"Epoch {epoch + 1}/{epochs} 结束 | 耗时: {end_time - start_time:.2f}s")
        print(f"Train Loss: {train_loss_epoch:.4f} | Train Acc: {train_acc_epoch:.2f}%")
        print(f"Test  Loss: {test_loss_epoch:.4f} | Test  Acc: {test_acc_epoch:.2f}%")
        print("-" * 60)

    print("训练结束，开始绘图...")

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(TRAIN_LOSS, label='Train Loss', color='blue')
    plt.plot(TEST_LOSS, label='Test Loss', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(TRAIN_ACC, label='Train Acc', color='blue')
    plt.plot(TEST_ACC, label='Test Acc', color='red', linestyle='--')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)

    plt.show()