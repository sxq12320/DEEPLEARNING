import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
from PIL import Image
import warnings
import time
import math


class VegetableFruitHybridNet(nn.Module):
    """针对蔬菜水果分类的混合注意力网络（针对112×112优化）"""

    def __init__(self, num_classes=36, attention_type='eca', dropout_rate=0.3):
        super(VegetableFruitHybridNet, self).__init__()

        # 初始卷积层（适应112×112输入）
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)  # 112×112 -> 56×56
        )

        # 注意力类型选择
        self.attention_type = attention_type

        # 第一阶段：56×56
        self.layer1 = nn.Sequential(
            LightHybridBlock(64, 64, 1, attention_type),
            LightHybridBlock(64, 64, 1, attention_type),
            nn.Dropout2d(dropout_rate / 2)
        )

        # 第二阶段：28×28
        self.layer2 = nn.Sequential(
            LightHybridBlock(64, 128, 2, attention_type),  # 下采样
            LightHybridBlock(128, 128, 1, attention_type),
            nn.Dropout2d(dropout_rate)
        )

        # 第三阶段：14×14
        self.layer3 = nn.Sequential(
            LightHybridBlock(128, 256, 2, attention_type),  # 下采样
            LightHybridBlock(256, 256, 1, attention_type),
            LightHybridBlock(256, 256, 1, attention_type),
            nn.Dropout2d(dropout_rate)
        )

        # 第四阶段：7×7
        self.layer4 = nn.Sequential(
            LightHybridBlock(256, 512, 2, attention_type),  # 下采样
            LightHybridBlock(512, 512, 1, attention_type),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(256, num_classes)
        )

        # 初始化权重
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # 特征提取
        x = self.stem(x)  # [B, 3, 112, 112] -> [B, 64, 56, 56]
        x = self.layer1(x)  # [B, 64, 56, 56] -> [B, 64, 56, 56]
        x = self.layer2(x)  # [B, 64, 56, 56] -> [B, 128, 28, 28]
        x = self.layer3(x)  # [B, 128, 28, 28] -> [B, 256, 14, 14]
        x = self.layer4(x)  # [B, 256, 14, 14] -> [B, 512, 1, 1]

        # 分类
        x = x.view(x.size(0), -1)  # 展平
        x = self.classifier(x)  # 分类

        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation注意力模块"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAMBlock(nn.Module):
    """CBAM注意力模块（通道+空间注意力）"""

    def __init__(self, channels, reduction=16):
        super(CBAMBlock, self).__init__()

        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        ca = self.channel_attention(x)
        x = x * ca

        # 空间注意力
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        sa_input = torch.cat([max_pool, avg_pool], dim=1)
        sa = self.spatial_attention(sa_input)

        return x * sa


class EfficientChannelAttention(nn.Module):
    """高效通道注意力（ECA）模块"""

    def __init__(self, channels, gamma=2, b=1):
        super(EfficientChannelAttention, self).__init__()
        t = int(abs((math.log(channels, 2) + b) / gamma))
        kernel_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=kernel_size,
                              padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv(y.squeeze(-1).transpose(-1, -2))
        y = y.transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class LightHybridBlock(nn.Module):
    """轻量级混合块（ResNet风格 + 注意力）"""

    def __init__(self, in_channels, out_channels, stride=1, attention_type='eca'):
        super(LightHybridBlock, self).__init__()

        # 残差连接
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 注意力机制
        if attention_type == 'se':
            self.attention = SEBlock(out_channels)
        elif attention_type == 'cbam':
            self.attention = CBAMBlock(out_channels)
        elif attention_type == 'eca':
            self.attention = EfficientChannelAttention(out_channels)
        else:
            self.attention = None

        # 下采样
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        # 应用注意力机制
        if self.attention is not None:
            out = self.attention(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


def safe_loader(path):
    '''安全地加载图像文件'''
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGBA').convert('RGB')


def get_stats(dataset):
    '''计算数据集的均值和标准差'''
    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4
    )

    mean = torch.zeros(3)
    mean_square = torch.zeros(3)
    total_pixels = 0

    for images, _ in dataloader:
        batch_size = images.size(0)
        height = images.size(2)
        width = images.size(3)
        pixels_per_image = height * width

        for i in range(3):
            channel_data = images[:, i, :, :]
            mean[i] += channel_data.sum()
            mean_square[i] += (channel_data ** 2).sum()

        total_pixels += batch_size * pixels_per_image

    mean = mean / total_pixels
    variance = (mean_square / total_pixels) - (mean ** 2)
    std = torch.sqrt(torch.clamp(variance, min=1e-6))

    return mean, std


def init_dataset():
    '''数据集的预处理'''
    warnings.filterwarnings("ignore", category=UserWarning)
    train_data_dir = r'E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\train'
    test_data_dir = r'E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\val'

    # 为计算统计量定义转换
    data_transforms_for_stats = transforms.Compose([
        transforms.Resize(128),  # 缩放到128×128
        transforms.CenterCrop(112),  # 中心裁剪到112×112
        transforms.ToTensor()  # 转换为张量[0,1]
    ])

    train_dataset_for_stats = datasets.ImageFolder(
        root=train_data_dir,
        transform=data_transforms_for_stats,
        loader=safe_loader
    )

    # 计算训练集的均值和标准差
    print("开始计算训练集的均值和标准差...")
    mean_train, std_train = get_stats(train_dataset_for_stats)

    # 定义训练和测试的转换（增强只在训练时使用）
    train_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(112),  # 随机裁剪增强
        transforms.RandomHorizontalFlip(p=0.5),  # 随机水平翻转
        transforms.RandomRotation(15),  # 随机旋转
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train.tolist(), std=std_train.tolist())
    ])

    test_transforms = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),  # 测试时只中心裁剪
        transforms.ToTensor(),
        transforms.Normalize(mean=mean_train.tolist(), std=std_train.tolist())
    ])

    # 加载数据集
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

    # 获取类别数量
    num_classes = len(full_train_dataset.classes)

    # 创建数据加载器
    train_loader = DataLoader(full_train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(full_test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)

    print(f"\n预处理完成！")
    print(f"训练集均值: {mean_train.tolist()}")
    print(f"训练集标准差: {std_train.tolist()}")
    print(f"训练集大小: {len(full_train_dataset)}")
    print(f"测试集大小: {len(full_test_dataset)}")
    print(f"训练批次数量: {len(train_loader)}")
    print(f"测试批次数量: {len(test_loader)}")
    print(f"类别数量: {num_classes}")
    print(f"类别名称: {full_train_dataset.classes}")

    return train_loader, test_loader, num_classes, full_train_dataset.classes


def train_epoch(model, device, train_loader, criterion, optimizer, epoch, num_epochs):
    """训练一个epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

        if batch_idx % 20 == 0:
            print(
                f'Epoch [{epoch + 1}/{num_epochs}], Step [{batch_idx + 1}/{len(train_loader)}], Loss: {loss.item():.4f}')

    train_loss = running_loss / len(train_loader)
    train_acc = 100. * correct / total

    return train_loss, train_acc


def evaluate(model, device, test_loader, criterion):
    """评估模型"""
    model.eval()
    test_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)

            test_loss += loss.item()
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    test_loss = test_loss / len(test_loader)
    test_acc = 100. * correct / total

    return test_loss, test_acc


def compute_confusion_matrix(model, device, test_loader, num_classes):
    """计算混淆矩阵"""
    model.eval()
    confusion_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)

            for t, p in zip(target.view(-1), predicted.view(-1)):
                confusion_matrix[t, p] += 1

    return confusion_matrix.numpy()


def plot_confusion_matrix(conf_matrix, class_names, accuracy):
    """绘制混淆矩阵"""
    num_classes = len(class_names)

    # 创建图形
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # 绘制混淆矩阵热力图
    im = ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    ax1.figure.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)

    # 设置坐标轴
    ax1.set(xticks=np.arange(num_classes),
            yticks=np.arange(num_classes),
            xticklabels=class_names,
            yticklabels=class_names,
            title=f'混淆矩阵 (总体准确率: {accuracy:.2f}%)',
            ylabel='真实类别',
            xlabel='预测类别')

    # 旋转x轴标签
    plt.setp(ax1.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # 在每个单元格中显示数字
    thresh = conf_matrix.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            ax1.text(j, i, format(conf_matrix[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if conf_matrix[i, j] > thresh else "black")

    # 计算每个类别的准确率
    class_accuracies = []
    for i in range(num_classes):
        correct = conf_matrix[i, i]
        total = np.sum(conf_matrix[i, :])
        if total > 0:
            acc = correct / total
        else:
            acc = 0.0
        class_accuracies.append(acc)

    # 绘制每个类别的准确率柱状图
    ax2.bar(range(num_classes), class_accuracies, color='skyblue', alpha=0.7)
    ax2.axhline(y=accuracy / 100, color='red', linestyle='--', label=f'总体准确率: {accuracy:.2f}%')
    ax2.set_xlabel('类别')
    ax2.set_ylabel('准确率')
    ax2.set_title('每个类别的准确率')
    ax2.set_xticks(range(num_classes))
    ax2.set_xticklabels(class_names, rotation=45, ha='right')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # 调整布局
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=150, bbox_inches='tight')
    plt.show()

    # 打印详细统计信息
    print(f"\n总体准确率: {accuracy:.2f}%")
    print("\n每个类别的准确率:")
    for i, class_name in enumerate(class_names):
        print(f"{class_name}: {class_accuracies[i] * 100:.2f}%")

    return class_accuracies


def plot_training_history(train_losses, train_accs, test_losses, test_accs):
    """绘制训练历史"""
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(14, 5))

    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Train Loss', linewidth=2)
    plt.plot(epochs, test_losses, 'r-', label='Test Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accs, 'b-', label='Train Accuracy', linewidth=2)
    plt.plot(epochs, test_accs, 'r-', label='Test Accuracy', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    # 初始化数据集
    print("初始化数据集...")
    train_loader, test_loader, num_classes, class_names = init_dataset()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用设备: {device}")

    # 创建模型
    print(f"\n创建混合注意力网络，类别数: {num_classes}")
    model = VegetableFruitHybridNet(
        num_classes=num_classes,  # 使用实际类别数
        attention_type='eca',  # 可以改为 'se' 或 'cbam'
        dropout_rate=0.3
    ).to(device)

    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型总参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)  # 标签平滑防止过拟合
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)

    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )

    # 训练记录
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    best_test_acc = 0.0

    # 训练参数
    num_epochs = 100
    print(f"\n开始训练，共 {num_epochs} 个epoch...")

    for epoch in range(num_epochs):
        start_time = time.time()

        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, device, train_loader, criterion, optimizer, epoch, num_epochs
        )

        # 评估
        test_loss, test_acc = evaluate(model, device, test_loader, criterion)

        # 更新学习率
        scheduler.step()

        # 记录结果
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        # 保存最佳模型
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'best_test_acc': best_test_acc,
                'class_names': class_names,
                'num_classes': num_classes,
            }, 'best_model.pth')
            print(f"保存最佳模型，测试准确率: {test_acc:.2f}%")

        end_time = time.time()
        epoch_time = end_time - start_time

        # 打印epoch结果
        print(f"\nEpoch [{epoch + 1:03d}/{num_epochs}] | Time: {epoch_time:.1f}s")
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Test  Loss: {test_loss:.4f} | Test  Acc: {test_acc:.2f}%")
        print(f"Best Test Acc: {best_test_acc:.2f}%")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print("-" * 60)

        # 提前停止条件（如果连续10个epoch测试准确率没有提升）
        if epoch > 20 and test_acc < max(test_accuracies[-10:]):
            print(f"连续10个epoch测试准确率没有提升，提前停止训练")
            break

    # 训练结束
    print("\n训练结束！")
    print(f"最佳测试准确率: {best_test_acc:.2f}%")

    # 绘制训练历史
    plot_training_history(train_losses, train_accuracies, test_losses, test_accuracies)

    # 加载最佳模型并最终评估
    print("\n加载最佳模型进行最终评估...")
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])

    final_test_loss, final_test_acc = evaluate(model, device, test_loader, criterion)
    print(f"最终测试结果 - Loss: {final_test_loss:.4f}, Acc: {final_test_acc:.2f}%")

    # 计算并绘制混淆矩阵
    print("\n计算混淆矩阵...")
    conf_matrix = compute_confusion_matrix(model, device, test_loader, num_classes)
    plot_confusion_matrix(conf_matrix, class_names, final_test_acc)


if __name__ == '__main__':
    main()