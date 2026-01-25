import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision 
from torchvision import transforms, datasets
import torch.optim as optim
from tqdm import tqdm
from Axeon import Axeon_Block 

# 假设这是你集成后的模型
class AxeonLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AxeonLeNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, kernel_size=5), # CIFAR10是3通道
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(12, 16, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        # 经过两次5x5卷积和池化，32x32会变成 5x5
        self.axeon_head = Axeon_Block(in_channels=16, num_classes=num_classes, feature_size=5)

    def forward(self, x):
        x = self.features(x)
        preds, prob_matrix = self.axeon_head(x)
        return preds, prob_matrix

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # 1. 加载并过滤数据集 (仅保留前3类: airplane, automobile, bird)
    full_train = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    full_test = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    
    train_idx = [i for i, label in enumerate(full_train.targets) if label < 3]
    test_idx = [i for i, label in enumerate(full_test.targets) if label < 3]
    
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_train, train_idx), batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(full_test, test_idx), batch_size=64, shuffle=False)
    
    print(f"使用设备: {device} | 训练样本数: {len(train_idx)}")

    # 2. 初始化 Axeon 版 LeNet
    model = AxeonLeNet(num_classes=3).to(device)
    # 注意：我们要对 prob_matrix 的均值求 CrossEntropy
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    epochs = 30 # 几何判定收敛通常较快
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    for epoch in range(epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [Axeon 训练]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # preds 是几何判定的索引，prob_matrix 是 C x N 的概率矩阵
            preds, prob_matrix = model(images)
            
            # 训练逻辑：计算所有通道投票的平均概率与标签的损失
            # prob_matrix: [B, C, N] -> mean: [B, N]
            avg_probs = torch.mean(prob_matrix, dim=1)
            loss = criterion(avg_probs, labels)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item() # 使用几何判定结果算准确率
            
            pbar.set_postfix(loss=loss.item(), acc=100*correct/total)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        test_loss, correct, total = 0.0, 0, 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds, prob_matrix = model(images)
                
                avg_probs = torch.mean(prob_matrix, dim=1)
                loss = criterion(avg_probs, labels)
                
                test_loss += loss.item()
                total += labels.size(0)
                correct += (preds == labels).sum().item() # 核心：使用回归直线的几何判定
        
        test_loss /= len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}, Train Acc: {train_acc:.2f}%, Test Acc: {test_acc:.2f}%')

    # 绘制曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Axeon Geometric Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(test_accs, label='Test Acc')
    plt.title('Axeon Geometric Accuracy')
    plt.legend()
    plt.savefig('axeon_results.png')
    plt.show()
    
    return model

if __name__ == "__main__":
    trained_model = train_model()