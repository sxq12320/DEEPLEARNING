import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torchvision 
from torchvision import transforms, datasets
import torch.optim as optim
import time
from tqdm import tqdm
import os
from torch.cuda.amp import autocast, GradScaler

class AlexNet(nn.Module):
    '''AlexNet 神经网络模块，这里就不再分两个GPU训练了，直接写在单个GPU上面进行训练
    '''
    def __init__(self , num_classes= 20):
        super(AlexNet , self).__init__()
        # input size = 224*224*3
        self.Conv1 = nn.Conv2d(in_channels = 3 , out_channels = 96 , kernel_size = 11 , stride = 4 , padding = 2)
        self.Relu1 = nn.ReLU(inplace=True)
        self.maxpool1 = nn.MaxPool2d(kernel_size=3 , stride = 2) 

        # input size = 27*27*96
        self.Conv2 = nn.Conv2d(in_channels = 96 , out_channels = 256 , kernel_size=5 , stride=1 , padding = 2)
        self.Relu2 = nn.ReLU(inplace=True)
        self.maxpool2 = nn.MaxPool2d(kernel_size = 3 , stride = 2)

        # input size = 13*13*256
        self.Conv3 = nn.Conv2d(in_channels = 256 , out_channels = 192*2 , kernel_size= 3 , stride = 1 , padding = 1)
        self.Relu3 = nn.ReLU(inplace=True)

        # input size = 13*13*384
        self.Conv4 = nn.Conv2d(in_channels = 384 , out_channels=384 , kernel_size = 3 , stride=1 , padding=1)
        self.Relu4 = nn.ReLU(inplace=True)

        # input size = 13*13*384
        self.Conv5 = nn.Conv2d(in_channels = 384 , out_channels =256 , kernel_size = 3 , stride = 1 , padding = 1)
        self.Relu5 = nn.ReLU(inplace=True)
        self.maxpool3 = nn.MaxPool2d(kernel_size = 3 , stride = 2)

        self.avgpool = nn.AdaptiveAvgPool2d((6,6))


        # input size = 6*6*256
        self.dropout1 = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(in_features = 256*6*6 , out_features=2048 , bias = True)
        self.Relu6 = nn.ReLU(inplace=True)

        # input size = 2048
        self.fc2 = nn.Linear(in_features=2048 , out_features=2048 , bias = True) 
        self.Relu7 = nn.ReLU(inplace=True)

        self.dropout2 = nn.Dropout(p=0.5)

        # input size = 2048
        self.fc3 = nn.Linear(in_features=2048 , out_features=num_classes , bias = True)
        self.Softmax = nn.Softmax(dim=1)

    def forward(self , x):
        x = self.Conv1(x)
        x = self.Relu1(x)
        x = self.maxpool1(x)

        x = self.Conv2(x)
        x = self.Relu2(x)
        x = self.maxpool2(x)

        x = self.Conv3(x)
        x = self.Relu3(x)

        x = self.Conv4(x)
        x = self.Relu4(x)

        x = self.Conv5(x)
        x = self.Relu5(x)
        x = self.maxpool3(x)

        x = self.avgpool(x)

        x = self.dropout1(x)

        x = x.view(x.size(0) , -1)

        x = self.fc1(x)
        x = self.Relu6(x)

        x = self.fc2(x)
        x = self.Relu7(x)

        x = self.dropout2(x)

        x = self.fc3(x)

        return x
    


class CIFAR_AlexNet(nn.Module):
    '''专为CIFAR-10优化的轻量级AlexNet，速度提升5-10倍'''
    def __init__(self, num_classes=10):
        super(CIFAR_AlexNet, self).__init__()
        # 为32x32输入优化
        self.features = nn.Sequential(
            # 输入: [batch, 3, 32, 32]
            nn.Conv2d(3, 64, kernel_size=5, stride=1, padding=2),  # 减小kernel_size
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [batch, 64, 16, 16]
            
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [batch, 192, 8, 8]
            
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # -> [batch, 384, 8, 8]
            
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),  # -> [batch, 256, 8, 8]
            
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # -> [batch, 256, 4, 4]
        )
        
        # 自适应池化到固定大小
        self.avgpool = nn.AdaptiveAvgPool2d((4, 4))  # 4x4 而不是 6x6
        
        # 大幅简化的全连接层
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 4 * 4, 512),  # 4096 -> 512
            nn.ReLU(inplace=True),
            
            nn.Dropout(0.5),
            nn.Linear(512, 256),  # 4096 -> 256
            nn.ReLU(inplace=True),
            
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x



def train_model():
    # 数据预处理 (使用32x32原始尺寸)
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

    # 加载数据集
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)

    # 数据加载器
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 初始化模型
    model = CIFAR_AlexNet(num_classes=10).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练参数
    epochs = 10
    train_losses, test_losses = [], []
    train_accs, test_accs = [], []

    # 训练循环
    for epoch in range(epochs):
        # 训练阶段
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # 带进度条的训练
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs} [训练]')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            # 统计
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # 更新进度条
            pbar.set_postfix(loss=loss.item(), acc=100*correct/total)
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # 验证阶段
        model.eval()
        test_loss = 0.0
        correct = 0
        total = 0
        
        # 带进度条的验证
        pbar = tqdm(test_loader, desc=f'Epoch {epoch+1}/{epochs} [验证]')
        with torch.no_grad():
            for images, labels in pbar:
                images, labels = images.to(device), labels.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix(loss=loss.item(), acc=100*correct/total)
        
        test_loss = test_loss / len(test_loader)
        test_acc = 100 * correct / total
        test_losses.append(test_loss)
        test_accs.append(test_acc)
        
        print(f'Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, '
              f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

    # 3. 绘制图片
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, 'b-', label='训练损失')
    plt.plot(test_losses, 'r-', label='测试损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('损失曲线')
    plt.legend()
    
    # 准确率曲线
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, 'b-', label='训练准确率')
    plt.plot(test_accs, 'r-', label='测试准确率')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title('准确率曲线')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()
    
    return model

if __name__ == "__main__":
    trained_model = train_model()