import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import gzip
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import warnings

class MobileNetV1_Block(nn.Module):
    def __init__(self , num_classes = 10 , in_channels = 3 , out_channels = 64 ,stride = 1):
        super(MobileNetV1_Block, self).__init__()
        # 深度可分离卷积 depthwise separable convolution
        self.conv1 = nn.Conv2d(
            in_channels=in_channels , 
            out_channels=in_channels , 
            kernel_size=3 , 
            stride=stride , 
            padding=1 , 
            groups=in_channels , 
            bias=False
            )
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 逐点卷积 pointwise convolution
        self.conv2 = nn.Conv2d(in_channels = in_channels , out_channels = out_channels , kernel_size=1 , stride=1 , padding=0 , bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self , x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x
    
class MobileNetV1(nn.Module):
    def __init__(self , num_classes = 10):
    
        super(MobileNetV1 , self).__init__()
        # 第一层卷积层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3 , 32 , kernel_size=3 , stride=1 , padding=1 , bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 第二层深度可分离卷积层
        self.layer2 = nn.Sequential(
            MobileNetV1_Block(in_channels = 32 , out_channels=64 , stride=1)
        )
        # 第三层深度可分离卷积层
        self.layer3 = nn.Sequential(
            MobileNetV1_Block(in_channels = 64 , out_channels=128 , stride=2)
        )
        # 第四层深度可分离卷积层
        self.layer4 = nn.Sequential(
            MobileNetV1_Block(in_channels = 128 , out_channels=128 , stride=1)
        )
        # 第五层深度可分离卷积层
        self.layer5 = nn.Sequential(
            MobileNetV1_Block(in_channels = 128 , out_channels=256 , stride=2)
        )
        # 第六层深度可分离卷积层
        self.layer6 = nn.Sequential(
            MobileNetV1_Block(in_channels = 256 , out_channels=256 , stride=1)
        )
        # 第七层深度可分离卷积层
        self.layer7 = nn.Sequential(
            MobileNetV1_Block(in_channels = 256 , out_channels=512 , stride=2)
        )
        # 后续五个深度可分离卷积层
        self.layer8 = nn.Sequential(
            MobileNetV1_Block(in_channels = 512 , out_channels=512 , stride=1),
            MobileNetV1_Block(in_channels = 512 , out_channels=512 , stride=1),
            MobileNetV1_Block(in_channels = 512 , out_channels=512 , stride=1),
            MobileNetV1_Block(in_channels = 512 , out_channels=512 , stride=1),
            MobileNetV1_Block(in_channels = 512 , out_channels=512 , stride=1)
        )
        # 第十三层深度可分离卷积层
        self.layer9 = nn.Sequential(
            MobileNetV1_Block(in_channels = 512 , out_channels=1024 , stride=2)
        )
        # 第十四层深度可分离卷积层
        self.layer10 = nn.Sequential(
            MobileNetV1_Block(in_channels = 1024 , out_channels=1024 , stride=2)
        )
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Linear(1024 , num_classes)

    def forward(self , x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.avgpool(x)
        x = x.view(-1 , 1024)
        x = self.fc(x)
        return x

        
def safe_loader(path):
    # 打开图片
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            # 关键步骤：
            # 1. convert('RGBA'): 先转成 4 通道，正确处理透明度（避免警告）
            # 2. convert('RGB'): 再转回 3 通道（丢弃透明通道，背景通常变白或黑）
            #    这样能保证所有图片进入网络时都是 (3, H, W)
            return img.convert('RGBA').convert('RGB')


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    Train_DIR = r"E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\train"
    Test_DIR = r"E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\val"

    Batch_size = 64
    Epochs = 10

    TRAIN_LOSS_HIST = []
    TEST_LOSS_HIST = []
    TRAIN_ACC_HIST = []
    TEST_ACC_HIST = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(Train_DIR):
        print(f"错误：找不到训练集路径 {Train_DIR}，请检查路径配置！")
        exit()

    if not os.path.exists(Test_DIR):
        print(f"错误：找不到测试集路径 {Test_DIR}，请检查路径配置！")
        exit()

    train_dataset = datasets.ImageFolder(root=Train_DIR, transform=transform , loader=safe_loader)
    test_dataset  = datasets.ImageFolder(root=Test_DIR, transform=transform , loader=safe_loader)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"检测到 {num_classes} 个分类: {class_names}")

    # trian_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
    # test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

    train_dataloader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True , num_workers=2)
    test_dataloader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=False , num_workers=2)

    model = MobileNetV1(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(Epochs):
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0
            
            # --- 训练循环 ---
            train_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{Epochs}] Train", leave=False)
            for i , (images , labels) in enumerate(train_bar):
                images = images.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs , labels)
                loss.backward()
                optimizer.step()

                # 累积统计
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total_train += labels.size(0)
                correct_train += (predicted == labels).sum().item()
                train_bar.set_postfix(loss=loss.item(), acc=100 * correct_train / total_train)

            # 计算 Epoch 平均值
            epoch_loss = running_loss / len(train_dataloader)
            epoch_acc = 100 * correct_train / total_train
            TRAIN_LOSS_HIST.append(epoch_loss)
            TRAIN_ACC_HIST.append(epoch_acc)

            
            model.eval()
            test_loss = 0.0
            correct_test = 0
            total_test = 0
            test_bar = tqdm(test_dataloader, desc=f"Epoch [{epoch+1}/{Epochs}] Test ", leave=False)
            with torch.no_grad():
                for images, labels in test_bar:
                    images = images.to(device)
                    labels = labels.to(device)

                    outputs = model(images)
                    loss = criterion(outputs, labels) 
                    test_loss += loss.item()
                    
                    _, predicted = torch.max(outputs.data, 1)
                    total_test += labels.size(0)
                    correct_test += (predicted == labels).sum().item()

            avg_test_loss = test_loss / len(test_dataloader)
            avg_test_acc = 100 * correct_test / total_test
            TEST_LOSS_HIST.append(avg_test_loss)
            TEST_ACC_HIST.append(avg_test_acc)

            # 打印本 Epoch 最终结果
            print(f"Epoch [{epoch+1}/{Epochs}] "
                f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
                f"Test Loss: {avg_test_loss:.4f} Acc: {avg_test_acc:.2f}%")

    # 绘图
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(TRAIN_LOSS_HIST , label = 'Train Loss')
    plt.plot(TEST_LOSS_HIST , label = 'Test Loss')
    plt.title('Loss')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot(TRAIN_ACC_HIST , label = 'Train Accuracy')
    plt.plot(TEST_ACC_HIST , label = 'Test Accuracy')
    plt.title('Accuracy')
    plt.legend()
    plt.show()