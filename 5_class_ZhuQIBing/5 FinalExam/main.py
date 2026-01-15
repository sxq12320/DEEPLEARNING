import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
import warnings
import csv
import time

# ============================================
# 1. 定义 CBAM 注意力机制模块
# ============================================
class CBAMBlock(nn.Module):
    '''CBAM注意力机制模块
       包含通道注意力机制模块以及空间注意力机制模块
       
    '''
    def __init__(self, outchannels, reduction=16):
        super(CBAMBlock, self).__init__()
        
        # 确保 hidden_channels 至少为 1，防止报错
        hidden_channels = max(1, outchannels // reduction)

        # 通道注意力机制 (Channel Attention)
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # 共享全连接层 (使用 Conv2d 实现 1x1 卷积，效果等同于 Linear 但无需 reshape)
        self.shared_mlp = nn.Sequential(
            nn.Conv2d(outchannels, hidden_channels, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, outchannels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

        # 空间注意力机制 (Spatial Attention)
        # kernel_size=7, padding=3 保证尺寸不变
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        # --- 1. 通道注意力 ---
        # x: [B, C, H, W]
        max_out = self.shared_mlp(self.maxpool(x)) # [B, C, 1, 1]
        avg_out = self.shared_mlp(self.avgpool(x)) # [B, C, 1, 1]
        
        channel_out = self.sigmoid(max_out + avg_out)
        x = x * channel_out # 广播机制相乘

        # --- 2. 空间注意力 ---
        # 在通道维度(dim=1)上求最大和平均
        max_out, _ = torch.max(x, dim=1, keepdim=True) # [B, 1, H, W]
        avg_out = torch.mean(x, dim=1, keepdim=True)   # [B, 1, H, W]
        
        spatial_out = torch.cat([max_out, avg_out], dim=1) # [B, 2, H, W]
        spatial_out = self.sigmoid(self.conv(spatial_out)) # [B, 1, H, W]
        
        x = x * spatial_out
        return x

# ============================================
# 2. 修改 MobileNetV1 Block 以集成 CBAM
# ============================================
class MobileNetV1_Block(nn.Module):
    def __init__(self, num_classes=10, in_channels=3, out_channels=64, stride=1, use_cbam=True):
        super(MobileNetV1_Block, self).__init__()
        self.use_cbam = use_cbam

        # 深度可分离卷积 depthwise separable convolution
        # 1. Depthwise
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)

        # 2. Pointwise
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # 3. CBAM 模块 (新增)
        if self.use_cbam:
            self.cbam = CBAMBlock(outchannels=out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        # 插入 CBAM 注意力机制
        if self.use_cbam:
            x = self.cbam(x)
            
        return x

# ============================================
# 3. MobileNet 网络结构
# ============================================
class MobileNetV1(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1, self).__init__()
        # 第一层卷积层
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        # 堆叠 Depthwise Blocks (默认 use_cbam=True)
        self.layer2 = nn.Sequential(MobileNetV1_Block(in_channels=32, out_channels=64, stride=1))
        self.layer3 = nn.Sequential(MobileNetV1_Block(in_channels=64, out_channels=128, stride=2))
        self.layer4 = nn.Sequential(MobileNetV1_Block(in_channels=128, out_channels=128, stride=1))
        self.layer5 = nn.Sequential(MobileNetV1_Block(in_channels=128, out_channels=256, stride=2))
        self.layer6 = nn.Sequential(MobileNetV1_Block(in_channels=256, out_channels=256, stride=1))
        self.layer7 = nn.Sequential(MobileNetV1_Block(in_channels=256, out_channels=512, stride=2))
        
        # 后续五个深度可分离卷积层
        self.layer8 = nn.Sequential(
            MobileNetV1_Block(in_channels=512, out_channels=512, stride=1),
            MobileNetV1_Block(in_channels=512, out_channels=512, stride=1),
            MobileNetV1_Block(in_channels=512, out_channels=512, stride=1),
            MobileNetV1_Block(in_channels=512, out_channels=512, stride=1),
            MobileNetV1_Block(in_channels=512, out_channels=512, stride=1)
        )
        
        self.layer9 = nn.Sequential(MobileNetV1_Block(in_channels=512, out_channels=1024, stride=2))
        self.layer10 = nn.Sequential(MobileNetV1_Block(in_channels=1024, out_channels=1024, stride=1)) # Stride 1 for small resolution adaptation
        
        # 平均池化层
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        # 全连接层
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
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
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

# ============================================
# 4. 辅助函数
# ============================================
def safe_loader(path):
    # 打开图片并处理透明度
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGBA').convert('RGB')

# ============================================
# 5. 主程序
# ============================================
if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # 配置路径 (请确保路径正确)
    Train_DIR = r"E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\train"
    Test_DIR = r"E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam\dataset\archive\labeled\val"

    Batch_size = 32
    Epochs = 100

    TRAIN_LOSS_HIST = []
    TEST_LOSS_HIST = []
    TRAIN_ACC_HIST = []
    TEST_ACC_HIST = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据预处理
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    if not os.path.exists(Train_DIR):
        print(f"错误：找不到训练集路径 {Train_DIR}")
        exit()
    if not os.path.exists(Test_DIR):
        print(f"警告：找不到测试集路径 {Test_DIR}")
        # exit() 

    # 加载数据
    train_dataset = datasets.ImageFolder(root=Train_DIR, transform=transform, loader=safe_loader)
    test_dataset  = datasets.ImageFolder(root=Test_DIR, transform=transform, loader=safe_loader)

    class_names = train_dataset.classes
    num_classes = len(class_names)
    print(f"检测到 {num_classes} 个分类: {class_names}")

    train_dataloader = DataLoader(dataset=train_dataset, batch_size=Batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=Batch_size, shuffle=False, num_workers=2)

    # 初始化模型
    print("初始化 MobileNetV1 + CBAM 模型...")
    model = MobileNetV1(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # 训练循环
    for epoch in range(Epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        train_bar = tqdm(train_dataloader, desc=f"Epoch [{epoch+1}/{Epochs}] Train", leave=False)
        for images, labels in train_bar:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            train_bar.set_postfix(loss=loss.item(), acc=100 * correct_train / total_train)

        epoch_loss = running_loss / len(train_dataloader)
        epoch_acc = 100 * correct_train / total_train
        TRAIN_LOSS_HIST.append(epoch_loss)
        TRAIN_ACC_HIST.append(epoch_acc)

        # 测试循环
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

        print(f"Epoch [{epoch+1}/{Epochs}] "
              f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | "
              f"Test Loss: {avg_test_loss:.4f} Acc: {avg_test_acc:.2f}%")

    # ============================================
    # 保存结果与绘图
    # ============================================
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_csv_name = f"training_log_{timestamp}.csv"
    plot_png_name = f"loss_acc_curve_{timestamp}.png"

    # 1. 打印最佳结果
    if len(TEST_ACC_HIST) > 0:
        best_acc = max(TEST_ACC_HIST)
        best_epoch = TEST_ACC_HIST.index(best_acc) + 1
        print("\n" + "="*30)
        print(f"训练结束！结果汇总")
        print("="*30)
        print(f"最佳测试准确率: {best_acc:.2f}%")
        print(f"出现在第 {best_epoch} 轮")
        print("="*30)

    # 2. 保存 CSV
    with open(log_csv_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Epoch', 'Train Loss', 'Train Acc (%)', 'Test Loss', 'Test Acc (%)'])
        for i in range(Epochs):
            writer.writerow([
                i + 1, 
                f"{TRAIN_LOSS_HIST[i]:.4f}", 
                f"{TRAIN_ACC_HIST[i]:.2f}", 
                f"{TEST_LOSS_HIST[i]:.4f}", 
                f"{TEST_ACC_HIST[i]:.2f}"
            ])
    print(f"[已保存] 训练日志数据: {log_csv_name}")

    # 3. 绘图
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(TRAIN_LOSS_HIST, label='Train Loss', color='blue')
    plt.plot(TEST_LOSS_HIST, label='Test Loss', color='red')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(TRAIN_ACC_HIST, label='Train Acc', color='blue')
    plt.plot(TEST_ACC_HIST, label='Test Acc', color='red')
    if len(TEST_ACC_HIST) > 0:
        plt.plot(best_epoch-1, best_acc, 'go', label=f'Best: {best_acc:.1f}%') 
        plt.text(best_epoch-1, best_acc, f' {best_acc:.1f}%', va='bottom')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(plot_png_name, dpi=300) 
    print(f"[已保存] 训练曲线图片: {plot_png_name}")
    plt.show()

    # ============================================
    # 混淆矩阵 (PyTorch 实现)
    # ============================================
    print("\n正在使用 PyTorch 计算混淆矩阵...")
    confusion_mat = torch.zeros(num_classes, num_classes, dtype=torch.int64)

    model.eval()
    with torch.no_grad():
        for images, labels in test_dataloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion_mat[t.long(), p.long()] += 1

    cm_data = confusion_mat.cpu().numpy()
    
    plt.figure(figsize=(12, 10))
    plt.imshow(cm_data, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(num_classes)
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)

    thresh = cm_data.max() / 2.
    for i in range(num_classes):
        for j in range(num_classes):
            plt.text(j, i, str(cm_data[i, j]),
                     horizontalalignment="center",
                     verticalalignment="center",
                     color="white" if cm_data[i, j] > thresh else "black")

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()

    cm_png_name = f"confusion_matrix_pytorch_{timestamp}.png"
    plt.savefig(cm_png_name, dpi=300)
    print(f"[已保存] 混淆矩阵图片: {cm_png_name}")
    plt.show()