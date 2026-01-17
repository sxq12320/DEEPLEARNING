import torch 
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageFile
import warnings
import random
import sys
import time
import itertools

# --- 核心优化模块 ---
from torch.cuda.amp import autocast, GradScaler

# 防止截断图片报错
ImageFile.LOAD_TRUNCATED_IMAGES = True

# -----------------------------------------------------------------------
# 1. 日志记录器 (同时输出到屏幕和txt文件)
# -----------------------------------------------------------------------
class Logger(object):
    def __init__(self, filename="training_log.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.log.flush()

# -----------------------------------------------------------------------
# 2. 模型定义: CBAM + MobileNetV1
# -----------------------------------------------------------------------
class CBAMBlock(nn.Module):
    def __init__(self, outchannels, reduction=4):
        super(CBAMBlock, self).__init__()
        # 通道注意力
        self.maxpool = nn.AdaptiveMaxPool2d(1)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(outchannels, outchannels // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(outchannels // reduction, outchannels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        # 空间注意力
        self.conv = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False)

    def forward(self, x):
        # Channel Attention
        max_out = self.fc2(self.relu(self.fc1(self.maxpool(x))))
        avg_out = self.fc2(self.relu(self.fc1(self.avgpool(x))))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        
        # Spatial Attention
        spatial_max = torch.max(x, dim=1, keepdim=True)[0]
        spatial_avg = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([spatial_max, spatial_avg], dim=1)))
        return spatial_out * x

class MobileNetV1_Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, use_cbam=False):
        super(MobileNetV1_Block, self).__init__()
        # 深度可分离卷积
        self.conv1 = nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.use_cbam = use_cbam
        if use_cbam:
            self.cbam = CBAMBlock(out_channels)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        if self.use_cbam:
            x = self.cbam(x)
        return x

class MobileNetV1_CBAM(nn.Module):
    def __init__(self, num_classes=10):
        super(MobileNetV1_CBAM, self).__init__()
        self.layer1 = nn.Sequential(nn.Conv2d(3, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True), CBAMBlock(32))
        self.layer2 = nn.Sequential(MobileNetV1_Block(32, 64, 1, False))
        self.layer3 = nn.Sequential(MobileNetV1_Block(64, 128, 2, False))
        self.layer4 = nn.Sequential(MobileNetV1_Block(128, 128, 1))
        self.layer5 = nn.Sequential(MobileNetV1_Block(128, 256, 2, False))
        self.layer6 = nn.Sequential(MobileNetV1_Block(256, 256, 1))
        self.layer7 = nn.Sequential(MobileNetV1_Block(256, 512, 2, False))
        self.layer8 = nn.Sequential(
            MobileNetV1_Block(512, 512, 1), MobileNetV1_Block(512, 512, 1), 
            MobileNetV1_Block(512, 512, 1), MobileNetV1_Block(512, 512, 1), 
            MobileNetV1_Block(512, 512, 1, False)
        )
        self.layer9 = nn.Sequential(MobileNetV1_Block(512, 1024, 2, False))
        self.layer10 = nn.Sequential(MobileNetV1_Block(1024, 1024, 2))
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.5)
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
        x = self.dropout(x)
        x = self.fc(x)
        return x

# -----------------------------------------------------------------------
# 3. 辅助功能函数 (加载、绘图)
# -----------------------------------------------------------------------
def safe_loader(path):
    """高效图片加载"""
    try:
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    except Exception as e:
        print(f"Warning: Error loading image {path} : {e}")
        return Image.new('RGB', (192, 192))

def plot_confusion_matrix_mpl(cm, classes, normalize=False, title='Confusion Matrix'):
    """使用 Matplotlib 绘制混淆矩阵"""
    plt.figure(figsize=(12, 10))
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("✅ 混淆矩阵已保存为 confusion_matrix.png")

# -----------------------------------------------------------------------
# 4. 主训练逻辑
# -----------------------------------------------------------------------
if __name__ == "__main__":
    # 重定向输出到日志文件
    sys.stdout = Logger("training_log.txt")
    warnings.filterwarnings("ignore")
    
    # --- 1. 参数设置 ---
    Train_DIR = r"E:\mastercode\data\RSCD\train"
    Test_DIR = r"E:\mastercode\data\RSCD\test"
    
    BATCH_SIZE = 32         # 建议 32 或 64
    EPOCHS = 100            # 训练轮数
    IMG_SIZE = 192          # 图片尺寸 (192比224快约30%)
    SAMPLE_RATIO = 0.01     # 0.01 表示使用1%的数据进行快速测试
    
    # 硬件加速配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True # 针对固定输入大小优化卷积
    
    print(f"[{time.strftime('%H:%M:%S')}] 设备: {device}")
    print(f"配置: Batch={BATCH_SIZE}, Size={IMG_SIZE}, Epochs={EPOCHS}")

    # --- 2. 数据准备 ---
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5434369376344288, 0.5481078096551103, 0.5279714255467811), 
                             std=(0.17983000422261916, 0.1788723625524178, 0.18184710069990692))
    ])
    
    if not os.path.exists(Train_DIR):
        print(f"错误: 找不到路径 {Train_DIR}")
        exit()

    print("\n正在加载数据集索引...")
    full_train_dataset = datasets.ImageFolder(Train_DIR, transform=transform, loader=safe_loader)
    full_test_dataset = datasets.ImageFolder(Test_DIR, transform=transform, loader=safe_loader)
    
    class_names = full_train_dataset.classes
    num_classes = len(class_names)
    print(f"类别数量: {num_classes} | 类别名称: {class_names}")
    
    # 抽样逻辑 (用于快速测试代码)
    if SAMPLE_RATIO < 1.0:
        print(f"⚠ 注意: 正在使用 {SAMPLE_RATIO*100}% 的数据进行测试运行...")
        train_idx = random.sample(range(len(full_train_dataset)), int(len(full_train_dataset)*SAMPLE_RATIO))
        test_idx = random.sample(range(len(full_test_dataset)), int(len(full_test_dataset)*SAMPLE_RATIO))
        train_dataset = Subset(full_train_dataset, train_idx)
        test_dataset = Subset(full_test_dataset, test_idx)
    else:
        train_dataset = full_train_dataset
        test_dataset = full_test_dataset

    # 优化 DataLoader
    num_workers = 2
    print(f"DataLoader Workers: {num_workers}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, 
                              num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, 
                             num_workers=num_workers, pin_memory=True, persistent_workers=(num_workers>0))

    # --- 3. 初始化模型 ---
    model = MobileNetV1_CBAM(num_classes=num_classes).to(device)
    
    # 尝试编译模型 (PyTorch 2.0+)
    try:
        model = torch.compile(model)
        print("✅ Model compiled successfully.")
    except:
        pass

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 混合精度 Scaler
    scaler = GradScaler()
    
    # --- 4. 训练循环 ---
    history = {'train_loss': [], 'test_loss': [], 'train_acc': [], 'test_acc': []}
    best_acc = 0.0
    start_time = time.time()

    print("\n🚀 开始训练...")
    
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        loop = tqdm(train_loader, desc=f"Ep [{epoch+1}/{EPOCHS}]", leave=True)
        
        for images, labels in loop:
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            
            optimizer.zero_grad()
            
            # --- 混合精度前向 ---
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            # --- 混合精度反向 ---
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.detach(), 1) # detach省显存
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100*correct_train/total_train)
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100 * correct_train / total_train
        
        # --- 测试验证 ---
        model.eval()
        test_loss = 0.0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                with autocast():
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                
                test_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        avg_test_acc = 100 * correct_test / total_test
        
        print(f" -> Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}% | Test Loss: {avg_test_loss:.4f} Acc: {avg_test_acc:.2f}%")
        
        history['train_loss'].append(epoch_loss)
        history['test_loss'].append(avg_test_loss)
        history['train_acc'].append(epoch_acc)
        history['test_acc'].append(avg_test_acc)
        
        if avg_test_acc > best_acc:
            best_acc = avg_test_acc
            torch.save(model.state_dict(), 'best_model.pth')
            print("🌟 最佳模型已保存")

    print(f"\n✅ 训练完成! 总耗时: {(time.time()-start_time)/60:.1f} min. 最佳精度: {best_acc:.2f}%")

    # --- 5. 结果可视化: 曲线图 ---
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['test_loss'], label='Test Loss')
    plt.title('Loss Curve')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['test_acc'], label='Test Acc')
    plt.title('Accuracy Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_curves.png')
    print("✅ 训练曲线已保存为 training_curves.png")

    # --- 6. 结果可视化: 纯手写混淆矩阵 (无 sklearn 依赖) ---
    print("\n正在生成混淆矩阵 (纯 PyTorch/Numpy 实现)...")
    
    # 加载最佳模型
    if os.path.exists('best_model.pth'):
        model.load_state_dict(torch.load('best_model.pth'))
    model.eval()
    
    # 1. 初始化混淆矩阵 (行: 真实标签, 列: 预测标签)
    # 使用 PyTorch Tensor 累加，速度更快
    conf_matrix = torch.zeros(num_classes, num_classes)
    
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Testing for Matrix"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            
            # 累加混淆矩阵
            for t, p in zip(labels.view(-1), preds.view(-1)):
                conf_matrix[t.long(), p.long()] += 1
                
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    # 转为 Numpy 方便绘图
    cm = conf_matrix.cpu().numpy().astype(int)
    
    # 2. 打印简易分类报告
    print("\n" + "="*40)
    print(f"总体准确率 (Overall Accuracy): {100 * total_correct / total_samples:.2f}%")
    print("="*40)
    print(f"{'Class':<20} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 58)
    
    for i in range(num_classes):
        # TP: 对角线元素
        tp = cm[i, i]
        # FP: 这一列的总和 - TP
        fp = cm[:, i].sum() - tp
        # FN: 这一行的总和 - TP
        fn = cm[i, :].sum() - tp
        
        # 防止除零错误
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{class_names[i]:<20} | {precision:.4f}     | {recall:.4f}     | {f1:.4f}")
    print("="*40)

    # 3. 绘制混淆矩阵 (使用 Matplotlib)
    plt.figure(figsize=(12, 10))
    
    # 归一化处理 (可选，让颜色显示百分比)
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在格子里填数字 (显示 具体数量 和 百分比)
    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        # 文字内容：数量 (百分比)
        text_str = f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})"
        
        plt.text(j, i, text_str,
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black",
                 fontsize=8) # 字体稍微调小一点以防重叠
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✅ 混淆矩阵已保存为 confusion_matrix.png")
    
    print("\n所有任务已完成。请查看 training_log.txt 获取详细日志。")


    # ... (你的画图代码) ...
    print("✅ 所有任务完成，60秒后将自动关机...")
    print("如果不希望关机，请在命令行输入 shutdown -a 取消")
    
    # Windows 关机命令 (-s:关机, -t 60:延迟60秒)
    os.system("shutdown -s -t 60")