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
from collections import defaultdict

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
# 3. 辅助功能函数
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

def get_stratified_indices(targets, ratio, min_samples_per_class=2, seed=42):
    """
    分层抽样：确保每个类别都按比例抽取样本
    Args:
        targets: 标签列表
        ratio: 抽样比例 (0-1)
        min_samples_per_class: 每个类别最少样本数
        seed: 随机种子
    Returns:
        indices: 抽取的索引列表
    """
    # 固定随机种子
    random.seed(seed)
    np.random.seed(seed)
    
    # 按类别组织索引
    class_indices = defaultdict(list)
    for idx, target in enumerate(targets):
        class_indices[target].append(idx)
    
    # 计算每个类别应该抽取的样本数
    selected_indices = []
    for class_label, indices in class_indices.items():
        n_samples = len(indices)
        n_select = max(min_samples_per_class, int(n_samples * ratio))
        
        # 如果要求抽取的数量超过实际数量，则全部抽取
        if n_select >= n_samples:
            selected_indices.extend(indices)
        else:
            # 随机抽取指定数量的样本
            selected = random.sample(indices, n_select)
            selected_indices.extend(selected)
    
    # 打乱顺序
    random.shuffle(selected_indices)
    return selected_indices

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
    EPOCHS = 50            # 训练轮数
    IMG_SIZE = 192          # 图片尺寸 (192比224快约30%)
    SAMPLE_RATIO = 0.05     # 0.05 表示使用5%的数据进行快速测试
    MIN_SAMPLES_PER_CLASS = 20  # 每个类别最少样本数
    
    # 硬件加速配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True # 针对固定输入大小优化卷积
    
    print(f"[{time.strftime('%H:%M:%S')}] 设备: {device}")
    print(f"配置: Batch={BATCH_SIZE}, Size={IMG_SIZE}, Epochs={EPOCHS}")
    print(f"分层抽样: 比例={SAMPLE_RATIO}, 每类最少样本={MIN_SAMPLES_PER_CLASS}")

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
    
    # 打印原始数据集统计信息
    train_targets = [full_train_dataset.targets[i] for i in range(len(full_train_dataset))]
    test_targets = [full_test_dataset.targets[i] for i in range(len(full_test_dataset))]
    
    # 统计每个类别的样本数量
    print("\n📊 原始数据集统计:")
    print(f"{'类别':<15} | {'训练集':<8} | {'测试集':<8} | {'总计':<8}")
    print("-" * 45)
    total_train = 0
    total_test = 0
    for i, class_name in enumerate(class_names):
        train_count = train_targets.count(i)
        test_count = test_targets.count(i)
        total_train += train_count
        total_test += test_count
        print(f"{class_name:<15} | {train_count:<8} | {test_count:<8} | {train_count+test_count:<8}")
    print("-" * 45)
    print(f"{'总计':<15} | {total_train:<8} | {total_test:<8} | {total_train+total_test:<8}")
    
    # 分层抽样逻辑 (用于快速测试代码)
    if SAMPLE_RATIO < 1.0:
        print(f"\n⚠ 注意: 正在使用分层抽样，比例={SAMPLE_RATIO*100}%")
        print("   确保每个类别都有均匀的数据分布...")
        
        # 使用分层抽样获取索引
        train_idx = get_stratified_indices(
            train_targets, 
            SAMPLE_RATIO, 
            min_samples_per_class=MIN_SAMPLES_PER_CLASS,
            seed=42
        )
        
        test_idx = get_stratified_indices(
            test_targets,
            SAMPLE_RATIO,
            min_samples_per_class=MIN_SAMPLES_PER_CLASS,
            seed=42
        )
        
        train_dataset = Subset(full_train_dataset, train_idx)
        test_dataset = Subset(full_test_dataset, test_idx)
        
        # 打印抽样后的统计信息
        print("\n📊 抽样后数据集统计:")
        print(f"{'类别':<15} | {'训练集':<8} | {'测试集':<8}")
        print("-" * 35)
        
        # 计算抽样后的类别分布
        sampled_train_targets = [train_targets[i] for i in train_idx]
        sampled_test_targets = [test_targets[i] for i in test_idx]
        
        for i, class_name in enumerate(class_names):
            train_count = sampled_train_targets.count(i)
            test_count = sampled_test_targets.count(i)
            print(f"{class_name:<15} | {train_count:<8} | {test_count:<8}")
        
        print(f"\n抽样后总样本数: 训练集={len(train_dataset)}, 测试集={len(test_dataset)}")
    else:
        train_dataset = full_train_dataset
        test_dataset = full_test_dataset

    # 优化 DataLoader
    num_workers = 2
    print(f"\nDataLoader Workers: {num_workers}")
    
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
    print("\n" + "="*50)
    print(f"总体准确率 (Overall Accuracy): {100 * total_correct / total_samples:.2f}%")
    print("="*50)
    print(f"{'类别':<20} | {'准确数':<8} | {'总数':<8} | {'准确率':<10}")
    print("-" * 55)
    
    for i in range(num_classes):
        correct = cm[i, i]
        total = cm[i, :].sum()
        accuracy = correct / total if total > 0 else 0
        print(f"{class_names[i]:<20} | {correct:<8} | {total:<8} | {accuracy:.4f}")
    print("="*50)

    # 3. 绘制混淆矩阵 (使用 Matplotlib)
    plt.figure(figsize=(12, 10))
    
    # 归一化处理
    cm_normalized = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-7)
    
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Normalized)')
    plt.colorbar()
    
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # 在格子里填数字 (显示具体数量)
    thresh = cm_normalized.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, f"{cm[i, j]}\n({cm_normalized[i, j]:.2f})",
                 horizontalalignment="center",
                 verticalalignment="center",
                 color="white" if cm_normalized[i, j] > thresh else "black",
                 fontsize=8)
    
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    print("\n✅ 混淆矩阵已保存为 confusion_matrix.png")
    
    print("\n所有任务已完成。请查看 training_log.txt 获取详细日志。")

    print("✅ 所有任务完成，60秒后将自动关机...")
    print("如果不希望关机，请在命令行输入 shutdown -a 取消")
    
    # Windows 关机命令 (-s:关机, -t 60:延迟60秒)
    os.system("shutdown -s -t 60")