import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import shutil
import re
import random
import matplotlib.pyplot as plt  # <---【新增】引入绘图库
from tqdm import tqdm

# ================== 1. 网络结构定义 ==================
class FullConnectedNet(nn.Module):
    def __init__(self , input_size , hidden_size_1 , hidden_size_2 , num_classed):
        super(FullConnectedNet, self).__init__()
        self.flatten = nn.Flatten()
        self.layer1 = nn.Linear(input_size , hidden_size_1)
        self.layer2 = nn.Linear(hidden_size_1 , hidden_size_2)
        self.layer3 = nn.Linear(hidden_size_2 , num_classed)
        self.relu = nn.ReLU()

    def forward(self , x):
        x = self.flatten(x)
        out = self.layer1(x)
        out = self.relu(out)
        out = self.layer2(out)
        out = self.relu(out)
        out = self.layer3(out)
        return out
    
class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Inception(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()
        
        # 线路1：1x1 卷积
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)

        # 线路2：1x1 卷积(降维) -> 3x3 卷积
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 线路3：1x1 卷积(降维) -> 5x5 卷积
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 线路4：3x3 最大池化 -> 1x1 卷积
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        # 并行计算四个分支
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 在通道维度 (dim=1) 拼接结果
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
    
class MyGoogLeNet(nn.Module):
    def __init__(self, num_classes=5):
        super(MyGoogLeNet, self).__init__()
        
        # --- 初始层 (Pre-layers) ---
        self.pre_layers = nn.Sequential(
            BasicConv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            BasicConv2d(64, 64, kernel_size=1),
            BasicConv2d(64, 192, kernel_size=3, padding=1),
            nn.MaxPool2d(3, stride=2, ceil_mode=True)
        )

        # --- Inception 块 (参数参考原论文) ---
        # 格式: (输入, 1x1, 3x3降维, 3x3, 5x5降维, 5x5, 池化投影)
        self.a3 = Inception(192, 64, 96, 128, 16, 32, 32)
        self.b3 = Inception(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(3, stride=2, ceil_mode=True)

        self.a4 = Inception(480, 192, 96, 208, 16, 48, 64)
        self.b4 = Inception(512, 160, 112, 224, 24, 64, 64)
        self.c4 = Inception(512, 128, 128, 256, 24, 64, 64)
        self.d4 = Inception(512, 112, 144, 288, 32, 64, 64)
        self.e4 = Inception(528, 256, 160, 320, 32, 128, 128)

        self.a5 = Inception(832, 256, 160, 320, 32, 128, 128)
        self.b5 = Inception(832, 384, 192, 384, 48, 128, 128)

        # --- 分类层 ---
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)
        x = self.a3(x)
        x = self.b3(x)
        x = self.maxpool(x)

        x = self.a4(x)
        x = self.b4(x)
        x = self.c4(x)
        x = self.d4(x)
        x = self.e4(x)
        x = self.maxpool(x)

        x = self.a5(x)
        x = self.b5(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# ================== 2. 数据整理函数 ==================
def init_picture():
    base_path = r"E:\mastercode\DEEPLEARNING\5_class_ZhuQIBing\5 FinalExam"
    source_dir = os.path.join(base_path, r"dataset\Natural Hand Digits Dataset")
    processed_dir = os.path.join(base_path, r"dataset\processed_data")
    final_split_dir = os.path.join(base_path, r"dataset\dataset_final_split")

    if os.path.exists(final_split_dir) and len(os.listdir(final_split_dir)) > 0:
        print("检测到数据已整理，直接使用。")
        return final_split_dir

    print("开始整理数据...")
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    if not os.path.exists(source_dir):
        print(f"错误：找不到源文件夹 {source_dir}")
        return None

    for folder_name in os.listdir(source_dir):
        folder_path = os.path.join(source_dir, folder_name)
        if os.path.isdir(folder_path):
            match = re.search(r"Digit (\d+)", folder_name)
            if match:
                digit_class = match.group(1)
                class_save_path = os.path.join(processed_dir, digit_class)
                os.makedirs(class_save_path, exist_ok=True)
                for filename in os.listdir(folder_path):
                    shutil.copy2(os.path.join(folder_path, filename), 
                                 os.path.join(class_save_path, f"{folder_name}_{filename}"))

    train_ratio = 0.8
    if os.path.exists(final_split_dir):
        shutil.rmtree(final_split_dir) 

    for split in ['train', 'val']:
        os.makedirs(os.path.join(final_split_dir, split), exist_ok=True)

    classes = [d for d in os.listdir(processed_dir) if os.path.isdir(os.path.join(processed_dir, d))]
    for cls in classes:
        cls_path = os.path.join(processed_dir, cls)
        images = os.listdir(cls_path)
        random.shuffle(images)
        split_idx = int(len(images) * train_ratio)
        splits = {'train': images[:split_idx], 'val': images[split_idx:]}
        for split_name, img_list in splits.items():
            save_path = os.path.join(final_split_dir, split_name, cls)
            os.makedirs(save_path, exist_ok=True)
            for img in img_list:
                shutil.copy2(os.path.join(cls_path, img), os.path.join(save_path, img))
    print("数据整理完成！")
    return final_split_dir

# ================== 3. 主程序 ==================
if __name__ == "__main__":
    data_root = init_picture()
    if data_root is None: exit()

    # --- 超参数 ---
    IMG_SIZE = 224  
    input_size = 3 * IMG_SIZE * IMG_SIZE 
    hidden_size_1 = 300
    hidden_size_2 = 300
    num_classes = 5
    learning_rate = 0.001
    batch_size = 32
    num_epochs = 15 #稍微增加轮数，曲线更好看

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"当前运行设备: {device}")

    # --- 数据加载 ---
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]),
        'val': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
        ]),
    }

    image_datasets = {x: datasets.ImageFolder(os.path.join(data_root, x), data_transforms[x])
                      for x in ['train', 'val']}
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=True)
                   for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    
    # --- 模型与优化器 ---
    model = MyGoogLeNet(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # <--- 【新增】用于记录历史数据
    history = {
        'train_loss': [],
        'val_loss': [],
        'train_acc': [],
        'val_acc': []
    }

    print(f"开始训练，共 {num_epochs} 轮...")
    
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 20)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloaders[phase], desc=f"[{phase}]", unit="batch")

            for inputs, labels in loop:
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                loop.set_postfix(loss=loss.item())

            epoch_loss = running_loss / dataset_sizes[phase]
            # .cpu().item() 确保转为普通的 Python float
            epoch_acc = (running_corrects.double() / dataset_sizes[phase]).cpu().item()

            print(f'{phase} Summary -> Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # <--- 【新增】保存每轮的数据
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc)
            else:
                history['val_loss'].append(epoch_loss)
                history['val_acc'].append(epoch_acc)

    print("训练完成！正在绘制曲线...")

    # ================== 【新增】绘图部分 ==================
    plt.figure(figsize=(12, 5))

    # 1. 绘制 Loss 曲线
    plt.subplot(1, 2, 1) # 1行2列，第1个图
    plt.plot(range(1, num_epochs+1), history['train_loss'], label='Train Loss')
    plt.plot(range(1, num_epochs+1), history['val_loss'], label='Val Loss', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)

    # 2. 绘制 Accuracy 曲线
    plt.subplot(1, 2, 2) # 1行2列，第2个图
    plt.plot(range(1, num_epochs+1), history['train_acc'], label='Train Accuracy')
    plt.plot(range(1, num_epochs+1), history['val_acc'], label='Val Accuracy', linestyle='--')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show() # 显示图片