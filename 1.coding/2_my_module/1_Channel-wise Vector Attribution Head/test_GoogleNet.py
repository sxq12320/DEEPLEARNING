import torch 
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# --- 1. 基础模块定义 ---

class InceptionBlock(nn.Module):
    def __init__(self , in_channels , out_1x1 , out_3x3_reduce , out_3x3 , out_5x5_reduce , out_5x5 , out_pool_proj):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )
        self.branch3x3 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_reduce, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )
        self.poolbranch = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool_proj, kernel_size=1),
            nn.BatchNorm2d(out_pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        return torch.cat([self.branch1x1(x), self.branch3x3(x), self.branch5x5(x), self.poolbranch(x)], dim=1)

# --- 2. 改进后的 Axeon_Block ---

class Axeon_Block(nn.Module):
    def __init__(self, in_channels, num_classes=10, feature_size=1):
        super(Axeon_Block, self).__init__()
        self.num_classes = num_classes
        
        # 优化 1: 缩小初始化权重，防止点群初始时过于分散
        self.projector_weight = nn.Parameter(
            torch.randn(in_channels, feature_size * feature_size, num_classes) * 0.01
        )
        self.projector_bias = nn.Parameter(torch.zeros(in_channels, num_classes))

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, H * W)
        
        # 投影到 N 维空间
        points = torch.matmul(x_flat.unsqueeze(2), self.projector_weight.unsqueeze(0)) 
        points = points.squeeze(2) + self.projector_bias.unsqueeze(0)
        
        # 每个通道独立归一化
        prob_matrix = F.softmax(points, dim=2) # [B, 1024, N]

        # 拟合直线 (SVD)
        centroid = torch.mean(prob_matrix, dim=1, keepdim=True)
        centered_points = prob_matrix - centroid
        
        # 优化 2: 增加 eps 保证数值稳定
        U, S, V = torch.linalg.svd(centered_points + 1e-7)
        v = V[:, :, 0] # 主成分方向向量 [B, N]

        # 优化 3: 解决 SVD 符号翻转问题 (反向传播稳定性关键)
        v = v * torch.sign(v.sum(dim=-1, keepdim=True) + 1e-7)

        # 计算对齐得分
        axes = torch.eye(self.num_classes).to(x.device)
        # 我们用 scores 作为判定依据，也作为 loss 的输入
        scores = torch.matmul(v, axes.T) 
        abs_scores = torch.abs(scores) 

        _, predictions = torch.max(abs_scores, dim=1)
        
        return predictions, prob_matrix, scores , S

# --- 3. 集成网络 ---

class GoogleNet_CIFAR10_Axeon(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogleNet_CIFAR10_Axeon, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            nn.Conv2d(64, 192, 3, 1, 1),
            nn.BatchNorm2d(192), nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2, 1),
            InceptionBlock(192, 64, 96, 128, 16, 32, 32),
            InceptionBlock(256, 128, 128, 192, 32, 96, 64),
            nn.MaxPool2d(3, 2, 1),
            InceptionBlock(480, 192, 96, 208, 16, 48, 64),
            InceptionBlock(512, 160, 112, 224, 24, 64, 64),
            InceptionBlock(512, 128, 128, 256, 24, 64, 64),
            InceptionBlock(512, 112, 144, 288, 32, 64, 64),
            InceptionBlock(528, 256, 160, 320, 32, 128, 128),
            nn.MaxPool2d(3, 2, 1),
            InceptionBlock(832, 256, 160, 320, 32, 128, 128),
            InceptionBlock(832, 384, 192, 384, 48, 128, 128),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.axeon_head = Axeon_Block(in_channels=1024, num_classes=num_classes, feature_size=1)

    def forward(self, x):
        x = self.features(x)
        return self.axeon_head(x)

# --- 4. 训练主函数 ---

from sklearn.metrics import average_precision_score

def train_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # ... [此处保留你提供的 transform 和 dataset 定义] ...
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
    train_dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./data', train=False, download=True, transform=test_transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1024, shuffle=True, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=2)

    model = GoogleNet_CIFAR10_Axeon(num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    # --- 记录器 ---
    history = {
        'train_loss': [], 'test_loss': [],
        'train_acc': [], 'test_acc': [],
        'line_intensity': [], 'mAP': []
    }

    for epoch in range(50):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        s1_list = []
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/50')
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            
            preds, prob_matrix, scores, S = model(images)
            
            # 记录拟合强度 (S1 占比)
            line_intensity = (S[:, 0] / (torch.sum(S, dim=1) + 1e-7)).mean().item()
            s1_list.append(line_intensity)

            avg_probs = torch.mean(prob_matrix, dim=1)
            loss = criterion(avg_probs, labels) + 0.5 * criterion(torch.abs(scores), labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            running_loss += loss.item()
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pbar.set_postfix(loss=f"{loss.item():.3f}", acc=f"{100.*correct/total:.1f}%")

        # --- 验证逻辑 (含 mAP) ---
        model.eval()
        t_loss, t_correct, t_total = 0, 0, 0
        all_scores, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                preds, prob_matrix, scores, S = model(images)
                
                avg_probs = torch.mean(prob_matrix, dim=1)
                t_loss += (criterion(avg_probs, labels) + 0.5 * criterion(torch.abs(scores), labels)).item()
                t_total += labels.size(0)
                t_correct += (preds == labels).sum().item()
                
                all_scores.append(torch.abs(scores).cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        # 计算 mAP
        y_true = np.eye(10)[np.concatenate(all_labels)]
        y_pred = np.concatenate(all_scores)
        current_map = average_precision_score(y_true, y_pred, average='macro')

        # 更新历史记录
        history['train_loss'].append(running_loss / len(train_loader))
        history['test_loss'].append(t_loss / len(test_loader))
        history['train_acc'].append(100. * correct / total)
        history['test_acc'].append(100. * t_correct / t_total)
        history['line_intensity'].append(np.mean(s1_list))
        history['mAP'].append(current_map)

        print(f"Epoch {epoch+1} Summary: Test Acc: {history['test_acc'][-1]:.2f}%, mAP: {current_map:.4f}, Intensity: {history['line_intensity'][-1]:.3f}")
        
    return model, history, test_loader, device

def plot_metrics(history):
    epochs = range(1, len(history['train_loss']) + 1)
    plt.style.use('ggplot')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Loss 曲线
    ax1.plot(epochs, history['train_loss'], label='Train Loss', color='blue')
    ax1.plot(epochs, history['test_loss'], label='Test Loss', color='red', linestyle='--')
    ax1.set_title('Loss Evolution')
    ax1.set_xlabel('Epochs')
    ax1.legend()

    # Acc & Intensity & mAP 曲线
    ax2.plot(epochs, history['test_acc'], label='Test Accuracy (%)', color='green')
    ax2.plot(epochs, [x * 100 for x in history['mAP']], label='mAP (%)', color='purple', linestyle='-.')
    # 增加第二个 y 轴来画拟合强度
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs, history['line_intensity'], label='Line Intensity (S1)', color='orange', alpha=0.5)
    ax2_twin.set_ylabel('Intensity Ratio')
    
    ax2.set_title('Performance & Geometry Intensity')
    ax2.set_xlabel('Epochs')
    ax2.legend(loc='lower right')
    plt.show()

import seaborn as sns

def visualize_10d_matrix(model, test_loader, device):
    model.eval()
    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        preds, prob_matrix, scores, S = model(images)
        
        # 选一个正确的样本展示
        sample_idx = 0 
        true_label = labels[sample_idx].item()
        # [1024, 10] 矩阵
        matrix_10d = prob_matrix[sample_idx].cpu().numpy()
        
        plt.figure(figsize=(12, 8))
        # 绘制 1024x10 的热力图 (只取前 100 通道方便观察)
        sns.heatmap(matrix_10d[:100, :], cmap="viridis")
        plt.title(f"Axeon 10D Geometric Matrix (Sample: Class {true_label})\n(1024 Channels x 10 Classes)")
        plt.xlabel("10-Dimensional Class Axes")
        plt.ylabel("Channel Index (Top 100)")
        plt.show()

        # 打印奇异值谱
        s_vals = S[sample_idx].cpu().numpy()
        plt.bar(range(10), s_vals / np.sum(s_vals))
        plt.title("Singular Value Spectrum (Energy Distribution)")
        plt.show()

if __name__ == "__main__":
    # 执行训练
    trained_model, test_loader, device = train_model()