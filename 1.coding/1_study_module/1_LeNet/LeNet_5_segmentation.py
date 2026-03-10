import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# ==========================================
# 1. 定义数据集类 (处理 VOC 格式)
# ==========================================
class VOCSegmentationDataset(Dataset):
    def __init__(self, root_dir, image_set='test'):
        self.root_dir = root_dir
        self.image_dir = os.path.join(self.root_dir, 'JPEGImages')
        self.mask_dir = os.path.join(self.root_dir, 'SegmentationClass')
        
        txt_path = os.path.join(self.root_dir, 'ImageSets', 'Segmentation', f'{image_set}.txt')
        with open(txt_path, 'r') as f:
            self.file_names = [line.strip() for line in f.readlines()]
            
        self.img_transform = transforms.Compose([
            transforms.Resize((256, 256)), 
            transforms.ToTensor()          
        ])
        
    def __len__(self):
        return len(self.file_names)
        
    def __getitem__(self, idx):
        file_name = self.file_names[idx]
        
        # 读取图片并预处理 [3, 256, 256]
        img_path = os.path.join(self.image_dir, file_name + '.jpg')
        image = Image.open(img_path).convert('RGB')
        image = self.img_transform(image) 
        
        # 读取标签并预处理 (严禁使用双线性插值，必须用 NEAREST)
        mask_path = os.path.join(self.mask_dir, file_name + '.png')
        mask = Image.open(mask_path)
        mask = mask.resize((256, 256), Image.NEAREST)
        
        mask_tensor = torch.from_numpy(np.array(mask)).long()
        return image, mask_tensor

# ==========================================
# 2. 定义 FCN_LeNet 模型 (适配彩色图与 21 类)
# ==========================================
class FCN_LeNet(nn.Module):
    def __init__(self, num_classes=21): 
        super(FCN_LeNet, self).__init__()
        self.features = nn.Sequential(
            # 修改点：in_channels 改为 3 (RGB 彩色图)
            nn.Conv2d(3, 6, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(16, 120, kernel_size=5, padding=2), 
            nn.ReLU(),
            nn.Conv2d(120, 84, kernel_size=1, padding=0), 
            nn.ReLU(),
            nn.Conv2d(84, num_classes, kernel_size=1) 
        )
        self.upsample = nn.ConvTranspose2d(
            num_classes, num_classes, kernel_size=8, stride=4, padding=2
        )

    def forward(self, x):
        original_size = x.shape[2:]
        x = self.features(x)
        x = self.classifier(x)
        x = self.upsample(x)
        if x.shape[2:] != original_size:
            x = F.interpolate(x, size=original_size, mode='bilinear', align_corners=False)
        return x

# ==========================================
# 3. 主程序：批量训练循环与验证可视化
# ==========================================
if __name__ == "__main__":
    # ！！！ 极其重要：请替换为你电脑上 VOCdevkit/VOC2007 文件夹的绝对路径 ！！！
    # 例如: VOC_ROOT = r"E:\mastercode\data\VOC\VOCtest_06-Nov-2007\VOCdevkit\VOC2007"
    VOC_ROOT = r"E:\mastercode\data\VOC\VOCtest_06-Nov-2007\VOCdevkit\VOC2007" 
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"当前使用的计算设备: {device}")

    # 1. 准备数据管道
    try:
        dataset = VOCSegmentationDataset(root_dir=VOC_ROOT, image_set='test')
        # batch_size=4 意味着每次同时送入 4 张图片计算梯度
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
        print(f"成功加载数据集，共找到 {len(dataset)} 张图片。")
    except Exception as e:
        print(f"数据集加载失败，请检查 VOC_ROOT 路径！错误信息: {e}")
        exit()

    # 2. 初始化网络
    # VOC 包含 20 个前景类别 + 1 个背景类 = 21 类
    model = FCN_LeNet(num_classes=21).to(device)
    
    # 核心技巧：忽略标签中的 255 (白色物体分界线)
    criterion = nn.CrossEntropyLoss(ignore_index=255)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 3. 开始真实训练循环
    num_epochs = 300 # 先跑 10 个 Epoch 看看整体流程
    print("\n--- 开始在 VOC 数据集上进行训练 ---")
    
    for epoch in range(num_epochs):
        model.train() 
        epoch_loss = 0.0
        
        # 遍历 DataLoader 吐出的每一个批次
        for batch_idx, (images, masks) in enumerate(dataloader):
            images, masks = images.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            # 打印每个小批次的进度 (防止你觉得程序卡死了)
            if (batch_idx + 1) % 50 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(dataloader)}], Loss: {loss.item():.4f}")
        
        # 打印当前 Epoch 的平均 Loss
        avg_loss = epoch_loss / len(dataloader)
        print(f"===> Epoch {epoch+1} 完成, 平均 Loss: {avg_loss:.4f}\n")

    print("--- 训练完成！抽取一张图片测试效果 ---")

    # 4. 可视化效果对比
    model.eval() 
    with torch.no_grad():
        test_img, test_mask = dataset[0] # 从数据集拿第一张图片
        test_img_batch = test_img.unsqueeze(0).to(device) 
        
        output = model(test_img_batch)
        pred_mask = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
    # 张量形状转换，适配 matplotlib 画图格式
    img_show = test_img.permute(1, 2, 0).numpy() # [C,H,W] 变 [H,W,C]
    mask_show = test_mask.numpy()

    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(img_show)
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Ground Truth Mask")
    mask_show[mask_show == 255] = 0 # 强行把边界 255 改成 0，防止破坏颜色映射表
    plt.imshow(mask_show, cmap='nipy_spectral', vmin=0, vmax=20) 
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Predicted Mask (LeNet)")
    plt.imshow(pred_mask, cmap='nipy_spectral', vmin=0, vmax=20)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()