import torch
import torch.nn as nn
import torch.nn.functional as F

class Axeon_Block(nn.Module):
    def __init__(self, in_channels, num_classes=3, feature_size=8):
        super(Axeon_Block, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feature_size = feature_size

        # 第一步：通道融合 (1x1 卷积)
        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=1,
            stride=1,
            padding=0
        )
        self.relu = nn.ReLU()

        # 第二步：定义每个通道独立的投影参数
        # 形状：[通道数, 像素总数, 分类数]
        self.projector_weight = nn.Parameter(
            torch.randn(in_channels, feature_size * feature_size, num_classes)
        )
        self.projector_bias = nn.Parameter(
            torch.zeros(in_channels, num_classes)
        )
    
    def forward(self, x):
        B, C, H, W = x.shape

        # --- Step 2: 通道融合 ---
        x = self.relu(self.conv1(x))  # [B, C, H, W]

        # --- Step 3: 对每个通道单独进行 Softmax 回归 ---
        # 1. 展平空间维度 [B, C, H*W]
        x_flat = x.view(B, C, H * W)
        
        # 2. 批量投影 (注意这里变量名必须与 __init__ 中一致)
        # x_flat.unsqueeze(2) -> [B, C, 1, H*W]
        # self.projector_weight.unsqueeze(0) -> [1, C, H*W, N]
        points = torch.matmul(x_flat.unsqueeze(2), self.projector_weight.unsqueeze(0)) 
        points = points.squeeze(2) + self.projector_bias.unsqueeze(0)  # [B, C, N]

        # 3. Softmax 归一化：将每个通道的投影转为概率坐标
        prob_matrix = F.softmax(points, dim=2)  # [B, C, N]

        # --- Step 4: 拟合直线 (线性回归) ---
        centroid = torch.mean(prob_matrix, dim=1, keepdim=True)  # [B, 1, N]
        centered_points = prob_matrix - centroid  # [B, C, N]
        
        # 使用 SVD 求主成分方向
        _, _, V = torch.linalg.svd(centered_points)
        v = V[:, :, 0]  # 方向向量 [B, N]

        # --- Step 5: 计算回归直线与坐标轴的距离 ---
        axes = torch.eye(self.num_classes).to(x.device)  # [N, N]
        
        scores = torch.abs(torch.matmul(v, axes.T))  # [B, N]

        _, predictions = torch.max(scores, dim=1)  # [B]
        
        return predictions, prob_matrix
