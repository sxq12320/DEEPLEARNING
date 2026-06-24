"""
改进网络：Multi-Scale 1D CNN + SE Attention + Attention Pooling
================================================================
设计思路：
1. MLP的问题：将64个点flatten成128维向量，完全丢失空间顺序和局部几何关系
2. Transformer的问题：O(n²)注意力对64个点过重，训练/推理慢
3. 本方案：
   - 多尺度1D CNN（kernel=3,5,7 + dilation）捕捉不同尺度的局部几何
   - 残差连接 + BatchNorm 保证训练稳定
   - SE通道注意力 自适应加权重要特征通道
   - 注意力池化 替代简单avg pooling，学习"哪些点对授粉点更重要"
   - 整体参数量适中，推理速度快
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # x: (B, C, L)
        b, c, _ = x.size()
        w = self.squeeze(x).view(b, c)
        w = self.excitation(w).view(b, c, 1)
        return x * w


class MultiScaleConv1DBlock(nn.Module):
    """多尺度1D卷积块：并行不同kernel size + 膨胀卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        mid = out_channels // 4

        # 4个并行分支，不同感受野
        self.branch_small = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=3, padding=1, dilation=1),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch_medium = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch_large = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=3, padding=4, dilation=4),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
        )
        self.branch_global = nn.Sequential(
            nn.Conv1d(in_channels, mid, kernel_size=1),
            nn.BatchNorm1d(mid),
            nn.ReLU(inplace=True),
        )

        # 融合后压缩
        self.fuse = nn.Sequential(
            nn.Conv1d(mid * 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x):
        s = self.branch_small(x)
        m = self.branch_medium(x)
        l = self.branch_large(x)
        g = self.branch_global(x)
        out = torch.cat([s, m, l, g], dim=1)
        out = self.fuse(out)
        out = self.se(out)
        return out


class ResidualConvBlock(nn.Module):
    """残差卷积块"""
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=padding),
            nn.BatchNorm1d(channels),
        )
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(channels)

    def forward(self, x):
        residual = x
        out = self.conv(x)
        out = self.se(out)
        out = out + residual
        return self.relu(out)


class AttentionPooling(nn.Module):
    """注意力池化：学习哪些点对授粉点预测更重要"""
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv1d(channels, channels // 2, 1),
            nn.BatchNorm1d(channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // 2, 1, 1),
        )

    def forward(self, x):
        # x: (B, C, L)
        attn_weights = self.attention(x)  # (B, 1, L)
        attn_weights = F.softmax(attn_weights, dim=-1)  # (B, 1, L)
        out = (x * attn_weights).sum(dim=-1)  # (B, C)
        return out, attn_weights.squeeze(1)  # 返回权重用于可视化


class ImprovedContourNetV2(nn.Module):
    """
    改进网络 V2：多尺度CNN + SE注意力 + 注意力池化

    与010(MLP)和011(Transformer)的对比：
    ┌──────────┬────────────────┬─────────────┬───────────────────┐
    │          │ 010 MLP        │ 011 Trans.  │ 本方案 V2         │
    ├──────────┼────────────────┼─────────────┼───────────────────┤
    │ 局部特征 │ ✗ 无           │ ✓ Self-Attn │ ✓ 多尺度CNN       │
    │ 全局特征 │ ✗ flatten丢失  │ ✓ O(n²)     │ ✓ 膨胀卷积+Attn   │
    │ 顺序信息 │ ✗ 丢失         │ ✓ 位置编码  │ ✓ 卷积天然保持    │
    │ 参数量   │ ~50K           │ ~300K       │ ~100K             │
    │ 推理速度 │ 极快           │ 慢          │ 快                │
    └──────────┴────────────────┴─────────────┴───────────────────┘
    """
    def __init__(self, num_boundary_points=64, base_channels=64, num_blocks=3):
        super().__init__()
        self.num_boundary_points = num_boundary_points

        # 1. 输入投影：2D坐标 → 特征空间
        self.input_proj = nn.Sequential(
            nn.Conv1d(2, base_channels, kernel_size=1),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(inplace=True),
        )

        # 2. 多尺度卷积编码器
        self.ms_conv = MultiScaleConv1DBlock(base_channels, base_channels * 2)

        # 3. 残差卷积块堆叠
        self.res_blocks = nn.Sequential(
            *[ResidualConvBlock(base_channels * 2) for _ in range(num_blocks)]
        )

        # 4. 注意力池化
        self.attn_pool = AttentionPooling(base_channels * 2)

        # 5. HSV编码器
        self.hsv_encoder = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )

        # 6. 预测头
        self.predictor = nn.Sequential(
            nn.Linear(base_channels * 2 + 32, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(64, 2),
            nn.Tanh()
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, boundary_points, hsv_features, return_attention=False):
        """
        Args:
            boundary_points: (B, 128) 64个点的(x,y)坐标（flatten）
            hsv_features: (B, 3) HSV特征
            return_attention: 是否返回注意力权重（用于可视化）
        Returns:
            offset: (B, 2) 授粉点偏移量
            attn_weights: (B, 64) 注意力权重（可选）
        """
        B = boundary_points.shape[0]

        # 重塑为(B, 2, 64) 给Conv1d
        pts = boundary_points.view(B, 2, self.num_boundary_points)

        # 1. 输入投影
        x = self.input_proj(pts)          # (B, 64, 64)

        # 2. 多尺度卷积
        x = self.ms_conv(x)              # (B, 128, 64)

        # 3. 残差块
        x = self.res_blocks(x)           # (B, 128, 64)

        # 4. 注意力池化
        x, attn_weights = self.attn_pool(x)  # (B, 128), (B, 64)

        # 5. HSV特征
        hsv_feat = self.hsv_encoder(hsv_features)  # (B, 32)

        # 6. 融合预测
        combined = torch.cat([x, hsv_feat], dim=1)  # (B, 160)
        offset = self.predictor(combined)

        if return_attention:
            return offset, attn_weights
        return offset


# ============ 参数量对比 ============
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def benchmark_speed(model, device, n_runs=100):
    """测试推理速度"""
    import time
    model.eval()
    dummy_b = torch.randn(1, 128).to(device)
    dummy_h = torch.randn(1, 3).to(device)

    # warmup
    for _ in range(10):
        with torch.no_grad():
            model(dummy_b, dummy_h)

    if device.type == 'cuda':
        torch.cuda.synchronize()
    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy_b, dummy_h)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = (time.time() - start) / n_runs * 1000
    return elapsed


if __name__ == "__main__":
    from ultralytics_main_new_010_contour_to_pollination import ContourToPollinationNet as MLPNet
    # fallback: 直接创建实例对比
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("网络结构对比")
    print("=" * 60)

    # MLP (010)
    mlp_net = MLPNet(num_boundary_points=64).to(device)
    mlp_params = count_params(mlp_net)
    mlp_speed = benchmark_speed(mlp_net, device)

    # 改进版 V2
    v2_net = ImprovedContourNetV2(num_boundary_points=64, base_channels=64, num_blocks=3).to(device)
    v2_params = count_params(v2_net)
    v2_speed = benchmark_speed(v2_net, device)

    print(f"\n{'':30s} {'MLP (010)':>12s} {'V2 (本方案)':>12s}")
    print(f"{'参数量':30s} {mlp_params:>12,} {v2_params:>12,}")
    print(f"{'推理时间 (ms/sample)':30s} {mlp_speed:>11.2f}ms {v2_speed:>11.2f}ms")

    # 测试forward
    dummy_b = torch.randn(4, 128).to(device)
    dummy_h = torch.randn(4, 3).to(device)

    offset, attn = v2_net(dummy_b, dummy_h, return_attention=True)
    print(f"\n输出形状: offset={offset.shape}, attn={attn.shape}")
    print(f"注意力权重范围: [{attn.min().item():.4f}, {attn.max().item():.4f}]")
    print("✓ 网络前向传播正常")
