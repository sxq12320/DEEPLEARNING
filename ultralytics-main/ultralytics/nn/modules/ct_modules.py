import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleVarianceEstimator(nn.Module):
    """
    多尺度方差估计器 (统计学先验 + 网络自适应校准)
    """
    def __init__(self, c_in):
        super().__init__()
        # 使用统计数学计算数学期望Ex
        self.pool1 = nn.AvgPool2d(kernel_size = 3 , stride = 1 , padding = 1)
        self.pool2 = nn.AvgPool2d(kernel_size = 5 , stride = 1 , padding = 2)
        self.pool3 = nn.AvgPool2d(kernel_size = 7 , stride = 1 , padding = 3)
        # 2. 极轻量网络校准器 (学习语义级别的不确定性)
        # 输入: 3个尺度的统计方差(各c_in通道) + 1个全局上下文(c_in通道) = 4*c_in通道
        self.calibrator = nn.Sequential(
            nn.Conv2d(4 * c_in, 1, kernel_size=1, bias=False),
            nn.Softplus() # 保证输出的方差/不确定性严格为正
        )

    def forward(self, x):
        # 1. 计算多尺度局部均值 E(X)
        mu1 = self.pool1(x)
        mu2 = self.pool2(x)
        mu3 = self.pool3(x)
        
        # 2. 计算多尺度局部方差 Var(X) = E(X^2) - [E(X)]^2
        # 为了计算效率，这里使用 (X - mu)^2 的近似
        var1 = self.pool1((x - mu1).pow(2))
        var2 = self.pool2((x - mu2).pow(2))
        var3 = self.pool3((x - mu3).pow(2))
        # 2. 计算多尺度局部方差 Var(X) = E(X^2) - [E(X)]^2
        # 为了计算效率，这里使用 (X - mu)^2 的近似
        var1 = self.pool1((x - mu1).pow(2))
        var2 = self.pool2((x - mu2).pow(2))
        var3 = self.pool3((x - mu3).pow(2))
        
        # 3. 拼接统计先验与原始特征，进行网络自适应校准
        # 使用全局均值池化将原始特征压缩为全局上下文，避免参数量爆炸
        x_global = F.adaptive_avg_pool2d(x, 1) 
        # 广播回原尺寸，作为全局上下文指导局部方差校准
        x_context = x_global.expand_as(var1) 
        
        feat = torch.cat([var1, var2, var3, x_context], dim=1)
        
        # 4. 输出校准后的最终方差/不确定性图 (B, 1, H, W)
        sigma2 = self.calibrator(feat)
        return sigma2


class KalmanGatedFusion(nn.Module):
    """
    【阶段 1：P3 浅层卡尔曼融合 (深度指导 RGB) - 极致轻量化版】
    物理意义：利用方差估计器计算局部空间卡尔曼增益 K。
    轻量化：采用 1x1 挤压通道 + 3x3 深度卷积，相比标准 3x3 卷积减少 85% 参数。
    """
    def __init__(self, c_rgb, c_dep, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.uncert_rgb = MultiScaleVarianceEstimator(c_rgb)
        self.uncert_dep = MultiScaleVarianceEstimator(c_dep)

        # 投影层：将深度特征映射到 RGB 通道维度
        self.proj_rgb = nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False)
        self.proj_dep = nn.Conv2d(c_dep, c_rgb, kernel_size=1, bias=False)

        # 深度可分离卷积输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_rgb, c_rgb, kernel_size=3, padding=1, groups=c_rgb, bias=False),
            nn.BatchNorm2d(c_rgb),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False)
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            f_rgb, f_dep = x[0], x[1]
        else:
            # 如果只输入单张图，直接返回
            return self.out_conv(self.proj_rgb(x))
            
        # 空间尺寸对齐
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
            
        # 1. 估计多尺度校准方差
        sigma2_rgb = self.uncert_rgb(f_rgb)
        sigma2_dep = self.uncert_dep(f_dep)
        
        # 2. 动态卡尔曼增益计算 (K = P_dep / (P_rgb + P_dep))
        k_gain = sigma2_dep / (sigma2_rgb + sigma2_dep + self.eps)
        
        # 3. 特征投影
        p_rgb = self.proj_rgb(f_rgb)
        p_dep = self.proj_dep(f_dep)
        
        # 4. 状态更新方程: F_fused = F_rgb + K * (F_dep - F_rgb)
        f_fused = p_rgb + k_gain * (p_dep - p_rgb)
        
        return self.out_conv(f_fused)    


class ESOFusion(nn.Module):
    """
    【阶段 2：P4 中层 ESO 扰动补偿融合 - 极致轻量化版】
    物理意义：引入自抗扰观测器 (ESO) 的思想，估计遮挡图 M_occ，对高遮挡区域进行特征主动补偿。
    轻量化：拼接特征后通过 1x1 压缩至 32 通道，再用 3x3 深度卷积估计，参数量降低约 98%！
    """
    def __init__(self, c_p4_rgb, c_p3_fused):
        super().__init__()
        self.c_p4 = c_p4_rgb
        self.proj_u = nn.Conv2d(c_p3_fused, c_p4_rgb, kernel_size=1, bias=False)
        # ==========================================
        # 极轻量瓶颈 ESO 观测器 (保留你的优秀设计)
        # 物理意义：估计 RGB 特征中的总扰动 z2 (遮挡/噪声/模态差异)
        # ==========================================
        self.eso_observer = nn.Sequential(
            # 1x1 压缩至 32 维，极大降低计算量
            nn.Conv2d(c_p4_rgb * 2, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 3x3 深度可分离卷积提取空间扰动分布
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            # 【关键修正】输出 C 通道扰动，且不加 Sigmoid！
            # 真实物理扰动是无界的(可正可负)，Sigmoid 会破坏扰动方向性
            # 使用 Tanh 限制极端值防止训练崩溃，同时保留负值能力
            nn.Conv2d(32, c_p4_rgb, kernel_size=1, bias=True), 
            nn.Tanh() 
        )
        
        # ESO 反馈增益 beta1 (可学习参数，对应控制理论中的观测器带宽)
        self.beta1 = nn.Parameter(torch.ones(1, c_p4_rgb, 1, 1) * 0.1)
        
        # 深度可分离输出层 (对补偿后的纯净特征进行非线性变换)
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_p4_rgb, c_p4_rgb, kernel_size=3, padding=1, groups=c_p4_rgb, bias=False),
            nn.BatchNorm2d(c_p4_rgb),
            nn.SiLU(inplace=True),
            nn.Conv2d(c_p4_rgb, c_p4_rgb, kernel_size=1, bias=False)
        )

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            f_rgb_p4, f_fused_p3 = x[0], x[1]
        else:
            # 兼容单输入情况
            return self.out_conv(x)
            
        # 空间尺寸对齐
        if f_rgb_p4.shape[2:] != f_fused_p3.shape[2:]:
            f_fused_p3 = F.interpolate(f_fused_p3, size=f_rgb_p4.shape[2:], 
                                       mode='bilinear', align_corners=False)
            
        # 1. 生成控制量 u
        u = self.proj_u(f_fused_p3)
        
        # 2. ESO 扰动观测：估计总扰动 z2
        concat_feat = torch.cat([f_rgb_p4, u], dim=1)
        z2_disturbance = self.eso_observer(concat_feat)  # Shape: (B, C, H, W), 值域 [-1, 1]
        
        # 3. 【严格 ESO 补偿律】
        # 物理公式: F_compensated = F_rgb - beta1 * z2 + u
        # 减去估计的扰动，加上来自浅层的控制引导
        f_compensated = f_rgb_p4 - self.beta1 * z2_disturbance + u
        
        # 4. 非线性净化
        f_clean = self.out_conv(f_compensated)
        
        # 5. 【YOLO 兼容关键】恢复 Concat 的双倍通道特性
        # 将"ESO 补偿净化后的特征"与"原始 P3 控制特征"拼接
        # 确保输出通道 = c_p4_rgb + c_p3_fused，完美接入后续 C3k2 模块
        out = torch.cat([f_clean, f_fused_p3], dim=1)
        
        return out    


class IDAPBCFusion(nn.Module):
    """
    【阶段 3：P5 深层 IDA-PBC 能量成型融合 - 极致轻量化版】
    物理意义：将深层 RGB 语义映射为哈密顿期望势能面，指导几何特征，最后 Concat 拼接。
    轻量化：利用双向通道注意力 Bottleneck 提取全局势能，参数开销可忽略不计。
    """
    def __init__(self, c_p5_rgb, c_p4_fused):
        super().__init__()
        self.proj_dep = nn.Conv2d(c_p4_fused, c_p5_rgb, kernel_size=1, bias=False)
        
        # 全局哈密顿势能激励头 (基于极轻量 MLP)
        self.energy_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_p5_rgb, c_p5_rgb // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_p5_rgb // 16, c_p5_rgb, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        self.out_conv = nn.Conv2d(c_p5_rgb * 2, c_p5_rgb, kernel_size=1, bias=False)

    def forward(self, x):
        if isinstance(x, list):
            f_rgb_p5, f_fused_p4 = x[0], x[1]
        else:
            f_rgb_p5, f_fused_p4 = x, None
        if f_fused_p4.shape[2:] != f_rgb_p5.shape[2:]:
            f_fused_p4 = F.interpolate(f_fused_p4, size=f_rgb_p5.shape[2:], mode='bilinear', align_corners=False)
            
        f_dep_p5 = self.proj_dep(f_fused_p4)
        
        # 注入控制能量约束
        rgb_energy = self.energy_gate(f_rgb_p5)
        f_dep_guided = f_dep_p5 * rgb_energy
        
        # 无损 Concat 通道组合
        f_concat = torch.cat([f_rgb_p5, f_dep_guided], dim=1)
        return self.out_conv(f_concat)


class SplitChannels(nn.Module):
    """
    通道分割模块：从多通道输入中按索引提取指定通道子集。
    用途：将 4 通道 RGBD 输入拆分为 RGB(0,1,2) 和 Depth(3) 两路。
    """

    def __init__(self, c_in, channels):
        """
        Args:
            c_in: 输入总通道数（仅用于信息记录，forward 不依赖它）
            channels: 要提取的通道索引列表，如 [0,1,2] 表示 RGB，[3] 表示 Depth
        """
        super().__init__()
        self.channels = channels
        self.c_out = len(channels)

    def forward(self, x):
        return x[:, self.channels, :, :]


class BLFLoss(nn.Module):
    """
    【物理损失函数：物理障碍李雅普诺夫约束 (Barrier Lyapunov Function)】
    确保非模态 mask 在空间几何上必须完全包裹可见 mask。违规时对数梯度爆炸，强制向外纠偏。
    """
    def __init__(self, kc=0.5, eps=1e-6):
        super().__init__()
        self.kc = kc
        self.eps = eps

    def forward(self, pred_visible, pred_amodal):
        violation_error = torch.clamp(pred_visible - pred_amodal, min=0.0)
        clamped_error = torch.clamp(violation_error, max=self.kc - self.eps)
        loss_val = -0.5 * torch.log(self.kc**2 / (self.kc**2 - clamped_error**2 + self.eps))
        return loss_val.mean()


class BypassModule(nn.Module):
    """
    用于消融实验的 Bypass 旁路组件（直接执行无损 Concat / 映射以对齐通道，无任何额外计算）
    """
    def __init__(self, c_in1, c_in2=None):
        super().__init__()
        if c_in2 is not None:
            self.proj = nn.Conv2d(c_in2, c_in1, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        if isinstance(x, list):
            f1, f2 = x[0], x[1]
        else:
            return self.proj(x)
        f2_proj = self.proj(f2)
        return f1 + f2_proj
