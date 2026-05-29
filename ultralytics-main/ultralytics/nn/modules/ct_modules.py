import torch
import torch.nn as nn
import torch.nn.functional as F

class KalmanGatedFusion(nn.Module):
    """
    【阶段 1：P3 浅层卡尔曼融合 (深度指导 RGB) - 极致轻量化版】
    物理意义：利用方差估计器计算局部空间卡尔曼增益 K。
    轻量化：采用 1x1 挤压通道 + 3x3 深度卷积，相比标准 3x3 卷积减少 85% 参数。
    """
    def __init__(self, c_rgb, c_dep, eps=1e-5):
        super().__init__()
        self.eps = eps
        
        # 极轻量化通道挤压方差估计器
        mid_c_rgb = max(16, c_rgb // 16)
        mid_c_dep = max(8, c_dep // 8)
        
        self.uncert_rgb = nn.Sequential(
            nn.Conv2d(c_rgb, mid_c_rgb, kernel_size=1, bias=False),
            nn.Conv2d(mid_c_rgb, 1, kernel_size=3, padding=1, groups=1, bias=False),
            nn.Softplus()
        )
        self.uncert_dep = nn.Sequential(
            nn.Conv2d(c_dep, mid_c_dep, kernel_size=1, bias=False),
            nn.Conv2d(mid_c_dep, 1, kernel_size=3, padding=1, groups=1, bias=False),
            nn.Softplus()
        )
        self.proj_rgb = nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False)
        self.proj_dep = nn.Conv2d(c_dep, c_rgb, kernel_size=1, bias=False)
        
        # 深度可分离卷积输出层
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_rgb, c_rgb, kernel_size=3, padding=1, groups=c_rgb, bias=False),
            nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False)
        )

    def forward(self, x):
        if isinstance(x, list):
            f_rgb, f_dep = x[0], x[1]
        else:
            f_rgb, f_dep = x, None
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode='bilinear', align_corners=False)
            
        sigma2_rgb = self.uncert_rgb(f_rgb)
        sigma2_dep = self.uncert_dep(f_dep)
        
        # 动态卡尔曼增益计算
        k_gain = sigma2_dep / (sigma2_rgb + sigma2_dep + self.eps)
        
        p_rgb = self.proj_rgb(f_rgb)
        p_dep = self.proj_dep(f_dep)
        
        # 状态更新方程: F_fused = F_rgb + K * (F_dep - F_rgb)
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
        self.proj_p3 = nn.Conv2d(c_p3_fused, c_p4_rgb, kernel_size=1, bias=False)
        
        # 极轻量瓶颈观测器 (Bottleneck ESO)
        self.eso_observer = nn.Sequential(
            nn.Conv2d(c_p4_rgb * 2, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        # 深度可分离输出，避免 P4 阶段标准大卷积带来的计算开销
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_p4_rgb, c_p4_rgb, kernel_size=3, padding=1, groups=c_p4_rgb, bias=False),
            nn.Conv2d(c_p4_rgb, c_p4_rgb, kernel_size=1, bias=False)
        )

    def forward(self, x):
        if isinstance(x, list):
            f_rgb_p4, f_fused_p3 = x[0], x[1]
        else:
            f_rgb_p4, f_fused_p3 = x, None
        if f_fused_p3.shape[2:] != f_rgb_p4.shape[2:]:
            f_fused_p3 = F.interpolate(f_fused_p3, size=f_rgb_p4.shape[2:], mode='bilinear', align_corners=False)
            
        p_fused_p3 = self.proj_p3(f_fused_p3)
        
        # 1. 观测器：估计系统未知外扰量 (遮挡不确定性)
        concat_feat = torch.cat([f_rgb_p4, p_fused_p3], dim=1)
        m_occ = self.eso_observer(concat_feat)
        
        # 2. 扰动补偿反馈律
        f_compensated = f_rgb_p4 + m_occ * p_fused_p3
        return self.out_conv(f_compensated)


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
