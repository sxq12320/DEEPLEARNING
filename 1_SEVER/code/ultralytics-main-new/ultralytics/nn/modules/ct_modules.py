import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleVarianceEstimator(nn.Module):
    """
    多尺度方差估计器 (统计学先验 + 网络自适应校准)
    输出 groups 个通道的不确定性图，保留不同通道组对边缘方向的响应差异。
    """
    def __init__(self, c_in, groups=4):
        super().__init__()
        self.groups = groups
        # 使用统计数学计算数学期望 E(X)
        self.pool1 = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.AvgPool2d(kernel_size=5, stride=1, padding=2)
        self.pool3 = nn.AvgPool2d(kernel_size=7, stride=1, padding=3)
        # 极轻量网络校准器 (学习语义级别的不确定性)
        # 输入: 3个尺度的统计方差(各c_in通道) + 1个全局上下文(c_in通道) = 4*c_in通道
        # 输出: groups 个通道的不确定性图，而非单一标量，保留通道组间差异
        self.calibrator = nn.Sequential(
            nn.Conv2d(4 * c_in, groups, kernel_size=1, bias=False),
            nn.Softplus()  # 保证输出的方差/不确定性严格为正
        )

    def forward(self, x):
        # 1. 计算多尺度局部均值 E(X)
        mu1 = self.pool1(x)
        mu2 = self.pool2(x)
        mu3 = self.pool3(x)

        # 2. 计算多尺度局部方差 Var(X) = E[(X - E[X])^2]
        var1 = self.pool1((x - mu1).pow(2))
        var2 = self.pool2((x - mu2).pow(2))
        var3 = self.pool3((x - mu3).pow(2))

        # 3. 拼接统计先验与原始特征，进行网络自适应校准
        # 使用全局均值池化将原始特征压缩为全局上下文，避免参数量爆炸
        x_global = F.adaptive_avg_pool2d(x, 1)
        # 广播回原尺寸，作为全局上下文指导局部方差校准
        x_context = x_global.expand_as(var1)

        feat = torch.cat([var1, var2, var3, x_context], dim=1)

        # 4. 输出校准后的最终方差/不确定性图 (B, groups, H, W)
        sigma2 = self.calibrator(feat)
        return sigma2


class KalmanGatedFusion(nn.Module):
    """
    【阶段 1：P3 浅层卡尔曼融合 (深度指导 RGB) - 极致轻量化版】
    物理意义：利用方差估计器计算局部空间卡尔曼增益 K。
    修正后的卡尔曼增益：K = sigma2_rgb / (sigma2_rgb + sigma2_dep + eps)
    - 当 RGB 不确定性大（暗光/弱纹理）时，K → 1，信任 Depth；
    - 当 Depth 不确定性大（噪声/空洞）时，K → 0，信任 RGB。
    这才是真正的卡尔曼最优估计：不确定性大的观测应被削弱。
    轻量化：采用 1x1 挤压通道 + 3x3 深度卷积，相比标准 3x3 卷积减少 85% 参数。
    """
    def __init__(self, c_rgb, c_dep, eps=1e-5, groups=4):
        super().__init__()
        self.eps = eps
        self.groups = groups
        self.uncert_rgb = MultiScaleVarianceEstimator(c_rgb, groups=groups)
        self.uncert_dep = MultiScaleVarianceEstimator(c_dep, groups=groups)

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

        # 1. 估计多尺度校准方差（不确定性）
        # sigma2_rgb: (B, groups, H, W)，sigma2_dep: (B, groups, H, W)
        sigma2_rgb = self.uncert_rgb(f_rgb)
        sigma2_dep = self.uncert_dep(f_dep)

        # 2. 修正后的卡尔曼增益计算
        # 物理意义：K = sigma2_rgb / (sigma2_rgb + sigma2_dep + eps)
        # sigma2 代表观测不确定性(Uncertainty)，K 越大表示 RGB 越不可靠，应更多信任 Depth
        # 当 RGB 不确定性大（暗光/弱纹理）→ K → 1 → 信任 Depth
        # 当 Depth 不确定性大（噪声/空洞）→ K → 0 → 信任 RGB
        k_gain = sigma2_rgb / (sigma2_rgb + sigma2_dep + self.eps)  # (B, groups, H, W)

        # 3. 将 groups 通道的卡尔曼增益映射到全部 c_rgb 通道
        # 每个 group 的增益被复制到对应通道组，保留不同通道组对边缘方向的响应差异
        c_rgb = f_rgb.shape[1]
        channels_per_group = c_rgb // self.groups
        k_gain_full = k_gain.repeat_interleave(channels_per_group, dim=1)  # (B, c_rgb, H, W)

        # 4. 特征投影
        p_rgb = self.proj_rgb(f_rgb)
        p_dep = self.proj_dep(f_dep)

        # 5. 卡尔曼状态更新方程: F_fused = F_rgb + K * (F_dep - F_rgb)
        # K → 1 时：F_fused → F_dep（RGB不可靠，信任Depth）
        # K → 0 时：F_fused → F_rgb（Depth不可靠，信任RGB）
        f_fused = p_rgb + k_gain_full * (p_dep - p_rgb)

        return self.out_conv(f_fused)


class ESOFusion(nn.Module):
    """
    【阶段 2：P4 中层 ESO 扰动补偿融合 - 两步迭代观测版】
    物理意义：引入自抗扰观测器 (ESO) 的动态迭代收敛过程。
    原版只有单次静态前馈，无法模拟 ESO 的收敛特性。
    修改为两步迭代观测：
      Step 1 (粗观测)：初步估计总扰动并补偿
      Step 2 (精观测)：基于初步补偿后的特征，估计残差扰动并精细补偿
    补偿增益 beta1/beta2 改为可学习的 Sigmoid 门控向量，避免训练初期梯度震荡。
    """
    def __init__(self, c_p4_rgb, c_p3_fused):
        super().__init__()
        self.c_p4 = c_p4_rgb
        self.proj_u = nn.Conv2d(c_p3_fused, c_p4_rgb, kernel_size=1, bias=False)

        # ==========================================
        # Step 1: 粗观测器 Observer_1
        # 物理意义：初步估计 RGB 特征中的总扰动 z2^(1) (遮挡/噪声/模态差异)
        # ==========================================
        self.eso_observer_1 = nn.Sequential(
            nn.Conv2d(c_p4_rgb * 2, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, c_p4_rgb, kernel_size=1, bias=True),
            nn.Tanh()  # 保留负值能力，限制极端值
        )

        # ==========================================
        # Step 2: 精观测器 Observer_2
        # 物理意义：基于初步补偿后的特征，估计残差扰动 z2^(2)
        # 输入为 [f^(1), u]，即初步补偿特征与控制量的拼接
        # ==========================================
        self.eso_observer_2 = nn.Sequential(
            nn.Conv2d(c_p4_rgb * 2, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, c_p4_rgb, kernel_size=1, bias=True),
            nn.Tanh()
        )

        # ESO 反馈增益 beta1, beta2 (可学习的 Sigmoid 门控向量)
        # 物理意义：对应控制理论中的观测器带宽，决定补偿强度
        # 使用 Sigmoid 门控：输出范围 (0, 1)，初始值约 0.1，避免训练初期梯度震荡
        # 网络自适应决定每个通道的补偿强度
        self.beta1_raw = nn.Parameter(torch.full((1, c_p4_rgb, 1, 1), -2.2))  # sigmoid(-2.2) ≈ 0.1
        self.beta2_raw = nn.Parameter(torch.full((1, c_p4_rgb, 1, 1), -2.2))  # sigmoid(-2.2) ≈ 0.1

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

        # 1. 生成控制量 u (来自浅层 P3 的融合特征)
        u = self.proj_u(f_fused_p3)

        # 2. 计算自适应补偿增益
        beta1 = torch.sigmoid(self.beta1_raw)  # (1, C, 1, 1), 初始 ≈ 0.1
        beta2 = torch.sigmoid(self.beta2_raw)  # (1, C, 1, 1), 初始 ≈ 0.1

        # ==========================================
        # Step 1: 粗观测 (Coarse Observation)
        # 拼接原始 RGB 特征与控制量，初步估计总扰动
        # z2^(1) = Observer_1([f_rgb, u])
        # ==========================================
        concat_feat_1 = torch.cat([f_rgb_p4, u], dim=1)
        z2_coarse = self.eso_observer_1(concat_feat_1)  # (B, C, H, W), 值域 [-1, 1]

        # 粗补偿：f^(1) = f_rgb - beta1 * z2^(1) + u
        # 减去估计的扰动，加上来自浅层的控制引导
        f_step1 = f_rgb_p4 - beta1 * z2_coarse + u

        # ==========================================
        # Step 2: 精观测 (Fine Observation)
        # 基于初步补偿后的特征，估计残差扰动
        # z2^(2) = Observer_2([f^(1), u])
        # ==========================================
        concat_feat_2 = torch.cat([f_step1, u], dim=1)
        z2_fine = self.eso_observer_2(concat_feat_2)  # (B, C, H, W), 值域 [-1, 1]

        # 精细补偿：f_comp = f^(1) - beta2 * z2^(2)
        # 进一步消除残余扰动，实现 ESO 的动态收敛
        f_compensated = f_step1 - beta2 * z2_fine

        # 3. 非线性净化
        f_clean = self.out_conv(f_compensated)

        # 4. 【YOLO 兼容关键】恢复 Concat 的双倍通道特性
        # 将"ESO 补偿净化后的特征"与"原始 P3 控制特征"拼接
        # 确保输出通道 = c_p4_rgb + c_p3_fused，完美接入后续 C3k2 模块
        out = torch.cat([f_clean, f_fused_p3], dim=1)

        return out


class IDAPBCFusion(nn.Module):
    """
    【阶段 3：P5 深层 IDA-PBC 能量成型融合 - 空间-通道联合势能版】
    物理意义：将深层 RGB 语义映射为哈密顿期望势能面，指导几何特征。
    修正：原版只有通道注意力(GAP)，AdaptiveAvgPool2d(1) 完全丢失了深层宝贵的空间语义位置信息。
    修改为空间-通道联合势能建模：
      - 通道势能 (Channel Energy)：GAP + MLP + Sigmoid → (B, C, 1, 1)，提取全局类别语义权重
      - 空间势能 (Spatial Energy)：Mean+Max 压缩 + 7x7 Conv + Sigmoid → (B, 1, H, W)，提取空间语义显著性图
    双重能量成型：f_dep_guided = f_dep_p5 * channel_energy * spatial_energy
    实现"知道是什么（通道）"且"知道在哪里（空间）"的深层语义对齐。
    """
    def __init__(self, c_p5_rgb, c_p4_fused):
        super().__init__()
        self.proj_dep = nn.Conv2d(c_p4_fused, c_p5_rgb, kernel_size=1, bias=False)

        # ==========================================
        # 通道势能分支 (Channel Energy Branch)
        # 物理意义：提取全局类别语义权重，回答"是什么"
        # GAP + MLP + Sigmoid → (B, C, 1, 1)
        # ==========================================
        self.channel_energy = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_p5_rgb, c_p5_rgb // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_p5_rgb // 16, c_p5_rgb, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # ==========================================
        # 空间势能分支 (Spatial Energy Branch)
        # 物理意义：提取空间语义显著性图，回答"在哪里"
        # 通道维度 Mean+Max 压缩 → (B, 2, H, W) → 7x7 Conv + Sigmoid → (B, 1, H, W)
        # 使用 7x7 大核卷积捕获深层大范围空间语义关系
        # ==========================================
        self.spatial_energy = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(c_p5_rgb * 2, c_p5_rgb, kernel_size=1, bias=False)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            f_rgb_p5, f_fused_p4 = x[0], x[1]
        else:
            f_rgb_p5, f_fused_p4 = x, None

        # 单输入降级：无深度特征时直接返回 RGB 投影
        if f_fused_p4 is None:
            return self.out_conv(torch.cat([f_rgb_p5, torch.zeros_like(f_rgb_p5)], dim=1))

        # 空间尺寸对齐
        if f_fused_p4.shape[2:] != f_rgb_p5.shape[2:]:
            f_fused_p4 = F.interpolate(f_fused_p4, size=f_rgb_p5.shape[2:], mode='bilinear', align_corners=False)

        f_dep_p5 = self.proj_dep(f_fused_p4)

        # ==========================================
        # 通道势能：全局类别语义权重 (B, C, 1, 1)
        # 回答"是什么"——哪些语义通道在当前场景中是重要的
        # ==========================================
        ch_energy = self.channel_energy(f_rgb_p5)  # (B, C, 1, 1)

        # ==========================================
        # 空间势能：空间语义显著性图 (B, 1, H, W)
        # 回答"在哪里"——哪些空间位置包含关键语义信息
        # 在通道维度上进行 Mean+Max 双重压缩，保留互补的空间统计信息
        # ==========================================
        rgb_mean = torch.mean(f_rgb_p5, dim=1, keepdim=True)  # (B, 1, H, W)
        rgb_max = torch.max(f_rgb_p5, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        spatial_input = torch.cat([rgb_mean, rgb_max], dim=1)  # (B, 2, H, W)
        sp_energy = self.spatial_energy(spatial_input)  # (B, 1, H, W)

        # ==========================================
        # 双重能量成型 (Dual Energy Shaping)
        # f_dep_guided = f_dep_p5 * channel_energy * spatial_energy
        # 通道权重决定"哪些语义通道应该被激活"
        # 空间权重决定"哪些空间位置应该被增强"
        # 两者联合实现深层语义对齐：既知道是什么，又知道在哪里
        # ==========================================
        f_dep_guided = f_dep_p5 * ch_energy * sp_energy

        # 无损 Concat 通道组合
        f_concat = torch.cat([f_rgb_p5, f_dep_guided], dim=1)
        return self.out_conv(f_concat)


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


class APFM(nn.Module):
    """
    Attention Parallel Feature Mixer (APFM)
    同时建模 Channel-wise 和 Spatial-wise 注意力，
    对两路互补特征 FA, FB 进行自适应加权融合。
    支持两路不同通道数输入（通过 1x1 Conv 投影对齐）。
    Args:
        c1 (int): 第一路输入特征通道数 (输出通道数也等于 c1)
        c2 (int): 第二路输入特征通道数
        reduction (int): 通道压缩比，默认为 4
    """
    def __init__(self, c1: int, c2: int, reduction: int = 4):
        super().__init__()
        # 如果通道数不同，投影第二路到 c1
        self.proj_b = nn.Conv2d(c2, c1, kernel_size=1, bias=False) if c2 != c1 else nn.Identity()
        in_channels = c1
        mid_channels = max(in_channels // reduction, 1)
        # ------------------------------------------------------------------
        # Channel Context 分支 (对应论文 Figure 2b 左侧)
        # GAP -> 1x1 Conv -> Norm -> SiLU -> 1x1 Conv -> Norm
        # ------------------------------------------------------------------
        self.channel_branch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
        )
        # ------------------------------------------------------------------
        # Spatial Context 分支 (对应论文 Figure 2b 右侧)
        # GMP -> 1x1 Conv -> Norm -> SiLU -> 1x1 Conv -> Norm
        # ------------------------------------------------------------------
        self.spatial_branch = nn.Sequential(
            nn.Conv2d(in_channels * 2, mid_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=mid_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(mid_channels, in_channels, kernel_size=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
        )
        # ------------------------------------------------------------------
        # 融合两路上下文后的 3x3 Conv (论文 Figure 2b 底部)
        # ------------------------------------------------------------------
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=1, num_channels=in_channels),
            nn.SiLU(inplace=True),
        )
        # 最终 Sigmoid: 生成融合权重 w ∈ (0, 1)
        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        if isinstance(x, (list, tuple)):
            fa, fb = x[0], x[1]
        else:
            return x
        # 空间尺寸对齐
        if fa.shape[2:] != fb.shape[2:]:
            fb = F.interpolate(fb, size=fa.shape[2:], mode='bilinear', align_corners=False)
        # 投影第二路通道到 c1
        fb = self.proj_b(fb)
        # 拼接两路特征: [B, 2C, H, W]
        f_cat = torch.cat([fa, fb], dim=1)
        # ---- Channel Context ----
        gap = F.adaptive_avg_pool2d(f_cat, 1)
        ch_ctx = self.channel_branch(gap)          # [B, C, 1, 1]
        # ---- Spatial Context ----
        gmp = F.adaptive_max_pool2d(f_cat, 1)
        sp_ctx = self.spatial_branch(gmp)          # [B, C, 1, 1]
        # ---- 合并两路上下文 ----
        ctx_cat = torch.cat([ch_ctx, sp_ctx], dim=1)
        ctx_fused = self.fusion_conv(ctx_cat)       # [B, C, 1, 1]
        # ---- 生成权重并自适应融合 ----
        w = self.sigmoid(ctx_fused)                 # [B, C, 1, 1]
        ff = w * fa + (1.0 - w) * fb               # [B, C, H, W]
        return ff


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
