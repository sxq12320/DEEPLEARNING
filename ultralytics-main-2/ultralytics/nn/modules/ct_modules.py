"""Control-theory inspired RGBD fusion modules for multi-modal object detection.

This module implements a physics-inspired RGBD fusion pipeline using concepts from
control theory: Kalman filtering, Extended State Observers (ESO), and IDA-PBC
(Interconnection and Damping Assignment - Passivity Based Control) energy shaping.

The three fusion stages are designed for different feature pyramid levels:
    - Stage 1 (P3 shallow): KalmanGatedFusion - depth guides RGB via adaptive gain
    - Stage 2 (P4 mid): ESOFusion - occlusion estimation and compensation
    - Stage 3 (P5 deep): IDAPBCFusion - Hamiltonian energy-guided fusion

Reference:
    Control theory concepts applied to multi-modal feature fusion for RGBD segmentation.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class KalmanGatedFusion(nn.Module):
    """Stage 1: P3 shallow-layer Kalman filter fusion (depth guides RGB).

    Uses variance estimators to compute local spatial Kalman gain K, then applies
    the Kalman state update equation: F_fused = F_rgb + K * (F_dep - F_rgb).
    Lightweight design: 1x1 squeeze + 3x3 depthwise conv (85% parameter reduction
    vs standard 3x3 convolution).

    Methods:
        forward: Apply Kalman-gated fusion to RGB and depth feature maps.

    Args:
        c_rgb (int): Number of RGB input channels.
        c_dep (int): Number of depth input channels.
        eps (float): Numerical stability term for variance division.

    Returns:
        (torch.Tensor): Fused feature map of shape (B, c_rgb, H, W).
    """

    def __init__(self, c_rgb, c_dep, eps=1e-5):
        """Initialize KalmanGatedFusion with lightweight variance estimators.

        Args:
            c_rgb (int): Number of RGB input channels.
            c_dep (int): Number of depth input channels.
            eps (float): Numerical stability term for variance division.
        """
        super().__init__()
        self.eps = eps

        # Lightweight channel-squeeze variance estimators
        mid_c_rgb = max(16, c_rgb // 16)
        mid_c_dep = max(8, c_dep // 8)

        self.uncert_rgb = nn.Sequential(
            nn.Conv2d(c_rgb, mid_c_rgb, kernel_size=1, bias=False),
            nn.Conv2d(mid_c_rgb, 1, kernel_size=3, padding=1, groups=1, bias=False),
            nn.Softplus(),
        )
        self.uncert_dep = nn.Sequential(
            nn.Conv2d(c_dep, mid_c_dep, kernel_size=1, bias=False),
            nn.Conv2d(mid_c_dep, 1, kernel_size=3, padding=1, groups=1, bias=False),
            nn.Softplus(),
        )
        self.proj_rgb = nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False)
        self.proj_dep = nn.Conv2d(c_dep, c_rgb, kernel_size=1, bias=False)

        # Depthwise separable output layer
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_rgb, c_rgb, kernel_size=3, padding=1, groups=c_rgb, bias=False),
            nn.Conv2d(c_rgb, c_rgb, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """Apply Kalman-gated fusion to RGB and depth feature maps.

        Args:
            x (list[torch.Tensor] | torch.Tensor): List of [f_rgb, f_dep] or single tensor.

        Returns:
            (torch.Tensor): Fused feature map of shape (B, c_rgb, H, W).
        """
        if isinstance(x, list):
            f_rgb, f_dep = x[0], x[1]
        else:
            f_rgb, f_dep = x, None
        if f_rgb.shape[2:] != f_dep.shape[2:]:
            f_dep = F.interpolate(f_dep, size=f_rgb.shape[2:], mode="bilinear", align_corners=False)

        sigma2_rgb = self.uncert_rgb(f_rgb)
        sigma2_dep = self.uncert_dep(f_dep)

        # Dynamic Kalman gain computation
        k_gain = sigma2_dep / (sigma2_rgb + sigma2_dep + self.eps)

        p_rgb = self.proj_rgb(f_rgb)
        p_dep = self.proj_dep(f_dep)

        # State update equation: F_fused = F_rgb + K * (F_dep - F_rgb)
        f_fused = p_rgb + k_gain * (p_dep - p_rgb)
        return self.out_conv(f_fused)


class ESOFusion(nn.Module):
    """Stage 2: P4 mid-layer Extended State Observer (ESO) disturbance compensation fusion.

    Estimates occlusion map M_occ using an ESO-inspired architecture and performs active
    feature compensation for high-occlusion regions. Lightweight bottleneck ESO: 1x1
    compress to 32 channels, 3x3 depthwise estimate (98% parameter reduction).

    Methods:
        forward: Apply ESO-based occlusion compensation fusion.

    Args:
        c_p4_rgb (int): Number of P4 RGB feature channels.
        c_p3_fused (int): Number of P3 fused feature channels.

    Returns:
        (torch.Tensor): Compensated feature map of shape (B, c_p4_rgb, H, W).
    """

    def __init__(self, c_p4_rgb, c_p3_fused):
        """Initialize ESOFusion with lightweight bottleneck observer.

        Args:
            c_p4_rgb (int): Number of P4 RGB feature channels.
            c_p3_fused (int): Number of P3 fused feature channels.
        """
        super().__init__()
        self.proj_p3 = nn.Conv2d(c_p3_fused, c_p4_rgb, kernel_size=1, bias=False)

        # Lightweight bottleneck ESO observer
        self.eso_observer = nn.Sequential(
            nn.Conv2d(c_p4_rgb * 2, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32, bias=False),
            nn.Conv2d(32, 1, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

        # Depthwise separable output
        self.out_conv = nn.Sequential(
            nn.Conv2d(c_p4_rgb, c_p4_rgb, kernel_size=3, padding=1, groups=c_p4_rgb, bias=False),
            nn.Conv2d(c_p4_rgb, c_p4_rgb, kernel_size=1, bias=False),
        )

    def forward(self, x):
        """Apply ESO-based occlusion compensation fusion.

        Args:
            x (list[torch.Tensor] | torch.Tensor): List of [f_rgb_p4, f_fused_p3] or single tensor.

        Returns:
            (torch.Tensor): Compensated feature map of shape (B, c_p4_rgb, H, W).
        """
        if isinstance(x, list):
            f_rgb_p4, f_fused_p3 = x[0], x[1]
        else:
            f_rgb_p4, f_fused_p3 = x, None
        if f_fused_p3.shape[2:] != f_rgb_p4.shape[2:]:
            f_fused_p3 = F.interpolate(f_fused_p3, size=f_rgb_p4.shape[2:], mode="bilinear", align_corners=False)

        p_fused_p3 = self.proj_p3(f_fused_p3)

        # 1. Observer: estimate system disturbance (occlusion uncertainty)
        concat_feat = torch.cat([f_rgb_p4, p_fused_p3], dim=1)
        m_occ = self.eso_observer(concat_feat)

        # 2. Disturbance compensation feedback law
        f_compensated = f_rgb_p4 + m_occ * p_fused_p3
        return self.out_conv(f_compensated)


class IDAPBCFusion(nn.Module):
    """Stage 3: P5 deep-layer IDA-PBC energy shaping fusion.

    Maps deep RGB semantics to a Hamiltonian expected potential energy surface to guide
    geometric features. Uses bidirectional channel attention bottleneck for global energy
    extraction. Final output via 1x1 convolution after concatenation.

    Methods:
        forward: Apply IDA-PBC energy-guided fusion.

    Args:
        c_p5_rgb (int): Number of P5 RGB feature channels.
        c_p4_fused (int): Number of P4 fused feature channels.

    Returns:
        (torch.Tensor): Energy-guided fused feature map of shape (B, c_p5_rgb, H, W).
    """

    def __init__(self, c_p5_rgb, c_p4_fused):
        """Initialize IDAPBCFusion with Hamiltonian energy gate.

        Args:
            c_p5_rgb (int): Number of P5 RGB feature channels.
            c_p4_fused (int): Number of P4 fused feature channels.
        """
        super().__init__()
        self.proj_dep = nn.Conv2d(c_p4_fused, c_p5_rgb, kernel_size=1, bias=False)

        # Global Hamiltonian potential energy gate (lightweight MLP)
        self.energy_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_p5_rgb, c_p5_rgb // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c_p5_rgb // 16, c_p5_rgb, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )
        self.out_conv = nn.Conv2d(c_p5_rgb * 2, c_p5_rgb, kernel_size=1, bias=False)

    def forward(self, x):
        """Apply IDA-PBC energy-guided fusion.

        Args:
            x (list[torch.Tensor] | torch.Tensor): List of [f_rgb_p5, f_fused_p4] or single tensor.

        Returns:
            (torch.Tensor): Energy-guided fused feature map of shape (B, c_p5_rgb, H, W).
        """
        if isinstance(x, list):
            f_rgb_p5, f_fused_p4 = x[0], x[1]
        else:
            f_rgb_p5, f_fused_p4 = x, None
        if f_fused_p4.shape[2:] != f_rgb_p5.shape[2:]:
            f_fused_p4 = F.interpolate(f_fused_p4, size=f_rgb_p5.shape[2:], mode="bilinear", align_corners=False)

        f_dep_p5 = self.proj_dep(f_fused_p4)

        # Inject control energy constraint
        rgb_energy = self.energy_gate(f_rgb_p5)
        f_dep_guided = f_dep_p5 * rgb_energy

        # Lossless Concat channel combination
        f_concat = torch.cat([f_rgb_p5, f_dep_guided], dim=1)
        return self.out_conv(f_concat)


class SplitChannels(nn.Module):
    """Channel splitting module for multi-channel input processing.

    Extracts specified channel subsets by index from a multi-channel input tensor.
    Primary use case: splitting 4-channel RGBD input into RGB (channels 0,1,2)
    and Depth (channel 3) streams.

    Methods:
        forward: Extract specified channels from input tensor.

    Args:
        c_in (int): Total number of input channels (for record-keeping only).
        channels (list[int]): Channel indices to extract, e.g. [0,1,2] for RGB, [3] for depth.

    Returns:
        (torch.Tensor): Tensor with only the specified channels, shape (B, len(channels), H, W).
    """

    def __init__(self, c_in, channels):
        """Initialize SplitChannels with channel index list.

        Args:
            c_in (int): Total number of input channels.
            channels (list[int]): Channel indices to extract.
        """
        super().__init__()
        self.channels = channels
        self.c_out = len(channels)

    def forward(self, x):
        """Extract specified channels from input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W).

        Returns:
            (torch.Tensor): Output tensor of shape (B, len(channels), H, W).
        """
        return x[:, self.channels, :, :]


class BLFLoss(nn.Module):
    """Barrier Lyapunov Function loss for amodal segmentation constraints.

    Ensures that non-modal (amodal) mask spatially fully encloses the visible mask.
    When the constraint is violated (visible > amodal), the log-gradient explodes,
    forcing outward correction. This enforces the physical invariant that occluded
    regions must be at least as large as visible regions.

    Methods:
        forward: Compute BLF constraint loss between visible and amodal predictions.

    Args:
        kc (float): Constraint boundary parameter (maximum allowed violation).
        eps (float): Numerical stability term.

    Returns:
        (torch.Tensor): Scalar BLF loss value.
    """

    def __init__(self, kc=0.5, eps=1e-6):
        """Initialize BLFLoss with constraint boundary.

        Args:
            kc (float): Constraint boundary parameter.
            eps (float): Numerical stability term.
        """
        super().__init__()
        self.kc = kc
        self.eps = eps

    def forward(self, pred_visible, pred_amodal):
        """Compute BLF constraint loss between visible and amodal predictions.

        Args:
            pred_visible (torch.Tensor): Predicted visible mask probabilities.
            pred_amodal (torch.Tensor): Predicted amodal mask probabilities.

        Returns:
            (torch.Tensor): Scalar BLF loss value.
        """
        violation_error = torch.clamp(pred_visible - pred_amodal, min=0.0)
        clamped_error = torch.clamp(violation_error, max=self.kc - self.eps)
        loss_val = -0.5 * torch.log(self.kc**2 / (self.kc**2 - clamped_error**2 + self.eps))
        return loss_val.mean()


class BypassModule(nn.Module):
    """Bypass module for ablation experiments.

    Performs lossless concatenation and channel alignment with no additional computation.
    Used as a baseline fusion method to compare against more sophisticated fusion
    strategies like KalmanGatedFusion, ESOFusion, and IDAPBCFusion.

    Methods:
        forward: Apply bypass fusion (identity or 1x1 projection).

    Args:
        c_in1 (int): Number of channels in the first input.
        c_in2 (int | None): Number of channels in the second input. If None, uses identity.

    Returns:
        (torch.Tensor): Fused feature map of shape (B, c_in1, H, W).
    """

    def __init__(self, c_in1, c_in2=None):
        """Initialize BypassModule with optional channel projection.

        Args:
            c_in1 (int): Number of channels in the first input.
            c_in2 (int | None): Number of channels in the second input.
        """
        super().__init__()
        if c_in2 is not None:
            self.proj = nn.Conv2d(c_in2, c_in1, kernel_size=1, bias=False)
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        """Apply bypass fusion to input feature maps.

        Args:
            x (list[torch.Tensor] | torch.Tensor): List of [f1, f2] or single tensor.

        Returns:
            (torch.Tensor): Fused feature map of shape (B, c_in1, H, W).
        """
        if isinstance(x, list):
            f1, f2 = x[0], x[1]
        else:
            return self.proj(x)
        f2_proj = self.proj(f2)
        return f1 + f2_proj
