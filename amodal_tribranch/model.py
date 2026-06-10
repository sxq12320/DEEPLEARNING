"""
Amodal Tri-Branch Segmentation Model based on YOLO11 Backbone & Neck.

Architecture:
  - Backbone: YOLO11 CSPDarknet (C3k2 + SPPF + C2PSA)
  - Neck: FPN + PAN (multi-scale feature fusion)
  - Head: Custom Tri-Branch Amodal Head (vis / occ / full)

The original YOLO Segment head (Proto + Mask Coefficients) is replaced by
three parallel pixel-level mask prediction branches for amodal segmentation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from copy import deepcopy

from ultralytics.nn.tasks import parse_model
from ultralytics.nn.modules.conv import Conv


class DilatedConvBNAct(nn.Module):
    """Dilated Convolution with BN and SiLU activation, for expanding receptive field."""

    def __init__(self, c1, c2, k=3, dilation=2, act=True):
        super().__init__()
        padding = dilation * (k - 1) // 2
        self.conv = nn.Conv2d(c1, c2, k, padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TriBranchAmodalHead(nn.Module):
    """Tri-branch amodal segmentation head.

    Takes P3 features from the YOLO neck (stride=8) and produces three
    parallel mask predictions: visible, occluded, and full (amodal).

    Architecture:
        Decoder: P3 -> upsample 2x -> shared features
        head_vis:  Conv -> Conv1x1 -> pred_vis   (standard conv)
        head_occ:  DilatedConv -> Conv1x1 -> pred_occ (dilated conv for larger RF)
        head_full: Concat(vis_feat, occ_feat) -> Conv -> Conv1x1 -> pred_full
    """

    def __init__(self, in_channels, mid_channels=128):
        super().__init__()

        # ---- Decoder: upsample P3 (stride=8) to stride=4 ----
        self.decoder = nn.Sequential(
            Conv(in_channels, mid_channels, k=3),
            nn.ConvTranspose2d(mid_channels, mid_channels, kernel_size=2, stride=2, bias=True),
            Conv(mid_channels, mid_channels, k=3),
        )

        # ---- Visible branch (standard convolution) ----
        self.vis_feat_conv = Conv(mid_channels, mid_channels // 2, k=3)
        self.vis_out_conv = nn.Conv2d(mid_channels // 2, 1, kernel_size=1)

        # ---- Occluded branch (dilated convolution for larger receptive field) ----
        self.occ_feat_conv = DilatedConvBNAct(mid_channels, mid_channels // 2, k=3, dilation=2)
        self.occ_out_conv = nn.Conv2d(mid_channels // 2, 1, kernel_size=1)

        # ---- Full branch (concat vis + occ intermediate features) ----
        self.full_feat_conv = Conv(mid_channels, mid_channels // 2, k=3)  # mid_channels = vis + occ concat
        self.full_out_conv = nn.Conv2d(mid_channels // 2, 1, kernel_size=1)

        # Initialize output conv biases to -2.0 for stable initial sigmoid ~0.12
        for m in [self.vis_out_conv, self.occ_out_conv, self.full_out_conv]:
            nn.init.constant_(m.bias, -2.0)

    def forward(self, p3_feat):
        """
        Args:
            p3_feat: [B, C, H/8, W/8] P3 feature map from YOLO neck

        Returns:
            pred_vis:  [B, 1, H/4, W/4] visible mask logits
            pred_occ:  [B, 1, H/4, W/4] occluded mask logits
            pred_full: [B, 1, H/4, W/4] full (amodal) mask logits
        """
        shared = self.decoder(p3_feat)  # [B, mid_ch, H/4, W/4]

        # Visible branch
        vis_feat = self.vis_feat_conv(shared)   # [B, mid_ch//2, H/4, W/4]
        pred_vis = self.vis_out_conv(vis_feat)  # [B, 1, H/4, W/4]

        # Occluded branch
        occ_feat = self.occ_feat_conv(shared)   # [B, mid_ch//2, H/4, W/4]
        pred_occ = self.occ_out_conv(occ_feat)  # [B, 1, H/4, W/4]

        # Full branch (concat intermediate features from vis and occ)
        full_feat = torch.cat([vis_feat, occ_feat], dim=1)  # [B, mid_ch, H/4, W/4]
        full_feat = self.full_feat_conv(full_feat)            # [B, mid_ch//2, H/4, W/4]
        pred_full = self.full_out_conv(full_feat)             # [B, 1, H/4, W/4]

        return pred_vis, pred_occ, pred_full


class AmodalYOLO(nn.Module):
    """Amodal segmentation model based on YOLO11 backbone + neck + tri-branch head.

    This model:
    1. Parses a YOLO11 YAML config to build the backbone and neck (FPN+PAN)
    2. Removes the original Segment head
    3. Attaches a custom TriBranchAmodalHead on the P3 feature map
    4. Provides a clean forward interface returning three mask predictions
    """

    def __init__(self, cfg, ch=4, nc=1, p3_layer_idx=16, head_mid_channels=128):
        """
        Args:
            cfg: path to YOLO11-seg YAML config file
            ch: input channels (4 for RGBD)
            nc: number of classes (unused in tri-branch head, kept for YAML compatibility)
            p3_layer_idx: layer index that outputs P3 features (default 16 for YOLO11-seg)
            head_mid_channels: intermediate channels in the tri-branch head
        """
        super().__init__()
        self.p3_layer_idx = p3_layer_idx

        # ---- Build backbone + neck from YAML ----
        with open(cfg, encoding='utf-8') as f:
            d = yaml.safe_load(f)

        d['channels'] = ch
        if nc and nc != d.get('nc', nc):
            d['nc'] = nc

        # Remove the last head entry (Segment head) — we replace it with our tri-branch head
        d['head'] = d['head'][:-1]

        self.backbone_neck, self.save = parse_model(deepcopy(d), ch=ch, verbose=True)

        # ---- Determine P3 output channels via dummy forward ----
        with torch.no_grad():
            dummy = torch.zeros(1, ch, 64, 64)
            p3_feat = self._extract_p3(dummy)
            p3_channels = p3_feat.shape[1]

        # ---- Tri-branch amodal head ----
        self.tri_head = TriBranchAmodalHead(p3_channels, mid_channels=head_mid_channels)

        # Print model info
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"[AmodalYOLO] P3 layer index: {p3_layer_idx}, P3 channels: {p3_channels}")
        print(f"[AmodalYOLO] Total params: {total_params:,} | Trainable: {trainable_params:,}")

    def _extract_p3(self, x):
        """Forward through backbone+neck and extract P3 features.

        Unlike the standard YOLO forward which only saves outputs in the save list,
        this method saves ALL intermediate outputs so we can access P3 regardless
        of whether it's in the save list.
        """
        y = []  # store ALL intermediate outputs
        for m in self.backbone_neck:
            if m.f != -1:  # not from previous layer
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            x = m(x)
            y.append(x)  # save ALL outputs
        return y[self.p3_layer_idx]

    def forward(self, x, target_size=None):
        """
        Args:
            x: [B, 4, H, W] RGBD input tensor
            target_size: (H, W) target size for output masks (for loss computation).
                         If None, outputs are at stride-4 resolution.

        Returns:
            pred_vis:  [B, 1, H_out, W_out] visible mask logits
            pred_occ:  [B, 1, H_out, W_out] occluded mask logits
            pred_full: [B, 1, H_out, W_out] full (amodal) mask logits
        """
        p3_feat = self._extract_p3(x)
        pred_vis, pred_occ, pred_full = self.tri_head(p3_feat)

        if target_size is not None:
            pred_vis = F.interpolate(pred_vis, size=target_size, mode='bilinear', align_corners=False)
            pred_occ = F.interpolate(pred_occ, size=target_size, mode='bilinear', align_corners=False)
            pred_full = F.interpolate(pred_full, size=target_size, mode='bilinear', align_corners=False)

        return pred_vis, pred_occ, pred_full


# ---- Quick test ----
if __name__ == '__main__':
    import sys
    yaml_path = sys.argv[1] if len(sys.argv) > 1 else 'ultralytics/cfg/models/11/yolo11-seg.yaml'
    model = AmodalYOLO(cfg=yaml_path, ch=4, nc=1, p3_layer_idx=16, head_mid_channels=128)
    x = torch.randn(2, 4, 640, 640)
    vis, occ, full = model(x, target_size=(640, 640))
    print(f"pred_vis:  {vis.shape}")
    print(f"pred_occ:  {occ.shape}")
    print(f"pred_full: {full.shape}")
