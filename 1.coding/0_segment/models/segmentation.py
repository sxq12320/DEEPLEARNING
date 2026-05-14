"""基础与复合分割网络模块。

包含诸如简单的 MiniSegNet、FPNSegNet 与 TS-Dual 结构。
"""

from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import RESNET_18_BACKBONE_CFG

from .backbones import MultiScaleResNet18
from .builder import build_backbone, build_head, build_neck, make_layers
from .modules import FPN


class MiniSegNet(nn.Module):
    """最小化分割网络，ResNet-18 backbone + 1x1 卷积 head。

    包含最基本的 BackBone 与 极简预测头的分割网络。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出通道数。
        backbone_cfg (list | None): Backbone 配置列表。
    """

    def __init__(self, in_ch=3, out_ch=1, backbone_cfg=None):
        """初始化最简分割网络类。"""
        super().__init__()
        cfg = backbone_cfg if backbone_cfg is not None else RESNET_18_BACKBONE_CFG
        self.backbone = make_layers(cfg)
        # 找到 backbone 最后一个特征图的通道数
        self.head = nn.Conv2d(512, out_ch, kernel_size=1)

    def forward(self, x):
        """前向传播并上采样到输入尺寸。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            torch.Tensor: 输出 logits，大小与输入一致。
        """
        feat = self.backbone(x)
        logits = self.head(feat)
        logits = F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return logits


class FPNSegNet(nn.Module):
    """FPN 多尺度分割网络。

    结构：
        MultiScaleResNet18 backbone → FPN neck → 多尺度特征融合 → 分割 head

    Backbone 输出 4 个尺度 [c2, c3, c4, c5]，
    FPN 自顶向下融合为 [p2, p3, p4, p5]，
    全部上采样至 p2 分辨率拼接后经 head 输出分割图。

    Attributes:
        in_ch (int): 输入通道数。
        out_ch (int): 输出类别数。
        backbone_channels (List[int] | None): backbone 各阶段通道数，默认 [64, 128, 256, 512]。
        fpn_channels (int): FPN 统一通道数。
    """

    def __init__(self, in_ch=3, out_ch=1, backbone_channels=None, fpn_channels=256):
        """初始化多尺度分割网络类。"""
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [64, 128, 256, 512]

        self.backbone = MultiScaleResNet18()
        self.fpn = FPN(in_channels_list=backbone_channels, out_channels=fpn_channels)

        fuse_in_ch = fpn_channels * len(backbone_channels)
        self.head = nn.Sequential(
            nn.Conv2d(fuse_in_ch, fpn_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(fpn_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_channels, out_ch, kernel_size=1),
        )

    def forward(self, x):
        """前向传播。

        Args:
            x (torch.Tensor): 输入图像 (B, 3, H, W)。

        Returns:
            torch.Tensor: 分割 logits，大小与输入一致。
        """
        features = self.backbone(x)  # [c2, c3, c4, c5]
        fpn_feats = self.fpn(features)  # [p2, p3, p4, p5]

        # 全部上采样到 p2 分辨率并拼接
        h, w = fpn_feats[0].shape[2:]
        fused = torch.cat(
            [
                F.interpolate(f, size=(h, w), mode="bilinear", align_corners=False)
                for f in fpn_feats
            ],
            dim=1,
        )

        logits = self.head(fused)
        logits = F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return logits


class TSDualSegDetNet(nn.Module):
    """TS-Dual 分割 + 检测联合模型。

    结构：backbone → neck → head
    输出包含 bbox 预测与 mask logits。

    Args:
        model_cfg (dict): 模型配置，包含 backbone/neck/head 三部分。
    """

    def __init__(self, model_cfg: dict):
        """初始化 TS-Dual 网络。

        Args:
            model_cfg (dict): 模型配置字典。
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.backbone = build_backbone(model_cfg["backbone"])
        self.neck = build_neck(model_cfg["neck"])
        self.head = build_head(model_cfg["head"])

    def forward(
        self,
        rgb: Union[torch.Tensor, dict],
        prompt: Optional[torch.Tensor] = None,
        depth: Optional[torch.Tensor] = None,
    ) -> dict:
        """前向传播。

        Args:
            rgb (torch.Tensor | dict): RGB 输入或输入字典。
            prompt (torch.Tensor | None): Mask 先验提示。
            depth (torch.Tensor | None): Depth 输入。

        Returns:
            dict: {"bbox": bbox_pred, "mask": mask_logits}。
        """
        rgb, prompt, depth = self._parse_inputs(rgb, prompt, depth)
        features = self.backbone(rgb, prompt, depth)
        neck_out = self.neck(features)
        bbox_pred, mask_logits = self.head(neck_out, input_shape=rgb.shape[2:])
        return {"bbox": bbox_pred, "mask": mask_logits}

    def _parse_inputs(
        self,
        rgb: Union[torch.Tensor, dict],
        prompt: Optional[torch.Tensor],
        depth: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
        """解析输入，兼容 dict 形式。

        Args:
            rgb (torch.Tensor | dict): RGB 输入或包含多模态的字典。
            prompt (torch.Tensor | None): Mask 先验提示。
            depth (torch.Tensor | None): Depth 输入。

        Returns:
            Tuple: (rgb, prompt, depth)。
        """
        if isinstance(rgb, dict):
            depth = rgb.get("depth")
            prompt = rgb.get("prompt")
            rgb = rgb.get("rgb")
        if rgb is None:
            raise ValueError("RGB 输入不能为空")
        return rgb, prompt, depth


def build_ts_dual_model(
    model_cfg: dict, num_classes: Optional[int] = None
) -> TSDualSegDetNet:
    """根据配置构建 TS-Dual 模型。

    Args:
        model_cfg (dict): 模型配置字典。
        num_classes (int | None): 分割类别数（用于覆盖 head 输出通道）。

    Returns:
        TSDualSegDetNet: 构建后的模型。
    """
    cfg = deepcopy(model_cfg)
    if num_classes is not None:
        head_args = cfg.setdefault("head", {}).setdefault("args", {})
        head_args["mask_out_ch"] = num_classes
    return TSDualSegDetNet(cfg)
