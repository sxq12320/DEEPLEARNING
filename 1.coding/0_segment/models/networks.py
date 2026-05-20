"""分割与检测组合网络。

包含 MiniSegNet、FPNSegNet、TS-Dual、YOLO11 等完整网络结构。
"""

from copy import deepcopy
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.config import RESNET_18_BACKBONE_CFG

from .backbones import MultiScaleResNet18, YOLO11Backbone
from utils.builder import build_backbone, build_head, build_neck, make_layers
from .heads import YOLO11Head
from .modules import FPN
from .necks import YOLO11Neck

# ======================================================================
# 分割网络
# ======================================================================

class MiniSegNet(nn.Module):
    """最小化分割网络，ResNet-18 backbone + 1x1 卷积 head。"""

    def __init__(self, in_ch=3, out_ch=1, backbone_cfg=None):
        super().__init__()
        cfg = backbone_cfg if backbone_cfg is not None else RESNET_18_BACKBONE_CFG
        self.backbone = make_layers(cfg)
        self.head = nn.Conv2d(512, out_ch, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        logits = F.interpolate(
            logits, size=x.shape[2:], mode="bilinear", align_corners=False
        )
        return logits


class FPNSegNet(nn.Module):
    """FPN 多尺度分割网络。

    MultiScaleResNet18 backbone → FPN neck → 多尺度特征融合 → 分割 head。
    """

    def __init__(self, in_ch=3, out_ch=1, backbone_channels=None, fpn_channels=256):
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
        features = self.backbone(x)
        fpn_feats = self.fpn(features)

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

    结构：backbone → neck → head，输出 bbox 预测与 mask logits。
    """

    def __init__(self, model_cfg: dict):
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
    """根据配置构建 TS-Dual 模型。"""
    cfg = deepcopy(model_cfg)
    if num_classes is not None:
        head_args = cfg.setdefault("head", {}).setdefault("args", {})
        head_args["mask_out_ch"] = num_classes
    return TSDualSegDetNet(cfg)


# ======================================================================
# 检测网络
# ======================================================================

class YOLO11Detector(nn.Module):
    """YOLO11 完整目标检测器。

    YOLO11Backbone → YOLO11Neck → YOLO11Head
    """

    def __init__(
        self, num_classes=80, reg_max=16, backbone_channels=None, depth_scale=1.0
    ):
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [16, 32, 64, 128, 256]

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.backbone_channels = backbone_channels
        self.depth_scale = depth_scale

        neck_channels = backbone_channels[2:]

        self.backbone = YOLO11Backbone(
            channels=backbone_channels, depth_scale=depth_scale
        )
        self.neck = YOLO11Neck(channels=neck_channels, depth_scale=depth_scale)
        self.head = YOLO11Head(
            num_classes=num_classes, reg_max=reg_max, channels=neck_channels
        )

    def forward(self, x):
        features = self.backbone(x)
        neck_feats = self.neck(features)
        cls_list, reg_list = self.head(neck_feats)
        return cls_list, reg_list, features, neck_feats

    def predict(self, x):
        features = self.backbone(x)
        neck_feats = self.neck(features)
        return self.head(neck_feats)
