"""YOLO11 完整检测器模块。

组合 backbone、neck、head 构建端到端的目标检测网络。
通过更换 channels / depth_scale 参数可灵活缩放为 nano / small / medium 规格。
"""

import torch
import torch.nn as nn

from .backbones import YOLO11Backbone
from .heads import YOLO11Head
from .necks import YOLO11Neck


class YOLO11Detector(nn.Module):
    """YOLO11 完整目标检测器。

    结构：
        YOLO11Backbone → YOLO11Neck → YOLO11Head
              ↓                ↓              ↓
        [P3,P4,P5]    [N3,N4,N5]    [cls_list, reg_list]

    Attributes:
        backbone (YOLO11Backbone): 特征提取骨干网络。
        neck (YOLO11Neck): PAN-FPN 颈部融合网络。
        head (YOLO11Head): 解耦检测头。
        num_classes (int): 目标类别数。
        reg_max (int): DFL 回归区间数。
        backbone_channels (List[int]): backbone 各阶段通道配置。
        neck_channels (List[int]): neck 输入通道配置。
        depth_scale (float): 深度缩放因子。
    """

    def __init__(
        self, num_classes=80, reg_max=16, backbone_channels=None, depth_scale=1.0
    ):
        """初始化 YOLO11 检测器。

        Args:
            num_classes (int): 检测类别数（不含背景）。
            reg_max (int): Distribution Focal Loss 的区间数量。
            backbone_channels (List[int] | None): backbone 宽度配置 [c1,c2,c3,c4,c5]，
                                                  默认 nano。
            depth_scale (float): 深度缩放因子（nano≈0.33, small≈0.67, medium=1.0）。
        """
        super().__init__()
        if backbone_channels is None:
            backbone_channels = [16, 32, 64, 128, 256]

        self.num_classes = num_classes
        self.reg_max = reg_max
        self.backbone_channels = backbone_channels
        self.depth_scale = depth_scale

        # neck 输入通道为 backbone 后三层输出的通道
        neck_channels = backbone_channels[2:]  # [c3, c4, c5]

        self.backbone = YOLO11Backbone(
            channels=backbone_channels, depth_scale=depth_scale
        )
        self.neck = YOLO11Neck(channels=neck_channels, depth_scale=depth_scale)
        self.head = YOLO11Head(
            num_classes=num_classes, reg_max=reg_max, channels=neck_channels
        )

    def forward(self, x):
        """前向传播完成完整的检测流程。

        Args:
            x (torch.Tensor): 输入图像批 (B, 3, H, W)。

        Returns:
            Tuple:
                cls_list (List[torch.Tensor]): 每层分类 logits，
                                               形状 (B, num_classes, Hi, Wi)。
                reg_list (List[torch.Tensor]): 每层回归输出，
                                               形状 (B, 4*reg_max, Hi, Wi)。
                features (List[torch.Tensor]): backbone 多尺度特征 [P3, P4, P5]，
                                               供损失函数使用。
                neck_feats (List[torch.Tensor]): neck 融合特征 [N3, N4, N5]。
        """
        features = self.backbone(x)  # [P3, P4, P5]
        neck_feats = self.neck(features)  # [N3, N4, N5]
        cls_list, reg_list = self.head(neck_feats)
        return cls_list, reg_list, features, neck_feats

    def predict(self, x):
        """仅返回检测头的原始输出，方便推理时解码。

        Args:
            x (torch.Tensor): 输入图像批 (B, 3, H, W)。

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                cls_list, reg_list。
        """
        features = self.backbone(x)
        neck_feats = self.neck(features)
        return self.head(neck_feats)
