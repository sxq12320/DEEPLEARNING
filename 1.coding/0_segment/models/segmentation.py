import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import RESNET_18_BACKBONE_CFG
from .blocks import Basic_Conv_Block
from .builder import make_layers


class MiniSegNet(nn.Module):
    """最小化分割网络，ResNet-18 backbone + 1x1 卷积 head。"""

    def __init__(self, in_ch=3, out_ch=1, backbone_cfg=None):
        """初始化分割网络。

        Args:
            in_ch (int): 输入通道数。
            out_ch (int): 输出通道数。
            backbone_cfg (list | None): Backbone 配置列表。
        """
        super().__init__()
        cfg = backbone_cfg if backbone_cfg is not None else RESNET_18_BACKBONE_CFG
        self.backbone = make_layers(cfg)
        self._use_encoder_decoder = cfg == RESNET_18_BACKBONE_CFG
        if self._use_encoder_decoder:
            self._stage_slices = [(2, 4), (4, 6), (6, 8), (8, 10)]
            self.up4 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
            self.dec4 = self._make_decoder_block(512, 256)
            self.up3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
            self.dec3 = self._make_decoder_block(256, 128)
            self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
            self.dec2 = self._make_decoder_block(128, 64)
            self.head = nn.Conv2d(64, out_ch, kernel_size=1)
        else:
            # 找到 backbone 最后一个特征图的通道数
            self.head = nn.Conv2d(512, out_ch, kernel_size=1)

    def _make_decoder_block(self, in_ch, out_ch):
        return nn.Sequential(
            Basic_Conv_Block(in_ch, out_ch, 3, 1, 1, 1, 1, "relu"),
            Basic_Conv_Block(out_ch, out_ch, 3, 1, 1, 1, 1, "relu"),
        )

    def forward(self, x):
        """前向传播并上采样到输入尺寸。

        Args:
            x (torch.Tensor): 输入图像张量。

        Returns:
            torch.Tensor: 输出 logits，大小与输入一致。
        """
        input_size = x.shape[2:]
        if not self._use_encoder_decoder:
            feat = self.backbone(x)
            logits = self.head(feat)
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
            return logits

        x = self.backbone[0](x)
        x = self.backbone[1](x)
        features = []
        for start, end in self._stage_slices:
            for idx in range(start, end):
                x = self.backbone[idx](x)
            features.append(x)

        feat1, feat2, feat3, feat4 = features
        dec4 = self.up4(feat4)
        if dec4.shape[2:] != feat3.shape[2:]:
            dec4 = F.interpolate(dec4, size=feat3.shape[2:], mode='bilinear', align_corners=False)
        dec4 = self.dec4(torch.cat([dec4, feat3], dim=1))

        dec3 = self.up3(dec4)
        if dec3.shape[2:] != feat2.shape[2:]:
            dec3 = F.interpolate(dec3, size=feat2.shape[2:], mode='bilinear', align_corners=False)
        dec3 = self.dec3(torch.cat([dec3, feat2], dim=1))

        dec2 = self.up2(dec3)
        if dec2.shape[2:] != feat1.shape[2:]:
            dec2 = F.interpolate(dec2, size=feat1.shape[2:], mode='bilinear', align_corners=False)
        dec2 = self.dec2(torch.cat([dec2, feat1], dim=1))

        logits = self.head(dec2)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits
