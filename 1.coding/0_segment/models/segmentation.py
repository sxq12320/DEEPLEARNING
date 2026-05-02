import torch
import torch.nn as nn
import torch.nn.functional as F
from configs.config import RESNET_18_BACKBONE_CFG
from .blocks import Basic_Conv_Block
from .builder import make_layers


class MiniSegNet(nn.Module):
    """最小化分割网络，ResNet-18 backbone + 1x1 卷积 head。"""

    _STEM_BLOCK_TYPES = ("basic_conv_block", "maxpool")
    _RESNET18_STAGE_BLOCK = "resnet_block_34"
    _RESNET18_STAGE_COUNTS = (2, 2, 2, 2)
    _RESNET18_STAGE_CHANNELS = (64, 128, 256, 512)
    _RESNET18_CFG_LENGTH = len(_STEM_BLOCK_TYPES) + sum(_RESNET18_STAGE_COUNTS)

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
        self._stem_slice = (0, len(self._STEM_BLOCK_TYPES))
        self._stage_slices = self._infer_stage_slices(cfg)
        self._use_encoder_decoder = self._stage_slices is not None
        if self._use_encoder_decoder:
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

    def _infer_stage_slices(self, cfg):
        if len(cfg) < self._RESNET18_CFG_LENGTH:
            return None
        if tuple(item[0] for item in cfg[:self._stem_slice[1]]) != self._STEM_BLOCK_TYPES:
            return None
        if any(
            item[0] != self._RESNET18_STAGE_BLOCK
            for item in cfg[self._stem_slice[1]:self._RESNET18_CFG_LENGTH]
        ):
            return None
        slices = []
        start = self._stem_slice[1]
        for count, channels in zip(self._RESNET18_STAGE_COUNTS, self._RESNET18_STAGE_CHANNELS):
            end = start + count
            if cfg[end - 1][2] != channels:
                return None
            slices.append((start, end))
            start = end
        if start != self._RESNET18_CFG_LENGTH:
            return None
        return slices

    def _align_like(self, source, target):
        if source.shape[2:] != target.shape[2:]:
            return F.interpolate(source, size=target.shape[2:], mode='bilinear', align_corners=False)
        return source

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

        for idx in range(*self._stem_slice):
            x = self.backbone[idx](x)
        features = []
        for start, end in self._stage_slices:
            for idx in range(start, end):
                x = self.backbone[idx](x)
            features.append(x)

        feat1, feat2, feat3, feat4 = features
        dec4 = self._align_like(self.up4(feat4), feat3)
        dec4 = self.dec4(torch.cat([dec4, feat3], dim=1))

        dec3 = self._align_like(self.up3(dec4), feat2)
        dec3 = self.dec3(torch.cat([dec3, feat2], dim=1))

        dec2 = self._align_like(self.up2(dec3), feat1)
        dec2 = self.dec2(torch.cat([dec2, feat1], dim=1))

        logits = self.head(dec2)
        logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)
        return logits
