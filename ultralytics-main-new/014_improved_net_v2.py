"""
Lightweight ROI heatmap network for the second-stage pollination keypoint model.

This version keeps the stage-1 YOLO segmentation unchanged and makes the second
stage cheaper with depthwise separable blocks plus strip-style context mixing.
"""

import time

import torch
import torch.nn as nn
import torch.nn.functional as F


def _to_2tuple(value):
    if isinstance(value, tuple):
        return value
    return (value, value)


class ConvBNAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1, act=True):
        super().__init__()
        kernel_size = _to_2tuple(kernel_size)
        padding = tuple(k // 2 for k in kernel_size)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class DepthwiseSeparableBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.depthwise = ConvBNAct(in_channels, in_channels, kernel_size=3, groups=in_channels)
        self.pointwise = ConvBNAct(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))


class ResidualDepthwiseBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(channels, channels, kernel_size=3, groups=channels),
            ConvBNAct(channels, channels, kernel_size=1, act=False),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(x + self.block(x))


class StripContextBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.pre = ConvBNAct(channels, channels, kernel_size=1)
        self.strip_h = ConvBNAct(channels, channels, kernel_size=(1, kernel_size), groups=channels)
        self.strip_v = ConvBNAct(channels, channels, kernel_size=(kernel_size, 1), groups=channels)
        self.mix = ConvBNAct(channels, channels, kernel_size=1, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        residual = x
        x = self.pre(x)
        x = self.strip_h(x)
        x = self.strip_v(x)
        x = self.mix(x)
        return self.act(residual + x)


class ROIHeatmapNet(nn.Module):
    """
    Lightweight ROI keypoint heatmap predictor.

    Input:
        roi: (B, 4, H, W), RGB channels normalized plus a binary mask channel.
    Output:
        heatmap logits: (B, 1, H/2, W/2) by default.
    """

    def __init__(self, in_channels=4, base_channels=16, output_size=None):
        super().__init__()
        c = base_channels
        self.output_size = output_size

        self.stem = nn.Sequential(
            ConvBNAct(in_channels, c, kernel_size=3),
            ResidualDepthwiseBlock(c),
        )
        self.enc1 = nn.Sequential(
            DepthwiseSeparableBlock(c, c * 2),
            ResidualDepthwiseBlock(c * 2),
        )
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            DepthwiseSeparableBlock(c * 2, c * 4),
            ResidualDepthwiseBlock(c * 4),
        )
        self.down2 = nn.MaxPool2d(2)
        self.context = nn.Sequential(
            StripContextBlock(c * 4, kernel_size=7),
            StripContextBlock(c * 4, kernel_size=7),
        )
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            ConvBNAct(c * 4, c * 2, kernel_size=1),
        )
        self.dec1 = nn.Sequential(
            DepthwiseSeparableBlock(c * 4, c * 2),
            ResidualDepthwiseBlock(c * 2),
        )
        self.head = nn.Sequential(
            ConvBNAct(c * 2, c, kernel_size=3),
            nn.Conv2d(c, 1, kernel_size=1),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, roi):
        x0 = self.stem(roi)         # H, W
        x1 = self.enc1(x0)          # H, W
        x1d = self.down1(x1)        # H/2, W/2
        x2 = self.enc2(x1d)         # H/2, W/2
        x2d = self.down2(x2)        # H/4, W/4
        x3 = self.context(x2d)      # H/4, W/4

        y = self.up1(x3)            # H/2, W/2
        y = self.dec1(torch.cat([y, x1d], dim=1))
        logits = self.head(y)

        if self.output_size is not None and logits.shape[-2:] != _to_2tuple(self.output_size):
            logits = F.interpolate(logits, size=self.output_size, mode="bilinear", align_corners=False)
        return logits


# Backward-compatible name for older imports in this project.
ImprovedContourNetV2 = ROIHeatmapNet


def count_params(model):
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def benchmark_speed(model, device, roi_size=128, n_runs=100):
    model.eval()
    dummy = torch.randn(1, 4, roi_size, roi_size, device=device)

    for _ in range(10):
        with torch.no_grad():
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()
    for _ in range(n_runs):
        with torch.no_grad():
            model(dummy)

    if device.type == "cuda":
        torch.cuda.synchronize()

    return (time.time() - start) / n_runs * 1000.0


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ROIHeatmapNet(in_channels=4, base_channels=16).to(device)
    dummy = torch.randn(2, 4, 128, 128, device=device)
    out = model(dummy)

    print("=" * 60)
    print("ROIHeatmapNet")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Params: {count_params(model):,}")
    print(f"Input: {tuple(dummy.shape)}")
    print(f"Output: {tuple(out.shape)}")
    print(f"Speed: {benchmark_speed(model, device):.2f} ms/sample")
