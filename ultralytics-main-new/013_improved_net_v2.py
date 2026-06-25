"""
ROI heatmap network for the second-stage pollination keypoint model.

This model follows direction A:
1. YOLO provides a flower segmentation mask.
2. The mask defines a flower ROI.
3. The second-stage network receives ROI RGB pixels plus the ROI mask.
4. The network predicts a keypoint heatmap instead of a center offset.
"""

import time

import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ROIHeatmapNet(nn.Module):
    """
    Lightweight U-Net style keypoint heatmap predictor.

    Input:
        roi: (B, 4, H, W), RGB channels normalized plus a binary mask channel.
    Output:
        heatmap logits: (B, 1, H/2, W/2) for the pollination keypoint.
    """

    def __init__(self, in_channels=4, base_channels=32):
        super().__init__()
        c = base_channels

        self.enc1 = ConvBlock(in_channels, c)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c, c * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * 2, c * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ConvBlock(c * 4, c * 8))

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 8, c * 4, kernel_size=1),
        )
        self.dec2 = ConvBlock(c * 8, c * 4)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(c * 4, c * 2, kernel_size=1),
        )
        self.dec1 = ConvBlock(c * 4, c * 2)

        self.head = nn.Sequential(
            nn.Conv2d(c * 2, c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),
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
        x1 = self.enc1(roi)      # H, W
        x2 = self.down1(x1)      # H/2, W/2
        x3 = self.down2(x2)      # H/4, W/4
        x4 = self.down3(x3)      # H/8, W/8

        y = self.up2(x4)         # H/4, W/4
        y = self.dec2(torch.cat([y, x3], dim=1))
        y = self.up1(y)          # H/2, W/2
        y = self.dec1(torch.cat([y, x2], dim=1))
        return self.head(y)


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
    model = ROIHeatmapNet(in_channels=4, base_channels=32).to(device)
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
