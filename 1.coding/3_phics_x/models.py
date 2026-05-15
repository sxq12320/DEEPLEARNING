import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import cba


class ResNet18(nn.Module):
    def __init__(self, in_ch: int):
        super(ResNet18, self).__init__()
        self.layers = nn.ModuleList(
            [
                cba(
                    in_channel=in_ch, out_channel=64, kernel_size=7, stride=2, padding=3
                ),
                nn.Sequential(
                    nn.Conv2d(
                        in_channels=in_ch,
                        out_channels=64,
                        kernel_size=7,
                        stride=2,
                        padding=3,
                        dilation=1,
                        groups=1,
                    ),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                ),
            ]
        )

    def forward(self, x):
        p = []
        out = x
        b, c, h, w = x.shape
        for layer in self.layers:
            out = layer(out)
            p.append(out)
        return out, p
