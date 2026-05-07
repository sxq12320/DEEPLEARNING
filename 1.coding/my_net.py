from uu import encode

import torch
import torch.nn as nn


class encoder(nn.Module):
    """
    Encoder 编码器
    """

    def __init__(self, in_ch: int = 3, alpha: int = 4):
        super(encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=in_ch,
                out_channels=in_ch * alpha,
                kernel_size=3,
                stride=2,
                padding=1,
            )
            # nn.Conv2d(
            #     in_channels=in_ch * alpha,
            #     out_channels=in_ch * alpha * 2,
            #     kernel_size=5,
            #     stride=2,
            #     padding=1,
            # ),
        )

    def forward(self, x):
        return self.encoder(x)


x = torch.randn(1, 3, 16, 16)
model = encoder(in_ch=x.size(1), alpha=4)
y = model(x)
print(y.size())
