# 引入库
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D


class AttentionParallelFeatureMixer(nn.Module):
    """
    Attention Parallel Feature Mixer模块搭建

    ---

    Args:
        in_channel_FA: int
            输入特征图FA的通道数
        in_channel_FB: int
            输入特征图FB的通道数
        reduction: int = 2
            降维倍数

    ---

    Returns：
        out: torch.Tensor
            输出特征图
    """

    def __init__(
        self,
        in_channel_FA: int,
        in_channel_FB: int,
        reduction: int = 2,
    ) -> None:
        super(AttentionParallelFeatureMixer, self).__init__()
        self.mid_channel = in_channel_FA + in_channel_FB

        # 通道注意力部分
        self.ch_att = nn.Sequential(
            nn.Conv2d(
                in_channels=self.mid_channel * 2,
                out_channels=self.mid_channel // reduction,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.mid_channel // reduction),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=self.mid_channel // reduction,
                out_channels=in_channel_FA,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channel_FA),
        )

        # 空间注意力部分
        self.sp_att = nn.Sequential(
            nn.Conv2d(
                in_channels=self.mid_channel,
                out_channels=self.mid_channel // reduction,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.mid_channel // reduction),
            nn.SiLU(inplace=True),
            nn.Conv2d(
                in_channels=self.mid_channel // reduction,
                out_channels=in_channel_FB,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channel_FB),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, FA, FB):
        B, CA, H, W = FA.shape
        B, CB, H, W = FB.shape
        mix_feature = torch.cat([FA, FB], dim=1)

        # 通道注意力部分
        GAP = F.adaptive_avg_pool2d(mix_feature, (1, 1))
        GMP = F.adaptive_max_pool2d(mix_feature, (1, 1))
        ch_feature = torch.cat([GAP, GMP], dim=1)
        ch_output = self.ch_att(ch_feature)

        # 空间注意力部分
        sp_output = self.sp_att(mix_feature)

        w = self.sigmoid(ch_output + sp_output)
        output = w * FA + (1 - w) * FB
        return output


if __name__ == "__main__":
    B, C, H, W = 2, 2, 10, 10
    FA = torch.randn(B, C, H, W)
    FB = torch.randn(B, C, H, W)
    model = AttentionParallelFeatureMixer(C, C, reduction=2)
    model.eval()

    with torch.no_grad():
        out = model(FA, FB)

        print("融合后的特征图：\n", out)

        # 此时再调用 .numpy() 就绝对不会报错了
        fa_pts = FA[0].permute(1, 2, 0).reshape(-1, 3).numpy()
        fb_pts = FB[0].permute(1, 2, 0).reshape(-1, 3).numpy()
        out_pts = out[0].permute(1, 2, 0).reshape(-1, 3).numpy()

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(
            fa_pts[:, 0],
            fa_pts[:, 1],
            fa_pts[:, 2],
            color="blue",
            label="FA (Input A)",
            s=50,
            marker="o",
            alpha=0.7,
        )
        ax.scatter(
            fb_pts[:, 0],
            fb_pts[:, 1],
            fb_pts[:, 2],
            color="red",
            label="FB (Input B)",
            s=50,
            marker="s",
            alpha=0.7,
        )
        ax.scatter(
            out_pts[:, 0],
            out_pts[:, 1],
            out_pts[:, 2],
            color="green",
            label="Fused Output",
            s=70,
            marker="^",
            alpha=0.9,
        )

        # Draw trajectories connecting FA -> Out -> FB to show how features are mixed
        for i in range(len(fa_pts)):
            # Connect FA to FB with a faint gray line
            ax.plot(
                [fa_pts[i, 0], fb_pts[i, 0]],
                [fa_pts[i, 1], fb_pts[i, 1]],
                [fa_pts[i, 2], fb_pts[i, 2]],
                color="gray",
                linestyle="--",
                alpha=0.4,
            )

        ax.set_xlabel("Channel 0 (X)")
        ax.set_ylabel("Channel 1 (Y)")
        ax.set_zlabel("Channel 2 (Z)")
        ax.set_title(
            "3D Feature Space Visualization (Batch 0)\nHow Attention Mixer Fuses FA and FB"
        )
        ax.legend()

        plt.tight_layout()
        plt.show()

        print("\n特征点转换成功！形状为：", out_pts.shape)
