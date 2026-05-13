"""目标检测解耦头模块。

实现 YOLO11 的解耦检测头（Decoupled Head），将分类和回归分支
分离为两条独立卷积路径，避免分类与定位任务间的特征冲突。
"""

import torch
import torch.nn as nn


class YOLO11Head(nn.Module):
    """YOLO11 解耦检测头。

    对每一层特征图，分别通过 cls_branch 和 reg_branch 两条分支
    预测类别 logits 和边界框分布（Distribution Focal Loss 形式）。

    每个检测层输出：
        cls_out  — (B, num_classes, H, W)
        reg_out  — (B, 4 * reg_max, H, W)，其中 reg_max 为 DFL 区间数

    Attributes:
        num_classes (int): 检测类别数。
        reg_max (int): DFL 回归区间数量（默认 16）。
        channels (List[int]): 各检测层输入通道数。
    """

    def __init__(self, num_classes=80, reg_max=16, channels=None):
        """初始化解耦检测头。

        Args:
            num_classes (int): 类别数量（不含背景，如 COCO=80）。
            reg_max (int): DFL 回归区间数。
            channels (List[int] | None): 各层输入通道列表，默认 nano 规格。
        """
        super().__init__()
        if channels is None:
            channels = [64, 128, 256]
        self.num_classes = num_classes
        self.reg_max = reg_max

        # 每个检测尺度的两条分支
        self.cls_branch = nn.ModuleList()
        self.reg_branch = nn.ModuleList()

        for ch in channels:
            # 分类分支：两层 3×3 卷积 + 1×1 分类输出
            cls_layers = []
            for _ in range(2):
                cls_layers.extend(
                    [
                        nn.Conv2d(
                            ch, ch, kernel_size=3, stride=1, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(ch),
                        nn.SiLU(inplace=True),
                    ]
                )
            cls_layers.append(nn.Conv2d(ch, num_classes, kernel_size=1))
            self.cls_branch.append(nn.Sequential(*cls_layers))

            # 回归分支：两层 3×3 卷积 + 1×1 回归输出（4 * reg_max 通道）
            reg_layers = []
            for _ in range(2):
                reg_layers.extend(
                    [
                        nn.Conv2d(
                            ch, ch, kernel_size=3, stride=1, padding=1, bias=False
                        ),
                        nn.BatchNorm2d(ch),
                        nn.SiLU(inplace=True),
                    ]
                )
            reg_layers.append(nn.Conv2d(ch, 4 * reg_max, kernel_size=1))
            self.reg_branch.append(nn.Sequential(*reg_layers))

    def forward(self, features):
        """前向传播，对每层多尺度特征并行计算分类与回归输出。

        Args:
            features (List[torch.Tensor]): Neck 输出的多尺度特征列表
                                           [feat_s, feat_m, feat_l]，
                                           形状均为 (B, ch_i, H_i, W_i)。

        Returns:
            Tuple[List[torch.Tensor], List[torch.Tensor]]:
                cls_outputs — 每层的分类 logits，形状 (B, num_classes, H_i, W_i)；
                reg_outputs — 每层的回归输出，形状 (B, 4*reg_max, H_i, W_i)。
        """
        cls_outputs = []
        reg_outputs = []
        for feat, cls_conv, reg_conv in zip(features, self.cls_branch, self.reg_branch):
            cls_outputs.append(cls_conv(feat))
            reg_outputs.append(reg_conv(feat))
        return cls_outputs, reg_outputs
