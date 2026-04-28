# 0_segment

面向图像分割任务的深度学习工程骨架。

## 项目结构

```text
0_segment/
├── configs/               # 全局配置
│   └── config.py          # ACTIVATION_MAP, RESNET_18_CFG, RESNET_18_BACKBONE_CFG
├── data/                  # 数据流水线
│   ├── dataset.py         # SegmentationDataset（真实/合成数据自动切换）
│   └── transforms.py      # 图像变换 + TXT/JSON/NPY → mask 转换
├── models/                # 模型模块
│   ├── registry.py        # BLOCK_REGISTRY + register_block 装饰器
│   ├── blocks.py          # 15 种基础网络层（Conv, ResBlock, CBAM 等）
│   ├── builder.py         # make_layers: cfg list → nn.Sequential
│   ├── backbones.py       # ResNet18 骨干网络
│   └── segmentation.py    # MiniSegNet 分割架构
├── engine/                # 训练与评估引擎
│   ├── losses.py          # SegmentationLoss
│   ├── metrics.py         # compute_iou, compute_dice, calculate_map
│   ├── trainer.py         # Trainer（训练循环 + 超参日志 + loss 曲线）
│   └── evaluator.py       # Evaluator（评估循环 + IoU 报告）
├── utils/                 # 工具
│   ├── common.py          # get_activation, autopad
│   └── visualize.py       # plot_loss_curve, plot_sample_prediction
├── scripts/               # CLI 入口
│   ├── train.py           # 训练
│   └── predict.py         # 预测
├── checkpoints/           # 模型权重
├── logs/                  # 日志
├── runs/                  # 输出
└── requirements.txt
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 最小验证（合成数据，2 epoch）
python scripts/train.py --epochs 2 --batch 4

# 真实数据训练
python scripts/train.py --image-dir data/images --mask-dir data/masks --epochs 50

# 预测
python scripts/predict.py --source image.jpg --weights runs/train/exp/weights/best.pt
```

## 导入规范

```python
from models import MiniSegNet, ResNet18, make_layers, BLOCK_REGISTRY, register_block
from data import SegmentationDataset, get_dataset_rgb
from engine import compute_iou, compute_dice, Trainer, Evaluator
from configs import ACTIVATION_MAP, RESNET_18_BACKBONE_CFG
from utils import get_activation, autopad
```

## 注册机制

通过配置列表动态构建网络：

```python
from models import register_block, BLOCK_REGISTRY, make_layers

@register_block("my_block")
class MyBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
    def forward(self, x):
        return self.conv(x)

cfg = [
    ["basic_conv_block", 3, 64, 3, 1, 1, 1, 1, "relu"],
    ["my_block", 64, 128],
]
net = make_layers(cfg)
```

## 支持的数据标签格式

- **mask**: 直接读取掩码图片
- **txt**: YOLO 格式（bbox / polygon）
- **json**: COCO 格式
- **npy**: NumPy 数组格式

## 可用模块列表

| 注册名 | 类 | 说明 |
|--------|-----|------|
| `conv` | Conv | 普通卷积 |
| `basic_conv_block` | Basic_Conv_Block | Conv + BN + Activation |
| `conv_block_nonb` | Conv_Block_NONB | Conv + Activation (无 BN) |
| `depthwise_conv` | DepthWise_Conv | 深度卷积 |
| `pointwise_conv` | PointWise_Conv | 逐点卷积 |
| `depthwise_separable_conv` | DepthWiseSeparable_Conv | 深度可分离卷积 |
| `resnet_block_34` | ResNetBlock_34 | ResNet-18/34 基本块 |
| `resnet_block_50` | ResNetBlock_50 | ResNet-50+ 瓶颈块 |
| `cbam` | CBAM | 完整 CBAM 注意力 |
| `cbam_channel_attention` | CBAM_Channel_Attention | CBAM 通道注意力 |
| `cbam_spatial_attention` | CBAM_Spatial_Attention | CBAM 空间注意力 |
| `maxpool` | MaxPool | 最大池化 |
| `adaptive_avg_pool` | AdaptiveAvgPool | 自适应平均池化 |
| `flatten` | Flatten | 展平 |
| `linear` | Linear | 全连接层 |
