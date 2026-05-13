# 0_segment — 图像分割工程骨架

面向图像分割任务的深度学习工程，基于 PyTorch，支持**模块化设计 + 注册机制 + 配置驱动组网**。

## 项目结构

```
0_segment/
├── train.py                 # 一键训练入口（CLI + JSON/YAML 配置合并）
├── requirements.txt         # 依赖
│
├── configs/                 # 全局配置
│   ├── config.py            # 激活函数映射表 + 网络结构配置列表
│   └── __init__.py
│
├── datasets/                # 数据流水线
│   ├── dataset.py           # SegmentationDataset（真实/合成数据自动切换）
│   ├── transforms.py        # TXT / JSON / NPY → mask 标签转换
│   └── __init__.py
│
├── models/                  # 模型库
│   ├── registry.py          # @register_block 注册机制
│   ├── blocks.py            # 基础模块（Conv, ResBlock, CBAM, Pool ...）
│   ├── builder.py           # make_layers(cfg) 动态组网
│   ├── backbones.py         # 骨干网络（ResNet-18 ...）
│   ├── segmentation.py      # 分割架构（MiniSegNet）
│   └── __init__.py
│
├── engine/                  # 训练/评估引擎
│   ├── losses.py            # 损失函数（BCE / CE）
│   ├── metrics.py           # 评估指标（IoU, Dice, mAP）
│   ├── trainer.py           # Trainer 训练循环
│   ├── evaluator.py         # Evaluator 评估循环
│   └── __init__.py
│
├── utils/                   # 通用工具
│   ├── common.py            # get_activation, autopad
│   ├── visualize.py         # loss 曲线 / 预测对比可视化
│   └── __init__.py
│
└── checkpoints/             # 运行输出（权重、日志、图表）
    └── results/
        └── <name>/
            ├── weights/     # best.pt / last.pt
            ├── logs.txt     # 超参数 + 逐 epoch 日志
            └── loss_curve.png
```

## 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 使用默认配置启动（自动回退到合成数据）
python train.py

# 覆盖超参数
python train.py --imgsz 256 --epochs 50 --batch 16 --lr 5e-4

# 使用 JSON 配置文件
python train.py --cfg configs/train.json

# 使用 YAML 配置文件
python train.py --cfg configs/train.yaml

# 打印合并后配置
python train.py --print-cfg

# 保存合并配置
python train.py --cfg configs/train.json --save-cfg configs/merged.json
```

## 训练参数

所有参数均可通过 **CLI** 或 **JSON/YAML 配置文件** 指定，参数优先级：`CLI > 配置文件 > 默认值`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cfg` | str | — | JSON/YAML 配置文件路径 |
| `--image-dir` | str | "" | 训练图像目录 |
| `--mask-dir` | str | "" | 训练掩码目录 |
| `--label-dir` | str | "" | 标签目录（别名） |
| `--label-type` | str | mask | 标签格式：mask / txt / json / npy |
| `--imgsz` | int | 128 | 输入图像尺寸 |
| `--epochs` | int | 20 | 训练轮数 |
| `--batch` | int | 8 | 批次大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--workers` | int | 0 | 数据加载线程数 |
| `--synthetic-length` | int | 32 | 合成数据集长度 |
| `--augment / --no-augment` | bool | True | 数据增强开关 |
| `--cpu` | flag | False | 强制 CPU 模式 |
| `--project` | str | checkpoints/results | 输出根目录 |
| `--name` | str | train | 实验名称 |
| `--seed` | int | 22 | 随机种子 |

## JSON 配置文件示例

```jsonc
// configs/train.json
{
    "image_dir": "data/images",
    "mask_dir": "data/masks",
    "label_type": "mask",
    "imgsz": 256,
    "epochs": 50,
    "batch": 16,
    "lr": 0.001,
    "project": "checkpoints/results",
    "name": "exp_01",
    "seed": 42
}
```

## 日志输出

训练日志采用 YOLO 风格格式：

```
======================================================================
                      Segmentation Training
======================================================================
image_size           256
batch_size           16
epochs               50
learning_rate        0.001
optimizer            Adam
device               NVIDIA GeForce RTX 3060
data_source          data/images
augment              True
seed                 42
project              checkpoints/results/exp_01
model_params         2,134,593 trainable / 2,134,593 total
======================================================================

     Epoch       loss                  elapsed    ETA
────────────────────────────────────────────────────────────
        1/50     0.234567              00:05      04:10
        2/50     0.123456              00:10      04:00
        ...
```

日志文件 `logs.txt` 会同步写入到输出目录。

---

## 开发指南：如何添加新模块

### 1. 添加新的基础网络层（blocks）

在 [models/blocks.py](models/blocks.py) 中定义新模块，使用 `@register_block` 装饰器注册：

```python
from .registry import register_block

@register_block('my_attention')
class MyAttention(nn.Module):
    """自定义注意力模块。"""
    def __init__(self, in_ch: int, reduction: int):
        super().__init__()
        # 定义网络层
        ...

    def forward(self, x):
        # 前向逻辑
        ...
```

注册后即可在配置列表中直接引用：

```python
MY_CFG = [
    ["basic_conv_block", 3, 64, 7, 2, 3, 1, 1, "relu"],
    ["my_attention", 64, 4],
    ...
]
```

### 2. 添加新的骨干网络（backbone）

在 [models/backbones.py](models/backbones.py) 中添加 Backbone 类：

```python
class MyBackbone(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.backbone = make_layers(cfg or MY_CFG)

    def forward(self, x):
        return self.backbone(x)
```

### 3. 添加新的分割架构（segmentation head）

在 [models/segmentation.py](models/segmentation.py) 中定义新的分割网络：

```python
class MySegNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=1, backbone_cfg=None):
        super().__init__()
        self.backbone = make_layers(backbone_cfg or DEFAULT_CFG)
        self.head = nn.Conv2d(512, out_ch, kernel_size=1)

    def forward(self, x):
        feat = self.backbone(x)
        logits = self.head(feat)
        return F.interpolate(logits, size=x.shape[2:], mode='bilinear', align_corners=False)
```

### 4. 添加新的损失函数

在 [engine/losses.py](engine/losses.py) 的 `SegmentationLoss` 中扩展 `loss_type`：

```python
# 在 __init__ 中添加新的分支
elif loss_type == "dice":
    self.criterion = DiceLoss(**kwargs)
```

### 5. 添加新的数据增强

在 [datasets/transforms.py](datasets/transforms.py) 中添加增强函数，然后在 [datasets/dataset.py](datasets/dataset.py) 的 `_apply_augmentation` 中调用。

### 6. 切换模型架构

在 [train.py](train.py) 的 `train()` 函数中将 `MiniSegNet` 替换为你的模型：

```python
from models import MySegNet
model = MySegNet().to(device)
```

### 注册机制概述

```
定义模块 → @register_block('name') → 写入 BLOCK_REGISTRY
                                          ↓
配置列表 → make_layers(cfg) → 查表获取构造器 → nn.Sequential
```

## 已注册模块一览

| 注册名 | 类名 | 说明 |
|--------|------|------|
| `conv` | Conv | 基础卷积 |
| `basic_conv_block` | Basic_Conv_Block | Conv + BN + Activation |
| `conv_block_nonb` | Conv_Block_NONB | Conv + Activation（无 BN） |
| `depthwise_conv` | DepthWise_Conv | 逐通道卷积 |
| `pointwise_conv` | PointWise_Conv | 1×1 逐点卷积 |
| `depthwise_separable_conv` | DepthWiseSeparable_Conv | 可分离卷积 |
| `resnet_block_34` | ResNetBlock_34 | ResNet-34 残差块 |
| `resnet_block_50` | ResNetBlock_50 | ResNet-50 瓶颈残差块 |
| `cbam_channel_attention` | CBAM_Channel_Attention | CBAM 通道注意力 |
| `cbam_spatial_attention` | CBAM_Spatial_Attention | CBAM 空间注意力 |
| `cbam` | CBAM | CBAM 组合注意力 |
| `maxpool` | MaxPool | 最大池化 |
| `adaptive_max_pool` | AdaptiveMaxPool | 自适应最大池化 |
| `adaptive_avg_pool` | AdaptiveAvgPool | 自适应平均池化 |
| `flatten` | Flatten | 展平层 |
| `linear` | Linear | 全连接层 |

## 支持的数据格式

| 格式 | label_type | 说明 |
|------|------------|------|
| mask 图片 | `mask` | 直接读取 PNG/JPG 等掩码图像 |
| YOLO TXT | `txt` | 矩形框或多边形标注 |
| COCO JSON | `json` | COCO 格式标注文件 |
| NumPy | `npy` | .npy 格式掩码数组 |

## 依赖

```
torch==2.2.2
torchvision==0.17.2
numpy>=1.24,<2.0
opencv-python>=4.8,<5.0
Pillow>=9.5,<11.0
matplotlib>=3.7,<3.9
PyYAML>=6.0       # 可选，使用 YAML 配置时安装
```
