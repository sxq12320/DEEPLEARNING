# 0_segment — 图像分割与目标检测工程骨架

基于 PyTorch 的深度学习工程，支持**图像分割**和**目标检测**两大任务。
采用**模块化设计 + 注册机制 + 配置驱动组网**架构，
可灵活更换 backbone / neck / head。

内置 **FPN 多尺度特征金字塔**（分割）和 **YOLO11 PAN-FPN**（检测）两套融合方案。

## 项目结构

```
0_segment/
├── train.py                 # 一键训练入口（CLI + JSON/YAML 配置合并）
├── requirements.txt         # 依赖
│
├── configs/                 # 全局配置
│   └── config.py            # 激活函数映射 + ResNet/YOLO 网络结构配置
│
├── datasets/                # 数据流水线
│   ├── dataset.py           # SegmentationDataset（真实/合成数据自动切换）
│   └── transforms.py        # TXT / JSON / NPY → mask 标签转换
│
├── models/                  # 模型库
│   ├── registry.py          # @register_block 注册机制
│   ├── blocks.py            # 基础模块（Conv, ResBlock, CBAM, C3K2, SPPF, FPN ...）
│   ├── builder.py           # make_layers(cfg) 动态组网
│   ├── backbones.py         # 骨干网络（ResNet / YOLO11Backbone）
│   ├── necks.py             # 颈部融合（YOLO11 PAN-FPN）
│   ├── heads.py             # 检测头（YOLO11 解耦头）
│   ├── segmentation.py      # 分割架构（MiniSegNet / FPNSegNet）
│   └── detection.py         # 检测架构（YOLO11Detector）
│
├── engine/                  # 训练/评估引擎
│   ├── losses.py            # 损失函数（分割 BCE/CE + 检测 CIoU/DFL/BCE）
│   ├── metrics.py           # 评估指标（IoU, Dice, mAP）
│   ├── trainer.py           # Trainer 训练循环
│   └── evaluator.py         # Evaluator 评估循环
│
├── utils/                   # 通用工具
│   ├── common.py            # get_activation, autopad
│   └── visualize.py         # loss 曲线 / 预测对比可视化
│
└── checkpoints/             # 运行输出（权重、日志、图表）
```

## 快速开始

```bash
pip install -r requirements.txt

# === 分割 ===
python train.py --model-type fpnseg

# === 检测 ===
# YOLO11 检测在代码中直接调用 API，详见下方示例
```

---

## 模型架构

### 1. 分割模型

| 模型 | 结构 | 参数 |
|------|------|------|
| MiniSegNet | ResNet18 backbone → 1×1 Conv → upsample | ~11M |
| FPNSegNet | MultiScaleResNet18 → FPN → fused head | ~14M |

### 2. YOLO11 检测模型

```
Input (B,3,H,W)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  YOLO11Backbone (CSP + SPPF)                      │
│                                                   │
│  stem: Conv(s=2) → Conv(s=2)   → (B,32, H/4)     │
│  stage1: C3K2(32→64)            → P3 (B,64, H/8)  │
│  stage2: Conv(s=2) + C3K2(64→128) → P4 (B,128,H/16)│
│  stage3: Conv(s=2)+C3K2(128→256)+SPPF → P5(B,256,H/32)│
└──────────────────────────────────────────────────┘
    │  [P3, P4, P5]
    ▼
┌──────────────────────────────────────────────────┐
│  YOLO11Neck (PAN-FPN)                              │
│                                                   │
│  Top-down:                                        │
│    P5 ──→ Upsample ──+──→ C3K2 → N4               │
│    P4 ───────────────┘                  ↓          │
│    N4 ──→ Upsample ──+──→ C3K2 → N3               │
│    P3 ───────────────┘                             │
│                                                   │
│  Bottom-up:                                       │
│    N3 ──→ Conv(s=2) ──+──→ C3K2 → N4_out          │
│    N4 ────────────────┘                  ↓          │
│    N4_out → Conv(s=2) ──+──→ C3K2 → N5_out        │
│    P5 ──────────────────┘                          │
└──────────────────────────────────────────────────┘
    │  [N3, N4_out, N5_out]
    ▼
┌──────────────────────────────────────────────────┐
│  YOLO11Head (Decoupled)                            │
│                                                   │
│  For each scale:                                  │
│    cls_branch: 2×Conv3x3 + Conv1x1(num_classes)   │
│    reg_branch: 2×Conv3x3 + Conv1x1(4×reg_max)     │
│                                                   │
│  Output: cls_list[N], reg_list[N]                 │
└──────────────────────────────────────────────────┘
```

#### 模型缩放规格

| 规格 | 通道 [c1~c5] | depth_scale | 参数量 | 适用场景 |
|------|-------------|-------------|--------|----------|
| nano | [16,32,64,128,256] | 0.33 | ~5.5M | 移动端 / CPU |
| small | [32,64,128,256,512] | 0.67 | ~16M | 速度/精度平衡 |
| medium | [64,128,256,512,512] | 1.0 | ~40M | 常规 GPU 训练 |

#### 检测损失函数

YOLO11 使用**任务对齐标签分配**（TaskAlignedAssigner）+ 三合一损失：

```
total_loss = 7.5 * box_loss + 0.5 * cls_loss + 1.5 * dfl_loss

box_loss = CIoU Loss（预测框 vs 真实框）
cls_loss = BCE Loss（分类 logits vs one-hot 标签）
dfl_loss = Distribution Focal Loss（边界框分布回归）
```

标签分配策略：每个真实框选择 $alignment_metric = cls_score^α × IoU^β$ 最高的 topk 个锚点作为正样本。

#### 编程接口示例

```python
from models import YOLO11Detector
from engine.losses import YOLODetectionLoss
from configs.config import YOLO11_CONFIGS

# 选择规格
cfg = YOLO11_CONFIGS['nano']

# 构建模型
model = YOLO11Detector(
    num_classes=80,           # 类别数
    reg_max=16,               # DFL 区间数
    backbone_channels=cfg['channels'],
    depth_scale=cfg['depth_scale'],
)

# 前向传播
x = torch.randn(2, 3, 640, 640)
cls_list, reg_list, features, neck_feats = model(x)

# 计算损失
criterion = YOLODetectionLoss(num_classes=80, reg_max=16)
targets = [
    {'labels': torch.tensor([1, 2]), 'boxes': torch.tensor([[0.1,0.1,0.5,0.5],[0.3,0.3,0.7,0.7]])},
    {'labels': torch.tensor([0]), 'boxes': torch.tensor([[0.2,0.2,0.6,0.6]])},
]
loss, items = criterion(cls_list, reg_list, targets, features)
```

## 训练参数

所有参数均可通过 **CLI** 或 **JSON/YAML 配置文件** 指定，参数优先级：`CLI > 配置文件 > 默认值`。

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--cfg` | str | — | JSON/YAML 配置文件路径 |
| `--model-type` | str | fpnseg | 模型架构：miniseg / fpnseg |
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

---

## 开发指南：如何添加新模块

### 1. 添加新的基础网络层（blocks）

在 [models/blocks.py](models/blocks.py) 中定义，使用 `@register_block` 装饰器注册：

```python
@register_block('my_attention')
class MyAttention(nn.Module):
    """自定义注意力模块。"""
    def __init__(self, in_ch: int, reduction: int):
        super().__init__()
        ...

    def forward(self, x):
        ...
```

### 2. 添加新的 Backbone

在 [models/backbones.py](models/backbones.py) 中添加。

**单尺度模式**（返回单个 Tensor）：
```python
class MyBackbone(nn.Module):
    def forward(self, x) -> torch.Tensor
```

**多尺度模式**（返回 List[Tensor]，供 neck 使用）：
```python
class MyMultiScaleBackbone(nn.Module):
    def forward(self, x) -> List[torch.Tensor]  # [feat1, feat2, feat3]
```

### 3. 添加新的 Neck（颈部融合）

在 [models/necks.py](models/necks.py) 中添加。

接口约定：接收多尺度特征列表，返回同序融合特征列表。
```python
class MyNeck(nn.Module):
    def forward(self, features: List[Tensor]) -> List[Tensor]
```

### 4. 添加新的 Head（检测头）

在 [models/heads.py](models/heads.py) 中添加。

检测头接口约定：
```python
class MyHead(nn.Module):
    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]
    # Returns: (cls_outputs, reg_outputs)
```

### 5. 组装完整检测器

在 [models/detection.py](models/detection.py) 中：

```python
class MyDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = MyBackbone()
        self.neck = MyNeck(...)
        self.head = MyHead(...)

    def forward(self, x):
        feats = self.backbone(x)
        fused = self.neck(feats)
        return self.head(fused)
```

### 6. 添加新的损失函数

在 [engine/losses.py](engine/losses.py) 中添加新 Loss 类即可。

### 注册机制概述

```
定义模块 → @register_block('name') → 写入 BLOCK_REGISTRY
                                          ↓
配置列表 → make_layers(cfg) → 查表获取构造器 → nn.Sequential
```

---

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
| `c3k2` | C3k2 | YOLO CSP 瓶颈块 |
| `bottleneck` | Bottleneck | YOLO 标准瓶颈模块 |
| `sppf` | SPPF | 空间金字塔快速池化 |
| `cbam_channel_attention` | CBAM_Channel_Attention | CBAM 通道注意力 |
| `cbam_spatial_attention` | CBAM_Spatial_Attention | CBAM 空间注意力 |
| `cbam` | CBAM | CBAM 组合注意力 |
| `fpn_lateral_conv` | FPNLateralConv | FPN 侧向 1×1 通道对齐 |
| `fpn_output_conv` | FPNOutputConv | FPN 输出 3×3 抗混叠 |
| `maxpool` | MaxPool | 最大池化 |
| `adaptive_max_pool` | AdaptiveMaxPool | 自适应最大池化 |
| `adaptive_avg_pool` | AdaptiveAvgPool | 自适应平均池化 |
| `flatten` | Flatten | 展平层 |
| `linear` | Linear | 全连接层 |

| 未注册顶层模块 | 说明 |
|---------------|------|
| `FPN` | FPN neck 容器，组合 lateral + output convs |
| `MultiScaleResNet18` | 分阶段 ResNet-18，输出 4 尺度特征 |
| `FPNSegNet` | backbone + FPN + head 分割网络 |
| `YOLO11Backbone` | YOLO11 骨干（C3K2 + SPPF） |
| `YOLO11Neck` | YOLO11 PAN-FPN 颈部 |
| `YOLO11Head` | YOLO11 解耦检测头 |
| `YOLO11Detector` | backbone + neck + head 完整检测器 |
| `TaskAlignedAssigner` | 任务对齐标签分配器 |
| `YOLODetectionLoss` | CIoU + DFL + BCE 检测损失 |

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
