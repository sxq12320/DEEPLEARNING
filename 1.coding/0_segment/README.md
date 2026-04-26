# 0_segment

一个面向图像分割任务的深度学习工程骨架，包含：

- 配置管理
- 数据读取与标签转换
- 可复用网络模块（Block）
- 模块注册与动态构建
- 最小可运行 train/eval 闭环

---

## 1. 项目结构

```text
0_segment/
├─ checkpoints/                # 模型权重与训练断点
├─ configs/
│  ├─ __init__.py
│  └─ config.py                # 激活函数映射、网络配置
├─ data/
│  ├─ __init__.py
│  ├─ datasets/
│  │  ├─ __init__.py
│  │  └─ rgb_dataset.py        # RGB 数据集实现
│  ├─ preprocessing/
│  │  ├─ __init__.py
│  │  └─ enhance_image.py      # 图像增强（当前为占位）
│  └─ transforms/
│     ├─ __init__.py
│     ├─ image_transform.py    # 图像尺寸变换
│     └─ label_transform.py    # TXT/JSON/NPY -> mask
├─ logs/                       # 日志目录
├─ losses/                     # 损失函数目录（待扩展）
├─ metrics/                    # 评估指标目录（待扩展）
├─ models/
│  ├─ __init__.py
│  ├─ architectures/
│  │  ├─ __init__.py
│  │  └─ model.py              # 架构聚合入口
│  ├─ backbones/
│  │  ├─ __init__.py
│  │  └─ resnet_backbone.py
│  ├─ blocks/
│  │  ├─ __init__.py
│  │  └─ blocks.py             # Conv/ResBlock/CBAM 等基础模块
│  ├─ builders/
│  │  ├─ __init__.py
│  │  └─ builder.py            # cfg -> nn.Sequential
│  └─ registries/
│     ├─ __init__.py
│     └─ registry.py           # BLOCK_REGISTRY 与装饰器
├─ utils/
│  ├─ __init__.py
│  └─ common.py                # 通用函数（激活函数、autopad）
├─ train.py                    # 最小可运行训练入口
├─ evaluate.py                 # 最小可运行评估入口
└─ requirements.txt            # 一键安装依赖
```

---

## 2. 模块职责

### 2.1 configs

- 文件：configs/config.py
- 作用：集中维护全局配置。
- 关键对象：
  - ACTIVATION_MAP: 激活函数映射表。
  - RESNET_18_CFG: 网络构建配置示例。

### 2.2 data

- data/datasets：负责数据集封装。
- data/transforms：负责图像与标签格式转换。
- data/preprocessing：负责增强和预处理。

### 2.3 models

- models/blocks：可复用基础网络层。
- models/registries：模块注册中心。
- models/builders：根据配置动态组网。
- models/backbones：骨干网络入口。

### 2.4 utils

- 仅保留通用工具能力，当前包含：
  - get_activation
  - autopad

---

## 3. 接口文档

### 3.1 数据接口

1) get_dataset_rgb

- 位置：data/datasets/rgb_dataset.py
- 原型：

```python
get_dataset_rgb(
    image_dir,
    label_dir,
    label_type="mask",
    target_size=(640, 640),
)
```

- 说明：读取 RGB 图像与标签，返回 PyTorch Dataset。
- 返回项：
  - image: Tensor, shape 通常为 [3, H, W]
  - label: Tensor, shape 通常为 [H, W, 1]

2) 标签转换函数

- 位置：data/transforms/label_transform.py
- 接口：
  - TXT2MASK(label_dir, image_name, target_size)
  - JSON2MASK(label_dir, image_name, target_size)
  - NPY2MASK(label_dir, image_name, target_size)

3) 图像变换函数

- 位置：data/transforms/image_transform.py
- 接口：
  - image_transform(image_path, target_size=(640, 640))

### 3.2 模型接口

1) 注册器

- 位置：models/registries/registry.py
- 接口：

```python
register_block(name)
BLOCK_REGISTRY
```

2) 构建器

- 位置：models/builders/builder.py
- 接口：

```python
make_layers(cfg) -> nn.Sequential
```

3) 常用 block（节选）

- 位置：models/blocks/blocks.py
- 接口（类）：
  - MaxPool
  - AdaptiveAvgPool
  - Conv
  - Basic_Conv_Block
  - Conv_Block_NONB
  - DepthWise_Conv
  - PointWise_Conv
  - DepthWiseSeparable_Conv
  - ResNetBlock_34
  - ResNetBlock_50
  - CBAM_Channel_Attention
  - CBAM_Spatial_Attention
  - CBAM
  - Flatten
  - Linear

### 3.3 训练入口接口

- 位置：train.py
- 目标：保证最小可运行训练闭环。
- 特性：
  - 无真实数据时，自动使用合成数据跑通训练。
  - 有真实数据目录时，自动切换到 data.datasets.get_dataset_rgb。
  - 默认保存 checkpoint 到 checkpoints/minimal_last.pt。

核心参数：

- --epochs
- --batch-size
- --lr
- --image-size
- --image-dir
- --label-dir
- --label-type
- --checkpoint-dir
- --checkpoint-name
- --cpu

### 3.4 评估入口接口

- 位置：evaluate.py
- 目标：基于 train.py 输出的 checkpoint 进行最小评估闭环验证。
- 特性：
  - 与 train.py 共用数据构建逻辑（支持真实数据与合成数据）。
  - 默认读取 checkpoints/minimal_last.pt。
  - 输出 loss 与 IoU 两个核心指标。

核心参数：

- --checkpoint
- --batch-size
- --image-size
- --threshold
- --image-dir
- --label-dir
- --label-type
- --cpu

---

## 4. 使用说明

### 4.1 环境准备

建议使用 Python 3.8+，并执行：

```bash
pip install -r requirements.txt
```

### 4.2 最小可运行（无真实数据）

```bash
python train.py --epochs 2 --batch-size 4
```

### 4.3 使用真实数据训练

```bash
python train.py \
  --epochs 10 \
  --batch-size 8 \
  --image-dir data/your_images \
  --label-dir data/your_labels \
  --label-type mask
```

### 4.4 评估（最小闭环）

先训练得到 checkpoint，再执行：

```bash
python evaluate.py --checkpoint checkpoints/minimal_last.pt
```

也支持真实数据评估：

```bash
python evaluate.py \
  --checkpoint checkpoints/minimal_last.pt \
  --image-dir data/your_images \
  --label-dir data/your_labels \
  --label-type mask
```

---

## 5. 新路径导入规范

项目内导入统一采用新路径，不再依赖旧入口。

推荐示例：

```python
from data.datasets import get_dataset_rgb
from data.transforms import image_transform, TXT2MASK
from data.preprocessing import enhance_image

from models.blocks import Basic_Conv_Block
from models.builders import make_layers
from models.registries import BLOCK_REGISTRY, register_block
```

---

## 6. 当前状态说明

- train.py 与 evaluate.py 可直接组成最小 train/eval 闭环。
- data/preprocessing/enhance_image.py 当前为占位实现（pass），不影响最小训练流程。
- losses 与 metrics 目录已预留，后续可按任务补充实现。