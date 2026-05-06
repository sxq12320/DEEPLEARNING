# mastercode

深度学习图像分割项目集合。

## 项目列表

| 项目 | 说明 |
|------|------|
| [0_segment](0_segment/) | 图像分割工程骨架 — 模块化设计、动态组网、ResNet 骨干 |
| 1_study_module | 学习模块 |
| 2_Unet | U-Net 实现 |
| 3_phics_x | PHICS-X 相关 |

---

## 0_segment — 图像分割工程骨架

面向图像分割任务的深度学习工程骨架，支持动态配置网络结构、多种数据格式和多模型架构。

### 项目结构

```
0_segment/
├── configs/            # 全局配置（激活函数映射、网络结构配置）
├── data/               # 数据流水线
│   ├── dataset.py      # 数据集类（支持真实/合成数据）
│   └── transforms.py   # 图像与标签转换（TXT/JSON/NPY → mask）
├── models/             # 模型模块
│   ├── registry.py     # 模块注册中心
│   ├── blocks.py       # 基础网络层（Conv、ResBlock、CBAM 等）
│   ├── builder.py      # 动态组网（cfg → nn.Sequential）
│   ├── backbones.py    # 骨干网络（ResNet-18）
│   └── segmentation.py # 分割架构（MiniSegNet）
├── engine/             # 训练与评估引擎
│   ├── losses.py       # 损失函数
│   ├── metrics.py      # 评估指标（IoU、Dice、mAP）
│   ├── trainer.py      # 训练循环
│   └── evaluator.py    # 评估循环
├── utils/              # 工具函数
│   ├── common.py       # 通用函数
│   └── visualize.py    # 可视化（loss 曲线、预测对比图）
├── scripts/            # CLI 入口
│   ├── train.py        # 训练入口
│   └── predict.py      # 预测入口
├── checkpoints/        # 模型权重
├── logs/               # 日志
├── runs/               # 训练/预测输出
└── requirements.txt    # 依赖
```

### 快速开始

```bash
# 安装依赖
pip install -r requirements.txt

# 一键训练（默认配置）
python train.py

# 覆盖超参数
python train.py --imgsz 256 --epochs 50 --batch 16 --lr 5e-4

# 使用 JSON 配置
python train.py --cfg configs/train.json

# 训练（自动使用合成数据验证链路）
python scripts/train.py --epochs 10 --batch 8

# 使用真实数据训练
python scripts/train.py \
  --image-dir data/images \
  --mask-dir data/masks \
  --epochs 50 --batch 16

# 预测
python scripts/predict.py \
  --source path/to/image.jpg \
  --weights runs/train/exp/weights/best.pt
```

### 一键训练入口

- 推荐使用 [train.py](train.py) 作为统一训练入口，支持 CLI 覆盖与 JSON 配置。
- 可直接编辑 [configs/train.json](configs/train.json)，也可运行 `python train.py --print-cfg` 查看合并后的配置。

### 核心特性

- **注册机制**：通过 `@register_block` 装饰器注册自定义网络层，`make_layers(cfg)` 动态构建网络
- **配置驱动**：网络结构完全由配置列表定义，无需修改代码即可更换架构
- **多标签格式**：支持 mask 图片、TXT（YOLO 格式）、JSON（COCO 格式）、NPY 四种标签格式
- **自动回退**：无真实数据时自动使用合成数据验证训练链路

### 可用模块

- **卷积块**: Conv, Basic_Conv_Block, DepthWise_Conv, PointWise_Conv, DepthWiseSeparable_Conv
- **残差块**: ResNetBlock_34, ResNetBlock_50
- **注意力**: CBAM（通道注意力 + 空间注意力）
- **池化**: MaxPool, AdaptiveAvgPool
- **其他**: Flatten, Linear
