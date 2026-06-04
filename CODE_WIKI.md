# Code Wiki — RGB-D Multi-Modal Detection & Segmentation Project

> 基于 PyTorch 的深度学习研究项目，聚焦于 **RGB-D 多模态目标检测与图像分割**，使用控制理论（卡尔曼滤波、ESO、IDAPBC、滑模控制）改进 RGB 与 Depth 模态间的特征融合。

---

## Table of Contents

1. [项目概述](#1-项目概述)
2. [仓库整体结构](#2-仓库整体结构)
3. [核心子项目：ultralytics-main](#3-核心子项目ultralytics-main)
4. [核心子项目：0_segment](#4-核心子项目0_segment)
5. [学习模块：1_study_module](#5-学习模块1_study_module)
6. [U-Net 实现：2_Unet](#6-u-net-实现2_unet)
7. [PHICS-X 模块：3_phics_x](#7-phics-x-模块3_phics_x)
8. [依赖关系总览](#8-依赖关系总览)
9. [项目运行方式](#9-项目运行方式)
10. [开发指南](#10-开发指南)

---

## 1. 项目概述

### 研究目标

利用控制理论方法改进 YOLO11 框架下的 RGB-D（彩色+深度）多模态特征融合，在目标检测和图像分割任务上提升精度。核心创新包括：

- **KalmanGatedFusion** — 基于卡尔曼滤波的门控融合（浅层 P3，抑制深度噪声）
- **ESOFusion** — 扩展状态观测器融合（中层 P4，补偿模态失配）
- **IDAPBCFusion** — 互联阻尼与无源控制融合（深层 P5，能量最优）
- **SMCScheduler** — 滑模控制学习率调度器（逃离鞍点/损失平台区）
- **MuSGD / Muon** — 正交化优化器

### 技术栈

| 层级 | 技术 |
|------|------|
| 深度学习框架 | PyTorch 2.2.2 + torchvision 0.17.2 |
| 目标检测框架 | 修改版 Ultralytics YOLO11 |
| 图像处理 | OpenCV 4.x, Pillow |
| 数据处理 | NumPy, YAML |
| 可视化 | Matplotlib, tqdm |
| 自定义算子 | Triton（LSNet SKA 内核） |

---

## 2. 仓库整体结构

```
workspace/
├── ultralytics-main/           # ★ 主要研究代码 — 修改版 YOLO + 控制理论融合
│   ├── ultralytics/            #   核心库
│   │   ├── nn/modules/         #   自定义模块（CT 融合、SMC 调度器）
│   │   ├── cfg/models/11/      #   YOLO11 配置文件
│   │   ├── engine/trainer.py   #   训练引擎（含 SMC 集成）
│   │   ├── optim/muon.py       #   Muon / MuSGD 优化器
│   │   └── rgb_d_dataset.py    #   RGBD 数据集加载器
│   └── *.py                    #   脚本（0xx=测试, 1xx=数据预处理, 2xx=配置, 3xx=优化器）
│
├── 1.coding/
│   ├── 0_segment/              # ★ 分割/检测工程骨架 — 注册机制 + 配置驱动
│   │   ├── configs/            #   全局配置
│   │   ├── datasets/           #   数据流水线
│   │   ├── models/             #   模型库（注册中心、模块、组网）
│   │   ├── engine/             #   训练/评估引擎
│   │   ├── utils/              #   工具函数
│   │   └── train.py            #   训练入口
│   │
│   ├── 1_study_module/         # 经典网络学习实现（16个子模块）
│   ├── 2_Unet/                 # U-Net 分割实现
│   └── 3_phics_x/              # PHICS-X 物理信息模块
│
├── 2_catoon/                   # 动画学习（Manim）
├── CLAUDE.md                   # 开发指引
└── SMCScheduler_开发记录.md     # SMC 调度器开发文档
```

---

## 3. 核心子项目：ultralytics-main

### 3.1 架构概述

这是对 Ultralytics YOLO 框架的修改版本，增加了 RGB-D 多模态输入支持和控制理论融合模块。

#### 整体数据流

```
4 通道 RGBD 输入
        │
   SplitChannels ──┬── RGB 流 (标准 YOLO11 卷积 + C3k2)
   [0,1,2]         │
                   ├── P3 (80×80) ── KalmanGatedFusion
                   ├── P4 (40×40) ── ESOFusion
                   └── P5 (20×20) ── IDAPBCFusion
                            │
   Depth 流 ────────┬── P3 (轻量卷积)
   [3]              ├── P4
                    └── P5
                            │
        SPPF → C2PSA → FPN+PAN Neck → Segment Head
                            │
                   SMCScheduler (优化器步级别)
```

### 3.2 控制理论融合模块

**文件**: `ultralytics/nn/modules/ct_modules.py`

| 类 | 作用层级 | 核心原理 | 输入 | 输出 |
|---|---------|---------|------|------|
| **MultiScaleVarianceEstimator** | 辅助 | 多尺度池化估计特征图空间方差（3×3/5×5/7×7 + 1×1 校准器，Softplus 保证正定） | `(B, C, H, W)` | `(B, 1, H, W)` 方差图 |
| **KalmanGatedFusion** | P3 (浅层) | 卡尔曼增益 `K = σ²_dep / (σ²_rgb + σ²_dep)`；状态更新 `F_fused = F_rgb + K × (F_dep - F_rgb)` | `[f_rgb, f_dep]` | 融合特征（RGB 通道数） |
| **ESOFusion** | P4 (中层) | 估计"总扰动" z2（遮挡/噪声/模态差异），补偿 `F_comp = F_rgb - β1×z2 + u`，与 P3 拼接输出 | `[f_rgb_p4, f_fused_p3]` | `(B, c_p4+c_p3, H, W)` |
| **IDAPBCFusion** | P5 (深层) | SE 风格能量门控调制深度特征：`F_dep_guided = F_dep × rgb_energy` | `[f_rgb_p5, f_fused_p4]` | `(B, c_p5, H, W)` |
| **BypassModule** | 所有层级 | 消融基线 — 简单投影+相加，无控制逻辑 | `[f1, f2]` | `f1 + proj(f2)` |
| **SplitChannels** | 输入层 | 从 4 通道 RGBD 中提取指定通道 | `(B, 4, H, W)` | 子集通道 |
| **BLFLoss** | 损失 | Barrier Lyapunov Function 损失，确保"amodal"预测包含"visible"掩码 | `pred_visible, pred_amodal` | 标量损失 |

### 3.3 YOLO11-CT 模型配置变体

**配置目录**: `ultralytics/cfg/models/11/`

| 配置文件 | P3 融合 | P4 融合 | P5 融合 | 用途 |
|---------|---------|---------|---------|------|
| `yolo11-base-rgbd.yaml` | BypassModule | BypassModule | BypassModule | 基线（无控制理论） |
| `yolo11-ct-A.yaml` | KalmanGatedFusion | BypassModule | BypassModule | 消融 A（仅卡尔曼） |
| `yolo11-ct-AB.yaml` | KalmanGatedFusion | ESOFusion | BypassModule | 消融 A+B |
| `yolo11-ct-ABC.yaml` | KalmanGatedFusion | ESOFusion | IDAPBCFusion | 完整模型（三模块全开） |

**共有特征**:
- `channels: 4` — 4 通道 RGBD 输入
- 双流骨干：RGB 流（标准 C3k2）+ Depth 流（仅卷积，无 C3k2）
- SplitChannels 在层 1 (`[0,1,2]`) 和层 9/11 (`[3]`) 分离 RGB 和 Depth
- 三个融合点分别在 P3/8, P4/16, P5/32
- 标准 YOLO11 FPN+PAN Neck + Segment Head

### 3.4 SMCScheduler — 滑模控制学习率调度器

**文件**: `ultralytics/nn/modules/smc_scheduler.py`

**核心机制**:

```
滑模面:  s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)
         ────   ───────────   ────────────────────
         系数    梯度绝对幅值      梯度变化率
```

**触发条件（OR 逻辑）**:
1. 滑模面停滞：`|s_t|/peak < threshold` 持续 `patience` 步
2. Loss 平台 + 部分滑模面停滞

**Escape 响应**（温和、非破坏性）:
- 注入相对梯度噪声（`noise_scale × ||grad||`，默认 0.003）
- 学习率提升 `lr_boost` 倍（默认 1.2×）
- β₁ 从 0.9 降至 0.85（增加动量响应性）

**安全措施**:
- 不直接推动参数
- 不重置 Adam 状态（保持 exp_avg / exp_avg_sq）
- Warmup 期间跳过全部 SMC 逻辑
- Peak 每步衰减 0.999，避免冷启动永久抬高

**集成到训练循环** (`ultralytics/engine/trainer.py`):

```python
# 训练前：构建 SMC 调度器
self.smc_scheduler = SMCScheduler(self.optimizer, total_steps=iterations, ...)

# 每步训练：
self.smc_scheduler.observe_gradients()   # optimizer.step() 之前
self.optimizer.step()
self.smc_scheduler.step(self.loss.item()) # optimizer.step() 之后

# 每 Epoch 结束：
self.smc_scheduler.on_train_epoch_end(epoch_loss)
```

**默认超参数** (`ultralytics/cfg/default.yaml`):

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `smc_surface_threshold` | 0.1 | 滑模面停滞阈值 |
| `smc_surface_patience` | 50 | 停滞步数触发 escape |
| `smc_lr_boost` | 1.2 | escape 时 LR 提升倍数 |
| `smc_noise_scale` | 0.003 | 梯度噪声相对量级 |
| `smc_beta1_low` | 0.85 | escape 时 β₁ 值 |

### 3.5 自定义优化器

**文件**: `ultralytics/optim/muon.py`

| 类 | 说明 |
|---|------|
| **Muon** | 纯 Muon 优化器 — 所有参数接收正交化更新（Newton-Schulz 5 次迭代），乘法权重衰减 |
| **MuSGD** | Muon + SGD 混合优化器 — `use_muon=True` 的参数同时接收 Muon 和 SGD 更新，`use_muon=False` 的参数仅 SGD |

**核心函数**:
- `zeropower_via_newtonschulz5(G)` — Newton-Schulz 迭代计算矩阵正交化（替代 SVD 的 UV^T）
- `muon_update(grad, momentum, beta, nesterov)` — 应用动量、正交化、缩放

**PIDAO 优化器** (`ultralytics/engine/trainer.py` 内)：多通道高阶 PID 优化器（P + I + 多阶 D），当前已注释未激活。

### 3.6 RGBD 数据集

**文件**: `ultralytics/rgb_d_dataset.py`

| 类 | 说明 |
|---|------|
| **RGBDDataset** | 继承 YOLODataset，读取深度图（.npy 或图像格式），resize 后与 RGB 拼接为 4 通道 `(H, W, 4)` |
| **RGBDDataset_pic** | 支持 16 位 PNG 深度图（IMREAD_ANYDEPTH），深度值除以 1000 转换为米 |

### 3.7 脚本编号规范

| 编号范围 | 用途 | 示例 |
|---------|------|------|
| `0xx` | 模型测试与评估 | `000_test.py`, `001_yolo11_test.py` |
| `1xx` | 数据预处理与格式转换 | `101_json2yolo.py`, `102_VOC_TO_YOLO.py` |
| `2xx` | 数据集 YAML 配置 | `201_caomei_data.yaml` |
| `3xx` | 自定义优化器实现 | `301_optimizer_PIDAO.py` |

---

## 4. 核心子项目：0_segment

### 4.1 架构概述

面向图像分割和目标检测的深度学习工程骨架，采用**模块化设计 + 注册机制 + 配置驱动组网**架构。

**支持 3 种模型系列**:
1. **MiniSegNet** — 极简分割网络（ResNet-18 backbone → 1×1 Conv → 上采样）
2. **FPNSegNet** — FPN 多尺度分割（MultiScaleResNet18 → FPN → 融合头）
3. **TSDualSegDetNet** — 双模态分割+检测（RGB+Depth+Prompt 输入，跨模态注意力）
4. **YOLO11Detector** — 完整 YOLO11 检测器（Backbone + PAN-FPN + 解耦头）

### 4.2 项目结构

```
0_segment/
├── train.py                 # 训练入口（CLI + JSON/YAML 配置合并）
├── requirements.txt         # 依赖
├── configs/
│   └── config.py            # 激活函数映射 + ResNet/YOLO 网络结构配置
├── datasets/
│   ├── dataset.py           # SegmentationDataset（真实/合成数据自动切换）
│   └── parsers.py           # TXT / JSON / NPY → mask 标签转换
├── models/
│   ├── modules.py           # 基础模块（Conv, ResBlock, CBAM, C3K2, SPPF, FPN ...）
│   ├── backbones.py         # 骨干网络（ResNet / YOLO11Backbone / TSDualBackbone）
│   ├── necks.py             # 颈部融合（YOLO11 PAN-FPN / AFPN / DyHead）
│   ├── heads.py             # 检测头（YOLO11 解耦头 / DecoupledSegDetHead）
│   ├── networks.py          # 完整网络架构（MiniSegNet, FPNSegNet, TSDualSegDetNet, YOLO11Detector）
│   └── __init__.py          # 统一导出
├── engine/
│   ├── losses.py            # 损失函数（分割 BCE/CE + 检测 CIoU/DFL/BCE）
│   └── metrics.py           # 评估指标（IoU, Dice, mAP）
├── utils/
│   ├── registry.py          # 模块注册中心
│   ├── builder.py           # make_layers + build_backbone/necks/heads
│   ├── common.py            # get_activation, autopad
│   └── visualize.py         # loss 曲线 / 预测对比可视化
└── checkpoints/             # 运行输出（权重、日志、图表）
```

### 4.3 注册机制

**文件**: `utils/registry.py`

四个独立注册表，通过装饰器注册模块类：

| 装饰器 | 注册表 | 用途 |
|--------|--------|------|
| `@register_block(name)` | `BLOCK_REGISTRY` | 基础网络层 |
| `@register_backbone(name)` | `BACKBONE_REGISTRY` | 骨干网络 |
| `@register_neck(name)` | `NECK_REGISTRY` | 颈部融合 |
| `@register_head(name)` | `HEAD_REGISTRY` | 预测头 |

**使用方式**:
```python
@register_block('my_attention')
class MyAttention(nn.Module):
    def __init__(self, in_ch, reduction): ...
    def forward(self, x): ...
```

注册后通过 `make_layers(cfg)` 或 `build_backbone/necks/heads(cfg)` 从配置动态构建。

### 4.4 已注册模块一览

#### 基础模块 (`models/modules.py`)

| 注册名 | 类名 | 说明 |
|--------|------|------|
| `conv` | Conv | 基础卷积（无 BN/激活） |
| `basic_conv_block` | Basic_Conv_Block | Conv + BN + Activation |
| `conv_block_nonb` | Conv_Block_NONB | Conv + Activation（无 BN） |
| `depthwise_conv` | DepthWise_Conv | 逐通道卷积（groups=in_ch） |
| `pointwise_conv` | PointWise_Conv | 1×1 逐点卷积 |
| `depthwise_separable_conv` | DepthWiseSeparable_Conv | 可分离卷积（DW + PW） |
| `resnet_block_34` | ResNetBlock_34 | ResNet-34 基础残差块（2×3×3 conv） |
| `resnet_block_50` | ResNetBlock_50 | ResNet-50 瓶颈块（1×1→3×3→1×1） |
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

#### 骨干网络 (`models/backbones.py`)

| 注册名 | 类名 | 说明 |
|--------|------|------|
| `resnet18` | ResNet18 | 配置驱动 ResNet-18（通过 make_layers） |
| `multiscale_resnet18` | MultiScaleResNet18 | 分阶段 ResNet-18，输出 4 尺度特征 `[c2, c3, c4, c5]` |
| `yolo11_backbone` | YOLO11Backbone | CSP 风格骨干：stem → C3K2 阶段 → SPPF，输出 `[P3, P4, P5]` |
| `ts_dual_backbone` | TSDualBackbone | 双分支（RGB+prompt + depth），3 尺度 CrossTokenStatsAttention → 1×1 融合 |

#### 颈部融合 (`models/necks.py`)

| 注册名 | 类名 | 说明 |
|--------|------|------|
| `yolo11_neck` | YOLO11Neck | PAN-FPN：top-down（上采样 + C3K2）+ bottom-up（下采样 + C3K2） |
| `afpn_neck` | AFPNNeck | 自适应 FPN：3 尺度渐进式融合 → 单一输出张量 |
| `dyhead_neck` | DyHeadNeck | Dynamic Head：ScaleAwareAttn → SpatialAwareAttn → TaskAwareAttn |

#### 检测头 (`models/heads.py`)

| 注册名 | 类名 | 说明 |
|--------|------|------|
| `yolo11_head` | YOLO11Head | 解耦检测头：每尺度 cls_branch + reg_branch |
| `decoupled_segdet_head` | DecoupledSegDetHead | 分割+检测头：bbox_branch + mask_branch |

### 4.5 网络架构 (`models/networks.py`)

| 类 | 架构 | 参数量 |
|---|------|--------|
| **MiniSegNet** | ResNet-18 backbone → 1×1 Conv head → bilinear upsample | ~11M |
| **FPNSegNet** | MultiScaleResNet18 → FPN → concat → 3×3 conv → 1×1 head → upsample | ~14M |
| **TSDualSegDetNet** | Config-driven: backbone → neck → head；接受 (rgb, prompt, depth) | — |
| **YOLO11Detector** | YOLO11Backbone → YOLO11Neck → YOLO11Head；含 `forward()` 和 `predict()` | 按规格 |

**YOLO11 检测器缩放规格**:

| 规格 | 通道 `[c1~c5]` | depth_scale | 参数量 | 适用场景 |
|------|---------------|-------------|--------|----------|
| nano | [16,32,64,128,256] | 0.33 | ~5.5M | 移动端 / CPU |
| small | [32,64,128,256,512] | 0.67 | ~16M | 速度/精度平衡 |
| medium | [64,128,256,512,512] | 1.0 | ~40M | 常规 GPU 训练 |

### 4.6 损失函数 (`engine/losses.py`)

#### 分割损失

| 类 | 说明 |
|---|------|
| **SegmentationLoss** | BCEWithLogitsLoss 或 CrossEntropyLoss 包装器 |
| **NWDLoss** | 归一化 Wasserstein 距离框损失 |
| **FourierLoss** | 频域损失：rfft2 → 幅度差 L1 |
| **SegDetLoss** | 组合损失：`mask_weight × mask_loss + fourier_weight × fourier_loss + bbox_weight × nwd_loss` |

#### YOLO 检测损失

| 名称 | 说明 |
|------|------|
| **TaskAlignedAssigner** | 动态标签分配：`alignment_metric = cls_score^α × IoU^β`，选择 topk 锚点 |
| **ciou_loss** | Complete IoU 损失：IoU + 中心距离 + 宽高比 |
| **distribution_focal_loss** | DFL：softmax 加权交叉熵到左右相邻 bin |
| **YOLODetectionLoss** | 完整 YOLO11 损失：`7.5 × box_loss + 0.5 × cls_loss + 1.5 × dfl_loss` |

### 4.7 评估指标 (`engine/metrics.py`)

| 函数 | 说明 |
|------|------|
| `compute_iou()` | 批量平均二元 IoU（Jaccard），带平滑 |
| `compute_dice()` | 批量平均 Dice 系数 |
| `compute_box_iou()` | 每样本框 IoU（xyxy 归一化坐标） |
| `compute_pr_from_iou()` | 多 IoU 阈值下 TP/FP/FN 统计 |
| `compute_ap()` | 从 PR 曲线计算 AP（包络插值） |
| `ap_per_class()` | Ultralytics 风格 mAP：按类别/阈值计算 PR |

### 4.8 数据流水线

**数据格式支持** (`datasets/parsers.py`):

| 格式 | label_type | 说明 |
|------|------------|------|
| mask 图片 | `mask` | 直接读取 PNG/JPG 等掩码图像 |
| YOLO TXT | `txt` | 矩形框或多边形标注 |
| COCO JSON | `json` | COCO 格式标注文件 |
| NumPy | `npy` | .npy 格式掩码数组 |

**数据集类** (`datasets/dataset.py`):

| 类 | 说明 | 返回 |
|---|------|------|
| **SegmentationDataset** | 单模态（RGB + mask），支持 4 种标签格式，无真实数据时自动使用合成数据 | `(img C,H,W, mask 1,H,W)` |
| **MultiModalSegmentationDataset** | 多模态（RGB + prompt + depth + mask） | `(rgb 3,H,W, prompt 1,H,W, depth 1,H,W, mask 1,H,W)` |

### 4.9 训练入口 (`train.py`)

**配置系统**（3 层优先级：默认值 < 配置文件 < CLI 参数）:

```bash
python train.py \
  --model-type ts_dual \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --depth-dir /path/to/depth \
  --prompt-dir /path/to/prompts \
  --imgsz 256 \
  --epochs 50 \
  --batch 16 \
  --lr 1e-3
```

**关键 CLI 参数**:

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `--model-type` | str | fpnseg | 模型：miniseg / fpnseg / ts_dual |
| `--imgsz` | int | 128 | 输入图像尺寸 |
| `--epochs` | int | 20 | 训练轮数 |
| `--batch` | int | 8 | 批次大小 |
| `--lr` | float | 1e-3 | 学习率 |
| `--label-type` | str | mask | 标签格式：mask / txt / json / npy |
| `--cfg` | str | — | JSON/YAML 配置文件路径 |
| `--cpu` | flag | False | 强制 CPU 模式 |
| `--seed` | int | 22 | 随机种子 |

**训练流程**:
1. 选择设备（CUDA / CPU）
2. 构建模型（根据 `--model-type`）
3. 构建数据集与 DataLoader
4. Adam 优化器 + 损失函数
5. 训练循环（tqdm 进度条，支持 batch/epoch 级别统计）
6. 每 Epoch 保存 best.pt / last.pt，记录日志，绘制 loss 曲线

---

## 5. 学习模块：1_study_module

位于 `1.coding/1_study_module/`，包含 16 个经典深度学习架构的学习实现。

### 5.1 经典 CNN 分类模型

| 模块 | 文件 | 架构要点 |
|------|------|----------|
| **LeNet-5** | `1_LeNet/LENET-5.py` | 最早 CNN：2 卷积层 (5×5, 6→16) + 2 平均池化 + 3 全连接 (120→84→classes) |
| **AlexNet** | `2_AlexNet/AlexNet.py` | 5 卷积层 (11×11, 5×5, 3×3) + 最大池化 + 3 全连接 (2048→2048→classes)，Dropout 0.5 |
| **VGGNet** | `3_VGGNet/VGGNet_A~E.py` | A-E 五个变体，堆叠 3×3 卷积替代大核；VGG-A 11 层：64→128→256→512→512 + 3 FC (4096→4096→1000) |
| **NIN** | `9_NIN/NIN.py` | Network In Network：用 1×1 卷积作为"微型 MLP"，全局平均池化替代全连接 |
| **GoogleNet** | `10_GoogleNet/GoogleNet.py` | Inception 模块：多尺度卷积 (1×1, 3×3, 5×5) 并行；辅助分类器 |

### 5.2 注意力与通道-空间模块

| 模块 | 文件 | 架构要点 |
|------|------|----------|
| **SEBlock** | `4_SEBlock/seblock.py` | Squeeze-and-Excitation：GAP → FC-reduce → ReLU → FC-expand → Sigmoid → 通道加权 |
| **CBAM** | `5_CBAM/cbam.py` | 通道注意力 (SE 类似) + 空间注意力 (通道方向 avg/max concat → 7×7 conv → sigmoid) 串联 |

### 5.3 残差与密集架构

| 模块 | 文件 | 架构要点 |
|------|------|----------|
| **ResNet** | `11_ResNet/ResNet.py` | 两种残差块：ResNetBlock_34（2×3×3 conv）、ResNetBlock_50（瓶颈 1×1→3×3→1×1）；支持 ResNet-18/34/50/101/152 |
| **DenseNet** | `12_DenseNet/DenseNet.ipynb` | 密集连接：4 个 DenseBlock (6,12,24,16 层)，Transition 层减半通道，特征复用 |

### 5.4 Transformer 系列

| 模块 | 文件 | 架构要点 |
|------|------|----------|
| **Transformer** | `7_TransFormer/1_based_TransFormer.py` | 从零实现：ScaledDotProductAttention → MultiHeadAttention → EncoderLayer → DecoderLayer → 完整 TransFormer；支持正弦位置编码、mask 生成、Xavier 初始化 |
| **ViT** | `8_VIT/VIT.ipynb` | Vision Transformer：patch embedding → transformer encoder → classification head |

### 5.5 轻量/高效网络

| 模块 | 文件 | 架构要点 |
|------|------|----------|
| **MobileNet-V1** | `6_MobileNet-V1/MobileNet_v1.py` | 深度可分离卷积（depthwise 3×3 + pointwise 1×1） |
| **MobileNet-V2** | `14_MobileNet-V2/MobileNet_v2.ipynb` | 倒残差块：expand → depthwise → project（线性瓶颈） |
| **ConvNeXt** | `13_ConvNEXT/convnext.ipynb` | ViT 启发的现代 CNN：大核深度卷积、LayerNorm（替代 BN）、GELU |
| **LSNet** | `16_LSNET/` | 轻量视觉网络：RepVGGDW 可融合深度残差、LSConv 动态空间核（Triton 自定义 CUDA）、多头自注意力（仅 stage 3）、FFN；变体 T/S/B |

### 5.6 分割模型：FCN

| 模块 | 文件 | 架构要点 |
|------|------|----------|
| **FCN** | `15_FCN/model.py, net.py` | FCN-32s/16s/8s 三个变体；手写 VGG16 backbone；FC 层卷积化（1×1 conv）；转置卷积上采样（双线性初始化）；FCN-8s 融合 pool3/4/5 跳跃连接 |

---

## 6. U-Net 实现：2_Unet

**目录**: `1.coding/2_Unet/`

### 架构

```
Input → Encoder ──────────────────────────────→ Decoder → Output
         │                                       │
  64 → ConvBlock → Down ── P1 ────┐         ┌──→ Up + ConvBlock → 64
                                    │         │
  128 → ConvBlock → Down ── P2 ──┐ │    ┌───→ Up + ConvBlock → 128
                                  │ │    │
  256 → ConvBlock → Down ── P3 ┐ │ │ ┌→ Up + ConvBlock → 256
                                │ │ │
  512 → ConvBlock → Down ── P4 ┘ │ │
                                  │ │
       1024 → ConvBlock (Bottleneck) │
                                     │
  1×1 Conv → 21 classes (VOC 分割)
```

### 关键文件

| 文件 | 职责 |
|------|------|
| `net.py` | U-Net 网络定义：ConvBlock (3×3 conv→BN→Dropout→LeakyReLU ×2)、DownSampleBlock (stride-2 卷积替代最大池化)、UpSampleBlock (最近邻插值 + 1×1 conv + 通道拼接) |
| `data.py` | VOC2007 数据集加载；图像增强（双边滤波 → CLAHE → 锐化）；掩码处理（最近邻 resize） |
| `train.py` | 训练循环：AdamW 优化器、CrossEntropyLoss (ignore_index=255)、YOLO 风格训练日志 |
| `utils.py` | 工具函数 |

**通道进度**: 64→128→256→512→1024 (bottleneck) →512→256→128→64

---

## 7. PHICS-X 模块：3_phics_x

**目录**: `1.coding/3_phics_x/`

| 文件 | 职责 |
|------|------|
| `models.py` | 模型定义（含 ResNet18 部分实现） |
| `modules.py` | 基础模块 |
| `utils.py` | 工具函数 |

### 核心组件

- **`cba`** — 可配置的 Conv→BN→Activation 模块，支持 15 种激活函数（通过 `ACTIVATION_MAP` 选择：relu, leakyrelu, prelu, relu6, silu, gelu, elu, selu, mish, hardswish, sigmoid, tanh, identity）
- **`MaxPool`** — F.max_pool2d 包装器
- **ResNet18** — 部分实现（仅初始层）

### 通用注意力混合模块 (`1.coding/modules.py`)

**Attention Parallel Feature Mixer**: 融合两个特征图 (FA, FB) 使用双重注意力
- 通道注意力：GAP + GMP → 1×1 conv bottleneck → sigmoid
- 空间注意力：1×1 conv bottleneck → sigmoid
- 输出：`w * FA + (1-w) * FB`，其中 `w = sigmoid(ch_output + sp_output)`

---

## 8. 依赖关系总览

### 主要依赖图

```
ultralytics-main/
├── 依赖 PyTorch 2.2.2 + torchvision
├── 依赖 OpenCV, NumPy, Pillow, matplotlib
├── 依赖 Ultralytics YOLO 框架基础
├── ct_modules.py ──→ tasks.py（解析 CT 模块）
├── smc_scheduler.py ──→ trainer.py（训练循环集成）
├── rgb_d_dataset.py ──→ data/dataset.py（继承 YOLODataset）
└── cfg/models/11/*.yaml ──→ nn/tasks.py（模型构建）

0_segment/
├── 依赖 PyTorch 2.2.2 + torchvision
├── 依赖 OpenCV, NumPy, Pillow, matplotlib
├── registry.py ──→ builder.py（动态组网）
├── modules.py ──→ backbones.py, necks.py, heads.py
├── backbones.py ──→ networks.py
├── necks.py ──→ networks.py
├── heads.py ──→ networks.py
├── losses.py ──→ train.py
├── metrics.py ──→ train.py
├── parsers.py ──→ dataset.py
└── dataset.py ──→ train.py

1_study_module/
└── 各模块独立，无跨模块依赖
```

### 跨子项目依赖

- `0_segment` 和 `ultralytics-main` **相互独立**，不共享代码
- 两者共享相同的基础依赖（PyTorch、OpenCV 等）
- 研究代码在 `ultralytics-main/`，工程骨架在 `0_segment/`

---

## 9. 项目运行方式

### 9.1 环境安装

```bash
# 基础环境
pip install torch==2.2.2 torchvision==0.17.2
pip install numpy opencv-python Pillow matplotlib tqdm pyyaml

# 0_segment 专用
cd 1.coding/0_segment
pip install -r requirements.txt
```

### 9.2 Ultralytics-MAIN 训练与测试

#### YOLO 训练

```bash
# 标准 YOLO 训练（使用自定义 RGB-D 配置）
yolo detect train data=201_caomei_data.yaml model=ultralytics/cfg/models/11/yolo11-ct-ABC.yaml epochs=100 imgsz=640

# 使用 SMC 优化器训练
yolo detect train data=201_caomei_data.yaml model=ultralytics/cfg/models/11/yolo11-ct-ABC.yaml \
  epochs=100 imgsz=640 optimizer=SMC smc_surface_patience=10

# 使用自定义优化器脚本
python 301_optimizer_PIDAO.py
```

#### 模型测试

```bash
# 通用测试
python 000_test.py

# YOLO11 测试
python 001_yolo11_test.py

# SMC 测试
python 001_yolo11_SMC_test.py

# Apple RGB-D Amodal 训练
python 006_Apple_Amodal_test.py
```

#### 数据预处理

```bash
# JSON 转 YOLO 格式
python 101_json2yolo.py

# VOC 转 YOLO 格式
python 102_VOC_TO_YOLO.py

# Kvasir 数据集拆分
python 103_kvasir2yolo.py
```

#### 2D 优化基准测试

```bash
# SMC 调度器在 MoG + Rosenbrock 上的测试
python test_smc_scheduler.py
```

### 9.3 0_Segment 训练与预测

```bash
cd 1.coding/0_segment

# 分割训练（FPN 多尺度）
python train.py --model-type fpnseg

# 极简分割训练
python train.py --model-type miniseg

# TS-Dual 多模态训练
python train.py --model-type ts_dual \
  --image-dir /path/to/images \
  --mask-dir /path/to/masks \
  --depth-dir /path/to/depth \
  --prompt-dir /path/to/prompts \
  --imgsz 256 --epochs 50

# 使用配置文件
python train.py --cfg config.json

# 使用合成数据自动验证
python train.py --epochs 10 --batch 8

# 预测
python scripts/predict.py --source path/to/image.jpg --weights runs/train/exp/weights/best.pt
```

### 9.4 学习模块运行

各模块独立，直接运行 `main.py` 或打开 Jupyter Notebook：

```bash
# LeNet
cd 1.coding/1_study_module/2_AlexNet && python main.py

# GoogleNet
cd 1.coding/1_study_module/10_GoogleNet && python main.py

# LSNet 训练
cd 1.coding/1_study_module/16_LSNET && bash train.sh

# U-Net
cd 1.coding/2_Unet && python train.py
```

---

## 10. 开发指南

### 10.1 添加新模块到 0_Segment

#### 添加基础网络层

在 `models/modules.py` 中定义并使用 `@register_block` 装饰器：

```python
@register_block('my_attention')
class MyAttention(nn.Module):
    def __init__(self, in_ch: int, reduction: int = 16):
        super().__init__()
        # ...

    def forward(self, x):
        # ...
```

#### 添加新 Backbone

在 `models/backbones.py` 中添加，注册为 `@register_backbone`：

```python
# 单尺度模式（返回单个 Tensor）
class MyBackbone(nn.Module):
    def forward(self, x) -> torch.Tensor: ...

# 多尺度模式（返回 List[Tensor]，供 neck 使用）
class MyMultiScaleBackbone(nn.Module):
    def forward(self, x) -> List[torch.Tensor]:  # [feat1, feat2, feat3]
        ...
```

#### 添加新 Neck

在 `models/necks.py` 中添加，接口约定：

```python
class MyNeck(nn.Module):
    def forward(self, features: List[Tensor]) -> List[Tensor]:
        # 接收多尺度特征列表，返回同序融合特征列表
        ...
```

#### 添加新 Head

在 `models/heads.py` 中添加：

```python
class MyHead(nn.Module):
    def forward(self, features: List[Tensor]) -> Tuple[List[Tensor], List[Tensor]]:
        # Returns: (cls_outputs, reg_outputs)
        ...
```

#### 组装完整检测器

在 `models/networks.py` 中：

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

#### 添加新损失函数

在 `engine/losses.py` 中添加新 Loss 类即可。

### 10.2 添加新 CT 融合模块到 Ultralytics

1. 在 `ultralytics/nn/modules/ct_modules.py` 中定义新模块类
2. 在 `ultralytics/nn/tasks.py` 的 import 和 `parse_model()` 中注册
3. 在 `ultralytics/cfg/models/11/` 中创建新 YAML 配置
4. 在 YAML 中使用新模块名替换相应层

### 10.3 注册机制流程

```
定义模块类
    │
    ▼
@register_block('name') / @register_backbone('name') / ...
    │
    ▼
写入对应 REGISTRY (Dict[str, Type])
    │
    ▼
配置列表 / 字典 → make_layers(cfg) / build_backbone(cfg)
    │
    ▼
查表获取构造器 → 实例化 → nn.Sequential / 完整网络
```

### 10.4 配置文件格式

**0_Segment 支持 JSON/YAML 配置**:

```json
{
  "model_type": "fpnseg",
  "image_dir": "/path/to/images",
  "mask_dir": "/path/to/masks",
  "imgsz": 128,
  "epochs": 50,
  "batch": 16,
  "lr": 0.001,
  "activation": "silu"
}
```

**Ultralytics 使用 YAML 配置**:

```yaml
# 数据集配置 (201_caomei_data.yaml)
path: /path/to/dataset
train: images/train
val: images/val
names: {0: class1, 1: class2}

# 模型配置 (yolo11-ct-ABC.yaml)
channels: 4
depth_multiple: 0.33
width_multiple: 0.25
backbone:
  - [SplitChannels, [[0,1,2]], 1, null]
  - [KalmanGatedFusion, [-1, -2], 1, null]
  ...
```
