# TSDualSegDetNet 网络详细解析

**作者**: 代码文学家  
**日期**: 2026-05-18  
**项目路径**: `E:\mastercode\1.coding\0_segment`

---

## 问题 1: Prompt 的数据来源

### 📂 数据加载路径

Prompt 是从 **`prompt_dir`** 目录读取的**单通道掩码图像**（二值掩码）。

### 🔍 代码证据

在 `datasets/dataset.py` 中的 `MultiModalSegmentationDataset` 类：

```python
class MultiModalSegmentationDataset(Dataset):
    def __init__(
        self,
        image_dir=None,        # RGB 图像目录
        label_dir=None,        # 分割标签目录
        depth_dir=None,        # Depth 图像目录
        prompt_dir=None,       # ⭐ Prompt 目录（掩码先验）
        ...
    ):
```

### 📥 读取流程

| 步骤 | 操作 | 代码位置 |
|------|------|----------|
| 1 | 根据文件名 stem 在 `prompt_dir` 中查找文件 | `_read_gray(self.prompt_dir, stem)` |
| 2 | 支持格式：`.png`, `.jpg`, `.npy` 等 | `_read_gray()` 方法 |
| 3 | 读取为灰度图 (单通道) | `cv2.imread(str(candidate), cv2.IMREAD_GRAYSCALE)` |
| 4 | 缩放到 `target_size` | `_resize_if_needed(prompt)` |
| 5 | 转为 Tensor (1, H, W) | `_to_single_channel_tensor()` |

### 📊 Prompt 张量形状

```python
prompt_tensor.shape = (B, 1, H, W)  # 二值掩码，值为 0 或 1
```

### 🔗 Prompt 与 RGB 的融合

在 `backbones.py` 的 `TSDualBackbone` 中：

```python
def forward(self, rgb, prompt=None, depth=None):
    # Prompt 与 RGB 拼接 → 4 通道输入
    rgb_in = torch.cat([rgb, prompt], dim=1)  # (B, 3+1, H, W) = (B, 4, H, W)
    
    # 送入 RGB 分支
    rgb_p2 = self.rgb_stem(rgb_in)  # 输入通道数为 4
```

### ✍️ Prompt 生成方式

Prompt 通常是**人工标注的粗掩码**或**其他模型生成的先验掩码**，用于引导网络关注目标区域。

---

## 问题 2: RGBStem 的详细拆解

### 🏗️ RGBStem 定义

在 `backbones.py` 的 `TSDualBackbone` 类中：

```python
self.rgb_stem = nn.Sequential(
    ConvBNAct(in_ch_rgb, c2, k=3, s=2, activation=activation),
    ConvBNAct(c2, c2, k=3, s=2, activation=activation),
)
```

其中 `in_ch_rgb = 4` (RGB + Prompt 拼接)

### 🔩 ConvBNAct 模块拆解

`ConvBNAct` 是定义在 `models/modules.py` 中的基础模块：

```python
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, d=1, g=1, activation="silu"):
        super().__init__()
        # 1. 卷积层
        self.conv = nn.Conv2d(
            in_ch, out_ch, 
            kernel_size=k, 
            stride=s, 
            padding=autopad(k, p, d),  # 自动计算 padding 保持尺寸
            dilation=d, 
            groups=g, 
            bias=False
        )
        # 2. 批归一化层
        self.bn = nn.BatchNorm2d(out_ch)
        # 3. 激活函数
        self.act = get_activation(activation, ACTIVATION_MAP)
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
```

**所以 `ConvBNAct` = `Conv2d` + `BatchNorm2d` + `Activation`**

### 📐 RGBStem 详细结构表

| 层级 | 操作 | 核大小 (k) | 步幅 (s) | 输入通道 | 输出通道 | 输出尺寸 |
|------|------|------------|-----------|----------|----------|----------|
| Stem[0] | Conv3x3 + BN + SiLU | 3 | 2 | 4 (RGB+Prompt) | c2 | (H/2, W/2) |
| Stem[1] | Conv3x3 + BN + SiLU | 3 | 2 | c2 | c2 | (H/4, W/4) |

**示例**：若 `channels=[32, 64, 128]`，则 `c2=32`

### 🔢 完整数据流

```python
输入: rgb_prompt = (B, 4, 640, 640)
    ↓
Conv3x3(s=2) + BN + SiLU → (B, 32, 320, 320)
    ↓
Conv3x3(s=2) + BN + SiLU → (B, 32, 160, 160)
    ↓
输出: rgb_p2 = (B, 32, 160, 160)
```

### 🎨 RGBStem 结构图

```
输入 (B, 4, H, W)
    ↓
[Conv3x3, s=2] + BN + SiLU
    ↓ (B, c2, H/2, W/2)
[Conv3x3, s=2] + BN + SiLU
    ↓ (B, c2, H/4, W/4)
输出: rgb_p2
```

---

## 问题 3: CrossTokenStatsAttention 详细解析

### ⚠️ 注意：名字有误导性

这个模块名字中的 "Token" 实际上是 **"Statistics"** (统计值)，不是 Transformer 中的 Token。

### 🧠 核心思想

使用**全局平均池化**得到通道维度的统计量，通过 1x1 卷积生成跨模态注意力门控权重，实现 RGB 与 Depth 特征的交互。

### 🔬 代码拆解

在 `models/modules.py` 中：

```python
class CrossTokenStatsAttention(nn.Module):
    def __init__(self, channels, reduction=4, activation="silu"):
        super().__init__()
        hidden = max(1, channels // reduction)
        
        # RGB → Depth 门控生成器
        self.rgb_to_depth = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),  # 降维
            get_activation(activation, ACTIVATION_MAP),             # SiLU
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False), # 升维
            nn.Sigmoid()                                            # 门控权重 (0~1)
        )
        
        # Depth → RGB 门控生成器
        self.depth_to_rgb = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            get_activation(activation, ACTIVATION_MAP),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, rgb_feat, depth_feat):
        # 1. 计算全局统计量 (B, C, 1, 1)
        rgb_stat = rgb_feat.mean(dim=(2, 3), keepdim=True)    # RGB 全局平均
        depth_stat = depth_feat.mean(dim=(2, 3), keepdim=True) # Depth 全局平均
        
        # 2. 跨模态门控生成
        rgb_gate = self.depth_to_rgb(depth_stat)   # 用 Depth 统计生成 RGB 门控
        depth_gate = self.rgb_to_depth(rgb_stat)   # 用 RGB 统计生成 Depth 门控
        
        # 3. 特征交互 (残差形式)
        rgb_out = rgb_feat + rgb_gate * depth_feat  # RGB + 门控加权 Depth
        depth_out = depth_feat + depth_gate * rgb_feat # Depth + 门控加权 RGB
        
        return rgb_out, depth_out
```

### 📊 模块详细拆解表

| 子模块 | 组成 | 输入 | 输出 | 功能 |
|--------|------|------|------|------|
| **统计提取** | `mean(dim=(2,3))` | (B, C, H, W) | (B, C, 1, 1) | 全局平均池化 |
| **RGB→Depth 门控** | Conv1x1(down) + SiLU + Conv1x1(up) + Sigmoid | (B, C, 1, 1) | (B, C, 1, 1) | 生成 Depth 引导的 RGB 门控 |
| **Depth→RGB 门控** | Conv1x1(down) + SiLU + Conv1x1(up) + Sigmoid | (B, C, 1, 1) | (B, C, 1, 1) | 生成 RGB 引导的 Depth 门控 |
| **特征融合** | 逐元素加法 + 乘法 | 两个 (B, C, H, W) | (B, C, H, W) | 残差式跨模态融合 |

### 🎯 流程图

```
输入: rgb_feat (B, C, H, W), depth_feat (B, C, H, W)
    ↓
[全局平均池化]
    ↓
rgb_stat = (B, C, 1, 1)      depth_stat = (B, C, 1, 1)
    ↓                            ↓
[Depth→RGB 门控生成器]          [RGB→Depth 门控生成器]
    ↓                            ↓
rgb_gate = (B, C, 1, 1)      depth_gate = (B, C, 1, 1)
    ↓                            ↓
rgb_out = rgb_feat + rgb_gate * depth_feat
depth_out = depth_feat + depth_gate * rgb_feat
    ↓
输出: (rgb_out, depth_out)
```

### 💡 关键特点

1. **轻量级**：只使用 1x1 卷积，计算量小
2. **全局上下文**：通过全局平均池化捕获通道间关系
3. **双向交互**：RGB ↔ Depth 互相引导
4. **残差连接**：保留原始特征信息

---

## 问题 4: Neck 的融合策略分析

### 🔍 YOLO11Neck 真的是渐进式融合吗？

**答案：不是！YOLO11Neck 使用的是 PAN-FPN（双路径融合），不是渐进式融合。**

### 📐 YOLO11Neck (PAN-FPN) 结构

在 `models/necks.py` 中：

```python
class YOLO11Neck(nn.Module):
    def __init__(self, channels=None, depth_scale=1.0):
        # channels = [c3, c4, c5]  # 输入通道数
        
        # ---- Top-Down 路径 (自顶向下) ----
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        
        # P5 → N4 融合
        self.top_down_conv1 = Conv(c5, c4, 1x1)  # 通道对齐
        self.top_down_c3k2_1 = C3k2(c4+c4, c4)   # 特征融合
        
        # N4 → N3 融合
        self.top_down_conv2 = Conv(c4, c3, 1x1)
        self.top_down_c3k2_2 = C3k2(c3+c3, c3)
        
        # ---- Bottom-Up 路径 (自底向上) ----
        # N3 → N4_out 融合
        self.bottom_up_conv1 = Conv(c3, c4, 3x3, s=2)  # 下采样
        self.bottom_up_c3k2_1 = C3k2(c4+c4, c4)
        
        # N4_out → N5_out 融合
        self.bottom_up_conv2 = Conv(c4, c5, 3x3, s=2)
        self.bottom_up_c3k2_2 = C3k2(c5+c5, c5)
```

### 🚀 PAN-FPN 数据流

```
输入: [P3, P4, P5]  (Backbone 输出)
    ↓
【Top-Down 路径】
P5 → Conv1x1 → Upsample → 与 P4 拼接 → C3k2 → N4
N4 → Conv1x1 → Upsample → 与 P3 拼接 → C3k2 → N3
    ↓
【Bottom-Up 路径】
N3 → Conv3x3(s=2) → 与 N4 拼接 → C3k2 → N4_out
N4_out → Conv3x3(s=2) → 与 P5 拼接 → C3k2 → N5_out
    ↓
输出: [N3, N4_out, N5_out]  (PAN 融合特征)
```

### 🆚 三种 Neck 对比表

| Neck 类型 | 融合策略 | 路径数量 | 代码类 | 适用场景 |
|-----------|----------|----------|--------|----------|
| **YOLO11Neck** | PAN-FPN (自顶向下 + 自底向上) | 2 | `YOLO11Neck` | YOLO 检测 |
| **AFPNNeck** | 渐进式融合 (Progressive) | 1 | `AFPNNeck` | 分割任务 |
| **DyHeadNeck** | 动态注意力融合 | 1 | `DyHeadNeck` | 多任务 |

### 📊 AFPNNeck (渐进式融合) 结构

```python
class AFPNNeck(nn.Module):
    def forward(self, features):
        # 输入: [P2, P3, P4]
        
        # Step 1: 侧向卷积统一通道数
        l2 = self.lateral[0](p2)  # (B, out_ch, H/4, W/4)
        l3 = self.lateral[1](p3)  # (B, out_ch, H/8, W/8)
        l4 = self.lateral[2](p4)  # (B, out_ch, H/16, W/16)
        
        # Step 2: 渐进式融合 (浅层 → 深层)
        p3_up = Upsample(l3) → 与 l2 拼接 → Conv3x3 → f_l1
        p4_up = Upsample(l4) → 与 f_l1 拼接 → Conv3x3 → f_l2
        
        # 输出: f_l2 (单尺度融合特征)
```

### ✅ 结论

1. **YOLO11Neck 不是渐进式融合**，而是 PAN-FPN (双路径融合)
2. **渐进式融合是 AFPNNeck**，它会逐步融合浅层到深层特征
3. 如果你的配置文件中 `neck` 选择 `yolo11_neck`，则使用的是 PAN-FPN

---

## 问题 5: DecoupledSegDetHead 详细解析

### 🎯 核心功能

`DecoupledSegDetHead` 是一个**解耦的检测+分割头**，同时输出边界框和分割掩码。

### 🔬 代码拆解

在 `models/heads.py` 中：

```python
class DecoupledSegDetHead(nn.Module):
    def __init__(self, in_channels=128, mask_out_ch=1, activation="silu", bbox_hidden=128):
        super().__init__()
        
        # ---- BBox 分支 ----
        self.bbox_branch = nn.Sequential(
            ConvBNAct(in_channels, bbox_hidden, k=3, s=1, activation=activation),
            ConvBNAct(bbox_hidden, bbox_hidden, k=3, s=1, activation=activation),
        )
        self.bbox_pool = nn.AdaptiveAvgPool2d(1)  # 全局平均池化 → (B, C, 1, 1)
        self.bbox_fc = nn.Linear(bbox_hidden, 4)     # 全连接层 → 4 个坐标
        
        # ---- Mask 分支 ----
        self.mask_branch = nn.Sequential(
            ConvBNAct(in_channels, in_channels, k=3, s=1, activation=activation),
            ConvBNAct(in_channels, in_channels, k=3, s=1, activation=activation),
            nn.Conv2d(in_channels, mask_out_ch, kernel_size=1),  # 1x1 卷积输出掩码
        )

    def forward(self, features, input_shape=None):
        # ---- BBox 分支 ----
        bbox_feat = self.bbox_branch(features)
        bbox_vec = self.bbox_pool(bbox_feat).flatten(1)  # (B, C)
        bbox_raw = torch.sigmoid(self.bbox_fc(bbox_vec))  # (B, 4) ∈ [0, 1]
        
        # 确保坐标有效性 (x1<x2, y1<y2)
        x1 = torch.min(bbox_raw[:, 0], bbox_raw[:, 2])
        y1 = torch.min(bbox_raw[:, 1], bbox_raw[:, 3])
        x2 = torch.max(bbox_raw[:, 0], bbox_raw[:, 2])
        y2 = torch.max(bbox_raw[:, 1], bbox_raw[:, 3])
        bbox_pred = torch.stack([x1, y1, x2, y2], dim=1)  # (B, 4)
        
        # ---- Mask 分支 ----
        mask_logits = self.mask_branch(features)  # (B, C, H, W)
        if input_shape is not None:
            mask_logits = F.interpolate(mask_logits, size=input_shape, mode="bilinear")
        
        return bbox_pred, mask_logits
```

### 📊 模块详细拆解表

| 分支 | 子模块 | 操作 | 输出形状 | 功能 |
|------|--------|------|----------|------|
| **BBox 分支** | `bbox_branch` | 2x(Conv3x3+BN+SiLU) | (B, 128, H, W) | 提取定位特征 |
| | `bbox_pool` | AdaptiveAvgPool2d(1) | (B, 128, 1, 1) | 全局上下文 |
| | `bbox_fc` | Linear(128, 4) + Sigmoid | (B, 4) | 预测归一化坐标 |
| | 后处理 | min/max 确保有效性 | (B, 4) | xyxy 格式框 |
| **Mask 分支** | `mask_branch` | 2x(Conv3x3+BN+SiLU) | (B, 128, H, W) | 提取分割特征 |
| | 输出层 | Conv1x1 | (B, C, H, W) | 分割 logits |

### 🎨 结构图

```
输入: features (B, 128, H, W)
    ↓
┌─────────────────────┬─────────────────────┐
│   BBox 分支         │   Mask 分支         │
│                    │                     │
│ [Conv3x3+BN+SiLU]  │  [Conv3x3+BN+SiLU] │
│  (128, 128)        │   (128, 128)        │
│        ↓            │         ↓            │
│ [Conv3x3+BN+SiLU]  │  [Conv3x3+BN+SiLU] │
│  (128, 128)        │   (128, 128)        │
│        ↓            │         ↓            │
│ [AdaptiveAvgPool]   │  [Conv1x1]          │
│  (128, 1, 1)      │   (C, H, W)         │
│        ↓            │                     │
│ [Linear(128,4)]    │  [Upsample]         │
│  (4,)              │   (C, H_orig, W_orig)│
│        ↓            │                     │
│ [Sigmoid + min/max]│                     │
│  (4,)              │                     │
└─────────────────────┴─────────────────────┘
    ↓                            ↓
bbox_pred (B, 4)           mask_logits (B, C, H, W)
```

### 🔢 输出详解

1. **bbox_pred**: (B, 4)，归一化坐标 [0, 1]
   - 格式: `[x1, y1, x2, y2]` (左上角 + 右下角)
   
2. **mask_logits**: (B, C, H, W)，分割 logits
   - C = `mask_out_ch` (通常为 1 或 num_classes)
   - 需要经过 Sigmoid 转为概率

---

## 附录：完整模块汇总表格

### 📊 骨干网络模块表

| 模块名称 | 类名 | 输入 | 输出 | 核心组件 | 适用任务 |
|----------|------|------|------|----------|----------|
| **标准 Stem** | `ConvBNAct × 2` | (B, 3, H, W) | (B, C, H/4, W/4) | Conv3x3(s=2)+BN+SiLU | 单模态 |
| **RGB Stem** | `ConvBNAct × 2` | (B, 4, H, W) | (B, C, H/4, W/4) | Conv3x3(s=2)+BN+SiLU | 双模态 (RGB+Prompt) |
| **Depth Stem** | `ConvBNAct × 2` | (B, 1, H, W) | (B, C, H/4, W/4) | Conv3x3(s=2)+BN+SiLU | 双模态 |
| **Stage 模块** | `BasicBlock × n` | (B, C, H/s, W/s) | (B, 2C, H/2s, W/2s) | [Conv3x3+BN+SiLU]×2 + Residual | 特征提取 |
| **SPPF** | `SPPF` | (B, C, H, W) | (B, C, H, W) | MaxPool5x5 × 3 + Concat + Conv1x1 | YOLO 系列 |

### 📊 注意力机制模块表

| 模块名称 | 类名 | 输入 | 输出 | 计算复杂度 | 功能 |
|----------|------|------|------|------------|------|
| **CrossTokenStats-Attention** | `CrossTokenStatsAttention` | (B,C,H,W)×2 | (B,C,H,W)×2 | O(C²) 很低 | 跨模态统计注意力 |
| **ScaleAware-Attention** | `ScaleAwareAttention` | (B,C,H,W) | (B,C,H,W) | O(C²) | 尺度感知注意力 |
| **SpatialAware-Attention** | `SpatialAwareAttention` | (B,C,H,W) | (B,C,H,W) | O(HW) | 空间感知注意力 |
| **TaskAware-Attention** | `TaskAwareAttention` | (B,C,H,W) | (B,C,H,W) | O(C) | 任务感知注意力 |

### 📊 颈部网络模块表

| 模块名称 | 类名 | 输入 | 输出 | 融合策略 | 参数量 |
|----------|------|------|------|----------|--------|
| **YOLO11Neck** | `YOLO11Neck` | [P3,P4,P5] | [N3,N4,N5] | PAN-FPN (双路径) | 中等 |
| **AFPNNeck** | `AFPNNeck` | [P2,P3,P4] | f_l2 | 渐进式融合 (单路径) | 较低 |
| **DyHeadNeck** | `DyHeadNeck` | (B,C,H,W) | Dict | 动态注意力融合 | 较高 |

### 📊 检测头模块表

| 模块名称 | 类名 | 输入 | 输出 | 解耦方式 | 适用任务 |
|----------|------|------|------|----------|----------|
| **YOLO11Head** | `YOLO11Head` | [N3,N4,N5] | cls+reg | 分类/回归分支分离 | 检测 |
| **DecoupledSegDetHead** | `DecoupledSegDetHead` | features | bbox+mask | BBox/Mask 分支分离 | 分割+检测 |

### 📊 损失函数模块表

| 损失组件 | 类名 | 输入 | 输出 | 公式/方法 | 权重 |
|----------|------|------|------|-----------|------|
| **分割损失** | `SegmentationLoss` | pred, target | scalar | BCE or CE | 1.0 |
| **频域损失** | `FourierLoss` | pred, target | scalar | FFT2D 幅值差异 | 0.2 |
| **框损失** | `NWDLoss` | pred_box, target_box | scalar | NWD (Wasserstein) | 1.0 |
| **CIoU 损失** | `ciou_loss()` | pred_box, target_box | scalar | Complete IoU | 7.5 (检测) |
| **DFL 损失** | `distribution_focal_loss()` | pred_dist, target | scalar | Distribution Focal Loss | 1.5 (检测) |

### 📊 优化器配置表

| 参数 | 值 | 说明 |
|------|------|------|
| **优化器** | `Adam` | 自适应学习率优化器 |
| **学习率 (lr)** | `1e-3` (默认) | 可通过 `--lr` 参数修改 |
| **权重衰减** | 0 (默认) | Adam 默认不使用权重衰减 |
| **betas** | (0.9, 0.999) | Adam 一阶、二阶动量衰减率 |
| **eps** | `1e-8` | 数值稳定项 |

---

## 总结

本文档详细解析了 `TSDualSegDetNet` 网络的 6 个核心问题：

1. ✅ **Prompt 来源**: 从 `prompt_dir` 读取的单通道掩码图像
2. ✅ **RGBStem 拆解**: 2 个 `ConvBNAct` (Conv+BN+SiLU)
3. ✅ **CrossTokenStatsAttention**: 跨模态统计注意力 (轻量级)
4. ✅ **Neck 融合策略**: YOLO11Neck 使用 PAN-FPN，不是渐进式融合
5. ✅ **DecoupledSegDetHead**: 解耦的检测+分割头
6. ✅ **损失函数和优化器**: SegDetLoss (BCE+Fourier+NWD) + Adam

**文档路径**: `E:\mastercode\TSDualSegDetNet_详细解析.md`

---

## 问题 6: 损失函数和优化器

### 📉 损失函数

在 `engine/losses.py` 中，`TSDualSegDetNet` 使用的是 `SegDetLoss`：

```python
class SegDetLoss(nn.Module):
    def __init__(
        self,
        mask_loss="bce",        # 分割损失类型
        mask_weight=1.0,        # 分割损失权重
        fourier_weight=0.2,     # Fourier 损失权重
        bbox_weight=1.0,         # NWD 损失权重
        nwd_constant=20.0,      # NWD 归一化常数
    ):
        self.mask_loss = SegmentationLoss(mask_loss)  # BCE or CrossEntropy
        self.fourier_loss = FourierLoss()              # 频域损失
        self.bbox_loss = NWDLoss(constant=nwd_constant) # 框损失

    def forward(self, mask_logits, mask_targets, bbox_pred, bbox_targets, bbox_valid=None):
        # 1. 分割损失
        mask_loss_val = self.mask_loss(mask_logits, mask_targets)
        
        # 2. Fourier 频域损失
        prob = torch.sigmoid(mask_logits)
        fourier_loss_val = self.fourier_loss(prob, mask_targets.float())
        
        # 3. NWD 框损失
        bbox_loss_val = self.bbox_loss(bbox_pred, bbox_targets, bbox_valid)
        
        # 总损失 = 加权求和
        total = (
            self.mask_weight * mask_loss_val +
            self.fourier_weight * fourier_loss_val +
            self.bbox_weight * bbox_loss_val
        )
        
        return total, {"mask": mask_loss_val, "fourier": fourier_loss_val, "bbox": bbox_loss_val}
```

### 📊 损失函数详细表

| 损失组件 | 类名 | 公式/方法 | 用途 | 权重 |
|----------|------|-----------|------|------|
| **分割损失** | `SegmentationLoss` | BCEWithLogitsLoss 或 CrossEntropyLoss | 分割掩码准确性 | `mask_weight=1.0` |
| **频域损失** | `FourierLoss` | FFT2D 幅值差异 | 形状一致性约束 | `fourier_weight=0.2` |
| **框损失** | `NWDLoss` | Normalized Wasserstein Distance | 边界框回归 | `bbox_weight=1.0` |

### 🔬 NWD 损失详解

```python
class NWDLoss(nn.Module):
    def forward(self, pred_boxes, target_boxes, valid_mask=None):
        # 1. 计算中心点距离
        pred_cx = (pred[:, 0] + pred[:, 2]) / 2.0
        pred_cy = (pred[:, 1] + pred[:, 3]) / 2.0
        center_dist = (pred_cx - tgt_cx)**2 + (pred_cy - tgt_cy)**2
        
        # 2. 计算尺度距离
        size_dist = ((pred_w - tgt_w)**2 + (pred_h - tgt_h)**2) / 4.0
        
        # 3. Wasserstein 距离
        wasserstein = sqrt(center_dist + size_dist)
        
        # 4. 映射为损失
        nwd = exp(-wasserstein / constant)
        loss = 1.0 - nwd
        
        return loss.mean()
```

### ⚙️ 优化器

在 `train.py` 中的**第 6 部分：训练主流程**中定义：

```python
# train.py 第 301 行
optimizer = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
```

**优化器类型**: `torch.optim.Adam`

**默认学习率**: `lr=1e-3` (可在配置中修改)

### 📊 优化器详细表

| 参数 | 值 | 说明 |
|------|------|------|
| **优化器** | `Adam` | 自适应学习率优化器 |
| **学习率 (lr)** | `1e-3` (默认) | 可通过 `--lr` 参数修改 |
| **权重衰减** | 0 (默认) | Adam 默认不使用权重衰减 |
| **betas** | (0.9, 0.999) | Adam 一阶、二阶动量衰减率 |
| **eps** | `1e-8` | 数值稳定项 |

### 🔧 如何修改优化器

如果你想使用其他优化器（如 SGD、AdamW），需要修改 `train.py`：

```python
# 改为 SGD
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=cfg["lr"],
    momentum=0.9,
    weight_decay=5e-4
)

# 改为 AdamW
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=cfg["lr"],
    weight_decay=1e-4
)
```

### 📹 完整训练配置表

| 超参数 | 默认值 | 命令行参数 | 说明 |
|--------|----------|------------|------|
| **epochs** | 20 | `--epochs` | 训练轮数 |
| **batch_size** | 8 | `--batch` | 批次大小 |
| **learning_rate** | 1e-3 | `--lr` | 学习率 |
| **optimizer** | Adam | (固定) | 优化器类型 |
| **image_size** | 128 | `--imgsz` | 输入图像尺寸 |
| **augment** | True | `--augment/--no-augment` | 数据增强 |
| **seed** | 22 | `--seed` | 随机种子 |
| **workers** | 0 | `--workers` | DataLoader 线程数 |