# YOLO11 XX-Former RGBD 双主干分割网络

基于 Ultralytics YOLO11 的 RGBD 双主干实例分割模型，采用 XX-Former 架构范式与门控融合机制。

## 整体架构

```
RGBD 输入 (B, 4, 640, 640)
    ├─ SplitChannels([0,1,2]) → RGB 流 (B, 3, 640, 640)
    │   └─ Conv下采样 + C3k2_XXFormer × 4 → P3 / P4 / P5
    │
    ├─ SplitChannels([3])     → Depth 流 (B, 1, 640, 640)
    │   └─ Conv + MaxPool下采样 + 通道对齐 → D3 / D4 / D5
    │
    └─ NeckGateFusion × 3（P3/P4/P5 深度辅助RGB融合）
        └─ PAN-FPN Head（C3k2_XXFormer 精炼）→ Segment 输出
```

**YAML 配置文件**：`ultralytics/cfg/models/11/yolo11-xxformer-rgbd.yaml`

---

## 模块说明

### 1. DualBranchTokenMixer — 双分支 Token Mixer

XX-Former 的空间混合核心，用两条并行卷积分支替代传统单路径或自注意力：

| 分支 | 操作 | 作用 |
|------|------|------|
| 局部分支 | 3×3 Depthwise SepConv → BN → 1×1 Conv → BN | 提取局部空间模式 |
| 全局分支 | 7×7 Depthwise Conv → BN → 1×1 Conv → BN | 获取全局空间上下文（显存友好） |

两路输出逐元素相加融合，无额外参数。

```python
# 输入: (B, dim, H, W)  输出: (B, dim, H, W)
local_out  = local_branch(x)
global_out = global_branch(x)
return local_out + global_out
```

### 2. DualBranchFFN — 双分支前馈网络

XX-Former 的通道精炼层，两条并行分支分别关注空间细节和多尺度上下文：

| 分支 | 操作 | 作用 |
|------|------|------|
| 分支A | 1×1 Conv → BN → GELU → 3×3 DWConv(d=1) → BN → 1×1 Conv → BN | 聚焦空间细节 |
| 分支B | 1×1 Conv → BN → GELU → 3×3 DWConv(d=2) → BN → 1×1 Conv → BN | 多尺度上下文（空洞卷积） |

两路输出逐元素相加融合。隐藏层维度 = `dim × expansion_ratio`（默认2倍）。

```python
# 输入: (B, dim, H, W)  输出: (B, dim, H, W)
return branch_a(x) + branch_b(x)
```

### 3. XXFormerBlock — XX-Former 基本块

将 DualBranchTokenMixer 和 DualBranchFFN 组合成标准 Transformer 风格的残差块：

```
输入 → BN → DualBranchTokenMixer → + 残差
     → BN → DualBranchFFN        → + 残差 → 输出
```

**参数**：
- `dim`：通道数
- `num_heads`：注意力头数（保留，API兼容）
- `attn_ratio`：注意力比例（保留，API兼容）
- `ffn_expansion`：FFN 隐藏层扩展比（默认2.0）

### 4. C3k2_XXFormer — XX-Former CSP 模块

C3k2 的 XX-Former 变体，可直接替换标准 C3k2，内部使用 XXFormerBlock 替代 Bottleneck：

```
输入 → cv1(1×1) → chunk split
      ├─ 路径A → XXFormerBlock × n（逐级串联）
      └─ 路径B（恒等）
      → cat → cv2(1×1) → 输出
```

**参数**：
- `c1 / c2`：输入/输出通道数
- `n`：XXFormerBlock 块数（默认1）
- `e`：通道扩展比（默认0.5）
- `num_heads / attn_ratio / ffn_expansion`：透传给 XXFormerBlock

### 5. NeckGateFusion — 颈部门控融合模块

替代 PAN-FPN 中的简单 Concat 拼接，用通道级门控实现自适应加权融合：

```
输入 [x1, x2]
  → 投影对齐（通道不同时 1×1 Conv + BN）
  → 空间对齐（尺寸不同时双线性插值）
  → Concat → GAP → MLP → Softmax → 权重 [g1, g2]
  → 输出 = g1·x1 + g2·x2
```

**参数**：
- `c1 / c2`：两路输入通道数（不同则自动投影）
- `reduction`：MLP 缩减比（默认16）

在本架构中用于 P3/P4/P5 三个层级的 RGB-Depth 融合（深度辅助RGB）。

### 6. C3k2_Neck — 门控融合 CSP 模块

C3k2 的颈部变体，在 CSP 结构内部使用 NeckGateFusion 替代标准 chunk+cat：

```
输入 → cv1(1×1) → chunk
      ├─ 路径A → Bottleneck × n
      └─ 路径B（恒等）
      → NeckGateFusion([A, B]) → cv2(1×1) → 输出
```

### 7. SplitChannels — 通道分离模块

将 RGBD 4通道输入按索引分离为 RGB(3ch) 和 Depth(1ch) 两路：

```python
# YAML 配置示例
SplitChannels, [[0, 1, 2]]   # → RGB  (B, 3, H, W)
SplitChannels, [[3]]          # → Depth (B, 1, H, W)
```

---

## 文件结构

```
ultralytics-main-2/
├── ultralytics/
│   ├── cfg/models/11/
│   │   └── yolo11-xxformer-rgbd.yaml      # RGBD 双主干 YAML 配置
│   ├── nn/modules/
│   │   ├── block.py                        # DualBranchTokenMixer, DualBranchFFN,
│   │   │                                   # XXFormerBlock, C3k2_XXFormer,
│   │   │                                   # NeckGateFusion, C3k2_Neck
│   │   ├── __init__.py                     # 模块注册
│   │   └── ct_modules.py                   # SplitChannels, BypassModule 等
│   ├── nn/tasks.py                         # 模型解析与模块注册
│   └── engine/model.py                     # ch 参数透传
└── README.md
```

## 使用方法

```python
from ultralytics import YOLO

# 加载 RGBD 双主干模型
model = YOLO("ultralytics/cfg/models/11/yolo11-xxformer-rgbd.yaml")

# 训练（需 RGBD 4通道数据集）
model.train(data="your_dataset.yaml", epochs=300, imgsz=640, batch=4)
```
