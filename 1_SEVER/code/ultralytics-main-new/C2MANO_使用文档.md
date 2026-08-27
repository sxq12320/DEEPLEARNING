# C2MANO 使用文档 —— MANO 多极注意力即插即用模块

> 模块文件：`ultralytics/nn/modules/mano.py`
> 示例配置：`0_orange_yaml/004_yolo11-seg-mano.yaml`
> 来源论文：*Linear Attention with Global Context: A Multipole Attention Mechanism for Vision and Physics*（MANO，ICCV-W 2025，arXiv:2507.02748）

---

## 1. 概述

**MANO（Multipole Attention Neural Operator）** 是一种受快速多极子方法（FMM）启发的注意力机制：把自注意力看作粒子间的多尺度交互，在**每个 head 内同时保持全局感受野**，且时间/内存复杂度对网格点数**线性**。

本仓库把它适配成一个符合 Ultralytics 契约的即插即用块 **`C2MANO`**：

- **NCHW 进出、通道不变、空间尺寸不变** —— 和 `C2PSA` 完全相同的契约，YAML 里能像 C2PSA 一样直接用。
- **纯 PyTorch**，无 einops / 无额外依赖。
- **对任意特征图尺寸鲁棒**（80/40/20 等非 2 的幂都能跑）。

> 本项目用途：替换 YOLO11 backbone 末端的 `C2PSA`，做「注意力机制」的单变量消融（E0=C2PSA vs E1=C2MANO）。

---

## 2. 模块架构

四层结构，自底向上封装：

```
C2MANO                         ← YAML 暴露的入口（保持通道，assert c1==c2）
└── n × MANOBlock              ← 注意力 + FFN，各带残差（类比 PSABlock）
        ├── MultipoleAttention ← 核心：多尺度金字塔 + 共享窗口注意力 + V-cycle 聚合
        │       └── _WindowMHSA← 窗口内标准多头自注意力
        └── FFN                ← Conv1×1 → 2c → Conv1×1
```

### 2.1 核心算法（MultipoleAttention）

```
输入 X [B,C,H,W]
  │
  ├─ 建金字塔：X₀=X, X₁=down(X₀), X₂=down(X₁), ...   （down = 共享卷积，逐级下采样）
  │
  ├─ 每一级都用【同一个】窗口注意力：Oₗ = WindowAttn(Xₗ)   （权重跨尺度共享）
  │
  └─ V-cycle 自粗到细聚合：
        res = O_粗
        res = O_次粗 + (1/1)·up(res)
        res = O_次次粗 + (1/2)·up(res)
        ...                                            （up = 共享转置卷积）
输出 res [B,C,H,W]   （通道、尺寸均不变）
```

- **共享注意力 + 共享采样核** → 参数量与金字塔层数 **无关**，非常省参。
- **总复杂度由最细尺度主导**，粗尺度开销可忽略 → 近似线性、却有全局感受野。

### 2.2 三个鲁棒性处理（相对论文参考实现的适配）

| 问题 | 参考实现 | 本实现的解决 |
|---|---|---|
| ① 张量布局 | NHWC | 全程 NCHW，直接对接 YOLO |
| ② 层数写死 | `levels = int(log(image_size, rate))` 在 `__init__` 固定 | **每次 forward 按实际 H,W 动态算** `levels`（`_num_levels`） |
| ③ 非 2 的幂尺寸 | conv 下采样，20→10→5→2.5 崩溃 | 下采样前 **pad 到整除**；上采样后用 `F.interpolate` **对齐到精确尺寸**；窗口注意力内部 pad 到窗口整除、算完裁回 |

---

## 3. 输入 / 输出

| 项 | 规格 |
|---|---|
| **输入** | `[B, C, H, W]`（NCHW，float） |
| **输出** | `[B, C, H, W]` —— **通道、尺寸完全不变** |
| **硬约束** | `C2MANO` 要求 `c1 == c2`（保持通道，故只能插在不改变通道的位置） |
| **对 H,W 要求** | 无（内部自动 pad / 对齐，任意尺寸可跑；已验证 640/512/608） |

---

## 4. 参数详解

### 4.1 `C2MANO`（YAML 入口）

```python
C2MANO(c1, c2, n=1, e=0.5, window_size=4, num_heads=None, sampling_rate=2, max_levels=3)
```

| 参数 | 类型 | 默认 | 说明 | 由谁提供 |
|---|---|---|---|---|
| `c1` | int | — | 输入通道 | **parse_model 自动填**（=上一层通道） |
| `c2` | int | — | 输出通道（须 = c1） | **parse_model 自动填**（=YAML 首参经 width 缩放） |
| `n` | int | 1 | 堆叠的 MANOBlock 数 | **parse_model 自动填**（=YAML repeats × depth） |
| `e` | float | 0.5 | 隐藏通道比例，内部 `c = int(c1*e)` | YAML 可选 |
| `window_size` | int | 4 | 非重叠窗口边长 | YAML 可选 |
| `num_heads` | int/None | None | 注意力头数，`None`→`max(c//32,1)` 并自动保证整除 | YAML 可选 |
| `sampling_rate` | int | 2 | 每级下/上采样因子 | YAML 可选 |
| `max_levels` | int | 3 | 金字塔最大层数（实际按尺寸自适应，不超过它） | YAML 可选 |

> ⚠️ `C2MANO` **没有**透传 `downsample`（采样方式）和 `shortcut`（残差开关）。如需改见 §8.2。

### 4.2 `MultipoleAttention`（核心，供直接调用）

```python
MultipoleAttention(dim, window_size=4, num_heads=None, sampling_rate=2, max_levels=3, downsample="conv")
```

- `downsample`：`"conv"`（可学习 Conv/ConvTranspose，忠于论文，默认）或 `"avg"`（AvgPool/最近邻上采样，**无参数、无 ConvTranspose**）。

### 4.3 关键取值建议

- **`window_size` 要配合特征图尺寸**：要达到 `max_levels` 层，需保证每级下采样后仍 ≥ 一个窗口。例如 P5=20×20、`sampling_rate=2`：
  - `window_size=4` → 20→10→5，可达 **3 级**（默认，小特征图友好）✅
  - `window_size=8` → 20→10（10<8 后停），只有 **2 级**
  - 插到 P3=80×80 这种大特征图时，`window_size=8` 也能多级，可适当调大。
- **`num_heads`** 留 `None` 即可（自动按通道选，且保证整除）。
- **`e`** 越小越省算力（默认 0.5，与 C2PSA 一致）。

---

## 5. YAML 写法

### 5.1 基本用法（等价替换 C2PSA）

```yaml
# backbone 末端，把原来的 [-1, 2, C2PSA, [1024]] 换成：
- [-1, 2, C2MANO, [1024]] # 全部用默认参数
```

`parse_model` 会自动：
1. `c1 = 上一层输出通道`；
2. `c2 = make_divisible(min(1024, max_channels) × width, 8)`（n 尺度 width=0.25 → **256**）；
3. 插入 repeats：`n = max(round(2 × depth), 1)`（n 尺度 depth=0.5 → **1**）。

→ 实际实例化为 `C2MANO(c1=256, c2=256, n=1)`，与 baseline C2PSA 完全对齐。

### 5.2 传可选参数（重要：顺序规则）

YAML 方括号里第 1 个数是 `c2`（会被 width 缩放），**之后的数按 `C2MANO` 签名顺序依次对应** `e, window_size, num_heads, sampling_rate, max_levels`：

```yaml
- [-1, 2, C2MANO, [1024]]                 # 全默认
- [-1, 2, C2MANO, [1024, 0.5]]            # e=0.5
- [-1, 2, C2MANO, [1024, 0.5, 8]]         # e=0.5, window_size=8
- [-1, 2, C2MANO, [1024, 0.5, 8, 4]]      # 再加 num_heads=4
- [-1, 2, C2MANO, [1024, 0.5, 4, null, 2, 3]]  # 全指定（num_heads 用 null=自动）
```

> 规则记忆：`[c2, e, window_size, num_heads, sampling_rate, max_levels]`，从左到右能写几个写几个，后面的用默认。

### 5.3 完整片段示例（`004_yolo11-seg-mano.yaml` 的关键行）

```yaml
nc: 1
scales:
  n: [0.50, 0.25, 1024]
backbone:
  # ... 0-9 层与 yolo11-seg 完全相同 ...
  - [-1, 1, SPPF, [1024, 5]]   # 9
  - [-1, 2, C2MANO, [1024]]    # 10  ← 这里替换了 C2PSA
head:
  # ... 与 yolo11-seg 完全相同 ...
  - [[16, 19, 22], 1, Segment, [nc, 32, 256]]
```

---

## 6. 使用示例

### 6.1 训练（本地 Windows，主 fork）

```bash
cd ultralytics-main-new
pip install -e .          # 首次或包结构变更后；editable 模式下改源码免重装

# E1：MANO 变体（从 yolo11n-seg.pt 迁移可匹配层，C2MANO 层从头学）
python train_citrus_seg.py --model 0_orange_yaml/004_yolo11-seg-mano.yaml \
    --pretrained yolo11n-seg.pt --name E1_mano --data 200orange_wuxi_seg.yaml
```

> ⚠️ 记得带 `--data`：脚本默认路径 `data/test/` 已废弃，用 `200orange_wuxi_seg.yaml`（指向 `data/orange_yolo`）。

### 6.2 CLI（不依赖驱动脚本，服务器友好）

```bash
yolo segment train model=0_orange_yaml/004_yolo11-seg-mano.yaml \
    data=<你的数据yaml> optimizer=AdamW epochs=300 imgsz=640 batch=4 device=0
```

### 6.3 Python 直接构建 / 单独调用模块

```python
import torch
from ultralytics import YOLO
from ultralytics.nn.modules import C2MANO, MultipoleAttention

# 整网
m = YOLO("0_orange_yaml/004_yolo11-seg-mano.yaml")
m.info()

# 单独用这个块（保持通道）
blk = C2MANO(256, 256, n=1)
y = blk(torch.randn(1, 256, 20, 20))   # -> [1, 256, 20, 20]

# 或只用核心多极注意力
attn = MultipoleAttention(128, window_size=4, max_levels=3)
y2 = attn(torch.randn(1, 128, 80, 80)) # -> [1, 128, 80, 80]
```

### 6.4 插到其它位置

因为保持通道，`C2MANO` 可插在任何「不改变通道」的位置，例如 neck 的某个 C3k2 之后追加一层：

```yaml
- [-1, 2, C3k2, [256, False]]  # 16 (P3/8-small)
- [-1, 1, C2MANO, [256]]       # 追加：对 P3 做多极注意力（通道不变）
```

> 注意插到 neck 会改变后续层索引，需同步调整 head 里 `Concat`/`Segment` 的引用层号。

---

## 7. 已验证结果

| 模型 | 层数 | 参数量 | GFLOPs | 验证项 |
|---|---|---|---|---|
| 001 baseline (C2PSA) | 204 | 2,876,848 | 10.5 | — |
| **004 (C2MANO)** | 200 | **2,988,851** | **11.2** | build + 前向(640/512/608) + 反向 ✅ |

主 fork 与 1_SEVER 副本均构建通过，参数量一致（+3.9%，与 baseline 同量级）。

---

## 8. 注意事项与调参

### 8.1 确定性训练（deterministic）

`downsample="conv"`（默认）用到 `ConvTranspose2d`。`train_citrus_seg.py` 开了 `deterministic=True`：
- Ultralytics 通常以 `warn_only` 方式启用确定性算法，**只警告不报错**，可正常训练。
- 若在你的环境真报「ConvTranspose 无确定性实现」错误，改用无参数采样 `downsample="avg"`（见 §8.2）。

### 8.2 如何暴露 / 切换 `downsample`

当前 `C2MANO` 未把 `downsample` 透传到 YAML。两种改法（任选其一）：

**（A）最省事：改 `MultipoleAttention` 默认**（`mano.py`）
```python
def __init__(self, dim, window_size=4, num_heads=None, sampling_rate=2, max_levels=3, downsample="avg"):
```

**（B）透传到 YAML：给 C2MANO/MANOBlock 加参数**
```python
# C2MANO.__init__ 末尾加 downsample="conv"，并传给 MANOBlock；
# MANOBlock.__init__ 加 downsample="conv"，传给 MultipoleAttention。
# 之后 YAML: [-1, 2, C2MANO, [1024, 0.5, 4, null, 2, 3, "avg"]]
```

### 8.3 4 文件注册机制（已完成，供迁移/排错参考）

新增自定义模块要走这 4 步（本模块已全部完成）：
1. **实现**：`ultralytics/nn/modules/mano.py`
2. **导出**：`ultralytics/nn/modules/__init__.py` —— import + 加入 `__all__`
3. **导入**：`ultralytics/nn/tasks.py` 顶部 import `C2MANO`
4. **注册**：`tasks.py` 的 `parse_model()` 里，把 `C2MANO` 加进 `base_modules` 和 `repeat_modules` 两个 frozenset（复用 C2PSA 通道契约，无需专门 `elif` 分支）

> 漏掉第 3/4 步会报「YAML 解析错误」而不是清晰的「未知模块」。

### 8.4 调参速查

| 想要 | 调什么 |
|---|---|
| 更省算力 | `e` 调小（如 0.25） |
| 更大感受野/更多层次 | 插到大特征图（P3/P4）并把 `window_size` 调大 |
| 小特征图（P5）保证多级 | `window_size` 保持 4（默认） |
| 免除 deterministic 隐患 | `downsample="avg"`（§8.2） |
| 更深的注意力 | YAML repeats 调大（如 `[-1, 3, C2MANO, [1024]]`） |

---

## 9. 参数速查表（一页纸）

```
C2MANO(c1, c2, n=1, e=0.5, window_size=4, num_heads=None, sampling_rate=2, max_levels=3)
        └c1=c2 必须相等；输入输出 [B,C,H,W] 通道尺寸都不变

YAML:  [-1, n, C2MANO, [c2, e, window_size, num_heads, sampling_rate, max_levels]]
                        └──── 只有 c2 必填，其余从左往右可选 ────┘

默认行为：window=4 / heads=自动 / rate=2 / 最多3级金字塔 / conv 采样
适用位置：任何 c_in==c_out 的地方（默认替换 backbone 第10层 C2PSA）
```
