# SMCScheduler 技术文档

> 版本：V3
> 最后更新：2026-06-04
> 文件路径：`ultralytics/nn/modules/smc_scheduler.py`

---

## 1. 研究背景与动机

在深度学习训练中，优化器常陷入**鞍点（saddle point）**和**损失平缓区（plateau）**，表现为梯度范数持续趋近于零、损失停滞不前。传统调度器（Cosine Annealing、ReduceLROnPlateau）仅基于损失曲线或固定时间表调整学习率，缺乏对**梯度动态变化**的实时感知能力。

滑模控制（Sliding Mode Control, SMC）是控制理论中用于处理系统不确定性和外部扰动的经典方法，其核心思想是设计一个**滑模面（sliding surface）**，通过检测系统状态是否偏离滑模面来判断是否需要施加控制输入。

本工作将 SMC 思想移植到深度学习优化器中：**以滑模面为传感器，实时感知梯度动态，仅在系统真正停滞时施加温和扰动**。

### 核心约束

**思路 2 的框架（改 Schedule），实现思路 1 的目的（逃离鞍点/平缓区）**——框架不可更改，即必须通过调度器（而非直接修改优化器算法）来实现逃离机制。

---

## 2. 理论基础

### 2.1 滑模面定义

$$s_t = c \cdot \|g_t\| + (\|g_t\| - \|g_{t-1}\|)$$

| 符号 | 含义 |
|------|------|
| $s_t$ | $t$ 时刻的滑模面值 |
| $c$ | 滑模面系数（默认 0.5），控制"绝对幅值"与"变化率"的权重 |
| $\|g_t\|$ | 第 $t$ 步的全局梯度 L2 范数 |
| $\|g_t\| - \|g_{t-1}\|$ | 梯度范数的变化率（离散导数） |

**物理意义**：滑模面 $s_t$ 同时捕捉梯度的**绝对幅值**和**动态变化**。当梯度既小（$\|g_t\| \to 0$）又静止（$\|g_t\| - \|g_{t-1}\| \to 0$）时，$s_t \to 0$，表示系统在滑模面上停滞——这正是鞍点或平坦区的特征。

**与单纯梯度范数检测的区别**：梯度范数小不一定意味着停滞（可能是稳定收敛中的正常状态），但梯度范数小**且**变化率也小，则几乎可以确定是鞍点或平坦区。滑模面比单一指标更鲁棒。

### 2.2 停滞检测

$$\text{surface\_ratio} = \frac{|s_t|}{s_t^{\text{peak}}}$$

- $s_t^{\text{peak}}$：滑模面绝对值的历史峰值（带衰减）
- 当 `surface_ratio` < `surface_threshold` 持续 `surface_patience` 步 → 判定为停滞 → 触发 escape

### 2.3 Escape 控制策略

Escape 模式下同时施加三个温和控制：

| 控制量 | 正常值 | Escape 值 | 说明 |
|--------|--------|-----------|------|
| 学习率倍率 | $1.0$ | $1.05$ | LR 提升 5%（V3 更温和） |
| β₁ | $0.9$ | $0.88$ | 动量轻微降低，增加对新梯度的敏感度 |
| 梯度噪声 | $0$ | 递减注入 | 最多 10 步，每步衰减 0.9 |

---

## 3. 架构设计

### 3.1 状态机

```
┌─────────┐    surface停滞      ┌──────────┐   surface恢复/超时    ┌──────────┐
│ Warmup  │ ──────────────────→ │  Escape  │ ──────────────────→ │ Cooldown │
│         │   (仅warmup后检测)   │          │                      │          │
└─────────┘                     └──────────┘                      └──────────┘
      │                              │                                  │
      │ warmup结束                   │ noise最多10步                     │ cooldown结束
      ▼                              ▼                                  ▼
┌─────────┐                    ┌──────────┐                       ┌─────────┐
│ Normal  │◄───────────────────│  Normal  │◄──────────────────────│ Normal  │
└─────────┘                    └──────────┘                       └─────────┘
```

四个状态：
- **Warmup**：前 N 步，仅执行 cosine LR 线性预热，SMC 完全休眠
- **Normal**：正常训练，cosine LR 衰减，SMC 监测滑模面
- **Escape**：滑模面停滞触发，注入梯度噪声 + LR/β₁ 微调，最长持续 `escape_max_duration` 步
- **Cooldown**：Escape 结束后的冷却期，禁止新的 Escape 触发

### 3.2 方法调用流程

```
每个训练步：
  ① smc.observe_gradients()     ← optimizer.step() 之前
  ② optimizer.step()
  ③ smc.step(loss_value)        ← optimizer.step() 之后

每个 epoch 结束：
  ④ smc.on_train_epoch_end(epoch_loss)
```

---

## 4. 核心代码详解

### 4.1 滑模面计算

```python
def _compute_sliding_surface(self, grad_norm):
    if self.prev_grad_norm is None:
        s_t = self.c * grad_norm
    else:
        s_t = self.c * grad_norm + (grad_norm - self.prev_grad_norm)
    self.prev_grad_norm = grad_norm
    self.s_t = s_t
    return s_t
```

首步无历史值，退化为 $s_t = c \cdot \|g_t\|$。后续步正常计算。

### 4.2 梯度噪声注入（observe_gradients）

```python
def observe_gradients(self):
    if self.step_count < self.warmup_steps:
        return  # warmup 期间不执行任何 SMC 逻辑

    gn = self._compute_grad_norm()
    s_t = self._compute_sliding_surface(gn)
    self._update_sliding_surface_stats(abs(s_t))

    # 仅在 escape 激活、噪声步数未超限、梯度有效时注入
    if self._in_escape and self._escape_step_counter < self.noise_max_steps and gn > 1e-12:
        # 噪声衰减：随 escape 持续逐步减小，避免后期累积破坏特征
        current_noise_scale = self.noise_scale * (self.noise_decay ** self._escape_step_counter)
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    grad_norm = p.grad.data.norm(2).item()
                    noise_std = current_noise_scale * max(grad_norm, 1e-8)  # 相对噪声
                    noise = torch.randn_like(p.grad.data) * noise_std
                    p.grad.data.add_(noise)
```

**关键设计**：
- **相对噪声**：`noise_std = scale × layer_grad_norm`，对所有层自适应，避免固定绝对噪声对小梯度层破坏过大
- **衰减注入**：每次 escape 最多 10 步噪声，每步衰减为上一步的 0.9 倍（0.001 → 0.0009 → ...），防止噪声累积
- **不修改参数本身**：噪声加在梯度上，由 AdamW 的更新规则自然消化

### 4.3 停滞检测与 Escape 控制（step）

```python
def step(self, loss_value=None):
    self.step_count += 1
    # ... loss 记录 ...

    if self.step_count < self.warmup_steps:
        # warmup: 仅应用 cosine LR
        return

    # peak 衰减（0.9999/step，极其缓慢，避免快速下降导致 ratio 失真）
    self.s_t_peak *= 0.9999

    # 滑模面停滞检测
    if self.s_t_peak > 1e-12 and self.s_t is not None:
        surface_ratio = abs(self.s_t) / self.s_t_peak
        if surface_ratio < self.surface_threshold:
            self._surface_counter += 1
        else:
            self._surface_counter = 0  # 恢复则立即重置

    # 触发条件：纯滑模面停滞 + 不在冷却期
    should_escape = (
        self._surface_counter >= self.surface_patience
        and self._cooldown_counter == 0
    )
```

**V3 关键改动**：触发条件移除了 step 级 loss plateau 的 OR 条件。原因：minibatch 级 loss 波动极大，用于判断 plateau 误触发率极高，escape 效果反而更差。Loss plateau 仅通过 `on_train_epoch_end` 在 epoch 级别检测（宏观、稳定），不参与 escape 触发。

### 4.4 Escape 持续与退出

```python
elif self._in_escape:
    # 退出条件：滑模面恢复 OR 超过最大持续时间
    if self._surface_counter == 0 or self._escape_step_counter >= self.escape_max_duration:
        self._in_escape = False
        self._cooldown_counter = self.escape_cooldown
```

**退出机制**：
- 滑模面恢复（`_surface_counter` 被重置为 0）→ 立即退出
- 超过最大持续步数（默认 20 步）→ 强制退出
- 退出后进入冷却期（默认 100 步），防止反复触发震荡

### 4.5 Epoch 级 Loss 监控

```python
def on_train_epoch_end(self, train_loss):
    """仅记录 epoch 级 loss plateau，不直接触发 escape"""
    if train_loss is not None:
        if self._best_loss is None or train_loss < self._best_loss * 0.999:
            self._best_loss = train_loss
            self._loss_plateau_count = 0
        else:
            self._loss_plateau_count += 1
```

该方法记录 epoch 级 loss plateau，供外部监控和日志使用，但**不参与 escape 触发逻辑**（V3 设计决策）。

---

## 5. 默认超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `c` | 0.5 | 滑模面系数 |
| `surface_threshold` | 0.05 | $|s_t|/s_t^{\text{peak}}$ 低于此值视为停滞 |
| `surface_patience` | 100 | 停滞持续步数后触发 escape |
| `lr_boost` | 1.05 | escape 时 LR 提升倍数 |
| `noise_scale` | 0.001 | 梯度噪声基准标准差（相对） |
| `noise_max_steps` | 10 | 单次 escape 最多注入噪声步数 |
| `noise_decay` | 0.9 | 噪声衰减系数 |
| `escape_cooldown` | 100 | escape 后冷却步数 |
| `escape_max_duration` | 20 | 单次 escape 最长持续步数 |
| `beta1_default` | 0.9 | 正常 β₁ |
| `beta1_low` | 0.88 | escape 时 β₁ |
| `warmup_steps` | 100 | warmup 步数（跳过 SMC） |

---

## 6. YOLO 框架集成

### 6.1 trainer.py 集成点

```python
# 创建（_build_train_pipeline 中）
elif name == "SMC":
    for pg in self.optimizer.param_groups:
        if "initial_lr" not in pg:
            pg["initial_lr"] = pg["lr"]
    self.smc_scheduler = SMCScheduler(self.optimizer, total_steps=iterations, ...)

# 训练循环（_do_train 中）
if self.smc_scheduler is not None:
    self.smc_scheduler.observe_gradients()      # ① step 前：观测梯度
self.optimizer_step()
if self.smc_scheduler is not None:
    self.smc_scheduler.step(self.loss.item())    # ② step 后：控制

# Epoch 结束
if self.smc_scheduler is not None and self.tloss is not None:
    epoch_loss = self.tloss.mean().item()
    self.smc_scheduler.on_train_epoch_end(epoch_loss)  # ③ epoch 级监控
```

### 6.2 使用示例

```python
from ultralytics import YOLO

yolo = YOLO("yolo11-base-rgbd.yaml")
yolo.train(
    data="206_Apple_Amodal.yaml",
    optimizer="SMC",              # 触发 SMCScheduler
    epochs=20,
    imgsz=540,
    batch=4,
    lr0=0.01,
    device=0,
    # SMC 超参数（可选，不传则用 default.yaml 默认值）
    smc_surface_patience=10,
    smc_lr_boost=1.05,
    smc_noise_scale=0.001,
    smc_beta1_low=0.88,
)
```

---

## 7. V1 → V2 → V3 版本演进

### V1（已废弃）

**核心问题**：参数推动 + Adam 状态重置

| 机制 | 问题 |
|------|------|
| 直接移动网络参数 | 破坏已学习的特征表示 |
| 重置 exp_avg / exp_avg_sq | 丢失动量信息，导致重收敛梯度爆炸 |
| 参数克隆/回滚 | OOM 风险，DDP 多卡不同步 |

**结果**：mAP 仅约 20%，训练指标震荡。

### V2

**改进**：移除参数推动和 Adam 状态重置，改为梯度噪声注入。

| 问题 | V2 方案 |
|------|---------|
| 参数破坏 | 仅在梯度上加噪声，不修改参数 |
| Adam 状态丢失 | 保持 exp_avg / exp_avg_sq |
| 噪声量纲 | 相对噪声：`noise_std = scale × layer_grad_norm` |
| Warmup 误触发 | warmup 期间跳过 SMC |
| Peak 冷启动 | 每步衰减 `peak *= 0.999` |

**遗留问题**：`_compute_sliding_surface` 返回的 $s_t$ 被丢弃，实际使用的是 `grad_norm_ema / grad_norm_peak`，**根本不是滑模面**，escape 逻辑沦为错误的启发式规则，极易误触发。

### V3（当前版本）

**核心修复**：

| 问题 | V3 方案 |
|------|---------|
| 滑模面未被使用 | 真正使用 $|s_t| / s_t^{\text{peak}}$ 进行检测 |
| step 级 loss plateau 误触发 | 移除 step 级 OR 条件，loss plateau 仅 epoch 级监控 |
| 噪声持续时间无限制 | 限制单次 escape 最多 10 步噪声注入 |
| 噪声累积 | 衰减注入：每步衰减 0.9 |
| Escape 无冷却 | 新增 cooldown 期（100 步），防止反复触发震荡 |
| 参数过于激进 | LR 1.2x → 1.05x，β₁ 0.85 → 0.88 |
| Peak 衰减过快 | 0.999 → 0.9999 |

---

## 8. 安全性保证

| 保证 | 实现方式 |
|------|---------|
| 不破坏特征表示 | 不修改参数，仅在梯度上加微量噪声 |
| 不丢失优化状态 | 保持 AdamW 的 exp_avg / exp_avg_sq |
| 不干扰正常训练 | warmup 期间完全跳过；cooldown 防震荡 |
| 不因噪声累积而崩溃 | 衰减注入 + 最大步数限制 |
| DDP 多卡兼容 | 噪声仅作用于梯度，各卡独立计算，无需同步 |
| AMP 兼容 | 噪声在 `torch.no_grad()` 外部，不影响 autocast |

---

## 9. 监控指标

通过 `smc.get_stats()` 可获取：

```python
{
    "avg_lr_ratio": 0.98,          # 平均 LR 倍率（<1 表示整体偏保守）
    "noise_injections": 12,        # 累计噪声注入次数
    "escape_events": 3,            # 累计 escape 触发次数
    "surface_ratio": 0.03,         # 当前 |s_t|/s_t_peak（<threshold 表示停滞）
    "s_t_peak": 2.34,             # 当前滑模面峰值
    "in_escape": False,            # 当前是否在 escape 模式
    "cooldown_counter": 45,        # 剩余冷却步数
}
```

训练日志中会出现：
```
[SMC] step=1200: escape triggered (surface_stall=100, ratio=0.0312)
[SMC] step=1220: escape deactivated (duration=20, cooldown=100)
```

---

## 10. 已知限制与后续方向

1. **噪声同种子问题**：DDP 多卡下各卡噪声独立生成，理论上 escape 行为略有差异。实际影响极小（噪声量级 0.001），但严格对齐需固定各卡种子。
2. **超参数敏感性**：`surface_threshold` 和 `surface_patience` 需要根据任务调整，目前仅在 2D 基准和 Apple Amodal 数据集上验证。
3. **与 Cosine LR 的交互**：SMC 的 LR 提升是在 cosine 基础上的乘性调整，训练后期 cosine 值很小时，escape 提升的绝对量也很小，可能不足以逃离深层平坦区。
4. **滑模面系数 $c$ 的自适应**：当前 $c$ 固定为 0.5，理论上可根据训练阶段自适应调整。
