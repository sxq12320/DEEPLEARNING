# SMCScheduler 开发记录

> 导出时间：2026-06-04
> 分支：main

---

## 1. 项目背景

在 YOLO11 RGB-D 多模态检测/分割项目中，使用控制理论（卡尔曼滤波、ESO、IDAPBC）改进 RGB-D 特征融合。本次工作是在此基础上，引入**滑模控制（SMC）**思想构建自适应学习率调度器，用于逃离训练中的鞍点/平缓区。

## 2. 核心设计

### 框架约束

**思路 2 的框架（改 Schedule），实现思路 1 的目的（逃离鞍点/平缓区）** — 不可更改。

### SMC 核心机制

```
滑模面:  s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)
         ────   ───────────   ────────────────────
         系数    梯度绝对幅值      梯度变化率
```

- `|s_t|` 持续偏低 → 系统在滑模面上停滞 → 触发 escape
- Escape = 梯度噪声注入 + LR 轻微提升 + β₁ 微调

### 对神经网络的安全措施

| 项目 | 说明 |
|------|------|
| 不推动参数 | 随机推动会破坏已学习的特征表示 |
| 不重置 Adam 状态 | 保持 exp_avg / exp_avg_sq，避免重收敛梯度爆炸 |
| β₁ 变化极小 | 0.9 → 0.85，保持收敛稳定性 |
| 噪声量级小 | 相对噪声：占梯度能量的 0.3% |

## 3. 文件清单

| 文件 | 作用 |
|------|------|
| `ultralytics/nn/modules/smc_scheduler.py` | SMCScheduler 核心实现 |
| `ultralytics/engine/trainer.py` | YOLO 训练循环集成 |
| `ultralytics/cfg/default.yaml` | SMC 超参数默认值 |
| `006_Apple_Amodal_test.py` | Apple RGB-D Amodal 数据集训练脚本 |
| `001_yolo11_SMC_test.py` | SMC 测试脚本（yolo11n-seg） |
| `test_smc_scheduler.py` | 2D 优化基准测试（MoG + Rosenbrock） |

## 4. SMCScheduler 最终版本核心逻辑

### 4.1 观测（observe_gradients）— 在 optimizer.step() 之前调用

```python
def observe_gradients(self):
    if self.step_count < self.warmup_steps:
        return  # warmup 期间不做任何 SMC 逻辑

    gn = self._compute_grad_norm()
    self._update_grad_norm_ema(gn)
    self._compute_sliding_surface(gn)

    # Escape 时：注入相对梯度噪声（能量占梯度能量的 noise_scale）
    if self._in_escape and gn > 1e-12:
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    grad_norm = p.grad.data.norm(2).item()
                    noise_std = self.noise_scale * max(grad_norm, 1e-8)
                    noise = torch.randn_like(p.grad.data) * noise_std
                    p.grad.data.add_(noise)
```

### 4.2 控制（step）— 在 optimizer.step() 之后调用

```python
def step(self, loss_value=None):
    self.step_count += 1
    # ... loss plateau 检测 ...

    # Warmup 保护
    if self.step_count < self.warmup_steps:
        # 只应用 cosine LR，不做 SMC 逻辑
        return

    # peak 缓慢衰减，避免冷启动和 spike 永久抬高
    self.grad_norm_peak *= 0.999

    # Escape 触发（OR 逻辑）
    cond_a = self._surface_counter >= self.surface_patience           # 纯滑模面停滞
    cond_b = (self._loss_plateau_count >= self.surface_patience * 2
              and self._surface_counter >= self.surface_patience // 2)  # loss plateau + 部分滑模面停滞
    should_escape = cond_a or cond_b

    # 应用控制
    ctrl = 1.0 if self._in_escape else 0.0
    lr_factor = cos_factor * (1.0 + (self.lr_boost - 1.0) * ctrl)
    b1 = self.beta1_default - (self.beta1_default - self.beta1_low) * ctrl
```

### 4.3 默认超参数

```yaml
# default.yaml
smc_surface_threshold: 0.1   # |s_t|/peak 低于此值视为停滞
smc_surface_patience: 50     # 停滞持续步数后触发 escape
smc_lr_boost: 1.2            # escape 时 LR 提升倍数
smc_noise_scale: 0.003       # 梯度噪声（相对梯度范数）
smc_beta1_low: 0.85          # escape 时 β₁
```

## 5. YOLO 集成方式

### trainer.py 关键集成点

```python
# 1. 创建 SMCScheduler（在 _build_train_pipeline 中）
elif name == "SMC":
    for pg in self.optimizer.param_groups:
        if "initial_lr" not in pg:
            pg["initial_lr"] = pg["lr"]
    self.smc_scheduler = SMCScheduler(self.optimizer, total_steps=iterations, ...)

# 2. 训练循环（在 _do_train 中）
if self.smc_scheduler is not None:
    self.smc_scheduler.observe_gradients()
self.optimizer_step()
if self.smc_scheduler is not None:
    self.smc_scheduler.step(self.loss.item())

# 3. Epoch 结束回调
if self.smc_scheduler is not None and self.tloss is not None:
    epoch_loss = self.tloss.mean().item()
    self.smc_scheduler.on_train_epoch_end(epoch_loss)
```

### 使用方式

```python
yolo.train(
    optimizer="SMC",          # 触发 SMCScheduler
    smc_surface_patience=10,  # 可选，覆盖默认值
    smc_lr_boost=1.2,
    smc_noise_scale=0.003,
    smc_beta1_low=0.85,
)
```

## 6. 曾遇到的问题及修复

| 问题 | 原因 | 修复 |
|------|------|------|
| `KeyError: 'initial_lr'` | SMC 模式跳过了 LambdaLR，param_groups 缺 initial_lr | 手动添加 `pg["initial_lr"] = pg["lr"]` |
| `RuntimeError: device mismatch` | escape_dir 在 CPU 创建，模型在 GPU | 改为 `torch.randn(total_params, device=device)` |
| `SyntaxError: invalid character` | Unicode 弯引号混入 trainer.py | 替换为 ASCII 直引号 |
| YOLO 训练 mAP 仅 20% | 旧版"参数推动+Adam状态重置"破坏了特征表示 | 重构为梯度噪声注入，不碰参数和 Adam 状态 |
| `AttributeError: on_train_epoch_end` | SMCScheduler 缺少该方法 | 添加 `on_train_epoch_end` 方法 |
| `SyntaxError: 'smc_plateau_patience' is not valid` | 脚本参数名与 default.yaml 不一致 | 统一为 `smc_surface_patience` |
| 梯度噪声量纲问题 | 固定绝对噪声对不同层效果差异极大 | 改为相对噪声：`noise_std = scale × layer_grad_norm` |
| Escape 触发过严 | AND 逻辑要求双条件同时满足 | 改为 OR 逻辑，任一条件满足即可 |
| peak 冷启动/spike 永久抬高 | peak 只增不减 | 每步衰减 `peak *= 0.999` |
| Warmup 期间误触发 | warmup 梯度天然变化大，滑模面计算无意义 | warmup 期间跳过全部 SMC 逻辑 |

## 7. 版本演进

### V1（已废弃）
- 参数推动 + Adam 状态重置 + 参数克隆/回滚
- 结果：OOM 风险、DDP 不兼容、mAP 仅 20%

### V2（当前版本）
- 仅梯度噪声注入 + LR/Beta 微调
- 保持 Adam 状态、参数不变
- Warmup 保护、peak 衰减、相对噪声、OR 触发逻辑
- 工程安全、DDP 兼容、理论自洽

## 8. 待完成事项

- [ ] 在 Apple RGB-D Amodal 数据集上验证 V2 版本效果（运行 `006_Apple_Amodal_test.py`）
- [ ] 更新 `test_smc_scheduler.py` 使用指定 MoG 函数和 4 个起始点
- [ ] SMCScheduler 与 AdamW 的 2D 基准对比（收敛速度、精度、跳出能力）
- [ ] 对照实验：Cosine / Cosine+ReduceLROnPlateau / Cosine+SMC / Cosine+SMC(无噪声)

## 9. 运行命令

```bash
# Apple RGB-D Amodal 训练（SMC）
python 006_Apple_Amodal_test.py

# SMC 基础测试
python 001_yolo11_SMC_test.py

# 2D 优化基准测试
python test_smc_scheduler.py
```
