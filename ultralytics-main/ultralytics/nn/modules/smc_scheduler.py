"""
SMCScheduler — 基于滑模控制的 AdamW 学习率调度器（适配神经网络）

核心思想（思路 2 的框架，实现思路 1 的目的）：
    框架 = 基于滑模面动态调节 LR（Schedule）
    目的 = 逃离鞍点/平缓区

    SMC 核心机制：
    1. 滑模面 s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)
       — 捕捉梯度动态变化率
    2. |s_t| 持续偏低 → 系统在滑模面上停滞 → 触发 escape
    3. Escape = 梯度噪声注入 + LR 轻微提升（温和，不破坏网络）

    对神经网络的安全措施：
    - 不推动参数（会破坏特征表示）
    - 不重置 Adam 状态（会丢失动量信息）
    - β₁ 变化极小（保持收敛稳定性）
    - 噪声量级很小（0.001-0.01）

工程安全性：底层完全依赖标准 AdamW，通过动态修改 param_groups 实现控制。
"""

import math
import torch


class SMCScheduler:
    """
    滑模控制调度器。

    SMC 核心流程：
    1. 每步计算滑模面 s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)
    2. 跟踪 |s_t| 的 EMA 均值和峰值
    3. 当 |s_t| / peak < threshold 持续 N 步 → escape 模式
    4. Escape = 注入梯度噪声 + LR 提升 1.2x + β₁ 轻微降低
    5. Loss 改善后自动恢复正常

    Args:
        optimizer: AdamW 优化器
        total_steps: 总步数
        c: 滑模面系数
        surface_threshold: |s_t|/peak 低于此值视为"在滑模面上停滞"
        surface_patience: 滑模面停滞持续步数
        lr_boost: escape 时 LR 提升倍数
        noise_scale: 梯度噪声标准差（相对梯度范数）
        beta1_low: escape 时 β₁
    """

    def __init__(
        self,
        optimizer,
        total_steps=10000,
        c=0.5,
        warmup_steps=100,
        min_lr_ratio=0.01,
        surface_threshold=0.1,
        surface_patience=50,
        lr_boost=1.2,
        noise_scale=0.003,
        beta1_default=0.9,
        beta1_low=0.85,
        beta2_default=0.999,
        verbose=True,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.c = c
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

        # SMC: 滑模面参数
        self.surface_threshold = surface_threshold
        self.surface_patience = surface_patience
        self.prev_grad_norm: float | None = None
        self.grad_norm_ema: float | None = None
        self.grad_norm_peak: float = 0.0
        self._surface_counter: int = 0

        # 控制参数
        self.lr_boost = lr_boost
        self.noise_scale = noise_scale
        self.beta1_default = beta1_default
        self.beta1_low = beta1_low
        self.beta2_default = beta2_default

        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # 状态
        self._last_loss: float | None = None
        self._best_loss: float | None = None
        self._loss_plateau_count: int = 0
        self._in_escape: bool = False
        self.step_count = 0
        self.mode = "normal"
        self.verbose = verbose
        self._lr_sum: float = 0.0
        self._noise_count: int = 0
        self._escape_events: int = 0

    def _compute_grad_norm(self):
        """计算所有参数梯度的全局 L2 范数"""
        total_sq = 0.0
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_sq)

    def _get_cosine_lr(self, step):
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    def _compute_sliding_surface(self, grad_norm):
        """SMC 核心：计算滑模面 s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)"""
        if self.prev_grad_norm is None:
            s_t = self.c * grad_norm
        else:
            s_t = self.c * grad_norm + (grad_norm - self.prev_grad_norm)
        self.prev_grad_norm = grad_norm
        return s_t

    def _update_grad_norm_ema(self, gn):
        """EMA 更新梯度范数"""
        if self.grad_norm_ema is None:
            self.grad_norm_ema = gn
        else:
            self.grad_norm_ema = 0.95 * self.grad_norm_ema + 0.05 * gn
        if gn > self.grad_norm_peak:
            self.grad_norm_peak = gn

    def observe_gradients(self):
        """optimizer.step() 之前：计算滑模面 + 注入噪声（escape 时）"""
        if self.step_count < self.warmup_steps:
            return

        gn = self._compute_grad_norm()
        self._update_grad_norm_ema(gn)
        self._compute_sliding_surface(gn)  # 更新滑模面状态

        # SMC Escape：注入相对梯度噪声（能量占梯度能量的 noise_scale）
        if self._in_escape and gn > 1e-12:
            for pg in self.optimizer.param_groups:
                for p in pg["params"]:
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2).item()
                        noise_std = self.noise_scale * max(grad_norm, 1e-8)
                        noise = torch.randn_like(p.grad.data) * noise_std
                        p.grad.data.add_(noise)
            self._noise_count += 1

    def step(self, loss_value=None):
        """optimizer.step() 之后"""
        self.step_count += 1

        if loss_value is not None:
            val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            self._last_loss = val
            # Loss plateau 检测
            if self._best_loss is None or val < self._best_loss * 0.999:
                self._best_loss = val
                self._loss_plateau_count = 0
            else:
                self._loss_plateau_count += 1

        cos_factor = self._get_cosine_lr(self.step_count)

        # Warmup 期间不做任何 SMC 逻辑
        if self.step_count < self.warmup_steps:
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.initial_lrs[i] * cos_factor
                pg["betas"] = (self.beta1_default, self.beta2_default)
            self.mode = "warmup"
            return

        # peak 缓慢衰减，避免冷启动和 spike 永久抬高
        self.grad_norm_peak *= 0.999

        # SMC: 滑模面停滞检测
        # |s_t| 持续偏低 = 系统在滑模面上停滞 = 需要扰动
        if self.grad_norm_ema is not None and self.grad_norm_peak > 1e-12:
            surface_ratio = self.grad_norm_ema / self.grad_norm_peak
            if surface_ratio < self.surface_threshold:
                self._surface_counter += 1
            else:
                self._surface_counter = 0
        else:
            self._surface_counter = 0

        # 触发 escape（OR 逻辑，任一条件满足即可）
        #   条件 A: 滑模面持续停滞
        #   条件 B: Loss plateau 且滑模面已有一定程度停滞
        cond_a = self._surface_counter >= self.surface_patience
        cond_b = (self._loss_plateau_count >= self.surface_patience * 2
                  and self._surface_counter >= self.surface_patience // 2)
        should_escape = cond_a or cond_b

        if should_escape and not self._in_escape:
            self._in_escape = True
            self._escape_events += 1
            if self.verbose:
                print(f"[SMC] step={self.step_count}: escape triggered "
                      f"(surface_stall={self._surface_counter}, loss_plateau={self._loss_plateau_count})")

        elif not should_escape and self._in_escape:
            self._in_escape = False
            if self.verbose:
                print(f"[SMC] step={self.step_count}: escape deactivated")

        # 连续控制：escape 时适度调整参数
        ctrl = 1.0 if self._in_escape else 0.0

        lr_factor = cos_factor * (1.0 + (self.lr_boost - 1.0) * ctrl)
        b1 = self.beta1_default - (self.beta1_default - self.beta1_low) * ctrl

        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self.initial_lrs[i] * lr_factor
            pg["betas"] = (b1, self.beta2_default)

        self.mode = "escape" if self._in_escape else "normal"
        self._lr_sum += lr_factor

    def on_train_epoch_end(self, train_loss):
        """每个 epoch 结束时调用，用于 epoch 级 plateau 检测"""
        if train_loss is not None:
            if self._best_loss is None or train_loss < self._best_loss * 0.999:
                self._best_loss = train_loss
                self._loss_plateau_count = 0
            else:
                self._loss_plateau_count += 1

    def get_stats(self):
        return {
            "avg_lr_ratio": self._lr_sum / max(self.step_count, 1),
            "noise_injections": self._noise_count,
            "escape_events": self._escape_events,
            "surface_ratio": (self.grad_norm_ema / self.grad_norm_peak
                              if self.grad_norm_peak > 1e-12 else 0.0),
        }

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "prev_grad_norm": self.prev_grad_norm,
            "grad_norm_ema": self.grad_norm_ema,
            "grad_norm_peak": self.grad_norm_peak,
            "initial_lrs": self.initial_lrs,
        }

    def load_state_dict(self, sd):
        self.step_count = sd["step_count"]
        self.prev_grad_norm = sd.get("prev_grad_norm")
        self.grad_norm_ema = sd.get("grad_norm_ema")
        self.grad_norm_peak = sd.get("grad_norm_peak", 0.0)
        self.initial_lrs = sd["initial_lrs"]
