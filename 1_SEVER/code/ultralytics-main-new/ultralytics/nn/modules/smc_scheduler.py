"""
SMCScheduler — 基于滑模控制的 AdamW 学习率调度器 V3

V3 关键修复（解决训练效果差、val loss 震荡、metrics 暴跌问题）：
1. 真正使用滑模面 |s_t| 进行停滞检测
   — V2 中 _compute_sliding_surface 返回的 s_t 被丢弃，实际使用的是 grad_norm_ema/peak，
     根本不是滑模面，导致 escape 逻辑沦为错误的 heuristics，极易误触发。
2. 大幅降低噪声强度并限制单次注入时长，避免破坏已收敛特征
3. 引入 escape cooldown 和最大持续时间，防止持续震荡
4. LR 提升更温和（1.05x 而非 1.2x），β₁ 变化更小（0.88 而非 0.85）
5. 分离 step 级与 epoch 级 loss 检测，避免 minibatch 波动误触发
6. 滑模面峰值衰减更慢（0.9999 而非 0.999）

对神经网络的安全措施（V2/V3 保持一致）：
- 不推动参数（会破坏特征表示）
- 不重置 Adam 状态（会丢失动量信息）
- β₁ 变化极小（保持收敛稳定性）
- 噪声量级很小且限时

工程安全性：底层完全依赖标准 AdamW，通过动态修改 param_groups 实现控制。
"""

import math
import torch


class SMCScheduler:
    """
    滑模控制调度器 V3。

    SMC 核心流程：
    1. 每步计算滑模面 s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)
    2. 跟踪 |s_t| 的峰值 s_t_peak
    3. 当 |s_t| / s_t_peak < threshold 持续 N 步 → escape 模式
    4. Escape = 限时梯度噪声注入 + LR 提升 1.05x + β₁ 轻微降低
    5. Escape 最多持续 M 步，结束后进入 cooldown 期

    Args:
        optimizer: AdamW 优化器
        total_steps: 总步数
        c: 滑模面系数
        surface_threshold: |s_t|/s_t_peak 低于此值视为"在滑模面上停滞"
        surface_patience: 滑模面停滞持续步数
        lr_boost: escape 时 LR 提升倍数
        noise_scale: 梯度噪声标准差（相对梯度范数）
        noise_max_steps: 单次 escape 事件最多注入噪声的步数
        noise_decay: 噪声随 escape 步数的衰减系数
        escape_cooldown: escape 结束后冷却步数，此期间不触发新 escape
        escape_max_duration: 单次 escape 最长持续步数
        beta1_low: escape 时 β₁
    """

    def __init__(
        self,
        optimizer,
        total_steps=10000,
        c=0.5,
        warmup_steps=100,
        min_lr_ratio=0.01,
        surface_threshold=0.05,      # V3: 更严格，0.1→0.05
        surface_patience=100,        # V3: 更保守，50→100
        lr_boost=1.05,               # V3: 更温和，1.2→1.05
        noise_scale=0.001,           # V3: 更低，0.003→0.001
        noise_max_steps=10,          # V3: 新增，单次 escape 最多注入 10 步噪声
        noise_decay=0.9,             # V3: 新增，噪声随 escape 步数衰减
        escape_cooldown=100,         # V3: 新增，escape 后冷却 100 步
        escape_max_duration=20,      # V3: 新增，单次 escape 最长 20 步
        beta1_default=0.9,
        beta1_low=0.88,              # V3: 变化更小，0.85→0.88
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

        # 滑模面状态（V3 真正保存并使用）
        self.s_t: float | None = None
        self.s_t_peak: float = 0.0
        self.prev_grad_norm: float | None = None

        # 控制参数
        self.lr_boost = lr_boost
        self.noise_scale = noise_scale
        self.noise_max_steps = noise_max_steps
        self.noise_decay = noise_decay
        self.escape_cooldown = escape_cooldown
        self.escape_max_duration = escape_max_duration
        self.beta1_default = beta1_default
        self.beta1_low = beta1_low
        self.beta2_default = beta2_default

        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # 状态
        self._last_loss: float | None = None
        self._best_loss: float | None = None
        self._loss_plateau_count: int = 0
        self._in_escape: bool = False
        self._escape_step_counter: int = 0
        self._cooldown_counter: int = 0
        self._surface_counter: int = 0
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
        self.s_t = s_t
        return s_t

    def _update_sliding_surface_stats(self, abs_s_t):
        """更新滑模面峰值"""
        if abs_s_t > self.s_t_peak:
            self.s_t_peak = abs_s_t

    def observe_gradients(self):
        """optimizer.step() 之前：计算滑模面 + 注入噪声（escape 时）"""
        if self.step_count < self.warmup_steps:
            return

        gn = self._compute_grad_norm()
        s_t = self._compute_sliding_surface(gn)
        self._update_sliding_surface_stats(abs(s_t))

        # SMC Escape：注入相对梯度噪声（仅在 escape 激活且未超过 max_steps 时）
        if self._in_escape and self._escape_step_counter < self.noise_max_steps and gn > 1e-12:
            # 噪声随 escape 持续衰减，避免后期噪声累积破坏特征
            current_noise_scale = self.noise_scale * (self.noise_decay ** self._escape_step_counter)
            for pg in self.optimizer.param_groups:
                for p in pg["params"]:
                    if p.grad is not None:
                        grad_norm = p.grad.data.norm(2).item()
                        noise_std = current_noise_scale * max(grad_norm, 1e-8)
                        noise = torch.randn_like(p.grad.data) * noise_std
                        p.grad.data.add_(noise)
            self._noise_count += 1
            self._escape_step_counter += 1

    def step(self, loss_value=None):
        """optimizer.step() 之后"""
        self.step_count += 1

        # 记录 loss 但不用于 step 级 plateau 检测（minibatch 波动太大，误触发率高）
        if loss_value is not None:
            val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            self._last_loss = val

        cos_factor = self._get_cosine_lr(self.step_count)

        # Warmup 期间不做任何 SMC 逻辑
        if self.step_count < self.warmup_steps:
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.initial_lrs[i] * cos_factor
                pg["betas"] = (self.beta1_default, self.beta2_default)
            self.mode = "warmup"
            return

        # 滑模面峰值缓慢衰减（V3: 0.999→0.9999，避免 peak 过快衰减导致 ratio 失真）
        self.s_t_peak *= 0.9999

        # SMC: 滑模面停滞检测 — V3 真正使用 |s_t| / s_t_peak
        surface_ratio = 0.0
        if self.s_t_peak > 1e-12 and self.s_t is not None:
            surface_ratio = abs(self.s_t) / self.s_t_peak
            if surface_ratio < self.surface_threshold:
                self._surface_counter += 1
            else:
                self._surface_counter = 0
        else:
            self._surface_counter = 0

        # cooldown 处理
        if self._cooldown_counter > 0:
            self._cooldown_counter -= 1

        # 触发条件：纯滑模面停滞 + 不在冷却期
        # V3: 移除 step 级 loss plateau 的 OR 条件，避免 minibatch 波动误触发
        #     loss plateau 仅通过 on_train_epoch_end 进行 epoch 级检测（宏观、稳定）
        should_escape = (
            self._surface_counter >= self.surface_patience
            and self._cooldown_counter == 0
        )

        # 激活/停用 escape
        if should_escape and not self._in_escape:
            self._in_escape = True
            self._escape_events += 1
            self._escape_step_counter = 0
            if self.verbose:
                print(f"[SMC] step={self.step_count}: escape triggered "
                      f"(surface_stall={self._surface_counter}, ratio={surface_ratio:.4f})")

        elif self._in_escape:
            # 停用条件：滑模面恢复 或 超过最大持续时间
            if self._surface_counter == 0 or self._escape_step_counter >= self.escape_max_duration:
                self._in_escape = False
                self._cooldown_counter = self.escape_cooldown
                if self.verbose:
                    print(f"[SMC] step={self.step_count}: escape deactivated "
                          f"(duration={self._escape_step_counter}, cooldown={self.escape_cooldown})")

        # 连续控制：escape 时适度调整参数（V3 更温和）
        ctrl = 1.0 if self._in_escape else 0.0
        lr_factor = cos_factor * (1.0 + (self.lr_boost - 1.0) * ctrl)
        b1 = self.beta1_default - (self.beta1_default - self.beta1_low) * ctrl

        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self.initial_lrs[i] * lr_factor
            pg["betas"] = (b1, self.beta2_default)

        self.mode = "escape" if self._in_escape else "normal"
        self._lr_sum += lr_factor

    def on_train_epoch_end(self, train_loss):
        """每个 epoch 结束时调用，用于 epoch 级 plateau 检测（宏观、稳定）"""
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
            "surface_ratio": (abs(self.s_t) / self.s_t_peak
                              if self.s_t is not None and self.s_t_peak > 1e-12 else 0.0),
            "s_t_peak": self.s_t_peak,
            "in_escape": self._in_escape,
            "cooldown_counter": self._cooldown_counter,
        }

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "prev_grad_norm": self.prev_grad_norm,
            "s_t": self.s_t,
            "s_t_peak": self.s_t_peak,
            "initial_lrs": self.initial_lrs,
            "_best_loss": self._best_loss,
            "_loss_plateau_count": self._loss_plateau_count,
            "_in_escape": self._in_escape,
            "_escape_step_counter": self._escape_step_counter,
            "_cooldown_counter": self._cooldown_counter,
            "_surface_counter": self._surface_counter,
        }

    def load_state_dict(self, sd):
        self.step_count = sd["step_count"]
        self.prev_grad_norm = sd.get("prev_grad_norm")
        self.s_t = sd.get("s_t")
        self.s_t_peak = sd.get("s_t_peak", 0.0)
        self.initial_lrs = sd["initial_lrs"]
        self._best_loss = sd.get("_best_loss")
        self._loss_plateau_count = sd.get("_loss_plateau_count", 0)
        self._in_escape = sd.get("_in_escape", False)
        self._escape_step_counter = sd.get("_escape_step_counter", 0)
        self._cooldown_counter = sd.get("_cooldown_counter", 0)
        self._surface_counter = sd.get("_surface_counter", 0)
