"""
SMCScheduler — 基于滑模控制 (Sliding Mode Control) 的 AdamW 学习率调度器

核心思想（思路 2 的框架，实现思路 1 的目的）：
    框架 = 基于滑模面动态调节 LR（Schedule）
    目的 = 逃离鞍点/平缓区/局部最优

    逃离机制（多方向探索 + loss 选择）：
    1. 检测 plateau → 触发逃离
    2. 尝试多个随机方向，每个方向推动 N 步
    3. 每个方向结束后检查 loss：若 loss 恢化则回退
    4. 保留 loss 改善最大的方向（可能收敛到更优 basin）

工程安全性：底层完全依赖标准 AdamW，通过动态修改 param_groups + state 实现控制。
"""

import math
import torch
from collections import deque


class SMCScheduler:
    def __init__(
        self,
        optimizer,
        total_steps=10000,
        warmup_steps=100,
        min_lr_ratio=0.01,
        plateau_threshold=0.15,
        plateau_patience=5,
        lr_boost=5.0,
        escape_duration=20,
        escape_push=0.05,
        max_escape_trials=8,
        beta1_default=0.9,
        beta1_low=0.1,
        beta2_default=0.999,
        beta2_low=0.9,
        momentum_reset_interval=5,
        verbose=True,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio
        self.plateau_threshold = plateau_threshold
        self.plateau_patience = plateau_patience
        self.grad_norm_history: deque[float] = deque(maxlen=300)
        self.grad_peak: float = 0.0
        self._plateau_counter: int = 0
        self.lr_boost = lr_boost
        self.escape_duration = escape_duration
        self.escape_push = escape_push
        self.max_escape_trials = max_escape_trials
        self.momentum_reset_interval = momentum_reset_interval
        self.beta1_default = beta1_default
        self.beta1_low = beta1_low
        self.beta2_default = beta2_default
        self.beta2_low = beta2_low
        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # 逃离状态
        self._escape_remaining: int = 0
        self._escape_dir: torch.Tensor | None = None
        self._pre_escape_state: dict | None = None
        self._pre_escape_loss: float | None = None
        self._last_loss: float | None = None
        self._revert_pending: bool = False
        self._escape_trials_left: int = 0  # 剩余尝试方向数
        self._best_escape_state: dict | None = None
        self._best_escape_loss: float | None = None

        self.step_count = 0
        self.mode = "normal"
        self.verbose = verbose
        self._lr_sum: float = 0.0
        self._noise_count: int = 0
        self._escape_count: int = 0
        self._revert_count: int = 0

    def _compute_grad_norm(self):
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

    def _is_plateau(self):
        if len(self.grad_norm_history) < self.plateau_patience:
            return False
        recent = list(self.grad_norm_history)[-self.plateau_patience:]
        avg = sum(recent) / len(recent)
        if self.grad_peak < 1e-12:
            return False
        return (avg / self.grad_peak) < self.plateau_threshold

    def _save_state(self):
        state = {}
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                state[p] = p.data.clone()
        return state

    def _restore_state(self, state):
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                if p in state:
                    p.data.copy_(state[p])

    def observe_gradients(self):
        """optimizer.step() 之前"""
        gn = self._compute_grad_norm()
        self.grad_norm_history.append(gn)
        if gn > self.grad_peak:
            self.grad_peak = gn

        # 1. 回退：上一步逃离后 loss 恢化
        if self._revert_pending and self._pre_escape_state is not None:
            self._restore_state(self._pre_escape_state)
            self._escape_remaining = 0
            self._revert_pending = False
            self._pre_escape_state = None
            self._pre_escape_loss = None
            self._plateau_counter = 0
            self._revert_count += 1
            # 保留当前最佳（如果有的话），尝试下一个方向
            self._escape_trials_left -= 1
            if self._escape_trials_left > 0:
                self._start_escape_trial()
            return

        # 2. 逃离进行中
        if self._escape_remaining > 0 and self._escape_dir is not None:
            for pg in self.optimizer.param_groups:
                for p in pg["params"]:
                    push = self._escape_dir[:p.numel()].reshape_as(p.data)
                    p.data.add_(push * self.escape_push)
                    if p.grad is not None:
                        p.grad.data.zero_()
            self._noise_count += 1
            if self._escape_remaining % self.momentum_reset_interval == 0:
                for state in self.optimizer.state.values():
                    if "exp_avg" in state:
                        state["exp_avg"].mul_(0.0)
                    if "exp_avg_sq" in state:
                        state["exp_avg_sq"].mul_(0.0)

    def _start_escape_trial(self):
        """开始一个新的逃离尝试"""
        self._escape_remaining = self.escape_duration
        self._pre_escape_loss = self._last_loss
        self._pre_escape_state = self._save_state()
        total_params = sum(p.numel() for pg in self.optimizer.param_groups for p in pg["params"])
        self._escape_dir = torch.randn(total_params)
        self._escape_dir = self._escape_dir / (self._escape_dir.norm() + 1e-8)

    def step(self, loss_value=None):
        """optimizer.step() 之后"""
        self.step_count += 1

        if loss_value is not None:
            val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            self._last_loss = val

        # 1. 逃离倒计时结束
        if self._escape_remaining > 0:
            self._escape_remaining -= 1
            if self._escape_remaining == 0 and self._pre_escape_loss is not None:
                if self._last_loss is not None and self._last_loss >= self._pre_escape_loss:
                    # Loss 没有改善 → 回退
                    self._revert_pending = True
                else:
                    # Loss 真正改善了 → 保留
                    self._escape_trials_left = 0
                    self._pre_escape_loss = None
                    self._pre_escape_state = None
                    self._plateau_counter = 0
            ctrl = 1.0
        else:
            ctrl = 0.0

        # 2. Cosine LR
        cos_factor = self._get_cosine_lr(self.step_count)

        # 3. Plateau detection
        if not self._revert_pending and self._escape_remaining <= 0:
            if self._is_plateau():
                self._plateau_counter += 1
            else:
                self._plateau_counter = 0

        # 4. 触发逃离（多方向尝试）
        if (self._plateau_counter >= self.plateau_patience
                and self._escape_remaining <= 0
                and not self._revert_pending
                and self._escape_trials_left <= 0):
            self._escape_count += 1
            self._escape_trials_left = self.max_escape_trials
            self._start_escape_trial()

        # 5. Apply LR / betas
        lr_factor = cos_factor * (1.0 + (self.lr_boost - 1.0) * ctrl)
        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self.initial_lrs[i] * lr_factor
        b1 = self.beta1_default - (self.beta1_default - self.beta1_low) * ctrl
        b2 = self.beta2_default - (self.beta2_default - self.beta2_low) * ctrl
        for pg in self.optimizer.param_groups:
            pg["betas"] = (b1, b2)

        self.mode = "escape" if ctrl > 0.3 else "normal"
        self._lr_sum += lr_factor

    def get_stats(self):
        return {
            "avg_lr_ratio": self._lr_sum / max(self.step_count, 1),
            "noise_injections": self._noise_count,
            "escape_events": self._escape_count,
            "reverts": self._revert_count,
            "grad_peak": self.grad_peak,
        }

    def state_dict(self):
        return {"step_count": self.step_count, "grad_peak": self.grad_peak, "initial_lrs": self.initial_lrs}

    def load_state_dict(self, sd):
        self.step_count = sd["step_count"]
        self.grad_peak = sd.get("grad_peak", 0.0)
        self.initial_lrs = sd["initial_lrs"]
