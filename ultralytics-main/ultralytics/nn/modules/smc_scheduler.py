"""
SMCScheduler — 基于滑模控制的 AdamW 调度器

核心：plateau 检测 → 定向逃离 → 快速重收敛 → loss 比较决定保留/回退
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
        escape_push=0.10,
        escape_push_steps=20,
        reconv_steps=40,
        reconv_lr_mult=3.0,
        beta1_default=0.9,
        beta1_low=0.1,
        beta2_default=0.999,
        beta2_low=0.9,
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

        self.escape_push = escape_push
        self.escape_push_steps = escape_push_steps
        self.reconv_steps = reconv_steps
        self.reconv_lr_mult = reconv_lr_mult
        self.beta1_default = beta1_default
        self.beta1_low = beta1_low
        self.beta2_default = beta2_default
        self.beta2_low = beta2_low
        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # State
        self._escape_dir: torch.Tensor | None = None
        self._last_loss: float | None = None
        self._in_escape: bool = False
        self.step_count = 0
        self.mode = "normal"
        self.verbose = verbose
        self._lr_sum: float = 0.0
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
        gn = self._compute_grad_norm()
        self.grad_norm_history.append(gn)
        if gn > self.grad_peak:
            self.grad_peak = gn

    def step(self, loss_value=None):
        self.step_count += 1
        if loss_value is not None:
            val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            self._last_loss = val

        if self._in_escape:
            return  # 逃离期间不干预

        cos_factor = self._get_cosine_lr(self.step_count)

        # Plateau detection
        if self._is_plateau():
            self._plateau_counter += 1
        else:
            self._plateau_counter = 0

        # 触发逃离：push → reconv → 比较 loss
        if self._plateau_counter >= self.plateau_patience:
            self._plateau_counter = 0
            self._in_escape = True
            self._escape_count += 1
            pre_escape_loss = self._last_loss
            pre_escape_state = self._save_state()

            # 1. 定向逃离（在参数所在设备上）
            first_param = next(p for pg in self.optimizer.param_groups for p in pg["params"])
            device = first_param.device
            total_params = sum(p.numel() for pg in self.optimizer.param_groups for p in pg["params"])
            escape_dir = torch.randn(total_params, device=device)
            escape_dir = escape_dir / (escape_dir.norm() + 1e-8)

            for step_i in range(self.escape_push_steps):
                for pg in self.optimizer.param_groups:
                    for p in pg["params"]:
                        push = escape_dir[:p.numel()].reshape_as(p.data)
                        p.data.add_(push * self.escape_push)
                        if p.grad is not None:
                            p.grad.data.zero_()

            # 2. 快速重收敛（高 LR + 正常梯度）
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.initial_lrs[0] * self.reconv_lr_mult
                pg["betas"] = (self.beta1_low, self.beta2_low)
            for state in self.optimizer.state.values():
                if "exp_avg" in state:
                    state["exp_avg"].mul_(0.0)
                if "exp_avg_sq" in state:
                    state["exp_avg_sq"].mul_(0.0)

            for pg in self.optimizer.param_groups:
                pg["lr"] = self.initial_lrs[0] * self.reconv_lr_mult
                pg["betas"] = (self.beta1_low, self.beta2_low)

            self._reconv_loss = self._last_loss
            self._reconv_step = 0

        # 重收敛期间：正常训练但用高 LR
        if hasattr(self, '_reconv_step') and self._reconv_step < self.reconv_steps:
            for pg in self.optimizer.param_groups:
                pg["lr"] = self.initial_lrs[0] * self.reconv_lr_mult
                pg["betas"] = (self.beta1_low, self.beta2_low)
            self._reconv_step += 1

            if self._reconv_step >= self.reconv_steps:
                # 重收敛结束，比较 loss
                post_escape_loss = self._last_loss
                if post_escape_loss is not None and pre_escape_loss is not None:
                    if post_escape_loss >= pre_escape_loss:
                        # Loss 没改善 → 回退
                        self._restore_state(pre_escape_state)
                        self._revert_count += 1
                    # 否则保留逃离结果
                # 恢复正常
                self._in_escape = False
                if hasattr(self, '_reconv_step'):
                    del self._reconv_step
                if hasattr(self, '_reconv_loss'):
                    del self._reconv_loss
                self._plateau_counter = 0
        else:
            # 正常训练
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.initial_lrs[i] * cos_factor
                pg["betas"] = (self.beta1_default, self.beta2_default)

        self._lr_sum += self.optimizer.param_groups[0]["lr"] / self.initial_lrs[0] if self.initial_lrs[0] > 0 else 1.0

    def get_stats(self):
        return {
            "avg_lr_ratio": self._lr_sum / max(self.step_count, 1),
            "escape_events": self._escape_count,
            "reverts": self._revert_count,
        }

    def state_dict(self):
        return {"step_count": self.step_count, "initial_lrs": self.initial_lrs}

    def load_state_dict(self, sd):
        self.step_count = sd["step_count"]
        self.initial_lrs = sd["initial_lrs"]
