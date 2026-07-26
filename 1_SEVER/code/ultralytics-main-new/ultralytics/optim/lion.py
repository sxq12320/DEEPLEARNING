# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Lion optimizer (EvoLved Sign Momentum).

Reference: Chen et al., "Symbolic Discovery of Optimization Algorithms" (NeurIPS 2023),
arXiv:2302.06675. 符号回归搜索发现的优化器：只跟踪动量、用 sign 更新，显存省一半；
在同等 batch 下通常需要比 AdamW 小 3-10x 的学习率（建议 lr0=0.001-0.003 配合本仓库协议）。
"""

from __future__ import annotations

import torch
from torch.optim.optimizer import Optimizer


class Lion(Optimizer):
    """Lion optimizer: sign-of-interpolated-momentum update with decoupled weight decay."""

    def __init__(self, params, lr: float = 1e-4, betas: tuple = (0.9, 0.99), weight_decay: float = 0.0):
        if lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """Perform a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if len(state) == 0:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                # decoupled weight decay
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)
                # update = sign(beta1 * m + (1 - beta1) * g)
                update = exp_avg.mul(beta1).add_(grad, alpha=1.0 - beta1).sign_()
                p.add_(update, alpha=-lr)
                # momentum update: m = beta2 * m + (1 - beta2) * g
                exp_avg.mul_(beta2).add_(grad, alpha=1.0 - beta2)

        return loss
