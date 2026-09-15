"""Shared numerics for the existing SMC/SMCAO AdamW controllers; no new optimizer."""

import math

import torch


class ControllerNumerics:
    """Avoid tensor-by-tensor GPU synchronization and competing LR schedules."""

    def _compute_grad_norm(self):
        by_device = {}
        for group in self.optimizer.param_groups:
            for parameter in group["params"]:
                if parameter.grad is not None:
                    grad = parameter.grad.detach()
                    by_device.setdefault(grad.device, []).append(grad.float().norm())
        return math.sqrt(sum(torch.stack(norms).square().sum().item() for norms in by_device.values()))

    def set_training_context(self, base_lrs, warmup_active):
        """The trainer owns warmup/linear/cosine timing, including accumulation."""
        if len(base_lrs) != len(self.optimizer.param_groups):
            raise ValueError("SMC base learning rates must match parameter groups")
        self._external_lrs = list(base_lrs)
        self._external_warmup = bool(warmup_active)
        factor = 1.0 if warmup_active else getattr(self, "_control_factor", 1.0)
        self._apply_lrs(factor)

    def _warming_up(self):
        return getattr(self, "_external_warmup", self.step_count < self.warmup_steps)

    def _apply_lrs(self, factor):
        self._control_factor = factor
        base = getattr(self, "_external_lrs", self.initial_lrs)
        for group, lr in zip(self.optimizer.param_groups, base):
            group["lr"] = lr * factor

    @staticmethod
    @torch.no_grad()
    def _add_relative_noise(grad, noise, scale):
        """Bound perturbation L2 by scale * gradient L2, independent of tensor size."""
        norm = grad.detach().float().norm()
        noise = noise.float()
        noise.mul_(norm * scale / noise.norm().clamp_min(1e-12))
        grad.add_(noise.to(grad.dtype))

    def state_dict(self):
        """Keep controller configuration and ALL counters; optimizer is saved separately."""
        return {key: value for key, value in vars(self).items() if key != "optimizer"}

    def load_state_dict(self, state):
        if len(state["initial_lrs"]) != len(self.optimizer.param_groups):
            raise ValueError("Cannot restore SMC state with different parameter groups")
        # Accept older checkpoints with partial controller state.
        for key, value in state.items():
            if key != "optimizer":
                setattr(self, key, value)
