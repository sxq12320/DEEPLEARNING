"""
SMCScheduler — 基于滑模控制 (Sliding Mode Control) 的 AdamW 学习率/动量调度器

理论基础：
    滑模面 s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)
    其中 ||g_t|| 为当前 step 所有参数梯度的全局 L2 范数。

三种工作模式：
    1. 正常模式 (Normal)：|s_t| 均值较高，梯度持续变化 → 保持初始 LR 和 β₁
    2. 鞍点逃离 (Saddle Escape)：|s_t| 均值极低，Loss 无明显下降 → 降低 β₁ 解耦动量、提升 LR
    3. 震荡平滑 (Chattering Damping)：|s_t| 方差极大，梯度剧烈震荡 → 增加 Weight Decay、缩小 LR

工程安全性：底层完全依赖标准 AdamW，仅通过动态修改 param_groups 实现控制。
"""

import math
import torch
from collections import deque


class SMCScheduler:
    """
    滑模控制调度器 — 包装 AdamW 优化器，基于梯度滑模面动态调节 LR / β₁ / Weight Decay。

    使用方式：
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scheduler = SMCScheduler(optimizer, total_steps=10000)
        for step in range(total_steps):
            loss = train_step()
            loss.backward()
            scheduler.observe_gradients()  # 在 optimizer.step() 之前调用
            optimizer.step()
            scheduler.step(loss)           # 在 optimizer.step() 之后调用

    Args:
        optimizer: 已创建的 AdamW 优化器实例
        total_steps: 总训练步数（用于判断是否进入鞍点逃离模式的 Loss 参考窗口）
        c: 滑模面系数 s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)
        window_size: 滑动窗口长度 W，用于计算 |s_t| 的均值和方差
        thr_reach: 正常模式阈值，窗口 |s_t| 均值 > thr_reach 时保持默认参数
        thr_saddle: 鞍点逃离阈值，窗口 |s_t| 均值 < thr_saddle 且 Loss 无下降时触发逃离
        thr_chatter_var: 震荡方差阈值，窗口 |s_t| 方差 > 此值时触发震荡平滑
        beta1_default: 默认 β₁ (AdamW 动量)
        beta1_escape: 鞍点逃离时的 β₁ (低动量，解耦历史梯度)
        lr_boost: 鞍点逃离时 LR 乘数
        escape_duration: 鞍点逃离持续步数
        lr_dampen: 震荡平滑时 LR 缩放因子 (< 1.0)
        wd_boost: 震荡平滑时 Weight Decay 增量
        loss_window: 判断 Loss 是否下降的窗口长度
        loss_threshold: Loss 下降阈值（相对变化率）
        verbose: 是否打印模式切换信息
    """

    def __init__(
        self,
        optimizer,
        total_steps=10000,
        c=0.5,
        window_size=50,
        thr_reach=1e-3,
        thr_saddle=1e-5,
        thr_chatter_var=1e-6,
        beta1_default=0.9,
        beta1_escape=0.1,
        lr_boost=2.0,
        escape_duration=30,
        lr_dampen=0.5,
        wd_boost=0.01,
        loss_window=20,
        loss_threshold=1e-4,
        verbose=True,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps

        # 滑模面参数
        self.c = c
        self.prev_grad_norm = None  # ||g_{t-1}||

        # 滑动窗口
        self.window_size = window_size
        self.s_window = deque(maxlen=window_size)

        # 阈值
        self.thr_reach = thr_reach
        self.thr_saddle = thr_saddle
        self.thr_chatter_var = thr_chatter_var

        # β₁ 控制
        self.beta1_default = beta1_default
        self.beta1_escape = beta1_escape

        # LR / WD 控制
        self.lr_boost = lr_boost
        self.lr_dampen = lr_dampen
        self.wd_boost = wd_boost

        # 鞍点逃离状态机
        self.escape_duration = escape_duration
        self.escape_counter = 0  # 剩余逃离步数

        # Loss 跟踪
        self.loss_window = loss_window
        self.loss_threshold = loss_threshold
        self.loss_history = deque(maxlen=loss_window)

        # 初始参数备份
        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self.initial_wds = [pg["weight_decay"] for pg in optimizer.param_groups]

        # 当前模式
        self.mode = "normal"  # "normal" | "escape" | "damping"

        self.verbose = verbose
        self.step_count = 0

    def _compute_grad_norm(self):
        """计算所有参数梯度的全局 L2 范数 ||g_t||"""
        total_norm_sq = 0.0
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    total_norm_sq += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm_sq)

    def _compute_sliding_surface(self, grad_norm):
        """计算滑模面 s_t = c × ||g_t|| + (||g_t|| - ||g_{t-1}||)"""
        if self.prev_grad_norm is None:
            s_t = self.c * grad_norm
        else:
            s_t = self.c * grad_norm + (grad_norm - self.prev_grad_norm)
        self.prev_grad_norm = grad_norm
        return s_t

    def _loss_decreasing(self):
        """判断 Loss 是否在近期有明显下降（兼容负数 Loss 和接近 0 的情况）"""
        if len(self.loss_history) < self.loss_window:
            return True  # 数据不足时假设正常
        recent = list(self.loss_history)
        first_half = sum(recent[: len(recent) // 2]) / (len(recent) // 2)
        second_half = sum(recent[len(recent) // 2 :]) / (len(recent) - len(recent) // 2)
        eps = 1e-8
        absolute_drop = first_half - second_half
        relative_drop = absolute_drop / (abs(first_half) + eps)
        return (absolute_drop > self.loss_threshold) or (relative_drop > self.loss_threshold)

    def _set_mode(self, new_mode):
        """切换模式并调整优化器参数"""
        if new_mode == self.mode and new_mode != "escape":
            return

        if self.verbose and new_mode != self.mode:
            print(f"[SMCScheduler] step={self.step_count} mode: {self.mode} → {new_mode}")

        self.mode = new_mode

        if new_mode == "normal":
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.initial_lrs[i]
                pg["betas"] = (self.beta1_default, pg["betas"][1])
                pg["weight_decay"] = self.initial_wds[i]

        elif new_mode == "escape":
            self.escape_counter = self.escape_duration
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.initial_lrs[i] * self.lr_boost
                pg["betas"] = (self.beta1_escape, pg["betas"][1])
                pg["weight_decay"] = self.initial_wds[i]

        elif new_mode == "damping":
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.initial_lrs[i] * self.lr_dampen
                pg["betas"] = (self.beta1_default, pg["betas"][1])
                pg["weight_decay"] = self.initial_wds[i] + self.wd_boost

    def observe_gradients(self):
        """在 optimizer.step() 之前调用，捕获真实的梯度状态并记录滑模面值"""
        grad_norm = self._compute_grad_norm()
        s_t = self._compute_sliding_surface(grad_norm)
        self.s_window.append(abs(s_t))

    def step(self, loss_value=None):
        """
        在 optimizer.step() 之后调用，传入当前 loss 用于状态机判断。

        Args:
            loss_value: 当前 step 的 loss 标量值（可选，但推荐传入）
        """
        self.step_count += 1

        if loss_value is not None:
            self.loss_history.append(
                loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            )

        # 窗口未满时保持正常模式
        if len(self.s_window) < self.window_size:
            return

        s_mean = sum(self.s_window) / len(self.s_window)
        s_var = sum((x - s_mean) ** 2 for x in self.s_window) / len(self.s_window)

        # 如果正在执行鞍点逃离，倒计时
        if self.escape_counter > 0:
            self.escape_counter -= 1
            if self.escape_counter == 0:
                self._set_mode("normal")
            return

        # ===== 模式判断（优先级：震荡平滑 > 鞍点逃离 > 正常） =====

        # 1. 震荡平滑：方差极大
        if s_var > self.thr_chatter_var and s_mean > self.thr_reach:
            self._set_mode("damping")
            return

        # 2. 鞍点逃离：均值极低且 Loss 无明显下降
        if s_mean < self.thr_saddle and not self._loss_decreasing():
            self._set_mode("escape")
            return

        # 3. 正常模式
        self._set_mode("normal")

    def state_dict(self):
        """导出调度器状态"""
        return {
            "step_count": self.step_count,
            "prev_grad_norm": self.prev_grad_norm,
            "s_window": list(self.s_window),
            "loss_history": list(self.loss_history),
            "escape_counter": self.escape_counter,
            "mode": self.mode,
            "initial_lrs": self.initial_lrs,
            "initial_wds": self.initial_wds,
        }

    def load_state_dict(self, state_dict):
        """恢复调度器状态"""
        self.step_count = state_dict["step_count"]
        self.prev_grad_norm = state_dict["prev_grad_norm"]
        self.s_window = deque(state_dict["s_window"], maxlen=self.window_size)
        self.loss_history = deque(state_dict["loss_history"], maxlen=self.loss_window)
        self.escape_counter = state_dict["escape_counter"]
        self.mode = state_dict["mode"]
        self.initial_lrs = state_dict["initial_lrs"]
        self.initial_wds = state_dict["initial_wds"]
