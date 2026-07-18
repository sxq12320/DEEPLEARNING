"""
SMCAO V2.2 — 滑模控制自适应优化器调度器（局部极小值逃逸增强版）
==================================================================

在 SMCScheduler V3 基础上，引入四项控制理论级改进以突破高维非凸空间的局部极小值：

1. 随机滑模面抖动 (Stochastic Dither Signal Injection)
   - 当陷入停滞时，向滑模面注入 Lévy 飞行重尾扰动，强制轨迹偏离滑模面
   - s = c(f)·∇f + v + σ_escape·ξ_levy
   - Lévy 分布的厚尾特性使参数有概率进行大跳步，跨越浅层势垒

2. 负阻尼能量注入 (Negative Damping & Tunneling)
   - 当梯度≈0且损失仍高时，临时切换阻尼系数为负值
   - a_eff = a₀·tanh(γ·(f - f_target))·sgn(‖∇f‖ - ε_g)
   - 负阻尼使速度指数级发散，驱动参数冲出当前洼地
   - 一旦梯度恢复，a_eff 自动回归正阻尼，系统重新收敛

3. 分数阶记忆效应 (Fractional-Order Memory Effect)
   - 引入梯度历史的指数加权积分，保留轨迹历史路径信息
   - s = c·∇f + v + κ_mem·D^{-α}∇f
   - D^{-α}∇f ≈ Σ_{k=0}^{W} ρ^k · g_{t-k}  (指数衰减窗口)
   - 累积的历史动量产生持续推力，帮助参数"飘"过平坦区

4. 自适应损失引导滑模面 (Loss-Driven Dynamic Surface Reshaping)
   - 滑模面系数 c 不再是常数，而是损失值的动态函数
   - c(f) = c₀·exp(-β/(f + ε))
   - 高损失区：c(f)≈c₀，强滑模约束快速滑行
   - 低损失区：c(f)变小，释放约束允许精修
   - 局部极小值（高f但∇f≈0）：c(f)保持大，速度主导冲过势垒

安全性保证：
  - 底层仍依赖标准 AdamW，通过动态修改 param_groups 实现控制
  - 不推动参数本身（不破坏特征表示）
  - 不重置 Adam 状态（不丢失动量信息）
  - 负阻尼有速度钳制（v_max），防止数值爆炸
  - Lévy 扰动仅在 escape 期间注入，且有衰减和时限
  - 分数阶记忆窗口有限（W 步），计算开销 O(W) 可控
"""

import math
import torch


def _levy_noise(shape, device, alpha=1.5, scale=1.0):
    """
    生成 Lévy 飞行噪声（Mantegna 算法）。

    Lévy 分布具有重尾特性，使扰动偶尔产生大幅跳步，
    有助于参数跨越局部势垒（区别于高斯扰动的局部探索）。

    Args:
        shape: 噪声张量形状
        device: 设备
        alpha: 稳定指数 (1 < α < 2)，越小尾部越重；α=2 退化为高斯
        scale: 缩放系数

    Returns:
        Lévy 噪声张量
    """
    # Mantegna 算法
    sigma_u = (
        math.gamma(1 + alpha) * math.sin(math.pi * alpha / 2)
        / (math.gamma((1 + alpha) / 2) * alpha * 2 ** ((alpha - 1) / 2))
    ) ** (1 / alpha)
    u = torch.randn(shape, device=device) * sigma_u
    v = torch.randn(shape, device=device).abs().clamp(min=1e-12)
    step = u / (v ** (1 / alpha))
    # 缩放并钳制极端值（安全措施）
    step = step * scale
    step = step.clamp(-10.0, 10.0)
    return step


class SMCAOV22Scheduler:
    """
    SMCAO V2.2 调度器 — 四项改进驱动的局部极小值逃逸。

    核心动力学：
        θ_next = θ + v · dt
        v_next = v + (-a_eff · v - κ_eff · sat(s/φ) - ki · Z) · dt
        Z_next = Z + s · dt

    其中：
        s = c(f) · ∇f + v + κ_mem · D^{-α}∇f + σ_escape · ξ_levy
        a_eff = a₀ · tanh(γ · (f - f_target)) · sgn(‖∇f‖ - ε_g)
        c(f) = c₀ · exp(-β / (f + ε))

    在 YOLO 训练循环中的调用方式（与 SMCScheduler 相同）：
        smc.observe_gradients()   # optimizer.step() 之前
        smc.step(loss_value)      # optimizer.step() 之后
        smc.on_train_epoch_end(epoch_loss)  # epoch 结束

    Args:
        optimizer: AdamW 优化器
        total_steps: 总训练步数
        c0: 滑模面基础系数
        warmup_steps: 预热步数
        min_lr_ratio: 最小学习率比例（cosine 终点）

        # 滑模面停滞检测
        surface_threshold: |s_t|/s_t_peak 低于此值视为停滞
        surface_patience: 停滞持续步数触发 escape

        # 机制1: 随机滑模面抖动
        dither_scale: Lévy 扰动基础缩放
        dither_alpha: Lévy 稳定指数 (1 < α < 2)
        dither_max_steps: 单次 escape 最多注入 Lévy 扰动的步数
        dither_decay: 扰动随步数的衰减系数

        # 机制2: 负阻尼能量注入
        a0: 基础阻尼系数（正阻尼）
        a_neg: 负阻尼幅度（escape 时 a = -a_neg）
        gamma_damping: 负阻尼激活的 tanh 增益
        grad_threshold: 梯度低于此值视为"梯度消失"

        # 机制3: 分数阶记忆效应
        kappa_mem: 分数阶记忆项系数
        alpha_frac: 分数阶阶数 (0 < α < 1)
        memory_window: 记忆窗口长度 W

        # 机制4: 自适应损失引导滑模面
        c_beta: c(f) = c0·exp(-beta/(f+eps)) 中的 beta
        c_eps: c(f) 中的 epsilon（防除零）

        # 通用控制
        lr_boost: escape 时 LR 提升倍数
        escape_cooldown: escape 结束后冷却步数
        escape_max_duration: 单次 escape 最长持续步数
        beta1_default: 默认 β₁
        beta1_low: escape 时 β₁
        beta2_default: 默认 β₂
        ki: 积分增益
        phi: sat 饱和边界层厚度
        v_max: 速度钳制上限
        verbose: 是否打印状态信息
    """

    def __init__(
        self,
        optimizer,
        total_steps=10000,
        c0=0.5,
        warmup_steps=100,
        min_lr_ratio=0.01,
        # 停滞检测
        surface_threshold=0.05,
        surface_patience=100,
        # 机制1: 随机滑模面抖动
        dither_scale=0.003,
        dither_alpha=1.5,
        dither_max_steps=10,
        dither_decay=0.9,
        # 机制2: 负阻尼能量注入
        a0=5.0,
        a_neg=1.0,
        gamma_damping=2.0,
        grad_threshold=0.01,
        # 机制3: 分数阶记忆
        kappa_mem=0.1,
        alpha_frac=0.5,
        memory_window=20,
        # 机制4: 自适应损失引导
        c_beta=1.0,
        c_eps=0.01,
        # 通用控制
        lr_boost=1.1,
        escape_cooldown=100,
        escape_max_duration=25,
        beta1_default=0.9,
        beta1_low=0.88,
        beta2_default=0.999,
        ki=0.001,
        phi=0.1,
        v_max=10.0,
        verbose=True,
    ):
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.c0 = c0
        self.warmup_steps = warmup_steps
        self.min_lr_ratio = min_lr_ratio

        # 停滞检测
        self.surface_threshold = surface_threshold
        self.surface_patience = surface_patience

        # 机制1: 随机滑模面抖动
        self.dither_scale = dither_scale
        self.dither_alpha = dither_alpha
        self.dither_max_steps = dither_max_steps
        self.dither_decay = dither_decay

        # 机制2: 负阻尼能量注入
        self.a0 = a0
        self.a_neg = a_neg
        self.gamma_damping = gamma_damping
        self.grad_threshold = grad_threshold

        # 机制3: 分数阶记忆
        self.kappa_mem = kappa_mem
        self.alpha_frac = alpha_frac
        self.memory_window = memory_window

        # 机制4: 自适应损失引导
        self.c_beta = c_beta
        self.c_eps = c_eps

        # 通用控制
        self.lr_boost = lr_boost
        self.escape_cooldown = escape_cooldown
        self.escape_max_duration = escape_max_duration
        self.beta1_default = beta1_default
        self.beta1_low = beta1_low
        self.beta2_default = beta2_default
        self.ki = ki
        self.phi = phi
        self.v_max = v_max
        self.verbose = verbose

        self.initial_lrs = [pg["lr"] for pg in optimizer.param_groups]

        # ── 滑模面状态 ──
        self.s_t: float | None = None
        self.s_t_peak: float = 0.0
        self.prev_grad_norm: float | None = None
        self.c_current: float = c0  # 动态 c(f) 当前值

        # ── 机制3: 分数阶记忆 ──
        # 存储最近 W 步的梯度范数，用于计算 D^{-α}∇f 的标量近似
        self._grad_norm_history: list[float] = []

        # ── 机制2: 负阻尼状态 ──
        self.a_eff: float = a0  # 当前有效阻尼系数
        self._velocity_norm: float = 0.0  # 当前速度范数

        # ── 积分项 Z ──
        self._integral_Z: float = 0.0

        # ── Escape 状态 ──
        self._in_escape: bool = False
        self._escape_step_counter: int = 0
        self._cooldown_counter: int = 0
        self._surface_counter: int = 0

        # ── 通用状态 ──
        self.step_count = 0
        self.mode = "normal"
        self._lr_sum: float = 0.0
        self._noise_count: int = 0
        self._escape_events: int = 0
        self._last_loss: float | None = None
        self._best_loss: float | None = None
        self._loss_plateau_count: int = 0

    # ================================================================
    #  基础工具方法
    # ================================================================

    def _compute_grad_norm(self):
        """计算所有参数梯度的全局 L2 范数"""
        total_sq = 0.0
        for pg in self.optimizer.param_groups:
            for p in pg["params"]:
                if p.grad is not None:
                    total_sq += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_sq)

    def _get_cosine_lr(self, step):
        """Cosine 学习率调度"""
        if step < self.warmup_steps:
            return step / max(self.warmup_steps, 1)
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        return self.min_lr_ratio + 0.5 * (1.0 - self.min_lr_ratio) * (1.0 + math.cos(math.pi * progress))

    @staticmethod
    def _sat(x, phi):
        """饱和函数 sat(x/φ)"""
        return max(-1.0, min(1.0, x / phi))

    # ================================================================
    #  机制4: 自适应损失引导滑模面 c(f)
    # ================================================================

    def _compute_adaptive_c(self, loss_val):
        """
        c(f) = c₀ · exp(-β / (f + ε))

        高损失区（远离最优）：c(f) ≈ c₀，强滑模约束
        低损失区（接近最优）：c(f) → 0，释放约束允许精修
        局部极小值（高f但∇f≈0）：c(f) 保持大，速度主导冲势垒
        """
        if loss_val is None or loss_val < 0:
            return self.c0
        # 防止 loss 过大导致 exp 溢出
        exponent = -self.c_beta / (loss_val + self.c_eps)
        exponent = max(-20.0, min(0.0, exponent))
        return self.c0 * math.exp(exponent)

    # ================================================================
    #  机制3: 分数阶记忆效应 D^{-α}∇f
    # ================================================================

    def _compute_fractional_memory(self):
        """
        分数阶积分的标量近似：
        D^{-α}∇f ≈ Σ_{k=0}^{W-1} ρ^k · g_{t-k}

        其中 ρ = (1 - α_frac) 为衰减系数，W 为记忆窗口。
        这保留了梯度历史的路径信息，在平坦区产生持续推力。
        """
        if not self._grad_norm_history:
            return 0.0

        rho = 1.0 - self.alpha_frac  # 衰减系数
        mem = 0.0
        for k, gn in enumerate(self._grad_norm_history):
            mem += (rho ** k) * gn
        return mem

    # ================================================================
    #  机制2: 负阻尼能量注入
    # ================================================================

    def _compute_effective_damping(self, grad_norm, loss_val):
        """
        a_eff = a₀ · tanh(γ · (f - f_target)) · sgn(‖∇f‖ - ε_g)

        当梯度≈0且损失仍高（卡在局部极小值）时：
          tanh > 0 且 sgn < 0 → a_eff < 0（负阻尼）
          系统从外界吸收能量，速度发散，冲出洼地

        当梯度恢复（已逃出局部极小值）时：
          sgn > 0 → a_eff > 0（正阻尼）
          系统重新收敛
        """
        if loss_val is None or self._best_loss is None:
            return self.a0

        # 损失残差：当前损失相对于最优损失的差距
        loss_residual = loss_val - self._best_loss
        if loss_residual <= 0:
            return self.a0  # 已经是最优，正常阻尼

        # tanh 调度：损失越高，激活越强
        tanh_val = math.tanh(self.gamma_damping * loss_residual)

        # 梯度消失检测
        if grad_norm < self.grad_threshold:
            # 梯度消失 + 高损失 → 负阻尼
            a_eff = -self.a_neg * tanh_val
        else:
            # 梯度正常 → 正阻尼
            a_eff = self.a0

        return a_eff

    # ================================================================
    #  滑模面计算（整合四项改进）
    # ================================================================

    def _compute_sliding_surface(self, grad_norm, loss_val):
        """
        完整滑模面：
        s = c(f) · ‖∇f‖ + v + κ_mem · D^{-α}∇f + σ_escape · ξ_levy

        其中各项对应：
        - c(f)·‖∇f‖: 自适应损失引导的梯度分量（机制4）
        - v: 速度分量（二阶动力学）
        - κ_mem·D^{-α}∇f: 分数阶记忆分量（机制3）
        - σ_escape·ξ_levy: 随机抖动分量（机制1，仅escape时注入）
        """
        # 机制4: 动态 c(f)
        c_f = self._compute_adaptive_c(loss_val)
        self.c_current = c_f

        # 速度分量（使用上一步的速度范数近似）
        v_component = self._velocity_norm

        # 机制3: 分数阶记忆分量
        mem_component = self.kappa_mem * self._compute_fractional_memory()

        # 基础滑模面
        s_t = c_f * grad_norm + v_component + mem_component

        # 机制1: 随机滑模面抖动（仅 escape 期间注入）
        # 注：Lévy 噪声的梯度注入在 observe_gradients() 中完成，
        # 这里仅对滑模面标量值添加扰动以影响停滞检测
        if self._in_escape and self._escape_step_counter < self.dither_max_steps:
            current_scale = self.dither_scale * (self.dither_decay ** self._escape_step_counter)
            # Lévy 标量扰动
            levy_s = _levy_noise((1,), device="cpu", alpha=self.dither_alpha, scale=current_scale)
            s_t += levy_s.item()

        self.prev_grad_norm = grad_norm
        self.s_t = s_t
        return s_t

    # ================================================================
    #  主接口：observe_gradients (optimizer.step() 之前调用)
    # ================================================================

    def observe_gradients(self):
        """
        optimizer.step() 之前调用：
        1. 计算滑模面（含四项改进）
        2. Escape 时注入 Lévy 噪声到梯度
        3. 更新分数阶记忆缓冲区
        """
        if self.step_count < self.warmup_steps:
            return

        gn = self._compute_grad_norm()

        # 更新分数阶记忆缓冲区
        self._grad_norm_history.insert(0, gn)
        if len(self._grad_norm_history) > self.memory_window:
            self._grad_norm_history.pop()

        # 计算完整滑模面
        s_t = self._compute_sliding_surface(gn, self._last_loss)
        self._update_sliding_surface_stats(abs(s_t))

        # 机制1: Escape 时注入 Lévy 飞行噪声到梯度
        if self._in_escape and self._escape_step_counter < self.dither_max_steps and gn > 1e-12:
            current_scale = self.dither_scale * (self.dither_decay ** self._escape_step_counter)
            for pg in self.optimizer.param_groups:
                for p in pg["params"]:
                    if p.grad is not None:
                        grad_norm_p = p.grad.data.norm(2).item()
                        scale_p = current_scale * max(grad_norm_p, 1e-8)
                        noise = _levy_noise(
                            p.grad.data.shape,
                            device=p.grad.data.device,
                            alpha=self.dither_alpha,
                            scale=scale_p,
                        )
                        p.grad.data.add_(noise)
            self._noise_count += 1
            self._escape_step_counter += 1

    def _update_sliding_surface_stats(self, abs_s_t):
        """更新滑模面峰值"""
        if abs_s_t > self.s_t_peak:
            self.s_t_peak = abs_s_t

    # ================================================================
    #  主接口：step (optimizer.step() 之后调用)
    # ================================================================

    def step(self, loss_value=None):
        """
        optimizer.step() 之后调用：
        1. 更新 cosine LR
        2. 检测停滞/触发 escape
        3. 计算负阻尼 a_eff
        4. 更新速度 v 和积分项 Z
        5. 动态调整 AdamW 参数 (lr, betas)
        """
        self.step_count += 1

        # 记录 loss
        if loss_value is not None:
            val = loss_value.item() if isinstance(loss_value, torch.Tensor) else loss_value
            self._last_loss = val

        cos_factor = self._get_cosine_lr(self.step_count)

        # Warmup 期间不做任何控制逻辑
        if self.step_count < self.warmup_steps:
            for i, pg in enumerate(self.optimizer.param_groups):
                pg["lr"] = self.initial_lrs[i] * cos_factor
                pg["betas"] = (self.beta1_default, self.beta2_default)
            self.mode = "warmup"
            return

        # 滑模面峰值缓慢衰减
        self.s_t_peak *= 0.9999

        # ── 停滞检测 ──
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

        # ── Escape 触发条件 ──
        # 结合滑模面停滞 + epoch 级 loss plateau
        should_escape = (
            self._surface_counter >= self.surface_patience
            and self._cooldown_counter == 0
        )
        # 额外触发：epoch 级 loss plateau（宏观、稳定）
        if not should_escape and self._loss_plateau_count >= 3 and self._cooldown_counter == 0:
            should_escape = True

        # 激活/停用 escape
        if should_escape and not self._in_escape:
            self._in_escape = True
            self._escape_events += 1
            self._escape_step_counter = 0
            if self.verbose:
                print(f"[SMCAO V2.2] step={self.step_count}: escape triggered "
                      f"(surface_stall={self._surface_counter}, "
                      f"loss_plateau={self._loss_plateau_count}, "
                      f"ratio={surface_ratio:.4f})")
        elif self._in_escape:
            if self._surface_counter == 0 or self._escape_step_counter >= self.escape_max_duration:
                self._in_escape = False
                self._cooldown_counter = self.escape_cooldown
                if self.verbose:
                    print(f"[SMCAO V2.2] step={self.step_count}: escape deactivated "
                          f"(duration={self._escape_step_counter}, cooldown={self.escape_cooldown})")

        # ── 机制2: 计算有效阻尼系数 ──
        gn = self.prev_grad_norm if self.prev_grad_norm is not None else 0.0
        self.a_eff = self._compute_effective_damping(gn, self._last_loss)

        # ── 二阶动力学更新速度 v 和积分项 Z ──
        # v_next = v + (-a_eff·v - κ_eff·sat(s/φ) - ki·Z) · dt
        # Z_next = Z + s · dt
        s_val = self.s_t if self.s_t is not None else 0.0
        dt = 0.01  # 时间步长（归一化）

        # 饱和函数
        sat_s = self._sat(s_val, self.phi)

        # 自适应 κ（滑模面上放大以钉住轨迹）
        kappa_eff = 1.0 + 0.5 / (self.phi + abs(s_val))

        # 速度更新
        v_accel = -self.a_eff * self._velocity_norm - kappa_eff * sat_s - self.ki * self._integral_Z
        self._velocity_norm = self._velocity_norm + v_accel * dt

        # 速度钳制（安全措施：防止负阻尼导致发散）
        if abs(self._velocity_norm) > self.v_max:
            self._velocity_norm = self.v_max * (1.0 if self._velocity_norm > 0 else -1.0)

        # 积分项更新
        self._integral_Z = self._integral_Z + s_val * dt

        # ── 参数组更新 ──
        ctrl = 1.0 if self._in_escape else 0.0

        # LR: cosine * (1 + boost * ctrl)
        # 负阻尼时额外提升 LR（能量注入）
        neg_damping_boost = 1.0
        if self.a_eff < 0:
            neg_damping_boost = 1.0 + abs(self.a_eff) * 0.1  # 负阻尼越强，LR 额外提升越大

        lr_factor = cos_factor * (1.0 + (self.lr_boost - 1.0) * ctrl) * neg_damping_boost
        b1 = self.beta1_default - (self.beta1_default - self.beta1_low) * ctrl

        for i, pg in enumerate(self.optimizer.param_groups):
            pg["lr"] = self.initial_lrs[i] * lr_factor
            pg["betas"] = (b1, self.beta2_default)

        self.mode = "escape" if self._in_escape else "normal"
        if self.a_eff < 0:
            self.mode = "tunneling"  # 负阻尼隧道模式
        self._lr_sum += lr_factor

    # ================================================================
    #  Epoch 级接口
    # ================================================================

    def on_train_epoch_end(self, train_loss):
        """每个 epoch 结束时调用，用于 epoch 级 plateau 检测"""
        if train_loss is not None:
            if self._best_loss is None or train_loss < self._best_loss * 0.999:
                self._best_loss = train_loss
                self._loss_plateau_count = 0
            else:
                self._loss_plateau_count += 1

    # ================================================================
    #  状态查询与序列化
    # ================================================================

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
            "a_eff": self.a_eff,
            "c_current": self.c_current,
            "velocity_norm": self._velocity_norm,
            "mode": self.mode,
            "loss_plateau": self._loss_plateau_count,
        }

    def state_dict(self):
        return {
            "step_count": self.step_count,
            "prev_grad_norm": self.prev_grad_norm,
            "s_t": self.s_t,
            "s_t_peak": self.s_t_peak,
            "c_current": self.c_current,
            "initial_lrs": self.initial_lrs,
            "_best_loss": self._best_loss,
            "_loss_plateau_count": self._loss_plateau_count,
            "_in_escape": self._in_escape,
            "_escape_step_counter": self._escape_step_counter,
            "_cooldown_counter": self._cooldown_counter,
            "_surface_counter": self._surface_counter,
            "_velocity_norm": self._velocity_norm,
            "_integral_Z": self._integral_Z,
            "_grad_norm_history": list(self._grad_norm_history),
            "_escape_events": self._escape_events,
            "_noise_count": self._noise_count,
            "a_eff": self.a_eff,
        }

    def load_state_dict(self, sd):
        self.step_count = sd["step_count"]
        self.prev_grad_norm = sd.get("prev_grad_norm")
        self.s_t = sd.get("s_t")
        self.s_t_peak = sd.get("s_t_peak", 0.0)
        self.c_current = sd.get("c_current", self.c0)
        self.initial_lrs = sd["initial_lrs"]
        self._best_loss = sd.get("_best_loss")
        self._loss_plateau_count = sd.get("_loss_plateau_count", 0)
        self._in_escape = sd.get("_in_escape", False)
        self._escape_step_counter = sd.get("_escape_step_counter", 0)
        self._cooldown_counter = sd.get("_cooldown_counter", 0)
        self._surface_counter = sd.get("_surface_counter", 0)
        self._velocity_norm = sd.get("_velocity_norm", 0.0)
        self._integral_Z = sd.get("_integral_Z", 0.0)
        self._grad_norm_history = sd.get("_grad_norm_history", [])
        self._escape_events = sd.get("_escape_events", 0)
        self._noise_count = sd.get("_noise_count", 0)
        self.a_eff = sd.get("a_eff", self.a0)
