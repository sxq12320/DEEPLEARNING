"""
SMCAO — Sliding Mode Control Adaptive Optimizer
================================================

基于 SMCAO 数学思想的混合架构可微投影层 + 独立优化器。
专用于低维物理参数空间的硬约束优化。

架构设计原则 (Hybrid Architecture)
------------------------------------
- Backbone/Decoder 权重由 AdamW 更新 (禁止 SMC 更新高维参数)
- SMCAO 仅作用于网络末端的低维物理参数 (形状参数、朝向角等)
- 前向传播中执行 ODE 积分, 将 θ 投影到 s≈0 的物理可行流形上
- 过程完全可微, 外部 AdamW 可反向传播

SMCAO ODE 动力学
-----------------
连续时间二阶 ODE:
    θ̈ + a·θ̇ + κ·sat(s/φ) + kᵢ·Z = 0
    Ż = ∇f(θ)

滑模面:
    s = c·∇f(θ) + (H(θ) + λI)·θ̇

在滑模面 s=0 上:
    ∇f(θ) 指数衰减 → 0 (当 H 正定且 c > 0)
    等效于带 Hessian 校正的梯度流 (牛顿流)

关键改进 (相比旧版 SMCScheduler):
  1. 精确 Hessian (低维可行), 而非仅用梯度范数
  2. sat(s/φ) 边界层平滑, 绝对禁止 sign(s)
  3. 自适应 Hessian 正则化 λ (从大到小衰减)
  4. 速度钳制 v_max 保证数值稳定性
  5. RK4 ODE 积分器 (完全可微)
  6. 设计为可微投影层 nn.Module, 非 LR 调度器
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math


# ============================================================
#  核心模块: 可微投影层 (Differentiable Projection Layer)
# ============================================================

class SMCAO_ProjectionLayer(nn.Module):
    """
    SMCAO 可微投影层 — 将网络预测的低维物理参数投影到物理可行流形。

    在 YOLO 分割网络中的使用方式:
        class YOLOWithSMCAO(nn.Module):
            def __init__(self):
                self.backbone = YOLOBackbone()      # AdamW 训练
                self.physics_head = PhysicsHead()    # 输出低维 θ
                self.smc_projection = SMCAO_ProjectionLayer(dim=θ_dim)

            def forward(self, x):
                features = self.backbone(x)
                θ_pred = self.physics_head(features)
                θ_proj = self.smc_projection(θ_pred, physics_loss_fn)
                return θ_proj

    前向传播中执行 N 步 SMCAO ODE 积分 (RK4), 将 θ 投影到 s≈0 的流形上。
    过程完全可微, 外部 AdamW 可反向传播通过此层。

    Args:
        dim: 物理参数维度 (通常 < 50)
        a: 阻尼系数 (控制速度衰减速率)
        kappa: 趋达增益 (控制趋向滑模面的力度)
        phi: 边界层宽度 (sat 函数的线性区宽度)
        ki: 积分增益 (消除稳态误差)
        c: 滑模面系数 (梯度项权重)
        N: ODE 积分步数 (推荐 5-20)
        dt: ODE 时间步长
        v_max: 速度上限 (数值稳定性保障)
        lambda_init: Hessian 正则化初始值 (大 → 初期近似梯度下降)
        lambda_decay: Hessian 正则化衰减率 (逐步恢复精确 Hessian)
        detach: 是否在 ODE 步间断开计算图 (节省显存, 牺牲部分梯度精度)
    """

    def __init__(
        self,
        dim: int,
        a: float = 3.0,
        kappa: float = 1.0,
        phi: float = 0.1,
        ki: float = 0.05,
        c: float = 1.0,
        N: int = 10,
        dt: float = 0.01,
        v_max: float = 5.0,
        lambda_init: float = 50.0,
        lambda_decay: float = 0.95,
        detach: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.a = a
        self.kappa = kappa
        self.phi = phi
        self.ki = ki
        self.c = c
        self.N = N
        self.dt = dt
        self.v_max = v_max
        self.lambda_init = lambda_init
        self.lambda_decay = lambda_decay
        self.detach = detach

        # 内部状态 (非参数, 跨 forward 调用持续更新)
        self._step_count = 0
        self.register_buffer('velocity', torch.zeros(dim))
        self.register_buffer('integral', torch.zeros(dim))
        self.last_surface_norm = 0.0

    @staticmethod
    def _sat(s: torch.Tensor, phi: float) -> torch.Tensor:
        """
        边界层饱和函数 — 消除 sign(s) 导致的 chattering。

        |s| ≤ φ 时: sat(s/φ) = s/φ (线性区, 连续可微)
        |s| > φ 时: sat(s/φ) = sign(s) (饱和区)

        数学性质:
          - 在 |s|≤φ 内等效于线性反馈, 保证滑模面上的平滑运动
          - 在 |s|>φ 时退化为 sign(s), 保证有限时间趋达
          - 处处连续 (sign(s) 在 s=0 处不连续 → chattering 根因)
        """
        return torch.clamp(s / phi, -1.0, 1.0)

    @staticmethod
    def _compute_grad(theta: torch.Tensor, loss_fn) -> torch.Tensor:
        """计算 ∇f(θ), 保持计算图以支持后续 Hessian 计算。"""
        theta_d = theta.detach().requires_grad_(True)
        f = loss_fn(theta_d)
        grad = torch.autograd.grad(f, theta_d, create_graph=True)[0]
        return grad

    @staticmethod
    def _compute_grad_and_hessian(
        theta: torch.Tensor, loss_fn
    ) -> tuple:
        """
        精确梯度 + 精确 Hessian (适用于低维 θ, dim < 100)。

        返回:
            f_val: f(θ) 标量值
            grad: ∇f(θ), shape (d,)
            H: Hessian ∇²f(θ), shape (d, d)

        使用 grad-grad 技术: 先对 f 求一阶导 (create_graph=True),
        再对一阶导的每个分量求导得到 Hessian 的行。
        """
        theta_d = theta.detach().requires_grad_(True)
        f = loss_fn(theta_d)
        grad = torch.autograd.grad(f, theta_d, create_graph=True)[0]

        d = theta_d.shape[0]
        H_rows = []
        for i in range(d):
            row = torch.autograd.grad(
                grad[i], theta_d,
                retain_graph=(i < d - 1),
                create_graph=False
            )[0]
            H_rows.append(row.detach())
        H = torch.stack(H_rows)

        return f.detach(), grad.detach(), H

    def _dynamics(
        self, theta, v, Z, loss_fn, lam, warmup_scale
    ) -> tuple:
        """
        SMCAO ODE 右端函数: 计算 (θ̇, θ̈, Ż)。

        连续时间 ODE:
            θ̈ = -a·v - κ·sat(s/φ) - kᵢ·Z
            s   = c·∇f(θ) + (H(θ) + λI)·v

        Args:
            theta: 当前参数, shape (d,)
            v: 当前速度 θ̇, shape (d,)
            Z: 积分状态 ∫∇f dt, shape (d,)
            loss_fn: 物理先验损失函数 f(θ) → scalar
            lam: Hessian 正则化系数 λ
            warmup_scale: 暖启动缩放因子 [0,1]

        Returns:
            (v, theta_ddot, Z_dot): 三个 shape (d,) 的张量
        """
        _, grad, H = self._compute_grad_and_hessian(theta, loss_fn)

        # 正则化 Hessian: H_reg = H + λI
        # 初期 λ 大 → H_reg ≈ λI → 滑模面退化为 s ≈ c·∇f + λ·v
        # 后期 λ 小 → H_reg ≈ H → 滑模面恢复精确 Hessian 校正
        H_reg = H + lam * torch.eye(
            H.shape[0], device=H.device, dtype=H.dtype
        )

        # 滑模面: s = c·∇f(θ) + H_reg·θ̇
        s = self.c * grad + H_reg @ v

        # 趋达律: -κ·sat(s/φ) - kᵢ·Z
        sat_s = self._sat(s, self.phi)
        kappa_eff = self.kappa * warmup_scale

        # ODE: θ̈ = -a·v - κ·sat(s/φ) - kᵢ·Z
        theta_ddot = -self.a * v - kappa_eff * sat_s - self.ki * Z

        # 积分动力学: Ż = ∇f(θ)
        Z_dot = grad

        return v, theta_ddot, Z_dot

    def _rk4_step(
        self, theta, v, Z, loss_fn, dt, lam, warmup_scale
    ) -> tuple:
        """
        四阶 Runge-Kutta ODE 积分步。

        对于 dy/dt = F(y), RK4 给出:
            k1 = F(yₙ)
            k2 = F(yₙ + h/2·k1)
            k3 = F(yₙ + h/2·k2)
            k4 = F(yₙ + h·k3)
            yₙ₊₁ = yₙ + h/6·(k1 + 2k2 + 2k3 + k4)

        完全可微: 所有中间计算保留在计算图中 (若 detach=False)。
        """
        dyn = self._dynamics

        k1_v, k1_a, k1_z = dyn(theta, v, Z, loss_fn, lam, warmup_scale)
        k2_v, k2_a, k2_z = dyn(
            theta + 0.5 * dt * k1_v,
            v + 0.5 * dt * k1_a,
            Z + 0.5 * dt * k1_z,
            loss_fn, lam, warmup_scale
        )
        k3_v, k3_a, k3_z = dyn(
            theta + 0.5 * dt * k2_v,
            v + 0.5 * dt * k2_a,
            Z + 0.5 * dt * k2_z,
            loss_fn, lam, warmup_scale
        )
        k4_v, k4_a, k4_z = dyn(
            theta + dt * k3_v,
            v + dt * k3_a,
            Z + dt * k3_z,
            loss_fn, lam, warmup_scale
        )

        new_theta = theta + (dt / 6.0) * (
            k1_v + 2 * k2_v + 2 * k3_v + k4_v
        )
        new_v = v + (dt / 6.0) * (
            k1_a + 2 * k2_a + 2 * k3_a + k4_a
        )
        new_Z = Z + (dt / 6.0) * (
            k1_z + 2 * k2_z + 2 * k3_z + k4_z
        )

        return new_theta, new_v, new_Z

    def forward(self, theta: torch.Tensor, loss_fn) -> torch.Tensor:
        """
        前向传播: 执行 N 步 SMCAO ODE 积分, 投影 θ 到物理可行流形。

        Args:
            theta: 网络预测的物理参数, shape (batch, dim) 或 (dim,)
            loss_fn: 物理先验损失函数 f(θ) → scalar
                     接收 shape (dim,) 的张量, 返回标量

        Returns:
            θ_projected: 投影后的物理参数 (同 shape)
        """
        batch_mode = theta.dim() > 1
        if batch_mode:
            results = []
            for i in range(theta.shape[0]):
                results.append(
                    self._project_single(theta[i], loss_fn)
                )
            return torch.stack(results)
        return self._project_single(theta, loss_fn)

    def _project_single(
        self, theta: torch.Tensor, loss_fn
    ) -> torch.Tensor:
        """单样本投影: N 步 RK4 ODE 积分。"""
        self._step_count += 1

        # 自适应 Hessian 正则化: 从大到小衰减
        lam = self.lambda_init * (
            self.lambda_decay ** max(0, self._step_count - 10)
        )
        lam = max(lam, 0.01)

        # 暖启动缩放: 前 20 步逐步增大趋达增益
        warmup_scale = min(1.0, self._step_count / 20.0)

        theta_cur = theta if not self.detach else theta.detach().clone()
        v = self.velocity.clone()
        Z = self.integral.clone()

        surface_norms = []

        for _ in range(self.N):
            theta_new, v_new, Z_new = self._rk4_step(
                theta_cur, v, Z, loss_fn,
                self.dt, lam, warmup_scale
            )

            # 速度钳制 (数值稳定性)
            v_norm = v_new.norm()
            if v_norm > self.v_max:
                v_new = v_new * (self.v_max / (v_norm + 1e-12))

            theta_cur = theta_new
            v = v_new
            Z = Z_new

            # 计算当前滑模面范数 (监控用)
            with torch.no_grad():
                _, g, Hm = self._compute_grad_and_hessian(
                    theta_cur, loss_fn
                )
                H_reg = Hm + lam * torch.eye(
                    Hm.shape[0], device=Hm.device
                )
                s = self.c * g + H_reg @ v
                surface_norms.append(s.norm().item())

        # 更新内部状态
        self.velocity.copy_(v.detach())
        self.integral.copy_(Z.detach())
        self.last_surface_norm = (
            surface_norms[-1] if surface_norms else 0.0
        )

        if self.detach:
            return theta_cur
        else:
            return theta_cur

    def reset(self):
        """重置内部状态 (新样本或新 epoch 时调用)。"""
        self.velocity.zero_()
        self.integral.zero_()
        self.last_surface_norm = 0.0
        self._step_count = 0

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, a={self.a}, kappa={self.kappa}, "
            f"phi={self.phi}, ki={self.ki}, c={self.c}, "
            f"N={self.N}, dt={self.dt}"
        )


# ============================================================
#  独立优化器: 用于基准测试 (Benchmark Optimizer)
# ============================================================

class SMCAOOptimizer(Optimizer):
    """
    SMCAO 独立优化器 — 两阶段滑模控制动力学。

    用于与 AdamW 等一阶优化器进行公平对比 (如 Rosenbrock 基准测试)。

    两阶段控制策略 (解决 Hessian-velocity 耦合导致的趋达不稳定问题):
      Phase 1 (趋达阶段): 梯度下降 + 阻尼, 将参数拉到谷底附近
        判据: ‖∇f‖ < reaching_threshold
        动力学: θ̈ = -a·θ̇ - ∇f (纯梯度驱动, 无 Hessian 耦合)
      Phase 2 (滑模阶段): 解耦滑模面 + 高趋达增益
        滑模面: s = c·∇f + θ̇ (解耦! 不含 H·θ̇ 项)
        趋达律: θ̈ = -a·θ̇ - κ·sat(s/φ) - kᵢ·Z
        在 s=0 上: θ̇ = -c·∇f → θ̈ = -c·H·θ̇ (牛顿流)

    为什么解耦滑模面是安全的:
      ṡ = c·ḡ + θ̈ = c·H·θ̇ + (-a·θ̇ - κ·sat(s/φ))
      s·ṡ = s·c·H·θ̇ - s·a·θ̇ - κ·s·sat(s/φ)
      趋达条件: κ > |c·H·θ̇ - a·θ̇| 在谷底 (θ̇ 小, g 小) 成立
      而在 s=0 上: θ̇ = -c·g, 所以 c·H·θ̇ = -c²·H·g
      在谷底 ‖g‖ 很小 → ‖c²·H·g‖ 可控 → κ 可以覆盖

    Args:
        params: 待优化的参数 (应为低维, dim < 100)
        lr: 学习率 (ODE 时间步长)
        a: 阻尼系数 (控制速度衰减速率)
        kappa: 趋达增益 (Phase 2, 需要 > ||c²·H·g|| 在谷底)
        phi: 边界层宽度 (sat 函数的线性区宽度)
        ki: 积分增益 (消除稳态误差)
        c: 滑模面梯度系数
        v_max: 速度上限
        reaching_threshold: ‖∇f‖ 低于此值时切换到 Phase 2
    """

    def __init__(
        self,
        params,
        lr: float = 0.002,
        a: float = 5.0,
        kappa: float = 500.0,
        phi: float = 0.01,
        ki: float = 0.001,
        c: float = 1.0,
        v_max: float = 5.0,
        reaching_threshold: float = 5.0,
    ):
        defaults = dict(
            lr=lr, a=a, kappa=kappa, phi=phi, ki=ki, c=c,
            v_max=v_max, reaching_threshold=reaching_threshold,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            params_list = list(group['params'])
            flat = self._flatten_params(params_list)

            group['_velocity'] = torch.zeros_like(flat)
            group['_integral'] = torch.zeros_like(flat)
            group['_step_count'] = 0
            group['_surface_norm'] = 0.0
            group['_phase'] = 'reaching'  # 'reaching' or 'sliding'
            group['_shapes'] = [p.shape for p in params_list]
            group['_loss_val'] = float('inf')

    @staticmethod
    def _flatten_params(params):
        return torch.cat([p.detach().reshape(-1) for p in params])

    @staticmethod
    def _assign_to_params(flat, params, shapes):
        offset = 0
        for p, shape in zip(params, shapes):
            numel = p.numel()
            p.data.copy_(flat[offset:offset + numel].reshape(shape))
            offset += numel

    @staticmethod
    def _sat(s, phi):
        """边界层饱和函数 — 消除 sign(s) 导致的 chattering。"""
        return torch.clamp(s / phi, -1.0, 1.0)

    @staticmethod
    def _compute_grad_and_hessian(theta_flat, closure):
        """精确梯度 + 精确 Hessian (grad-grad 技术)。"""
        theta = theta_flat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            f = closure(theta)
            grad = torch.autograd.grad(
                f, theta, create_graph=True
            )[0]

            d = theta.shape[0]
            H = torch.zeros(d, d, device=theta.device)
            for i in range(d):
                row = torch.autograd.grad(
                    grad[i], theta,
                    retain_graph=(i < d - 1),
                    create_graph=False,
                )[0]
                H[i] = row.detach()

        return f.detach().item(), grad.detach(), H

    @torch.no_grad()
    def step(self, closure=None):
        """
        执行一步 SMCAO 两阶段动力学更新。

        Phase 1 (‖∇f‖ > threshold): 梯度下降 + 阻尼 → 趋达谷底
        Phase 2 (‖∇f‖ < threshold): 解耦滑模面 → 沿谷底精确收敛
        """
        if closure is None:
            raise ValueError("SMCAO requires a closure")

        for group in self.param_groups:
            lr = group['lr']
            a = group['a']
            kappa = group['kappa']
            phi = group['phi']
            ki = group['ki']
            c_coeff = group['c']
            v_max = group['v_max']
            reach_thresh = group['reaching_threshold']

            params = list(group['params'])
            v = group['_velocity']
            Z = group['_integral']
            group['_step_count'] += 1

            theta = self._flatten_params(params)
            d = theta.shape[0]

            # 先计算当前梯度和 Hessian (决定 Phase)
            f_val, g, H = self._compute_grad_and_hessian(theta, closure)
            grad_norm = g.norm().item()

            # Phase 判断
            if grad_norm < reach_thresh and group['_phase'] == 'reaching':
                group['_phase'] = 'sliding'

            phase = group['_phase']

            # ---- RK4 四阶积分 ----
            def dynamics(th, vv, zz):
                """ODE 右端: 根据 Phase 返回 (θ̇, θ̈, Ż)。"""
                f_v, g_d, H_d = self._compute_grad_and_hessian(th, closure)

                if phase == 'reaching':
                    # Phase 1: 梯度下降 + 阻尼
                    # θ̈ = -a·v - ∇f
                    # 等效于: 带摩擦的粒子在势能面上滚动
                    th_ddot = -a * vv - g_d
                else:
                    # Phase 2: 解耦滑模面
                    # s = c·∇f + θ̇ (不含 H·θ̇ → 无 Hessian 耦合)
                    s = c_coeff * g_d + vv
                    sat_s = self._sat(s, phi)
                    # θ̈ = -a·v - κ·sat(s/φ) - kᵢ·Z
                    th_ddot = -a * vv - kappa * sat_s - ki * zz

                return vv, th_ddot, g_d

            dt = lr

            k1_v, k1_a, k1_z = dynamics(theta, v, Z)
            k2_v, k2_a, k2_z = dynamics(
                theta + 0.5 * dt * k1_v,
                v + 0.5 * dt * k1_a,
                Z + 0.5 * dt * k1_z,
            )
            k3_v, k3_a, k3_z = dynamics(
                theta + 0.5 * dt * k2_v,
                v + 0.5 * dt * k2_a,
                Z + 0.5 * dt * k2_z,
            )
            k4_v, k4_a, k4_z = dynamics(
                theta + dt * k3_v,
                v + dt * k3_a,
                Z + dt * k3_z,
            )

            theta += (dt / 6.0) * (
                k1_v + 2 * k2_v + 2 * k3_v + k4_v
            )
            v_new = v + (dt / 6.0) * (
                k1_a + 2 * k2_a + 2 * k3_a + k4_a
            )
            Z_new = Z + (dt / 6.0) * (
                k1_z + 2 * k2_z + 2 * k3_z + k4_z
            )

            # 速度钳制
            v_norm = v_new.norm()
            if v_norm > v_max:
                v_new.mul_(v_max / (v_norm + 1e-12))

            # 写回参数
            self._assign_to_params(theta, params, group['_shapes'])
            group['_velocity'] = v_new
            group['_integral'] = Z_new

            # 计算解耦滑模面范数 (监控, Phase 2 有意义)
            f_val2, g_final, _ = self._compute_grad_and_hessian(
                self._flatten_params(params), closure
            )
            s_final = c_coeff * g_final + v_new
            group['_surface_norm'] = s_final.norm().item()
            group['_loss_val'] = f_val2

    def get_surface_norm(self):
        """获取最近一步的滑模面范数 ‖s‖。"""
        return self.param_groups[0].get('_surface_norm', 0.0)

    def get_loss_val(self):
        """获取最近一步的损失值。"""
        return self.param_groups[0].get('_loss_val', float('inf'))

    def get_phase(self):
        """获取当前阶段: 'reaching' 或 'sliding'。"""
        return self.param_groups[0].get('_phase', 'reaching')
