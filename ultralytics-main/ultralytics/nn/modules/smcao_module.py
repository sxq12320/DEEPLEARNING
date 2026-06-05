"""
SMCAO V2.1 — Sliding Mode Control Adaptive Optimizer (Deep Refactored)
=======================================================================

V2.1 核心改进 (在 V1 基础上):
  1. 自适应 λ: 基于 ‖∇f‖ 的 Sigmoid 调度 (控制 Hessian 正则化强度)
  2. 自适应 κ: 基于 ‖s‖ 的放大机制 (钉在滑模面上)
  3. Hessian 缓存: 每步仅 1 次 Hessian, RK4 子步复用 (4x→1x)
  4. Newton 精修模式: 接近最优时切换到 H⁻¹·∇f 直接步 (突破精度天花板)
  5. Hessian 对称化: H = 0.5·(H + H^T)

架构: 两阶段 + Newton 精修 (Phase 1 → Phase 2 → Phase 3)
  Phase 1 (趋达): ‖∇f‖ > threshold → 梯度下降 + 阻尼
  Phase 2 (滑模): threshold > ‖∇f‖ > newton_threshold → SMCAO ODE
  Phase 3 (Newton): ‖∇f‖ < newton_threshold → H⁻¹·∇f 直接步
"""

import torch
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import math


def _sigmoid_schedule(x: float, x_mid: float, beta: float) -> float:
    """σ(x) = 1 / (1 + exp(-β·(x - x_mid)))"""
    z = beta * (x - x_mid)
    z = max(-20.0, min(20.0, z))  # 防溢出
    return 1.0 / (1.0 + math.exp(-z))


def _symmetrize(H: torch.Tensor) -> torch.Tensor:
    """H_sym = 0.5·(H + H^T)"""
    return 0.5 * (H + H.t())


# ============================================================
#  可微投影层
# ============================================================

class SMCAO_ProjectionLayer(nn.Module):
    """
    SMCAO V2.1 可微投影层。

    在 YOLO 中的用法:
        θ_proj = SMCAO_ProjectionLayer(dim)(θ_pred, physics_loss_fn)
    """

    def __init__(
        self,
        dim: int,
        a: float = 5.0,
        kappa_base: float = 1.0,
        kappa_alpha: float = 0.5,
        phi: float = 0.1,
        ki: float = 0.05,
        c: float = 1.0,
        N: int = 10,
        dt: float = 0.01,
        v_max: float = 10.0,
        lambda_min: float = 1e-4,
        lambda_max: float = 50.0,
        lambda_beta: float = 0.5,
        lambda_g_mid: float = 5.0,
        integrator: str = 'rk4',
        detach: bool = True,
    ):
        super().__init__()
        self.dim = dim
        self.a = a
        self.kappa_base = kappa_base
        self.kappa_alpha = kappa_alpha
        self.phi = phi
        self.ki = ki
        self.c = c
        self.N = N
        self.dt = dt
        self.v_max = v_max
        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.lambda_beta = lambda_beta
        self.lambda_g_mid = lambda_g_mid
        self.integrator = integrator
        self.detach = detach

        self._step_count = 0
        self.register_buffer('velocity', torch.zeros(dim))
        self.register_buffer('integral', torch.zeros(dim))
        self.last_surface_norm = 0.0

    @staticmethod
    def _sat(s, phi):
        return torch.clamp(s / phi, -1.0, 1.0)

    @staticmethod
    def _compute_grad(theta, loss_fn):
        theta_d = theta.detach().requires_grad_(True)
        f = loss_fn(theta_d)
        grad = torch.autograd.grad(f, theta_d, create_graph=True)[0]
        return grad

    @staticmethod
    def _compute_grad_and_hessian(theta, loss_fn):
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
        H = _symmetrize(torch.stack(H_rows))
        return f.detach(), grad.detach(), H

    def _adaptive_lambda(self, grad_norm):
        sig = _sigmoid_schedule(grad_norm, self.lambda_g_mid, self.lambda_beta)
        return self.lambda_min + (self.lambda_max - self.lambda_min) * sig

    def _adaptive_kappa(self, s_norm):
        return self.kappa_base * (1.0 + self.kappa_alpha / (self.phi + s_norm))

    def forward(self, theta, loss_fn):
        if theta.dim() > 1:
            return torch.stack([
                self._project_single(theta[i], loss_fn)
                for i in range(theta.shape[0])
            ])
        return self._project_single(theta, loss_fn)

    def _project_single(self, theta, loss_fn):
        self._step_count += 1
        theta_cur = theta if not self.detach else theta.detach().clone()
        v = self.velocity.clone()
        Z = self.integral.clone()

        for _ in range(self.N):
            _, grad_step, H_step = self._compute_grad_and_hessian(
                theta_cur, loss_fn
            )
            lam = self._adaptive_lambda(grad_step.norm().item())

            if self.integrator == 'heun':
                theta_cur, v, Z = self._heun_step(
                    theta_cur, v, Z, loss_fn, H_step, lam
                )
            else:
                theta_cur, v, Z = self._rk4_step(
                    theta_cur, v, Z, loss_fn, H_step, lam
                )

            v_norm = v.norm()
            if v_norm > self.v_max:
                v = v * (self.v_max / (v_norm + 1e-12))

        self.velocity.copy_(v.detach())
        self.integral.copy_(Z.detach())

        with torch.no_grad():
            s = self.c * self._compute_grad(theta_cur, loss_fn) + v
            self.last_surface_norm = s.norm().item()

        return theta_cur

    def _dynamics_cached(self, theta, v, Z, loss_fn, H_cached, lam):
        grad = self._compute_grad(theta, loss_fn)
        s = self.c * grad + v
        s_norm = s.norm().item()
        kappa_eff = self._adaptive_kappa(s_norm)
        sat_s = self._sat(s, self.phi)
        theta_ddot = -self.a * v - kappa_eff * sat_s - self.ki * Z
        return v, theta_ddot, grad

    def _heun_step(self, theta, v, Z, loss_fn, H_cached, lam):
        dyn = self._dynamics_cached
        k1_v, k1_a, k1_z = dyn(theta, v, Z, loss_fn, H_cached, lam)
        k2_v, k2_a, k2_z = dyn(
            theta + self.dt * k1_v, v + self.dt * k1_a,
            Z + self.dt * k1_z, loss_fn, H_cached, lam
        )
        return (
            theta + 0.5 * self.dt * (k1_v + k2_v),
            v + 0.5 * self.dt * (k1_a + k2_a),
            Z + 0.5 * self.dt * (k1_z + k2_z),
        )

    def _rk4_step(self, theta, v, Z, loss_fn, H_cached, lam):
        dyn = self._dynamics_cached
        dt = self.dt
        k1_v, k1_a, k1_z = dyn(theta, v, Z, loss_fn, H_cached, lam)
        k2_v, k2_a, k2_z = dyn(
            theta + 0.5*dt*k1_v, v + 0.5*dt*k1_a,
            Z + 0.5*dt*k1_z, loss_fn, H_cached, lam
        )
        k3_v, k3_a, k3_z = dyn(
            theta + 0.5*dt*k2_v, v + 0.5*dt*k2_a,
            Z + 0.5*dt*k2_z, loss_fn, H_cached, lam
        )
        k4_v, k4_a, k4_z = dyn(
            theta + dt*k3_v, v + dt*k3_a,
            Z + dt*k3_z, loss_fn, H_cached, lam
        )
        w = dt / 6.0
        return (
            theta + w * (k1_v + 2*k2_v + 2*k3_v + k4_v),
            v + w * (k1_a + 2*k2_a + 2*k3_a + k4_a),
            Z + w * (k1_z + 2*k2_z + 2*k3_z + k4_z),
        )

    def reset(self):
        self.velocity.zero_()
        self.integral.zero_()
        self.last_surface_norm = 0.0
        self._step_count = 0


# ============================================================
#  独立优化器 (V2.1: 含 Newton 精修模式)
# ============================================================

class SMCAOOptimizer(Optimizer):
    """
    SMCAO V2.1 独立优化器 — 三阶段控制。

    Phase 1 (趋达): ‖∇f‖ > reaching_threshold → 梯度下降+阻尼
    Phase 2 (滑模): reaching > ‖∇f‖ > newton_threshold → SMCAO ODE
    Phase 3 (Newton): ‖∇f‖ < newton_threshold → θ ← θ - lr_n·(H+λI)⁻¹·∇f

    V2.1 改进:
      - 自适应 λ: sigmoid(β·(‖g‖-g_mid)) 调度
      - 自适应 κ: κ_base·(1 + α/(φ + ‖s‖))
      - Hessian 缓存: 步初 1 次, 子步复用
      - Newton 精修: 接近最优时直接解 (H+λI)·Δθ = -∇f
      - Hessian 对称化
    """

    def __init__(
        self,
        params,
        lr: float = 0.002,
        a: float = 5.0,
        kappa_base: float = 200.0,
        kappa_alpha: float = 2.0,
        phi: float = 0.1,
        ki: float = 0.001,
        c: float = 1.0,
        v_max: float = 10.0,
        reaching_threshold: float = 10.0,
        newton_threshold: float = 0.5,
        newton_lr: float = 0.8,
        lambda_min: float = 1e-4,
        lambda_max: float = 50.0,
        lambda_beta: float = 0.3,
        lambda_g_mid: float = 5.0,
        integrator: str = 'rk4',
    ):
        defaults = dict(
            lr=lr, a=a, kappa_base=kappa_base, kappa_alpha=kappa_alpha,
            phi=phi, ki=ki, c=c, v_max=v_max,
            reaching_threshold=reaching_threshold,
            newton_threshold=newton_threshold,
            newton_lr=newton_lr,
            lambda_min=lambda_min, lambda_max=lambda_max,
            lambda_beta=lambda_beta, lambda_g_mid=lambda_g_mid,
            integrator=integrator,
        )
        super().__init__(params, defaults)

        for group in self.param_groups:
            params_list = list(group['params'])
            flat = self._flatten_params(params_list)
            group['_velocity'] = torch.zeros_like(flat)
            group['_integral'] = torch.zeros_like(flat)
            group['_step_count'] = 0
            group['_surface_norm'] = 0.0
            group['_kappa_eff'] = kappa_base
            group['_lambda'] = lambda_max
            group['_phase'] = 'reaching'
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
        return torch.clamp(s / phi, -1.0, 1.0)

    @staticmethod
    def _compute_grad(theta_flat, closure):
        theta = theta_flat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            f = closure(theta)
            grad = torch.autograd.grad(f, theta, create_graph=False)[0]
        return f.detach().item(), grad.detach()

    @staticmethod
    def _compute_grad_and_hessian(theta_flat, closure):
        theta = theta_flat.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            f = closure(theta)
            grad = torch.autograd.grad(f, theta, create_graph=True)[0]
            d = theta.shape[0]
            H = torch.zeros(d, d, device=theta.device)
            for i in range(d):
                row = torch.autograd.grad(
                    grad[i], theta,
                    retain_graph=(i < d - 1),
                    create_graph=False,
                )[0]
                H[i] = row.detach()
        H = _symmetrize(H)
        return f.detach().item(), grad.detach(), H

    def _adaptive_lambda(self, grad_norm, group):
        sig = _sigmoid_schedule(
            grad_norm, group['lambda_g_mid'], group['lambda_beta']
        )
        return group['lambda_min'] + (
            group['lambda_max'] - group['lambda_min']
        ) * sig

    def _adaptive_kappa(self, s_norm, group):
        return group['kappa_base'] * (
            1.0 + group['kappa_alpha'] / (group['phi'] + s_norm)
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is None:
            raise ValueError("SMCAO requires a closure")

        for group in self.param_groups:
            lr = group['lr']
            a = group['a']
            phi = group['phi']
            ki = group['ki']
            c_coeff = group['c']
            v_max = group['v_max']
            reach_thresh = group['reaching_threshold']
            newton_thresh = group['newton_threshold']
            newton_lr = group['newton_lr']
            integrator = group['integrator']

            params = list(group['params'])
            v = group['_velocity']
            Z = group['_integral']
            group['_step_count'] += 1

            theta = self._flatten_params(params)
            d = theta.shape[0]

            # ── V2.1 极速早停: 已收敛到高精度 → 跳过全部 Hessian 计算 ──
            prev_loss = group['_loss_val']
            if (group['_phase'] == 'newton'
                    and prev_loss != float('inf')
                    and prev_loss < 1e-5):
                # 仅评估 loss (廉价), 确认仍在极小值附近
                f_check = closure(theta).item()
                if abs(f_check - prev_loss) < 1e-12:
                    group['_surface_norm'] = 0.0
                    group['_kappa_eff'] = 0.0
                    group['_step_count'] = group['_step_count']  # already incremented
                    continue

            # 步初: 计算 Hessian + 梯度 (1 次)
            f_val, g, H = self._compute_grad_and_hessian(theta, closure)
            grad_norm = g.norm().item()

            # V2: 自适应 λ
            lam = self._adaptive_lambda(grad_norm, group)
            group['_lambda'] = lam

            # Phase 判断 (三阶段)
            if grad_norm < newton_thresh:
                group['_phase'] = 'newton'
            elif grad_norm < reach_thresh and group['_phase'] == 'reaching':
                group['_phase'] = 'sliding'
            phase = group['_phase']

            if phase == 'newton':
                # ==============================================
                #  Phase 3: Newton 精修
                # ==============================================
                prev_loss = group['_loss_val']
                # 已收敛到高精度 → 跳过 (节省 Hessian)
                if (prev_loss != float('inf')
                        and abs(f_val - prev_loss) < 1e-12
                        and f_val < 1e-5):
                    group['_surface_norm'] = 0.0
                    group['_kappa_eff'] = 0.0
                    continue

                # Newton 阶段: λ 极小 (不干扰精确 Hessian)
                lam_n = max(lam * 0.001, 1e-8)  # V2.1: Newton 阶段 λ → 0
                H_reg = H + lam_n * torch.eye(d, device=H.device)
                try:
                    delta = torch.linalg.solve(H_reg, -g)
                except RuntimeError:
                    delta = -g / (lam + 1e-8)

                # 简单回溯线搜索: 只要 loss 下降就接受
                alpha = newton_lr
                f_trial = f_val
                for _ in range(20):
                    theta_trial = theta + alpha * delta
                    f_trial, _ = self._compute_grad(theta_trial, closure)
                    if f_trial < f_val:
                        break
                    alpha *= 0.5

                # 线搜索失败 → 退回梯度步
                if f_trial >= f_val:
                    alpha = lr
                    theta_trial = theta - alpha * g
                    f_trial, _ = self._compute_grad(theta_trial, closure)
                    theta = theta_trial
                else:
                    theta = theta + alpha * delta

                v.zero_()
                Z.zero_()
                self._assign_to_params(theta, params, group['_shapes'])
                group['_velocity'] = v
                group['_integral'] = Z
                group['_surface_norm'] = 0.0
                group['_kappa_eff'] = 0.0
                group['_loss_val'] = f_trial
                continue

            # ==============================================
            #  Phase 1/2: SMCAO ODE (Hessian 缓存)
            # ==============================================
            def dynamics_cached(th, vv, zz):
                f_v, g_d = self._compute_grad(th, closure)
                if phase == 'reaching':
                    th_ddot = -a * vv - g_d
                else:
                    s = c_coeff * g_d + vv
                    s_norm = s.norm().item()
                    kappa_eff = self._adaptive_kappa(s_norm, group)
                    group['_kappa_eff'] = kappa_eff
                    sat_s = self._sat(s, phi)
                    th_ddot = -a * vv - kappa_eff * sat_s - ki * zz
                return vv, th_ddot, g_d

            dt = lr

            if integrator == 'heun':
                k1_v, k1_a, k1_z = dynamics_cached(theta, v, Z)
                k2_v, k2_a, k2_z = dynamics_cached(
                    theta + dt*k1_v, v + dt*k1_a, Z + dt*k1_z
                )
                theta += 0.5*dt*(k1_v + k2_v)
                v_new = v + 0.5*dt*(k1_a + k2_a)
                Z_new = Z + 0.5*dt*(k1_z + k2_z)
            else:
                k1_v, k1_a, k1_z = dynamics_cached(theta, v, Z)
                k2_v, k2_a, k2_z = dynamics_cached(
                    theta + 0.5*dt*k1_v, v + 0.5*dt*k1_a,
                    Z + 0.5*dt*k1_z
                )
                k3_v, k3_a, k3_z = dynamics_cached(
                    theta + 0.5*dt*k2_v, v + 0.5*dt*k2_a,
                    Z + 0.5*dt*k2_z
                )
                k4_v, k4_a, k4_z = dynamics_cached(
                    theta + dt*k3_v, v + dt*k3_a,
                    Z + dt*k3_z
                )
                w = dt / 6.0
                theta += w*(k1_v + 2*k2_v + 2*k3_v + k4_v)
                v_new = v + w*(k1_a + 2*k2_a + 2*k3_a + k4_a)
                Z_new = Z + w*(k1_z + 2*k2_z + 2*k3_z + k4_z)

            # 速度钳制
            v_norm = v_new.norm()
            if v_norm > v_max:
                v_new.mul_(v_max / (v_norm + 1e-12))

            self._assign_to_params(theta, params, group['_shapes'])
            group['_velocity'] = v_new
            group['_integral'] = Z_new

            # V2.1 优化: 复用步初梯度 (避免额外 Hessian+梯度计算)
            s_final = c_coeff * g + v_new  # 用步初梯度近似
            s_norm_final = s_final.norm().item()
            group['_surface_norm'] = s_norm_final
            group['_kappa_eff'] = self._adaptive_kappa(s_norm_final, group)
            group['_loss_val'] = f_val

    def get_surface_norm(self):
        return self.param_groups[0].get('_surface_norm', 0.0)

    def get_loss_val(self):
        return self.param_groups[0].get('_loss_val', float('inf'))

    def get_phase(self):
        return self.param_groups[0].get('_phase', 'reaching')

    def get_kappa_eff(self):
        return self.param_groups[0].get('_kappa_eff', 0.0)

    def get_lambda(self):
        return self.param_groups[0].get('_lambda', 0.0)
