"""
SMCAO V2.1 vs AdamW — Rosenbrock 极限精度基准测试
===================================================

V2.1 测试目标:
    1. 精度突破 1e-5 (Newton 精修模式)
    2. 耗时 < 6s (Hessian 缓存 + RK4)
    3. 滑模面贴合度 > 95% (Phase 2 期间)
    4. 5 子图可视化 (含 κ_eff 曲线)
"""

import sys, os, time
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ultralytics.nn.modules.smcao_module import SMCAOOptimizer


def rosenbrock(theta):
    x, y = theta[0], theta[1]
    return (1.0 - x)**2 + 100.0*(y - x**2)**2

GLOBAL_OPT = np.array([1.0, 1.0])


def run_adamw(init_pos, lr, steps, seed=42):
    torch.manual_seed(seed)
    xy = nn.Parameter(torch.tensor([float(init_pos[0]), float(init_pos[1])]))
    opt = torch.optim.AdamW([xy], lr=lr, betas=(0.9, 0.999), weight_decay=1e-4)
    traj = [xy.detach().numpy().copy()]
    losses = []
    t0 = time.time()
    for _ in range(steps):
        opt.zero_grad()
        loss = rosenbrock(xy)
        loss.backward()
        opt.step()
        traj.append(xy.detach().numpy().copy())
        losses.append(loss.item())
    return {'trajectory': np.array(traj), 'losses': np.array(losses),
            'time_sec': time.time() - t0}


def run_smcao(init_pos, lr, steps, seed=42, **kw):
    torch.manual_seed(seed)
    xy = nn.Parameter(torch.tensor([float(init_pos[0]), float(init_pos[1])]))
    opt = SMCAOOptimizer(
        [xy], lr=lr,
        a=kw.get('a', 5.0),
        kappa_base=kw.get('kappa_base', 200.0),
        kappa_alpha=kw.get('kappa_alpha', 2.0),
        phi=kw.get('phi', 0.1),
        ki=kw.get('ki', 0.001),
        c=kw.get('c', 1.0),
        v_max=kw.get('v_max', 10.0),
        reaching_threshold=kw.get('reaching_threshold', 10.0),
        newton_threshold=kw.get('newton_threshold', 0.5),
        newton_lr=kw.get('newton_lr', 0.8),
        lambda_min=kw.get('lambda_min', 1e-4),
        lambda_max=kw.get('lambda_max', 50.0),
        lambda_beta=kw.get('lambda_beta', 0.3),
        lambda_g_mid=kw.get('lambda_g_mid', 5.0),
        integrator=kw.get('integrator', 'rk4'),
    )

    traj = [xy.detach().numpy().copy()]
    losses, surface_norms, phases, kappa_effs, lambdas = [], [], [], [], []

    def closure(th):
        return rosenbrock(th)

    t0 = time.time()
    for i in range(steps):
        opt.step(closure)
        with torch.no_grad():
            pos = xy.detach().numpy().copy()
            loss_val = rosenbrock(xy).item()
        traj.append(pos)
        losses.append(loss_val)
        surface_norms.append(opt.get_surface_norm())
        phases.append(opt.get_phase())
        kappa_effs.append(opt.get_kappa_eff())
        lambdas.append(opt.get_lambda())

        if (i+1) % 1000 == 0:
            print(f"  SMCAO {i+1}/{steps}: loss={loss_val:.2e}, "
                  f"‖s‖={surface_norms[-1]:.2e}, phase={phases[-1]}, "
                  f"κ_eff={kappa_effs[-1]:.1f}, λ={lambdas[-1]:.4f}")
    elapsed = time.time() - t0

    phase_counts = {}
    for p in phases:
        phase_counts[p] = phase_counts.get(p, 0) + 1
    print(f"  Phase: {phase_counts}")

    return {
        'trajectory': np.array(traj), 'losses': np.array(losses),
        'surface_norms': np.array(surface_norms), 'phases': phases,
        'kappa_effs': np.array(kappa_effs), 'lambdas': np.array(lambdas),
        'time_sec': elapsed,
    }


def compute_metrics(result, name, phi=0.1):
    losses = result['losses']
    final_pos = result['trajectory'][-1]
    final_loss = float(losses[-1])
    dist = float(np.linalg.norm(final_pos - GLOBAL_OPT))

    s01 = s001 = s1e5 = -1
    for i, l in enumerate(losses):
        if l < 0.1 and s01 == -1: s01 = i + 1
        if l < 0.01 and s001 == -1: s001 = i + 1
        if l < 1e-5 and s1e5 == -1: s1e5 = i + 1

    last1k = losses[-1000:]
    std = float(np.std(last1k))
    mean = float(np.mean(last1k))

    # 滑模面贴合度 (Phase 2 期间 ‖s‖<φ 的比例)
    sm_ratio = -1.0
    if 'surface_norms' in result and 'phases' in result:
        sn = result['surface_norms']
        ph = result['phases']
        sliding_sn = [sn[i] for i in range(len(sn)) if ph[i] == 'sliding']
        if sliding_sn:
            sm_ratio = float(np.sum(np.array(sliding_sn) < phi) / len(sliding_sn))

    return {
        'name': name, 'final_loss': final_loss,
        'min_loss': float(np.min(losses)), 'dist_to_opt': dist,
        'final_pos': final_pos, 'steps_to_01': s01, 'steps_to_001': s001,
        'steps_to_1e5': s1e5, 'stability_std': std, 'stability_mean': mean,
        'sliding_mode_ratio': sm_ratio, 'time_sec': result['time_sec'],
    }


def plot_comparison(smcao_r, adamw_r, sm_m, aw_m, save_path, phi=0.1):
    fig, axes = plt.subplots(2, 3, figsize=(22, 11))
    fig.suptitle('SMCAO V2.1 vs AdamW — Rosenbrock Extreme Benchmark',
                 fontsize=16, fontweight='bold', y=0.98)

    # (1) 2D Trajectory
    ax = axes[0, 0]
    xx, yy = np.linspace(-1.5, 1.5, 500), np.linspace(-0.5, 2.0, 500)
    X, Y = np.meshgrid(xx, yy)
    Z = np.log10((1-X)**2 + 100*(Y-X**2)**2 + 1e-10)
    cf = ax.contourf(X, Y, Z, levels=50, cmap='viridis', alpha=0.8)
    plt.colorbar(cf, ax=ax, shrink=0.8)
    for label, traj, col in [('AdamW', adamw_r['trajectory'], '#d62728'),
                              ('SMCAO', smcao_r['trajectory'], '#1f77b4')]:
        ax.plot(traj[:, 0], traj[:, 1], '-', color=col, lw=0.9, alpha=0.8, label=label)
        ax.plot(traj[0, 0], traj[0, 1], 's', color=col, ms=8, mec='k', mew=1.5, zorder=5)
        ax.plot(traj[-1, 0], traj[-1, 1], 'X', color=col, ms=10, mec='k', mew=1.5, zorder=5)
    ax.plot(1, 1, '*', color='gold', ms=16, mec='k', mew=1.5, zorder=6, label='Opt')
    ax.set_xlim(-1.5, 1.5); ax.set_ylim(-0.5, 2.0)
    ax.set_title('(1) 2D Trajectory'); ax.legend(fontsize=8)

    # (2) Loss (Log)
    ax = axes[0, 1]
    sr = np.arange(len(adamw_r['losses']))
    ax.semilogy(sr, adamw_r['losses'], '-', color='#d62728', lw=1, label='AdamW')
    ax.semilogy(sr, smcao_r['losses'], '-', color='#1f77b4', lw=1, label='SMCAO')
    ax.axhline(1e-5, color='purple', ls=':', lw=0.8, alpha=0.6)
    ax.text(sr[-1]*0.98, 1.2e-5, '1e-5', fontsize=7, color='purple')
    # Newton phase 标注
    if 'phases' in smcao_r:
        ph = smcao_r['phases']
        newton_start = None
        for i in range(len(ph)):
            if ph[i] == 'newton' and (i == 0 or ph[i-1] != 'newton'):
                newton_start = i; break
        if newton_start is not None:
            ax.axvline(newton_start, color='orange', ls='--', lw=1.5, alpha=0.7,
                       label=f'Newton@{newton_start}')
    ax.set_title('(2) Loss Convergence (Log)'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (3) Last 1000 Steps
    ax = axes[0, 2]
    zs = max(0, len(adamw_r['losses']) - 1000)
    zs_arr = np.arange(zs, len(adamw_r['losses']))
    ax.plot(zs_arr, adamw_r['losses'][zs:], '-', color='#d62728', lw=0.8,
            label=f'AdamW std={aw_m["stability_std"]:.2e}')
    ax.plot(zs_arr, smcao_r['losses'][zs:], '-', color='#1f77b4', lw=0.8,
            label=f'SMCAO std={sm_m["stability_std"]:.2e}')
    ax.set_title('(3) Last 1000 Steps'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (4) Sliding Surface
    ax = axes[1, 0]
    sn = smcao_r['surface_norms']
    ax.semilogy(np.arange(len(sn)), np.maximum(sn, 1e-15), '-', color='#1f77b4', lw=1, label='‖s(t)‖')
    ax.axhline(phi, color='green', ls='--', lw=1, label=f'φ={phi}')
    if 'phases' in smcao_r:
        ph = smcao_r['phases']
        for i in range(1, len(ph)):
            if ph[i] == 'sliding' and ph[i-1] == 'reaching':
                ax.axvline(i, color='orange', ls='--', lw=1.5, label=f'Sliding@{i}')
                break
            if ph[i] == 'newton' and ph[i-1] != 'newton':
                ax.axvline(i, color='red', ls=':', lw=1.5, label=f'Newton@{i}')
                break
    ax.set_title('(4) Sliding Surface ‖s(t)‖'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # (5) κ_eff
    ax = axes[1, 1]
    ke = smcao_r['kappa_effs']
    ax.plot(np.arange(len(ke)), ke, '-', color='#9467bd', lw=1, label='κ_eff(t)')
    ax.axhline(sm_m.get('kappa_base', 200), color='gray', ls=':', lw=0.8, label='κ_base')
    ax.set_title('(5) Adaptive κ_eff'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    # (6) λ
    ax = axes[1, 2]
    la = smcao_r['lambdas']
    ax.semilogy(np.arange(len(la)), np.maximum(la, 1e-10), '-', color='#2ca02c', lw=1, label='λ(t)')
    ax.set_title('(6) Adaptive λ'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[可视化] 已保存: {save_path}")


def print_metrics(sm, aw):
    print("\n" + "=" * 78)
    print("  SMCAO V2.1 vs AdamW — 核心指标对比")
    print("=" * 78)
    print(f"\n  [最终精度]")
    print(f"    Final Loss:    SMCAO={sm['final_loss']:.6e}  AdamW={aw['final_loss']:.6e}")
    print(f"    Min Loss:      SMCAO={sm['min_loss']:.6e}  AdamW={aw['min_loss']:.6e}")
    print(f"    Dist→(1,1):    SMCAO={sm['dist_to_opt']:.6e}  AdamW={aw['dist_to_opt']:.6e}")
    print(f"\n  [收敛速度]")
    for label, sk, ak in [("Loss<0.1", 'steps_to_01', 'steps_to_01'),
                           ("Loss<0.01", 'steps_to_001', 'steps_to_001'),
                           ("Loss<1e-5", 'steps_to_1e5', 'steps_to_1e5')]:
        sv = str(sm[sk]) if sm[sk] != -1 else "N/A"
        av = str(aw[ak]) if aw[ak] != -1 else "N/A"
        print(f"    {label:12s}: SMCAO={sv:>6s}  AdamW={av:>6s}")
    print(f"\n  [训练稳定性]")
    ratio = aw['stability_std'] / max(sm['stability_std'], 1e-30)
    print(f"    Std:    SMCAO={sm['stability_std']:.4e}  AdamW={aw['stability_std']:.4e}  ratio={ratio:.1f}x")
    print(f"\n  [V2 新增: 滑模面贴合度 (Phase 2 期间)]")
    r = sm['sliding_mode_ratio']
    if r >= 0:
        print(f"    ‖s‖<φ 比例: {r*100:.1f}%  {'✓ >95%' if r > 0.95 else '✗ <95%'}")
    else:
        print(f"    N/A (无 Phase 2 步)")
    print(f"\n  [耗时]")
    print(f"    SMCAO: {sm['time_sec']:.2f}s  AdamW: {aw['time_sec']:.2f}s")

    print(f"\n  [综合评价]")
    wins = 0; total = 7
    checks = [
        ("Final Loss 更低", sm['final_loss'] < aw['final_loss']),
        ("Dist→(1,1) 更近", sm['dist_to_opt'] < aw['dist_to_opt']),
        ("Loss<0.1 更快", (sm['steps_to_01'] < aw['steps_to_01']
                           if sm['steps_to_01'] != -1 and aw['steps_to_01'] != -1
                           else sm['steps_to_01'] != -1)),
        ("Loss<0.01 更快", (sm['steps_to_001'] < aw['steps_to_001']
                            if sm['steps_to_001'] != -1 and aw['steps_to_001'] != -1
                            else sm['steps_to_001'] != -1)),
        ("Loss<1e-5 达到", sm['steps_to_1e5'] != -1),
        ("稳定性更好", sm['stability_std'] < aw['stability_std']),
        ("耗时 <6s", sm['time_sec'] < 6.0),
    ]
    for desc, win in checks:
        wins += int(win)
        print(f"    {desc:25s} → {'✓' if win else '✗'}")
    print(f"    综合: {wins}/{total} 项通过")
    print("=" * 78)


if __name__ == "__main__":
    print("=" * 78)
    print("  SMCAO V2.1 vs AdamW — Rosenbrock 极限精度测试")
    print("  起点: (-1, 1), 步数: 5000")
    print("  V2.1: 自适应λ/κ + Hessian缓存 + Newton精修")
    print("=" * 78)

    INIT = (-1.0, 1.0)
    STEPS = 5000
    SEED = 42

    # ---- SMCAO V2.1 参数 ----
    SMCAO_LR = 0.002
    SMCAO_P = dict(
        a=5.0,
        kappa_base=200.0,       # 趋达增益基值
        kappa_alpha=2.0,        # 趋达增益放大系数
        phi=0.1,                # 边界层宽度
        ki=0.001,
        c=1.0,
        v_max=10.0,             # 速度上限
        reaching_threshold=10.0,
        newton_threshold=0.1,   # ‖∇f‖<0.1 时切换到 Newton (更保守, 确保在谷底)
        newton_lr=1.0,          # Newton 步长 (1.0=完全 Newton 步)
        lambda_min=1e-4,        # Hessian 正则化下限
        lambda_max=50.0,        # Hessian 正则化上限
        lambda_beta=0.3,
        lambda_g_mid=5.0,
        integrator='rk4',       # RK4 (精度优先)
    )

    print(f"\n[1/2] AdamW (lr=0.001, {STEPS} steps)...")
    aw_r = run_adamw(INIT, lr=0.001, steps=STEPS, seed=SEED)
    print(f"  AdamW: {aw_r['time_sec']:.2f}s, loss={aw_r['losses'][-1]:.6e}")

    print(f"\n[2/2] SMCAO V2.1 (lr={SMCAO_LR}, {STEPS} steps, integrator=rk4)...")
    sm_r = run_smcao(INIT, lr=SMCAO_LR, steps=STEPS, seed=SEED, **SMCAO_P)
    print(f"  SMCAO: {sm_r['time_sec']:.2f}s, loss={sm_r['losses'][-1]:.6e}")

    phi = SMCAO_P['phi']
    aw_m = compute_metrics(aw_r, "AdamW", phi=phi)
    sm_m = compute_metrics(sm_r, "SMCAO", phi=phi)
    sm_m['kappa_base'] = SMCAO_P['kappa_base']
    print_metrics(sm_m, aw_m)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "smcao_v2_vs_adamw_rosenbrock.png")
    plot_comparison(sm_r, aw_r, sm_m, aw_m, save_path, phi=phi)

    print("\n" + "=" * 78)
    print("  测试完成!")
    print("=" * 78)


"""
========================================================================
  V2.1 理论解释: 为什么 Newton 精修能突破精度天花板
========================================================================

1. SMCAO ODE 的精度瓶颈
   在 Phase 2 (滑模阶段), 等效动力学为:
     θ̇ = -c·∇f (在滑模面 s=0 上)
     θ̈ = -c·H·θ̇ = c²·H·∇f
   这是一阶梯度流, 收敛速率:
     ‖θ_k - θ*‖ ≈ (1 - c·lr·λ_min)^k · ‖θ_0 - θ*‖
   Rosenbrock λ_min ≈ 0.4, c=1, lr=0.002:
     (1 - 0.0008)^5000 ≈ e^{-4} ≈ 0.018
   所以 SMCAO ODE 的理论精度底线 ≈ 0.018, 无法突破 1e-5。

2. Newton 精修的二次收敛
   在 Phase 3 (‖∇f‖ < newton_threshold), 函数近似二次:
     f(θ) ≈ f(θ*) + 0.5·(θ-θ*)^T·H·(θ-θ*)
   Newton 步: θ ← θ - H⁻¹·∇f
   对于二次函数, 1 步精确到最优。
   对于 Rosenbrock (非完全二次), 收敛是二次的:
     ‖θ_{k+1} - θ*‖ ≤ C·‖θ_k - θ*‖²
   如果 ‖θ_k - θ*‖ ≈ 0.1, 则:
     步 1: 0.01
     步 2: 0.0001
     步 3: 1e-8
   3 步 Newton 即可从 0.1 精度达到 1e-8!

3. 自适应 λ 在 Newton 阶段的作用
   λ = λ_min + (λ_max - λ_min)·sigmoid(β·(‖g‖ - g_mid))
   当 ‖g‖ 很小 (Newton 阶段):
     sigmoid → 0, λ ≈ λ_min = 1e-4
     H_reg = H + 1e-4·I ≈ H (精确 Hessian)
     Newton 步精确, 不受正则化干扰

4. Armijo 线搜索保证全局收敛
   Newton 步可能过冲 (非二次区域), Armijo 条件:
     f(θ + α·Δθ) < f(θ) - c·α·‖∇f·Δθ‖
   自动缩减步长 α, 保证每步都减小 loss。
========================================================================
"""
