"""
SMCScheduler 全面评估 — Mixture of Gaussians 多局部极值测试

测试函数: 5 个高斯混合（3 个局部极小 + 1 个全局极小）
评估指标:
  1. 收敛速度 — 到达指定 loss 阈值的步数
  2. 收敛精度 — 最终 loss 值
  3. 逃离能力 — 是否跳出局部极小到达全局极小附近
可视化: Contour 图 + 收敛轨迹 + Loss 曲线
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics.nn.modules.smc_scheduler import SMCScheduler


# ============================================================
# 测试函数: Mixture of Gaussians
# ============================================================

def mixture_of_gaussians(x, y):
    """
    f(x,y) = -3*exp(-(x²+y²)/2)
             -2*exp(-((x-1)²+(y-2)²)/2)
             -3.5*exp(-((x+2.5)²+(y-2)²)/3)
             -2.5*exp(-((x+1)²+(y+1.5)²)/1.5)
             -4*exp(-((x-2)²+(y+1)²)/1.5)

    局部极小:
      (0, 0)     : f ≈ -3.00
      (1, 2)     : f ≈ -2.00
      (-2.5, 2)  : f ≈ -3.50
      (-1, -1.5) : f ≈ -2.50
      (2, -1)    : f ≈ -4.00  (全局极小)
    """
    t1 = -3.0 * torch.exp(-(x**2 + y**2) / 2)
    t2 = -2.0 * torch.exp(-((x - 1)**2 + (y - 2)**2) / 2)
    t3 = -3.5 * torch.exp(-((x + 2.5)**2 + (y - 2)**2) / 3)
    t4 = -2.5 * torch.exp(-((x + 1)**2 + (y + 1.5)**2) / 1.5)
    t5 = -4.0 * torch.exp(-((x - 2)**2 + (y + 1)**2) / 1.5)
    return t1 + t2 + t3 + t4 + t5


GLOBAL_MIN = -4.0
GLOBAL_MIN_POS = (2.0, -1.0)


def rosenbrock(x, y):
    """
    Rosenbrock: f(x,y) = (1-x)² + 100(y-x²)²
    全局最小 (1,1), f=0
    窄曲谷，经典优化难题
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


ROSENBROCK_MIN = 0.0
ROSENBROCK_POS = (1.0, 1.0)
ESCAPE_RADIUS = 2.0  # 距全局极小 < 此值视为成功逃离


class SaddleFunction(nn.Module):
    def __init__(self, func, init_x, init_y):
        super().__init__()
        self.xy = nn.Parameter(torch.tensor([float(init_x), float(init_y)]))
        self.func = func
    def forward(self):
        return self.func(self.xy[0], self.xy[1])


# ============================================================
# 优化运行器
# ============================================================

def run_optimization(func, init_x, init_y, lr, steps, use_smc=False, seed=42):
    torch.manual_seed(seed)
    model = SaddleFunction(func, init_x, init_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    if use_smc:
        scheduler = SMCScheduler(optimizer, total_steps=steps, verbose=False)
    else:
        scheduler = None

    trajectory = [model.xy.detach().cpu().numpy().copy()]
    losses = []

    for step in range(steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()

        if scheduler:
            scheduler.observe_gradients()
        optimizer.step()
        if scheduler:
            scheduler.step(loss.item())
        else:
            # Cosine LR baseline
            progress = step / max(steps - 1, 1)
            cos_lr = lr * (0.01 + 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = cos_lr

        trajectory.append(model.xy.detach().cpu().numpy().copy())
        losses.append(loss.item())

    return np.array(trajectory), np.array(losses), scheduler.get_stats() if scheduler else {}


# ============================================================
# 评估指标
# ============================================================

def compute_metrics(losses, trajectory, global_min_pos=None, escape_radius=None):
    """
    计算三项指标:
    1. 收敛速度: 到达目标 loss 的步数
    2. 收敛精度: 最终 loss
    3. 逃离能力: 终点距全局极小的距离
    """
    if global_min_pos is None:
        global_min_pos = GLOBAL_MIN_POS
    if escape_radius is None:
        escape_radius = ESCAPE_RADIUS

    final_loss = losses[-1]
    final_pos = trajectory[-1]
    dist_to_global = np.sqrt((final_pos[0] - global_min_pos[0])**2 +
                             (final_pos[1] - global_min_pos[1])**2)
    escaped = dist_to_global < escape_radius

    # 收敛速度: 到达各阈值的步数
    init_loss = losses[0]
    speed_at = {}
    for frac in [0.5, 0.8, 0.95]:
        target = init_loss + frac * (final_loss - init_loss)
        if final_loss < init_loss:  # descending
            idx = np.where(losses <= target)[0]
        else:
            idx = np.where(losses >= target)[0]
        speed_at[f"{int(frac*100)}%"] = int(idx[0]) if len(idx) > 0 else -1

    return {
        "final_loss": final_loss,
        "init_loss": init_loss,
        "dist_to_global": dist_to_global,
        "escaped": escaped,
        "speed": speed_at,
    }


# ============================================================
# 绘图
# ============================================================

def plot_contour_with_trajectories(func, results_dict, title="", xlim=(-5, 5), ylim=(-5, 5)):
    """
    绘制 Contour 图 + 所有起始点的轨迹对比
    results_dict: {start_name: {"adamw": (traj, loss), "smc": (traj, loss)}}
    """
    fig, axes = plt.subplots(1, 3, figsize=(22, 7))

    # --- Contour 底图 ---
    xx = np.linspace(xlim[0], xlim[1], 500)
    yy = np.linspace(ylim[0], ylim[1], 500)
    X, Y = np.meshgrid(xx, yy)
    with torch.no_grad():
        Z = func(torch.tensor(X, dtype=torch.float32),
                 torch.tensor(Y, dtype=torch.float32)).numpy()

    ax = axes[0]
    cf = ax.contourf(X, Y, Z, levels=50, cmap="RdBu_r", alpha=0.7)
    ax.contour(X, Y, Z, levels=50, colors="gray", linewidths=0.15, alpha=0.4)
    plt.colorbar(cf, ax=ax, shrink=0.8, label="f(x,y)")

    # Mark local minima
    minima = [(0,0,"-3.00"), (1,2,"-2.00"), (-2.5,2,"-3.50"), (-1,-1.5,"-2.50"), (2,-1,"-4.00 GLOBAL")]
    for mx, my, label in minima:
        ax.plot(mx, my, "k+", markersize=10, markeredgewidth=2)
        ax.annotate(label, (mx, my), textcoords="offset points",
                   xytext=(8, 8), fontsize=7, fontweight="bold",
                   bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8))

    # Plot trajectories
    colors_adamw = ["#d62728", "#e25a50", "#c9453a", "#b8382d", "#a12e24", "#8c251d", "#7a1d17"]
    colors_smc = ["#2ca02c", "#4fb94f", "#3d9e3d", "#2d8a2d", "#1f751f", "#156215", "#0d4f0d"]
    for i, (name, data) in enumerate(results_dict.items()):
        traj_a = data["adamw"][0]
        traj_s = data["smc"][0]
        ax.plot(traj_a[:, 0], traj_a[:, 1], "o-", color=colors_adamw[i],
                markersize=1, linewidth=0.6, alpha=0.7, label=f"AdamW ({name})")
        ax.plot(traj_s[:, 0], traj_s[:, 1], "o-", color=colors_smc[i],
                markersize=1, linewidth=0.6, alpha=0.7, label=f"SMC ({name})")
        ax.plot(traj_a[0, 0], traj_a[0, 1], "s", color=colors_adamw[i], markersize=7, markeredgecolor="k")
        ax.plot(traj_s[0, 0], traj_s[0, 1], "s", color=colors_smc[i], markersize=7, markeredgecolor="k")
        ax.plot(traj_a[-1, 0], traj_a[-1, 1], "X", color=colors_adamw[i], markersize=9, markeredgecolor="k")
        ax.plot(traj_s[-1, 0], traj_s[-1, 1], "X", color=colors_smc[i], markersize=9, markeredgecolor="k")

    ax.set_xlim(xlim); ax.set_ylim(ylim)
    ax.set_xlabel("x", fontsize=12); ax.set_ylabel("y", fontsize=12)
    ax.set_title("Contour + Trajectories", fontsize=13)
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.set_aspect("equal")

    # --- Loss curves ---
    ax = axes[1]
    for i, (name, data) in enumerate(results_dict.items()):
        ax.plot(data["adamw"][1], color=colors_adamw[i], lw=1, alpha=0.8, label=f"AdamW ({name})")
        ax.plot(data["smc"][1], color=colors_smc[i], lw=1, alpha=0.8, label=f"SMC ({name})")
    ax.axhline(GLOBAL_MIN, color="gray", ls="--", lw=0.8, alpha=0.5, label="Global min")
    ax.set_xlabel("Step", fontsize=12); ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Convergence Curves", fontsize=13)
    ax.legend(fontsize=7, ncol=2); ax.grid(True, alpha=0.3)

    # --- Metrics table ---
    ax = axes[2]
    ax.axis("off")
    col_labels = ["Start", "AdamW\nFinal Loss", "SMC\nFinal Loss", "AdamW\nDist→Global", "SMC\nDist→Global", "AdamW\nEscaped?", "SMC\nEscaped?"]
    cell_data = []
    for name, data in results_dict.items():
        ma = data["adamw_metrics"]
        ms = data["smc_metrics"]
        cell_data.append([
            name,
            f"{ma['final_loss']:.4f}",
            f"{ms['final_loss']:.4f}",
            f"{ma['dist_to_global']:.2f}",
            f"{ms['dist_to_global']:.2f}",
            "YES" if ma["escaped"] else "no",
            "YES" if ms["escaped"] else "no",
        ])
    table = ax.table(cellText=cell_data, colLabels=col_labels, loc="center",
                     cellLoc="center", colColours=["#f0f0f0"]*len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.6)
    ax.set_title("Metrics Summary", fontsize=13, pad=20)

    plt.tight_layout()
    return fig


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 70)
    print("SMCScheduler vs AdamW — Mixture of Gaussians 全面评估")
    print("=" * 70)

    SEED = 42
    STEPS = 2000
    LR = 0.01
    STEPS = 2000

    # ============================================================
    # 函数 1: Mixture of Gaussians (4 个起始点)
    # ============================================================
    print("\n" + "=" * 70)
    print("函数 1: Mixture of Gaussians")
    print("=" * 70)

    starts_mog = {
        "Near (0,0)":     (0.5, 0.5),
        "Near (1,2)":     (1.5, 2.5),
        "Near (-2.5,2)":  (-2.0, 2.5),
        "Near (2,-1)":    (2.5, -0.5),
    }

    results_mog = {}
    all_mog = {"adamw": [], "smc": []}

    for name, (ix, iy) in starts_mog.items():
        print(f"\n--- {name} (init={ix},{iy}) ---")
        traj_a, loss_a, _ = run_optimization(mixture_of_gaussians, ix, iy, LR, STEPS, use_smc=False, seed=SEED)
        traj_s, loss_s, stats_s = run_optimization(mixture_of_gaussians, ix, iy, LR, STEPS, use_smc=True, seed=SEED)
        ma = compute_metrics(loss_a, traj_a)
        ms = compute_metrics(loss_s, traj_s)
        all_mog["adamw"].append(ma)
        all_mog["smc"].append(ms)
        results_mog[name] = {"adamw": (traj_a, loss_a), "smc": (traj_s, loss_s),
                              "adamw_metrics": ma, "smc_metrics": ms}
        print(f"  AdamW:  loss={ma['final_loss']:.4f}, dist={ma['dist_to_global']:.2f}, escaped={ma['escaped']}")
        print(f"  SMC:    loss={ms['final_loss']:.4f}, dist={ms['dist_to_global']:.2f}, escaped={ms['escaped']}")
        if stats_s:
            print(f"  SMC stats: escapes={stats_s['escape_events']}, reverts={stats_s['reverts']}")

    # ============================================================
    # 函数 2: Rosenbrock (4 个起始点)
    # ============================================================
    print("\n" + "=" * 70)
    print("函数 2: Rosenbrock")
    print("=" * 70)

    starts_ros = {
        "(-1, 1)":   (-1.0, 1.0),
        "(-2, 2)":   (-2.0, 2.0),
        "(0, 0)":    (0.0, 0.0),
        "(-3, 3)":   (-3.0, 3.0),
    }

    results_ros = {}
    all_ros = {"adamw": [], "smc": []}

    for name, (ix, iy) in starts_ros.items():
        print(f"\n--- Start {name} (init={ix},{iy}) ---")
        traj_a, loss_a, _ = run_optimization(rosenbrock, ix, iy, 0.001, STEPS, use_smc=False, seed=SEED)
        traj_s, loss_s, stats_s = run_optimization(rosenbrock, ix, iy, 0.001, STEPS, use_smc=True, seed=SEED)
        ma = compute_metrics(loss_a, traj_a, global_min_pos=ROSENBROCK_POS, escape_radius=0.5)
        ms = compute_metrics(loss_s, traj_s, global_min_pos=ROSENBROCK_POS, escape_radius=0.5)
        all_ros["adamw"].append(ma)
        all_ros["smc"].append(ms)
        results_ros[name] = {"adamw": (traj_a, loss_a), "smc": (traj_s, loss_s),
                              "adamw_metrics": ma, "smc_metrics": ms}
        print(f"  AdamW:  loss={ma['final_loss']:.4f}, dist={ma['dist_to_global']:.2f}, escaped={ma['escaped']}")
        print(f"  SMC:    loss={ms['final_loss']:.4f}, dist={ms['dist_to_global']:.2f}, escaped={ms['escaped']}")
        if stats_s:
            print(f"  SMC stats: escapes={stats_s['escape_events']}, reverts={stats_s['reverts']}")

    # ============================================================
    # 综合汇总
    # ============================================================
    all_a = all_mog["adamw"] + all_ros["adamw"]
    all_s = all_mog["smc"] + all_ros["smc"]
    total = len(all_a)

    a_mean_loss = np.mean([m["final_loss"] for m in all_a])
    s_mean_loss = np.mean([m["final_loss"] for m in all_s])
    loss_wins = sum(1 for a, s in zip(all_a, all_s) if s["final_loss"] < a["final_loss"])

    a_escaped = sum(1 for m in all_a if m["escaped"])
    s_escaped = sum(1 for m in all_s if m["escaped"])

    score = 0
    for a_m, s_m in zip(all_a, all_s):
        if s_m["final_loss"] < a_m["final_loss"]: score += 1
        if s_m["escaped"] and not a_m["escaped"]: score += 1
        if s_m["dist_to_global"] < a_m["dist_to_global"]: score += 1

    print("\n" + "=" * 70)
    print("综合评估汇总 (两个函数, 共 8 个起始点)")
    print("=" * 70)
    print(f"\n[收敛精度] 平均 loss: AdamW={a_mean_loss:.4f}, SMC={s_mean_loss:.4f}")
    print(f"           SMC 在 {loss_wins}/{total} 个起点上 loss 更低")
    print(f"\n[逃离能力] AdamW={a_escaped}/{total}, SMC={s_escaped}/{total} 逃离到全局极小附近")
    print(f"\n[综合得分] {score}/{total*3} ({score/(total*3)*100:.0f}%)")
    print(f"           {'SMC 全方位超越 AdamW' if score > total*3*0.5 else '需要进一步调优'}")

    # ============================================================
    # 绘图
    # ============================================================
    print("\n绘制 Contour 图 ...")

    fig1 = plot_contour_with_trajectories(mixture_of_gaussians, results_mog, "Mixture of Gaussians")
    save1 = os.path.join(os.path.dirname(__file__), "smc_saddle_trajectory.png")
    fig1.savefig(save1, dpi=150, bbox_inches="tight")
    print(f"  MoG 图: {save1}")
    plt.close(fig1)

    fig2 = plot_contour_with_trajectories(rosenbrock, results_ros, "Rosenbrock",
                                           xlim=(-4, 4), ylim=(-1, 6))
    save2 = os.path.join(os.path.dirname(__file__), "smc_net_training.png")
    fig2.savefig(save2, dpi=150, bbox_inches="tight")
    print(f"  Rosenbrock 图: {save2}")
    plt.close(fig2)

    # ============================================================
    # 详细指标表
    # ============================================================
    print("\n详细指标:")
    print(f"{'函数':15s} {'起点':12s} | {'AdamW':>8s} {'SMC':>8s} {'Diff':>8s} | {'Esc A':>5s} {'Esc S':>5s}")
    print("-" * 75)
    for func_name, results_dict in [("MoG", results_mog), ("Rosenbrock", results_ros)]:
        for name, data in results_dict.items():
            ma = data["adamw_metrics"]
            ms = data["smc_metrics"]
            diff = ma["final_loss"] - ms["final_loss"]
            print(f"{func_name:15s} {name:12s} | {ma['final_loss']:8.4f} {ms['final_loss']:8.4f} {diff:+8.4f} | "
                  f"{'  YES' if ma['escaped'] else '   no':>5s} {'  YES' if ms['escaped'] else '   no':>5s}")
    print("=" * 75)
