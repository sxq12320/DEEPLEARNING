"""
SMCScheduler vs AdamW — 2D 优化函数全面评估

测试函数:
  1. Mixture of Gaussians: 5 个高斯混合，全局极小在 (8,-1)，f=-8
  2. Rosenbrock: 窄曲谷，全局极小在 (1,1)，f=0

评估指标:
  1. 收敛速度 — 到达各 loss 阈值的步数（越少越好）
  2. 收敛精度 — 最终 loss 值（越小越好）
  3. 逃离能力 — 是否跳出局部极小，终点距全局极小的距离

可视化:
  - Contour 图 + 优化轨迹对比
  - Loss 收敛曲线
  - 详细指标汇总表
"""

import math
import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics.nn.modules.smc_scheduler import SMCScheduler


# ============================================================
# 测试函数 1: Mixture of Gaussians
# ============================================================

def mixture_of_gaussians(x, y):
    """
    f(x,y) = -3·exp(-(x²+y²)/2)
             -2·exp(-((x-3)²+(y-3)²)/2)
             -4·exp(-((x+3)²+(y-2)²)/3)
             -2.5·exp(-((x+2)²+(y+3)²)/1.5)
             -8·exp(-((x-8)²+(y+1)²)/1.5)

    局部极小:
      ( 0,  0)  : f ≈ -3.00
      ( 3,  3)  : f ≈ -2.00
      (-3,  2)  : f ≈ -4.00
      (-2, -3)  : f ≈ -2.50
      ( 8, -1)  : f ≈ -8.00  ← 全局极小
    """
    t1 = -3.0  * torch.exp(-(x**2 + y**2) / 2)
    t2 = -2.0  * torch.exp(-((x - 3)**2 + (y - 3)**2) / 2)
    t3 = -4.0  * torch.exp(-((x + 3)**2 + (y - 2)**2) / 3)
    t4 = -2.5  * torch.exp(-((x + 2)**2 + (y + 3)**2) / 1.5)
    t5 = -8.0  * torch.exp(-((x - 8)**2 + (y + 1)**2) / 1.5)
    return t1 + t2 + t3 + t4 + t5


GLOBAL_MIN_VAL = -8.0
GLOBAL_MIN_POS = (8.0, -1.0)


# ============================================================
# 测试函数 2: Rosenbrock
# ============================================================

def rosenbrock(x, y):
    """
    f(x,y) = (1-x)² + 100·(y-x²)²
    全局最小 (1,1), f=0
    """
    return (1 - x) ** 2 + 100 * (y - x ** 2) ** 2


ROSENBROCK_MIN = 0.0
ROSENBROCK_POS = (1.0, 1.0)


# ============================================================
# 模型包装
# ============================================================

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
            # Cosine LR 作为 AdamW 的 baseline schedule
            progress = step / max(steps - 1, 1)
            cos_lr = lr * (0.01 + 0.5 * (1.0 + math.cos(math.pi * progress)))
            for pg in optimizer.param_groups:
                pg["lr"] = cos_lr

        trajectory.append(model.xy.detach().cpu().numpy().copy())
        losses.append(loss.item())

    stats = scheduler.get_stats() if scheduler else {}
    return np.array(trajectory), np.array(losses), stats


# ============================================================
# 评估指标
# ============================================================

def compute_metrics(losses, trajectory, global_min_pos, escape_radius):
    """
    三项核心指标:
      1. 收敛速度: loss 从初始值下降 50%/80%/95% 各需多少步
      2. 收敛精度: 最终 loss
      3. 逃离能力: 终点距全局极小的距离 + 是否在 escape_radius 内
    """
    final_loss = losses[-1]
    init_loss = losses[0]
    final_pos = trajectory[-1]
    dist_to_global = float(np.sqrt(
        (final_pos[0] - global_min_pos[0])**2 +
        (final_pos[1] - global_min_pos[1])**2
    ))
    escaped = dist_to_global < escape_radius

    # 收敛速度: 到达各阈值的步数
    speed = {}
    for frac in [0.5, 0.8, 0.95]:
        target = init_loss + frac * (final_loss - init_loss)
        if final_loss < init_loss:
            idx = np.where(losses <= target)[0]
        else:
            idx = np.where(losses >= target)[0]
        speed[f"{int(frac*100)}%"] = int(idx[0]) if len(idx) > 0 else -1

    return {
        "final_loss": final_loss,
        "init_loss": init_loss,
        "final_pos": final_pos,
        "dist_to_global": dist_to_global,
        "escaped": escaped,
        "speed": speed,
    }


# ============================================================
# 绘图
# ============================================================

def plot_contour_with_trajectories(func, results_dict, title,
                                    xlim, ylim, local_minima):
    """
    三面板图: Contour+轨迹 | Loss 曲线 | 指标汇总表
    """
    fig, axes = plt.subplots(1, 3, figsize=(24, 7))

    # --- Contour 底图 ---
    xx = np.linspace(xlim[0], xlim[1], 600)
    yy = np.linspace(ylim[0], ylim[1], 600)
    X, Y = np.meshgrid(xx, yy)
    with torch.no_grad():
        Z = func(torch.tensor(X, dtype=torch.float32),
                 torch.tensor(Y, dtype=torch.float32)).numpy()

    ax = axes[0]
    levels = np.linspace(Z.min(), Z.max(), 60)
    cf = ax.contourf(X, Y, Z, levels=levels, cmap="RdBu_r", alpha=0.75)
    ax.contour(X, Y, Z, levels=levels, colors="gray", linewidths=0.12, alpha=0.35)
    plt.colorbar(cf, ax=ax, shrink=0.8, label="f(x,y)")

    # 标注局部极小
    for mx, my, label in local_minima:
        is_global = "GLOBAL" in label.upper()
        marker = "*" if is_global else "+"
        ms = 14 if is_global else 10
        mc = "gold" if is_global else "black"
        ax.plot(mx, my, marker, color=mc, markersize=ms, markeredgewidth=2)
        ax.annotate(label, (mx, my), textcoords="offset points",
                    xytext=(10, 10), fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    # 颜色映射: 4 个起始点 → 4 组颜色
    cmap_adamw = ["#d62728", "#e07b39", "#b03060", "#8c564b"]
    cmap_smc   = ["#2ca02c", "#17becf", "#9467bd", "#e377c2"]

    for i, (name, data) in enumerate(results_dict.items()):
        ca = cmap_adamw[i % len(cmap_adamw)]
        cs = cmap_smc[i % len(cmap_smc)]
        traj_a = data["adamw"][0]
        traj_s = data["smc"][0]

        # AdamW 轨迹
        ax.plot(traj_a[:, 0], traj_a[:, 1], "-", color=ca, lw=0.8, alpha=0.75)
        ax.plot(traj_a[0, 0], traj_a[0, 1], "s", color=ca, ms=7, mec="k", mew=1)
        ax.plot(traj_a[-1, 0], traj_a[-1, 1], "X", color=ca, ms=9, mec="k", mew=1)

        # SMC 轨迹
        ax.plot(traj_s[:, 0], traj_s[:, 1], "-", color=cs, lw=0.8, alpha=0.75)
        ax.plot(traj_s[0, 0], traj_s[0, 1], "s", color=cs, ms=7, mec="k", mew=1)
        ax.plot(traj_s[-1, 0], traj_s[-1, 1], "X", color=cs, ms=9, mec="k", mew=1)

        # 起始点标签
        ax.annotate(f"A{i+1}", (traj_a[0, 0], traj_a[0, 1]),
                    textcoords="offset points", xytext=(-12, -8),
                    fontsize=7, fontweight="bold", color=ca)
        ax.annotate(f"S{i+1}", (traj_s[0, 0], traj_s[0, 1]),
                    textcoords="offset points", xytext=(4, -8),
                    fontsize=7, fontweight="bold", color=cs)

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(f"{title}: Contour + Trajectories", fontsize=13)
    ax.set_aspect("equal")

    # 手工图例
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color="gray", lw=2, label="AdamW (dashed)"),
        Line2D([0], [0], color="gray", lw=2, linestyle="--", label="SMC (solid)"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor="gray",
               markersize=7, label="Start"),
        Line2D([0], [0], marker="X", color="w", markerfacecolor="gray",
               markersize=9, label="End"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")

    # --- Loss 曲线 ---
    ax = axes[1]
    for i, (name, data) in enumerate(results_dict.items()):
        ca = cmap_adamw[i % len(cmap_adamw)]
        cs = cmap_smc[i % len(cmap_smc)]
        ax.plot(data["adamw"][1], color=ca, lw=1, alpha=0.85,
                label=f"AdamW ({name})")
        ax.plot(data["smc"][1], color=cs, lw=1, alpha=0.85, ls="--",
                label=f"SMC ({name})")

    true_min = GLOBAL_MIN_VAL if "MoG" in title else ROSENBROCK_MIN
    ax.axhline(true_min, color="gray", ls=":", lw=1, alpha=0.6, label="Global min")
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"{title}: Convergence Curves", fontsize=13)
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    # --- 指标汇总表 ---
    ax = axes[2]
    ax.axis("off")
    col_labels = [
        "Start", "AdamW\nFinal Loss", "SMC\nFinal Loss",
        "AdamW\nDist→Global", "SMC\nDist→Global",
        "AdamW\nEscaped?", "SMC\nEscaped?"
    ]
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

    table = ax.table(cellText=cell_data, colLabels=col_labels,
                     loc="center", cellLoc="center",
                     colColours=["#e8e8e8"] * len(col_labels))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.7)
    ax.set_title(f"{title}: Metrics Summary", fontsize=13, pad=25)

    plt.tight_layout()
    return fig


def print_summary(results_dict):
    """打印某函数的详细指标表"""
    print(f"\n{'Start':14s} | {'AdamW Loss':>10s} {'SMC Loss':>10s} {'Diff':>10s} | "
          f"{'A Dist':>7s} {'S Dist':>7s} | {'A Esc':>5s} {'S Esc':>5s}")
    print("-" * 85)
    for name, data in results_dict.items():
        ma = data["adamw_metrics"]
        ms = data["smc_metrics"]
        diff = ma["final_loss"] - ms["final_loss"]
        print(f"{name:14s} | {ma['final_loss']:10.4f} {ms['final_loss']:10.4f} {diff:+10.4f} | "
              f"{ma['dist_to_global']:7.2f} {ms['dist_to_global']:7.2f} | "
              f"{'  YES' if ma['escaped'] else '   no':>5s} "
              f"{'  YES' if ms['escaped'] else '   no':>5s}")


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 80)
    print("  SMCScheduler vs AdamW — 2D 优化函数全面评估")
    print("  测试函数: MoG (5 高斯混合) + Rosenbrock")
    print("  每个函数 4 个起始点, 评估: 收敛速度 / 精度 / 逃离能力")
    print("=" * 80)

    SEED = 42
    STEPS = 2000

    # ============================================================
    # 函数 1: Mixture of Gaussians
    # ============================================================
    print("\n" + "=" * 80)
    print("  函数 1: Mixture of Gaussians")
    print("  全局极小: (8, -1), f=-8.0")
    print("=" * 80)

    mog_minima = [
        (0, 0, "Min1 (0,0) f=-3"),
        (3, 3, "Min2 (3,3) f=-2"),
        (-3, 2, "Min3 (-3,2) f=-4"),
        (-2, -3, "Min4 (-2,-3) f=-2.5"),
        (8, -1, "GLOBAL (8,-1) f=-8"),
    ]

    starts_mog = {
        "Near (0,0)":    (0.5, 0.5),
        "Near (3,3)":    (3.5, 3.5),
        "Near (-3,2)":   (-2.5, 2.5),
        "Near (-2,-3)":  (-1.5, -2.5),
    }

    results_mog = {}

    for name, (ix, iy) in starts_mog.items():
        print(f"\n--- {name}  init=({ix},{iy}) ---")
        traj_a, loss_a, _ = run_optimization(
            mixture_of_gaussians, ix, iy, lr=0.01, steps=STEPS, use_smc=False, seed=SEED)
        traj_s, loss_s, stats_s = run_optimization(
            mixture_of_gaussians, ix, iy, lr=0.01, steps=STEPS, use_smc=True, seed=SEED)

        ma = compute_metrics(loss_a, traj_a, GLOBAL_MIN_POS, escape_radius=2.0)
        ms = compute_metrics(loss_s, traj_s, GLOBAL_MIN_POS, escape_radius=2.0)

        results_mog[name] = {
            "adamw": (traj_a, loss_a), "smc": (traj_s, loss_s),
            "adamw_metrics": ma, "smc_metrics": ms,
        }
        print(f"  AdamW: loss={ma['final_loss']:.4f}, dist={ma['dist_to_global']:.2f}, escaped={ma['escaped']}")
        print(f"  SMC:   loss={ms['final_loss']:.4f}, dist={ms['dist_to_global']:.2f}, escaped={ms['escaped']}")
        if stats_s:
            print(f"  SMC stats: escapes={stats_s.get('escape_events', 0)}")

    print_summary(results_mog)

    # ============================================================
    # 函数 2: Rosenbrock
    # ============================================================
    print("\n" + "=" * 80)
    print("  函数 2: Rosenbrock")
    print("  全局极小: (1, 1), f=0")
    print("=" * 80)

    ros_minima = [
        (1, 1, "GLOBAL (1,1) f=0"),
    ]

    starts_ros = {
        "(-1, 1)":   (-1.0, 1.0),
        "(-2, 2)":   (-2.0, 2.0),
        "(0, 0)":    (0.0, 0.0),
        "(-3, 3)":   (-3.0, 3.0),
    }

    results_ros = {}

    for name, (ix, iy) in starts_ros.items():
        print(f"\n--- Start {name}  init=({ix},{iy}) ---")
        traj_a, loss_a, _ = run_optimization(
            rosenbrock, ix, iy, lr=0.001, steps=STEPS, use_smc=False, seed=SEED)
        traj_s, loss_s, stats_s = run_optimization(
            rosenbrock, ix, iy, lr=0.001, steps=STEPS, use_smc=True, seed=SEED)

        ma = compute_metrics(loss_a, traj_a, ROSENBROCK_POS, escape_radius=0.5)
        ms = compute_metrics(loss_s, traj_s, ROSENBROCK_POS, escape_radius=0.5)

        results_ros[name] = {
            "adamw": (traj_a, loss_a), "smc": (traj_s, loss_s),
            "adamw_metrics": ma, "smc_metrics": ms,
        }
        print(f"  AdamW: loss={ma['final_loss']:.4f}, dist={ma['dist_to_global']:.2f}, escaped={ma['escaped']}")
        print(f"  SMC:   loss={ms['final_loss']:.4f}, dist={ms['dist_to_global']:.2f}, escaped={ms['escaped']}")
        if stats_s:
            print(f"  SMC stats: escapes={stats_s.get('escape_events', 0)}")

    print_summary(results_ros)

    # ============================================================
    # 综合汇总
    # ============================================================
    all_a = [d["adamw_metrics"] for d in results_mog.values()] + \
            [d["adamw_metrics"] for d in results_ros.values()]
    all_s = [d["smc_metrics"] for d in results_mog.values()] + \
            [d["smc_metrics"] for d in results_ros.values()]
    total = len(all_a)

    a_mean_loss = np.mean([m["final_loss"] for m in all_a])
    s_mean_loss = np.mean([m["final_loss"] for m in all_s])
    loss_wins = sum(1 for a, s in zip(all_a, all_s) if s["final_loss"] < a["final_loss"])
    a_escaped = sum(1 for m in all_a if m["escaped"])
    s_escaped = sum(1 for m in all_s if m["escaped"])
    a_mean_dist = np.mean([m["dist_to_global"] for m in all_a])
    s_mean_dist = np.mean([m["dist_to_global"] for m in all_s])

    # 综合得分: loss 更好 +1, 逃离更好 +1, 距离更近 +1
    score = 0
    score_details = {"loss": 0, "escape": 0, "dist": 0}
    for a_m, s_m in zip(all_a, all_s):
        if s_m["final_loss"] < a_m["final_loss"]:
            score += 1; score_details["loss"] += 1
        if s_m["escaped"] and not a_m["escaped"]:
            score += 1; score_details["escape"] += 1
        if s_m["dist_to_global"] < a_m["dist_to_global"]:
            score += 1; score_details["dist"] += 1

    print("\n" + "=" * 80)
    print("  综合评估汇总 (2 个函数, 共 8 个起始点)")
    print("=" * 80)
    print(f"\n  [收敛精度]  平均 loss:  AdamW={a_mean_loss:.4f}  SMC={s_mean_loss:.4f}")
    print(f"              SMC 在 {loss_wins}/{total} 个起点上 loss 更低")
    print(f"\n  [逃离能力]  AdamW={a_escaped}/{total}  SMC={s_escaped}/{total} 逃离到全局极小附近")
    print(f"\n  [平均距离]  AdamW={a_mean_dist:.2f}  SMC={s_mean_dist:.2f}")
    print(f"\n  [综合得分]  {score}/{total*3} ({score/(total*3)*100:.0f}%)")
    print(f"    收敛精度: {score_details['loss']}/{total}")
    print(f"    逃离能力: {score_details['escape']}/{total}")
    print(f"    距离优势: {score_details['dist']}/{total}")
    verdict = "SMC 全方位超越 AdamW ✓" if score > total * 3 * 0.6 else "需要进一步调优"
    print(f"    结论: {verdict}")
    print("=" * 80)

    # ============================================================
    # 绘图
    # ============================================================
    print("\n绘制 Contour 图 ...")

    fig1 = plot_contour_with_trajectories(
        mixture_of_gaussians, results_mog,
        title="Mixture of Gaussians",
        xlim=(-6, 11), ylim=(-6, 6),
        local_minima=mog_minima,
    )
    save1 = os.path.join(os.path.dirname(__file__), "smc_saddle_trajectory.png")
    fig1.savefig(save1, dpi=150, bbox_inches="tight")
    print(f"  MoG 图: {save1}")
    plt.close(fig1)

    fig2 = plot_contour_with_trajectories(
        rosenbrock, results_ros,
        title="Rosenbrock",
        xlim=(-4, 4), ylim=(-1, 6),
        local_minima=ros_minima,
    )
    save2 = os.path.join(os.path.dirname(__file__), "smc_net_training.png")
    fig2.savefig(save2, dpi=150, bbox_inches="tight")
    print(f"  Rosenbrock 图: {save2}")
    plt.close(fig2)

    print("\n完成！")
