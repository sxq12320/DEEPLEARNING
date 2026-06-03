"""
SMCScheduler 可视化测试 — 2D 鞍点逃离轨迹对比

对比：
    1. 标准 AdamW（被困在鞍点附近）
    2. AdamW + SMCScheduler（滑模控制逃离鞍点）

合成函数：
    Monkey Saddle: f(x, y) = x³ - 3xy²
    （原点为鞍点，三条上升脊和三条下降谷交替排列）
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ultralytics.nn.modules.smc_scheduler import SMCScheduler


# ============================================================
# 合成函数定义
# ============================================================

def monkey_saddle(x, y):
    """Monkey Saddle: f(x,y) = x³ - 3xy²，原点为鞍点"""
    return x ** 3 - 3 * x * y ** 2


def monkey_saddle_with_noise(x, y):
    """带噪声的 Monkey Saddle，增加逃离难度"""
    base = monkey_saddle(x, y)
    noise = 0.01 * (x ** 2 + y ** 2)  # 轻微碗形正则化，防止发散
    return base + noise


class SaddleFunction(nn.Module):
    """将 2D 合成函数包装为 nn.Module，参数为 (x, y)"""

    def __init__(self, func, init_x=-0.5, init_y=-0.5):
        super().__init__()
        self.xy = nn.Parameter(torch.tensor([init_x, init_y], dtype=torch.float32))
        self.func = func

    def forward(self):
        return self.func(self.xy[0], self.xy[1])


# ============================================================
# 训练轨迹记录
# ============================================================

def run_optimization(func, init_x, init_y, lr, steps, use_smc=False, seed=42):
    """
    运行优化并记录轨迹。

    Args:
        func: 合成函数
        init_x, init_y: 初始点
        lr: 学习率
        steps: 迭代步数
        use_smc: 是否启用 SMCScheduler
        seed: 随机种子

    Returns:
        trajectory: np.array shape (steps+1, 2)
        losses: np.array shape (steps+1,)
        modes: list of mode strings (only when use_smc=True)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = SaddleFunction(func, init_x, init_y)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    scheduler = None
    if use_smc:
        scheduler = SMCScheduler(
            optimizer,
            total_steps=steps,
            c=0.5,
            window_size=30,
            thr_reach=5e-4,
            thr_saddle=1e-5,
            thr_chatter_var=1e-7,
            beta1_default=0.9,
            beta1_escape=0.1,
            lr_boost=3.0,
            escape_duration=20,
            lr_dampen=0.5,
            wd_boost=0.005,
            loss_window=15,
            loss_threshold=5e-5,
            verbose=False,
        )

    trajectory = [model.xy.detach().cpu().numpy().copy()]
    losses = []
    modes = []

    for step in range(steps):
        optimizer.zero_grad()
        loss = model()
        loss.backward()

        if scheduler:
            scheduler.observe_gradients()

        optimizer.step()

        if scheduler:
            scheduler.step(loss.item())
            modes.append(scheduler.mode)

        trajectory.append(model.xy.detach().cpu().numpy().copy())
        losses.append(loss.item())

    return np.array(trajectory), np.array(losses), modes


# ============================================================
# 可视化
# ============================================================

def plot_contour_with_trajectories(traj_adamw, traj_smc, losses_adamw, losses_smc,
                                   modes, func, xlim=(-2, 2), ylim=(-2, 2)):
    """绘制等高线 + 双轨迹对比图"""

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # --- 共用等高线底图 ---
    xx = np.linspace(xlim[0], xlim[1], 300)
    yy = np.linspace(ylim[0], ylim[1], 300)
    X, Y = np.meshgrid(xx, yy)
    Z = func(X, Y)
    Z_clip = np.clip(Z, -5, 5)

    # 非均匀 levels：中心加密、外围稀疏，突出鞍点附近细节
    levels = np.concatenate([
        np.linspace(-5, -0.5, 15),
        np.linspace(-0.4, 0.4, 20),
        np.linspace(0.5, 5, 15),
    ])

    # ============================================================
    # 子图 1：等高线 + 轨迹对比
    # ============================================================
    ax = axes[0]
    ax.contourf(X, Y, Z_clip, levels=levels, cmap="RdBu_r", alpha=0.6)
    ax.contour(X, Y, Z_clip, levels=levels, colors="gray", linewidths=0.3, alpha=0.5)
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.axvline(0, color="k", linewidth=0.5, linestyle="--", alpha=0.3)
    ax.plot(0, 0, "k+", markersize=12, markeredgewidth=2, label="Saddle (0,0)")

    # AdamW 轨迹
    ax.plot(traj_adamw[:, 0], traj_adamw[:, 1], "o-", color="#e74c3c",
            markersize=1.5, linewidth=0.8, alpha=0.8, label="AdamW (baseline)")
    ax.plot(traj_adamw[0, 0], traj_adamw[0, 1], "s", color="#e74c3c",
            markersize=8, markeredgecolor="k")
    ax.plot(traj_adamw[-1, 0], traj_adamw[-1, 1], "X", color="#e74c3c",
            markersize=10, markeredgecolor="k")

    # SMC 轨迹
    ax.plot(traj_smc[:, 0], traj_smc[:, 1], "o-", color="#2ecc71",
            markersize=1.5, linewidth=0.8, alpha=0.8, label="AdamW + SMC")
    ax.plot(traj_smc[0, 0], traj_smc[0, 1], "s", color="#2ecc71",
            markersize=8, markeredgecolor="k")
    ax.plot(traj_smc[-1, 0], traj_smc[-1, 1], "X", color="#2ecc71",
            markersize=10, markeredgecolor="k")

    # 标注鞍点逃离区域
    ax.annotate("Saddle Escape\nTriggered", xy=(traj_smc[50, 0], traj_smc[50, 1]),
                fontsize=8, color="#2ecc71", fontweight="bold",
                arrowprops=dict(arrowstyle="->", color="#2ecc71"),
                xytext=(traj_smc[50, 0] + 0.5, traj_smc[50, 1] + 0.5))

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title("Optimization Trajectory on Monkey Saddle", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")

    # ============================================================
    # 子图 2：Loss 曲线对比
    # ============================================================
    ax = axes[1]
    ax.plot(losses_adamw, color="#e74c3c", linewidth=1.2, label="AdamW", alpha=0.8)
    ax.plot(losses_smc, color="#2ecc71", linewidth=1.2, label="AdamW + SMC", alpha=0.8)
    ax.set_xlabel("Step", fontsize=12)
    ax.set_ylabel("Loss (f(x,y))", fontsize=12)
    ax.set_title("Loss Curves", fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    # ============================================================
    # 子图 3：SMC 模式状态机 — 状态时序色带 (State Sequence Ribbon)
    # ============================================================
    ax = axes[2]
    if modes:
        mode_map = {"normal": 0, "escape": 1, "damping": 2}
        mode_values = np.array([mode_map.get(m, 0) for m in modes]).reshape(1, -1)

        cmap = matplotlib.colormaps.get_cmap("Set1").resampled(3)
        ax.imshow(mode_values, aspect="auto", cmap=cmap, interpolation="nearest",
                  extent=[0, len(modes), 0, 1])

        ax.set_yticks([0.16, 0.5, 0.83])
        ax.set_yticklabels(["Damping", "Normal", "Escape"], fontsize=10)
        ax.set_xlabel("Step", fontsize=12)
        ax.set_title("SMCScheduler Mode Transitions", fontsize=13)

        from matplotlib.patches import Patch
        colors = [cmap(0), cmap(1), cmap(2)]
        legend_elements = [
            Patch(facecolor=colors[0], label="Normal"),
            Patch(facecolor=colors[1], label="Saddle Escape"),
            Patch(facecolor=colors[2], label="Chattering Damping"),
        ]
        ax.legend(handles=legend_elements, fontsize=9, loc="upper right",
                  bbox_to_anchor=(1.05, 1))
    else:
        ax.text(0.5, 0.5, "No SMC\n(Baseline Only)", ha="center", va="center",
                fontsize=16, color="gray", transform=ax.transAxes)
        ax.set_title("SMCScheduler Mode", fontsize=13)

    plt.tight_layout()
    return fig


# ============================================================
# 主程序
# ============================================================

if __name__ == "__main__":
    print("=" * 60)
    print("SMCScheduler — 2D Monkey Saddle 鞍点逃离可视化测试")
    print("=" * 60)

    # 参数设置
    INIT_X, INIT_Y = -1.0, -1.0  # 起始点（远离鞍点，但在鞍点吸引域内）
    LR = 0.01
    STEPS = 500
    SEED = 42

    func = monkey_saddle

    # --- 运行标准 AdamW ---
    print("\n[1/2] Running standard AdamW ...")
    traj_adamw, losses_adamw, _ = run_optimization(
        func, INIT_X, INIT_Y, LR, STEPS, use_smc=False, seed=SEED
    )
    final_dist_adamw = np.sqrt(traj_adamw[-1, 0] ** 2 + traj_adamw[-1, 1] ** 2)
    print(f"  终点: ({traj_adamw[-1, 0]:.4f}, {traj_adamw[-1, 1]:.4f})")
    print(f"  距鞍点距离: {final_dist_adamw:.4f}")
    print(f"  最终 Loss: {losses_adamw[-1]:.6f}")

    # --- 运行 AdamW + SMCScheduler ---
    print("\n[2/2] Running AdamW + SMCScheduler ...")
    traj_smc, losses_smc, modes = run_optimization(
        func, INIT_X, INIT_Y, LR, STEPS, use_smc=True, seed=SEED
    )
    final_dist_smc = np.sqrt(traj_smc[-1, 0] ** 2 + traj_smc[-1, 1] ** 2)
    escape_count = modes.count("escape")
    damping_count = modes.count("damping")
    print(f"  终点: ({traj_smc[-1, 0]:.4f}, {traj_smc[-1, 1]:.4f})")
    print(f"  距鞍点距离: {final_dist_smc:.4f}")
    print(f"  最终 Loss: {losses_smc[-1]:.6f}")
    print(f"  鞍点逃离触发: {escape_count} 次")
    print(f"  震荡平滑触发: {damping_count} 次")

    # --- 绘图 ---
    print("\n绘制等高线 + 轨迹对比图 ...")
    fig = plot_contour_with_trajectories(
        traj_adamw, traj_smc, losses_adamw, losses_smc, modes, func,
        xlim=(-2, 2), ylim=(-2, 2)
    )

    save_path = os.path.join(os.path.dirname(__file__), "smc_saddle_trajectory.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"图片已保存: {save_path}")
    plt.close(fig)
