"""Original mathematical illustrations and traceable empirical plots for J11.

Analytic examples are explicitly distinct from experimental observations.
Generated JSON/CSV records are the source data for the PDF, not new experiments.
"""

import csv
import hashlib
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sympy as sp
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

SKILL = Path("C:/Users/33836/.agents/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL))
from audit_panel_alignment import require_matplotlib_panel_alignment

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
ASSETS = HERE / "figures"
WORK = Path("E:/mastercode/_work")
BLUE, PURPLE, ORANGE = "#2879AD", "#7754A0", "#B67C35"
TEXT, GRAY = "#20374B", "#7D8A96"
PALE, LAVENDER = "#EAF3F9", "#F0EBF6"
WIDTH = 170 / 25.4

for font in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/arial.ttf"):
    font_manager.fontManager.addfont(font)
matplotlib.rcParams.update(
    {
        "font.family": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
        "font.size": 8.2,
        "axes.labelsize": 8.2,
        "axes.titlesize": 9,
        "axes.titleweight": "bold",
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
        "axes.unicode_minus": False,
        "mathtext.fontset": "dejavusans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "text.color": TEXT,
        "axes.labelcolor": TEXT,
        "axes.edgecolor": GRAY,
        "xtick.color": TEXT,
        "ytick.color": TEXT,
        "savefig.dpi": 300,
    }
)


def label(ax, letter):
    ax.annotate(
        letter,
        (0, 1),
        xycoords="axes fraction",
        xytext=(-12, 9),
        textcoords="offset points",
        weight="bold",
        fontsize=10,
    )


def save(fig, stem, panels=True):
    fig.canvas.draw()
    require_matplotlib_panel_alignment(
        fig,
        json_out=ASSETS / f"{stem}.alignment.json",
        tolerance_pt=1.5,
        require_panel_labels=panels,
        strict=True,
    )
    fig.savefig(ASSETS / f"{stem}.pdf")
    fig.savefig(ASSETS / f"{stem}.svg")
    fig.savefig(ASSETS / f"{stem}.png", dpi=600)
    plt.close(fig)


def box(ax, x, y, w, h, text, face=PALE, edge=BLUE, fontsize=8.2):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.04", fc=face, ec=edge, lw=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize, linespacing=1.5)


def arrow(ax, start, end, color=GRAY, dashed=False):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=8,
            lw=1,
            color=color,
            linestyle="--" if dashed else "-",
            shrinkA=1,
            shrinkB=1,
        )
    )


def symbolic_checks():
    d, delta = sp.symbols("d delta", positive=True)
    iou = (d - delta) ** 2 / (2 * d**2 - (d - delta) ** 2)
    a, b, c, v1, v2, v3 = sp.symbols("a b c v1 v2 v3", real=True)
    z = sp.exp(a) + sp.exp(b) + sp.exp(c)
    alpha = sp.exp(a) / z
    mixture = (sp.exp(a) * v1 + sp.exp(b) * v2 + sp.exp(c) * v3) / z
    assert sp.simplify(sp.diff(mixture, a) - alpha * (v1 - mixture)) == 0
    eta, beta, gamma, norm = sp.symbols("eta beta gamma norm", positive=True)
    bound = (-eta * (1 - gamma) + beta * eta**2 * (1 + gamma) ** 2 / 2) * norm**2
    limit = 2 * (1 - gamma) / (beta * (1 + gamma) ** 2)
    assert sp.simplify(bound.subs(eta, limit)) == 0
    null = sp.Matrix([[1, 1]]) * sp.Matrix([1, -1])
    assert null == sp.zeros(1, 1)
    assert [str(iou.subs({d: x, delta: 1})) for x in (4, 8, 16)] == ["9/23", "49/79", "225/287"]
    return {
        "sympy_version": sp.__version__,
        "fusion_gradient": "exact residual = 0",
        "iou_one_pixel_xy": {str(x): str(iou.subs({d: x, delta: 1})) for x in (4, 8, 16)},
        "compression_kernel": [1, -1],
        "gradient_budget_boundary": "exact residual = 0",
        "scope": "Exact identities under stated assumptions; no AP or global convergence theorem.",
    }


def small_geometry(audit):
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.75))
    fig.subplots_adjust(left=0.085, right=0.98, bottom=0.22, top=0.85, wspace=0.45)
    ax, bx = axes
    offset = np.linspace(0, 3.9, 200)
    for side, color in ((4, ORANGE), (8, PURPLE), (16, BLUE)):
        overlap = (side - offset) ** 2
        ax.plot(offset, overlap / (2 * side**2 - overlap), color=color, lw=1.6, label=f"边长 {side} px")
    ax.axhline(0.5, color=GRAY, lw=0.8, ls="--")
    ax.set(xlabel="x、y 同时偏移 / px", ylabel="掩膜 IoU", ylim=(0, 1.04), xlim=(0, 3.9), title="方形掩膜：解析示例")
    ax.legend(loc="upper right", frameon=False)
    splits = audit["dataset"]["splits"]
    bins, rows = [], []
    for name in ("train", "val"):
        x = splits[name]["counts"]
        values = [x["min_side_lt8"], x["min_side_lt16"] - x["min_side_lt8"], x["instances"] - x["min_side_lt16"]]
        bins.append(100 * np.array(values) / x["instances"])
        rows.extend(
            {"split": name, "range": r, "count": n, "instances": x["instances"]}
            for r, n in zip(("lt8", "8to16", "ge16"), values)
        )
    bins = np.array(bins)
    xpos = np.arange(3)
    for k, (name, color) in enumerate((("训练集", GRAY), ("验证集", BLUE))):
        bars = bx.bar(xpos + (k - 0.5) * 0.33, bins[k], 0.30, color=color, label=name)
        bx.bar_label(bars, fmt="%.1f", padding=3, fontsize=8)
    bx.set(
        xticks=xpos,
        xticklabels=("< 8", "8–16", "≥ 16"),
        ylabel="实例比例 / %",
        xlabel="640 输入下 GT 短边 / px",
        ylim=(0, 94),
        title="正式划分：静态几何审计",
    )
    bx.legend(loc="upper left", frameon=False)
    label(ax, "a")
    label(bx, "b")
    with (ASSETS / "small_geometry_source.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    save(fig, "02_small_geometry")


def rearrangement():
    fig, ax = plt.subplots(figsize=(WIDTH, 1.8))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.97)
    ax.set(xlim=(0, 10), ylim=(0, 2.5))
    ax.axis("off")
    colors = ("#BCD9EB", "#DDD0EE", "#F3DCC0", "#D2E7DC")
    for index, (x, y) in enumerate(((0.4, 1.35), (1.05, 1.35), (0.4, 0.7), (1.05, 0.7))):
        ax.add_patch(Rectangle((x, y), 0.62, 0.62, fc=colors[index], ec="white"))
        ax.text(x + 0.31, y + 0.31, "abcd"[index], ha="center", va="center", fontsize=11)
    ax.text(1.02, 0.3, "一个 2×2 相位块", ha="center", fontsize=8.2)
    box(ax, 2.05, 0.92, 1.25, 0.88, "Pixel\nunshuffle")
    arrow(ax, (1.7, 1.37), (2.03, 1.37))
    for index in range(4):
        box(ax, 3.85 + index * 0.53, 1.13, 0.44, 0.48, "abcd"[index], face=colors[index], edge=GRAY)
    ax.text(4.83, 0.76, "四个通道：只是重排", ha="center", fontsize=8.2)
    arrow(ax, (3.32, 1.37), (3.82, 1.37))
    box(ax, 6.28, 0.92, 1.2, 0.88, "学习投影\nW + BN")
    arrow(ax, (5.98, 1.37), (6.26, 1.37))
    box(ax, 8.1, 0.92, 1.45, 0.88, "紧凑特征\n未必可逆", face="#FBF1E5", edge=ORANGE)
    arrow(ax, (7.5, 1.37), (8.08, 1.37))
    ax.text(3.06, 2.22, "置换可逆：UᵀU = I", fontsize=9, color=BLUE, ha="center")
    ax.text(7.9, 2.22, "秩不足：可能丢失方向", fontsize=9, color=ORANGE, ha="center")
    save(fig, "03_phase_rearrangement", panels=False)


def competitive():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.45))
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.25, top=0.82, wspace=0.40)
    ax, bx = axes
    ax.set(xlim=(0, 5), ylim=(0, 3.2))
    ax.axis("off")
    for y, text in ((2.15, "浅层细节"), (1.05, "同尺度特征"), (-0.05, "深层语义")):
        box(ax, 0.05, y + 0.1, 1.35, 0.65, text)
        arrow(ax, (1.44, y + 0.43), (2.07, 1.43))
    box(ax, 2.1, 1.1, 1.25, 0.7, "对齐后\n竞争加权", face=LAVENDER, edge=PURPLE)
    arrow(ax, (3.39, 1.43), (3.85, 1.43))
    box(ax, 3.9, 1.1, 0.9, 0.7, "精炼\n输出")
    ax.set_title("一次式融合；权重和为 1", pad=13)
    weight = np.linspace(0, 1, 150)
    value = 1.0 * weight + 0.1 * (1 - weight)
    bx.plot(weight, value, color=BLUE, lw=1.8)
    bx.scatter([1 / 3], [0.4], s=30, color=ORANGE, zorder=3)
    bx.set(xlim=(0, 1), ylim=(0, 1.08), xlabel="细节源权重 α", ylabel="混合响应", title="构造示例：细节 1，其他源 0.1")
    bx.text(0.46, 0.12, "均匀混合：响应 0.4\n竞争权重并非天然正确", fontsize=8, color=ORANGE)
    label(ax, "a")
    label(bx, "b")
    save(fig, "04_competitive_fusion")


def differential():
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.5))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.23, top=0.83, wspace=0.42)
    p, q = np.array([0.5, 0.2, 0.2, 0.1]), np.array([0.1, 0.2, 0.2, 0.5])
    xpos = np.arange(4)
    axes[0].bar(xpos - 0.16, p, 0.30, color=BLUE, label="读取分布 A₁")
    axes[0].bar(xpos + 0.16, q, 0.30, color=PURPLE, label="读取分布 A₂")
    axes[0].set(
        xticks=xpos,
        xticklabels=("1", "2", "3", "4"),
        xlabel="压缩上下文槽位",
        ylabel="权重",
        ylim=(0, 0.7),
        title="人工概率示例：各行和均为 1",
    )
    axes[0].legend(frameon=False, loc="upper center", ncol=1)
    vals = [
        float((p - q) @ np.ones(4)),
        float((p - q) @ np.array([1, 0, 0, 0])),
        float((p - q) @ np.array([0, 0, 0, 1])),
    ]
    bars = axes[1].bar(np.arange(3), vals, color=(GRAY, BLUE, ORANGE), width=0.55)
    axes[1].bar_label(bars, fmt="%.1f", padding=4, fontsize=8)
    axes[1].axhline(0, color=GRAY, lw=0.8)
    axes[1].set(
        xticks=range(3),
        xticklabels=("常量", "槽位 1 突起", "槽位 4 突起"),
        ylabel="差分读出",
        ylim=(-0.62, 0.62),
        title="消常量 ≠ 正确识别果实",
    )
    label(axes[0], "a")
    label(axes[1], "b")
    save(fig, "05_differential_context")


def decoder():
    fig, ax = plt.subplots(figsize=(WIDTH, 2.3))
    fig.subplots_adjust(left=0.02, right=0.98, bottom=0.05, top=0.96)
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")
    box(ax, 0.1, 1.93, 1.75, 0.72, "P3/P4/P5\n8400 个候选")
    box(ax, 2.45, 1.93, 1.7, 0.72, "实例系数 C\n每候选 32 维")
    arrow(ax, (1.9, 2.29), (2.4, 2.29))
    box(ax, 0.1, 0.42, 1.75, 0.76, "P1/P2 + P3\n细节 + 语义")
    box(ax, 2.45, 0.42, 1.7, 0.76, "32 张原型 P\n320×320")
    arrow(ax, (1.9, 0.80), (2.4, 0.80))
    box(ax, 5.13, 1.08, 1.60, 0.85, "线性合成\nL = CP", face=LAVENDER, edge=PURPLE)
    arrow(ax, (4.2, 2.29), (5.1, 1.70))
    arrow(ax, (4.2, 0.80), (5.1, 1.30))
    box(ax, 7.55, 1.08, 2.0, 0.85, "Sigmoid + 裁剪\n每果实一张掩膜")
    arrow(ax, (6.78, 1.50), (7.5, 1.50))
    ax.text(5.9, 0.20, "原型更细不能凭空产生检测候选", ha="center", color=ORANGE, fontsize=9)
    save(fig, "06_mask_decoder", panels=False)


def paired_evidence(evidence):
    selected = []
    for short, prefix in (("I20", "I20_corrected"), ("V12_04", "V12_04_shared")):
        item = next(r for r in evidence["paired"] if r["name"].startswith(prefix) and r["mode"] == "fine")
        g, t = item["summary"]["global"], item["summary"]["trustedmask"]
        assert g["tiny_n"] == t["tiny_n"] == 161
        selected.append(
            {
                "name": short,
                "global_n": g["tiny_matched"],
                "fine_n": t["tiny_matched"],
                "tiny_n": 161,
                "global_ms": g["median_ms"],
                "fine_ms": t["median_ms"],
                "source": item["path"],
            }
        )
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 2.6))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.23, top=0.82, wspace=0.4)
    x = np.arange(2)
    for mode, shift, color in (("global", -0.16, GRAY), ("fine", 0.16, BLUE)):
        rates = [100 * row[f"{mode}_n"] / 161 for row in selected]
        bars = axes[0].bar(x + shift, rates, 0.30, color=color, label="全图" if mode == "global" else "细切片 + 合并")
        axes[0].bar_label(bars, labels=[f"{r[f'{mode}_n']}/161" for r in selected], padding=3, fontsize=8)
    axes[0].set(
        xticks=x,
        xticklabels=("I20", "V12_04"),
        ylabel="Tiny recall / %",
        ylim=(0, 100),
        title="同权重配对诊断：旧验证集",
    )
    axes[0].legend(loc="upper left", frameon=False)
    ratios = [r["fine_ms"] / r["global_ms"] for r in selected]
    bars = axes[1].bar(x, ratios, color=ORANGE, width=0.52)
    axes[1].bar_label(bars, labels=[f"{r:.2f}×" for r in ratios], padding=5, fontsize=8.5)
    axes[1].axhline(1, color=GRAY, lw=0.8, ls="--")
    axes[1].set(
        xticks=x, xticklabels=("I20", "V12_04"), ylabel="耗时 / 全图耗时", ylim=(0, 4.5), title="召回收益伴随多视图代价"
    )
    label(axes[0], "a")
    label(axes[1], "b")
    with (ASSETS / "paired_evidence_source.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(selected[0]))
        writer.writeheader()
        writer.writerows(selected)
    save(fig, "07_paired_evidence")


def endpoint():
    fig, ax = plt.subplots(figsize=(WIDTH, 2.05))
    fig.subplots_adjust(left=0.085, right=0.96, bottom=0.24, top=0.84)
    recall, precision = np.array([0.2, 0.5, 0.8]), np.array([1.0, 0.9, 0.8])
    ax.plot(recall, precision, "o-", color=BLUE, lw=1.8, label="构造观测点（非实验）")
    ax.plot([0.8, 0.8, 1.0], [0.8, 0, 0], color=GRAY, ls="--", lw=1.4, label="零端点补齐：非真实新增观测")
    ax.set(
        xlim=(0, 1.02),
        ylim=(0, 1.07),
        xlabel="Recall：横轴不是置信度",
        ylabel="Precision",
        title="解释绘图约定，不篡改实际 PR 或 AP",
    )
    ax.legend(loc="lower left", frameon=False)
    save(fig, "08_pr_endpoint", panels=False)


def gradient_budget():
    fig, ax = plt.subplots(figsize=(WIDTH, 1.95))
    fig.subplots_adjust(left=0.09, right=0.96, bottom=0.25, top=0.84)
    gamma = np.linspace(0, 0.99, 180)
    boundary = 2 * (1 - gamma) / (1 + gamma) ** 2
    ax.plot(gamma, boundary, color=BLUE, lw=1.8)
    ax.fill_between(gamma, 0, boundary, color=PALE)
    ax.text(0.10, 0.30, "满足假设时：主损失一步下降区域", fontsize=8.5, color=BLUE)
    ax.set(
        xlim=(0, 1),
        ylim=(0, 2.15),
        xlabel="辅助梯度预算 γ",
        ylabel="βη 的允许上限",
        title="条件性数学建议；J11 未实现此控制律",
    )
    save(fig, "09_gradient_budget", panels=False)


def main():
    ASSETS.mkdir(parents=True, exist_ok=True)
    audit_path = WORK / "20261001_j_strict_review_v3/audit.json"
    evidence_path = WORK / "20261001_j/evidence/bounded_evidence.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
    checks = symbolic_checks()
    old = ROOT / "docs/J_SERIES_20261001/paper_figures_readable_20261001/J11_通俗版架构图.pdf"
    shutil.copy2(old, ASSETS / "01_architecture_overview.pdf")
    small_geometry(audit)
    rearrangement()
    competitive()
    differential()
    decoder()
    paired_evidence(evidence)
    endpoint()
    gradient_budget()
    sources = [
        audit_path,
        evidence_path,
        old,
        ROOT / "0_orange_yaml/J_series/J11_synthesis.yaml",
        ROOT / "ultralytics/nn/modules/citrus_j.py",
        ROOT / "ultralytics/utils/citrus_j_loss.py",
    ]
    record = {
        "checks": checks,
        "sources_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "scope": "No training, no model changes, no dataset modifications; analytic examples labelled.",
    }
    (HERE / "verification.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(checks, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
