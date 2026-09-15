"""Editable code-faithful paper diagrams; export SVG/PDF/300dpi PNG locally."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/E_V9_REVIEW_20260915/figures"
INK = "#203040"
BLUE = "#0072B2"
ORANGE = "#A45A00"
GREEN = "#007958"
FILL = {"old": "#EAF3FA", "new": "#FFF1DE", "out": "#E5F5ED", "loss": "#F5EDF4"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8, "svg.fonttype": "none", "pdf.fonttype": 42})


def canvas(title, height=4.6):
    fig, ax = plt.subplots(figsize=(7.2, height))
    fig.subplots_adjust(left=0.025, right=0.975, top=0.92, bottom=0.055)
    ax.set(xlim=(0, 12), ylim=(0, 7))
    ax.axis("off")
    fig.suptitle(title, fontsize=11, fontweight="bold", color=INK, y=0.975)
    return fig, ax, []


def node(ax, registry, x, y, w, h, label, kind="old", fs=8):
    color = ORANGE if kind == "new" else GREEN if kind == "out" else BLUE
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.035,rounding_size=0.09",
        facecolor=FILL[kind],
        edgecolor=color,
        linewidth=0.9,
        zorder=3,
    )
    ax.add_patch(patch)
    text = ax.text(
        x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs, color=INK, zorder=4, linespacing=1.3
    )
    registry.append((patch, text))


def route(ax, points, color=BLUE, dashed=False):
    for a, b in zip(points[:-2], points[1:-1]):
        ax.plot([a[0], b[0]], [a[1], b[1]], color=color, lw=0.95, linestyle="--" if dashed else "-", zorder=1)
    ax.add_patch(
        FancyArrowPatch(
            points[-2],
            points[-1],
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=0.95,
            color=color,
            linestyle="--" if dashed else "-",
            zorder=2,
        )
    )


def save(fig, registry, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    issues = []
    for patch, text in registry:
        pb, tb = patch.get_window_extent(renderer), text.get_window_extent(renderer)
        if tb.x0 < pb.x0 or tb.x1 > pb.x1 or tb.y0 < pb.y0 or tb.y1 > pb.y1:
            issues.append(text.get_text())
    for ext in ("svg", "pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, facecolor="white")
    plt.close(fig)
    return {
        "figure": name,
        "text_outside_nodes": issues,
        "minimum_node_font_pt": min(t.get_fontsize() for _, t in registry),
        "vector_formats": ["SVG editable text", "PDF embedded fonts"],
        "png_dpi": 300,
        "palette": "Okabe-Ito-inspired dark blue/orange/green; labeled nodes do not rely on colour",
    }


def architecture():
    fig, ax, reg = canvas("E V9 | Detection-preserving, task-separated mask architecture")
    node(ax, reg, 0.12, 4.7, 1.25, 0.85, "RGB view\n640 × 640")
    node(
        ax,
        reg,
        1.85,
        3.6,
        2.05,
        2.3,
        "Shared hybrid\nbackbone\n\nC2 / C3 / C4: C3k2\nC5: partial mixing\nSPPF + C2PSA",
        fs=7.5,
    )
    node(ax, reg, 4.5, 4.65, 2.2, 1.15, "Asymmetric neck\nP3 / P4 / C5\nP4 reconstruction kept", fs=7.5)
    node(ax, reg, 7.3, 4.65, 2.2, 1.15, "Instance towers\nboxes · scores\nmask coefficients", fs=7.5)
    node(ax, reg, 10.25, 3.0, 1.55, 1.25, "Instance\nmasks +\nquality scores", "out", fs=7.5)
    node(ax, reg, 1.85, 1.6, 2.05, 1.0, "Semantic-guided\nP2 detail\n16 channels", fs=7.5)
    node(ax, reg, 4.5, 1.6, 2.2, 1.0, "Mask-only fusion\nnative C4 / C5\n+ local residual", "new", fs=7.5)
    node(ax, reg, 7.3, 1.6, 2.2, 1.0, "Compact mask basis\n32-channel decoder\nP3 → P2", "new", fs=7.5)
    node(ax, reg, 10.25, 1.6, 1.55, 1.0, "Phase detail\nrefinement\nstride 2", fs=7.5)
    route(ax, [(1.37, 5.12), (1.85, 5.12)])
    route(ax, [(3.9, 5.12), (4.5, 5.12)])
    route(ax, [(6.7, 5.12), (7.3, 5.12)])
    route(ax, [(9.5, 5.12), (11.02, 5.12), (11.02, 4.25)])
    route(ax, [(2.85, 3.6), (2.85, 2.6)])
    ax.text(2.96, 3.05, "C2", fontsize=7, color=BLUE)
    route(ax, [(4.75, 4.65), (4.2, 4.05), (4.2, 2.9), (3.5, 2.9), (3.5, 2.6)])
    ax.text(4.24, 3.55, "P3", fontsize=7, color=BLUE)
    route(ax, [(3.9, 2.1), (4.5, 2.1)], ORANGE)
    route(ax, [(3.9, 3.9), (5.6, 3.9), (5.6, 2.6)], ORANGE)
    ax.text(4.45, 4.02, "C4 / C5 context", fontsize=7, color=ORANGE)
    route(ax, [(6.7, 2.1), (7.3, 2.1)], ORANGE)
    route(ax, [(7.0, 5.12), (7.0, 3.3), (8.4, 3.3), (8.4, 2.6)])
    ax.text(7.1, 3.43, "Relayed P3", fontsize=7, color=BLUE)
    route(ax, [(9.5, 2.1), (10.25, 2.1)])
    route(ax, [(11.02, 2.6), (11.02, 3.0)], GREEN)
    # Inherited detail relay returns to P3 BEFORE the mask-only correction.
    route(ax, [(2.15, 2.6), (2.15, 3.2), (1.55, 3.2), (1.55, 6.3), (7.0, 6.3), (7.0, 5.12)], BLUE, True)
    ax.plot(7.0, 5.12, "o", color=BLUE, ms=3, zorder=5)
    ax.text(4.5, 6.47, "Inherited detail → P3 relay, AFTER the neck (one pass)", fontsize=7, ha="center", color=BLUE)
    route(ax, [(2.1, 3.6), (1.45, 3.6), (1.45, 0.9), (11.02, 0.9), (11.02, 1.6)], BLUE, True)
    ax.text(
        6.6,
        1.04,
        "Stem /2 → exact phase rearrangement → learned fine-mask correction",
        fontsize=7,
        ha="center",
        color=BLUE,
    )
    ax.text(
        6,
        0.24,
        "V9_08 candidate: 2.151 M parameters · 7.093 GFLOPs/view @640 · accuracy not yet measured",
        fontsize=7.5,
        ha="center",
        color=INK,
    )
    return save(fig, reg, "E_V9_architecture")


def modules():
    fig, ax, reg = canvas("E V9 | Mask fusion, efficient decoding and targeted supervision", height=5.6)
    ax.text(
        0.12, 6.8, "a   Mask-only context / detail reconciliation (16 channels)", fontsize=9, weight="bold", color=INK
    )
    node(ax, reg, 0.15, 5.55, 1.55, 0.65, "C4 / C5\n1×1 projections", fs=7.5)
    node(ax, reg, 2.25, 5.55, 2.0, 0.65, "Mean + 3×3\nupsample to P2", "new", fs=7.5)
    node(ax, reg, 0.15, 4.4, 1.55, 0.65, "P2 detail D\nsemantic-guided", fs=7.5)
    node(ax, reg, 2.25, 4.4, 2.0, 0.65, "L = AvgPool(D)\nH = D − L", "new", fs=7.5)
    node(ax, reg, 4.85, 5.0, 2.15, 1.15, "Gate: [S, L, |H|]\n1×1 → softmax\nw₁ + w₂ = 1", "new", fs=7.5)
    node(ax, reg, 7.55, 5.0, 2.15, 1.15, "U = w₁(S − L)\n+ w₂ DWConv(H)", "new", fs=7.5)
    node(ax, reg, 10.25, 5.0, 1.6, 1.15, "D′ = D\n+ tanh(a) U\na starts at 0.1", "out", fs=7.2)
    route(ax, [(1.7, 5.87), (2.25, 5.87)], ORANGE)
    route(ax, [(4.25, 5.87), (4.85, 5.87)], ORANGE)
    ax.text(4.42, 6.04, "S", fontsize=7.5, color=ORANGE)
    route(ax, [(1.7, 4.72), (2.25, 4.72)], ORANGE)
    route(ax, [(4.25, 4.72), (5.4, 4.72), (5.4, 5.0)], ORANGE)
    route(ax, [(7, 5.57), (7.55, 5.57)], ORANGE)
    route(ax, [(9.7, 5.57), (10.25, 5.57)], ORANGE)
    ax.text(
        6,
        4.03,
        "Correction feeds masks only; no C4/C5-width computation at P2; no CPU edge operator.",
        fontsize=7.5,
        ha="center",
        color=INK,
    )
    ax.axhline(3.73, color="#D8DEE5", lw=0.7)
    ax.text(
        0.12,
        3.4,
        "b   Replace the wide prototype stack; retain fine phase decoding",
        fontsize=9,
        weight="bold",
        color=INK,
    )
    node(ax, reg, 0.15, 2.15, 2.0, 0.8, "P3 → 32 channels\nupsample to P2", "new", fs=7.5)
    node(ax, reg, 2.8, 2.15, 2.0, 0.8, "Add projected D′\n16 → 32 channels", "new", fs=7.5)
    node(ax, reg, 5.45, 2.15, 2.0, 0.8, "Depthwise 3×3\n+ pointwise 1×1", "new", fs=7.5)
    node(ax, reg, 8.1, 2.15, 1.45, 0.8, "32 mask\nbases", "out", fs=7.5)
    node(ax, reg, 10.2, 2.15, 1.65, 0.8, "Inherited phase\nrefinement /2", fs=7.5)
    for x1, x2 in ((2.15, 2.8), (4.8, 5.45), (7.45, 8.1), (9.55, 10.2)):
        route(ax, [(x1, 2.55), (x2, 2.55)], ORANGE)
    ax.axhline(1.85, color="#D8DEE5", lw=0.7)
    ax.text(0.12, 1.54, "c   Training-only additions, tested separately", fontsize=9, weight="bold", color=INK)
    node(
        ax,
        reg,
        0.15,
        0.35,
        3.5,
        0.85,
        "Tiny Dice: visible area <256 px²\nmean mask coefficients per GT\nbase BCE kept for every positive",
        "loss",
        fs=7.5,
    )
    node(
        ax,
        reg,
        4.15,
        0.35,
        3.5,
        0.85,
        "Quality negatives: ≤32 per image\nexclude all GT boxes +8 px margin\nno new inference branch",
        "loss",
        fs=7.5,
    )
    node(
        ax,
        reg,
        8.15,
        0.35,
        3.7,
        0.85,
        "Keep visible-boundary / neighbor loss\nno convex fill or circle target\nCopy-Paste recipe inherited",
        "loss",
        fs=7.5,
    )
    return save(fig, reg, "E_V9_modules")


if __name__ == "__main__":
    report = [architecture(), modules()]
    (OUT / "layout_checks.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
