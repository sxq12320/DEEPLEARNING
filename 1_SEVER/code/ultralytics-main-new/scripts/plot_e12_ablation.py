"""Render E12 mechanism evidence from measured tensors, predictions and paired GT records."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np


def settings():
    available = {f.name for f in font_manager.fontManager.ttflist}
    cjk = next((f for f in ("Microsoft YaHei", "Noto Sans CJK SC", "SimHei") if f in available), None)
    plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ([cjk] if cjk else []) + ["Arial", "DejaVu Sans"],
                         "font.size": 8, "axes.titlesize": 9,
                         "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.right": False, "axes.spines.top": False})
    return lambda chinese, english: chinese if cjk else english


def export(fig, axes, prefix):
    """The experiment runs on a standard training server. Use the journal-layout auditor when installed."""
    fig.canvas.draw()
    try:
        from audit_panel_alignment import require_matplotlib_panel_alignment
    except ImportError:
        sizes = np.array([a.get_window_extent().transformed(fig.dpi_scale_trans.inverted()).size*72 for a in axes])
        if np.ptp(sizes, axis=0).max() > 1.5:
            raise ValueError("Misaligned evidence panels.")
        with prefix.with_suffix(".alignment.json").open("w", encoding="utf-8") as file:
            json.dump({"check": "equal panel sizes", "sizes_pt": sizes.tolist()}, file)
    else:
        require_matplotlib_panel_alignment(fig, axes=axes, json_out=prefix.with_suffix(".alignment.json"),
                                           tolerance_pt=1.5, strict=True)
    fig.savefig(prefix.with_suffix(".svg"), facecolor="white")
    fig.savefig(prefix.with_suffix(".pdf"), dpi=300, facecolor="white")
    fig.savefig(prefix.with_suffix(".png"), dpi=600, facecolor="white")
    plt.close(fig)
    try:
        from tifffile import imwrite
    except ImportError:
        return
    from PIL import Image
    with Image.open(prefix.with_suffix(".png")) as png:
        imwrite(prefix.with_suffix(".tiff"), np.asarray(png.convert("RGB")), photometric="rgb",
                compression="zlib", resolution=(600, 600), metadata=None)


def contour(ax, mask, color, line):
    if mask.any() and not mask.all():
        ax.contour(mask.astype(float), levels=[.5], colors=[color], linestyles=[line], linewidths=.9)


def cases_figure(directory, cases, protocol, tr):
    if not cases:
        return
    mode = protocol["mechanism"]
    cn_title = {"detail": "持续细节流", "route": "共享识别校正", "phase": "相位掩膜细化"}[mode]
    heat_name = {"detail": tr("头部 P2 校正幅度", "P2 update at mask head"),
                 "route": tr("P3 校正幅度", "P3 recognition update"),
                 "phase": tr("原型细化幅度", "Prototype update")}[mode]
    frozen = protocol["control_type"] == "frozen"
    ctr = tr("关闭模块出口", "Module exits off") if frozen else tr("移除后重训", "Retrained control")
    categories = {"Recovered": "找回", "Contour gain": "轮廓改善", "Lost": "丢失",
                  "Contour loss": "轮廓变差", "Both miss": "均漏检", "Both match": "均命中"}
    arrays = []
    for case in cases:
        with np.load(directory/case["file"], allow_pickle=False) as archive:
            arrays.append({k: archive[k] for k in archive.files})
    vmax = max(float(np.percentile(np.concatenate([d["update"].ravel() for d in arrays]), 99)), 1e-12)
    width, height = 7.20, 1.25 + 1.86*len(cases)
    fig, axs = plt.subplots(len(cases), 4, squeeze=False, figsize=(width, height))
    fig.subplots_adjust(left=.034, right=.977, top=1-.60/height, bottom=.81/height,
                        wspace=.12, hspace=.33)
    fig.text(.034, 1-.15/height, tr("同一果实上检验"+cn_title, "Testing the "+mode+" pathway on the same fruit"),
             fontsize=12, fontweight="bold", va="center")
    titles = [tr("原图与人工标注", "RGB and annotation"), heat_name, ctr, tr("完整 E12", "Full E12")]
    for j, title in enumerate(titles):
        axs[0, j].set_title(title, pad=5, fontweight="bold")
    for i, (case, data) in enumerate(zip(cases, arrays)):
        for j, ax in enumerate(axs[i]):
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if j == 1:
                ax.imshow(data["update"], cmap="magma", vmin=0, vmax=vmax, interpolation="nearest")
            else:
                ax.imshow(data["rgb"], interpolation="nearest")
            contour(ax, data["gt"], "#69DFFF", "dashed")
            if j >= 2:
                tag = "control" if j == 2 else "full"
                contour(ax, data[tag], "#FFAA43", "solid")
                matched = case[tag+"_hit"]
                status = tr("命中", "Hit") if matched else tr("未匹配", "Miss")
                label = "%s   IoU %.3f" % (status, case[tag+"_shown_iou"])
            elif j == 0:
                label = "%s   %d px²" % (tr(categories[case["category"]], case["category"]), case["area640"])
            else:
                label = Path(case["image"]).stem + "   #" + str(case["gt_index"])
            ax.set_xlabel(label, fontsize=7, labelpad=3)
    fig.text(.034, .26/height, tr("青虚线为标注，橙实线为预测。所有列采用同一裁剪。亮色表示较大校正。",
                                  "Cyan dashed: annotation. Orange: prediction. Identical crop in each row."),
             fontsize=7)
    fig.text(.034, .10/height,
             tr("热图为实际更新绝对值的通道均值，共用色阶 0–%.3g。案例按预定规则选取。" % vmax,
                "Update magnitude, mean over channels. Shared range 0–%.3g. Cases selected by fixed rules." % vmax),
             fontsize=7)
    export(fig, list(axs.flat), directory/"01_case_comparison")


def paired_figure(directory, summary, protocol, tr):
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.20))
    fig.subplots_adjust(left=.157, right=.982, bottom=.29, top=.75, wspace=.24)
    groups = ("tiny", "small", "medium", "large", "all")
    y = np.arange(len(groups))[::-1]
    fig.text(.025, .918, tr("完整 E12 相对对照的逐果实变化", "Paired change with the full E12 model"),
             fontsize=12, fontweight="bold")
    if protocol["control_type"] == "frozen":
        comparison = tr("同一权重，关闭指定通路作为对照", "Same weights, specified exits disabled")
    else:
        comparison = tr("对照为独立重训的删减模型", "Control trained separately without the component")
    fig.text(.025, .84, "%s  ·  %d %s" % (comparison, protocol["source_images"], tr("张验证图" if
             protocol["split"] == "val" else "张测试图", protocol["split"]+" images")), fontsize=8)
    for ax, key, interval, title in (
        (axs[0], "delta_recall_pp", "delta_recall_image_bootstrap_ci95_pp", tr("命中率变化", "Recall change")),
        (axs[1], "delta_common_band_pp", "delta_common_band_image_bootstrap_ci95_pp",
         tr("共同命中果实的轮廓变化", "Contour change on shared hits")),
    ):
        ax.axvline(0, lw=.8, color="#828B96", linestyle="--")
        for yy, group in zip(y, groups):
            result = summary[group]
            v, ci = result[key], result[interval]
            if v is None or ci is None:
                continue
            color = "#147E87" if v >= 0 else "#B5582C"
            ax.plot(ci, [yy, yy], color=color, linewidth=1.7, solid_capstyle="round")
            ax.scatter([v], [yy], s=25, c=color, zorder=3)
        ax.set_ylim(-.55, 4.55)
        ax.set_yticks(y)
        ax.tick_params(axis="y", length=0, pad=4)
        ax.grid(axis="y", color="#DEE4E6", linewidth=.5)
        ax.set_axisbelow(True)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=9, pad=7)
        ax.set_xlabel(tr("完整减对照（百分点）", "Full minus control (pp)"))
    axs[0].set_yticklabels(["<256", "256–1023", "1024–4095", "≥4096", tr("全部", "All")])
    axs[0].set_ylabel(tr("标注面积（640栅格，px²）", "GT area at 640 (px²)"), fontsize=7)
    axs[1].set_yticklabels([])
    fig.text(.025, .105, tr("点为配对差值，线为按图像配对重采样的 95% 区间。", "Points: paired differences. Lines: 95% image-bootstrap intervals."), fontsize=7)
    fig.text(.025, .05, tr("左图含全部标注。右图只含双方命中实例，使用半径 2 px 的轮廓带重叠率。", "Left: all GT. Right: two-sided 2 px contour-band IoU on shared matches only."), fontsize=7)
    export(fig, list(axs), directory/"02_paired_effects")


def observed_pr_figure(directory, ranking, protocol, tr):
    with (directory / "observed_pr.csv").open(encoding="utf-8-sig", newline="") as file:
        records = list(csv.DictReader(file))
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.55))
    fig.subplots_adjust(left=.099, right=.972, bottom=.28, top=.72, wspace=.25)
    fig.text(.03, .948, tr("实测精确率–召回率曲线", "Observed precision–recall curves"), fontsize=12, fontweight="bold")
    ctrl = (tr("关闭指定通路", "Exits disabled") if protocol["control_type"] == "frozen"
            else tr("移除后重训", "Retrained control"))
    handles = []
    for arm, label, color, style in (("control", ctrl, "#B45C37", "--"), ("full", tr("完整 E12", "Full E12"), "#007B83", "-")):
        rows = [r for r in records if r["arm"] == arm]
        recall = np.array([float(r["recall"]) for r in rows])
        precision = np.array([float(r["precision"]) for r in rows])
        for j, ax in enumerate(axs):
            visible = precision if j == 0 else np.where((recall >= .60) & (precision >= .58), precision, np.nan)
            line, = ax.plot(recall, visible, color=color, linestyle=style, lw=1.1, label=label)
            if j == 0:
                handles.append(line)
            s = ranking[arm]
            if s["r_at_precision90"] is not None:
                ax.scatter(s["r_at_precision90"], s["precision_at_selected_threshold"], s=21,
                           marker="o" if arm == "full" else "s", facecolors="none", edgecolors=color, zorder=4)
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.022, .89), ncol=2, frameon=False, fontsize=8)
    axs[0].set(xlim=(0, 1), ylim=(0, 1.015), ylabel=tr("精确率", "Precision"))
    axs[1].set(xlim=(.60, 1), ylim=(.58, 1.015))
    for ax, title in zip(axs, (tr("全部阈值", "All thresholds"), tr("高召回区域", "High-recall detail"))):
        ax.axhline(.90, lw=.7, color="#777777", linestyle=":", zorder=0)
        ax.set_title(title, pad=4)
        ax.set_xlabel(tr("召回率", "Recall"))
        ax.grid(color="#DEE4E6", linewidth=.45)
        ax.set_axisbelow(True)
    fig.text(.03, .099, tr("空心标记为精确率 ≥90% 时的最大实测召回，点线为 90% 精确率。", 
                            "Markers show the maximum observed recall at precision ≥90%. Dotted line: 90% precision."), fontsize=7)
    fig.text(.03, .043, tr("整图，Mask IoU 0.50。曲线在实测范围结束，不向召回率 1 补点。", 
                            "Whole image, mask IoU 0.50. Each curve ends at its last observed threshold."), fontsize=7)
    export(fig, list(axs), directory / "03_observed_pr")


def small_fruit_figure(directory, fixed, aligned, tr):
    if aligned["groups"] is None:
        return
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.6))
    fig.subplots_adjust(left=.085, right=.98, bottom=.295, top=.73, wspace=.24)
    fig.text(.030, .944, tr("细节分支的收益是否取决于阈值", "Do detail-path gains depend on the operating point?"),
             fontsize=12, fontweight="bold")
    handles = []
    for ax, key, title in zip(axs, ("tiny", "small"), (tr("极小果", "Tiny"), tr("小果", "Small"))):
        base, matched = fixed[key], aligned["groups"][key]
        n = base["n"]
        ax.set_title("%s   n = %d" % (title, n))
        ax.set_xticks([0, 1])
        ax.set_xticklabels([tr("分数 ≥0.25", "Score ≥0.25"), tr("全局 P ≥90%", "Overall P ≥90%")])
        ax.set_xlim(-.5, 1.5)
        ax.set_ylim(0, 100)
        for arm, offset, color, label in (("control", -.18, "#B45C37", tr("对照", "Control")),
                                          ("full", .18, "#007B83", tr("完整 E12", "Full E12"))):
            values = np.array([base[arm+"_hits"], matched[arm]])
            heights = values*100/n if n else [0., 0.]
            bars = ax.bar(np.array([0, 1])+offset, heights, width=.32, color=color, label=label)
            if ax is axs[0]:
                handles.append(bars)
            for bar, hits in zip(bars, values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+2, str(hits), ha="center", fontsize=7)
    axs[0].set_ylabel(tr("组内召回率（%）", "Within-group recall (%)"))
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(.020, .883), ncol=2, frameon=False)
    fig.text(.03, .105, tr("柱顶为命中果实数。按 640 栅格面积分组，极小果 <256 px²，小果 256–1023 px²。",
                           "Labels give hit counts. GT areas at 640: tiny <256 px², small 256–1023 px²."), fontsize=7)
    fig.text(.03, .048, tr("右侧工作点为各模型在整体验证集达到 ≥90% 精确率时的最大实测召回阈值。",
                           "At the right operating point, each arm uses its own threshold selected on all validation GT."), fontsize=7)
    export(fig, list(axs), directory/"04_small_fruit_operating_points")


def render(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory/name).read_text(encoding="utf-8"))
    tr = settings()
    protocol, summary = read("protocol.json"), read("summary.json")
    cases_figure(directory, read("cases.json"), protocol, tr)
    paired_figure(directory, summary, protocol, tr)
    if (directory / "observed_pr.csv").is_file():
        observed_pr_figure(directory, read("ranking.json"), protocol, tr)
    if protocol["mechanism"] == "detail" and (directory/"precision90_by_size.json").is_file():
        small_fruit_figure(directory, summary, read("precision90_by_size.json"), tr)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path, help="Existing VIS_E12_ABLATION output")
    render(parser.parse_args().output)
