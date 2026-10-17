"""Plot cached exploration evidence. No inference, filtering, or synthetic observations.

Figure contract: native and derived fine views rescue small GT at an FP cost.
Three complementary roles are miss-stage decomposition, scale stratification,
and the precision-recall tradeoff at a fixed operating threshold. All panels
use one frozen historical model, old validation members, and common640 masks.
Counts are finite-sample descriptions, not seed means. No error bars are invented.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--figure-skill-scripts", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Preserve completed figures; choose a new output directory")
    sys.path.insert(0, str(args.figure_skill_scripts))
    from audit_panel_alignment import require_matplotlib_panel_alignment

    args.output.mkdir(parents=True)
    with (args.run / "records.csv").open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if float(row["threshold"]) == 0.25]
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    modes = ("global", "upsample_linear", "upsample_area", "fine")
    labels = ("Global", "Derived LINEAR", "Derived AREA", "Native fine")
    colors = ("#7A8591", "#648EAF", "#D39B54", "#317A70")
    mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                         "font.size": 7, "svg.fonttype": "none", "pdf.fonttype": 42,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.7,
                         "legend.frameon": False})
    fig, axes = plt.subplots(1, 3, figsize=(7.204724, 3.346457))  # 183 by 85 mm
    fig.subplots_adjust(left=0.085, right=0.985, bottom=0.33, top=0.80, wspace=0.45)
    for label, ax in zip("abc", axes):
        ax.annotate(label, xy=(0, 1), xycoords="axes fraction", xytext=(-14, 12),
                    textcoords="offset points", fontweight="bold", fontsize=9)

    bottom = np.zeros(4)
    stages = (("S0", "No box witness", "#B8BEC4"), ("S1", "Low score", "#92B6D0"),
              ("S2", "Removal proxy", "#DAB272"), ("S3", "Mask failure proxy", "#8EA89C"),
              ("matching_competition", "Matching conflict", "#AD96B2"), ("unresolved", "Unresolved", "#CE9494"))
    for stage, label, color in stages:
        counts = np.asarray([sum(row["mode"] == mode and row["stage"] == stage for row in rows) for mode in modes])
        if counts.any():
            axes[0].barh(np.arange(4), counts, left=bottom, color=color, height=0.65, label=label)
            bottom += counts
    for y, total in enumerate(bottom):
        axes[0].text(total + 3, y, str(int(total)), va="center", fontsize=7)
    axes[0].set_yticks(np.arange(4), ("Global", "LINEAR", "AREA", "Native"))
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, bottom.max() * 1.20)
    axes[0].set_xlabel("Missed GT count")
    axes[0].set_title("Work-threshold misses", fontsize=8, pad=14)
    axes[0].legend(loc="upper left", bbox_to_anchor=(0, -0.31), fontsize=6, borderaxespad=0)

    bounds = (0, 8, 16, 32, 64, float("inf"))
    bins = ("<8", "8–16", "16–32", "32–64", "≥64")
    source_data = []
    for mode, label, color in zip(modes, labels, colors):
        recalls = []
        for index in range(5):
            stratum = [row for row in rows if row["mode"] == mode
                       and bounds[index] <= float(row["size640"]) < bounds[index + 1]]
            hits = sum(row["stage"] == "S4" for row in stratum)
            recalls.append(100 * hits / len(stratum) if stratum else np.nan)
            source_data.append(dict(mode=mode, size_bin=bins[index], gt=len(stratum), hit=hits,
                                    recall_percent=recalls[-1]))
        axes[1].plot(np.arange(5), recalls, "o-", label=label, color=color, linewidth=1.15, markersize=3)
    axes[1].set_xticks(np.arange(5), bins, rotation=30, ha="right", rotation_mode="anchor")
    axes[1].set_ylim(0, 105)
    axes[1].set_ylabel("Mask recall (%)")
    axes[1].set_xlabel("Visible-mask size (px)")
    axes[1].set_title("The smallest fruit gain most", fontsize=8, pad=14)
    axes[1].grid(axis="y", alpha=0.17)
    axes[1].legend(loc="upper left", bbox_to_anchor=(0, -0.31), fontsize=6, borderaxespad=0)

    for mode, label, color in zip(modes, labels, colors):
        record = audit["pr_raw_prefix"][mode]
        recall = audit["metric_reproduction"][mode]["recall25"] * 100
        precision = record["precision25"] * 100
        axes[2].scatter(recall, precision, color=color, s=32, zorder=3)
        source_data.append(dict(mode=mode, size_bin="all", gt=audit["targets"], hit=record["hits25"],
                                recall_percent=recall, precision_percent=precision, fp=record["fp25"]))
    axes[2].set_xlim(76, 93)
    axes[2].set_ylim(63, 91)
    axes[2].set_xlabel("Mask recall (%)")
    axes[2].set_ylabel("Precision (%)")
    axes[2].set_title("Recall has an FP cost", fontsize=8, pad=14)
    axes[2].grid(alpha=0.17)
    fig.suptitle("Frozen I07 · old-split exploration · score ≥0.25 · mask IoU ≥0.50", fontsize=9, y=0.975)
    fig.text(0.985, 0.022, "193 images · 1,049 GT · one weight · common640, not native EV6 AP", ha="right", fontsize=6)
    fig.canvas.draw()
    require_matplotlib_panel_alignment(fig, json_out=str(args.output / "diagnosis.alignment.json"),
                                       overlay_svg=str(args.output / "diagnosis.alignment.svg"),
                                       tolerance_pt=1.5, gutter_tolerance_pt=1.5,
                                       require_panel_labels=True, strict=True)
    fig.savefig(args.output / "diagnosis.svg")
    fig.savefig(args.output / "diagnosis.pdf")
    fig.savefig(args.output / "diagnosis.png", dpi=600)
    fig.savefig(args.output / "diagnosis.tiff", dpi=600, pil_kwargs={"compression": "tiff_lzw"})
    plt.close(fig)
    with (args.output / "source_data.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        fields = sorted(set().union(*(row.keys() for row in source_data)))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(source_data)


if __name__ == "__main__":
    main()
