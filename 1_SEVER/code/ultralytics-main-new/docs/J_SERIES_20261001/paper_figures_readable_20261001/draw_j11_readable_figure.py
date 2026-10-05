"""Editable, plain-language J11 architecture diagrams in Chinese and English."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import yaml

ROOT = Path(__file__).resolve().parents[1]
TEXT = "#152b43"
BLUE = "#387cb6"
PURPLE = "#8056a4"
GRAY = "#637284"

LABELS = {
    "zh": {
        "title": "J11：细节保留与上下文辅助的柑橘幼果实例分割",
        "input": "果树 RGB 图像",
        "encoder": "① 提取图像线索",
        "preprocess": "颜色 + 明暗 + 局部对比",
        "shallow": "浅层：保留边缘与纹理",
        "deep": "深层：提取环境与语义",
        "fusion": "② 汇合多尺度信息\n让细节与语义互补",
        "identify": "③ 找到每个果实\n确定位置、判断是不是果实",
        "mask": "④ 描出可见轮廓\n用保留的细节画准边界",
        "output": "⑤ 输出独立实例\n每个果实一张掩膜",
        "scene": "浅层前景提示：只辅助分类",
        "bypass": "保留的浅层细节",
        "footer": "实线：主要信息流    蓝色虚线：细节直达轮廓解码    紫色虚线：辅助分类，不硬删候选",
    },
    "en": {
        "title": "J11: Detail-preserving citrus instance segmentation",
        "input": "RGB orchard\nimage",
        "encoder": "1  Encode image cues",
        "preprocess": "RGB + luminance\n+ local contrast",
        "shallow": "Shallow features\nEdges and texture",
        "deep": "Deeper features\nContext and semantics",
        "fusion": "2  One-pass fusion\nDetail + semantics\nacross scales",
        "identify": "3  Locate instances\nBoxes, scores and\ninstance coefficients",
        "mask": "4  Decode visible masks\nDetail + semantics\n+ instance coefficients",
        "output": "5  Instance masks\nSeparate masks\nfor each fruit",
        "scene": "Shallow scene cue to classification only",
        "bypass": "Retained early detail",
        "footer": "Solid: main flow    Blue dashed: detail bypass    Purple dashed: soft class guidance; no hard filtering",
    },
}


def box(ax, xy, size, text, face="#e9f3fb", edge=BLUE, font=8.2, weight="normal", zorder=2):
    x, y = xy
    width, height = size
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.12,rounding_size=0.65",
            facecolor=face,
            edgecolor=edge,
            linewidth=0.85,
            zorder=zorder,
        )
    )
    if text:
        ax.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=font,
            color=TEXT,
            weight=weight,
            linespacing=1.5,
            zorder=3,
        )


def arrow(ax, points, color=GRAY, dashed=False):
    for index, (start, end) in enumerate(zip(points, points[1:])):
        ax.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>" if index == len(points) - 2 else "-",
                mutation_scale=7,
                linewidth=1.05,
                color=color,
                linestyle=(0, (3, 2)) if dashed else "solid",
                shrinkA=0,
                shrinkB=0,
                zorder=1,
            )
        )


def draw(output, language, audit_scripts):
    from audit_panel_alignment import require_matplotlib_panel_alignment

    labels = LABELS[language]
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Microsoft YaHei", "Arial", "DejaVu Sans"],
            "font.size": 8,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    # 180 x 115 mm; literal inches also allow static size preflight to inspect this export.
    fig, ax = plt.subplots(figsize=(7.086614173228347, 4.52755905511811))
    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.98)
    ax.set(xlim=(0, 100), ylim=(0, 66))
    ax.axis("off")
    ax.text(50, 63, labels["title"], fontsize=10, weight="bold", color=TEXT, ha="center", va="center")
    box(ax, (2, 30), (13, 13), labels["input"], face="#f2f5f8", edge=GRAY, font=8.2)

    # One sequential encoder, not a persistent parallel high-resolution backbone.
    box(ax, (21, 19), (26, 38), "", face="#f7fbfe", edge="#b4ccdf", zorder=0)
    ax.text(34, 54.1, labels["encoder"], fontsize=9.2, weight="bold", ha="center", va="center", color=TEXT)
    box(ax, (24, 43), (20, 8), labels["preprocess"], font=8 if language == "zh" else 7.8)
    box(ax, (24, 33), (20, 8), labels["shallow"], font=8 if language == "zh" else 7.8)
    box(ax, (24, 22), (20, 8), labels["deep"], face="#eeebf7", edge=PURPLE, font=8 if language == "zh" else 7.8)
    arrow(ax, [(15.2, 36.5), (18, 36.5), (18, 47), (23.8, 47)])
    arrow(ax, [(34, 42.8), (34, 41.2)])
    arrow(ax, [(34, 32.8), (34, 30.2)])

    box(
        ax,
        (54, 30),
        (16, 13),
        labels["fusion"],
        face="#fff4e5",
        edge="#b18442",
        font=8.1 if language == "zh" else 7.7,
        weight="bold",
    )
    arrow(ax, [(44.2, 37), (53.8, 37)])
    arrow(ax, [(44.2, 26), (50, 26), (50, 33), (53.8, 33)])
    box(
        ax,
        (77, 30),
        (21, 13),
        labels["identify"],
        face="#eeebf7",
        edge=PURPLE,
        font=8.1 if language == "zh" else 7.8,
        weight="bold",
    )
    arrow(ax, [(70.2, 36.5), (76.8, 36.5)])

    # Only the class-tower input receives the learned shallow foreground prior.
    arrow(ax, [(44.2, 39.7), (49, 39.7), (49, 58.5), (87.5, 58.5), (87.5, 43.2)], PURPLE, True)
    ax.text(70.5, 60.1, labels["scene"], fontsize=7.5, color=PURPLE, ha="center", va="center")

    box(
        ax,
        (54, 7),
        (23, 13),
        labels["mask"],
        face="#e9f3fb",
        edge=BLUE,
        font=8.1 if language == "zh" else 7.8,
        weight="bold",
    )
    box(
        ax,
        (83, 7),
        (15, 13),
        labels["output"],
        face="#f0f5f9",
        edge=GRAY,
        font=8.1 if language == "zh" else 7.6,
        weight="bold",
    )
    arrow(ax, [(62, 29.8), (62, 20.2)])
    arrow(ax, [(92, 29.8), (92, 24.5), (73, 24.5), (73, 20.2)])
    arrow(ax, [(77.2, 13.5), (82.8, 13.5)])
    arrow(ax, [(23.8, 37), (19, 37), (19, 12), (53.8, 12)], BLUE, True)
    ax.text(35, 14.1, labels["bypass"], fontsize=8, color=BLUE, ha="center", va="center")
    ax.text(50, 2.5, labels["footer"], fontsize=7.2, color=GRAY, ha="center", va="center")

    stem = output / ("J11_通俗版架构图" if language == "zh" else "J11_architecture_paper_EN")
    require_matplotlib_panel_alignment(
        fig,
        json_out=str(stem) + ".alignment.json",
        strict=True,
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
    )
    fig.savefig(str(stem) + ".svg")
    fig.savefig(str(stem) + ".pdf")
    fig.savefig(str(stem) + ".png", dpi=600)
    plt.close(fig)
    return stem


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--audit-scripts", required=True, type=Path)
    args = parser.parse_args()
    sys.path.insert(0, str(args.audit_scripts))
    config_path = ROOT / "0_orange_yaml/J_series/J11_synthesis.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["head"][0][0] == [4, 2, 7]
    assert config["head"][1][0] == [7, 4, 10]
    assert config["head"][2][0] == [10, 7]
    assert config["head"][3][0] == [11, 12, 13, 2, 0]
    assert config["head"][3][3][3:7] == [2, True, 0, 0]
    if args.output.exists():
        raise FileExistsError("Use a new output folder, preserving existing figures")
    args.output.mkdir(parents=True)
    font_path = Path("C:/Windows/Fonts/msyh.ttc")
    if font_path.exists():
        font_manager.fontManager.addfont(str(font_path))
    stems = [draw(args.output, language, args.audit_scripts) for language in ("zh", "en")]
    (args.output / "source_identity.json").write_text(
        json.dumps(
            {
                "model": "J11_synthesis",
                "yaml_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
                "module_sha256": hashlib.sha256((ROOT / "ultralytics/nn/modules/citrus_j.py").read_bytes()).hexdigest(),
                "figure_role": "architecture only; no claim of measured optimum or performance gains",
                "dimensions_mm": [180, 115],
                "png_dpi": 600,
                "backend": "matplotlib; editable SVG/PDF text; simplified semantic-level diagram",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Exported: " + ", ".join(str(stem) for stem in stems))


if __name__ == "__main__":
    main()
