# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""F 系列 yaml 生成器：柑橘远距离小目标改进实验矩阵（F01-F38，共 37 个）.

针对痛点：远处柑橘幼果极小（<32px/<16px）、模糊、发黑、估计标注。
运行：python _generate_far_yamls.py   （在本目录下生成/覆盖所有 F*.yaml）

拓扑家族与索引说明见各构造函数 docstring；leave-one-out 消融（F32-F37）与 F31 完全同拓扑
（用 nn.Identity / 原始模块占位），保证逐层索引对齐、预训练迁移一致，是干净消融的关键。
"""

from __future__ import annotations

from pathlib import Path

OUT_DIR = Path(__file__).parent

SCALES = """
# Parameters
nc: 80 # number of classes (overridden by the dataset YAML, e.g. nc=1 for citrus)
scales: # model compound scaling constants
  # [depth, width, max_channels]
  n: [0.50, 0.25, 1024]
  s: [0.50, 0.50, 1024]
  m: [0.50, 1.00, 512]
  l: [1.00, 1.00, 512]
  x: [1.00, 1.50, 512]
"""

UP_STOCK = ("nn.Upsample", '[None, 2, "nearest"]')
UP_DY = ("DySample", "[]")
UP_CARAFE = ("CARAFE", "[]")
CAT_STOCK = ("Concat", "[1]")
CAT_BIFPN = ("BiFPNConcat", "[]")
SPPF_STOCK = ("SPPF", "[1024, 5]")
SPPF_LSKA = ("SPPF_LSKA", "[1024, 5]")
SPPF_RFB = ("RFB", "[1024]")


def down_args(mod: str, c: int) -> str:
    """Args token for a downsampling module at output channels c."""
    return {"SPDConv": f"[{c}, 3]", "HWDown": f"[{c}]"}.get(mod, f"[{c}, 3, 2]")


def stock_backbone(down="Conv", bb="C3k2", sppf=SPPF_STOCK):
    """标准 yolo11 骨干（索引 0-10）：P2=2, P3=4, P4=6, P5=8, C2PSA=10."""
    return [
        "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
        "- [-1, 2, C3k2, [256, False, 0.25]] # 2-P2/4",
        f"- [-1, 1, {down}, {down_args(down, 256)}] # 3-P3/8",
        f"- [-1, 2, {bb}, [512, False, 0.25]] # 4-P3/8",
        f"- [-1, 1, {down}, {down_args(down, 512)}] # 5-P4/16",
        f"- [-1, 2, {bb}, [512, True]] # 6-P4/16",
        f"- [-1, 1, {down}, {down_args(down, 1024)}] # 7-P5/32",
        f"- [-1, 2, {bb}, [1024, True]] # 8-P5/32",
        f"- [-1, 1, {sppf[0]}, {sppf[1]}] # 9",
        "- [-1, 2, C2PSA, [1024]] # 10",
    ]


def stock_head(up=UP_STOCK, cat=CAT_STOCK, neck="C3k2"):
    """标准 23 层头（11-23），Segment 输入 [16, 19, 22]."""
    return [
        f"- [-1, 1, {up[0]}, {up[1]}] # 11",
        f"- [[-1, 6], 1, {cat[0]}, {cat[1]}] # 12: cat backbone P4",
        f"- [-1, 2, {neck}, [512, False]] # 13-P4/16",
        f"- [-1, 1, {up[0]}, {up[1]}] # 14",
        f"- [[-1, 4], 1, {cat[0]}, {cat[1]}] # 15: cat backbone P3",
        f"- [-1, 2, {neck}, [256, False]] # 16-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 17",
        f"- [[-1, 13], 1, {cat[0]}, {cat[1]}] # 18: cat head P4",
        f"- [-1, 2, {neck}, [512, False]] # 19-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 20",
        f"- [[-1, 10], 1, {cat[0]}, {cat[1]}] # 21: cat head P5",
        f"- [-1, 2, {neck}, [1024, True]] # 22-P5/32",
        "- [[16, 19, 22], 1, Segment, [nc, 32, 256]] # 23-Segment(P3, P4, P5)",
    ]


def attn_head(attn: str, attn_args: str = "[]", up=UP_STOCK, cat=CAT_STOCK):
    """注意力头（11-24）：在颈部 P3 输出后插入注意力（P3 同时是 mask 原型来源），Segment 输入 [17, 20, 23]."""
    return [
        f"- [-1, 1, {up[0]}, {up[1]}] # 11",
        f"- [[-1, 6], 1, {cat[0]}, {cat[1]}] # 12: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 13-P4/16",
        f"- [-1, 1, {up[0]}, {up[1]}] # 14",
        f"- [[-1, 4], 1, {cat[0]}, {cat[1]}] # 15: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 16-P3/8",
        f"- [-1, 1, {attn}, {attn_args}] # 17-P3 attention (小目标路径 + mask 原型来源)",
        "- [-1, 1, Conv, [256, 3, 2]] # 18",
        f"- [[-1, 13], 1, {cat[0]}, {cat[1]}] # 19: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 20-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 21",
        f"- [[-1, 10], 1, {cat[0]}, {cat[1]}] # 22: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 23-P5/32",
        "- [[17, 20, 23], 1, Segment, [nc, 32, 256]] # 24-Segment(P3, P4, P5)",
    ]


def dfem_backbone(down="Conv", sppf=SPPF_STOCK, dfem="DFEM", bb="C3k2"):
    """增强骨干（0-11）：增强模块插在骨干 P3 之后。P2=2, P3(增强后)=5, P4=7, C2PSA=11."""
    return [
        "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
        f"- [-1, 2, {bb}, [256, False, 0.25]] # 2-P2/4",
        f"- [-1, 1, {down}, {down_args(down, 256)}] # 3-P3/8",
        f"- [-1, 2, {bb}, [512, False, 0.25]] # 4-P3/8",
        f"- [-1, 1, {dfem}, []] # 5-P3 增强模块（原创）",
        f"- [-1, 1, {down}, {down_args(down, 512)}] # 6-P4/16",
        f"- [-1, 2, {bb}, [512, True]] # 7-P4/16",
        f"- [-1, 1, {down}, {down_args(down, 1024)}] # 8-P5/32",
        f"- [-1, 2, {bb}, [1024, True]] # 9-P5/32",
        f"- [-1, 1, {sppf[0]}, {sppf[1]}] # 10",
        "- [-1, 2, C2PSA, [1024]] # 11",
    ]


def dfem_head(up=UP_STOCK, cat=CAT_STOCK, neckattn=None):
    """DFEM 拓扑的头（12-24/25）。neckattn 非空时在颈部 P3 后插入（如 LIAM / nn.Identity 占位）."""
    lines = [
        f"- [-1, 1, {up[0]}, {up[1]}] # 12",
        f"- [[-1, 7], 1, {cat[0]}, {cat[1]}] # 13: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 14-P4/16",
        f"- [-1, 1, {up[0]}, {up[1]}] # 15",
        f"- [[-1, 5], 1, {cat[0]}, {cat[1]}] # 16: cat backbone P3 (DFEM 增强后)",
        "- [-1, 2, C3k2, [256, False]] # 17-P3/8",
    ]
    if neckattn is None:
        lines += [
            "- [-1, 1, Conv, [256, 3, 2]] # 18",
            f"- [[-1, 14], 1, {cat[0]}, {cat[1]}] # 19: cat head P4",
            "- [-1, 2, C3k2, [512, False]] # 20-P4/16",
            "- [-1, 1, Conv, [512, 3, 2]] # 21",
            f"- [[-1, 11], 1, {cat[0]}, {cat[1]}] # 22: cat head P5",
            "- [-1, 2, C3k2, [1024, True]] # 23-P5/32",
            "- [[17, 20, 23], 1, Segment, [nc, 32, 256]] # 24-Segment(P3, P4, P5)",
        ]
    else:
        lines += [
            f"- [-1, 1, {neckattn}, []] # 18-P3 颈部注意力槽位",
            "- [-1, 1, Conv, [256, 3, 2]] # 19",
            f"- [[-1, 14], 1, {cat[0]}, {cat[1]}] # 20: cat head P4",
            "- [-1, 2, C3k2, [512, False]] # 21-P4/16",
            "- [-1, 1, Conv, [512, 3, 2]] # 22",
            f"- [[-1, 11], 1, {cat[0]}, {cat[1]}] # 23: cat head P5",
            "- [-1, 2, C3k2, [1024, True]] # 24-P5/32",
            "- [[18, 21, 24], 1, Segment, [nc, 32, 256]] # 25-Segment(P3, P4, P5)",
        ]
    return lines


def p2_head(up=UP_STOCK, cat=CAT_STOCK):
    """P2 检测头（骨干为标准 0-10）：11-29，Segment 输入 [19, 22, 25, 28]，mask 原型来自 P2/4."""
    return [
        f"- [-1, 1, {up[0]}, {up[1]}] # 11",
        f"- [[-1, 6], 1, {cat[0]}, {cat[1]}] # 12: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 13-P4/16",
        f"- [-1, 1, {up[0]}, {up[1]}] # 14",
        f"- [[-1, 4], 1, {cat[0]}, {cat[1]}] # 15: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 16-P3/8",
        f"- [-1, 1, {up[0]}, {up[1]}] # 17",
        f"- [[-1, 2], 1, {cat[0]}, {cat[1]}] # 18: cat backbone P2",
        "- [-1, 2, C3k2, [128, False]] # 19-P2/4 高分辨率小目标层",
        "- [-1, 1, Conv, [128, 3, 2]] # 20",
        f"- [[-1, 16], 1, {cat[0]}, {cat[1]}] # 21: cat head P3",
        "- [-1, 2, C3k2, [256, False]] # 22-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 23",
        f"- [[-1, 13], 1, {cat[0]}, {cat[1]}] # 24: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 25-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 26",
        f"- [[-1, 10], 1, {cat[0]}, {cat[1]}] # 27: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 28-P5/32",
        "- [[19, 22, 25, 28], 1, Segment, [nc, 32, 256]] # 29-Segment(P2, P3, P4, P5)",
    ]


def p2_full_head(up=UP_DY, cat=CAT_BIFPN):
    """F38 专用：DFEM 骨干（0-11）+ P2 头（12-30），Segment 输入 [20, 23, 26, 29]."""
    return [
        f"- [-1, 1, {up[0]}, {up[1]}] # 12",
        f"- [[-1, 7], 1, {cat[0]}, {cat[1]}] # 13: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 14-P4/16",
        f"- [-1, 1, {up[0]}, {up[1]}] # 15",
        f"- [[-1, 5], 1, {cat[0]}, {cat[1]}] # 16: cat backbone P3 (DFEM 增强后)",
        "- [-1, 2, C3k2, [256, False]] # 17-P3/8",
        f"- [-1, 1, {up[0]}, {up[1]}] # 18",
        f"- [[-1, 2], 1, {cat[0]}, {cat[1]}] # 19: cat backbone P2",
        "- [-1, 2, C3k2, [128, False]] # 20-P2/4 高分辨率小目标层",
        "- [-1, 1, Conv, [128, 3, 2]] # 21",
        f"- [[-1, 17], 1, {cat[0]}, {cat[1]}] # 22: cat head P3",
        "- [-1, 2, C3k2, [256, False]] # 23-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 24",
        f"- [[-1, 14], 1, {cat[0]}, {cat[1]}] # 25: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 26-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 27",
        f"- [[-1, 11], 1, {cat[0]}, {cat[1]}] # 28: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 29-P5/32",
        "- [[20, 23, 26, 29], 1, Segment, [nc, 32, 256]] # 30-Segment(P2, P3, P4, P5)",
    ]


def csfg_head():
    """CSFG 头（11-24）：P2(层2) 细节经 CSFG 注入颈部 P3（层16），Segment 输入 [17, 20, 23]."""
    return [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11',
        "- [[-1, 6], 1, Concat, [1]] # 12: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 13-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
        "- [[-1, 4], 1, Concat, [1]] # 15: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 16-P3/8",
        "- [[2, 16], 1, CSFG, []] # 17: P2 细节门控注入 P3（原创，P2 头的轻量替代）",
        "- [-1, 1, Conv, [256, 3, 2]] # 18",
        "- [[-1, 13], 1, Concat, [1]] # 19: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 20-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 21",
        "- [[-1, 10], 1, Concat, [1]] # 22: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 23-P5/32",
        "- [[17, 20, 23], 1, Segment, [nc, 32, 256]] # 24-Segment(P3, P4, P5)",
    ]


def hvi_dfem_layers():
    """F23 专用：HVIEnhance 前端（层0）+ 全体索引 +1 + DFEM 在骨干 P3 后（层6）."""
    backbone = [
        "- [-1, 1, HVIEnhance, [16, 2]] # 0: HVI 低照度增强前端 (3->3)",
        "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
        "- [-1, 2, C3k2, [256, False, 0.25]] # 3-P2/4",
        "- [-1, 1, Conv, [256, 3, 2]] # 4-P3/8",
        "- [-1, 2, C3k2, [512, False, 0.25]] # 5-P3/8",
        "- [-1, 1, DFEM, []] # 6-P3 DFEM 双域频率增强（原创）",
        "- [-1, 1, Conv, [512, 3, 2]] # 7-P4/16",
        "- [-1, 2, C3k2, [512, True]] # 8-P4/16",
        "- [-1, 1, Conv, [1024, 3, 2]] # 9-P5/32",
        "- [-1, 2, C3k2, [1024, True]] # 10-P5/32",
        "- [-1, 1, SPPF, [1024, 5]] # 11",
        "- [-1, 2, C2PSA, [1024]] # 12",
    ]
    head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13',
        "- [[-1, 8], 1, Concat, [1]] # 14: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 15-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
        "- [[-1, 6], 1, Concat, [1]] # 17: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 18-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 19",
        "- [[-1, 15], 1, Concat, [1]] # 20: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 21-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 22",
        "- [[-1, 12], 1, Concat, [1]] # 23: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 24-P5/32",
        "- [[18, 21, 24], 1, Segment, [nc, 32, 256]] # 25-Segment(P3, P4, P5)",
    ]
    return backbone, head


def dfem_p2p3_layers(mod: str = "DFEM"):
    """F20/F49 共用：指定模块同时插在骨干 P2（层3）与 P3（层6）之后."""
    backbone = [
        "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
        "- [-1, 2, C3k2, [256, False, 0.25]] # 2-P2/4",
        f"- [-1, 1, {mod}, []] # 3-P2 {mod}（原创）",
        "- [-1, 1, Conv, [256, 3, 2]] # 4-P3/8",
        "- [-1, 2, C3k2, [512, False, 0.25]] # 5-P3/8",
        f"- [-1, 1, {mod}, []] # 6-P3 {mod}（原创）",
        "- [-1, 1, Conv, [512, 3, 2]] # 7-P4/16",
        "- [-1, 2, C3k2, [512, True]] # 8-P4/16",
        "- [-1, 1, Conv, [1024, 3, 2]] # 9-P5/32",
        "- [-1, 2, C3k2, [1024, True]] # 10-P5/32",
        "- [-1, 1, SPPF, [1024, 5]] # 11",
        "- [-1, 2, C2PSA, [1024]] # 12",
    ]
    head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13',
        "- [[-1, 8], 1, Concat, [1]] # 14: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 15-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
        "- [[-1, 6], 1, Concat, [1]] # 17: cat backbone P3 (DFEM 增强后)",
        "- [-1, 2, C3k2, [256, False]] # 18-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 19",
        "- [[-1, 15], 1, Concat, [1]] # 20: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 21-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 22",
        "- [[-1, 12], 1, Concat, [1]] # 23: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 24-P5/32",
        "- [[18, 21, 24], 1, Segment, [nc, 32, 256]] # 25-Segment(P3, P4, P5)",
    ]
    return backbone, head


def emit(fname: str, comment_lines: list[str], backbone: list[str], head: list[str]) -> None:
    """Write one model yaml with header comments."""
    lines = ["# Ultralytics \U0001F680 AGPL-3.0 License - https://ultralytics.com/license", "#"]
    lines += [f"# {c}" if c else "#" for c in comment_lines]
    lines.append(SCALES)
    lines.append("backbone:")
    lines.append("  # [from, repeats, module, args]")
    lines += [f"  {b}" for b in backbone]
    lines.append("")
    lines.append("head:")
    lines += [f"  {h}" for h in head]
    lines.append("")
    (OUT_DIR / fname).write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {fname}")


def main() -> None:  # noqa: PLR0915
    train_cmd = "python train_citrus_seg.py --model 0_orange_yaml/1_far_small/{f} --pretrained yolo11n-seg.pt --name {n}"

    # ---------------- A 组：P2 高分辨率检测层 ----------------
    emit(
        "F01_yolo11-seg-p2.yaml",
        [
            "F01: +P2/4 高分辨率检测层（Segment 增至 4 层，mask 原型改由 P2 生成）。",
            "动机：640 输入下 34.9-40.5% 实例至少一边 <32px，P3/8 上 <16px 目标不足 2 个特征格。",
            "文献: QueryDet (Yang et al. CVPR 2022, arXiv:2103.09136); MAE-YOLOv8 绿果小目标用 p2 分支",
            "      (Liu et al. 2024, doi:10.1016/j.compag.2024.109458)。",
            "消融角色: A 组单模块 | " + train_cmd.format(f="F01_yolo11-seg-p2.yaml", n="F01_p2"),
        ],
        stock_backbone(),
        p2_head(),
    )
    emit(
        "F02_yolo11-seg-p2-spd.yaml",
        [
            "F02: P2 检测层 + SPD-Conv 无损下采样（骨干 3 处步长卷积替换）。",
            "动机: P2 层保住分辨率后，下采样仍丢 3/4 像素；SPD 重排到通道维实现信息无损。",
            "文献: SPD-Conv (Sunkara & Luo, ECML-PKDD 2022, arXiv:2208.03641)。",
            "消融角色: A 组组合 | " + train_cmd.format(f="F02_yolo11-seg-p2-spd.yaml", n="F02_p2_spd"),
        ],
        stock_backbone(down="SPDConv"),
        p2_head(),
    )

    # ---------------- B 组：下采样改进 ----------------
    emit(
        "F03_yolo11-seg-spdconv.yaml",
        [
            "F03: SPD-Conv 替换骨干全部步长下采样（P3/P4/P5 三处）。",
            "动机: 远处 <16px 柑橘经步长卷积两次下采样后特征几乎消失；SPD 无损重排保信息。",
            "文献: Sunkara & Luo, ECML-PKDD 2022, arXiv:2208.03641。",
            "消融角色: B 组单模块 | " + train_cmd.format(f="F03_yolo11-seg-spdconv.yaml", n="F03_spd"),
        ],
        stock_backbone(down="SPDConv"),
        stock_head(),
    )
    emit(
        "F04_yolo11-seg-hwd.yaml",
        [
            "F04: Haar 小波下采样 HWDown 替换骨干全部步长下采样。",
            "动机: 与 SPD 同为无损下采样，但按频带分解——高频子带显式保留模糊小果的微弱边缘。",
            "文献: Xu et al., Pattern Recognition 2023, doi:10.1016/j.patcog.2023.109819。",
            "消融角色: B 组单模块（与 F03 互为对照）| " + train_cmd.format(f="F04_yolo11-seg-hwd.yaml", n="F04_hwd"),
        ],
        stock_backbone(down="HWDown"),
        stock_head(),
    )

    # ---------------- C 组：C3k2 块变体 ----------------
    emit(
        "F05_yolo11-seg-c3k2faster.yaml",
        [
            "F05: 骨干 C3k2 → C3k2_Faster（FasterNet PConv 部分卷积）。",
            "动机: 轻量化对照——为后续加 P2/注意力腾出参数预算，验证骨干冗余度。",
            "文献: Chen et al., CVPR 2023, arXiv:2303.03667。",
            "消融角色: C 组单模块（轻量化）| " + train_cmd.format(f="F05_yolo11-seg-c3k2faster.yaml", n="F05_faster"),
        ],
        stock_backbone(bb="C3k2_Faster"),
        stock_head(),
    )
    emit(
        "F06_yolo11-seg-c3k2wt.yaml",
        [
            "F06: 骨干 P3-P5 C3k2 → C3k2_WT（小波卷积 bottleneck，频域大感受野）。",
            "动机: 模糊 = 高频衰减；WTConv 在小波域分频带卷积，可学习放大残存高频、低频补结构。",
            "文献: Finder et al., ECCV 2024, arXiv:2407.05848。",
            "消融角色: C 组单模块（抗模糊）| " + train_cmd.format(f="F06_yolo11-seg-c3k2wt.yaml", n="F06_wt"),
        ],
        stock_backbone(bb="C3k2_WT"),
        stock_head(),
    )
    emit(
        "F07_yolo11-seg-c3k2dwr.yaml",
        [
            "F07: 颈部 C3k2 → C3k2_DWR（多膨胀率残差，扩大颈部语义感受野）。",
            "动机: 远处小果需要大范围枝叶上下文佐证；DWR 以并联膨胀卷积零成本扩感受野。",
            "文献: Wei et al., DWRSeg 2022, arXiv:2212.01173。",
            "消融角色: C 组单模块（上下文）| " + train_cmd.format(f="F07_yolo11-seg-c3k2dwr.yaml", n="F07_dwr"),
        ],
        stock_backbone(),
        stock_head(neck="C3k2_DWR"),
    )

    # ---------------- D 组：注意力（统一插在颈部 P3 输出，mask 原型来源处）----------------
    attn_specs = [
        ("F08", "ema", "EMA", "[]", "EMA 高效多尺度注意力", "Ouyang et al., ICASSP 2023, arXiv:2305.13563"),
        ("F09", "simam", "SimAM", "[]", "SimAM 无参数能量注意力（0 参数增量）", "Yang et al., ICML 2021 (PMLR v139)"),
        ("F10", "cbam", "CBAM", "[]", "CBAM 通道+空间串联注意力（经典基线）", "Woo et al., ECCV 2018, arXiv:1807.06521"),
        ("F11", "coordatt", "CoordAtt", "[]", "Coordinate Attention 坐标注意力", "Hou et al., CVPR 2021, arXiv:2103.02907"),
        ("F12", "ela", "ELA", "[]", "ELA 高效局部注意力（条带池化+分组1D卷积）", "Xu & Wan 2024, arXiv:2403.01123"),
        ("F13", "caa", "CAA", "[]", "CAA 上下文锚点注意力（大条带核远程上下文）", "Cai et al., CVPR 2024, arXiv:2403.06258"),
    ]
    for num, tag, mod, margs, zh, ref in attn_specs:
        emit(
            f"{num}_yolo11-seg-{tag}.yaml",
            [
                f"{num}: 颈部 P3 输出后插入 {zh}。",
                "动机: P3 是小目标检测与 mask 原型的共同来源；注意力在此放大与叶片同色的弱小果信号。",
                f"文献: {ref}。",
                f"消融角色: D 组注意力横评 | " + train_cmd.format(f=f"{num}_yolo11-seg-{tag}.yaml", n=f"{num}_{tag}"),
            ],
            stock_backbone(),
            attn_head(mod, margs),
        )

    # ---------------- E 组：SPPF / 全局上下文 ----------------
    emit(
        "F14_yolo11-seg-sppf-lska.yaml",
        [
            "F14: SPPF → SPPF_LSKA（多尺度池化聚合特征上加大核分离注意力）。",
            "动机: 判断远处暗点是否为柑橘依赖全局上下文；LSKA 让 SPPF 聚合具备空间选择性。",
            "文献: Lau et al., Expert Syst. Appl. 2024, doi:10.1016/j.eswa.2023.121352。",
            "消融角色: E 组单模块 | " + train_cmd.format(f="F14_yolo11-seg-sppf-lska.yaml", n="F14_lska"),
        ],
        stock_backbone(sppf=SPPF_LSKA),
        stock_head(),
    )
    emit(
        "F15_yolo11-seg-rfb.yaml",
        [
            "F15: SPPF → RFB（多膨胀率感受野块）。",
            "动机: 模拟人类视觉离心感受野，'小感受野看果、大感受野看枝叶'并联进行。",
            "文献: Liu et al., ECCV 2018, arXiv:1711.07767。",
            "消融角色: E 组单模块（与 F14 互为对照）| " + train_cmd.format(f="F15_yolo11-seg-rfb.yaml", n="F15_rfb"),
        ],
        stock_backbone(sppf=SPPF_RFB),
        stock_head(),
    )

    # ---------------- F 组：颈部融合与上采样 ----------------
    emit(
        "F16_yolo11-seg-bifpn.yaml",
        [
            "F16: 颈部 4 处 Concat → BiFPNConcat（可学习加权融合）。",
            "动机: 深层语义在 Concat 中淹没浅层细节；可学习权重让网络自动上调小目标浅层贡献。",
            "文献: Tan et al., EfficientDet CVPR 2020, arXiv:1911.09070。",
            "消融角色: F 组单模块 | " + train_cmd.format(f="F16_yolo11-seg-bifpn.yaml", n="F16_bifpn"),
        ],
        stock_backbone(),
        stock_head(cat=CAT_BIFPN),
    )
    emit(
        "F17_yolo11-seg-carafe.yaml",
        [
            "F17: 颈部 2 处最近邻上采样 → CARAFE 内容感知重组上采样。",
            "动机: 最近邻把小果 1 个特征像素复制 4 份（无新信息）；CARAFE 按内容聚合邻域重建细节。",
            "文献: Wang et al., ICCV 2019, arXiv:1905.02188。",
            "消融角色: F 组单模块 | " + train_cmd.format(f="F17_yolo11-seg-carafe.yaml", n="F17_carafe"),
        ],
        stock_backbone(),
        stock_head(up=UP_CARAFE),
    )
    emit(
        "F18_yolo11-seg-dysample.yaml",
        [
            "F18: 颈部 2 处最近邻上采样 → DySample 动态点采样上采样（超轻量）。",
            "动机: 与 CARAFE 同目的但代价更低（nano 模型友好），小目标边界恢复更好。",
            "文献: Liu et al., ICCV 2023, arXiv:2308.15085。",
            "消融角色: F 组单模块（与 F17 互为对照）| " + train_cmd.format(f="F18_yolo11-seg-dysample.yaml", n="F18_dysample"),
        ],
        stock_backbone(),
        stock_head(up=UP_DY),
    )

    # ---------------- G 组：原创模块 ----------------
    emit(
        "F19_yolo11-seg-dfem.yaml",
        [
            "F19: 骨干 P3 后插入 DFEM 双域频率增强模块（原创）。",
            "动机: 直接针对'远处柑橘模糊(高频衰减)+发黑(弱响应)'——频带可学习增益 + 暗区响应补偿。",
            "文献基础: FreqFusion (TPAMI 2024, arXiv:2408.12879) + PE-YOLO (ICANN 2023, arXiv:2307.10953)，组合原创。",
            "消融角色: G 组原创单模块 | " + train_cmd.format(f="F19_yolo11-seg-dfem.yaml", n="F19_dfem"),
        ],
        dfem_backbone(),
        dfem_head(),
    )
    emit(
        "F20_yolo11-seg-dfem-p2p3.yaml",
        [
            "F20: DFEM 同时插在骨干 P2 与 P3 之后（双位置增强）。",
            "动机: P2 特征含最多小目标细节，先于 P3 增强可能更早止损；验证 DFEM 位置敏感性。",
            "消融角色: G 组位置消融（vs F19）| " + train_cmd.format(f="F20_yolo11-seg-dfem-p2p3.yaml", n="F20_dfem_p2p3"),
        ],
        *dfem_p2p3_layers(),
    )
    emit(
        "F21_yolo11-seg-liam.yaml",
        [
            "F21: 颈部 P3 输出后插入 LIAM 亮度不变注意力模块（原创）。",
            "动机: 近亮果/远暗果亮度分布差异大，IN 对齐亮度统计量 + 无参能量注意力突出弱信号。",
            "文献基础: IBN-Net (ECCV 2018, arXiv:1807.09441) + SimAM (ICML 2021)，门控级联原创。",
            "消融角色: G 组原创单模块（与 D 组注意力同位置可横评）| " + train_cmd.format(f="F21_yolo11-seg-liam.yaml", n="F21_liam"),
        ],
        stock_backbone(),
        attn_head("LIAM", "[]"),
    )
    emit(
        "F22_yolo11-seg-csfg.yaml",
        [
            "F22: CSFG 跨级小目标特征引导（原创）——P2 细节经 SPD 无损对齐+内容门控注入颈部 P3。",
            "动机: P2 检测头涨点但代价大；CSFG 以 <0.1 GFLOPs 代价把 P2 细节送进 P3，轻量替代。",
            "文献基础: Gold-YOLO GD (NeurIPS 2023, arXiv:2309.11331) + ASF-YOLO (doi:10.1016/j.imavis.2024.104957)，组合原创。",
            "消融角色: G 组原创单模块（vs F01 P2 头的性价比对照）| " + train_cmd.format(f="F22_yolo11-seg-csfg.yaml", n="F22_csfg"),
        ],
        stock_backbone(),
        csfg_head(),
    )
    emit(
        "F23_yolo11-seg-hvi-dfem.yaml",
        [
            "F23: HVIEnhance 低照度增强前端（图像域）+ DFEM（特征域）双重增强。",
            "动机: 发黑柑橘先在图像域做 HVI 色彩空间增强，再在特征域做频率/暗区补偿，互补验证。",
            "文献: HVI/CIDNet (Yan et al. 2024, arXiv:2402.05809)；DFEM 见 F19。",
            "消融角色: G 组组合（vs F19 与已有 010_hvi）| " + train_cmd.format(f="F23_yolo11-seg-hvi-dfem.yaml", n="F23_hvi_dfem"),
        ],
        *hvi_dfem_layers(),
    )

    # ---------------- H 组：两两组合（交互效应）----------------
    emit(
        "F24_yolo11-seg-spd-dysample.yaml",
        [
            "F24: SPD 无损下采样 + DySample 动态上采样（信息保持型采样闭环）。",
            "消融角色: H 组交互（F03 x F18）| " + train_cmd.format(f="F24_yolo11-seg-spd-dysample.yaml", n="F24_spd_dy"),
        ],
        stock_backbone(down="SPDConv"),
        stock_head(up=UP_DY),
    )
    emit(
        "F25_yolo11-seg-spd-ema.yaml",
        [
            "F25: SPD 无损下采样 + 颈部 P3 EMA 注意力。",
            "消融角色: H 组交互（F03 x F08）| " + train_cmd.format(f="F25_yolo11-seg-spd-ema.yaml", n="F25_spd_ema"),
        ],
        stock_backbone(down="SPDConv"),
        attn_head("EMA", "[]"),
    )
    emit(
        "F26_yolo11-seg-dfem-liam.yaml",
        [
            "F26: DFEM（骨干 P3 增强）+ LIAM（颈部 P3 注意力）——两个原创模块的协同。",
            "消融角色: H 组交互（F19 x F21）| " + train_cmd.format(f="F26_yolo11-seg-dfem-liam.yaml", n="F26_dfem_liam"),
        ],
        dfem_backbone(),
        dfem_head(neckattn="LIAM"),
    )
    emit(
        "F27_yolo11-seg-bifpn-dysample.yaml",
        [
            "F27: BiFPN 加权融合 + DySample 动态上采样（颈部整体升级）。",
            "消融角色: H 组交互（F16 x F18）| " + train_cmd.format(f="F27_yolo11-seg-bifpn-dysample.yaml", n="F27_bifpn_dy"),
        ],
        stock_backbone(),
        stock_head(up=UP_DY, cat=CAT_BIFPN),
    )
    emit(
        "F28_yolo11-seg-dfem-spd.yaml",
        [
            "F28: DFEM 频域增强 + SPD 无损下采样（先增强、再无损传递）。",
            "消融角色: H 组交互（F19 x F03）| " + train_cmd.format(f="F28_yolo11-seg-dfem-spd.yaml", n="F28_dfem_spd"),
        ],
        dfem_backbone(down="SPDConv"),
        dfem_head(),
    )

    # ---------------- I 组：最终组合 CitrusFar-Seg 及 leave-one-out ----------------
    emit(
        "F30_yolo11-seg-ours-lite.yaml",
        [
            "F30: CitrusFar-Seg-Lite = SPD 下采样 + DySample 上采样 + 颈部 P3 EMA。",
            "定位: 参数增量最小的三件套，若 F31 过重可作为论文主推轻量版。",
            "消融角色: I 组组合 | " + train_cmd.format(f="F30_yolo11-seg-ours-lite.yaml", n="F30_ours_lite"),
        ],
        stock_backbone(down="SPDConv"),
        attn_head("EMA", "[]", up=UP_DY),
    )
    full_note = "F31 拓扑（DFEM@5 + SPD 下采样 + SPPF_LSKA + BiFPN + DySample + LIAM@18），"
    emit(
        "F31_yolo11-seg-ours-full.yaml",
        [
            "F31: CitrusFar-Seg-Full（本课题主推组合，6 组件）：",
            "  DFEM（原创，频域+暗区增强）+ SPD-Conv（无损下采样）+ SPPF_LSKA（全局上下文）",
            "  + BiFPNConcat（加权融合）+ DySample（动态上采样）+ LIAM（原创，亮度不变注意力）。",
            "训练时配合 --iou-type NWDWise（原创损失）与可选 --slide 构成完整方法。",
            "消融角色: I 组主模型；F32-F37 为其 leave-one-out | "
            + train_cmd.format(f="F31_yolo11-seg-ours-full.yaml", n="F31_ours_full"),
        ],
        dfem_backbone(down="SPDConv", sppf=SPPF_LSKA),
        dfem_head(up=UP_DY, cat=CAT_BIFPN, neckattn="LIAM"),
    )
    loo = [
        ("F32", "no-dfem", "去 DFEM（层5 → nn.Identity）", dict(dfem="nn.Identity", down="SPDConv", sppf=SPPF_LSKA),
         dict(up=UP_DY, cat=CAT_BIFPN, neckattn="LIAM")),
        ("F33", "no-spd", "去 SPD（下采样还原步长 Conv）", dict(down="Conv", sppf=SPPF_LSKA),
         dict(up=UP_DY, cat=CAT_BIFPN, neckattn="LIAM")),
        ("F34", "no-lska", "去 LSKA（还原 SPPF）", dict(down="SPDConv", sppf=SPPF_STOCK),
         dict(up=UP_DY, cat=CAT_BIFPN, neckattn="LIAM")),
        ("F35", "no-bifpn", "去 BiFPN（还原 Concat）", dict(down="SPDConv", sppf=SPPF_LSKA),
         dict(up=UP_DY, cat=CAT_STOCK, neckattn="LIAM")),
        ("F36", "no-dysample", "去 DySample（还原最近邻上采样）", dict(down="SPDConv", sppf=SPPF_LSKA),
         dict(up=UP_STOCK, cat=CAT_BIFPN, neckattn="LIAM")),
        ("F37", "no-liam", "去 LIAM（层18 → nn.Identity）", dict(down="SPDConv", sppf=SPPF_LSKA),
         dict(up=UP_DY, cat=CAT_BIFPN, neckattn="nn.Identity")),
    ]
    for num, tag, zh, bkw, hkw in loo:
        emit(
            f"{num}_yolo11-seg-ours-{tag}.yaml",
            [
                f"{num}: {full_note}{zh}。",
                "与 F31 逐层索引完全对齐（占位模块），预训练迁移与层深一致——干净 leave-one-out。",
                f"消融角色: I 组 leave-one-out | " + train_cmd.format(f=f"{num}_yolo11-seg-ours-{tag}.yaml", n=f"{num}_{tag.replace('-', '_')}"),
            ],
            dfem_backbone(**bkw),
            dfem_head(**hkw),
        )
    emit(
        "F38_yolo11-seg-ours-full-p2.yaml",
        [
            "F38: CitrusFar-Seg-Full-P2 = F31 全部组件 + P2/4 高分辨率检测层（性能上限探索版）。",
            "mask 原型来自 P2/4（160x160），远处小果掩码质量最优；代价为最大的显存与延迟。",
            "消融角色: I 组上限版（vs F31 权衡）| " + train_cmd.format(f="F38_yolo11-seg-ours-full-p2.yaml", n="F38_ours_p2"),
        ],
        dfem_backbone(down="SPDConv", sppf=SPPF_LSKA),
        p2_full_head(up=UP_DY, cat=CAT_BIFPN),
    )

    # ---------------- J 组：架构级原创（大改网络拓扑，论文核心创新候选）----------------
    emit(
        "F40_yolo11-seg-hrstream.yaml",
        [
            "F40: 双流高分辨率辅助流（架构级原创）——P2 细节流与主干并行，三路 BiFPN 融合进 P3。",
            "  主干不变；新增 aux 流: backbone P2 → C3k2(P2 分辨率精炼) → SPD 无损对齐 P3；",
            "  颈部 P3 融合点改为三输入 BiFPNConcat(语义上采样 + backbone P3 + aux 细节)。",
            "动机: 远处小果的判别信息大量停留在 P2 分辨率，但 nano 主干在 P2 只有 1 个 C3k2；",
            "  辅助流以极小代价补足高分辨率处理深度（HRNet 保持高分辨率表征思想的轻量化实现）。",
            "文献基础: HRNet (Wang et al., TPAMI 2020, arXiv:1908.07919) + Gold-YOLO 汇聚分发",
            "  (NeurIPS 2023, arXiv:2309.11331)；双流拓扑组合为本课题原创。",
            "消融角色: J 组架构创新 | " + train_cmd.format(f="F40_yolo11-seg-hrstream.yaml", n="F40_hrstream"),
        ],
        stock_backbone(),
        [
            "- [2, 2, C3k2, [128, False]] # 11: HR 辅助流——P2 分辨率细节精炼（从骨干层2引出）",
            "- [-1, 1, SPDConv, [256, 3]] # 12: 辅助流无损下采样到 P3 尺度",
            '- [10, 1, nn.Upsample, [None, 2, "nearest"]] # 13: 语义流自顶向下（从 C2PSA 层10）',
            "- [[-1, 6], 1, Concat, [1]] # 14: cat backbone P4",
            "- [-1, 2, C3k2, [512, False]] # 15-P4/16",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
            "- [[-1, 4, 12], 1, BiFPNConcat, []] # 17: 三路融合——语义 + backbone P3 + HR 细节流",
            "- [-1, 2, C3k2, [256, False]] # 18-P3/8",
            "- [-1, 1, Conv, [256, 3, 2]] # 19",
            "- [[-1, 15], 1, Concat, [1]] # 20: cat head P4",
            "- [-1, 2, C3k2, [512, False]] # 21-P4/16",
            "- [-1, 1, Conv, [512, 3, 2]] # 22",
            "- [[-1, 10], 1, Concat, [1]] # 23: cat head P5",
            "- [-1, 2, C3k2, [1024, True]] # 24-P5/32",
            "- [[18, 21, 24], 1, Segment, [nc, 32, 256]] # 25-Segment(P3, P4, P5)",
        ],
    )
    freq_detail_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11: 语义路径自顶向下',
        "- [[-1, 6], 1, Concat, [1]] # 12: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 13-P4/16 语义",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
        "- [[-1, 4], 1, Concat, [1]] # 15: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 16-P3/8 语义",
        "- [2, 1, DFEM, []] # 17: 细节路径——骨干 P2 双域频率增强（原创模块）",
        "- [-1, 1, SPDConv, [256, 3]] # 18: 细节无损对齐 P3 尺度",
        "- [[16, 18], 1, BiFPNConcat, []] # 19: 细节-语义加权融合",
        "- [-1, 2, C3k2_WT, [256, False]] # 20-P3/8 输出（小波卷积融合块，抗模糊）",
        "- [-1, 1, Conv, [256, 3, 2]] # 21",
        "- [[-1, 13], 1, BiFPNConcat, []] # 22: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 23-P4/16 输出",
        "- [-1, 1, Conv, [512, 3, 2]] # 24",
        "- [[-1, 10], 1, BiFPNConcat, []] # 25: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 26-P5/32 输出",
        "- [[20, 23, 26], 1, Segment, [nc, 32, 256]] # 27-Segment(P3, P4, P5)",
    ]
    emit(
        "F41_yolo11-seg-freqdetail-pan.yaml",
        [
            "F41: FreqDetail-PAN 细节-语义双路颈部（架构级原创，整个 neck 重设计）：",
            "  语义路径 = 标准自顶向下；细节路径 = P2 → DFEM 频域增强 → SPD 无损对齐；",
            "  两路在 P3 用 BiFPN 加权融合，再过 C3k2_WT 小波融合块；自底向上全部换 BiFPN。",
            "动机: PAN 颈部的 P3 输出被深层语义主导，模糊/发黑小果的高频细节在融合前就已丢失；",
            "  本设计让细节以独立通路直达融合点，且频域增强发生在最高分辨率处。",
            "文献基础: DFEM 见 F19；WTConv (ECCV 2024, arXiv:2407.05848)；加权融合 (EfficientDet)。",
            "消融角色: J 组架构创新（vs F19/F22 单点版）| " + train_cmd.format(f="F41_yolo11-seg-freqdetail-pan.yaml", n="F41_freqdetail"),
        ],
        stock_backbone(),
        freq_detail_head,
    )
    sh_backbone = [
        "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
        "- [-1, 4, C3k2, [256, False, 0.25]] # 2-P2/4（加深: 2→4 repeats）",
        "- [-1, 1, Conv, [256, 3, 2]] # 3-P3/8",
        "- [-1, 4, C3k2, [512, False, 0.25]] # 4-P3/8（加深: 2→4 repeats）",
        "- [-1, 1, Conv, [512, 3, 2]] # 5-P4/16",
        "- [-1, 2, C3k2, [512, True]] # 6-P4/16",
        "- [-1, 1, Conv, [768, 3, 2]] # 7-P5/32（削窄: 1024→768）",
        "- [-1, 2, C3k2, [768, True]] # 8-P5/32（削窄）",
        "- [-1, 1, SPPF, [768, 5]] # 9",
        "- [-1, 2, C2PSA, [768]] # 10",
    ]
    sh_head_tail = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11',
        "- [[-1, 6], 1, Concat, [1]] # 12: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 13-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
        "- [[-1, 4], 1, Concat, [1]] # 15: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 16-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 17",
        "- [[-1, 13], 1, Concat, [1]] # 18: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 19-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 20",
        "- [[-1, 10], 1, Concat, [1]] # 21: cat head P5",
        "- [-1, 2, C3k2, [768, True]] # 22-P5/32（随骨干削窄）",
        "- [[16, 19, 22], 1, Segment, [nc, 32, 256]] # 23-Segment(P3, P4, P5)",
    ]
    emit(
        "F42_yolo11-seg-shallowheavy.yaml",
        [
            "F42: Shallow-Heavy 骨干（架构级原创）——计算重分配：P2/P3 加深一倍，P5 削窄 25%。",
            "动机: 34.9-40.5% 实例 <32px，其判别特征只存在于 P2/P3；stock 骨干却把最多参数",
            "  放在 P5/32（远处小果在此已 <1 格）。把算力搬到小目标所在的分辨率，参数量近似持平。",
            "文献基础: QueryDet 高分辨率查询 (CVPR 2022, arXiv:2103.09136)；深度/宽度再分配思想",
            "  (EfficientNet, ICML 2019, arXiv:1905.11946)。重分配方案为本课题原创。",
            "消融角色: J 组架构创新 | " + train_cmd.format(f="F42_yolo11-seg-shallowheavy.yaml", n="F42_shallowheavy"),
        ],
        sh_backbone,
        sh_head_tail,
    )
    f43_head = [ln for ln in freq_detail_head]
    f43_head[15] = "- [-1, 2, C3k2, [768, True]] # 26-P5/32 输出（随骨干削窄）"
    emit(
        "F43_yolo11-seg-citrusfar-v2.yaml",
        [
            "F43: CitrusFar-Seg-V2（最大架构创新组合）= Shallow-Heavy 骨干 + FreqDetail-PAN 颈部。",
            "  训练配合 --iou-type NWDWise（原创损失）即构成完整的论文候选方法：",
            "  浅层重分配（治小）+ 频域细节双路颈（治模糊/发黑）+ 尺度自适应损失（治估计标注）。",
            "消融角色: J 组终极组合（vs F31 模块堆叠版做架构对照）| "
            + train_cmd.format(f="F43_yolo11-seg-citrusfar-v2.yaml", n="F43_citrusfar_v2"),
        ],
        sh_backbone,
        f43_head,
    )

    # ---------------- K 组：部署导向轻量化（单片机/嵌入式，量化与端侧算子友好）----------------
    # 设计纪律：只用 conv / pool / concat / slice 级算子——不用 FFT(DFEM)、grid_sample(DySample)、
    # unfold(CARAFE)、InstanceNorm(LIAM)，保证 ONNX -> NCNN / RKNN / TFLite-int8 一键可转。
    edge_backbone = [
        "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
        "- [-1, 3, C3k2_Faster, [256, False, 0.25]] # 2-P2/4（PConv 加深不加价）",
        "- [-1, 1, HWDown, [256]] # 3-P3/8（Haar 小波无损下采样，纯 slice+1x1 端侧友好）",
        "- [-1, 3, C3k2_Faster, [512, False, 0.25]] # 4-P3/8（PConv 加深）",
        "- [-1, 1, HWDown, [512]] # 5-P4/16",
        "- [-1, 2, C3k2_Faster, [512, True]] # 6-P4/16",
        "- [-1, 1, HWDown, [768]] # 7-P5/32（削窄 1024→768）",
        "- [-1, 2, C3k2_Faster, [768, True]] # 8-P5/32",
        "- [-1, 1, SPPF_LSKA, [768, 5]] # 9（LSKA=纯 1D 深度卷积，端侧友好的大感受野）",
        "- [-1, 2, C2PSA, [768]] # 10",
    ]
    edge_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11（最近邻上采样，端侧零成本）',
        "- [[-1, 6], 1, BiFPNConcat, []] # 12: cat backbone P4（加权融合仅 2 个标量参数）",
        "- [-1, 2, C3k2_Faster, [512, False]] # 13-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
        "- [[-1, 4], 1, BiFPNConcat, []] # 15: cat backbone P3",
        "- [-1, 2, C3k2_Faster, [256, False]] # 16-P3/8",
        "- [[2, 16], 1, CSFG, []] # 17: P2 细节门控注入（原创，avgpool+conv+sigmoid 全端侧友好）",
        "- [-1, 1, Conv, [256, 3, 2]] # 18",
        "- [[-1, 13], 1, BiFPNConcat, []] # 19: cat head P4",
        "- [-1, 2, C3k2_Faster, [512, False]] # 20-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 21",
        "- [[-1, 10], 1, BiFPNConcat, []] # 22: cat head P5",
        "- [-1, 2, C3k2_Faster, [768, True]] # 23-P5/32",
        "- [[17, 20, 23], 1, Segment, [nc, 32, 256]] # 24-Segment(P3, P4, P5)",
    ]
    emit(
        "F44_yolo11-seg-citrusfar-edge.yaml",
        [
            "F44: CitrusFar-Edge（部署导向轻量化创新）——面向单片机/嵌入式的小目标分割网络。",
            "  = Shallow-Heavy 重分配（P2/P3 加深、P5 削窄）+ C3k2_Faster 全网 PConv 化",
            "  + SPD 无损下采样 + BiFPN 加权融合 + SPPF_LSKA 大感受野 + CSFG 小目标引导。",
            "  全网仅 conv/pool/concat/slice 算子：无 FFT、无 grid_sample、无 unfold、无 IN，",
            "  ONNX→NCNN/RKNN/TFLite-INT8 直转；PConv 对 MCU 内存带宽尤其友好（3/4 通道直通）。",
            "文献: FasterNet (CVPR 2023, arXiv:2303.03667); SPD-Conv (arXiv:2208.03641);",
            "  LSKA (doi:10.1016/j.eswa.2023.121352); 重分配与 CSFG 为本课题原创。",
            "消融角色: K 组部署主推 | " + train_cmd.format(f="F44_yolo11-seg-citrusfar-edge.yaml", n="F44_edge"),
        ],
        edge_backbone,
        edge_head,
    )
    nano_backbone = [
        "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
        "- [-1, 2, C3k2_Faster, [256, False, 0.25]] # 2-P2/4",
        "- [-1, 1, HWDown, [256]] # 3-P3/8（Haar 小波下采样）",
        "- [-1, 2, C3k2_Faster, [384, False, 0.25]] # 4-P3/8（384 削窄）",
        "- [-1, 1, HWDown, [384]] # 5-P4/16",
        "- [-1, 1, C3k2_Faster, [384, True]] # 6-P4/16（384 削窄）",
        "- [-1, 1, HWDown, [512]] # 7-P5/32（512 极限削窄）",
        "- [-1, 1, C3k2_Faster, [512, True]] # 8-P5/32",
        "- [-1, 1, SPPF_LSKA, [512, 5]] # 9",
        "- [-1, 1, C3k2_Faster, [512, True]] # 10（去 C2PSA——注意力 MatMul 不利于 INT8 量化）",
    ]
    nano_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11',
        "- [[-1, 6], 1, BiFPNConcat, []] # 12: cat backbone P4",
        "- [-1, 1, C3k2_Faster, [384, False]] # 13-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
        "- [[-1, 4], 1, BiFPNConcat, []] # 15: cat backbone P3",
        "- [-1, 1, C3k2_Faster, [256, False]] # 16-P3/8",
        "- [[2, 16], 1, CSFG, []] # 17: P2 细节门控注入（原创）",
        "- [-1, 1, Conv, [256, 3, 2]] # 18",
        "- [[-1, 13], 1, BiFPNConcat, []] # 19: cat head P4",
        "- [-1, 1, C3k2_Faster, [384, False]] # 20-P4/16",
        "- [-1, 1, Conv, [384, 3, 2]] # 21",
        "- [[-1, 10], 1, BiFPNConcat, []] # 22: cat head P5",
        "- [-1, 1, C3k2_Faster, [512, True]] # 23-P5/32",
        "- [[17, 20, 23], 1, Segment, [nc, 32, 256]] # 24-Segment(P3, P4, P5)",
    ]
    emit(
        "F45_yolo11-seg-citrusfar-edge-nano.yaml",
        [
            "F45: CitrusFar-Edge-Nano（极限压缩版）——P4/P5 通道 384/512、颈部单 repeat、",
            "  去 C2PSA（注意力 MatMul 对 INT8 量化不友好），其余同 F44。",
            "  目标 <1.5M 参数；配合 INT8 PTQ 后权重 <1.5MB，可进高端 MCU（如 RV1106/K230）。",
            "  精度换算力的兜底方案：若 F44 在端侧仍超预算则退到本配置 + CWD 蒸馏补精度。",
            "消融角色: K 组极限版（vs F44 权衡）| " + train_cmd.format(f="F45_yolo11-seg-citrusfar-edge-nano.yaml", n="F45_edge_nano"),
        ],
        nano_backbone,
        nano_head,
    )

    # ---------------- L 组：XX-Former 范式原创模块（MetaFormer 结构，论文主打创新）----------------
    former_backbone = stock_backbone()
    former_backbone[10] = "- [-1, 1, FarFormer, [2, 8]] # 10: FarFormer x2（替换 C2PSA，原创）"
    emit(
        "F46_yolo11-seg-farformer.yaml",
        [
            "F46: FarFormer 远场感知 Former（原创，MetaFormer 范式）替换 P5 端 C2PSA。",
            "  Token Mixer = LGFM：α·LRSA（低分辨率全局注意力，QKV 池化到 8x8，近线性代价全图上下文）",
            "                + (1-α)·HFBranch（Haar 高频子带细节），α 可学习通道门控；",
            "  FFN = MSDFFFN（5x5/7x7 通道拆分+洗牌的多尺度动态混合）。",
            "融合来源: LRFormer (TPAMI 2025, IEEE doc 11029508) + WTConv (ECCV 2024, arXiv:2407.05848)",
            "  + SRConvNet DML (IJCV 2025, doi:10.1007/s11263-024-02147-y)；组合与门控原创。",
            "消融角色: L 组 Former 单模块（vs C2PSA 基线 / vs F14）| "
            + train_cmd.format(f="F46_yolo11-seg-farformer.yaml", n="F46_farformer"),
        ],
        former_backbone,
        stock_head(),
    )
    emit(
        "F47_yolo11-seg-lumiformer.yaml",
        [
            "F47: LumiFormer 亮度感知 Former（原创，MetaFormer 范式）插在颈部 P3 输出。",
            "  Token Mixer = 频域通道注意力（rFFT 幅谱去直流→通道权重，选出'有结构'通道）",
            "                → 暗区空间调制（响应亮度图→暗区门控，放大发黑区域特征）串联；",
            "  FFN = EDFFN（末端可学习频带筛选，init 恒等）。",
            "融合来源: HS-FPN HFP (AAAI 2025, arXiv:2412.10116) + CIDNet/PE-YOLO 暗区增强",
            "  (CVPR 2025, doi:10.1109/CVPR52734.2025.00533; arXiv:2307.10953) + EVSSM EDFFN",
            "  (CVPR 2025, arXiv:2405.14343)；串联组合原创。",
            "消融角色: L 组 Former 单模块（与 D 组注意力同位置横评）| "
            + train_cmd.format(f="F47_yolo11-seg-lumiformer.yaml", n="F47_lumiformer"),
        ],
        stock_backbone(),
        attn_head("LumiFormer", "[1]"),
    )
    citrusformer_backbone = [ln for ln in sh_backbone]
    citrusformer_backbone[10] = "- [-1, 1, FarFormer, [2, 8]] # 10: FarFormer x2 @P5（替换 C2PSA，原创）"
    citrusformer_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11',
        "- [[-1, 6], 1, BiFPNConcat, []] # 12: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 13-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
        "- [[-1, 4], 1, BiFPNConcat, []] # 15: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 16-P3/8",
        "- [-1, 1, LumiFormer, [1]] # 17-P3 LumiFormer（原创，小目标+mask 原型路径）",
        "- [-1, 1, Conv, [256, 3, 2]] # 18",
        "- [[-1, 13], 1, BiFPNConcat, []] # 19: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 20-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 21",
        "- [[-1, 10], 1, BiFPNConcat, []] # 22: cat head P5",
        "- [-1, 2, C3k2, [768, True]] # 23-P5/32（随骨干削窄）",
        "- [[17, 20, 23], 1, Segment, [nc, 32, 256]] # 24-Segment(P3, P4, P5)",
    ]
    emit(
        "F48_yolo11-seg-citrusformer-net.yaml",
        [
            "F48: CitrusFormer-Net（论文主打候选架构）= Shallow-Heavy 骨干 + FarFormer@P5",
            "  + LumiFormer@颈部P3 + 全 BiFPN 加权融合；训练配 --iou-type NWDWise 构成完整方法。",
            "  三个原创组件各治一痛点：Shallow-Heavy 治'小'（算力搬到高分辨率）、",
            "  FarFormer 治'远+模糊'（全局上下文+高频细节）、LumiFormer 治'暗'（频域通道+暗区调制）。",
            "消融角色: L 组终极架构（vs F43 非 Former 版 / F46 F47 单点版做 LOO）| "
            + train_cmd.format(f="F48_yolo11-seg-citrusformer-net.yaml", n="F48_citrusformer"),
        ],
        citrusformer_backbone,
        citrusformer_head,
    )

    # ---------------- N 组：数据驱动第二轮原创（依据 _dataset_analysis.md 量化证据）----------------
    # 数据证据：47.9% 实例 <32px；小果 V=103 vs 大果 132（更暗）；模糊度差 20 倍；|Δa*| 2.2 vs 2.9（更伪装）
    emit(
        "F49_yolo11-seg-tdam.yaml",
        [
            "F49: TDAM 纹理差异放大模块（原创，COD 伪装目标机制迁移）插在骨干 P2+P3。",
            "数据依据: 小果 |Δa*|=2.2 几乎与叶片同色（伪装），但果面(光滑球面)与叶面(叶脉纹理)的",
            "  纹理统计不同；多尺度 center-surround 差分放大这种差异。纯 pool/conv 算子，端侧可转。",
            "文献: SINet (CVPR 2020) 感受野对比 + PFNet (CVPR 2021) distraction mining",
            "  + Zhai 2024 绿果=COD 立论 (doi:10.1016/j.compag.2024.109356)；组合原创。",
            "消融角色: N 组原创单模块（vs F19 DFEM 同位置对照）| "
            + train_cmd.format(f="F49_yolo11-seg-tdam.yaml", n="F49_tdam"),
        ],
        *dfem_p2p3_layers(mod="TDAM"),
    )
    lce_backbone = [
        "- [-1, 1, LCE, [4, 16]] # 0: LCE 暗区门控曲线增强前端 (3->3, 原创)",
        "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
        "- [-1, 2, C3k2, [256, False, 0.25]] # 3-P2/4",
        "- [-1, 1, Conv, [256, 3, 2]] # 4-P3/8",
        "- [-1, 2, C3k2, [512, False, 0.25]] # 5-P3/8",
        "- [-1, 1, Conv, [512, 3, 2]] # 6-P4/16",
        "- [-1, 2, C3k2, [512, True]] # 7-P4/16",
        "- [-1, 1, Conv, [1024, 3, 2]] # 8-P5/32",
        "- [-1, 2, C3k2, [1024, True]] # 9-P5/32",
        "- [-1, 1, SPPF, [1024, 5]] # 10",
        "- [-1, 2, C2PSA, [1024]] # 11",
    ]
    lce_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 12',
        "- [[-1, 7], 1, Concat, [1]] # 13: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 14-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 15',
        "- [[-1, 5], 1, Concat, [1]] # 16: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 17-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 18",
        "- [[-1, 14], 1, Concat, [1]] # 19: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 20-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 21",
        "- [[-1, 11], 1, Concat, [1]] # 22: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 23-P5/32",
        "- [[17, 20, 23], 1, Segment, [nc, 32, 256]] # 24-Segment(P3, P4, P5)",
    ]
    emit(
        "F50_yolo11-seg-lce.yaml",
        [
            "F50: LCE 暗区门控曲线增强前端（原创组合，图像域，端侧友好）。",
            "数据依据: 小果 V 中位数 103 vs 大果 132 且比背景更暗（-9~-12）——欠曝是小果专属退化；",
            "  LCE 用 Zero-DCE 曲线 LE(x)=x+A·x·(1-x) 迭代提亮，暗区门控保护近处亮果不过曝。",
            "  全 conv/mul 算子（比 HVIEnhance 更端侧友好），A init=0 恒等起步、预训练权重安全迁移。",
            "文献: Zero-DCE (CVPR 2020, doi:10.1109/CVPR42600.2020.00185) + PE-YOLO 暗区思想；门控组合原创。",
            "消融角色: N 组原创前端（vs 010_hvi 图像域对照）| "
            + train_cmd.format(f="F50_yolo11-seg-lce.yaml", n="F50_lce"),
        ],
        lce_backbone,
        lce_head,
    )
    f51_backbone = [
        "- [-1, 1, LCE, [4, 16]] # 0: LCE 暗区曲线增强（原创）",
        "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
        "- [-1, 2, C3k2, [256, False, 0.25]] # 3-P2/4",
        "- [-1, 1, TDAM, []] # 4-P2 TDAM 纹理放大（原创）",
        "- [-1, 1, Conv, [256, 3, 2]] # 5-P3/8",
        "- [-1, 2, C3k2, [512, False, 0.25]] # 6-P3/8",
        "- [-1, 1, TDAM, []] # 7-P3 TDAM（原创）",
        "- [-1, 1, Conv, [512, 3, 2]] # 8-P4/16",
        "- [-1, 2, C3k2, [512, True]] # 9-P4/16",
        "- [-1, 1, Conv, [1024, 3, 2]] # 10-P5/32",
        "- [-1, 2, C3k2, [1024, True]] # 11-P5/32",
        "- [-1, 1, SPPF, [1024, 5]] # 12",
        "- [-1, 2, C2PSA, [1024]] # 13",
    ]
    f51_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
        "- [[-1, 9], 1, Concat, [1]] # 15: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 16-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 17',
        "- [[-1, 7], 1, Concat, [1]] # 18: cat backbone P3 (TDAM 后)",
        "- [-1, 2, C3k2, [256, False]] # 19-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 20",
        "- [[-1, 16], 1, Concat, [1]] # 21: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 22-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 23",
        "- [[-1, 13], 1, Concat, [1]] # 24: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 25-P5/32",
        "- [[19, 22, 25], 1, Segment, [nc, 32, 256]] # 26-Segment(P3, P4, P5)",
    ]
    emit(
        "F51_yolo11-seg-lce-tdam.yaml",
        [
            "F51: LCE（治暗）+ TDAM@P2P3（治伪装）——数据体检两大退化的联合修复。",
            "消融角色: N 组组合（F49 x F50）| " + train_cmd.format(f="F51_yolo11-seg-lce-tdam.yaml", n="F51_lce_tdam"),
        ],
        f51_backbone,
        f51_head,
    )
    f52_backbone = [
        "- [-1, 1, LCE, [4, 16]] # 0: LCE 暗区曲线增强（端侧友好）",
        "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
        "- [-1, 3, C3k2_Faster, [256, False, 0.25]] # 3-P2/4（PConv 加深）",
        "- [-1, 1, TDAM, []] # 4-P2 TDAM 纹理放大（pool/conv 端侧友好）",
        "- [-1, 1, HWDown, [256]] # 5-P3/8（Haar 小波下采样）",
        "- [-1, 3, C3k2_Faster, [512, False, 0.25]] # 6-P3/8",
        "- [-1, 1, HWDown, [512]] # 7-P4/16",
        "- [-1, 2, C3k2_Faster, [512, True]] # 8-P4/16",
        "- [-1, 1, HWDown, [768]] # 9-P5/32（削窄）",
        "- [-1, 2, C3k2_Faster, [768, True]] # 10-P5/32",
        "- [-1, 1, SPPF_LSKA, [768, 5]] # 11",
        "- [-1, 2, C2PSA, [768]] # 12",
    ]
    f52_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13',
        "- [[-1, 8], 1, BiFPNConcat, []] # 14: cat backbone P4",
        "- [-1, 2, C3k2_Faster, [512, False]] # 15-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
        "- [[-1, 6], 1, BiFPNConcat, []] # 17: cat backbone P3",
        "- [-1, 2, C3k2_Faster, [256, False]] # 18-P3/8",
        "- [[4, 18], 1, CSFG, []] # 19: P2(TDAM 后) 细节门控注入（原创）",
        "- [-1, 1, Conv, [256, 3, 2]] # 20",
        "- [[-1, 15], 1, BiFPNConcat, []] # 21: cat head P4",
        "- [-1, 2, C3k2_Faster, [512, False]] # 22-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 23",
        "- [[-1, 12], 1, BiFPNConcat, []] # 24: cat head P5",
        "- [-1, 2, C3k2_Faster, [768, True]] # 25-P5/32",
        "- [[19, 22, 25], 1, Segment, [nc, 32, 256]] # 26-Segment(P3, P4, P5)",
    ]
    emit(
        "F52_yolo11-seg-citrusfar-edge-v2.yaml",
        [
            "F52: CitrusFar-Edge-V2（部署主推升级版）= F44 + LCE（治暗）+ TDAM（治伪装）。",
            "  全部组件仍为 conv/pool/concat/slice 算子——ONNX→NCNN/RKNN/TFLite-INT8 直转；",
            "  数据体检显示暗与伪装在小果上最严重，而小果正是端侧采摘最远视距的目标。",
            "消融角色: N 组部署组合（vs F44 的 LOO：去 LCE / 去 TDAM）| "
            + train_cmd.format(f="F52_yolo11-seg-citrusfar-edge-v2.yaml", n="F52_edge_v2"),
        ],
        f52_backbone,
        f52_head,
    )
    f53_backbone = [
        "- [-1, 1, LCE, [4, 16]] # 0: LCE 暗区曲线增强（原创）",
        "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
        "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
        "- [-1, 4, C3k2, [256, False, 0.25]] # 3-P2/4（Shallow-Heavy 加深）",
        "- [-1, 1, TDAM, []] # 4-P2 TDAM 纹理放大（原创）",
        "- [-1, 1, Conv, [256, 3, 2]] # 5-P3/8",
        "- [-1, 4, C3k2, [512, False, 0.25]] # 6-P3/8（加深）",
        "- [-1, 1, Conv, [512, 3, 2]] # 7-P4/16",
        "- [-1, 2, C3k2, [512, True]] # 8-P4/16",
        "- [-1, 1, Conv, [768, 3, 2]] # 9-P5/32（削窄）",
        "- [-1, 2, C3k2, [768, True]] # 10-P5/32",
        "- [-1, 1, SPPF, [768, 5]] # 11",
        "- [-1, 1, FarFormer, [2, 8]] # 12: FarFormer x2 @P5（替换 C2PSA，原创）",
    ]
    f53_head = [
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13',
        "- [[-1, 8], 1, BiFPNConcat, []] # 14: cat backbone P4",
        "- [-1, 2, C3k2, [512, False]] # 15-P4/16",
        '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
        "- [[-1, 6], 1, BiFPNConcat, []] # 17: cat backbone P3",
        "- [-1, 2, C3k2, [256, False]] # 18-P3/8",
        "- [-1, 1, LumiFormer, [1]] # 19-P3 LumiFormer（原创）",
        "- [-1, 1, Conv, [256, 3, 2]] # 20",
        "- [[-1, 15], 1, BiFPNConcat, []] # 21: cat head P4",
        "- [-1, 2, C3k2, [512, False]] # 22-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 23",
        "- [[-1, 12], 1, BiFPNConcat, []] # 24: cat head P5",
        "- [-1, 2, C3k2, [768, True]] # 25-P5/32",
        "- [[19, 22, 25], 1, Segment, [nc, 32, 256]] # 26-Segment(P3, P4, P5)",
    ]
    emit(
        "F53_yolo11-seg-citrusformer-plus.yaml",
        [
            "F53: CitrusFormer-Net-Plus（精度上限主打）= F48 + LCE（治暗）+ TDAM（治伪装）。",
            "  五个原创组件覆盖数据体检的全部退化：Shallow-Heavy(小) + FarFormer(远/糊)",
            "  + LumiFormer(暗-特征域) + LCE(暗-图像域) + TDAM(伪装)；训练配 --iou-type NWDWise",
            "  --tal-metric NWD --tal-min-pos 构成完整方法（GA-TAL 治 <16px 正样本饥饿）。",
            "消融角色: N 组终极（vs F48 增量 = LCE+TDAM 贡献）| "
            + train_cmd.format(f="F53_yolo11-seg-citrusformer-plus.yaml", n="F53_citrusformer_plus"),
        ],
        f53_backbone,
        f53_head,
    )
    fla_backbone = stock_backbone()
    fla_backbone[10] = '- [-1, 1, FarFormer, [2, 8, "fla"]] # 10: FarFormer x2 (FLA 线性注意力 mixer)'
    emit(
        "F54_yolo11-seg-farformer-fla.yaml",
        [
            "F54: FarFormer 的 FLA 线性注意力变体（token mixer 消融：LRSA vs 线性注意力）。",
            "依据: theme7 调研裁决——P5 仅 400 token，Mamba selective scan 无优势且部署困难；",
            "  MLLA (NeurIPS 2024, arXiv:2405.16605) 证明 Mamba≈门控线性注意力，故用",
            "  FLatten 式 focused 线性注意力 (ICCV 2023, arXiv:2308.00442) 做可导出替代。",
            "消融角色: L 组 mixer 消融（vs F46）| "
            + train_cmd.format(f="F54_yolo11-seg-farformer-fla.yaml", n="F54_farformer_fla"),
        ],
        fla_backbone,
        stock_head(),
    )

    # ---------------- O 组：频域专线原创 ----------------
    emit(
        "F55_yolo11-seg-mwca.yaml",
        [
            "F55: MWCA 多级小波跨频带注意力（原创频域模块）插在骨干 P3 之后。",
            "数据依据: 小果与大果 Laplacian 模糊度差 20 倍——判别信息随距离在频带间迁移；",
            "  MWCA 用 2 级 Haar 分解出 7 个子带 + 跨频带注意力自适应选频带 + 高频显著图门控低频。",
            "  全 slice/conv/linear 算子（无 FFT），端侧可转。",
            "文献: FEDER 频率分解辨伪装 (CVPR 2023) + WTConv (ECCV 2024) + HS-FPN (AAAI 2025)；组合原创。",
            "消融角色: O 组频域单模块（vs F19 DFEM(FFT 频带增益) / F06 C3k2_WT 同位置对照）| "
            + train_cmd.format(f="F55_yolo11-seg-mwca.yaml", n="F55_mwca"),
        ],
        dfem_backbone(dfem="MWCA"),
        dfem_head(),
    )
    emit(
        "F56_yolo11-seg-freqsuite.yaml",
        [
            "F56: CitrusFreq-Seg 频域主线组合 = HWDown 小波下采样 + C3k2_WT 小波骨干 + MWCA@P3。",
            "  训练配 --freq-loss 0.1（FFL 频域掩码对齐损失，ICCV 2021 arXiv:2012.12821 迁移）",
            "  构成'分解-卷积-注意力-监督'全频域链路——频域创新的完整故事线。",
            "消融角色: O 组频域组合（LOO: 去 MWCA=F04+F06 / 去 WT / 去 FFL 训练项）| "
            + train_cmd.format(f="F56_yolo11-seg-freqsuite.yaml", n="F56_freqsuite") + " --freq-loss 0.1",
        ],
        dfem_backbone(down="HWDown", dfem="MWCA", bb="C3k2_WT"),
        dfem_head(),
    )

    # ---------------- P 组：顶会新范式 P5-mixer 横评（与 C2PSA/F46/F54 同槽位）----------------
    hco_backbone = stock_backbone()
    hco_backbone[10] = "- [-1, 1, HCO, [2]] # 10: HCO 热传导算子 x2（替换 C2PSA，vHeat 范式）"
    emit(
        "F57_yolo11-seg-hco.yaml",
        [
            "F57: HCO 热传导算子（vHeat 物理范式）替换 P5 端 C2PSA。",
            "  新范式: 特征传播 = 热传导方程，频域指数核 exp(-‖ω‖²k)，k=可学习每通道'传播距离'。",
            "  O(N log N) 全局混合，比注意力便宜；k 可视化 = '每通道看多远'（论文可解释性卖点）。",
            "文献: vHeat (Wang et al., 2024, arXiv:2405.16555)；周期边界 FFT 近似实现为本课题适配。",
            "消融角色: P 组 P5-mixer 横评（C2PSA vs F46 FarFormer vs F54 FLA vs F57 HCO vs F58 超图）| "
            + train_cmd.format(f="F57_yolo11-seg-hco.yaml", n="F57_hco"),
        ],
        hco_backbone,
        stock_head(),
    )
    hyper_backbone = stock_backbone()
    hyper_backbone[10] = "- [-1, 1, HyperACE, [8]] # 10: 超图关联增强（替换 C2PSA，8 条自适应软超边）"
    emit(
        "F58_yolo11-seg-hyperace.yaml",
        [
            "F58: HyperACE-lite 超图关联增强（Hyper-YOLO/YOLOv13 范式）替换 P5 端 C2PSA。",
            "  新范式: 卷积/注意力只有成对关联；超图以 8 条自适应软超边做节点→超边→节点两跳传递，",
            "  建模'同一果串/同一枝条'的多对多高阶关联——单个远处暗果证据不足时由群体模式互相佐证",
            "  （数据依据: 47-58.5% 图像存在密集相邻实例）。O(E·N·C) 代价极小。",
            "文献: Hyper-YOLO (TPAMI 2025, arXiv:2408.04804); YOLOv13 (arXiv:2506.17733)；lite 软超边实现为本课题适配。",
            "消融角色: P 组 P5-mixer 横评 | " + train_cmd.format(f="F58_yolo11-seg-hyperace.yaml", n="F58_hyperace"),
        ],
        hyper_backbone,
        stock_head(),
    )

    emit(
        "F59_yolo11-seg-c3k2ls.yaml",
        [
            "F59: 骨干 C3k2 → C3k2_LS（LSNet 'See Large, Focus Small' 动态卷积 bottleneck）。",
            "  新算子: 大核感知上下文 + 小核动态聚合细节的仿生卷积——与'远处小果需上下文佐证'同构；",
            "  YOLOv10 团队出品、全标准算子构成、ONNX 可导出（theme10 裁决: backbone 侧最稳的新算子）。",
            "文献: LSNet (Wang et al., CVPR 2025, arXiv:2503.23135)；fork 内置官方结构 (lsnet.py)。",
            "消融角色: C 组追加（vs F05 PConv / F06 WTConv 同位置横评）| "
            + train_cmd.format(f="F59_yolo11-seg-c3k2ls.yaml", n="F59_c3k2ls"),
        ],
        stock_backbone(bb="C3k2_LS"),
        stock_head(),
    )

    # ---------------- Q 组：纹理先验主线（用户提出的"去颜色、纹理增强"思想的可行化）----------------
    tgp_backbone = [ln.replace("LCE, [4, 16]] # 0: LCE 暗区门控曲线增强前端 (3->3, 原创)",
                               "TGP, []] # 0: TGP 纹理先验前端——去颜色+多尺度LCN纹理图+可靠性门控 (3->3, ~20参数, 原创)")
                    for ln in lce_backbone]
    emit(
        "F60_yolo11-seg-tgp.yaml",
        [
            "F60: TGP 纹理先验前端（原创，用户思想的可行化）——去颜色 + 多尺度 LCN 纹理图 + 可靠性门控。",
            "设计逻辑: ①数据证据: |Δa*|≈2-3 颜色判别力低→判别力在纹理；②V=max(RGB) 去色相/饱和度，",
            "  LCN t=(V-μ)/(σ+ε) 再去亮度绝对值——只留纹理且光照不变（顺带治远处发黑）；",
            "  ③远处糊果纹理不可靠→局部 σ 即置信度，c=sigmoid(a·σ+b) 门控，糊果处自动回退 RGB 主流；",
            "  ④γ init=0 恒等起步。参数 ~20 个、FLOPs≈0——轻量化硬约束的极致案例。",
            "文献: LCN (Jarrett et al., ICCV 2009) + Zhai 2024 伪装立论；金字塔+门控组合原创。",
            "消融角色: Q 组单模块（vs F50 LCE / 010 HVI 前端三路横评）| "
            + train_cmd.format(f="F60_yolo11-seg-tgp.yaml", n="F60_tgp"),
        ],
        tgp_backbone,
        lce_head,
    )
    f61_backbone = [ln.replace("LCE, [4, 16]] # 0: LCE 暗区曲线增强（原创）",
                               "TGP, []] # 0: TGP 纹理先验前端（原创）") for ln in f51_backbone]
    emit(
        "F61_yolo11-seg-tgp-tdam.yaml",
        [
            "F61: 纹理主线组合 = TGP 前端（图像域纹理先验）+ TDAM@P2P3（特征域纹理放大）。",
            "  与频域主线 F56 对称成线：图像域先验 → 特征域放大 → (可选 --freq-loss 频域监督)，",
            "  构成'纹理感知'完整故事——直击 |Δa*|≈2.2 的绿绿伪装痛点。",
            "消融角色: Q 组组合（LOO: F60 只有前端 / F49 只有 TDAM）| "
            + train_cmd.format(f="F61_yolo11-seg-tgp-tdam.yaml", n="F61_tgp_tdam"),
        ],
        f61_backbone,
        f51_head,
    )

    # ---------------- R 组：颈部融合范式升级（漏检专项）----------------
    hsf_head = [
        "- [[6, 10], 1, HSF, []] # 11: P4 = 高层筛选融合(低层bb-P4, 高层C2PSA-P5)——替代上采样+Concat",
        "- [-1, 2, C3k2, [512, False]] # 12-P4/16（输入通道减半，参数下降）",
        "- [[4, 12], 1, HSF, []] # 13: P3 = 高层筛选融合(低层bb-P3, 高层neck-P4)",
        "- [-1, 2, C3k2, [256, False]] # 14-P3/8",
        "- [-1, 1, Conv, [256, 3, 2]] # 15",
        "- [[-1, 12], 1, Concat, [1]] # 16: cat head P4（自底向上保持标准）",
        "- [-1, 2, C3k2, [512, False]] # 17-P4/16",
        "- [-1, 1, Conv, [512, 3, 2]] # 18",
        "- [[-1, 10], 1, Concat, [1]] # 19: cat head P5",
        "- [-1, 2, C3k2, [1024, True]] # 20-P5/32",
        "- [[14, 17, 20], 1, Segment, [nc, 32, 256]] # 21-Segment(P3, P4, P5)",
    ]
    emit(
        "F62_yolo11-seg-hsf.yaml",
        [
            "F62: 颈部自顶向下融合 Concat → HSF 高层筛选融合（漏检专项 + 轻量化双赢）。",
            "漏检根因: 低层特征背景噪声大，Concat 拼接稀释远处小果微弱信号；HSF 用高层语义生成",
            "  通道筛选权重过滤低层，只保留语义相关响应再相加——且输出通道不翻倍，后续 C3k2 参数下降。",
            "文献: HS-FPN (Chen et al., Comput. Biol. Med. 2024, doi:10.1016/j.compbiomed.2024.107917，",
            "  微小目标专用轻量筛选式 FPN)。",
            "消融角色: R 组融合范式横评（Concat 基线 vs F16 BiFPN vs F62 HSF）| "
            + train_cmd.format(f="F62_yolo11-seg-hsf.yaml", n="F62_hsf"),
        ],
        stock_backbone(),
        hsf_head,
    )

    emit(
        "F63_yolo11-seg-c3k2sxq.yaml",
        [
            "F63: 骨干 C3k2 → C3k2_SXQ（自研三合一 bottleneck：部分卷积×大核DW×卷积门控）。",
            "  解决: 标准 Bottleneck 贵(全通道 3x3x2)/感受野小(3x3)/静态混合；SXQBottleneck 参数仅其 49%，",
            "  感受野 7x7，门控让'信细节还是信上下文'由内容决定（远处糊果区自动偏上下文支）。",
            "文献融合: FasterNet (CVPR 2023, arXiv:2303.03667) × ConvNeXt 大核 (CVPR 2022, arXiv:2201.03545)",
            "  × TransNeXt Convolutional GLU (CVPR 2024, arXiv:2311.17132)；三合一组合原创。",
            "消融角色: C 组四路块横评（F05 PConv / F06 WT / F59 LS / F63 SXQ）；SXQNet 家族颈部块的独立证据行 | "
            + train_cmd.format(f="F63_yolo11-seg-c3k2sxq.yaml", n="F63_c3k2sxq"),
        ],
        stock_backbone(bb="C3k2_SXQ"),
        stock_head(),
    )

    emit(
        "F64_yolo11-seg-c3k2moce.yaml",
        [
            "F64: 骨干 C3k2 → C3k2_MoCE（卷积专家混合——LLM MoE 思想的轻量 CNN 落地）。",
            "  4 个 5x5 DW 专家核 + GAP 软路由按内容组合（专家≈近亮/远暗/糊/伪装四种成像条件）；",
            "  CondConv 式核组合：FLOPs≈单个 DW 卷积；软路由保 ONNX 可导。router 可视化=专家分工。",
            "文献: MoE (arXiv:1701.06538) → CondConv (arXiv:1904.04971) → DynamicConv (arXiv:1912.03458)；",
            "  软路由可导出引 Soft MoE (arXiv:2308.00951)。先例边界: YOLO-Master (arXiv:2512.23273) 已做",
            "  MoE-in-YOLO，本行只可声称组合创新（nano分割+成像条件路由+全软路由DW专家），写作须做差异对比。",
            "消融角色: C 组块横评第五路 | " + train_cmd.format(f="F64_yolo11-seg-c3k2moce.yaml", n="F64_moce"),
        ],
        stock_backbone(bb="C3k2_MoCE"),
        stock_head(),
    )
    hr_backbone = stock_backbone()
    hr_backbone[10] = "- [-1, 1, HyperRes, [4]] # 10: 双流超连接残差堆叠 x4（替换 C2PSA；'换方向的残差'迁移）"
    emit(
        "F65_yolo11-seg-hyperres.yaml",
        [
            "F65: P5 端 C2PSA → HyperRes 双流超连接残差堆叠（LLM 界可学习多流残差思想迁移）。",
            "  两条残差流 + 每块 8 个可学习混合标量；init 精确等价标准残差链（单测验证），",
            "  训练中残差'方向/速率'可学习——Shallow-Heavy 加深浅层后的梯度路径增强件。",
            "文献: Hyper-Connections (ByteDance, ICLR 2025, arXiv:2409.19606) 2流 lite 版；谱系 mHC (DeepSeek,",
            "  arXiv:2512.24880) → Attention Residuals (Kimi Team, arXiv:2603.15031, 深度注意力残差=可升级方向)。",
            "消融角色: P 组 P5-mixer 第六路 | " + train_cmd.format(f="F65_yolo11-seg-hyperres.yaml", n="F65_hyperres"),
        ],
        hr_backbone,
        stock_head(),
    )

    # ================ SXQNet 家族 V2-V10（V1 = SXQNet-seg.yaml 手写旗舰）================
    # 家族设计哲学：V1 均衡；V2-V10 各把一条痛点主轴推到极致（场景轴家族，非单纯宽深缩放）。
    # 每版回答一个问题：果园部署时主导痛点是什么，就选哪版。全部 ≤ 基线参数量（V6/V10 P2 版 FLOPs 例外）。
    sxq = "python train_citrus_seg.py --model 0_orange_yaml/1_far_small/{f} --pretrained yolo11n-seg.pt --name {n}"
    emit(
        "SXQNet-V2-nano.yaml",
        [
            "SXQNet-V2 Nano（极限端侧轴）：回答'单片机放得下吗'。目标 <1.6M、INT8 <1.6MB。",
            "  = TGP+LCE 双前端（~0 代价）+ C3k2_Faster 全网 + HWDown + SPPF_LSKA + CSFG + BiFPN；",
            "  无 FFT/无超图 matmul/无 IN——纯 conv/pool/slice，NCNN/RKNN/TFLite-INT8 直转。",
            "消融: vs F45（无前端版）量化 TGP+LCE 增益 | " + sxq.format(f="SXQNet-V2-nano.yaml", n="SXQ_V2"),
        ],
        [
            "- [-1, 1, TGP, []] # 0: 纹理先验前端（原创, ~20参数）",
            "- [-1, 1, LCE, [4, 16]] # 1: 暗区曲线前端（原创）",
            "- [-1, 1, Conv, [64, 3, 2]] # 2-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 3-P2/4",
            "- [-1, 2, C3k2_Faster, [256, False, 0.25]] # 4-P2/4",
            "- [-1, 1, HWDown, [256]] # 5-P3/8",
            "- [-1, 2, C3k2_Faster, [384, False, 0.25]] # 6-P3/8（384 削窄）",
            "- [-1, 1, HWDown, [384]] # 7-P4/16",
            "- [-1, 1, C3k2_Faster, [384, True]] # 8-P4/16",
            "- [-1, 1, HWDown, [512]] # 9-P5/32（512 极限削窄）",
            "- [-1, 1, C3k2_Faster, [512, True]] # 10-P5/32",
            "- [-1, 1, SPPF_LSKA, [512, 5]] # 11",
            "- [-1, 1, C3k2_Faster, [512, True]] # 12（去 C2PSA 保 INT8 友好）",
        ],
        [
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13',
            "- [[-1, 8], 1, BiFPNConcat, []] # 14",
            "- [-1, 1, C3k2_Faster, [384, False]] # 15-P4",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
            "- [[-1, 6], 1, BiFPNConcat, []] # 17",
            "- [-1, 1, C3k2_Faster, [256, False]] # 18-P3",
            "- [[4, 18], 1, CSFG, []] # 19: P2 细节注入（原创）",
            "- [-1, 1, Conv, [256, 3, 2]] # 20",
            "- [[-1, 15], 1, BiFPNConcat, []] # 21",
            "- [-1, 1, C3k2_Faster, [384, False]] # 22-P4",
            "- [-1, 1, Conv, [384, 3, 2]] # 23",
            "- [[-1, 12], 1, BiFPNConcat, []] # 24",
            "- [-1, 1, C3k2_Faster, [512, True]] # 25-P5",
            "- [[19, 22, 25], 1, Segment, [nc, 32, 256]] # 26",
        ],
    )
    emit(
        "SXQNet-V3-freq.yaml",
        [
            "SXQNet-V3 Freq（频域轴）：回答'模糊主导时怎么办'（模糊度差 20 倍）。全频域链路：",
            "  HWDown(下采样) + C3k2_WT(块) + MWCA@P3(注意力) + PCFA@P4(部分通道频域,原创) + HCO@P5(热传导)；",
            "  训练配 --freq-loss 0.1 构成'分解-卷积-注意力-物理算子-监督'五层频域故事。",
            "消融: vs F56（无 PCFA/HCO 版）| " + sxq.format(f="SXQNet-V3-freq.yaml", n="SXQ_V3") + " --freq-loss 0.1",
        ],
        [
            "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
            "- [-1, 2, C3k2_WT, [256, False, 0.25]] # 2-P2/4",
            "- [-1, 1, HWDown, [256]] # 3-P3/8",
            "- [-1, 2, C3k2_WT, [512, False, 0.25]] # 4-P3/8",
            "- [-1, 1, MWCA, []] # 5-P3 跨频带注意力（原创）",
            "- [-1, 1, HWDown, [512]] # 6-P4/16",
            "- [-1, 2, C3k2_WT, [512, True]] # 7-P4/16",
            "- [-1, 1, PCFA, []] # 8-P4 部分通道频域注意力（原创，代价 1/4）",
            "- [-1, 1, HWDown, [768]] # 9-P5/32",
            "- [-1, 2, C3k2_WT, [768, True]] # 10-P5/32",
            "- [-1, 1, SPPF, [768, 5]] # 11",
            "- [-1, 1, HCO, [2]] # 12: 热传导算子 x2（vHeat 范式）",
        ],
        [
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13',
            "- [[-1, 8], 1, BiFPNConcat, []] # 14",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 15-P4",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
            "- [[-1, 5], 1, BiFPNConcat, []] # 17",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 18-P3",
            "- [-1, 1, Conv, [256, 3, 2]] # 19",
            "- [[-1, 15], 1, BiFPNConcat, []] # 20",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 21-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 22",
            "- [[-1, 12], 1, BiFPNConcat, []] # 23",
            "- [-1, 2, C3k2_SXQ, [768, True]] # 24-P5",
            "- [[18, 21, 24], 1, Segment, [nc, 32, 256]] # 25",
        ],
    )
    emit(
        "SXQNet-V4-former.yaml",
        [
            "SXQNet-V4 Former（全 Former 轴）：回答'MetaFormer 范式化到什么程度最划算'。",
            "  = Shallow-Heavy + FarFormer@P5 + LumiFormer@P4与P3 双位置（vs F48 仅 P3 一处）。",
            "消融: vs F48 增量 = P4 位 LumiFormer | " + sxq.format(f="SXQNet-V4-former.yaml", n="SXQ_V4"),
        ],
        [ln for ln in sh_backbone[:-1]] + ["- [-1, 1, FarFormer, [2, 8, \"lrsa\", True]] # 10: FarFormer x2 @P5（DyT 免归一化版，企业技巧融合）"],
        [
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 11',
            "- [[-1, 6], 1, BiFPNConcat, []] # 12",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 13-P4",
            "- [-1, 1, LumiFormer, [1, True]] # 14-P4 LumiFormer（DyT 版，双位置之一）",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 15',
            "- [[-1, 4], 1, BiFPNConcat, []] # 16",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 17-P3",
            "- [-1, 1, LumiFormer, [1, True]] # 18-P3 LumiFormer（DyT 版，原创）",
            "- [-1, 1, Conv, [256, 3, 2]] # 19",
            "- [[-1, 14], 1, BiFPNConcat, []] # 20",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 21-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 22",
            "- [[-1, 10], 1, BiFPNConcat, []] # 23",
            "- [-1, 2, C3k2_SXQ, [768, True]] # 24-P5",
            "- [[18, 21, 24], 1, Segment, [nc, 32, 256]] # 25",
        ],
    )
    emit(
        "SXQNet-V5-hyper.yaml",
        [
            "SXQNet-V5 Hyper（关系建模轴）：回答'密集粘连错检主导时怎么办'（47-58.5% 图密集相邻）。",
            "  = HyperACE 双层超图（P4+P5 群体关联）+ HSF 筛选融合 + CSFG——错检专攻版。",
            "消融: vs F58（单层超图）| " + sxq.format(f="SXQNet-V5-hyper.yaml", n="SXQ_V5"),
        ],
        [
            "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
            "- [-1, 2, C3k2_SXQ, [256, False, 0.25]] # 2-P2/4",
            "- [-1, 1, Conv, [256, 3, 2]] # 3-P3/8",
            "- [-1, 2, C3k2_SXQ, [512, False, 0.25]] # 4-P3/8",
            "- [-1, 1, Conv, [512, 3, 2]] # 5-P4/16",
            "- [-1, 2, C3k2_MoCE, [512, True]] # 6-P4/16（MoCE 成像条件专家，企业技巧融合）",
            "- [-1, 1, HyperACE, [8]] # 7-P4 超图关联（果串在 P4 尺度成簇）",
            "- [-1, 1, Conv, [1024, 3, 2]] # 8-P5/32",
            "- [-1, 2, C3k2_MoCE, [1024, True]] # 9-P5/32（MoCE）",
            "- [-1, 1, SPPF, [1024, 5]] # 10",
            "- [-1, 1, HyperACE, [8]] # 11-P5 超图关联（跨区域群体证据）",
        ],
        [
            "- [[7, 11], 1, HSF, []] # 12: P4 筛选融合",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 13-P4",
            "- [[4, 13], 1, HSF, []] # 14: P3 筛选融合",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 15-P3",
            "- [[2, 15], 1, CSFG, []] # 16: P2 细节注入（原创）",
            "- [-1, 1, Conv, [256, 3, 2]] # 17",
            "- [[-1, 13], 1, Concat, [1]] # 18",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 19-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 20",
            "- [[-1, 11], 1, Concat, [1]] # 21",
            "- [-1, 2, C3k2_SXQ, [1024, True]] # 22-P5",
            "- [[16, 19, 22], 1, Segment, [nc, 32, 256]] # 23",
        ],
    )
    emit(
        "SXQNet-V6-p2.yaml",
        [
            "SXQNet-V6 P2（小目标精度轴）：回答'不计 FLOPs 时小果精度能到多高'。",
            "  = TGP 前端 + Shallow-Heavy×HWDown 骨干 + P2 四层头（mask 原型来自 P2/4）+ 全 BiFPN。",
            "消融: vs V1（三层头）量化 P2 层增益；FLOPs 高是本轴的 trade-off，参数仍轻 | "
            + sxq.format(f="SXQNet-V6-p2.yaml", n="SXQ_V6"),
        ],
        [
            "- [-1, 1, TGP, []] # 0: 纹理先验前端（原创）",
            "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
            "- [-1, 4, C3k2_SXQ, [256, False, 0.25]] # 3-P2/4（加深）",
            "- [-1, 1, HWDown, [256]] # 4-P3/8",
            "- [-1, 4, C3k2_SXQ, [512, False, 0.25]] # 5-P3/8（加深）",
            "- [-1, 1, HWDown, [512]] # 6-P4/16",
            "- [-1, 2, C3k2_SXQ, [512, True]] # 7-P4/16",
            "- [-1, 1, HWDown, [768]] # 8-P5/32（削窄）",
            "- [-1, 2, C3k2_SXQ, [768, True]] # 9-P5/32",
            "- [-1, 1, SPPF_LSKA, [768, 5]] # 10",
            "- [-1, 2, C2PSA, [768]] # 11",
        ],
        [
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 12',
            "- [[-1, 7], 1, BiFPNConcat, []] # 13",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 14-P4",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 15',
            "- [[-1, 5], 1, BiFPNConcat, []] # 16",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 17-P3",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 18',
            "- [[-1, 3], 1, BiFPNConcat, []] # 19",
            "- [-1, 2, C3k2_SXQ, [128, False]] # 20-P2/4 高分辨率层",
            "- [-1, 1, Conv, [128, 3, 2]] # 21",
            "- [[-1, 17], 1, BiFPNConcat, []] # 22",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 23-P3",
            "- [-1, 1, Conv, [256, 3, 2]] # 24",
            "- [[-1, 14], 1, BiFPNConcat, []] # 25",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 26-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 27",
            "- [[-1, 11], 1, BiFPNConcat, []] # 28",
            "- [-1, 2, C3k2_SXQ, [768, True]] # 29-P5",
            "- [[20, 23, 26, 29], 1, Segment, [nc, 32, 256]] # 30-Segment(P2,P3,P4,P5)",
        ],
    )
    emit(
        "SXQNet-V7-fast.yaml",
        [
            "SXQNet-V7 Fast（延迟轴）：回答'实测延迟优先时留什么'。全 PConv + HSF（通道减半）",
            "  + SimAM（0 参数注意力）+ 去 C2PSA + nearest 上采样——每层都选延迟最优件。",
            "消融: vs V2（V2 重参数量，V7 重延迟——两种轻量化的区别行）| " + sxq.format(f="SXQNet-V7-fast.yaml", n="SXQ_V7"),
        ],
        [
            "- [-1, 1, Conv, [64, 3, 2]] # 0-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 1-P2/4",
            "- [-1, 2, C3k2_Faster, [256, False, 0.25]] # 2-P2/4",
            "- [-1, 1, Conv, [256, 3, 2]] # 3-P3/8（步长卷积=最快下采样）",
            "- [-1, 2, C3k2_Faster, [512, False, 0.25]] # 4-P3/8",
            "- [-1, 1, Conv, [512, 3, 2]] # 5-P4/16",
            "- [-1, 2, C3k2_Faster, [512, True]] # 6-P4/16",
            "- [-1, 1, Conv, [1024, 3, 2]] # 7-P5/32",
            "- [-1, 2, C3k2_Faster, [1024, True]] # 8-P5/32",
            "- [-1, 1, SPPF, [1024, 5]] # 9",
            "- [-1, 1, C3k2_Faster, [1024, True]] # 10（去 C2PSA）",
        ],
        [
            "- [[6, 10], 1, HSF, []] # 11: P4 筛选融合（通道不翻倍=更快）",
            "- [-1, 1, C3k2_Faster, [512, False]] # 12-P4",
            "- [[4, 12], 1, HSF, []] # 13: P3 筛选融合",
            "- [-1, 1, C3k2_Faster, [256, False]] # 14-P3",
            "- [-1, 1, SimAM, []] # 15: 零参数注意力",
            "- [-1, 1, Conv, [256, 3, 2]] # 16",
            "- [[-1, 12], 1, Concat, [1]] # 17",
            "- [-1, 1, C3k2_Faster, [512, False]] # 18-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 19",
            "- [[-1, 10], 1, Concat, [1]] # 20",
            "- [-1, 1, C3k2_Faster, [1024, True]] # 21-P5",
            "- [[15, 18, 21], 1, Segment, [nc, 32, 256]] # 22",
        ],
    )
    emit(
        "SXQNet-V8-texture.yaml",
        [
            "SXQNet-V8 Texture（纹理轴，用户思想主线版）：回答'绿绿伪装主导时怎么办'（|Δa*|=2.2）。",
            "  = TGP 前端（图像域先验）+ TDAM@P2P3（特征域差分放大）+ CSFG + LIAM——纹理全链路。",
            "消融: vs F61（无 CSFG/LIAM）| " + sxq.format(f="SXQNet-V8-texture.yaml", n="SXQ_V8"),
        ],
        [
            "- [-1, 1, TGP, []] # 0: 纹理先验前端（原创）",
            "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
            "- [-1, 2, C3k2_SXQ, [256, False, 0.25]] # 3-P2/4",
            "- [-1, 1, TDAM, []] # 4-P2 纹理差分（原创）",
            "- [-1, 1, Conv, [256, 3, 2]] # 5-P3/8",
            "- [-1, 2, C3k2_SXQ, [512, False, 0.25]] # 6-P3/8",
            "- [-1, 1, TDAM, []] # 7-P3 纹理差分（原创）",
            "- [-1, 1, Conv, [512, 3, 2]] # 8-P4/16",
            "- [-1, 2, C3k2_SXQ, [512, True]] # 9-P4/16",
            "- [-1, 1, Conv, [1024, 3, 2]] # 10-P5/32",
            "- [-1, 2, C3k2_SXQ, [1024, True]] # 11-P5/32",
            "- [-1, 1, SPPF, [1024, 5]] # 12",
            "- [-1, 2, C2PSA, [1024]] # 13",
        ],
        [
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 14',
            "- [[-1, 9], 1, Concat, [1]] # 15",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 16-P4",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 17',
            "- [[-1, 7], 1, Concat, [1]] # 18",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 19-P3",
            "- [[4, 19], 1, CSFG, []] # 20: P2(TDAM后) 细节注入（原创）",
            "- [-1, 1, LIAM, []] # 21: 亮度不变注意力（原创）",
            "- [-1, 1, Conv, [256, 3, 2]] # 22",
            "- [[-1, 16], 1, Concat, [1]] # 23",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 24-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 25",
            "- [[-1, 13], 1, Concat, [1]] # 26",
            "- [-1, 2, C3k2_SXQ, [1024, True]] # 27-P5",
            "- [[21, 24, 27], 1, Segment, [nc, 32, 256]] # 28",
        ],
    )
    emit(
        "SXQNet-V9-dark.yaml",
        [
            "SXQNet-V9 Dark（暗光轴）：回答'逆光/夜间/阴天主导时怎么办'（小果 V=103 且比背景暗）。",
            "  = LCE（图像域曲线）+ DFEM@P3（特征域频率+暗区）+ LumiFormer@P3颈部（Former 级）",
            "  ——暗光三级火箭；训练配 --aug-preset dark。",
            "消融: vs F23（HVI+DFEM 版做图像域前端横评）| " + sxq.format(f="SXQNet-V9-dark.yaml", n="SXQ_V9") + " --aug-preset dark",
        ],
        [
            "- [-1, 1, LCE, [4, 16]] # 0: 暗区曲线前端（原创）",
            "- [-1, 1, Conv, [64, 3, 2]] # 1-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 2-P2/4",
            "- [-1, 2, C3k2_SXQ, [256, False, 0.25]] # 3-P2/4",
            "- [-1, 1, Conv, [256, 3, 2]] # 4-P3/8",
            "- [-1, 2, C3k2_SXQ, [512, False, 0.25]] # 5-P3/8",
            "- [-1, 1, DFEM, []] # 6-P3 双域频率增强（原创）",
            "- [-1, 1, Conv, [512, 3, 2]] # 7-P4/16",
            "- [-1, 2, C3k2_SXQ, [512, True]] # 8-P4/16",
            "- [-1, 1, Conv, [1024, 3, 2]] # 9-P5/32",
            "- [-1, 2, C3k2_SXQ, [1024, True]] # 10-P5/32",
            "- [-1, 1, SPPF, [1024, 5]] # 11",
            "- [-1, 2, C2PSA, [1024]] # 12",
        ],
        [
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 13',
            "- [[-1, 8], 1, Concat, [1]] # 14",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 15-P4",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
            "- [[-1, 6], 1, Concat, [1]] # 17",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 18-P3",
            "- [-1, 1, LumiFormer, [1]] # 19: 亮度感知 Former（原创）",
            "- [-1, 1, Conv, [256, 3, 2]] # 20",
            "- [[-1, 15], 1, Concat, [1]] # 21",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 22-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 23",
            "- [[-1, 12], 1, Concat, [1]] # 24",
            "- [-1, 2, C3k2_SXQ, [1024, True]] # 25-P5",
            "- [[19, 22, 25], 1, Segment, [nc, 32, 256]] # 26",
        ],
    )
    emit(
        "SXQNet-V10-max.yaml",
        [
            "SXQNet-V10 Max（上限轴）：回答'全部手段齐上性能天花板在哪'。= V1 骨干/颈部 + P2 四层头。",
            "  参数仍 ≤ 基线；FLOPs 为全家最高（P2 头代价），作性能上界与蒸馏教师候选。",
            "训练: " + sxq.format(f="SXQNet-V10-max.yaml", n="SXQ_V10") + " --iou-type NWDWise --tal-metric NWD --tal-min-pos --freq-loss 0.1",
        ],
        [
            "- [-1, 1, TGP, []] # 0: 纹理先验前端（原创）",
            "- [-1, 1, LCE, [4, 16]] # 1: 暗区曲线前端（原创）",
            "- [-1, 1, Conv, [64, 3, 2]] # 2-P1/2",
            "- [-1, 1, Conv, [128, 3, 2]] # 3-P2/4",
            "- [-1, 4, C3k2_LS, [256, False, 0.25]] # 4-P2/4",
            "- [-1, 1, TDAM, []] # 5-P2 纹理差分（原创）",
            "- [-1, 1, HWDown, [256]] # 6-P3/8",
            "- [-1, 4, C3k2_LS, [512, False, 0.25]] # 7-P3/8",
            "- [-1, 1, MWCA, []] # 8-P3 跨频带注意力（原创）",
            "- [-1, 1, HWDown, [512]] # 9-P4/16",
            "- [-1, 2, C3k2_LS, [512, True]] # 10-P4/16",
            "- [-1, 1, HWDown, [768]] # 11-P5/32",
            "- [-1, 2, C3k2_LS, [768, True]] # 12-P5/32",
            "- [-1, 1, SPPF_LSKA, [768, 5]] # 13",
            "- [-1, 1, HyperACE, [8]] # 14: 超图关联（原创适配）",
            "- [-1, 1, HyperRes, [2]] # 15: 双流超连接残差 x2（企业技巧融合）",
        ],
        [
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 16',
            "- [[-1, 10], 1, BiFPNConcat, []] # 17",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 18-P4",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 19',
            "- [[-1, 8], 1, BiFPNConcat, []] # 20",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 21-P3",
            '- [-1, 1, nn.Upsample, [None, 2, "nearest"]] # 22',
            "- [[-1, 5], 1, BiFPNConcat, []] # 23",
            "- [-1, 2, C3k2_SXQ, [128, False]] # 24-P2/4 高分辨率层",
            "- [-1, 1, Conv, [128, 3, 2]] # 25",
            "- [[-1, 21], 1, BiFPNConcat, []] # 26",
            "- [-1, 2, C3k2_SXQ, [256, False]] # 27-P3",
            "- [-1, 1, Conv, [256, 3, 2]] # 28",
            "- [[-1, 18], 1, BiFPNConcat, []] # 29",
            "- [-1, 2, C3k2_SXQ, [512, False]] # 30-P4",
            "- [-1, 1, Conv, [512, 3, 2]] # 31",
            "- [[-1, 15], 1, BiFPNConcat, []] # 32",
            "- [-1, 2, C3k2_SXQ, [768, True]] # 33-P5",
            "- [[24, 27, 30, 33], 1, Segment, [nc, 32, 256]] # 34-Segment(P2,P3,P4,P5)",
        ],
    )

    print("done.")


if __name__ == "__main__":
    main()
