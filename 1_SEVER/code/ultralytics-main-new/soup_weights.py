"""soup_weights.py — Model Soup 权重汤：把多个 seed/配方的 best.pt 权重平均，白捡精度.

LLM/大模型界标配技巧的检测落地：同架构多次训练（不同 seed 或轻微配方差异）的权重
做逐参数平均，通常免费 +0.3~1.0 精度且推理零开销（uniform soup）。
Reference: Model Soups (Wortsman et al., ICML 2022, arXiv:2203.05482)；SWA (arXiv:1803.05407)；
LAWA 训练末期权重平均 (arXiv:2209.14981)。
注意（theme14 核验）：
1) 只能平均**同一架构、同一训练协议**的权重（3 seeds 的 F53 可以；F53 与 F52 不可以）；
2) **平均后必须在训练集上重估 BN 统计量**（forward 几百个 batch 刷新 running mean/var）再评测，
   否则 BN 统计与平均权重失配会掉点——用 --recalib 提示流程；
3) 进阶：greedy soup（按验证集逐个尝试加入，优于 uniform）可手动按本工具多次组合实现。

用法：
    python soup_weights.py --weights run1/best.pt run2/best.pt run3/best.pt --out soup.pt
    python eval_citrus_seg.py --weights soup.pt   # 与单 seed 对比，必须在同一 test split
"""

from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", nargs="+", required=True, help="2 个以上同架构 best.pt")
    ap.add_argument("--out", default="soup.pt")
    args = ap.parse_args()
    assert len(args.weights) >= 2, "至少 2 个权重才能做汤"

    ckpts = [torch.load(w, map_location="cpu", weights_only=False) for w in args.weights]
    base = ckpts[0]
    model = base["ema"] or base["model"]
    sd = model.state_dict()
    keys = list(sd.keys())
    for c in ckpts[1:]:
        m2 = c["ema"] or c["model"]
        sd2 = m2.state_dict()
        assert list(sd2.keys()) == keys, "架构不一致，不能平均（Model Soup 只适用于同架构同协议）"
    for k in keys:
        if sd[k].dtype.is_floating_point:
            acc = sd[k].float().clone()
            for c in ckpts[1:]:
                acc += (c["ema"] or c["model"]).state_dict()[k].float()
            sd[k] = (acc / len(ckpts)).to(sd[k].dtype)
    model.load_state_dict(sd)
    base["ema"] = model
    base["model"] = model
    torch.save(base, args.out)
    print(f"uniform soup of {len(ckpts)} checkpoints -> {args.out}")


if __name__ == "__main__":
    main()
