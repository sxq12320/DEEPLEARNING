"""verify_far_yamls.py — F 系列 yaml 全量构建/前向/反向冒烟 + 参数量/GFLOPs 报表.

用法（无需 pip install -e，本脚本自动把 fork 加入 sys.path）:
    python verify_far_yamls.py            # 全部 F*.yaml
    python verify_far_yamls.py F31        # 只测名字含 F31 的
输出: 逐行 OK/FAIL + params/GFLOPs，并写 0_orange_yaml/1_far_small/_verify_report.csv。
大实验前先跑本脚本 + 3-epoch smoke（见 AGENTS.md）。
"""

from __future__ import annotations

import csv
import glob
import os
import sys

FORK_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, FORK_ROOT)

import torch  # noqa: E402

from ultralytics.nn.tasks import SegmentationModel  # noqa: E402

# 反向传播冒烟名单（训练态 forward + sum-backward，验证梯度图完整）
BACKWARD_SMOKE = {"F19", "F22", "F23", "F31", "F38", "F40", "F41", "F42", "F43", "F46", "F47", "F48",
                  "F49", "F50", "F51", "F52", "F53", "F54", "F55", "F56", "F57", "F58", "F60", "F61",
                  "F64", "F65", "SXQ"}


def _flops_640(model) -> float:
    """thop 在 640 全分辨率直测（×2 换算 MACs→FLOPs，与 Ultralytics 口径一致）.

    不用 ultralytics get_flops 的 stride-trick：含固定代价模块（如 LRSA 的 8x8 注意力）时
    其 (640/stride)^2 外推会严重高估。thop 失败（如 HVIEnhance）返回 0。
    """
    try:
        import thop

        flops, _ = thop.profile(model, inputs=(torch.zeros(1, 3, 640, 640),), verbose=False)
        return flops * 2 / 1e9
    except Exception:  # noqa: BLE001
        return 0.0


def check_one(yaml_path: str) -> tuple[str, str, float, float]:
    name = os.path.basename(yaml_path)
    try:
        model = SegmentationModel(cfg=yaml_path, ch=3, nc=1, verbose=False)
        n_p = sum(p.numel() for p in model.parameters()) / 1e6
        model.eval()
        flops = _flops_640(model)
        with torch.no_grad():
            model(torch.zeros(1, 3, 640, 640))
        if name[:3] in BACKWARD_SMOKE:
            model.train()
            out = model(torch.zeros(2, 3, 640, 640))
            loss = sum(o.sum() for o in _flatten(out))
            loss.backward()
            n_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
            n_all = sum(1 for _ in model.parameters())
            note = f"OK (backward: {n_grad}/{n_all} params got grad)"
        else:
            note = "OK"
        return name, note, round(n_p, 3), round(flops, 1)
    except Exception as e:  # noqa: BLE001
        return name, f"FAIL {type(e).__name__}: {e}", 0.0, 0.0


def _flatten(out):
    if isinstance(out, torch.Tensor):
        if out.is_floating_point():
            yield out
    elif isinstance(out, (list, tuple)):
        for o in out:
            yield from _flatten(o)
    elif isinstance(out, dict):
        for o in out.values():
            yield from _flatten(o)


def main() -> int:
    pat = sys.argv[1] if len(sys.argv) > 1 else ""
    yamls = sorted(
        glob.glob(os.path.join(FORK_ROOT, "0_orange_yaml", "1_far_small", "F*.yaml"))
        + glob.glob(os.path.join(FORK_ROOT, "0_orange_yaml", "1_far_small", "SXQNet*.yaml"))
    )
    yamls = [y for y in yamls if pat in os.path.basename(y)]
    rows, n_fail = [], 0
    for y in yamls:
        name, note, n_p, flops = check_one(y)
        rows.append((name, note, n_p, flops))
        status = note if note.startswith("OK") else note[:120]
        print(f"{name:44s} {n_p:7.3f}M {flops:6.1f}G  {status}")
        if not note.startswith("OK"):
            n_fail += 1
    report = os.path.join(FORK_ROOT, "0_orange_yaml", "1_far_small", "_verify_report.csv")
    with open(report, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["yaml", "status", "params_M", "GFLOPs_640"])
        w.writerows(rows)
    print(f"\n{len(rows) - n_fail}/{len(rows)} passed. report -> {report}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
