"""Preserve official AP; expose empirical PR endpoints and pre/post-NMS failures.

This is a CPU single-class diagnostic, not an alternate leaderboard metric.
The plotted empirical curve ends at observed recall without synthetic zeros.
"""

# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.diagnose_citrus_candidate_stages import StageValidator
from ultralytics import YOLO


def render_pr(pr, output):
    """Dependency-free SVG; JSON retains every empirical point."""
    index = np.unique(np.linspace(0, len(pr["recall"]) - 1, min(2500, len(pr["recall"]))).astype(int))
    points = " ".join(f"{70 + 600 * pr['recall'][i]:.2f},{440 - 380 * pr['precision'][i]:.2f}" for i in index)
    cutoff = 70 + 600 * pr["maximum_recall"]
    ticks = "".join(
        f'<text x="{70 + 120*i}" y="465" text-anchor="middle">{i/5:.1f}</text>'
        f'<text x="55" y="{445 - 76*i}" text-anchor="end">{i/5:.1f}</text>' for i in range(6)
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="730" height="530" viewBox="0 0 730 530">
<rect width="730" height="530" fill="white"/>
<g font-family="Arial,sans-serif" font-size="14" fill="#243447">
<text x="70" y="27" font-size="20">Empirical mask PR — no synthetic tail</text>
<text x="70" y="49">Observed max recall: {pr['maximum_recall']:.3f}; official AP unchanged</text>
<rect x="{cutoff:.2f}" y="60" width="{670-cutoff:.2f}" height="380" fill="#e9eef4"/>
<path d="M70 60V440H670" fill="none" stroke="#243447"/>
<polyline points="{points}" fill="none" stroke="#145da0" stroke-width="1.5"/>
<path d="M{cutoff:.2f} 60V440" stroke="#b44a33" stroke-dasharray="5 4"/>
{ticks}<text x="345" y="493">Recall</text>
<text transform="translate(20 280) rotate(-90)">Precision</text>
<text x="70" y="518" font-size="12">Shaded region: recall not reached at evaluated conf / NMS / max_det settings.</text>
</g></svg>'''
    (output / "empirical_mask_pr.svg").write_text(svg, encoding="utf-8")


class EmpiricalPRValidator(StageValidator):
    def get_stats(self):
        stats = {k: np.concatenate(v, 0) for k, v in self.metrics.stats.items()}
        if len(np.unique(stats["target_cls"])) != 1:
            raise ValueError("This diagnostic currently supports one class only")
        order = np.argsort(-stats["conf"], kind="stable")
        tp = stats["tp_m"][order, 0].astype(float).cumsum()
        precision = tp / np.arange(1, len(tp) + 1)
        recall = tp / len(stats["target_cls"])
        self.empirical_pr = dict(
            recall=recall.tolist(), precision=precision.tolist(),
            confidence=stats["conf"][order].tolist(),
            maximum_recall=float(recall[-1]) if len(recall) else 0.0,
            final_precision=float(precision[-1]) if len(precision) else 0.0,
            gt_instances=len(stats["target_cls"]), predictions=len(tp),
        )
        return super().get_stats()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--render-only", action="store_true", help="Render existing diagnostic.json without inference")
    args = parser.parse_args()
    if args.render_only:
        render_pr(json.loads((args.output / "diagnostic.json").read_text())["empirical_pr"], args.output)
        return
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.mkdir(parents=True)
    torch.set_num_threads(2)
    validator = EmpiricalPRValidator(
        args=dict(data=str(args.data.resolve()), device="cpu", imgsz=640, batch=1,
                  workers=0, cache=True, plots=False, half=False, conf=0.001,
                  iou=0.7, max_det=300, rect=False, overlap_mask=True, mask_ratio=4),
        save_dir=args.output / "evaluation",
    )
    validator.stage_records = []
    metrics = validator(model=YOLO(str(args.weights)).model)
    records = validator.stage_records
    groups = {"all": records, "tiny": [r for r in records if r["area"] < 256]}
    summary = {k: dict(n=len(v), buckets=dict(Counter(r["bucket"] for r in v))) for k, v in groups.items()}
    payload = dict(
        weights=str(args.weights), metrics=metrics, summary=summary,
        empirical_pr=validator.empirical_pr, instances=records,
        limits="CPU FP32, batch1, no server pixel hash comparison. PR uses official post-NMS mask TP@0.5; "
        "failure buckets instead use fixed conf .25 and box-first greedy matching. Tiny=raster area<256 at640. "
        "Neither subset recall nor greedy buckets are AP_small or causal proof. Original metrics are unchanged.",
    )
    (args.output / "diagnostic.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    pr = validator.empirical_pr
    render_pr(pr, args.output)
    print(json.dumps(dict(summary=summary, endpoint={k: v for k, v in pr.items() if not isinstance(v, list)}), indent=2))


if __name__ == "__main__":
    main()
