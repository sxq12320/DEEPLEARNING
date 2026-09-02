# Task Assignment: Re-Challenge Ablation Protocol, Validation Gates & Metrics

## Objective
Verify that the updates made to Section 5 of `20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md` fully resolve all previous concerns raised by `challenger_ablation_1`.

## Inputs
- Primary Deliverable: `E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`
- Previous Challenger Report: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/challenger_ablation_1/handoff.md`
- Worker Refinement Report: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/worker_refine_1/handoff.md`

## Re-Challenge Criteria
1. Verify that Gate 1 properly scopes the $\le 3.20\text{ M}$ hard cap to G07 and specifies $\pm 2.0\%$ target envelopes for G00–G06.
2. Verify that Gate 2 includes CUDA synchronization.
3. Verify that Gate 3 has warmup-aware gradient norms and divergence ceilings.
4. Verify that Gate 4 properly defines screening targets across the ablation matrix.
5. Verify that $E_{\text{merge}}$ uses asymmetric ground-truth coverage ($\text{Cov} \ge 0.30$), $E_{\text{split}}$ uses fragment purity and minimum relative area, $\Delta\text{Solidity}$ uses pixel-based summation, and $\text{AP}_{\text{tiny}}$ uses standard COCO `areaRng = [0, 256]`.
6. Deliver verdict: `APPROVE` or `REQUEST_CHANGES` in `.agents/challenger_ablation_2/handoff.md`.
