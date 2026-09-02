# Task Assignment: Ablation Matrix, Gates & Metrics Stress Test

## Objective
Adversarially challenge the 8-model ablation protocol, 4 pre-experiment validation gates, and challenge metrics in `20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md` (R4).

## Inputs
- Primary Deliverable: `E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`
- Request: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/ORIGINAL_REQUEST.md`
- Project Plan: `E:/mastercode/1_SEVER/code/ultralytics-main-new/PROJECT.md`

## Challenge Criteria
1. Evaluate whether the 8-model ablation progression (G00–G07) cleanly isolates every variable without confounded factors.
2. Evaluate whether the 4 pre-experiment gates have concrete, automatable, unambiguous pass/fail criteria.
3. Evaluate the mathematical soundness and practical measurability of the specialized challenge metrics ($\text{AP}_{\text{tiny}}$, $\Delta\text{Solidity}$, $E_{\text{split}}$, $E_{\text{merge}}$).
4. Deliver verdict: `APPROVE` or `REQUEST_CHANGES` in `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/challenger_ablation_1/handoff.md`.
