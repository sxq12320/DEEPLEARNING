# Task Assignment: Refine Master Design Document per Challenger Feedback

## Objective
Update `E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md` to incorporate the high-value refinements identified by `challenger_ablation_1`.

## Inputs
- Target Deliverable: `E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`
- Challenger Feedback: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/challenger_ablation_1/handoff.md`
- Gate Status: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/orchestrator_1/GATE_STATUS.md`

## Required Edits to `20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`
1. **Section 5.2 (Pre-Experiment Validation Gates)**:
   - **Gate 1 (Dry-Run YAML Build Gate)**: Clarify that the strict $\le 3.20\text{ M}$ hard cap applies to the final proposed production model G07. For intermediate ablation stages G00–G06, Gate 1 validates parameter counts against their model-specific design envelopes ($\pm 2\%$ of target: G00: 2.84M, G01: 2.84M, G02: 2.98M, G03: 3.22M, G04: 3.24M, G05: 3.38M, G06: 3.12M).
   - **Gate 2 (GPU Speed & Memory Gate)**: Add explicit CUDA synchronization requirement (`torch.cuda.synchronize()`) before and after timing loops to avoid asynchronous timing distortions.
   - **Gate 3 (3-Epoch Smoke Convergence Gate)**: Account for YOLO 3-epoch warmup where learning rate ramps up. Criteria: finite bounded gradient norm ($\|\mathbf{g}\|_2 \in [0.001, 20.0]$, zero NaN/Inf) and divergence cap $\mathcal{L}_{\text{epoch3}} \le 2.0 \times \mathcal{L}_{\text{epoch1}}$.
   - **Gate 4 (50-Epoch Fast Screening Gate)**: Clarify screening gate behavior across the ablation spectrum (G01 sanity baseline $\Delta\text{mAP} \approx 0.0\% \pm 0.3\%$; exploratory architecture variants requiring $\Delta\text{Mask mAP50-95} \ge +1.5\%$ before graduating to full 300-epoch runs).
2. **Section 5.3 (Target Challenge Metrics & Error Quantification Protocol)**:
   - **$E_{\text{merge}}$ (Merge Error Rate)**: Update formula to use ground-truth coverage/recall ($\text{Cov}(M_i^{\text{pred}}, M_j^{\text{gt}}) = \frac{|M_i^{\text{pred}} \cap M_j^{\text{gt}}|}{|M_j^{\text{gt}}|} \ge 0.30$) rather than symmetric IoU, ensuring that large mask blobs merging $\ge 2$ ground-truth fruits are correctly counted.
   - **$E_{\text{split}}$ (Split Error Rate)**: Update formula to include minimum fragment relative area threshold ($|M_i^{\text{pred}} \cap M_j^{\text{gt}}| \ge 0.05 |M_j^{\text{gt}}|$) and fragment purity ($\ge 0.50$), preventing tiny single-pixel noise from inflating split counts while capturing genuine split fragments.
   - **$\Delta\text{Solidity}$ (Solidity Deficit)**: Explicitly note pixel-summation $\sum M$ for area computation to capture interior solar glare holes and waxy reflection washouts.
   - **$\text{AP}_{\text{tiny}}$**: Note implementation via standard COCO API `areaRng = [0, 256]` ($16\times 16\text{ px}$).

## Mandatory Integrity Warning
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

## Handoff
Save the updated document to `E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`.
Write your handoff report to `.agents/worker_refine_1/handoff.md`.
