## 2026-08-27T07:20:25Z

You are Worker 3 (Experiment Plan, Reproducibility & Visualization Lead).

MANDATORY FIRST STEP:
Read the authoritative user request at:
E:\mastercode\.agents\ORIGINAL_REQUEST.md
Also read repository rules in:
E:\mastercode\AGENTS.md
Read the survey handoff reports:
- E:\mastercode\.agents\explorer_audit_1\handoff.md
- E:\mastercode\.agents\explorer_lit_1\handoff.md
- E:\mastercode\.agents\explorer_repo_1\handoff.md

MANDATORY INTEGRITY WARNING:
DO NOT CHEAT. All implementations must be genuine. DO NOT hardcode test results, create dummy/facade implementations, or circumvent the intended task. A teamwork_preview_auditor will independently verify your work. Integrity violations WILL be detected and your work WILL be rejected.

Your working directory is:
E:\mastercode\.agents\worker_batch3_1\

Deliverable Files to Generate in E:\mastercode\3_研究生\architecture_search_20260827\:
1. 09_ablation_and_experiment_plan.md
   - Formal 3-seed benchmark plan (seeds 42, 43, 44) on group-aware deduplicated dataset reporting mean +- std.
   - Comprehensive factorial ablation matrix (Baseline S00, +RepContext S01, +ScaleFusion, +Lite Head S04, +Training-Aux BQ S09 -> Final CitrusB-Seg).
   - Challenge subset construction and evaluation protocol:
     * `strip_occlusion_concave` (Solidity < 0.85)
     * `touching_cluster` (Inter-instance distance <= 4px)
     * `extreme_scale_tiny` (Area < 32^2 px^2, short edge < 16px)
     * `camouflage_low_contrast` (Delta E_Lab < 15)
   - Cross-family comparison baseline protocol: YOLOv8n-seg, YOLO11n-seg, YOLO26n-seg, RTMDet-Ins-tiny, Mask R-CNN R50-FPN, RF-DETR Seg Nano, SOLOv2-Light R18-FPN, U-Net + marker-controlled watershed.
2. 10_reproducibility_checklist.md
   - Exact step-by-step reproduction guide: environment setup, PyTorch/CUDA versions, training execution commands, evaluation commands.
   - Exact hyperparameter specifications (300 epochs, AdamW, lr0=0.001, lrf=0.01, imgsz=640, batch=4, close_mosaic=10).
   - Validation & invalidation criteria.
3. references.bib
   - Comprehensive, fully verified BibTeX bibliography of all 28+ papers cited across the project (CVPR, ICCV, ECCV, NeurIPS, TPAMI, TIP, CEA, Frontiers, etc.) with accurate authors, titles, venues, years, and authentic DOIs / arXiv IDs.
4. architecture_overview.mmd
   - Detailed Mermaid flowchart illustrating the complete CitrusB-Seg end-to-end dataflow:
     * Backbone feature stages (P1 to P5) with SPPFRepContext and C2PSA
     * Neck FPN/PAN bidirectional flow with CitrusScaleFusion at P3
     * Lite Decoupled Segmentation Heads (Boxes, Classes, Mask Protos, Coefficients)
     * Training-time auxiliary branch (P2/P3 boundary loss, sparse query loss, contrast loss) and inference disconnection.

Write all 4 files completely to E:\mastercode\3_研究生\architecture_search_20260827\, verify their syntax and contents, write your handoff report to E:\mastercode\.agents\worker_batch3_1\handoff.md, and send a completion message.
