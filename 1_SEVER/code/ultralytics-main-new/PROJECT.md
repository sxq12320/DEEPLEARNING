# Project: 自动控制理论驱动的柑橘幼果实例分割网络规划方案 (Citrus Control Backbone Design)

## Architecture
A publication-grade deep learning architecture bridging classical control theory (closed-loop feedback, state observer, PID-style spatial-semantic-gradient regulation, Lyapunov-bounded residual stability) with YOLO11 instance segmentation. The architecture integrates proven high-efficiency components:
- Control-inspired Backbone (`C3k2Ctrl` / `ObserverBlock`) with closed-loop error correction and state estimation.
- LSKA strip pooling (7/11/21 kernels) in SPPF for large anisotropic receptive fields along foliage/branches.
- CARAFE content-aware reconstruction upsampler in Neck for sharp mask boundaries.
- Haar Wavelet Downsampling (`HWDown`) for lossless high-frequency preservation during downsampling.
- `SegmentCitrusLite` compact decoupled head for nano-scale parameter efficiency.

## Feature Inventory
| # | Feature | Description | Milestone | Source | Status |
|---|---------|-------------|-----------|--------|:------:|
| 1 | Failure Mode Analysis | In-depth analysis of open-loop CNN degradation under green-on-green camouflage, specular glare, and strip occlusions | M1 | ORIGINAL_REQUEST R1.1 | DONE |
| 2 | State Observer & Feedback Formulation | Continuous & discrete state observer equations (r, x_hat, e=r-y, K(s) / Luenberger observer) | M1 | ORIGINAL_REQUEST R1.2 | DONE |
| 3 | Tri-Branch PID-Inspired Balance | Proportional (spatial details), Integral (historical semantics), Derivative (boundary gradients) mathematical formulation | M1 | ORIGINAL_REQUEST R1.3 | DONE |
| 4 | Control-Driven Backbone Block (`C3k2Ctrl`) | Detailed internal mechanics, channel configs, residual bounds, and 100% official YOLO11 weight key compatibility | M2 | ORIGINAL_REQUEST R2.1 | DONE |
| 5 | Proven Components Integration | SPPF-LSKA (7/11/21), CARAFE Neck, HWDown, SegmentCitrusLite head specifications | M2 | ORIGINAL_REQUEST R2.2 | DONE |
| 6 | Visual & Structural Blueprints | Complete layer-by-layer YAML specification, ASCII diagrams, and Mermaid signal flowcharts | M2 | ORIGINAL_REQUEST R2.3 | DONE |
| 7 | Complexity Budget & Hardware Constraints | Nano scale parameters <= 3.20 M, GFLOPs <= 11.5 G, GPU latency <= 1.20x YOLO11n-seg, zero unverified heavy redundancy | M3 | ORIGINAL_REQUEST R3.1-3.3 | DONE |
| 8 | 8-Model Ablation Matrix | Rigorous 8-stage ablation protocol isolating individual control and structural components | M4 | ORIGINAL_REQUEST R4.1 | DONE |
| 9 | Pre-Experiment Gates & Metrics | 4 validation gates (Dry-run YAML, GPU speed, 3-epoch smoke, 50-epoch screening) and specialized challenge metrics (mAP, AP-tiny, solidity deficit, split/merge error) | M4 | ORIGINAL_REQUEST R4.2-4.3 | DONE |
| 10| Full Publication-Grade Document Synthesis | Deliverable markdown `20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md` | M5 | ORIGINAL_REQUEST Deliverable | DONE |

## Milestones
| # | Name | Scope | Dependencies | Status |
|---|------|-------|-------------|:------:|
| M0 | Survey & Codebase Mining | Explore YOLO11 codebase, existing modules, parameter accounting, and control theory frameworks | none | DONE |
| M1 | Control Theory & Mathematical Modeling | R1: Theoretical equations, failure modes, state observer, PID tri-branch | M0 | DONE |
| M2 | Architectural Specification & YAML Blueprint | R2: C3k2Ctrl, SPPF-LSKA, CARAFE, HWDown, SegmentCitrusLite, YAML, diagrams | M0, M1 | DONE |
| M3 | Complexity & Hardware Guardrails | R3: Exact parameter, GFLOPs, latency budget, layer-by-layer calculations | M0, M2 | DONE |
| M4 | Ablation Protocol & Experimental Roadmap | R4: 8-model ablation matrix, 4 pre-experiment gates, specialized challenge metrics | M1, M2, M3 | DONE |
| M5 | Comprehensive Synthesis & Multi-Agent Verification | Publication-grade deliverable generation, review, audit, and sign-off | M1, M2, M3, M4 | DONE |

## Interface Contracts
- Document Output: `E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`
- Gate Verification: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/orchestrator_1/GATE_STATUS.md` (Gate Result: PASS)
