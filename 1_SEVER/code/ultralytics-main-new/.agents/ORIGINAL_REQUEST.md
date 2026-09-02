# Original User Request

## 2026-09-02T13:17:47+08:00

Formulate a comprehensive, publication-grade network architecture design and planning document (自动控制理论驱动的柑橘幼果实例分割网络规划方案) that bridges classical control theory (closed-loop feedback, state observer, PID-style frequency regulation) with YOLO instance segmentation, detailing how to reform the backbone while harmoniously integrating proven winners (LSKA, CARAFE, HWDown, Lite Head).

Working directory: E:/mastercode/1_SEVER/code/ultralytics-main-new
Integrity mode: development

## Requirements

### R1. Mathematical & Control-Theory Theoretical Grounding (自动控制理论与神经网络机制建模)
Formulate the theoretical foundations mapping classical control theory to deep feature representations:
1. Explain how open-loop CNN feedforward propagation fails under foliage camouflage, strong solar glares, and strip-like occlusion.
2. Formulate the Closed-Loop Feedback Error Regulation (闭环负反馈误差校正) and State Observer (状态观测器) mechanisms: define the reference signal r, state estimation x_hat, error signal e = r - y, and feedback regulator K(s) in continuous/discrete feature space.
3. Formulate the PID-inspired tri-branch dynamic balance (Proportional spatial details, Integral historical semantics, Derivative boundary gradients).

### R2. End-to-End Architectural Specification (端到端网络结构全景设计)
Provide an exact, layer-by-layer architectural blueprint:
1. Backbone design: Specify the internal mechanics of the new control-inspired block (e.g., C3k2Ctrl / ObserverBlock), including channel configurations, residual bounds, and 100% official pretrained weight key compatibility.
2. Neck and Head integration: Explicitly define the connection with verified top-tier components:
   - SPPF: LSKA strip pooling (7/11/21 kernels)
   - Neck: CARAFE content-aware reconstruction upsampler
   - Downsampling: Haar Wavelet Downsampling (HWDown)
   - Head: SegmentCitrusLite compact decoupled head
3. Include clear ASCII diagrams and Mermaid flowchart diagrams illustrating the closed-loop signal routing and data flow.

### R3. Strict Complexity Budget & Hardware Constraints (算力与硬件资源预算)
Establish strict architectural guardrails:
1. Model capacity: Total parameters strictly constrained to <= 3.20 M (Nano scale), GFLOPs@640 strictly <= 11.5 G.
2. Latency constraint: Target GPU forward/backward step time must not exceed 1.20x relative to the official YOLO11n-seg baseline.
3. Zero-redundancy rule: Avoid any heavy multi-head additions or unverified heavy attention layers.

### R4. Complete Ablation Protocol & Experimental Roadmap (完整的消融矩阵与实验方案)
Formulate the staged experimental verification roadmap:
1. 8-model ablation matrix isolating: Baseline -> Control Backbone only -> Observer Feedback only -> Joint Control Backbone + LSKA -> Joint + CARAFE -> Full Proposed Method.
2. Pre-experiment gates: Dry-run YAML build test, GPU speed gate, 3-epoch smoke test, 50-epoch screening gate against G00 baseline.
3. Target challenge metrics definition: Precision, Recall, Mask mAP50-95, Mask mAP50, AP-tiny, Solidity deficit analysis, and split/merge error quantification.

Scope Directive: Deliverable is a complete, publication-ready planning and specification document (Markdown) in the workspace. No code editing or model training in this phase.

## Acceptance Criteria
- Save design document as E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md.
- Contains complete mathematical equations, layer-by-layer YAML structure specifications, ASCII/Mermaid flowcharts, and 8-stage ablation matrix.
