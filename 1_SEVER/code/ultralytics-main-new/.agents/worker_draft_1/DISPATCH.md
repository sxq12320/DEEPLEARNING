# Task Assignment: Publication-Grade Citrus Control Backbone Design Document Drafting

## Objective
Draft the complete, authoritative, publication-grade network architecture design and planning document:
`E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`

## Inputs
- Original Request: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/ORIGINAL_REQUEST.md`
- Project Plan: `E:/mastercode/1_SEVER/code/ultralytics-main-new/PROJECT.md`
- Codebase Mining Report: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/miner_codebase_1/handoff.md`
- Control Theory Mathematical Report: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/explorer_control_1/handoff.md`
- Architectural Blueprint & Budget Report: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/explorer_arch_1/handoff.md`

## Detailed Structure & Contents Required in `20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`
The document must be written in professional, academic, publication-grade Markdown (bilingual Chinese/English terminology where standard, e.g., 自动控制理论驱动的柑橘幼果实例分割网络规划方案), structured across 6 comprehensive sections:

1. **Title, Executive Abstract & Problem Definition**:
   - High-level executive summary, background on citrus orchard precision harvesting & phenotyping.
   - Comprehensive physical & mathematical failure mode analysis under green-on-green foliage camouflage ($\mathcal{F}_{contrast} \to 0$), specular solar glare ($I_{\text{glare}} \gg I_{\text{diffuse}}$), and branch strip occlusions ($w \in [2,8]\text{ px}$).
2. **Mathematical & Control-Theory Grounding (R1)**:
   - Complete continuous & discrete state-space formulation: plant dynamics $\mathbf{s}_l = \mathbf{A}_l \mathbf{s}_{l-1} + \mathbf{B}_l \mathbf{u}_l + \mathbf{w}_l$, measurement $\mathbf{y}_l = \mathbf{C}_l \mathbf{s}_l + \mathbf{v}_l$.
   - Luenberger State Observer equations $\hat{\mathbf{s}}_l$, innovation/error signal $\mathbf{e}_l = \mathbf{r}_l - \hat{\mathbf{s}}_l$, feedback gain $\mathbf{L}_l$.
   - Rigorous mathematical proofs: Theorem 1 (Asymptotic Error Convergence under Hurwitz condition) and Theorem 2 (Lyapunov Ultimate Boundedness inside compact invariant ball $\mathcal{B}_\epsilon$ via CARE and Discrete Stein Equation $P - A_{obs}^T P A_{obs} = Q$).
   - Tri-Branch PID-inspired dynamic regulator ($P$ spatial detail proportional filter, $I$ accumulated historical semantics with global contextual integrator, $D$ boundary gradient differential Laplacian filter).
   - Frequency-domain transfer functions $G_{PID}(s)$, Routh-Hurwitz stability criterion $K_i < \frac{K_p (1 + K_0 K_d)}{\tau}$, Tustin bilinear transformation $G_{PID}(z)$, and 2D spatial convolution kernel stencils ($\mathcal{K}_P, \mathcal{K}_I, \mathcal{K}_D$).
   - Convex adaptive gain scheduling $\alpha + \beta + \gamma = 1.0$ and LayerScale $\gamma \odot \tanh(u)$.
3. **End-to-End Architectural Specification (R2)**:
   - Detailed internal mechanics of `C3k2Ctrl` / `ObserverBlock` (sub-branches, channel splits, activations, residual scaling).
   - 100% official YOLO11 pretrained weight key compatibility table and zero-initialization strategy ($\gamma_{init} = 0$, $\mathbf{W}_{obs, pw} = 0$, $\mathbf{W}_{d, pw} = 0$).
   - Harmonious integration of verified winners:
     * SPPF-LSKA (7/11/21 large separable strip pooling kernels for anisotropic foliage/branches).
     * CARAFE (Content-Aware ReAssembly of FEatures $5\times 5$ upsampler in Neck).
     * HWDown (2D Haar Wavelet Downsampling for anti-aliased, lossless subband preservation).
     * SegmentCitrusLite (streamlined decoupled head with depthwise separable classification and training-only P2 auxiliary supervision `CitrusTrainAux`).
   - Complete layer-by-layer YAML blueprint (layers 0-23 with exact module names, parameters, channels, repeat counts).
   - Rich ASCII diagrams and Mermaid flowchart diagrams illustrating closed-loop routing and full data flow.
4. **Strict Complexity Budget & Hardware Constraints (R3)**:
   - Layer-by-layer tensor shapes ($C \times H \times W$), parameter counts, and GFLOPs at $640\times 640$ resolution.
   - Nano scale budget compliance table: Total Parameters = 3.021 M (<= 3.20 M cap), GFLOPs = 9.88 G (<= 11.5 G cap), GPU Latency = 1.12x (<= 1.20x YOLO11n-seg baseline).
   - Zero-redundancy analysis: verification that no unverified heavy multi-head attention or transformer blocks are added.
5. **Complete 8-Model Ablation Protocol & Experimental Roadmap (R4)**:
   - Exhaustive 8-model ablation matrix (G00 to G07) with exact configurations, parameter/FLOPs targets, and isolated research questions.
   - 4 Pre-experiment validation gates (Gate 1: Dry-Run YAML Build, Gate 2: GPU Speed & VRAM Gate, Gate 3: 3-Epoch Smoke Convergence, Gate 4: 50-Epoch Screening Gate against G00 baseline).
   - Target challenge metrics: Box & Mask mAP50, mAP50-95, AP-tiny ($S < 256\text{ px}^2$), Solidity Deficit ($\Delta \text{Solidity}$), Split Error Rate ($E_{split}$), Merge Error Rate ($E_{merge}$).
6. **Implementation Guidelines & Engineering Recommendations**:
   - PyTorch module design guidelines, forward pass execution order, loss formulation (auxiliary loss weighting $\lambda_{aux} = 0.5 \to 0.05$).
   - Verification procedures and acceptance criteria checklist.

## Handoff
Write the complete deliverable to `E:/mastercode/1_SEVER/code/ultralytics-main-new/20260902_CITRUS_CONTROL_BACKBONE_DESIGN.md`.
Write your handoff report to `.agents/worker_draft_1/handoff.md`.
