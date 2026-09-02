# Task Assignment: Architecture Specification, Budget & Ablation Roadmap

## Objective
Design the complete architectural blueprint (C3k2Ctrl, SPPF-LSKA, CARAFE, HWDown, SegmentCitrusLite), calculate exact layer-by-layer parameter and GFLOPs budget for Nano scale, and define the 8-stage ablation matrix & experimental validation gates.

## Inputs
- Request: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/ORIGINAL_REQUEST.md`
- Project Plan: `E:/mastercode/1_SEVER/code/ultralytics-main-new/PROJECT.md`

## Instructions
1. Specify the concrete internal structure of `C3k2Ctrl` / `ObserverBlock`:
   - Exact sub-branches, channel splits/expansions, activation functions, residual scaling $\gamma$,
   - Strategy for 100% official YOLO11 weight key compatibility (e.g. standard C3k2 convolutional weights cleanly mapping to primary feedforward path, with feedback/observer weights zero-initialized or structured as additive residual deltas).
2. Detail integration of proven components:
   - SPPF-LSKA: Large Separable Kernel Attention with strip pooling (7/11/21 kernels) replacing standard $5\times 5$ maxpool in SPPF,
   - CARAFE: Content-Aware ReAssembly of FEatures upsampler in Neck,
   - HWDown: Haar Wavelet 2D Downsampling (LL, LH, HL, HH subbands) for anti-aliased, lossless high-frequency spatial downsampling,
   - SegmentCitrusLite: Compact decoupled segmentation head with lightweight prototype mask generation and depthwise separable convolutions.
3. Strict Complexity Budget & Hardware Constraints:
   - Provide exact layer-by-layer tensor shape, parameter count, and GFLOPs calculation for $640\times 640$ input,
   - Verify: Total Params <= 3.20 M (Nano scale), Total GFLOPs <= 11.5 G, GPU latency <= 1.20x YOLO11n-seg.
4. Experimental Protocol:
   - Formulate the 8-model ablation matrix: G00 (Baseline) -> G01 (Control Backbone only) -> G02 (Observer Feedback only) -> G03 (PID Tri-Branch only) -> G04 (Control + LSKA) -> G05 (Control + LSKA + CARAFE) -> G06 (Control + LSKA + CARAFE + HWDown) -> G07 (Full Proposed with SegmentCitrusLite).
   - 4 Pre-experiment gates: Dry-run YAML build, GPU speed gate, 3-epoch smoke test, 50-epoch screening gate.
   - Challenge metrics: mAP50, mAP50-95, AP-tiny, Solidity deficit, Split/Merge error quantification.
5. Provide a detailed report at `.agents/explorer_arch_1/handoff.md`.
