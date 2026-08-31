## 2026-08-27T07:13:08Z
You are the Repository & Engineering Verification Lead (teamwork_preview_explorer).

MANDATORY FIRST STEP:
Read the authoritative user request at:
E:\mastercode\.agents\ORIGINAL_REQUEST.md
Also read repository rules in:
E:\mastercode\AGENTS.md

Your working directory is:
E:\mastercode\.agents\explorer_repo_1\

Task:
Perform open-source repository auditing, module feasibility analysis, and concrete architecture formulation:
1. Audit >=10 official / reputable open-source GitHub repositories for candidate modules (e.g., StarNet, MobileNetV4, RepNCSPELAN, PointRend, BiFPN, Dynamic Snake Conv, EMA, DCNv3/v4, Boundary IoU, etc.):
   - Record Repo URL, Organization/Author, Star count, License, PyTorch implementation quality, CUDA dependency status.
   - Verify compatibility with Ultralytics YOLO11 (pure PyTorch vs custom CUDA C++ extensions; strict rejection of Mamba/selective scan or non-deployable ops).
2. Calculate exact parameter and FLOP budgets:
   - Hard constraints: Params <= 2.85M, GFLOPs <= 10.0G (at 640x640), CPU latency <= 150ms, GPU latency <= 8ms.
   - Derive channel scaling, repeat counts, and layer connections for:
     * Candidate A (Conservative Pruning / High-efficiency baseline)
     * Candidate B (⭐ Recommended Primary Method: CitrusB-Seg)
     * Candidate C (Aggressive Dual-Stream / Boundary-Enhanced exploration)
3. Detail the complete architecture specification for CitrusB-Seg:
   - Backbone, Neck, Head design
   - Full YAML configuration draft matching Ultralytics format
   - Channel dimensions, stride progression, receptive field analysis, FLOP/parameter breakdown table.
4. Write your detailed findings, repository matrix, and architecture specs to E:\mastercode\.agents\explorer_repo_1\handoff.md and send a completion message.
