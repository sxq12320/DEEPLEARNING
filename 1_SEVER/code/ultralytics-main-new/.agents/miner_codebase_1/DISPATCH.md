# Task Assignment: Codebase & YOLO11 Architecture Mining

## Objective
Investigate the `ultralytics-main-new` codebase to extract exact architectural templates, module definitions, YAML syntax, weight key conventions, and existing implementations.

## Inputs
- Project root: `E:/mastercode/1_SEVER/code/ultralytics-main-new`
- Request: `E:/mastercode/1_SEVER/code/ultralytics-main-new/.agents/ORIGINAL_REQUEST.md`
- Project Plan: `E:/mastercode/1_SEVER/code/ultralytics-main-new/PROJECT.md`

## Instructions
1. Inspect YOLO11 segmentation configuration (`yolo11n-seg.yaml` or similar in `ultralytics/cfg/models/11/`).
2. Examine `C3k2`, `C2f`, `Bottleneck`, `SPPF`, `Conv`, `Segment` head in `ultralytics/nn/modules/`.
3. Check for any existing custom modules or references (e.g., LSKA, CARAFE, HWDown, Haar wavelet, custom heads) in the repository.
4. Analyze how `parse_model` in `ultralytics/nn/tasks.py` constructs layers from YAML, how channels are scaled (`scales: {n: ...}`), and how pretrained weight loading maps keys.
5. Provide a detailed report at `.agents/miner_codebase_1/handoff.md`.
