# Repository Guidelines

## Current Research Priority

This repository supports a master's thesis on vision for citrus bagging. The immediate goal is to publish two connected papers:

1. Lightweight, high-accuracy instance segmentation of immature citrus fruit.
2. Precise citrus peduncle-point localization using the fruit instances/ROIs produced by paper 1.

Keep paper 1 focused on RGB immature-fruit instance segmentation. Do not mix in RGB-D, amodal segmentation, OBB, robotic control, or multi-task pose heads unless a later task explicitly requires them. The current research source of truth is `3_研究生/柑橘套袋视觉_完整研究执行计划.md`.

The current visual-problem framing is not generic "occlusion and small objects." Focus on strip-like leaf/branch occlusion that creates deeply concave visible masks, the topology conflict between preserving one occluded fruit and separating adjacent touching fruits, and extreme within-image scale span. Quantify these with solidity/convex-hull deficits, neighboring-instance gaps, split/merge errors, and per-image scale ratios before claiming a method solves them.

## Project Structure

`ultralytics-main-new/` is the active codebase, a customized Ultralytics fork. Citrus model YAMLs are in `ultralytics-main-new/0_orange_yaml/`, training/evaluation drivers are `train_citrus_seg.py` and `eval_citrus_seg.py`, and experiment artifacts are under `ultralytics-main-new/1_results/`.

The current dataset is `data/test/`: 941 RGB images and 4,576 labeled instances. Treat the existing train/val/test split as preliminary because frames from the same burst sequence cross split boundaries. Formal paper experiments must use a group-aware split.

Other directories are independent legacy or side projects. Do not refactor them while working on citrus experiments.

## Baselines and Experiment Discipline

Use YOLO11n-seg as the current primary ablation baseline because it is nano-scale, already trained locally, and directly comparable with recent citrus literature. Do not limit comparison experiments to YOLO. The minimum cross-family set is YOLOv8n-seg, YOLO11n-seg, YOLO26n-seg, RTMDet-Ins-tiny, Mask R-CNN R50-FPN, and RF-DETR Seg Nano. For the journal-strength comparison, add SOLOv2-Light R18-FPN as a box-free, location-based baseline; it is not the primary ablation model, and its inclusion replaces the optional CondInst/SparseInst slot. Run a 50-epoch YOLO11n versus RTMDet-Ins-tiny screening before considering any primary-baseline switch.

Include `U-Net + marker-controlled watershed` as a semantic-to-instance auxiliary baseline. U-Net alone is not an instance segmentation model: merge instance masks into binary foreground for training, split predictions with a validation-tuned distance-transform watershed, and report both semantic Dice/mIoU and instance Mask AP. DeepLabV3+ or SegFormer-B0 plus the same watershed may be added as one optional semantic comparison. Use a mature `segmentation_models_pytorch` or MMSegmentation implementation; `1.coding/2_Unet/` is legacy learning code and is not paper-ready.

Existing runs `001`-`003` are preliminary. Their results are not interchangeable with new runs unless the data split, initialization, optimizer, learning rate, dropout, image size, seed, and evaluation split are identical. The current scripts contain conflicting protocols; resolve this before formal experiments.

For final tables, report mask mAP50-95, mask mAP50, precision, recall, AP by object scale, Params, GFLOPs, measured latency, and challenge-subset performance. For semantic models also report Dice, mIoU, and Boundary F1. Run screening experiments once, then repeat the primary baseline and final method with three seeds and report mean plus standard deviation.

## Development Commands

Run commands inside the YOLO fork:

```powershell
cd E:\mastercode\ultralytics-main-new
pip install -e .
python train_citrus_seg.py --model yolo11n-seg.pt --name E0_baseline
python eval_citrus_seg.py --weights 1_results\ORANGE_WUXI_SEG\E0_baseline\weights\best.pt
pytest tests
```

Use focused tests and a short 1-3 epoch smoke run before any 300-epoch experiment.

## Coding and Model Integration

Use Python with 4-space indentation and the repository's 120-column limit. Follow Ruff/isort/YAPF and Google-style docstrings.

When adding a YOLO module:

1. Implement it under `ultralytics/nn/modules/`.
2. Export it from `ultralytics/nn/modules/__init__.py`.
3. Import it in `ultralytics/nn/tasks.py`.
4. Register its channel/repeat behavior in `parse_model()`.
5. Add a minimal YAML and test model build, forward, backward, and FLOPs.

Prefer one coherent, task-specific method over stacking published attention, convolution, and upsampling blocks.

## Data, Results, and Git

Do not commit datasets, weights, `runs/`, large result images, archives, or videos. Keep numbered experiment names and never overwrite a completed run. Record the exact command, Git state, dataset split version, hardware, and final metrics for every paper experiment.

Do not revert unrelated user changes. Use concise scoped commits such as `citrus: add cross-family baseline configs`.
