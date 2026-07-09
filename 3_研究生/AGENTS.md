# Repository Guidelines

## Project Structure & Module Organization

This repository is a monorepo of independent computer-vision research projects. There is no top-level Python package or shared build step. The main codebase is `ultralytics-main-new/`, a customized Ultralytics YOLO11 fork for RGB-D detection and segmentation. Tests live in `ultralytics-main-new/tests/`, model configs in `ultralytics-main-new/mine_yaml*/`, and dataset YAMLs at the fork root.

Other projects include `1.coding/0_segment/` for a PyTorch segmentation/detection framework, `1.coding/1_study_module/` for classic network reproductions, `1.coding/2_Unet/` for U-Net training, and `2_catoon/` for Manim scenes. Local datasets under `data/` are environment-specific.

## Build, Test, and Development Commands

Always `cd` into the relevant subproject before running commands.

```bash
cd ultralytics-main-new && pip install -e .
pytest tests
yolo segment train model=mine_yaml/11_ours_final_complete.yaml data=206_Apple_Amodal.yaml epochs=300 imgsz=640
```

Use editable install so `from ultralytics import YOLO` resolves to this fork. Numbered scripts at the YOLO fork root, such as `006_Apple_Amodal_test.py`, are real training entry points.

For `1.coding/0_segment/`:

```bash
pip install -r requirements.txt
python train.py --model-type fpnseg --image-dir <imgs> --mask-dir <masks> --label-type mask
```

For animations, run `manim -pql <file>.py <SceneClass>` inside `2_catoon/`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep lines near 120 columns. The YOLO fork uses Ruff, isort, YAPF, Google-style docstrings, and pytest settings from `pyproject.toml`. Keep experiment scripts and YAMLs descriptive, following numeric prefixes like `013_train_improved_v2.py`.

When adding YOLO custom modules, implement the module, export it in `ultralytics/nn/modules/__init__.py`, import it in `ultralytics/nn/tasks.py`, and register it in `parse_model()`.

## Testing Guidelines

Prefer focused pytest targets, for example `pytest tests/test_cli.py`. For training changes, add a smoke run or document the command, dataset YAML, image size, epoch count, and device. Verify hardcoded Windows paths before trusting metrics. In `0_segment`, invalid image or mask paths silently fall back to synthetic tensors.

## RGB-D Data Convention

Dataset YAMLs such as `206_Apple_Amodal.yaml` set `channels: 4`. The dataloader uses `cv2.IMREAD_UNCHANGED` to preserve depth. Channel order is `[B, G, R, Depth]`; `.npy` arrays load directly without re-caching.

## Commit & Pull Request Guidelines

Use concise imperative commit messages with scope, such as `ultralytics: fix RGB-D dataloader caching`. Pull requests should name the affected subproject, summarize model/config changes, list commands run, and include metrics or screenshots when visual output changes.

## Security & Configuration Tips

Do not commit datasets, pretrained weights, `runs/`, `results/`, archives, or videos. Many scripts and YAMLs contain local `E:\mastercode\...` paths; update them carefully and avoid overwriting local dataset configs without checking the current setup.
