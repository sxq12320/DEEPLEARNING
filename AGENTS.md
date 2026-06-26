# Repository Guidelines

## Project Structure & Module Organization

This repository is a monorepo of independent computer-vision research projects. There is no top-level Python package or shared build step. The main codebase is `ultralytics-main-new/`, a customized Ultralytics YOLO11 fork for RGB-D detection and segmentation experiments. Tests for that fork live in `ultralytics-main-new/tests/`, model configs in `ultralytics-main-new/mine_yaml*/`, and dataset YAMLs at the fork root.

Other subprojects are standalone: `1.coding/0_segment/` contains a registry-based PyTorch segmentation/detection framework, `1.coding/1_study_module/` contains classic network reproductions, `1.coding/2_Unet/` contains a U-Net training script, and `2_catoon/` contains Manim animation scenes. Local datasets are under `data/` and are not expected to be reproducible from Git alone.

## Build, Test, and Development Commands

Always `cd` into the relevant subproject before running commands.

```bash
cd ultralytics-main-new && pip install -e .
pytest tests
yolo segment train model=mine_yaml/11_ours_final_complete.yaml data=206_Apple_Amodal.yaml epochs=300 imgsz=640
```

Use editable install for the YOLO fork so `from ultralytics import YOLO` resolves to this repository. The numbered scripts at fork root (`006_Apple_Amodal_test.py`, `013_train_improved_v2.py`, etc.) are the real training entry points — they call `YOLO(<yaml>).train(...)` with paths baked in. Equivalent CLI form shown above.

For the custom framework:

```bash
cd 1.coding/0_segment
pip install -r requirements.txt
python train.py --model-type fpnseg --image-dir <imgs> --mask-dir <masks> --label-type mask
```

**Gotcha:** if `--image-dir`/`--mask-dir` don't exist, the 0_segment framework silently falls back to synthetic random tensors instead of erroring — check paths if metrics look meaningless.

For animations, run `manim -pql <file>.py <SceneClass>` inside `2_catoon/`.

## Coding Style & Naming Conventions

Use Python with 4-space indentation and keep lines near the configured 120-column limit. The YOLO fork uses settings from `pyproject.toml`: Ruff, isort, YAPF, Google-style docstrings, and pytest. Keep experiment scripts and YAMLs descriptively named, following existing numeric prefixes such as `013_train_improved_v2.py` and `V4-06_final_complete.yaml`.

When adding YOLO custom modules, wire them across 4 files:
1. Implement `nn.Module` in `ultralytics/nn/modules/custom_blocks.py` (or a sibling file)
2. Export it in `ultralytics/nn/modules/__init__.py` (import + `__all__`)
3. Import it in `ultralytics/nn/tasks.py` (top-of-file imports)
4. Register it in `parse_model()` in `ultralytics/nn/tasks.py` (add to `base_modules` frozenset, or add a dedicated `elif` branch for non-standard channel math)

Forgetting step 3 or 4 produces a YAML-parse error rather than a clear "unknown module" message.

## Testing Guidelines

Run focused pytest targets when possible, for example `pytest tests/test_cli.py`. For training changes, add a small smoke run or document the exact command, dataset YAML, image size, epoch count, and device used. Check hardcoded Windows paths before trusting metrics.

## RGB-D Data Convention

Dataset YAMLs (`206_Apple_Amodal.yaml`, etc.) set `channels: 4`. The dataloader reads images with `cv2.IMREAD_UNCHANGED` to preserve the 4th (Depth) channel. Channel order is `[B, G, R, Depth]`. For `.npy` sources, arrays load directly without re-caching.

## Commit & Pull Request Guidelines

Recent commits use short, informal messages. Prefer concise imperative messages with a scope, such as `ultralytics: fix RGB-D dataloader caching`. Pull requests should identify the affected subproject, summarize model/config changes, include commands run, and attach metrics or screenshots for visual outputs.

## Security & Configuration Tips

Do not commit datasets, pretrained weights, `runs/`, `results/`, archives, or videos. Many scripts and YAML files contain local `E:\mastercode\...` paths; update them carefully when moving environments and avoid overwriting local dataset configs without checking the current user setup.

## CI Workflow

`.github/workflows/main.yml` triggers `scripts/update_readme.py` when an Issue labeled `blog` is created/edited, appending its title+link to `README.md` and auto-committing. Don't hand-edit the auto-generated blog-list section of `README.md`.