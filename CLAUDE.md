# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

Deep learning research project focused on **RGB-D multi-modal object detection and segmentation**, using control theory (Kalman filter, ESO, IDAPBC) to improve feature fusion between RGB and depth modalities. The repo contains two main codebases plus learning experiments.

## Repository Structure

| Directory | Purpose |
|-----------|---------|
| `ultralytics-main/` | Modified Ultralytics YOLO framework with custom control-theory fusion modules — the primary research codebase |
| `1.coding/0_segment/` | Modular image segmentation framework skeleton with dynamic network building |
| `1.coding/1_study_module/` | Individual module experiments |
| `2_catoon/` | Learning exercises (LeNet etc.) |
| `data/` | Dataset storage (gitignored) |

## Key Development Commands

### Training (ultralytics-main)

```bash
# Standard YOLO training with custom RGB-D config
yolo detect train data=201_caomei_data.yaml model=cfg/models/11/yolo11-ct-ABC.yaml epochs=100 imgsz=640

# Training with custom optimizer (PIDAO)
python 301_optimizer_PIDAO.py

# Run custom test scripts (numbered by category: 0xx=test, 1xx=data prep, 2xx=config, 3xx=optimizer)
python 000_test.py
python 001_yolo11_test.py
```

### Training (1.coding/0_segment)

```bash
pip install -r requirements.txt
python scripts/train.py --epochs 10 --batch 8
python scripts/predict.py --source path/to/image.jpg --weights runs/train/exp/weights/best.pt
```

### Data Preparation Scripts

```bash
# Convert JSON annotations to YOLO format
python 101_json2yolo.py

# Convert Pascal VOC to YOLO format
python 102_VOC_TO_YOLO.py

# Split Kvasir dataset to YOLO format
python 103_kvasir2yolo.py
```

## Architecture: YOLO11-CT (Control Theory) RGB-D Models

The core research contribution is a series of YOLO11 model variants for RGB-D fusion, each progressively adding control-theory modules at different feature pyramid levels:

| Config | P3 (80×80) | P4 (40×40) | P5 (20×20) |
|--------|-----------|-----------|-----------|
| `yolo11-base-rgbd.yaml` | BypassModule | BypassModule | BypassModule |
| `yolo11-ct-A.yaml` | KalmanGatedFusion | BypassModule | BypassModule |
| `yolo11-ct-AB.yaml` | KalmanGatedFusion | ESOFusion | BypassModule |
| `yolo11-ct-ABC.yaml` | KalmanGatedFusion | ESOFusion | IDAPBCFusion |

### Custom Modules

- **`ultralytics-main/ultralytics/nn/modules/ct_modules.py`** — Control-theory fusion modules:
  - `MultiScaleVarianceEstimator` — multi-scale variance estimation for adaptive fusion gating
  - `KalmanGatedFusion` — Kalman-filter-inspired gated fusion (shallow layer, suppresses depth noise)
  - `ESOFusion` — Extended State Observer fusion (mid layer, compensates modal misalignment)
  - `IDAPBCFusion` — Interconnected Damping & Passive-Based Control fusion (deep layer, energy-optimal)
- **`ultralytics-main/ultralytics/nn/modules/smc_scheduler.py`** — Sliding mode control learning rate scheduler
- **`ultralytics-main/ultralytics/optim/muon.py`** — Custom optimizer
- **`ultralytics-main/ultralytics/rgb_d_dataset.py`** — RGBD dataset class that concatenates RGB + depth into 4-channel input

## Script Numbering Convention (ultralytics-main)

Scripts follow a category-based numbering scheme:
- `0xx` — Model testing and evaluation scripts
- `1xx` — Data preprocessing and format conversion
- `2xx` — Dataset YAML configuration files
- `3xx` — Custom optimizer implementations

## Data Format

Dataset YAML configs (e.g., `201_caomei_data.yaml`) follow YOLO format and point to local image/label directories. Depth maps are stored separately and loaded by `RGBDDataset` at training time — they can be `.npy` or standard image formats (PNG, TIFF). If a depth file is missing, training will raise `FileNotFoundError`.

## Custom Model Configs

All custom YOLO11 variant configs live in `ultralytics-main/ultralytics/cfg/models/11/`. Naming convention:
- `yolo11-base-rgbd.yaml` — RGB-D baseline (bypass fusion only)
- `yolo11-ct-A.yaml`, `yolo11-ct-AB.yaml`, `yolo11-ct-ABC.yaml` — progressive control-theory ablation
- `yolo11-seg.yaml` — segmentation variant
- `yolo11-seg-RGBandD.yaml` — RGB-D segmentation variant
- `0_yolo11-seg-RGBandD.yaml` through `6_yolo11_seg_Dense_C3K2LS_NeckGate.yaml` — progressive architectural ablations
