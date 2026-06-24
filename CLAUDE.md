# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a **monorepo of independent deep-learning CV sub-projects**, not a single installable package. There is **no top-level build, lint, test, or dependency setup** — each sub-project has its own entry point, dependencies, and conventions. Always `cd` into the relevant sub-project before running anything.

Two cross-cutting facts that will bite you:
- **Scripts and dataset YAMLs use hardcoded Windows absolute paths** (`E:\mastercode\...`). The numbered training scripts and `*.yaml` data configs in `ultralytics-main-new/`, and the data path in `1.coding/2_Unet/train.py`, must be edited if the repo moves or runs on another machine.
- **Weights and data are untracked.** `.gitignore` excludes `*.pt`, `*.pth`, `/data/`, `runs/`, `weights/`, archives, and videos. Datasets live under `data/` locally only; don't assume they're reproducible from git.

## Sub-project map

| Path | What it is | How to run |
|------|-----------|-----------|
| `ultralytics-main-new/` | **Customized Ultralytics YOLO11 fork** for RGB-D apple-occlusion detection (main research) | `pip install -e .` then run numbered scripts or `yolo` CLI — see below |
| `1.coding/0_segment/` | **Custom registry + config-driven** segmentation/detection framework | `python train.py --model-type fpnseg ...` |
| `1.coding/1_study_module/` | Classic architecture reproductions (LeNet → ConvNeXt/LSNet), 16 self-contained numbered dirs | `cd <arch> && python main.py` or `python <Arch>.py` (independent, no shared runner) |
| `1.coding/2_Unet/` | U-Net on Pascal VOC2007 | `python train.py` (VOC path hardcoded in script) |
| `1.coding/3_phics_x/` | PHICS-X model **library** on ResNet18 base — imported, not run directly | — |
| `1.coding/4_real-esrgan/` | Standalone Real-ESRGAN super-resolution (own README) | see its README |
| `2_catoon/` | Manim NN-teaching animations | `manim -pql <file>.py <SceneClass>` |
| `3_研究生/` | Master's-thesis research archive (mushroom cutting-point & watermelon pollination) — **read-only markdown, no code** | — |
| `data/` | Datasets (MNIST, CIFAR-10, VOC, Apple_RGB_D_Amodal, shr_watermelon, …) + `json2yolo_pose.py`, `hebing.py` converters | — |
| `test.ipynb` | Notebook: YOLO-TXT → COCO-JSON conversion for the watermelon dataset | — |

Root `.pt` files (`yolo11n-pose.pt`, `yolo11n-seg.pt`, `yolo11s-pose.pt`, `yolo26n.pt`) are pretrained weights used by `ultralytics-main-new/` scripts for transfer learning / baselines.

## `ultralytics-main-new/` — the YOLO11 RGB-D fork

A fork of Ultralytics customized for **4-channel RGB-D** apple-occlusion detection using **pure CNN + frequency-domain + dynamic gating** (deliberately no Transformer/Mamba). Standard Ultralytics internals are unchanged; focus on the custom additions below.

**Install editable first.** `cd ultralytics-main-new && pip install -e .` so that `from ultralytics import YOLO` resolves to *this* fork. If a pip-installed `ultralytics` shadows it, the custom modules won't be found by the YAML parser.

### Custom modules and how they're wired in (multi-file mechanism)

The custom building blocks live in `ultralytics/nn/modules/`:
- `custom_blocks.py` — **SFM** (Strip-Freq Mixer, replaces C3k2/C2f), **WCAF** (Wavelet-Cross-Attention Fusion, replaces neck Concat), **DGFFN** (Dilated-Gated FFN), plus Haar DWT/IDWT helpers
- `mobilenetv3_rgb.py`, `mobilenetv4_rgb.py` — RGB-branch backbones
- `starnet_depth.py`, `shufflenetv2_depth.py` — Depth-branch backbones
- `scale_aware_fusion.py` — `ScaleAwareFusion_Depth2RGB` (P3), `_Bidirectional` (P4), `_RGBLed` (P5)

**To add or trace a custom module you must follow it across 4 files** (this is the key thing to understand here):
1. Implement the `nn.Module` in `ultralytics/nn/modules/custom_blocks.py` (or a sibling file)
2. Export it in `ultralytics/nn/modules/__init__.py` (import + add to `__all__`)
3. Import it in `ultralytics/nn/tasks.py` (top-of-file imports, ~line 100)
4. Register it inside `parse_model()` in `ultralytics/nn/tasks.py`: add to the `base_modules` frozenset (and `repeat_modules` if it takes a repeats arg). Modules with non-standard channel math need a dedicated `elif m is ...:` branch — **WCAF** is the worked example (it reads `ch[f[0]]`/`ch[f[1]]` for the RGB/Depth inputs).

Forgetting step 3 or 4 produces a YAML-parse error rather than a clear "unknown module" message.

### Experiment configs

Model-architecture YAMLs are the ablation/comparison matrix:
- `mine_yaml/` — `01_baseline_rgb_only` … `11_ours_final_complete` (the documented ablation ladder), plus older `ablation*_*.yaml` variants
- `mine_yaml_v4/` — `V4-01` … `V4-10`, the current iteration

In these YAMLs the dual-stream backbone and `ScaleAwareFusion_*` fusion layers appear as normal `[from, repeats, Module, [args]]` entries; `from` is a **list** (e.g. `[[2, 6], 1, ScaleAwareFusion_Depth2RGB, [128]]`) to pull from both RGB and Depth streams.

### RGB-D data convention

Dataset YAMLs (`205_*`–`208_*.yaml`, e.g. `206_Apple_Amodal.yaml`) set **`channels: 4`**. The dataloader (`ultralytics/data/base.py`) then reads images with `cv2.IMREAD_UNCHANGED` and, for `.npy` sources, loads the array directly **without re-caching** to preserve the 4th (Depth) channel. Channel order is `[B, G, R, Depth]`.

### Custom optimizers

`ultralytics/engine/trainer.py` adds optimizers beyond stock Ultralytics — notably **PIDAO** (multi-channel high-order PID) and **MuSGD**/SMCAO (see `007_smcao_v22_vs_adamw.py`). Select via `optimizer="PIDAO"` in `model.train(...)` or `optimizer=PIDAO` on the CLI.

### Running training

The **real entry points are the numbered driver scripts** at the fork root (`006_Apple_Amodal_test.py`, `008_shr_watermelon.py`, `013_train_improved_v2.py`, …), each calling `YOLO(<yaml or .pt>).train(...)` with paths/project/name baked in. Run one with `python 006_Apple_Amodal_test.py`. Equivalent CLI form:

```bash
yolo segment train model=mine_yaml/11_ours_final_complete.yaml data=206_Apple_Amodal.yaml \
  optimizer=PIDAO epochs=300 batch=2 imgsz=640 device=0
yolo segment val   model=results/<run>/weights/best.pt data=206_Apple_Amodal.yaml imgsz=640
```

Outputs go to `ultralytics-main-new/results/<name>/` (`weights/best.pt`, `results.csv`, `args.yaml`).

## `1.coding/0_segment/` — custom modular framework

A PyTorch segmentation/detection framework built around a **registry + config-driven assembly** pattern. Read `1.coding/0_segment/README.md` and `AGENTS.md` for the authoritative details; the architecture spans these files:

- `utils/registry.py` — four registries (`BLOCK_`, `BACKBONE_`, `NECK_`, `HEAD_REGISTRY`) and their `@register_block/_backbone/_neck/_head` decorators (lookups are lowercased)
- `utils/builder.py` — `make_layers()` builds `nn.Sequential` from a **list-style** block config; `build_backbone/neck/build_head()` instantiate from **dict-style** `{"name": ..., "args": {...}}` config
- `configs/config.py` — activation map + architecture presets (e.g. `TS_DUAL_MODEL_CFG`)
- `models/{modules,backbones,necks,heads}.py` — registered components; `models/networks.py` — assembled end-to-end models (`MiniSegNet`, `FPNSegNet`, `YOLO11Detector`, `TSDualSegDetNet`), which are **not** auto-registered
- `engine/losses.py`, `engine/metrics.py` — loss/metric library

**Build flow:** config selects a component by `name` → registry lookup → class instantiated with `args`. To add a component, decorate its class with the matching `@register_*` in the right `models/*.py` file; to add a whole model, assemble it in `networks.py` and export from `models/__init__.py`.

**Run:**
```bash
cd 1.coding/0_segment && pip install -r requirements.txt
python train.py --model-type fpnseg --image-dir <imgs> --mask-dir <masks> --label-type mask --epochs 50
python train.py --model-type ts_dual --image-dir <rgb> --mask-dir <mask> --depth-dir <depth> --prompt-dir <prior>
python train.py --cfg configs/train.json --print-cfg   # config file + CLI override; CLI wins
```
- `--model-type`: `miniseg` | `fpnseg` | `ts_dual`. `--label-type`: `mask` | `txt` | `json` | `npy`.
- **Gotcha:** if `--image-dir`/`--mask-dir` don't exist, the dataset **silently falls back to synthetic random tensors** instead of erroring — check your paths if metrics look meaningless.
- Outputs: `checkpoints/results/<name>/{weights/{best,last}.pt, logs.txt, loss_curve.png}`.

## Other sub-projects

- **`1.coding/1_study_module/`** — each architecture is a standalone folder trained independently on CIFAR-10/MNIST (auto-downloaded to a local `data/`). No shared trainer; some share `1.coding/modules.py` and `1.coding/transformer.py` at the `1.coding/` level. Run the folder's `main.py` or `<Arch>.py`.
- **`1.coding/2_Unet/`** — `python train.py`; data path (`data/VOC/.../VOC2007`) is hardcoded, weights save to `params/unet.pth`, visualizations to `train_images/`.
- **`2_catoon/`** — Manim scenes under `0_Learning/` and `1_LeNet/`; render with `manim -pql <file>.py <SceneClass>` (`-q l/m/h` quality). No custom `manim.cfg`.

## CI: issue-driven README blog list

`.github/workflows/main.yml` triggers `scripts/update_readme.py` when an Issue labeled **`blog`** is created/edited, appending its title+link to `README.md` and auto-committing as `github-actions[bot]`. **Don't hand-edit the auto-generated blog-list section** of `README.md` — it's machine-managed.
