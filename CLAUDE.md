# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a **monorepo of independent deep-learning CV sub-projects**, not a single installable package. There is **no top-level build, lint, test, or dependency setup** — each sub-project has its own entry point, dependencies, and conventions. **Always `cd` into the relevant sub-project before running anything.**

Two cross-cutting facts that will bite you:
- **Scripts and dataset YAMLs use hardcoded Windows absolute paths** (`E:\mastercode\...`). The training driver scripts and `*.yaml` data configs in `ultralytics-main-new/`, and the data path in `1.coding/2_Unet/train.py`, must be edited if the repo moves or runs on another machine.
- **Weights and data are untracked.** `.gitignore` excludes `*.pt`, `*.pth`, `/data/`, `runs/`, `results/`, `weights/`, archives, and videos. Datasets (MNIST, CIFAR-10, VOC, Apple_RGB_D_Amodal, shr_watermelon, jeruk_split, orange_wuxi, caomei, …) live under `data/` locally only and are not expected to be reproducible from git alone.

## Sub-project map

| Path | What it is | How to run |
|------|-----------|-----------|
| `ultralytics-main-new/` | **Customized Ultralytics YOLO11 fork** — RGB-D apple-occlusion research **plus** a stock-YOLO orange/citrus seg line (the current active work) | `pip install -e .`, then run a `train_orange_wuxi_*` script or the `yolo` CLI — see below |
| `1.coding/0_segment/` | **Custom registry + config-driven** segmentation/detection framework (PyTorch) | `python train.py --model-type fpnseg ...` |
| `1.coding/1_study_module/` | Classic architecture reproductions (LeNet → ConvNeXt/LSNet), 16 self-contained numbered dirs | `cd <arch> && python main.py` / `python <Arch>.py` — but 4 dirs are notebook-only (see below) |
| `1.coding/2_Unet/` | U-Net on Pascal VOC2007 | `python train.py` (VOC path hardcoded in script) |
| `1.coding/3_phics_x/` | PHICS-X model **library** on ResNet18 base — imported, not run directly | — |
| `1.coding/4_real-esrgan/` | Standalone Real-ESRGAN super-resolution — **vendored `xinntao/Real-ESRGAN` v0.3.0 as a nested git repo** (hence the single `??` entry in the parent's `git status`) | see below / its README |
| `2_catoon/` | Manim NN-teaching animations (`0_Learning/`, `1_LeNet/`) | `manim -pql <file>.py <SceneClass>` |
| `3_研究生/` | Master's-thesis research archive — citrus-bagging robot, mushroom cutting-point, watermelon pollination, tomato picking, bolt/nut recycling. Mostly markdown/HTML reports + flowchart & video-frame assets, plus some Python (`sam_amodal_pear.py`, `sd_diffusers/`) | — |
| `data/` | Datasets + `json2yolo_pose.py`, `hebing.py` format converters | — |
| `test.ipynb` | Notebook: YOLO-TXT → COCO-JSON conversion for the watermelon dataset | — |

Root `.pt` files (`yolo11n-pose.pt`, `yolo11n-seg.pt`, `yolo11s-pose.pt`, `yolo26n.pt`) are pretrained weights used by `ultralytics-main-new/` scripts for transfer learning / baselines. The fork root (`ultralytics-main-new/`) additionally keeps its own base weights — `yolo11n-seg.pt`, `yolo26n.pt`, and `yolo26n-seg.pt` (the last is **fork-only**, not at repo root) — used by the orange/citrus training scripts.

## `ultralytics-main-new/` — the YOLO11 RGB-D fork

A fork of Ultralytics customized for **4-channel RGB-D** apple-occlusion detection using **pure CNN + frequency-domain + dynamic gating** (deliberately no Transformer/Mamba). Standard Ultralytics internals are unchanged; focus on the custom additions below. The fork also hosts a **separate, plain 3-channel orange/citrus segmentation workflow** (its own subsection) that uses stock Ultralytics — keep the two lines distinct.

**Install editable first.** `cd ultralytics-main-new && pip install -e .` so that `from ultralytics import YOLO` resolves to *this* fork. If a pip-installed `ultralytics` shadows it, the custom modules won't be found by the YAML parser. After install: `pytest tests` to run the fork's test suite (e.g., `pytest tests/test_cli.py` for a focused target).

### Custom modules and how they're wired in (multi-file mechanism)

The custom building blocks live in `ultralytics/nn/modules/`:
- `custom_blocks.py` — **SFM** (Strip-Freq Mixer, replaces C3k2/C2f), **WCAF** (Wavelet-Cross-Attention Fusion, replaces neck Concat), **DGFFN** (Dilated-Gated FFN), plus Haar DWT/IDWT helpers
- `mobilenetv3_rgb.py`, `mobilenetv4_rgb.py` — RGB-branch backbones
- `starnet_depth.py`, `shufflenetv2_depth.py` — Depth-branch backbones
- `scale_aware_fusion.py` — `ScaleAwareFusion_Depth2RGB` (P3), `_Bidirectional` (P4), `_RGBLed` (P5)
- `ct_modules.py` — cross-task fusion modules used by some V4 ablations
- `rgbd_fusion_neck.py` — `RGBDFusionNeck` + P3/P4/P5 fusion modules; **exported in `__init__.py` but NOT imported/registered in `tasks.py`**, so it is currently unreachable from model YAMLs (experimental/unwired — complete the 4 steps below before using it)

**To add or trace a custom module you must follow it across 4 files** (this is the key thing to understand here):
1. Implement the `nn.Module` in `ultralytics/nn/modules/custom_blocks.py` (or a sibling file)
2. Export it in `ultralytics/nn/modules/__init__.py` (import + add to `__all__`)
3. Import it in `ultralytics/nn/tasks.py` (top-of-file imports, ~line 100)
4. Register it inside `parse_model()` in `ultralytics/nn/tasks.py`: add to the `base_modules` frozenset (and `repeat_modules` if it takes a repeats arg). Modules with non-standard channel math need a dedicated `elif m is ...:` branch — **WCAF** is the worked example (it reads `ch[f[0]]`/`ch[f[1]]` for the RGB/Depth inputs).

Forgetting step 3 or 4 produces a YAML-parse error rather than a clear "unknown module" message.

Other forks to know about:
- `ultralytics-main-new/ultralytics/data/base.py` — modified dataloader for `channels: 4` RGB-D (see below).
- `ultralytics-main-new/ultralytics/engine/trainer.py` — registers custom optimizers.
- `ultralytics-main-new/ultralytics/cfg/models/11/yolo11-rgbd.yaml` — stock RGB-D model config used by the CLI example.

### Experiment configs

Model-architecture YAMLs are the ablation/comparison matrix:
- `mine_yaml/` — `01_baseline_rgb_only.yaml` … `11_ours_final_complete.yaml` (the documented ablation ladder), plus older `ablation*_*.yaml` variants
- `mine_yaml_v4/` — `V4-01_baseline_rgb_only.yaml` … `V4-10_p5_rgbled_ablation.yaml`, the current iteration
- `gpt_yaml_yolo/` — separate folder of YAML variants (experimental)

In these YAMLs the dual-stream backbone and `ScaleAwareFusion_*` fusion layers appear as normal `[from, repeats, Module, [args]]` entries; `from` is a **list** (e.g. `[[2, 6], 1, ScaleAwareFusion_Depth2RGB, [128]]`) to pull from both RGB and Depth streams.

### RGB-D data convention

The RGB-D research targets **`channels: 4`**, but only `206_Apple_Amodal.yaml` actually sets it — the other root dataset YAMLs are standard 3-channel RGB and **omit the key**: `205_jeurk_spilt_data.yaml` (jeruk detection), `207_shr_watermelon.yaml` / `207_shr_watermelon_6pt.yaml` (watermelon pose), `208_shr_watermelon_seg.yaml` (watermelon seg). For a `channels: 4` config the dataloader (`ultralytics/data/base.py`) reads images with `cv2.IMREAD_UNCHANGED` and, for `.npy` sources, loads the array directly **without re-caching** to preserve the 4th (Depth) channel; channel order is `[B, G, R, Depth]`. The orange config `data/test/orange_wuxi_seg.yaml` is likewise plain 3-channel and is the only seg dataset off the RGB-D path.

### Custom optimizers

`ultralytics/engine/trainer.py` adds optimizers beyond stock Ultralytics — **PIDAO** (multi-channel high-order PID, defined inline in `trainer.py`), **MuSGD** (from `ultralytics/optim/muon.py`), and **SMCAO** (`ultralytics/nn/modules/smcao_v22_scheduler.py`, `SMCAOV22Scheduler`). Select via `optimizer="PIDAO"` in `model.train(...)` or `optimizer=PIDAO` on the CLI.

### Running training — two distinct lines

**1. RGB-D apple/watermelon research (the custom architecture).** The historical driver scripts followed a `NNN_name.py` pattern at the fork root (`006_Apple_Amodal_test.py`, `008_shr_watermelon_seg.py`, `013_train_improved_v2.py`, `015_train_distill_v2.py`, `016_train_watermelon_seg_p2.py`, `98_visualize_compare.py`, `99_compare_models.py`, …), each calling `YOLO(<yaml or .pt>).train(...)` with paths baked in. **These are currently deleted from the working tree** — still in git history, so `git checkout -- <file>` restores them (the deletions are uncommitted). The model YAMLs (`mine_yaml*`), data YAMLs (`205_*`–`208_*`), custom modules, and optimizers they drove all still exist, so the CLI form still works:

```bash
yolo segment train model=mine_yaml/11_ours_final_complete.yaml data=206_Apple_Amodal.yaml \
  optimizer=PIDAO epochs=300 batch=2 imgsz=640 device=0
yolo segment val   model=results/<run>/weights/best.pt data=206_Apple_Amodal.yaml imgsz=640
```

Outputs go to `ultralytics-main-new/results/<name>/` (`weights/best.pt`, `results.csv`, `args.yaml`).

**2. Orange/citrus (Wuxi) seg — the current live entry points.** See the next subsection.

### Orange/Citrus (Wuxi) stock-YOLO segmentation (separate from the RGB-D research)

A self-contained **3-channel RGB, single-class (`nc: 1`, `orange_immature`) instance-segmentation** line on **stock Ultralytics** — plain `yolo11n-seg.pt` / `yolo26n-seg.pt` + `optimizer="AdamW"`, **no SFM/WCAF, no `channels: 4`, no PIDAO**. Outputs land in `1_results/ORANGE_WUXI_SEG/<name>/` (note the `1_results/` prefix and ALL-CAPS subfolder), **not** `results/`. Pipeline (all scripts at the fork root, hardcoded `E:\mastercode` paths):

1. **Convert** — `python convert_orange_wuxi_to_yolo.py --overwrite`: LabelMe polygons → YOLO-seg `.txt`. Reads `data/orange_wuxi/annotions_x/` + `img/` (note the misspelled `annotions_x`; a 2nd batch sits in `annotion_x_2/` + `img_2/`), keeps only `orange_immature`, splits 0.7/0.2/0.1 train/val/test (seed `20260708`), and writes `data/test/{images,labels}/{train,val,test}/` + `orange_wuxi_seg.yaml` + `conversion_report.json`. Refuses a non-empty `--out` without `--overwrite`.
2. **Train** — `python train_orange_wuxi_yolo11n_seg.py` (v1: `yolo11n-seg.pt`); `_v2.py` / `_v3.py` (both load `yolo26n-seg.pt` **despite the `_yolo11n_` filename** — naming lag). All: AdamW, `epochs=300 patience=100 imgsz=640 batch=4 lr0=0.01 amp=0`, seed 42, `data=data/test/orange_wuxi_seg.yaml`, `project=1_results/ORANGE_WUXI_SEG`.
3. **Visualize** — `python make_orange_mask_compare.py`: 3-panel Original | GT | Pred mask overlays (hardcoded to a specific run's `best.pt`).

Run names in `1_results/ORANGE_WUXI_SEG/` form a ladder `000`–`006` (odd = yolo11n, even = yolo26n; `*_test_eval` = test-split evals). The `…imgs` tags (`361`/`597`) are historical snapshots — the dataset has since grown (702 JSONs → 491/140/71), so metrics across runs are **not** directly comparable.

Fork-root helpers for the watermelon-pose work also remain: `keypoint_map_utils.py` (numpy OKS + single-keypoint mAP), `vis_6pt.py`, `vis_compare.py`.

## `1.coding/0_segment/` — custom modular framework

A PyTorch segmentation/detection framework built around a **registry + config-driven assembly** pattern. Read `1.coding/0_segment/README.md` for the authoritative details (it has the full module list and architecture diagrams); the key parts to know:

- `utils/registry.py` — four registries (`BLOCK_REGISTRY`, `BACKBONE_REGISTRY`, `NECK_REGISTRY`, `HEAD_REGISTRY`) and their `@register_block/_backbone/_neck/_head` decorators (lookups are lowercased)
- `utils/builder.py` — `make_layers()` builds `nn.Sequential` from a **list-style** block config; `build_backbone/neck/build_head()` instantiate from **dict-style** `{"name": ..., "args": {...}}` config
- `configs/config.py` — activation map + architecture presets (e.g. `TS_DUAL_MODEL_CFG`, `YOLO11_CONFIGS` with `nano/small/medium` scales)
- `models/{modules,backbones,necks,heads}.py` — registered components
- `models/networks.py` — all assembled end-to-end models: `MiniSegNet`, `FPNSegNet`, `YOLO11Detector`, and `TSDualSegDetNet` (RGB + Mask prior + Depth → segmentation + bounding boxes)
- `engine/{losses,metrics,trainer,evaluator}.py` — loss library (BCE/CE; CIoU/DFL/BCE + TaskAlignedAssigner), IoU/Dice/mAP, training loops
- `datasets/dataset.py` — auto-switches real↔synthetic when paths are missing (silent fallback; see gotcha below)

**Build flow:** config selects a component by `name` → registry lookup → class instantiated with `args`. To add a component, decorate its class with the matching `@register_*` in the right `models/*.py` file; to add a whole model, assemble it in `models/networks.py`.

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

## `1.coding/1_study_module/` — architecture reproduction set

Sixteen self-contained dirs, `1_LeNet/` … `16_LSNET/`, naming maps: 1 LeNet-5 · 2 AlexNet · 3 VGGNet (A–E) · 4 SEBlock · 5 CBAM · 6 MobileNet-V1 · 7 Transformer · 8 ViT · 9 NIN · 10 GoogleNet · 11 ResNet · 12 DenseNet · 13 ConvNeXt · 14 MobileNet-V2 · 15 FCN · 16 LSNet. Most ship a `main.py` (or `<Arch>.py`) and train independently on CIFAR-10/MNIST auto-downloaded to `./data/`; **four dirs are Jupyter-notebook-only** — `8_VIT`, `12_DenseNet`, `13_ConvNEXT`, `14_MobileNet-V2` (open the `.ipynb`; `python main.py` doesn't apply there). There is **no shared trainer** — each dir is standalone. (The stray `1_study_module/transformer.py` is an unrelated `.npy`→16-bit-PNG depth-map converter that nothing imports — not a shared dependency.)

## Other sub-projects

- **`1.coding/2_Unet/`** — `python train.py`; split into `data.py` (VOC loader), `net.py` (U-Net), `utils.py`. Data path (`data/VOC/.../VOC2007`) is hardcoded, weights save to `params/unet.path` (literal `.path` extension, **not** `.pth`), visualizations to `train_images/`.
- **`1.coding/3_phics_x/`** — `models.py` + `modules.py` + `utils.py`; PHICS-X built on a ResNet18 base. Imported as a library, no CLI.
- **`1.coding/4_real-esrgan/`** — vendored Real-ESRGAN (`xinntao` v0.3.0) as a **nested git repo** (own `.git`, so the parent `git status` shows it as one untracked `??` entry — not a submodule). Install with `pip install -r requirements.txt` **then `python setup.py develop`** (builds the local `realesrgan` package). Run `python inference_realesrgan.py` / `inference_realesrgan_video.py`; the local `download_model.py` fetches weights to a hardcoded `weights/` path. `cog_predict.py` is a Cog/Replicate `Predictor` class (run via `cog predict -i img=@...`, **not** `python cog_predict.py`).
- **`2_catoon/`** — Manim scenes under `0_Learning/` and `1_LeNet/`; render with `manim -pql <file>.py <SceneClass>` (`-q l/m/h` quality). No custom `manim.cfg`; `test.py` and `test.html` exist for ad-hoc experiments.
- **`data/`** — datasets plus converters: `hebing.py` (label merging) and `json2yolo_pose.py` (keypoint format conversion). All datasets are git-ignored.

## Style & commit conventions

- Python with 4-space indentation; line length targeted near 120 columns. `pyproject.toml` in the YOLO fork enables Ruff, isort, YAPF, Google-style docstrings, and pytest.
- Numbered scripts and YAMLs follow `NNN_name_vX.py` / `NNN_ablation_topic.yaml` (e.g. `013_train_improved_v2.py`, `V4-06_final_complete.yaml`).
- Recent commits use short, informal Chinese/English messages. Prefer concise imperative with a scope, e.g. `ultralytics: fix RGB-D dataloader caching`.
- Pull requests identify the affected sub-project, summarize model/config changes, list commands run, and attach metrics or screenshots for visual outputs.

## CI: issue-driven README blog list

`.github/workflows/main.yml` runs `.github/scripts/update_readme.py` when an Issue carrying the **`blog`** label is opened/edited/labeled, to append the Issue title+link+date to `README.md` after a `<!-- BLOG_LIST -->` marker and auto-commit as `github-actions[bot]`. **This automation is currently non-functional on two counts:** the workflow's run step (`main.yml:29`) invokes `python scripts/update_readme.py` — the wrong path, since the script lives under `.github/scripts/` — so the job fails as written; and `README.md` has no `<!-- BLOG_LIST -->` marker, so the script would no-op anyway. Fix both (correct the workflow path **and** add the marker) before relying on it; until then there is no machine-managed blog section to avoid hand-editing.
