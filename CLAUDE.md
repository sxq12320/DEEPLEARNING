# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a monorepo centered on a **master's thesis on vision for citrus bagging**, plus independent side projects. There is **no top-level build, lint, test, or dependency setup** — each sub-project has its own entry point and dependencies; **always `cd` into the relevant sub-project before running anything**. Repo-wide research rules (priorities, baseline matrix, experiment discipline, git hygiene) live in **`AGENTS.md`** — treat it as authoritative; `README.md` (Chinese) is the narrative overview.

Cross-cutting facts that will bite you:
- **Scripts and dataset YAMLs use hardcoded absolute paths** — `E:\mastercode\...` on this machine, `/data/sxq/...` in server-side code. They must be edited if the repo moves.
- **Weights and data are untracked.** `.gitignore` excludes `*.pt`/`*.pth`/`*.onnx`, `/data/`, `runs/`, archives, and videos. Datasets live under `data/` locally only.
- **The YOLO fork exists in two copies**: `ultralytics-main-new/` (local dev tree) and `1_SEVER/code/ultralytics-main-new/` (server code mirror — **newer**, holds the F-series configs). See the 1_SEVER section before touching either. (`1.coding/` and `9_archive/` mentioned by older docs no longer exist.)

## Current research priority (short form of AGENTS.md)

Two connected papers: (1) **lightweight, high-accuracy RGB instance segmentation of immature citrus** (the active work), (2) peduncle-point localization using paper 1's ROIs. Paper 1 stays strictly RGB single-class (`orange_immature`) — **no RGB-D, amodal, OBB, or pose mixing**. The research source of truth is `3_研究生/柑橘套袋视觉_完整研究执行计划.md`. The legacy RGB-D apple-occlusion line (`channels: 4`, SFM/WCAF, `206_Apple_Amodal.yaml`) still lives in the fork but is dormant.

## Sub-project map

| Path | What it is | How to run |
|------|-----------|-----------|
| `ultralytics-main-new/` | **Customized Ultralytics fork — active citrus seg code** | `pip install -e .`, then `train_citrus_seg.py` / `eval_citrus_seg.py` (below) |
| `1_SEVER/code/` | **Server code mirror** (newer fork copy + `baseline_choice/` deploy copy) — read-mostly | see below |
| `4_baseline_choice/` | Cross-framework baseline comparison project (local full-dev copy) | `run_*.py` + `configs/baselines.yaml` — see its guides |
| `2_catoon/` | Manim teaching animations (`0_Learning`, `1_LeNet`, `3_mech_course` L01–L10) | `manim -pql <file>.py <SceneClass>` |
| `3_研究生/` | Research plans, literature surveys, historical archive | — |
| `5_novels/` | Side project: ten ~100k-char web novels; resume via `NOVELS_PLAN.md` + `NOVELS_PROGRESS.md` | — |
| `data/` | Datasets (git-ignored) + converters in `data/tools/` (`hebing.py`, `json2yolo_pose.py`) | — |

Root-level scratch files (`aisheer.py`, `niuq.py`, `test.py`, `PAT_ch_prime.png`) are standalone one-off experiments (Escher-spiral image transforms, a matplotlib diagram), not part of any sub-project.

## `ultralytics-main-new/` — the active citrus line

**Install editable first**: `cd ultralytics-main-new && pip install -e .` so `from ultralytics import YOLO` resolves to this fork. Tests: `pytest tests`.

### Drivers and the fixed protocol

`train_citrus_seg.py` fixes every hyperparameter except architecture (`FIXED`: AdamW, lr0=0.01, dropout=0.0, seed=42, deterministic, amp=0, patience=100; imgsz locked at 640) so E0/E1…E4 are clean one-variable ablations. Only `--model --name --data --pretrained --epochs --batch --imgsz --device` are CLI knobs.

```powershell
python train_citrus_seg.py --model yolo11n-seg.pt --name E0_yolo11n_seg_baseline_941
python train_citrus_seg.py --model 0_orange_yaml/004_yolo11-seg-mano.yaml --pretrained yolo11n-seg.pt --name E1_mano
python train_citrus_seg.py --model yolo11n-seg.pt --name E0_smoke --epochs 3    # always smoke before 300ep
python eval_citrus_seg.py --weights 1_results\ORANGE_WUXI_SEG\<run>\weights\best.pt
```

Gotchas:
- The local driver's default `DATA` points at `data/test/orange_wuxi_seg.yaml`, **which no longer exists** — pass `--data 200orange_wuxi_seg.yaml` (points at `data/orange_yolo`).
- `train_citrus_seg.py` has been broken twice by hand edits: a hyperparameter must live in exactly one place (the `FIXED` dict **or** a CLI flag, never both); after editing, run `python train_citrus_seg.py --help` to sanity-check before launching anything.
- `eval_citrus_seg.py` appends one row per split to `1_results/ORANGE_WUXI_SEG/results_summary.csv` — that CSV is the single results table; never hand-copy numbers across protocols.
- Runs land in `ultralytics-main-new/1_results/ORANGE_WUXI_SEG/<name>/`. Keep numbered run names and **never overwrite a completed run**.
- Preliminary runs `001`–`003` used a different protocol (lr0=0.001, dropout=0.1, trained from YAML) — their metrics are **not comparable** with current runs.

### Dataset

Current dataset: **`data/orange_yolo/`** — 965 RGB images (train 676 / val 193 / test 96), single class `orange_immature`, with a **group-aware split** (`group_split_manifest.csv`, `group_edges.csv`, `split_audit_report.json` keep same-burst frames together). `data/orange_yolo_cleaned_min4px/` is a cleaned variant. The older 941-image `data/test/` split referenced in `AGENTS.md`/`README.md` had burst-sequence leakage across splits and is no longer on disk.

### Model YAMLs and custom modules

`0_orange_yaml/` is the local ablation ladder: `001_yolo11-seg` baseline; `002_*starnet*` (official `-s1`/`-s2` supersede the first starnet version); `003_mobilenetv4` (negative result, abandoned); `004`–`008` C2MANO placements (all / P3 / P4 / P5 / P345); `010` HVI; `011` HVI+MANO; `012` P2-CFS. Fork-root dataset YAMLs: `200orange_wuxi_seg.yaml` (current), `205_jeurk_spilt_data.yaml`, `206_Apple_Amodal.yaml` (legacy RGB-D, `channels: 4`).

**Adding a custom module means touching 4 files** (the key mechanism here):
1. Implement the `nn.Module` under `ultralytics/nn/modules/` (e.g. `mano.py`, `p2_cfs_attention.py`)
2. Export it in `ultralytics/nn/modules/__init__.py` (import + `__all__`)
3. Import it in `ultralytics/nn/tasks.py` (top-of-file imports)
4. Register it in `parse_model()`: add to the `base_modules` frozenset; modules with non-standard channel math need a dedicated `elif m is ...:` branch

Forgetting step 3 or 4 produces a YAML-parse error, not a clear "unknown module" message. Currently registered for the citrus line: `C2MANO` (`mano.py`), `P2CFSAttention` + `SegmentP2CFS` head (`p2_cfs_attention.py`), `HVIEnhance` (`hvi_enhance.py`), StarNet block (`starnet.py`). Module documentation lives in root `模块使用说明.md`. Legacy RGB-D modules (`custom_blocks.py` SFM/WCAF/DGFFN, `scale_aware_fusion.py`, `mobilenetv3_rgb`/`mobilenetv4_rgb`/`starnet_depth`/`shufflenetv2_depth`, `ct_modules.py`) remain registered; `rgbd_fusion_neck.py` is exported in `__init__.py` but **still not registered in `tasks.py`** — unreachable from YAMLs.

Custom optimizers beyond stock Ultralytics (`engine/trainer.py`): **PIDAO**, **MuSGD** (`ultralytics/optim/muon.py`), **SMCAO** (`smcao_v22_scheduler.py`) — select via `optimizer="PIDAO"`.

## `1_SEVER/` — server code mirror (read-mostly)

`1_SEVER/code/` is a copy-back of the Linux server's `/data/sxq/` code. Two subtrees:
- `1_SEVER/code/ultralytics-main-new/` — **newer than the local fork**. Adds the 73-config F-series ladder + SXQNet V1–V10 family (`0_orange_yaml/1_far_small/F01…F73`), `verify_far_yamls.py` (build/forward/params/GFLOPs self-check of all 73 YAMLs), `tests/test_citrus_far.py` (56 tests), `BASELINES.md`, extended `train_citrus_seg.py` CLI flags (`--iou-type/--inner-ratio/--nwd-ratio/--slide` losses, `--tal-metric/--tal-min-pos` GA-TAL assigner, `--freq-loss`, `--aug-preset`, `--optimizer Lion`), and `1_batch/` (server batch-run ledger: `batch_ledger.json` + `logs/`; run names `<yamlstem>_<epochs>ep`). Its drivers hardcode server paths — data `/data/sxq/datasets/orange_yolo`, results `/data/sxq/results/000_anyothers/`.
- `1_SEVER/code/baseline_choice/` — server deploy copy of the baseline suite (same code as `4_baseline_choice/`; `platform_paths()` switches Windows↔Linux paths by `os.name`).

**Rules:** it is a mirror. When logic edits are needed (to copy back to the server), change logic only — **never modify the `SERVER_*` constants or any `/data/sxq/...` path**, never "fix" them to Windows paths, and keep the relative layout intact. Root `README_改进总览.md` is the master copy of the F-series design doc (the one inside 1_SEVER is its synced server-side twin).

## `4_baseline_choice/` — cross-family baseline suite

Self-contained engineering for paper 1's cross-family comparisons; config center is `configs/baselines.yaml` (seeds, primary metric `mask_ap_50_95`, efficiency + semantic metric lists). Entry points: `run_yolo_baselines.py` (YOLOv8n/YOLO11n/YOLO26n-seg), `run_mmdet.py` (RTMDet-Ins-tiny, SOLOv2-Light), `run_maskrcnn.py` (torchvision Mask R-CNN R50-FPN), `run_rfdetr.py`, `run_unet.py` (U-Net + marker-controlled watershed). Drivers auto-call `scripts/prepare_dataset.py` to convert `data/orange_yolo` into each framework's format; grouped 4-fold CV utilities are `scripts/build_grouped_citrus_cv.py` / `run_build_grouped_citrus_cv.py`. Vendored: `detectron2-main/`, `UNet_server_package/`. Per-framework deps in `requirements-*.txt`; tests: `pytest tests` from `4_baseline_choice/`. **Read `全基线对比一键运行指南.md` and `基线网络与数据集转换使用指南.md` before running.**

## Other notes

- `2_catoon/` — Manim scenes; `-q l/m/h` for quality. `3_mech_course/` is a 10-lesson series (`L01`–`L10/scenes.py`).
- `3_研究生/` — research archive; its own `AGENTS.md` is stale (describes the pre-citrus layout) — the root `AGENTS.md` supersedes it.
- Only claim "lightweight/efficient" — edge/Jetson deployment has not been tested.

## Style & commit conventions

Python, 4-space indentation, ~120-column lines. The fork's `pyproject.toml` enables Ruff, isort, YAPF, Google-style docstrings, and pytest. Numbered naming: `NNN_name_vX.py`, `NNN_ablation_topic.yaml`, `F##_<arch>.yaml`. Concise scoped commits (`citrus: add cross-family baseline configs`). Never commit datasets, weights, `runs/`, large result images, archives, or videos; record the exact command, Git state, dataset-split version, hardware, and final metrics for every paper experiment.

## CI: issue-driven README blog list (still broken)

`.github/workflows/main.yml` runs `.github/scripts/update_readme.py` when an Issue with the **`blog`** label is opened/edited/labeled, to append the Issue to a blog list in `README.md`. **Still non-functional on two counts:** the run step invokes `python scripts/update_readme.py` (wrong path — the script lives under `.github/scripts/`), and `README.md` has no `<!-- BLOG_LIST -->` marker, so the script would no-op. Fix both before relying on it.
