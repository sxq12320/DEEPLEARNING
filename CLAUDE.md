# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository nature

This is a monorepo centered on a **master's thesis on vision for citrus bagging**, plus independent side projects. There is **no top-level build, lint, test, or dependency setup** — each sub-project has its own entry point and dependencies; **always `cd` into the relevant sub-project before running anything**. Repo-wide research rules (priorities, baseline matrix, experiment discipline, git hygiene) live in **`AGENTS.md`** — treat it as authoritative; `README.md` (Chinese) is the narrative overview.

Cross-cutting facts that will bite you:
- **Scripts and dataset YAMLs use hardcoded absolute paths** — `E:\mastercode\...` on this machine, `/data/sxq/...` in server-side code. They must be edited if the repo moves.
- **Weights and data are untracked.** `.gitignore` excludes `*.pt`/`*.pth`/`*.onnx`, `/data/`, `runs/`, archives, and videos. Datasets live under `data/` locally only.
- **The YOLO fork exists in two copies**: `ultralytics-main-new/` (local dev tree — **stale, ~2026-08**) and `1_SEVER/code/ultralytics-main-new/` (server code mirror — **the active codebase**, holds all SAGE/E/I-series configs). All new work happens in the 1_SEVER copy and is re-uploaded to the server. (`1.coding/` and `9_archive/` mentioned by older docs no longer exist.)

## Current research priority (short form of AGENTS.md)

Two connected papers: (1) **lightweight, high-accuracy RGB instance segmentation of immature citrus** (the active work), (2) peduncle-point localization using paper 1's ROIs. Paper 1 stays strictly RGB single-class (`orange_immature`) — **no RGB-D, amodal, OBB, or pose mixing**. The research source of truth is `3_研究生/柑橘套袋视觉_完整研究执行计划.md`. The legacy RGB-D apple-occlusion line (`channels: 4`, SFM/WCAF, `206_Apple_Amodal.yaml`) still lives in the fork but is dormant.

## Sub-project map

| Path | What it is | How to run |
|------|-----------|-----------|
| `ultralytics-main-new/` | Customized Ultralytics fork — **local dev copy, stale since ~2026-08** | legacy reference only |
| `1_SEVER/code/ultralytics-main-new/` | **The active citrus seg codebase** (server mirror, SAGE→E→I_V1) | `pip install -e .` there; `RUN_CITRUS_<series>.py` / `*_batch.py` / `pytest tests` |
| `1_SEVER/code/` | Server mirror root (also holds `baseline_choice/` deploy copy) | see below |
| `4_baseline_choice/` | Cross-framework baseline comparison project (local full-dev copy) | `run_*.py` + `configs/baselines.yaml` — see its guides |
| `2_catoon/` | Manim teaching animations (`0_Learning`, `1_LeNet`, `3_mech_course` L01–L10) | `manim -pql <file>.py <SceneClass>` |
| `3_研究生/` | Research plans, literature surveys, historical archive | — |
| `5_novels/` | Side project: ten ~100k-char web novels; resume via `NOVELS_PLAN.md` + `NOVELS_PROGRESS.md` | — |
| `data/` | Datasets (git-ignored) + converters in `data/tools/` (`hebing.py`, `json2yolo_pose.py`) | — |

Root-level scratch files (`aisheer.py`, `niuq.py`, `test.py`, `PAT_ch_prime.png`) are standalone one-off experiments (Escher-spiral image transforms, a matplotlib diagram), not part of any sub-project.

## `ultralytics-main-new/` — legacy local dev copy (stale)

This root-level fork froze around 2026-08 and no longer receives work. The active codebase is
`1_SEVER/code/ultralytics-main-new/` — same layout, plus all SAGE/E/I-series modules, `protocols/`,
`docs/`, per-series `*_batch.py` runners, `citrus_foreground.py` (`RUNNERS` registry) and
`RUN_CITRUS_<series>.py` foreground entries. The notes below describe the shared conventions that
still apply to the active copy: `train_citrus_seg.py`/`eval_citrus_seg.py` drivers, the
`0_orange_yaml/` ablation ladder, and the 4-file module-registration mechanism
(`ultralytics/nn/modules/` → `__init__.py` → `tasks.py` imports → `parse_model()` sets).
In the active copy, series suites additionally live in `citrus_*_suite.py`, YAML generators in
`scripts/generate_citrus_*_yaml.py`, fixed training args in `citrus_protocol.py` + `protocols/`,
and contract tests in `tests/test_citrus_*.py`.

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

Formal dataset: **`data/orange_yolo_grouped_dedup_20260820/`** — 965 RGB images (train 676 / val 193 / test 96; 5,890 instances), single class `orange_immature`, group-aware split + dedup (`audit/`, `group_split_manifest.csv`). The older `data/orange_yolo/` (same 965 images, different split) has **123/303 groups crossing splits** — never use it for formal claims. Note the E V11/V12 server runs used yet another val membership (193 images / 1,049 instances vs grouped_dedup's 1,181); their rankings are screening evidence only — formal conclusions need grouped_dedup reruns at three seeds.

### Model YAMLs and custom modules

`0_orange_yaml/` is the local ablation ladder: `001_yolo11-seg` baseline; `002_*starnet*` (official `-s1`/`-s2` supersede the first starnet version); `003_mobilenetv4` (negative result, abandoned); `004`–`008` C2MANO placements (all / P3 / P4 / P5 / P345); `010` HVI; `011` HVI+MANO; `012` P2-CFS. Fork-root dataset YAMLs: `200orange_wuxi_seg.yaml` (current), `205_jeurk_spilt_data.yaml`, `206_Apple_Amodal.yaml` (legacy RGB-D, `channels: 4`).

**Adding a custom module means touching 4 files** (the key mechanism here):
1. Implement the `nn.Module` under `ultralytics/nn/modules/` (e.g. `mano.py`, `p2_cfs_attention.py`)
2. Export it in `ultralytics/nn/modules/__init__.py` (import + `__all__`)
3. Import it in `ultralytics/nn/tasks.py` (top-of-file imports)
4. Register it in `parse_model()`: add to the `base_modules` frozenset; modules with non-standard channel math need a dedicated `elif m is ...:` branch

Forgetting step 3 or 4 produces a YAML-parse error, not a clear "unknown module" message. Currently registered for the citrus line: `C2MANO` (`mano.py`), `P2CFSAttention` + `SegmentP2CFS` head (`p2_cfs_attention.py`), `HVIEnhance` (`hvi_enhance.py`), StarNet block (`starnet.py`). Module documentation lives in root `模块使用说明.md`. Legacy RGB-D modules (`custom_blocks.py` SFM/WCAF/DGFFN, `scale_aware_fusion.py`, `mobilenetv3_rgb`/`mobilenetv4_rgb`/`starnet_depth`/`shufflenetv2_depth`, `ct_modules.py`) remain registered; `rgbd_fusion_neck.py` is exported in `__init__.py` but **still not registered in `tasks.py`** — unreachable from YAMLs.

Custom optimizers beyond stock Ultralytics (`engine/trainer.py`): **PIDAO**, **MuSGD** (`ultralytics/optim/muon.py`), **SMCAO** (`smcao_v22_scheduler.py`) — select via `optimizer="PIDAO"`.

## `1_SEVER/` — server code mirror (the active codebase)

`1_SEVER/code/` is a copy-back of the Linux server's `/data/sxq/` code. Two subtrees:
- `1_SEVER/code/ultralytics-main-new/` — **the active codebase**. Beyond the historical ladders it now holds the SAGE V4R–V8 and E V1–V12 families plus the current **I_V1** series (`0_orange_yaml/<Series>_series/`, 423 YAMLs, indexed in `MODEL_INDEX.csv`). Workflow conventions: `citrus_<series>_suite.py` (NAMES/FACTORS/SUITES), `scripts/generate_citrus_*_yaml.py` (refuse to overwrite), `protocols/citrus_paper1_formal_v2_ram.yaml` (fixed protocol), `citrus_foreground.py` `RUNNERS` + `RUN_CITRUS_<series>.py` (VS Code ▶ foreground sequential training), `tests/test_citrus_*.py` contract tests, `docs/` design+review docs. Latest: `docs/I_V1_DESIGN_20260920.md`, `docs/I_V1_REVIEW_20260920/reviewer{1,2,3}.md`. Its drivers hardcode server paths — data `/data/sxq/datasets/...`, results `/data/sxq/results/<SERIES>/...`.
- `1_SEVER/code/baseline_choice/` — server deploy copy of the baseline suite (same code as `4_baseline_choice/`; `platform_paths()` switches Windows↔Linux paths by `os.name`).

**Rules:** it is a mirror. When logic edits are needed (to copy back to the server), change logic only — **never modify the `SERVER_*` constants or any `/data/sxq/...` path**, never "fix" them to Windows paths, and keep the relative layout intact.

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
