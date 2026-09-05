# SAGE series

_Immature-citrus instance segmentation · SAGE V5 candidates / preserved V4R evidence · 2026-09-04_

## Current SAGE V5 candidates

Use `RUN_SAGE_V5.py` for VS Code foreground sequential training. `SAGE50--56` are seven new candidates;
`SAGE30` and `SAGE42` are the two reference models in the V5 matrix. No V4R YAML is overwritten.

Read the [evidence and design](../../docs/SAGE_V5_EVIDENCE_AND_DESIGN.md),
[training instructions and fixed hyperparameters](../../docs/SAGE_V5_TRAINING.md), and
[verification record](../../reports/sage_v5_20260904/VERIFICATION.md) first.
Default `screen` selects 30/42/50/51/52. `DRY_RUN=False` actually trains; True only builds and exits.
No new candidate has a verified full-training accuracy result yet.

---

## Preserved SAGE V4R reconstruction

`SAGE40--48` are the preserved V4R experiments and `SAGE30` is their fixed-protocol official control. The complete
architecture, loss definitions, evidence limits and run order are documented in
[`docs/SAGE_V4_RECONSTRUCTED_GUIDE.md`](../../docs/SAGE_V4_RECONSTRUCTED_GUIDE.md). Use `RUN_SAGE_V4.py` for a
visible, sequential VS Code run. `SAGE00--35` remain unchanged below for reproducibility; V4R does not make their
historical results directly comparable when dataset or protocol differs.

## 📋 Which files are current

SAGE means **Semantic-Aligned Geometry-Evidence**. `SAGE00--04` and `SAGE10--17` remain unchanged for reproducibility.
The v3 experiments are `SAGE20--27`: a low-resolution axial shape-context backbone and a topology-supervised
innovation pyramid. Every YAML, including current V4R, remains loadable through the standard `YOLO(yaml).load(pt)` API.

| Range | Status | Purpose |
| --- | --- | --- |
| `SAGE00--04` | Legacy v1 | Mask-only progressive fusion |
| `SAGE10` | Paired control | Exact YOLO11n-seg |
| `SAGE11--15` | Primary v2 | Conservative pretrained-residual route |
| `SAGE16--17` | Aggressive v2 | Complete PAN replacement control |
| `SAGE20--26` | Preserved v3 | Shape-context and innovation-correction causal ablations |
| `SAGE27` | Exploratory v3 | Narrow-FLOPs candidate; not in the default queue |
| `SAGE30--35` | Preserved v4 | First V4 attempt; retained unchanged |
| `SAGE40--48` | Preserved V4R | Asymmetric neck, mask correction, geometry and P4 isolation |
| `SAGE50--56` | New V5 candidates | Dual-use detail route, late prototypes, geometry and isolated P5 WT |

## 📚 SAGE-v2 ablation matrix

| YAML | Backbone | Fusion / head | Loss role |
| --- | --- | --- | --- |
| `SAGE10_official_control.yaml` | Official | Official PAN + Segment | Official |
| `SAGE11_deep_backbone.yaml` | P4/P5 `C3k2SAGE` | Official | Official |
| `SAGE12_residual_pyramid.yaml` | Official | Residual topology pyramid | Official |
| `SAGE13_topology_supervision.yaml` | Official | Residual topology pyramid | Shared topology |
| `SAGE14_joint_core.yaml` | P4/P5 `C3k2SAGE` | Residual topology pyramid | Shared topology |
| `SAGE15_full_task_loss.yaml` | P4/P5 `C3k2SAGE` | Residual topology pyramid | Full task loss |
| `SAGE16_replace_neck.yaml` | Official | PAN replaced by SAGE pyramid | Shared topology |
| `SAGE17_joint_replace.yaml` | P4/P5 `C3k2SAGE` | PAN replaced by SAGE pyramid | Shared topology |

The shared topology target has four states: background/context, fruit interior, visible boundary and adjacent-instance
separator. P2 detail is rearranged into P3, but SAGE deliberately does not add a P2 prediction tower. This is meant to
protect tiny boundaries without repeating the expensive high-resolution route that slowed earlier models.

## 📚 SAGE-v3 ablation matrix

| YAML | Backbone | Fusion / head | Training role |
| --- | --- | --- | --- |
| `SAGE20_shape_context_backbone.yaml` | P4/P5 axial shape context | Official | Backbone isolation |
| `SAGE21_innovation_pyramid.yaml` | Official | Innovation pyramid | Stock loss isolation |
| `SAGE22_contrast_topology.yaml` | Official | Innovation + four-state topology | Explicit topology |
| `SAGE23_joint_core_v3.yaml` | Axial shape context | Innovation + four-state topology | Primary core |
| `SAGE24_style_robust.yaml` | SAGE23 + training-only style swap | Same as SAGE23 | Colour-reliance ablation |
| `SAGE25_quality_aligned.yaml` | Same as SAGE23 | Same as SAGE23 | Varifocal PR-ranking ablation |
| `SAGE26_occlusion_topology.yaml` | Same as SAGE23 | Same as SAGE23 | Concavity/separator loss ablation |
| `SAGE27_joint_lite.yaml` | Narrow axial context | 24-channel innovation route | Hardware Pareto exploration |

## ⚡ Compute and initialization audit

Numbers below use `nc=80` only for exact comparison with the official checkpoint; training overrides `nc` from the
dataset YAML.

| Model | Params | GFLOPs@640 | Compatible pretrained params |
| --- | ---: | ---: | ---: |
| SAGE10 | 2.877M | 10.529 | 100.00% |
| SAGE11 | 2.984M | 10.679 | 96.42% |
| SAGE12/13 | 2.940M | 10.938 | 97.85% |
| SAGE14/15 | 3.047M | 11.087 | 94.42% |
| SAGE16 | 2.237M | 9.555 | 93.11% |
| SAGE17 | 2.344M | 9.704 | 88.86% |
| SAGE20 | 2.888M | 10.417 | 98.41% |
| SAGE21/22 | 2.941M | 11.040 | 96.68% |
| SAGE23--26 | 2.986M | 11.102 | 95.20% |
| SAGE27 | 2.940M | 10.833 | 96.68% |

The implementation excludes deformable convolution, `grid_sample`, CARAFE, `unfold`, dynamic kernels, Mamba and
full-resolution attention matrices. Local CPU screening at 256 pixels measured SAGE20, SAGE21, SAGE23 and SAGE27 at
approximately `1.120x`, `1.174x`, `1.205x` and `1.233x` of the control's complete forward/backward time. The narrower
SAGE27 was not faster and is excluded from default screening. Target-GPU benchmarking remains mandatory.

## 🎯 Recommended experiment order

1. Run the v3 batch script with `--suite all --dry-run`
2. Benchmark SAGE20, SAGE21 and SAGE23 on an otherwise idle target GPU
3. Run v3 `--suite smoke --epochs 3`
4. Run v3 `--suite screen --epochs 50`
5. Test SAGE24--26 only if the joint core survives screening
6. Promote only a same-protocol Pareto survivor to 300 epochs and three seeds

Do not launch all eight v3 candidates for 300 epochs. SAGE23 is a hypothesis, not a declared winner. SAGE27 demonstrates
that lower FLOPs can still be slower and should not be promoted without a target-GPU result.

See [`20260902_CITRUS_SAGE_DESIGN.md`](../../20260902_CITRUS_SAGE_DESIGN.md) and
[`20260902_CITRUS_SAGE_SERVER.md`](../../20260902_CITRUS_SAGE_SERVER.md) for v2. Current v3 evidence and commands are in
[`20260902_CITRUS_SAGE_V3_DESIGN.md`](../../20260902_CITRUS_SAGE_V3_DESIGN.md) and
[`20260902_CITRUS_SAGE_V3_SERVER.md`](../../20260902_CITRUS_SAGE_V3_SERVER.md).
