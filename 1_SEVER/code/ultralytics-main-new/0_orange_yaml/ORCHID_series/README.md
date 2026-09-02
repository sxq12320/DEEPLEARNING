# ORCHID series

ORCHID means **Object-Region Conditioned Hierarchical Information Decoupling**. It tests a different fusion
paradigm for immature-citrus instance segmentation: detection and mask evidence are not forced to share one
fully fused feature pyramid, and high-resolution detail is admitted only around coarse tiny-fruit candidates.

| YAML | Structural question | Training-only override | Params | GFLOPs @640 |
|---|---|---:|---:|---:|
| `ORCHID00_official_control.yaml` | Exact paired YOLO11n-seg control | none | 2.877M | 10.529 |
| `ORCHID01_task_decoupled.yaml` | Does a separate raw-feature mask path help by itself? | none | 2.902M | 10.805 |
| `ORCHID02_latent_query_router.yaml` | Can mask loss learn a causal P2 gate without direct query labels? | none | 2.902M | 10.805 |
| `ORCHID03_supervised_query_router.yaml` | Does explicit tiny-centre supervision stabilize routing? | `citrus_query=0.10` | 2.902M | 10.805 |
| `ORCHID04_single_canvas_neck.yaml` | Can one P3 canvas replace the complete recurrent PAN? | `citrus_query=0.10` | 2.237M | 9.660 |
| `ORCHID05_decam_reference.yaml` | Does local candidate/background difference reduce fruit-leaf camouflage? | query 0.10, contrast 0.05 | 2.904M | 10.826 |
| `ORCHID06_query_router_lite.yaml` | Can the supervised route retain accuracy with one-block towers? | `citrus_query=0.10` | 2.741M | 9.894 |

`ORCHID00` is intentionally excluded from `--suite all`; use `--suite control` only for an explicit paired audit.
All other models use the locked protocol in `protocols/citrus_paper1_formal_v1.yaml` and the same
`yolo11n-seg.pt` initialization.

Recommended order:

1. `--suite smoke --epochs 3` tests ORCHID03 and ORCHID04.
2. `--suite screen --epochs 50` tests the structural chain ORCHID01--05.
3. Promote only Pareto survivors to 300 epochs and three seeds.

Do not interpret the table as an accuracy claim. Params/GFLOPs are verified locally; accuracy requires training
on the locked group-aware split.
