# CitrusSwift-Seg controlled architecture suite

These ten YAMLs are a causal ablation suite, not ten cosmetic variants. They all keep the YOLO11n scale and one-class
training is applied by the dataset config at runtime.

| YAML | Controlled change |
|---|---|
| `00_reference.yaml` | unchanged YOLO11n-seg reference |
| `01_repcontext_backbone.yaml` | deploy-fusible P5 context |
| `02_lska_backbone.yaml` | P5 LSKA only |
| `03_train_aux_head.yaml` | P2/P3 training-only boundary/query/contrast heads |
| `04_lite_head.yaml` | shorter box/class/mask-coefficient prediction heads |
| `05_fpn_only_neck.yaml` | top-down-only neck speed boundary |
| `06_asym_pan_neck.yaml` | retain P3-to-P4 bottom-up; remove P4-to-P5 bottom-up |
| `07_lska_asym_pan.yaml` | backbone-context and neck-topology interaction |
| `08_citrus_swift_full.yaml` | complete latency-aware candidate |
| `09_dense_topology_control.yaml` | previous dense-P2 design as a same-protocol control |

Run them only through `20260824_citrus_swift_batch.py`, which locks the training protocol and records the experiment
ledger. The recommended order is 1-epoch smoke, 50-epoch screening, then 300 epochs and three seeds for the reference
and promoted finalist only.
