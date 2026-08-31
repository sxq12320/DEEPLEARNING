# Light architecture literature and code search log

Search date: 2026-08-30  
Question: Which lightweight backbone and non-rigid multi-scale neck designs can preserve tiny immature-citrus detail while reducing realized latency for RGB instance segmentation?

## Sources searched

- CVF Open Access (CVPR/ICCV/WACV primary papers)
- arXiv primary manuscript records
- NeurIPS Proceedings
- IEEE publication metadata
- Official author GitHub repositories and released configs/checkpoints
- Existing local repositories under `C:/Users/33836/Desktop/github`

## Search strings

- `lightweight backbone latency memory access object detection instance segmentation official code`
- `partial convolution FasterNet CVPR 2023 official`
- `RepViT MobileOne FastViT EfficientViT dense prediction official code`
- `lightweight high resolution backbone small object segmentation Lite-HRNet`
- `AFPN asymptotic feature pyramid adaptive spatial fusion official code`
- `Gold-YOLO gather distribute neck NeurIPS official code`
- `DAMO-YOLO RepGFPN lightweight head official code`
- `small object high resolution sparse query QueryDet official code`
- `PointRend boundary refinement instance segmentation official code`
- `occluded touching object concave mask instance segmentation boundary topology`

## Inclusion criteria

1. Primary paper with explicit architecture evidence and reproducible experimental tables.
2. Official or author-linked source code when available.
3. Evidence on detection, instance/semantic segmentation, high-resolution prediction, or realized hardware latency.
4. Design can be expressed with stable PyTorch operators and integrated through a standard Ultralytics YAML.
5. Plausible nano-scale budget after adding a segmentation head.

## Exclusion criteria

1. Classification-only claims with no dense-prediction evidence and no safe pyramid interface.
2. Custom CUDA/sparse/deformable kernels that would make the user's server environment fragile.
3. Persistent full-resolution branches whose measured cost conflicts with the current speed problem.
4. Architectures already exceeding the target budget before adding the neck and segmentation head.
5. Secondary citrus-YOLO papers that only stack plug-and-play modules without isolating structural effects.

## Evidence matrix

| Work | Primary source | Official code | Evidence used | Decision |
|---|---|---|---|---|
| FasterNet | https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html | https://github.com/JierunChen/FasterNet | PConv reduces redundant spatial compute and memory access; latency must be measured | Adopt principle |
| AFPN | https://arxiv.org/abs/2306.15988 | https://github.com/gyyang23/AFPN | Adjacent progressive fusion and adaptive spatial weighting | Adopt compressed topology |
| Gold-YOLO | https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html | https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO | Gather/distribute alternative to conventional FPN/PAN | Adopt topology vocabulary/principle |
| DAMO-YOLO | https://arxiv.org/abs/2211.15444 | https://github.com/tinyvision/DAMO-YOLO | Efficient RepGFPN and large-neck/small-head analysis | Adopt small-head constraint, not full neck |
| RepViT | https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html | https://github.com/THU-MIG/RepViT | Strong mobile latency/accuracy design | Candidate, excluded this round due transfer/interface risk |
| MobileOne | https://openaccess.thecvf.com/content/CVPR2023/html/Vasu_MobileOne_An_Improved_One_Millisecond_Mobile_Backbone_CVPR_2023_paper.html | https://github.com/apple/ml-mobileone | Device latency is not predicted by params/FLOPs | Use evaluation principle |
| FastViT | https://openaccess.thecvf.com/content/ICCV2023/html/Vasu_FastViT_A_Fast_Hybrid_Vision_Transformer_Using_Structural_Reparameterization_ICCV_2023_paper.html | https://github.com/apple/ml-fastvit | RepMixer reduces memory-access overhead | Candidate, excluded from Light core |
| Lite-HRNet | https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html | https://github.com/HRNet/Lite-HRNet | High-resolution multi-branch position detail | Exclude persistent P2 cost |
| EfficientViT dense | https://openaccess.thecvf.com/content/ICCV2023/html/Cai_EfficientViT_Lightweight_Multi-Scale_Attention_for_High-Resolution_Dense_Prediction_ICCV_2023_paper.html | https://github.com/mit-han-lab/efficientvit | Hardware-aware high-resolution dense prediction | Candidate, defer attention branch |
| QueryDet | https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html | https://github.com/ChenhongyiYang/QueryDet-PyTorch | Sparse high-res calculation avoids background waste | Design constraint; custom sparse execution deferred |
| PointRend | https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html | https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend | Refine uncertain boundary points rather than whole dense map | Future head ablation, not core |
| RapidNet-Ti | https://openaccess.thecvf.com/content/WACV2025/ | https://github.com/XiangxiangSUN/RapidNet | Pure CNN/dilated representation | Rejected after local source profile: ~6.63M and ~51.9 ms at 320 before detector head |

## Synthesis

The chosen architecture combines two compatible and independently testable principles: partial-channel spatial mixing in a non-CSP backbone, and adjacent progressive adaptive fusion in a non-PAN neck. It intentionally avoids combining every eligible paper. The Light00/Light01/Light02 ablation separates backbone, neck, and their interaction; Light03/Light04 then test the deployment-quality Pareto endpoints.

