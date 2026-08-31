# PR-tail diagnosis and Light redesign evidence

Search date: 2026-08-31

## Research question

How should a lightweight immature-citrus instance-segmentation model improve the low-confidence precision/recall trade-off without repeating the slow P2 dual-stream and frequency-neck failures observed in G0830?

## Queries

- `VarifocalNet CVPR 2021 official paper Varifocal Loss dense object detector GitHub`
- `Generalized Focal Loss NeurIPS 2020 official paper quality focal loss GitHub`
- `QueryDet CVPR 2022 official paper small object high resolution sparse query GitHub`
- `FasterNet CVPR 2023 official paper partial convolution dense prediction GitHub`

## Primary papers and official code

| Work | Primary paper | Official code | Evidence used |
|---|---|---|---|
| VarifocalNet | https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html | https://github.com/hyz-xmaster/VarifocalNet | Dense candidate ranking benefits when the classification score jointly represents object presence and localization quality; Varifocal Loss asymmetrically down-weights negatives and emphasizes high-quality positives. |
| Generalized Focal Loss | https://papers.neurips.cc/paper_files/paper/2020/hash/f0bda020d2470f2e74990a07a607ebd9-Abstract.html | https://github.com/implus/GFocal | A joint continuous classification-quality representation avoids train/inference inconsistency in dense detection ranking. |
| QueryDet | https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html | https://github.com/ChenhongyiYang/QueryDet-PyTorch | Persistent dense high-resolution computation is expensive; high-resolution features should be restricted to candidate locations or a limited path. The paper reports +2.0 AP-small and about 3x high-resolution acceleration on COCO, but this does not guarantee citrus gains. |
| FasterNet | https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html | https://github.com/JierunChen/FasterNet | Partial-channel spatial mixing can reduce memory traffic, but realized latency must be measured and compression should not be assumed accuracy-neutral. |
| Mask Scoring R-CNN | https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html | https://github.com/zjhuang22/maskscoring_rcnn | Classification confidence and mask IoU are misaligned in instance segmentation; an explicit mask-IoU score can improve mask ranking and PR/AP without changing the predicted masks. |
| NWD | https://arxiv.org/abs/2110.13389 | https://github.com/jwwangchn/NWD | IoU supervision is unstable for boxes only a few pixels wide. NWD is retained as a separate tiny-localization loss ablation, not mixed into the structural Light claim. |
| RTMDet | https://arxiv.org/abs/2212.07784 | https://github.com/open-mmlab/mmdetection/tree/3.x/configs/rtmdet | Soft-label dynamic assignment and balanced backbone/neck capacity are relevant comparison principles; RTMDet-Ins-tiny remains a required non-YOLO baseline rather than a module donor. |

## Local evidence that constrains the redesign

- Ultralytics `compute_ap()` appends a zero-precision sentinel at the model's maximum achieved recall and at recall 1.0. Therefore the final vertical drop and zero tail are plotting/evaluation conventions; removing them would falsify the standard AP curve rather than improve the model.
- G0830 G00 and G02 both reach only about 0.88 mask recall at confidence 0.0. The unresolved issue is the approximately 12% unmatched ground-truth ceiling plus false positives admitted at low confidence.
- At the confusion-matrix operating point (`conf=0.25`, box IoU 0.45), G00 has TP/FP/FN = 1676/410/422 and G02 has 1671/439/427. Thus the bilateral P2 backbone is worse than the official control on both FP and FN at this threshold.
- G03 frequency neck is slower and less accurate; frequency fusion is excluded from the revised Light core.
- Full generic-backbone replacement previously reduced accuracy. Revised Light must preserve the shallow P2/P3 extraction stages and compress only P4/P5.

## Design boundary

The evidence supports two independently testable changes, not a module stack:

1. Stage-wise compression: preserve the official shallow P2/P3 stages and replace only deep P4/P5 extraction with partial-convolution residual stages.
2. Near-identity progressive fusion: retain adjacent-scale fusion but initialize each source injection as a small residual around the destination feature, avoiding repeated 0.5/0.5 averaging of weak tiny-object evidence.

Quality-aware classification is a separate loss ablation (`citrus_vfl`), because it targets candidate ranking and cannot recover ground truths for which the network produces no matching candidate.

Mask-quality calibration and NWD are also separated: the former targets the ordering of mask hypotheses, while the latter targets tiny-box localization. Neither is allowed to mask an unsuccessful Light backbone/neck ablation.
