# SAGE-v3 evidence and source-code audit

_Immature-citrus instance segmentation · 2026-09-02_

---

## Abstract

This audit evaluates the two user-provided control-inspired design documents and the 108-file
`Plug-play-modules-main` collection against the actual citrus task and completed local experiments. The collection
contains 107 Python files, but filename counts show that it is not an all-CVPR repository: 16 paths mention CVPR,
5 ICCV, 5 ECCV, 4 AAAI, 3 ICLR, 3 NeurIPS and 33 arXiv. The repository is therefore an idea index, not evidence that
every block is peer-reviewed, efficient or transferable. Five primary sources were retained: ConDSeg for explicit
foreground/background/uncertainty reasoning, PKINet and CGRSeg for axial shape context, PIDNet for complementary
state/detail reasoning, and CrossNorm for a training-only appearance intervention. VarifocalNet is retained only as
an optional score-alignment loss. Their original blocks are not copied. SAGE-v3 re-expresses the shared principles as
one low-resolution shape-context backbone and one topology-supervised innovation pyramid built from accelerator-
friendly operations.

## Audit method

The audit used four levels of evidence:

1. Read both supplied Markdown documents completely and compare their hashes.
2. Inventory every file in `C:\Users\33836\Desktop\Plug-play-modules-main`, then scan for dynamic sampling,
   `unfold/fold`, FFT, large attention matrices and framework-specific dependencies.
3. Read the most task-relevant implementations in full and verify them against the paper or author repository.
4. Check each proposed transfer against the local results: HWD, frequency necks, dynamic resampling and fragmented
   lightweight paths have already produced poor accuracy or speed and cannot be reintroduced without isolation.

The two supplied control documents are byte-identical. They are treated as one design proposal, not two independent
sources of evidence.

## What was adopted and what was rejected

| Source | Original problem and mechanism | SAGE-v3 transfer | Rejected portion |
| --- | --- | --- | --- |
| ConDSeg, AAAI 2025[^1] | Low contrast, soft boundaries and co-occurring foreground/background; decouples foreground, background and uncertainty before contrast-driven aggregation | One citrus-specific four-state topology map: context, fruit interior, visible boundary and instance separator | Original two-stage medical pipeline, three large decoder towers and `unfold/fold` dynamic aggregation |
| PKINet, CVPR 2024[^2] | Large scale span and contextual ambiguity; poly-kernel inception plus context-anchor attention | Low-resolution horizontal/vertical axial context at P4/P5 | Five sequential 3/5/7/9/11 depthwise kernels, rotated-detection stack and MMRotate dependencies |
| CGRSeg, ECCV 2024[^3] | Efficient spatial reconstruction; rectangular self-calibration captures axial global context | Shape-context gate combined with local 3×3 measurement in the backbone | Full MMSegmentation decoder, Dynamic Prototype Guided head and framework dependencies |
| PIDNet, CVPR 2023[^4] | Real-time semantic segmentation with complementary detail, context and boundary branches | Semantic prediction, local innovation and local contrast are complementary signals in each fusion transition | Claiming the learned nonlinear network is a literal PID controller or has classical closed-loop stability |
| CrossNorm, ICCV 2021[^5] | Distribution-shift robustness by exchanging feature mean and variance during training | Optional P4 training-only statistics exchange with a conservative convex blend | SelfNorm inference branch and unconditional replacement throughout the network |
| VarifocalNet, CVPR 2021[^6] | Align dense-classification ranking with localization quality through an IoU-aware score | Optional `citrus_vfl=1.0` loss ablation using existing task-aligned soft targets | Copying VFNet's FCOS/ATSS detector, star-shaped representation or adding another head |

## Control-document critique

### Useful ideas

- A residual update should be bounded and initialized near the pretrained network.
- Deep semantic prediction can be compared with an aligned local feature to form an informative residual.
- Detail, context and boundary evidence should have different roles instead of being concatenated blindly.
- The final method needs staged gates: build, backward, FLOPs, speed, smoke, screen and only then final training.

### Claims that cannot be made

- A learned feature tensor is not automatically a Luenberger state estimate, and the input feature is not a known
  ground-truth latent state.
- `tanh` bounds one residual coefficient; it does not prove Lyapunov stability of the complete nonlinear CNN.
- Replacing downsampling, SPPF, neck or head layers cannot preserve official weights bit-for-bit for those layers.
- Proposed parameter counts, latency ratios and AP gains are hypotheses until measured by the implemented model.
- The two identical documents do not constitute independent corroboration.

SAGE-v3 therefore uses the neutral term **innovation correction** for `measurement - prediction`. The control-system
interpretation remains an engineering analogy and is never presented as a stability theorem.

## Why the selected mechanism matches this dataset

| Citrus failure | Required evidence | SAGE-v3 mechanism | Required evaluation |
| --- | --- | --- | --- |
| Tiny fruit disappears after downsampling | High-resolution local change plus semantic confirmation | P2 local contrast is losslessly rearranged to P3; no P2 prediction tower | AP-tiny, recall by area bin |
| Green fruit resembles leaves | Shape, context and appearance invariance | P4/P5 axial context; optional training-only statistic exchange | Camouflage subset AP and false-positive rate |
| Strip-like leaf/branch occlusion creates concave masks | Long-axis context plus visible-boundary evidence | Horizontal/vertical context and topology boundary state | AP by solidity/convex-hull deficit |
| Touching fruits merge while occluded fruit fragments | Boundary and separator must be distinct | Four-state topology predicts visible boundary separately from inter-instance separator | Split/merge error rate and gap-bin AP |
| PR tail collapses at high recall | Candidate ranking must correlate with quality | Isolated Varifocal loss experiment | PR curve, calibration error and fixed-recall precision |

## Rejected Plug-play candidates

| Candidate type | Why it was not transferred |
| --- | --- |
| DySample/deformable sampling | `grid_sample` introduces a latency-sensitive dynamic hot path; earlier dynamic necks were already slow |
| ConDSeg aggregation block | Two `unfold/fold` passes and dynamic `k^4` attention are too expensive at citrus feature resolutions |
| Full PKIBlock | Five spatial-kernel stages and a large FFN conflict with the nano-scale speed target |
| CARAFE/HWD/frequency stacks | Local experiments have already shown negative or confounded accuracy and speed evidence |
| Mamba variants | User explicitly does not want a Mamba dependency; many repository files also depend on external scan kernels |
| Generic channel/spatial attention stacks | They do not encode the fruit/leaf, boundary/separator or scale-specific problem and repeat failed module stacking |
| RGB-D/3D modules | Outside the current RGB paper scope |

## Reproducible source snapshots

The author repositories were downloaded to `C:\Users\33836\Desktop\github`:

| Repository | Commit |
| --- | --- |
| ConDSeg | `b4c22c399e72ec858026abc0e87143f3c53fe12d` |
| CGRSeg | `0bc4d30556c6e380ba1e7ea8ab692f84d849ac61` |
| PKINet | `a33aa22d188c9946cc83fba60e3bb8ac0ec82ff7` |
| crossnorm-selfnorm | `58e8c739c124eb183ad21a08701516509453762e` |
| PIDNet | `4c158cf24ce432f0a8cb43364fae38d93cee0dc3` |

ConDSeg does not ship a root license file and its README limits use to research/education. No ConDSeg source code was
copied. PKINet, CGRSeg and CrossNorm use Apache-2.0 licenses; SAGE-v3 still implements an independent, task-specific
mechanism rather than transplanting their classes.

## References

[^1]: Lei, M., Wu, H., Lv, X., & Wang, X. (2025). “ConDSeg: A General Medical Image Segmentation Framework via Contrast-Driven Feature Enhancement.” _AAAI_. <https://ojs.aaai.org/index.php/AAAI/article/view/32482>

[^2]: Cai, Z. et al. (2024). “Poly Kernel Inception Network for Remote Sensing Detection.” _CVPR_. <https://openaccess.thecvf.com/content/CVPR2024/papers/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.pdf>

[^3]: Ni, Z. et al. (2024). “Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation.” _ECCV_. <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06867.pdf>

[^4]: Xu, J., Xiong, Z., & Bhattacharyya, S. P. (2023). “PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers.” _CVPR_. <https://arxiv.org/abs/2206.02066>

[^5]: Tang, Z. et al. (2021). “CrossNorm and SelfNorm for Generalization Under Distribution Shifts.” _ICCV_. <https://openaccess.thecvf.com/content/ICCV2021/html/Tang_CrossNorm_and_SelfNorm_for_Generalization_Under_Distribution_Shifts_ICCV_2021_paper.html>

[^6]: Zhang, H. et al. (2021). “VarifocalNet: An IoU-Aware Dense Object Detector.” _CVPR_. <https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html>
