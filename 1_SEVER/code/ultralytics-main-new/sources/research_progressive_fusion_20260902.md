# Progressive feature-fusion evidence for immature-citrus instance segmentation

Search date: 2026-09-02

## Research question

How should the neck depart from repeated FPN/PAN top-down and bottom-up fusion while preserving tiny-fruit detail,
suppressing green-leaf camouflage, retaining concave visible-mask boundaries, and avoiding the latency and transfer
failures observed in the local Light/G0830 experiments?

## Primary papers and official code

| Work | Primary paper | Official code | Design evidence retained |
|---|---|---|---|
| AFPN | https://arxiv.org/abs/2306.15988 | https://github.com/gyyang23/AFPN | Fuse adjacent low-level features first, then asymptotically incorporate higher levels to reduce large semantic gaps; spatially adaptive fusion resolves location-wise conflicts. |
| Gold-YOLO | https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html | https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO | Global gather-and-distribute can replace repeated local pairwise exchange, but a citrus nano model should restrict this mechanism to the small-object canvas rather than reproduce its full heavy implementation. |
| FaPN | https://openaccess.thecvf.com/content/ICCV2021/html/Huang_FaPN_Feature-Aligned_Pyramid_Network_for_Dense_Image_Prediction_ICCV_2021_paper.html | https://github.com/ShihuaHuang95/FaPN-full | Direct addition of resized high-level and local features produces contextual misalignment, especially at boundaries; explicit alignment and low-level feature selection improve dense prediction. |
| FreqFusion | https://arxiv.org/abs/2408.12879 | https://github.com/Linwei-Chen/FreqFusion | Direct fusion can disturb within-object high frequencies and blur displaced boundaries; adaptive low-pass, resampling, and high-pass paths target semantic consistency and boundary detail separately. |
| ASFF | https://arxiv.org/abs/1911.09516 | https://github.com/ruinmessi/ASFF | Pixel-wise scale weights are useful after features are aligned, because conflicting scale evidence varies by location. |
| EfficientDet / BiFPN | https://openaccess.thecvf.com/content_CVPR_2020/papers/Tan_EfficientDet_Scalable_and_Efficient_Object_Detection_CVPR_2020_paper.pdf | https://github.com/google/automl/tree/master/efficientdet | Weighted bidirectional fusion is a valid efficiency baseline, but repeated fusion is not automatically optimal for a small citrus dataset or realized GPU latency. |

## Local-code audit

- `CitrusLightAFPN` already implements adjacent progressive low-to-high gathering followed by high-to-low semantic
  distribution. Therefore merely proposing an AFPN is not novel for this repository.
- Its P2 route uses projection plus average pooling, which can attenuate tiny/high-frequency evidence before fusion.
- Its full neck is newly initialized, reducing direct reuse of the pretrained YOLO11 PAN path.
- Its near-identity source injection is deliberately weak at initialization; a semantic contribution must pass several
  such gates before reaching P3.
- `CitrusORCHIDNeck` already replaces recurrent PAN with a single P3 canvas, but it resizes P4 directly, globally pools
  P5, and average-pools P2. It does not perform progressive boundary alignment.
- The G0830 frequency-neck run finished below its official control, so frequency operators must be used at a single
  demonstrably necessary transition rather than across the whole pyramid.

## Recommended synthesis: SAGE-Fuse

**SAGE-Fuse: Semantic-Aligned Geometry-Evidence Fusion** is a task-decoupled, P3-centred fusion graph rather than
another attention stack.

1. Preserve the official YOLO11 backbone and PAN tensors as a pretrained identity path.
2. Relay semantic evidence progressively from P5 to P4 and from P4 to P3. Align only the P4-to-P3 transition, where
   boundary displacement most directly affects small masks.
3. Rearrange P2 to P3 with PixelUnshuffle/space-to-depth instead of average pooling, retaining four sub-pixel samples
   in the channel dimension.
4. Compute a low-cost agreement gate between the aligned semantic evidence and P2 geometry evidence. The gate admits
   detail only where fruit-level semantics and local shape/edge evidence agree, rather than amplifying all green leaf
   texture.
5. Inject the fused evidence as a zero/small-initialized residual into the P3 mask-prototype path. Keep box/class
   prediction on the native PAN P3/P4/P5 path during the first ablation stage.

This architecture is materially different from both the current Light AFPN and ORCHID single-canvas neck: it keeps
the pretrained PAN, uses a progressive aligned semantic relay, preserves P2 samples, and routes the result by task.

## Controlled ablations

| ID | Change from the same native control | Causal question |
|---|---|---|
| F0 | Exact YOLO11n-seg PAN | Paired control |
| F1 | Add P5-to-P4-to-P3 semantic relay as a mask-only residual | Does progressive semantics improve camouflage discrimination? |
| F2 | F1 plus one P4-to-P3 alignment unit | Is boundary/scale misalignment the bottleneck? |
| F3 | F2 plus lossless P2-to-P3 rearrangement | Does retained geometry improve tiny masks and Boundary F1? |
| F4 | F3 plus semantic-geometry agreement gate | Can leaf-texture false positives be suppressed without losing recall? |
| F5 | Optional mild P3 detection residual | Does the evidence help candidate recall, or should it remain mask-only? |

Promote models only after build/forward/backward/FLOPs checks, a 1-3 epoch smoke run, and a paired 50-epoch screen.
Report mask AP50-95/AP50, precision, recall, AP-tiny/AP-small, Boundary F1, split/merge errors, challenge-subset scores,
parameters, GFLOPs, and measured latency. Keep all optimizer, initialization, augmentation, AMP, image size, seed, and
data-split settings identical.

