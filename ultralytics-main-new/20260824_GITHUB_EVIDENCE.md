# Official-code evidence manifest

All repositories below were shallow-cloned to `C:\Users\33836\Desktop\github`. They are research evidence and
implementation references, not runtime dependencies of CitrusTopo-Seg.

| Repository | Official URL | Pinned commit | License | What was actually used |
|---|---|---|---|---|
| SFM | https://github.com/Linwei-Chen/SFM | `b08e63ee49a5e5d4e304ada368142a9f5972113a` | MIT | Audited ARS saliency-density grid resampling; established that the older Haar block was not an official SFM port |
| SCSegamba | https://github.com/Karl1109/SCSegamba | `cc74c4606f03bc308b1ef1c07ef6229f9123c519` | Apache-2.0 | Audited SAVSS/GBC/PAF; excluded from the final runtime because the official path requires Mamba extensions |
| BMask R-CNN | https://github.com/hustvl/BMaskR-CNN | `c74b0bd3ed47bf4aaa5c211c4e31eddc78fdc636` | Apache-2.0 | Mask-to-boundary and boundary-to-mask mutual feature fusion; BCE plus Dice boundary supervision |
| RefineMask | https://github.com/zhanggang001/RefineMask | `633ed2be1b36b3f3c798be484b6a5117004faab8` | Apache-2.0 | Evidence for concentrating later supervision on uncertain predicted/target boundary bands |
| QueryDet | https://github.com/ChenhongyiYang/QueryDet-PyTorch | `feebf218d53d59ba054132dfa6ef84159f793967` | MIT | Sparse small-object candidate/query heatmap concept; adapted as a lightweight P2 auxiliary query, not claimed as an exact port |
| SegFix / OpenSeg | https://github.com/openseg-group/openseg.pytorch | `aefc75517b09068d7131a69420bc5f66cb41f0ee` | MIT | Audited boundary and direction supervision; retained boundary supervision but omitted direction classes for simplicity |
| NWD | https://github.com/jwwangchn/NWD | `9775ac2d24354895fbdf5b8db1f1600c01c54e33` | Apache-2.0 | Verified the normalized Wasserstein formula; kept out of the full recipe after harmful local combined-loss evidence |
| Lite-HRNet | https://github.com/HRNet/Lite-HRNet | `7b9049d264fa40402a27d1f175deff3b46a6b91b` | Apache-2.0 | Persistent high-resolution and cross-resolution weighting evidence; used to justify a shallow P2 path rather than replacing YOLO wholesale |
| LSKA | https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention | `bb2a8d2cfd7e9ff48a34306197249d90153c8d4e` | Apache-2.0 | Official factorized LSKA-23 operator adapted at P5 with a zero-initialized residual |

The final implementation is a disclosed synthesis. `SPPFLSKAResidual` closely adapts the official LSKA operator;
the topology head and losses are task-specific adaptations grounded in BMask R-CNN, QueryDet, and RefineMask rather
than line-for-line ports of those frameworks.
