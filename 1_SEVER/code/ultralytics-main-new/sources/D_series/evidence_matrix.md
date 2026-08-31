# CitrusD 证据矩阵

| 论文 | 经验证的原始主张 | 本项目采用的最小思想 | 没有宣称的内容 |
|---|---|---|---|
| Gated-SCNN, ICCV 2019 | 独立浅层 shape stream；高层语义门控可去除形状噪声，并改善薄/小物体与边界 | P2 形状流与 P3/P4/P5 语义门控分离 | 不声称其 Cityscapes 增益会迁移到柑橘 |
| PiDiNet, ICCV 2021 | PDC 显式编码局部像素差分，轻量边缘检测有效 | 深度可分 PDC 作为形状流更新；普通卷积 D02 为因果对照 | 不把边缘检测 F-score 当成 Mask AP |
| PIDNet, CVPR 2023 | 直接混合上下文和细节会使细节被淹没；边界引导的选择性融合有效 | 语义只控制结构更新，并用 agreement/boundary gate 注入 P3 | 不复制 PID 三分支全网，也不宣称实时速度 |
| Lite-HRNet, CVPR 2021 | 小模型可持续维持高分辨率表示，并进行高效跨分辨率交互 | 保留窄 stride-4 流，而非增加稠密 P2 检测头 | 不使用完整 HRNet 网格，避免算力膨胀 |
| QueryDet, CVPR 2022 | 高分辨率有利于小目标，但稠密 P2 head 极昂贵；候选查询可减少无效计算 | 只在训练时使用 tiny-centre 查询监督，推理仍为 P3-P5 | 当前实现不等于稀疏卷积 QueryDet，不能声称其 3x 加速 |
| FreqFusion, TPAMI 2024 | 自适应低/高通与重采样改善融合的一致性和边界 | 暂未采用；保留为 D 核心验证后的独立颈部候选 | 不把模块库代码直接复制进 D，也不预报收益 |

主要原文：

- https://openaccess.thecvf.com/content_ICCV_2019/html/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.html
- https://openaccess.thecvf.com/content/ICCV2021/html/Su_Pixel_Difference_Networks_for_Efficient_Edge_Detection_ICCV_2021_paper.html
- https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html
- https://openaccess.thecvf.com/content/CVPR2021/papers/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.pdf
- https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html
- https://doi.org/10.1109/TPAMI.2024.3449959

