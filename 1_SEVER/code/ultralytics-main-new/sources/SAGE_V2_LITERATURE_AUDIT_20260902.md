# SAGE-v2 文献与开源代码审计

_检索日期：2026-09-02 · 目标：未成熟柑橘轻量实例分割_

---

## 📋 检索问题

本次检索不以“找到更多模块”为目标，而回答四个结构问题：如何在不维持昂贵 P2 检测塔的前提下保护
超小目标；如何用高层上下文抑制绿色叶片假阳性；如何保留条带遮挡产生的凹形可见边界；如何分开相邻
实例同时控制训练延迟。外部证据只有在能映射到上述问题、存在论文或官方代码、并符合当前算子与预训练
约束时才进入实现。

用户提供的 Awesome 仓库是低层视觉专题索引，主要覆盖图像恢复、超分辨率、去噪与增强，并非 CVPR
检测/实例分割论文全集。[^1] 因此检索同时扩展到 CVF Open Access、arXiv 与作者官方仓库。

---

## 🔍 检索与复现范围

| 来源 | 用途 | 审计方式 |
| --- | --- | --- |
| CVF Open Access | 论文原文 | 核对任务、结构与消融 |
| arXiv | 补充论文版本 | 核对方法细节 |
| 作者 GitHub | 实现原型 | 检查算子、依赖、热路径 |
| 本地历史结果 | 负迁移证据 | 固定协议下比较精度和速度 |

搜索词覆盖 `tiny object detection`、`camouflaged instance segmentation`、`boundary refinement`、
`feature pyramid alignment`、`real-time segmentation backbone`、`sparse high-resolution query`、
`touching instance separation` 和 `ranking loss instance segmentation`。

---

## 📚 采用矩阵

| 来源 | 原论文核心 | 代码观察 | 采用内容 | 实现位置 |
| --- | --- | --- | --- | --- |
| RepViT[^2] | 移动 CNN 的 token/channel mixer 与结构重参数化 | 官方实现使用低成本局部混合 | P4/P5 低分辨率残差 | `C3k2SAGE` |
| PIDNet[^3] | 细节、上下文与边界三分支；PagFM 选择融合 | 相似度图逐像素调制语义 | 相邻尺度相似度融合 | `_SAGESimilarityFuse` |
| QueryDet[^4] | 候选查询避免全图高分辨率计算 | 查询监督与高分辨率塔解耦 | 候选热图辅助监督 | `SegmentCitrusSAGEV2` |
| Mask Transfiner[^5] | 聚焦错误/不确定区域细化掩膜 | 原方案含专用细化解码器 | 只迁移边界错误监督 | shared topology loss |
| RefineMask[^6] | 多阶段利用细粒度特征修边 | 对边界质量直接建模 | P2 几何进入 P3 | `CitrusSAGEPyramid` |
| FaPN[^7] | 先对齐后融合跨尺度特征 | 官方路径依赖可变形卷积 | 迁移对齐原则 | regular-conv correction |
| BCNet[^8] | 显式建模遮挡层关系 | 面向 amodal/occlusion | 迁移 separator 状态 | topology channel 3 |
| Rank & Sort[^9] | 分类分数与定位质量排序 | 需重写排序目标 | 暂留独立损失实验 | 未混入首轮结构 |

SAGE-v2 对这些工作做的是受约束的思想迁移，而不是声称复现原方法。例如普通卷积相邻尺度校正不能命名为
FaPN，四状态可见拓扑监督也不能声称实现 BCNet 的 amodal 双层分割。

---

## ❌ 拒绝矩阵

| 候选 | 拒绝直接移植的原因 | 本任务证据 |
| --- | --- | --- |
| FreqFusion[^10] | 动态核、CARAFE/重采样路径增加热路径复杂度 | G03 Mask mAP50-95 为 0.65631 且训练约 4.43 小时 |
| FaPN DCN | 额外编译依赖与不规则访存 | 服务器环境与轻量目标不匹配 |
| CAMixerSR[^11] | 窗口划分、动态路由与变形/重排更适合超分辨率 | Light 系列已经暴露“参数少但训练慢”风险 |
| 完整 P2 prediction head | 640 分辨率下持续高分辨率计算昂贵 | G02 约 2.2 倍训练时间且收益很小 |
| Mamba/VSS | 需要额外环境与扫描核，用户明确不安装 Mamba | 不满足部署合同 |
| 全局 self-attention | P2/P3 token 数过大 | 不满足 nano 与速度目标 |
| 多个独立辅助头 | 重复计算且梯度目标可能冲突 | 改为一张共享四状态图 |

拒绝不表示这些方法在原论文无效，只表示它们不满足当前数据、硬件、已有实验和标准 YOLO YAML 入口的
联合约束。

---

## 📦 本地开源仓库快照

| 仓库 | 本地目录 | 提交 |
| --- | --- | --- |
| Awesome low-level vision | `Desktop/github/Awesome-CVPR...Low-Level-Vision` | `f5a77650035ee47058aebbcdeccc181def67efde` |
| CAMixerSR | `Desktop/github/CAMixerSR` | `d690b12bb2a0a185dc0049538b377d569f520497` |
| FaPN | `Desktop/github/FaPN` | `4d400719d3f2b9a4dc38c2132b9984be66219719` |
| SHViT | `Desktop/github/SHViT` | `6a729ccf18e0b941714b529638bd3d9bacebcef0` |
| RepViT | `Desktop/github/RepViT` | 本地已有 |
| PIDNet | `Desktop/github/PIDNet` | 本地已有 |
| QueryDet | `Desktop/github/QueryDet-PyTorch` | 本地已有 |
| Mask Transfiner | `Desktop/github/MaskTransfiner` | 本地已有 |
| RefineMask | `Desktop/github/RefineMask` | 本地已有 |
| FreqFusion | `Desktop/github/FreqFusion` | 本地已有 |
| Rank & Sort | `Desktop/github/RankSortLoss` | 本地已有 |
| DCNet | `Desktop/github/DCNet` | 本地已有 |

仓库仅作为实现审计材料，未把第三方源文件直接复制进 Ultralytics fork。当前实现使用项目已有的 PyTorch
与 Ultralytics 基础算子，避免引入新的编译扩展和许可证不清晰的代码片段。

---

## 🎯 对 SAGE-v2 的约束结论

1. 主干只改 P4/P5，并保留官方 `C3k2` 参数路径，降低小数据集随机初始化风险
2. P2 只传递局部几何证据，不新增预测尺度
3. 跨尺度只做相邻融合，避免 P5 直接跃迁到高分辨率造成语义/空间错位
4. 融合门必须接受边界与 separator 的任务监督
5. 保留官方 PAN 的残差版本是主候选，删除 PAN 的版本只能作为激进结构对照
6. 任何新方法必须先通过标准 YAML 构建、前向、反向、GFLOPs 与目标 GPU step-time 门禁

---

## 🔗 参考文献

[^1]: Kobaayyy. "Awesome CVPR Low-Level Vision." _GitHub_. https://github.com/Kobaayyy/Awesome-CVPR2026-CVPR2025-CVPR2024-CVPR2021-CVPR2020-Low-Level-Vision

[^2]: Wang et al. (2024). "RepViT: Revisiting Mobile CNN From ViT Perspective." _CVPR_. https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html

[^3]: Xu et al. (2023). "PIDNet: A Real-Time Semantic Segmentation Network Inspired by PID Controllers." _CVPR_. https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html

[^4]: Yang et al. (2022). "QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection." _CVPR_. https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html

[^5]: Ke et al. (2022). "Mask Transfiner for High-Quality Instance Segmentation." _CVPR_. https://openaccess.thecvf.com/content/CVPR2022/html/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.html

[^6]: Zhang et al. (2021). "RefineMask: Towards High-Quality Instance Segmentation With Fine-Grained Features." _CVPR_. https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html

[^7]: Huang et al. (2021). "FaPN: Feature-Aligned Pyramid Network for Dense Image Prediction." _ICCV_. https://openaccess.thecvf.com/content/ICCV2021/html/Huang_FaPN_Feature-Aligned_Pyramid_Network_for_Dense_Image_Prediction_ICCV_2021_paper.html

[^8]: Ke et al. (2021). "Deep Occlusion-Aware Instance Segmentation With Overlapping BiLayers." _CVPR_. https://openaccess.thecvf.com/content/CVPR2021/html/Ke_Deep_Occlusion-Aware_Instance_Segmentation_With_Overlapping_BiLayers_CVPR_2021_paper.html

[^9]: Oksuz et al. (2021). "Rank & Sort Loss for Object Detection and Instance Segmentation." _ICCV_. https://openaccess.thecvf.com/content/ICCV2021/html/Oksuz_Rank__Sort_Loss_for_Object_Detection_and_Instance_Segmentation_ICCV_2021_paper.html

[^10]: Chen et al. (2024). "FreqFusion: Frequency-Aware Feature Fusion for Dense Image Prediction." _arXiv_. https://arxiv.org/abs/2408.12879

[^11]: Wang et al. (2024). "CAMixerSR: Only Details Need More Attention." _CVPR_. https://openaccess.thecvf.com/content/CVPR2024/papers/Wang_CAMixerSR_Only_Details_Need_More_Attention_CVPR_2024_paper.pdf

---

_Last updated: 2026-09-02_
