# SAGE-v4 研究来源与实现核对

## 检索边界

2026-09-03：本地没有可用的 parallel-cli / Parallel / OpenRouter 凭据，因此用网页检索与作者仓库替代。
只以论文原文页面和作者开源代码作为技术依据，不把模块汇总库的文件名当成出版证明。CVF 页面部分请求返回 403，
由可用的论文索引摘要、arXiv 和官方 GitHub 交叉核对。没有声称复现论文原模型的实验精度。

## 已阅读和迁移的思想

| 来源 | 原论文结论或范围 | 本任务取舍 | 阅读的开源实现 |
|---|---|---|---|
| [PIDNet，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) | 受 PID 启发的实时语义分割，区分细节、语义与边界 | 借鉴分工与相邻尺度受限融合；不声称实例分割已验证，不照搬控制稳定性 | [作者代码](https://github.com/XuJiacong/PIDNet)，本地 `models/model_utils.py` 的 PagFM、Light_Bag、Bag |
| [BMask R-CNN，ECCV 2020](https://arxiv.org/abs/2007.08921) | 边界与掩膜相互学习以改善实例定位 | 使用独立边界辅助监督；不用完整 RoI 检测器 | [作者代码](https://github.com/hustvl/BMaskR-CNN)，`projects/BMaskR-CNN/bmaskrcnn/mask_head.py` |
| [ReZero，UAI 2021](https://proceedings.mlr.press/v161/bachlechner21a.html) | 零初始化残差门有利于深网络信号传播，也用于语言建模 | 使用相近的小幅、逐通道残差；本代码初始化 0.01，不是精确 ReZero 复现 | [作者代码](https://github.com/majumderb/rezero)，`rezero/transformer/rztx.py` |
| [Gated CNN，ICML 2017](https://proceedings.mlr.press/v70/dauphin17a.html) | 用卷积与门控实现语言建模、避免逐 token 递归 | 借鉴值分支×门分支，避免为“跨领域创新”引入大语言模型本体 | 原理论通过 MambaOut 作者实现交叉核对 |
| [MambaOut，CVPR 2025](https://arxiv.org/abs/2405.07992) | 去掉 SSM 的 gated CNN 在分类有效，但不能概括为所有检测/分割都不需要 SSM | 只移植门控卷积思路，P4 可选替换；不是采用 Mamba | [作者代码](https://github.com/yuweihao/MambaOut)，`models/mambaout.py` 的 GatedCNNBlock |
| [ConDSeg，AAAI 2025](https://arxiv.org/abs/2412.08345) | 对比驱动的前背景表征和聚合 | 不直接移植动态 unfold/fold 算子，避免重现 Light 延迟 | [作者代码](https://github.com/Mengqi-Lei/ConDSeg)，汇总库 ContrastDrivenFeatureAggregation 对照 |

本次新增下载两个官方仓库到桌面 `github`，未运行其安装器或训练代码：

- `C:/Users/33836/Desktop/github/MambaOut`，commit `9f2f2343eb0f99f2cf3ba6b92290b5a81be2bad1`。
- `C:/Users/33836/Desktop/github/rezero`，commit `e2c94a825c5564217e8cf4d75a28d59cab1d7029`。

其他作者仓库已在桌面存在，直接阅读，未覆盖。新网络代码为独立实现，没有把存在许可证疑问的汇总库源码整段复制进项目。

## 不采用的候选

阅读汇总库 GatedCNNBlock、Moga Block、ContrastDrivenFeatureAggregation、RCM 等候选：多序 DW 卷积、
轴向上下文和频率增强均有原场景价值，但历史 SAGE20、G03 等没有提供足够的任务增益支持。本轮不叠加这些分支。
应先通过实例尺度、可见掩膜凹度、邻居间隙分层评价确定具体收益，再决定是否引入。
