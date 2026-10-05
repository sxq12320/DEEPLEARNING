# J 系列文献与开源取舍记录

检索日期：2026-10-01。检索源包括 CVF Open Access、arXiv、ACL Anthology，并核对 OpenReview/作者 GitHub。属于**面向工程问题的范围检索**，不是穷尽式系统综述，不虚构检索命中总数或 PRISMA 排除数量。

检索围绕四个问题：tiny citrus 的分辨率损失；同色背景中的对比证据；轻量且非反复金字塔的融合/掩膜解码；实例、可见边界与背景拒绝的监督。LLM/NLP 借鉴限于可解释计算原则，不移植大语言模型或 token 推理成本。以下“相关”不等于“在柑橘任务已有效”。

## 1. 核心方法与代码核对

| 文献/状态 | 开源与本次阅读 | 借鉴及重要限制 |
|---|---|---|
| [FasterNet，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html) | [作者仓库](https://github.com/JierunChen/FasterNet)；本地 FasterNet/models/fasternet.py 的 Partial_conv3、MLPBlock | 部分通道做空间混合；不照搬完整主干，也不由 FLOPs 推出 FPS；J 加入独立乘性门控 |
| [GLU Variants Improve Transformer，2020 预印本](https://arxiv.org/abs/2002.05202) | 方法公式阅读，未作为新仓库移植 | 借鉴 value/gate 乘性选择；J 的 sigmoid gate 不等于 SwiGLU，未声称新门控公式 |
| [Differential Transformer，ICLR 2025](https://arxiv.org/abs/2410.05258) | [Microsoft unilm](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)；本文思路核对，未安装/执行 | 两个 attention 图之差；语言建模证据不证明果叶混淆改善；J 固定 lambda=1 便于条件性推导 |
| [Linear Differential Vision Transformer，2025](https://arxiv.org/abs/2511.00833) | [作者 LinearDiff](https://github.com/LeapLabTHU/LinearDiff)；新增下载，追读 VisualContrastAttention | 小记忆与视觉对比；J 不复现两阶段、位置token、RMSNorm；原代码拼接 softmax 与 J 分别 softmax 有区别 |
| [PIDNet，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) | 本地 PIDNet/models/model_utils.py 的语义/细节/边界相关结构 | 任务分路、语义与边界证据；不是把 PID 控制器或稳定性证明直接移入训练 |
| [FreqFusion，2024 预印本及后续期刊工作](https://arxiv.org/abs/2408.12879) | [作者仓库](https://github.com/Linwei-Chen/FreqFusion)；本地实现中采样/重组路径核对 | 跨尺度融合不应淹没边界；本次不采用重 CARAFE/unfold 或动态采样依赖 |
| [LSNet，CVPR 2025](https://arxiv.org/abs/2503.23135) | [作者仓库](https://github.com/THU-MIG/lsnet)及本地代码 | 广域感知与局部聚合分工；不加入它的定制内核或把局部边缘都视为果实 |
| [D-FINE-seg，2026 预印本](https://arxiv.org/abs/2602.23043) | [作者仓库](https://github.com/ArgoHA/D-FINE-seg)；新增下载，阅读 dfine_seg/model/dfine_criterion.py 的 cropped BCE/Dice | 使用实例区域内归一化监督作为参考；不搬它的完整 detector，不将预印本标成顶会论文 |
| [BDNet，CVPR 2026](https://openaccess.thecvf.com/content/CVPR2026/html/Guan_BDNetBio-Inspired_Dual-Backbone_Small_Object_Detection_Network_CVPR_2026_paper.html) | 论文方法线索；本次未找到并验证官方代码 | 不对称细节/语义角色有借鉴价值；遥感检测不是绿色果实可见实例分割，不直接照抄双重主干 |

核心 J 模块为独立、带来源说明的任务适配。现有训练/切片/评估功能按原实现复用，属于工程依赖，不包装成新贡献。模块仓库里“CVPR”文件名不构成录用、准确性或许可证证据，应回到作者原论文/原仓库核实。

## 2. 拓宽思路后的候选与排除理由

| 候选文献（原始来源） | 本次深度 | 取舍 |
|---|---|---|
| [Gated Delta Networks，ICLR 2025](https://arxiv.org/abs/2412.06464) | 方法概要；[官方代码](https://github.com/NVlabs/GatedDeltaNet)核对可用性 | 学习有限状态/门控预算原则，未直接迁移递推内核；不需要 Mamba |
| [DINT Transformer，EMNLP 2025](https://aclanthology.org/2025.emnlp-main.495/) | ACL 摘要与元数据 | 差分与信息保持的权衡；未把名称中的 integral 当作 PID 积分稳定性证据 |
| [Deep Kernel Fusion for Transformers，ACL 2026](https://aclanthology.org/2026.acl-short.15/) | 摘要与元数据 | 算子融合/内存访问提醒；未安装定制编译内核，实际速度仍需 GPU 实测 |
| [SparseInst，CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Cheng_Sparse_Instance_Activation_for_Real-Time_Instance_Segmentation_CVPR_2022_paper.pdf) | 实例表示/解码思路；已有本地仓库 | 借鉴共享掩膜表示与任务分离；本次保留成熟候选接口，不在同批次同时改 Hungarian matching |
| [FastInst，CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.html) | 架构与本地代码候选 | 无框 query 路线值得后续对比；尚无本任务匹配协议结果，暂不作为 J 主线必然更优的依据 |
| [PointRend，CVPR 2020](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) | 边界精化机制 | 不确定点恢复需候选先检出；不能以边界后处理替代漏检修复，本次不加入额外迭代成本 |
| [VarifocalNet，CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_VarifocalNet_An_IoU-Aware_Dense_Object_Detector_CVPR_2021_paper.html) | 质量分数监督原则 | 历史质量头值得保持对照，但本次不同时改变所有分配/评分因子，避免无法归因 |
| [NWD，2021 预印本](https://arxiv.org/abs/2110.13389)及[2022版本](https://arxiv.org/abs/2206.13996) | tiny box 误差/分配机制 | 框损失鲁棒性不等于可见掩膜精度；J 先固定 TAL/CIoU/DFL，待确认候选分配瓶颈再单独测 |
| [AdamW，ICLR 2019](https://openreview.net/pdf?id=Bkg6RiCqY7) | 解耦衰减方法 | 采用成熟优化器，不将 SGD/cosine 对照包装成新优化算法 |
| [Foveated Instance Segmentation，CVPR 2025](https://openaccess.thecvf.com/content/CVPR2025/html/Zeng_Foveated_Instance_Segmentation_CVPR_2025_paper.html) | 论文摘要/机制筛查 | 提示区域预算分配，但依赖注视位置，不等于无提示全图自动检测；不直接用人工注视作为推理条件 |
| [HyPiDecoder，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Zhou_HyPiDecoder_Hybrid_Pixel_Decoder_for_Efficient_Segmentation_and_Detection_ICCV_2025_paper.html) | 官方元数据/摘要，全文访问受限 | 提醒 decoder 不应被忽略；不声称已完整复现/深读 |
| [MOBIUS，ICCV 2025](https://openaccess.thecvf.com/content/ICCV2025/html/Segu_MOBIUS_Big-to-Mobile_Universal_Instance_Segmentation_via_Multi-modal_Bottleneck_Fusion_and_ICCV_2025_paper.html) | 摘要/方法筛查 | teacher/压缩值得另立蒸馏对照，但需要新训练协议，不加入本次 scratch 主线 |
| [Is There Really a Camouflaged Object?，2026 预印本](https://arxiv.org/abs/2608.11135) | 摘要/任务设置 | 提醒现实任务需拒绝背景，不只对有目标图片分割；本次不新增伪造叶片类别或人工删标签 |
| [Discover, Segment, Select，2026 预印本](https://arxiv.org/abs/2602.19944) | 摘要/候选筛查 | 渐进式搜索与选择仅作机制参考，不能让第一阶段漏检成为不可恢复的硬筛选 |
| [ViCoSAM3，2026 预印本](https://arxiv.org/abs/2609.15418) | 摘要筛查 | 大模型伪装场景先验不能直接等价于轻量部署；本次不增加 SAM3 运行依赖 |
| [GTR，2026 预印本](https://arxiv.org/abs/2609.26590) | 摘要/代码需求筛查 | 递推视觉与预算有启发，但定制内核、重复更新成本和未匹配数据使它不适合直接上服务器 |

共筛查 25 个主题条目，其中 NWD 两版本合并为一条。不是宣称 25 篇都逐页精读，更不是宣称 2026 预印本都已经顶会验证。核心实施依据优先采用有可核对实现、算子简单、与本数据难点直接关联的方法。

## 3. 对第三方模块合集的实际审阅

此外追读桌面 `Plug-play-modules-main` 中的 StarNet、CGAFusion、MSAA，以及已有 FreqFusion 文件；并回查原文/作者仓库，不因合集命名认定可直接复用：

- [Rewrite the Stars / StarNet](https://arxiv.org/abs/2403.19967) 提供乘性交互思路。本地副本的 `_init_weights` 中 `nn.Linear or nn.Conv2d` 在 Python 中只得到第一个类，不能按注释理解为同时匹配两类；未把这份副本直接作为 J 的初始化实现。
- [DEA-Net 作者仓库](https://github.com/cecret3350/DEA-Net) 的内容引导融合值得参考，但所读合集版本 PixelAttention 内 sigmoid 后，CGAFusion 又 sigmoid，最终权重落在约 (0.5,0.731)；加上 initial=x+y，整个混合也不是规范凸组合。此处只描述**本地副本**，未判定原作者所有版本都如此。J 用单次 softmax 和明确归一化做自己的源竞争对照。
- [CM-UNet](https://arxiv.org/abs/2405.10530) 的 MSAA 学习多尺度证据，但本地副本有 3/5/7 并行卷积、前置 concat，并声明未在 forward 使用的 down_2。本文任务的低延迟目标不支持不经预算测试直接照搬；J 先投影再对齐，按目标尺度做一次融合，也不引入 Mamba。

这三项是补充审阅，不计入前述25条检索主题。没有修改或执行桌面模块合集里的代码。它们的优点是启发结构设计，不是作为未经检查的“即插即用涨点保证”。

## 4. 下载与来源可追溯性

新增仓库均为只读研究副本，未覆盖已有 github 文件：

- `C:/Users/33836/Desktop/github/J_20261001_LinearDiff`，commit `3fb5ee100cfdd5c6e482cf1bade58a61dc24e3d1`。仓库根未发现独立 LICENSE 文件，不将其假定为 MIT；本次没有直接复制它的模块源码进 J。
- `C:/Users/33836/Desktop/github/J_20261001_D-FINE-seg`，commit `0c7537fe01455062711f87cbecb04bed7b3a9ed5`。仓库 LICENSE 为 Apache-2.0。本次参考 cropped mask loss 机制，未移植完整网络或依赖。

已有桌面仓库用于源代码阅读，不会自动安装或执行其中脚本。J 仍只需本项目已有 PyTorch/Ultralytics 环境。开源使用须保留来源与许可证义务；论文需要清楚区分原作者思想、复用实现和本任务独立改写，不能以更换名称宣称原创。

## 5. 检索轨迹与未解决问题

代表查询：`tiny object instance segmentation high resolution decoder`、`competitive multi-scale fusion efficient segmentation`、`differential attention visual contrast pooled tokens`、`2026 camouflaged object presence segmentation`、`LLM gated delta networks differential transformer`、`FasterNet memory access FLOPS`。

工具原始摘要中偶发不相关搜索结果已排除；例如不能把不相关的流体力学 arXiv 编号当作 LinearDiff 引文。没有用博主标题或模块文件名替代原始文献证据。

仍需实验证明：1）主干压窄后是否丢失果实语义；2）高分辨率是否同时提高叶枝误报；3）差分公共成分抵消会不会抹掉弱小果实；4）soft scene 支路在稀疏前景下是否学成过低响应；5）边界监督是否改善真实 merge/split，而非仅增大 loss。筛选应允许结果推翻当前假设，不预先决定综合模型必胜。
