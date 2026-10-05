# J_V2 文献—代码—实验决策记录

检索日期：2026-10-03。这是面向工程决策的主题综述，**不是穷尽式系统综述或复现报告**。
文献里的检测 AP、语义 IoU、遥感/红外结果，不能当成我们 RGB 可见实例分割的预期涨点。

## 检索与证据等级

通过 CVF Open Access、arXiv、OpenReview 的第一方页面/论文，以及作者 GitHub 核验。
`parallel-cli` 在本机不可用，改用网络检索，没有安装付费服务。
完成18组关键词检索及随后第一方页面核查；没有记录所有搜索引擎返回条数，不伪造 PRISMA 数字。
以下是归一化检索主题，不是逐字的工具调用转录：

| 轮次 | 查询主题 | 目的 |
|---|---|---|
| 1，3组 | CVPR2026 BDNet；CVPR2025轻量融合/小目标；2026实例分割 | 近期架构与目标发现 |
| 2，4组 | RepViT；RepNeXt；ESOD；2026轻量neck | 实际部署算子与分辨率预算 |
| 3，4组 | Differential Transformer；ESOD；QueryDet；CVPR2026 tiny segmentation | NLP迁移与稀疏搜索的条件 |
| 4，4组 | FasterNet；Rewrite the Stars；PIDNet；Rank & Sort | 延迟、乘性特征、控制启发与排序损失 |
| 5，3组 | CVPR2026 lightweight segmentation；2026 tiny detection；Mask Scoring | 最新补充与掩膜质量排序 |

采用：与发现/边界/成本有关、原始来源可核验、能形成明确对照的思想。
不直接采用：必须红外/事件流、依赖大规模预训练、没有匹配数据证据的重注意力、不可验证的涨点宣称。
若页面403但检索返回官方摘要，标记摘要级，不宣称读完全文。

## 主题一：小目标首先是候选发现与空间预算问题

| 来源及身份 | 实际核验层级 | 对本任务的决策 |
|---|---|---|
| [QueryDet，CVPR2022](https://arxiv.org/abs/2103.09136)；[作者代码](https://github.com/ChenhongyiYang/QueryDet-PyTorch) | 深化审查指定全文§3.3/4.2–4.4、Tables1/3/4；未复现 | Table3纯HR掉点，监督重平衡后才改善；P2仅为覆盖假设。作者稀疏执行不是我们dense P2或裁图多前向。 |
| [ESOD，TIP2024](https://arxiv.org/abs/2407.16424)；[作者代码](https://github.com/alibaba/esod) | 第一方摘要/仓库 | 思路是特征共享和区域搜索；我们的旧J2图像裁剪多次前向**不是ESOD复现**。共享特征稀疏执行留作后续，需要真实硬件收益。 |
| [TinyFormer，2026-05-24预印本](https://arxiv.org/abs/2605.25046)；[作者代码](https://github.com/mmpmmpmmpjosh/TinyFormer) | 第一方摘要、README、`dinov3_adapter.py`的SSA/返回尺度、`lite_encoder.py` | 浅层捷径补偿粗token空间丢失值得借鉴；不照搬ViT/DETR、Objects365预训练和其TensorRT延迟。J36保留细层候选与短融合，不声称实现PBM/SSA。 |
| [DM-EFS，ICCV2025](https://openaccess.thecvf.com/content/ICCV2025/html/Sharma_DM-EFS_Dynamically_Multiplexed_Expanded_Features_Set_Form_for_Robust_and_ICCV_2025_paper.html) | 官方摘要级 | 更丰富细层集合也有空间计算成本；应同时报告成本，不能把P2大卷积当免费召回。J32/J40是显式昂贵对照。 |
| [CSCWD，2026-09-24预印本](https://arxiv.org/abs/2609.30395) | 第一方摘要级；尚无本地复现 | 跨尺度教师监督可以不改变学生推理图，是精度恢复候选。**未实现**：当前没有在同一正式train划分上确定的教师；也不是当前J36训练的一部分。 |

历史同权重切片改善tiny匹配但耗时增加，这是本任务的数据支持；上述方法给出实现选项，而非覆盖掉该反证。

## 主题二：不应把“全部高频”误当成“全部果实细节”

| 来源及身份 | 实际核验层级 | 对本任务的决策 |
|---|---|---|
| [SET，CVPR2025](https://openaccess.thecvf.com/content/CVPR2025/html/Sun_SET_Spectral_Enhancement_for_Tiny_Object_Detection_CVPR_2025_paper.html) | 官方摘要/论文检索片段；本地README及`mmdet/models/detectors/fcos_set.py`前130行 | 背景噪声也含高频，增强所有边缘可能增强叶脉。源码含额外前向/梯度步骤，推理无额外负担不等于训练免费；本轮不迁移其昂贵对抗训练。 |
| [BDNet，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Guan_BDNetBio-Inspired_Dual-Backbone_Small_Object_Detection_Network_CVPR_2026_paper.html) | CVF官方检索摘要；直接页面403；未验证官方源码 | 互补结构/颜色归纳偏置合理，但RGB派生灰度不是新传感信息，也不自动带来优势。采用可关闭的局部对比残差，不直接建两个同等重主干。 |
| [Seeing Through the Noise，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Yuan_Seeing_Through_the_Noise_Improving_Infrared_Small_Target_Detection_and_Segmentation_from_Noise_Suppression_Perspective_CVPR_2026_paper.html) | 官方摘要级 | 噪声抑制提醒可借鉴；红外背景统计不可直接推到绿色果叶。不能据此给叶片硬阈值、补圆或填凸包。 |
| [DyFCLT，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/Li_DyFCLT_Dynamic_Frequency-Decoupled_Cross-Modal_Learning_Transformer_for_Multimodal_Tiny_Object_CVPR_2026_paper.html) | 官方摘要级 | RGB-T跨模态收益需要热成像，不纳入RGB论文范围。 |
| [SpTopoNet，CVPR2026](https://openaccess.thecvf.com/content/CVPR2026/html/He_Towards_Persistence_Learning_Topological_Constraints_for_Event-based_Small_Object_Detection_CVPR_2026_paper.html) | 官方摘要级 | 事件时序持久性不等于单帧可见mask的连通/凸性约束；不把遮挡凹陷错误填回去。 |

J35→J36才是局部对比的归因对照。中心—周围算子的常量偏置抵消可推导，
但“果叶误报减少”必须实测，不能由算子证明直接推出。

## 主题三：低参数不等于低耗时；部署图和训练图要分开

| 来源及身份 | 实际核验层级 | 对本任务的决策 |
|---|---|---|
| [RepViT，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/html/Wang_RepViT_Revisiting_Mobile_CNN_From_ViT_Perspective_CVPR_2024_paper.html)；[代码](https://github.com/THU-MIG/RepViT) | 官方论文入口；本地`model/repvit.py`的ConvBN融合/RepVGGDW | 分开空间/通道混合、冻结BN后合并线性分支。J2RepMix独立实现3×3/1×1/identity分支，并测试数值等价；不是原RepViT网络复现。 |
| [RepNeXt，2024预印本](https://arxiv.org/abs/2406.16004)；[代码](https://github.com/suous/RepNeXt) | 第一方摘要；本地`model/repnext.py`的分支与融合 | 部署重参数化是确定的代数性质，训练涨点不是；不搬多组大核/多尺度空间分支增加成本。 |
| [FasterNet，CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/papers/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.pdf) | 官方论文检索/PDF来源 | 减FLOPs不保证吞吐，内存访问和算子执行重要。此前J的压缩路线掉点、当前新图CPU也未证明更快，所以先筛选并实测GPU，不能宣布成功。 |
| [Rewrite the Stars，CVPR2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Ma_Rewrite_the_Stars_CVPR_2024_paper.pdf) | 官方论文来源；Plug-play本地StarNet前110行 | 乘性混合有表达作用，但高分辨率上7×7 DW和扩张GLU仍有代价；不重复此前失败的GLU堆叠路线。 |
| [PIDNet，CVPR2023](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html) | 官方摘要级 | 细节、语义、边界分工可借鉴；不是把任意残差冠名闭环控制。本轮仅证明有界注入，**没有证明动力系统稳定或PID最优**。 |

本轮图结构为非CSP编码器、一次top-down短路径、低宽度P2候选、共享预测卷积、P3窄原型。
没有DCN/CUDA扩展、Mamba、FFT、排序扫描、自注意力矩阵、循环多轮编码。
没有这些算子只能减少某些工程风险，不保证所有GPU更快。

## 主题四：LLM迁移和loss只在缺陷证据充分时用

[Differential Transformer，ICLR2025](https://openreview.net/pdf?id=OvoCm1gGhN)提供差分注意力抑制噪声的思路，
但我们J10的上下文变体没有追回主干容量损失。语言任务上的降噪不能推出果叶区分提升，
本轮不用它覆盖更直接的候选/边界问题。

[Mask Scoring R-CNN，CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html)
及[Rank & Sort，2021](https://arxiv.org/abs/2107.11669)说明排序与质量估计值得单独研究，
但排序不能凭空生成漏掉的果实。本轮先固定AdamW与原生分类/定位/分配，
只把已有可见mask、等实例tiny Dice作为gain=0.1的一项损失消融；没有自动控制优化器的虚构收益。

下一步如果候选召回已改善但排序差，才引入质量排序对照；统一单调温度校准不能改变固定候选的排序AP；
已发现原型复制缺陷，先J36/J41单项修复，再J41/J42联合比较P3/P2来源与输出分辨率；
如果轻量容量再次崩溃，保留J29而不是无限增加微模块。

## 本地源码快照与许可边界

| 仓库 | 当前本地位置（Desktop/github下） | Git提交 |
|---|---|---|
| RepViT | `RepViT/` | `298f42075eda5d2e6102559fad260c970769d34e` |
| RepNeXt | `RepNeXt/` | `f515377837207e883cc9d6bcc38f9c98fcd2bc1f` |
| SET | `SET/` | `9208fbc4cfe571be4c15dccad8db1665cfdcb9d6` |
| ESOD | `esod/` | `bde3571bd7db697e441eef0278cd425e888ea026` |
| TinyFormer（本次新增下载） | `J2_20261003_TinyFormer/` | `9075d9fdf5ce989ca3c7e830a780dfc7fbbe4ec1` |

仅做源码阅读，不安装作者训练依赖，不下载其权重，不发送私有图像/标签/结果。
RepViT/RepNeXt根许可为Apache-2.0；TinyFormer根许可说明Apache-2.0，
但DINOv3派生文件仍声明DINOv3 License Agreement，**不可假定整个目录所有内容同一许可**。
本轮新模块没有复制这些文件；借鉴算法也必须引用思想来源，重新写代码不使既有方法变成原创。

## 可检验的贡献边界

1. 候选发现和掩膜生成分离预算是否能改善微果召回—成本曲线，而非只提高图分辨率。
2. 短语义注入是否比PAN的多次往返更利于我们的遮挡可见实例，需J30/J31及J32后续对照。
3. 局部对比残差是否减少叶片误报而保留薄片果边界，需J35/J36和人工难例子集。

这三项仍是研究假设；新命名、组合或严谨代数推导均不是尚未做出的实证结果。

## 深化审查补充：原文反证比模块名称更重要

另保存 `E:/mastercode/_work/20261003_j2_deep/literature_audit.md`，
对10篇原始论文指定方法、消融和限制章节复核，逐项注明L1摘要/L2指定全文/L3复现级别；没有L3。
此次不把初版摘要阅读改写成逐页阅读全文。

| 原始文献 | 本次深化关注 | 与当前代码的边界 |
|---|---|---|
| [NWD，2021/2022预印本](https://arxiv.org/abs/2110.13389) | §3.1–3.3，小框IoU位移敏感性 | box损失不是mask形状/果叶判别；未默认替换CIoU |
| [TOOD，ICCV2021](https://arxiv.org/abs/2108.07755) | §3.2，分类与定位任务对齐 | 当前已有TAL；新增P2改变最小资格窗口，需记录而非宣称新发明 |
| [YOLACT，ICCV2019](https://arxiv.org/abs/1904.02689) | §3.1–3.4、§7，共享原型装配与拥挤失败 | K非实例上限；输出shape相同不代表空间表达相同 |
| [PointRend，CVPR2020](https://arxiv.org/abs/1912.08193) | 细特征与实例粗mask联合条件，不确定点补充覆盖 | 没有在当前图中实现完整PointRend；共享叶脉边缘不能独自区分实例 |
| [Making CNNs Shift-Invariant Again，ICML2019](https://arxiv.org/abs/1904.11486) | 降采样前低通及过强平滑的取舍 | 高通contrast不是抗混叠；分类不变性不是定位不变性 |
| [Mask Scoring R-CNN，CVPR2019](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html) | quality估计作用于已生成候选，存在前置截断 | 不能生成漏掉的果；当前未新增质量头 |
| [DEIM，CVPR2025](https://arxiv.org/abs/2412.04234) | one-to-one监督、质量目标的有效梯度 | 当前TAL为one-to-many，不直接搬Dense O2O/MAL |
| [ESPCN，CVPR2016](https://www.cv-foundation.org/openaccess/content_cvpr_2016/html/Shi_Real-Time_Single_Image_CVPR_2016_paper.html) | 低栅格卷积+可学子像素输出 | J41只借用相位解码算子，不称实现图像超分或新增观测 |
| [ICNR，2017预印本](https://arxiv.org/abs/1707.02937) | 重复相位初始化避免初始棋盘差异 | 训练后仍可出周期伪影；新旧函数等价只对初始化成立 |

QueryDet、FasterNet、TinyFormer的深化查证亦见该报告；TinyFormer/CSCWD保持2026预印本标识。
本轮新检索补充ESPCN、ICNR、对象分组/Slot Attention及近期tiny mask路线。
[Slot Attention，NeurIPS2020](https://papers.nips.cc/paper/2020/hash/8511df98c02ab60aea1b2356c013bc0f-Abstract.html)
提供对象分组启发，但没有本任务实例监督与预算证据，不把全局迭代slot直接堆到dense P2。

真正的迁移链是：**任务测量→代码反例→原论文机制/负消融→单项修复→固定协议检验**。
旧已验证有效锚点保持，未证实的新修正不替代它；文献外部数值不作我们的预期提升。
