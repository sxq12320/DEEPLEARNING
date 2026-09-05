# SAGE V5：从候选召回瓶颈到双用途细节路由

日期：2026-09-04。状态：探索性新架构，尚无完整训练精度结论。旧 SAGE V4R、数据集图像/标签及训练结果均保留。

## 结论先行

这次不是所有改动都失败了。V4R 的非对称路由保住精度并降低参数量，语义指导细节进一步改善掩膜精度；额外回投影和 P4 主干替换没有继续提高峰值。几何约束有后期收益，但还会损失小果召回。下一版必须把“找不到果实”和“已经找到但边界不准”分开解决。

因此实现 SAGE50–56 七个新 YAML，以 SAGE30 和 SAGE42 作双参照。核心 SAGE52 把**同一份语义筛选后的高分辨率细节同时送往 P3 候选预测与掩膜原型**，并将原型中的高开销空间卷积移到上采样之前。没有叠加第二套金字塔、全图 P2 检测塔或迭代控制器。是否优于 SAGE42，必须由实验回答。

## 1. 审计范围与可信度

全历史索引位于 `reports/sage_v5_20260904/history/`：154 份 CSV、140 份不同内容，14 份是重复展示件；对应 249 个当时已有 YAML。自动逐轮读 CSV、匹配 YAML、记录超参数；人工重点复核有代表性的正向与负向对照、当前 V4R 模块和损失实现。**不能声称旧模型每次运行都使用了当前这份源码**：历史大部分缺少源代码快照，文件名对应不等于版本相同，也不能把不同数据和 AMP 的行混成一张因果排行榜。

当前 V4R 十组最可比：全部 300 轮、seed=42、AMP=False。除模型名、模型路径、输出路径外，保存的训练参数一致。实际加载 676 张训练图/4324 个实例，193 张验证图/1049 个实例；没有把同名 NPY 当成额外样本。本次改代码之前，运行目录记录的实现 SHA256 均与本地对应源文件匹配。详细报告为 `reports/sage_v5_20260904/audit/v4r_audit.json`。这比依赖目录名称可靠：该目录虽然叫 SCREEN_50EP，实际并非 50 轮。

### V4R 同协议结果

以下百分数均取 Mask AP50–95 最佳轮，AP50 不单独挑另一轮。尾20均值是训练后期表现，不是三种子重复。

| 模型 | Mask AP50–95 | 同轮 Mask AP50 | 尾20 AP50–95 | 解读 |
| --- | ---: | ---: | ---: | --- |
| SAGE30 原生对照 | 66.719 | 82.437 | 65.899 | 当前组内参照 |
| SAGE40 非对称路由 | 66.863 | 82.248 | 66.002 | 参数降约18.5%，精度基本保住 |
| SAGE41 直接细节 | 67.246 | 82.538 | 66.284 | 细节分支有正向信号 |
| SAGE42 语义指导细节 | **67.502** | 82.982 | 66.483 | 本组峰值最高；对30是+0.783个百分点 |
| SAGE43 回投影 | 67.367 | 82.730 | 66.001 | 对42峰值−0.135，尾20−0.481 |
| SAGE44 边界约束 | 67.433 | 82.716 | 66.432 | 对43略正向，非大幅提升 |
| SAGE45 邻接约束 | 67.291 | **83.101** | 66.731 | AP50与严格AP的取舍 |
| SAGE46 两项几何约束 | 67.491 | 82.450 | **67.085** | 后期最好；不能据此认定召回最好 |
| SAGE47 P4 Faster | 67.002 | 82.555 | 66.093 | 对43下降，不默认推广 |
| SAGE48 P4门控 | 66.841 | 82.729 | 66.015 | 对43下降，不默认推广 |

SAGE42 和46的峰值仅差0.011个百分点，不足以宣布谁是稳定冠军。原图混淆矩阵中30为TP821/FP149/FN228，42为831/167/218，46为815/119/234。它们是框匹配统计，而且通常来自最终验证的官方 best.pt；不是同一最佳掩膜轮的逐实例边界证据。

### 历史里值得保留和必须谨慎的结论

| 对照 | 已保存数据中的信号 | 本次用途 |
| --- | --- | --- |
| G08→G10 | AP50–95 67.259→67.681；AP50 82.638→83.824 | 只支持该配方下P5小波替换的潜力；SAGE56单独再测 |
| G10 standard→full recipe | 67.681→64.033 | 多项附加损失不可一股脑加满 |
| N02 standard→full recipe | 67.341→65.006 | 同样不支持“损失越多越好” |
| S00→S06非对称 | 60.740→61.346，AP50略降 | 历史支持精度/效率取舍，V4R再次检查 |
| S06→S07加LSKA | 61.346→60.510 | LSKA并非换个位置就稳定涨点 |
| G0830 G00/G02/G03/G04 | 67.031/67.241/65.631/66.515 | 双路有弱信号；频率/RepMixer未普遍奏效，G02还混有损失因素 |
| SAGE20/21/22/23 | 66.621/66.935/66.593/66.717 | 结构与监督要分开；旧V3加载重复样本，不能直接对比本次V4R |
| T系列 | 多个只跑16/186/195/240轮 | 不用不等训练预算宣布赢家 |
| F14 LSKA、F17 CARAFE | 历史AP50–95为67.599、67.170 | 数据、AMP及初始化与早期基线不统一，不把全部增益算给模块 |

Light 的“训练巨慢”是需要服务器实测的运行问题。本地结果索引里没有可公平对应的完整 Light 性能组，不能虚构其精度或速度表。

## 2. 当前真正的瓶颈

加载 SAGE30、42、46 的 `best_mask.pt`，在本地匹配文件名的193张验证图上统一重评：CPU FP32、batch1、640、rect=False、NMS IoU=.7、max_det=300。**这是新的诊断协议，不替换服务器 CSV。** 本地Torch2.8 CPU与服务器Torch1.13 GPU、验证形状/掩膜处理等存在差异，重新计算AP也不完全相同。文件名相同不证明图像字节相同；没有读测试集。

按实际输入尺度的stride4栅格掩膜面积分组。下面是Mask IoU≥.5的一对一匹配召回，不是COCO AP_small。

| 模型 | 极小果153个：conf=.001 | 极小果：conf=.25 | 较大果617个：conf=.001 | 较大果：conf=.25 |
| --- | ---: | ---: | ---: | ---: |
| SAGE30 | 45.75% | 18.30% | 99.19% | 96.92% |
| SAGE42 | **53.59%** | **19.61%** | **99.35%** | 96.76% |
| SAGE46 | 45.75% | 15.03% | 98.54% | 96.76% |

极小果指输入尺度面积<16×16，较大组指≥32×32，不能把这里的“较大”冒充COCO large。SAGE42的中间组279个果实在conf=.25时召回77.42%。这表明极小果的候选覆盖及置信度排序问题远大于普通大小果实。

V4R 的 `detail` 只加到 `proto`，Detect/分类/框预测使用的P3没有这条直接细节输入。并非梯度完全断开，而是前向信息只直接服务于掩膜。SAGE V5新增的是这条缺失的信息路径。

遮挡/邻接不能忽略：在本地stride4诊断中316个GT的solidity<.9，479个GT在8输入像素膨胀邻域内有另一实例。两者是**分辨率相关代理**，不是人工已验证的遮挡/接触标签。SAGE42的分裂代理高于30（3.53%对2.29%），46降为1.91%；代理会混入重复检测，正式结论须按预测实例ID进行人工核验。真实凹陷的可见掩膜不能强制补成圆形或凸包。

颜色方面，局部Lab均值差<10的359个实例中，42在conf=.25的召回约78.27%，与其全体80.36%比较，尚不足以证明“颜色依赖是第一瓶颈”。均值差也受阴影、边缘和尺度影响。颜色伪装仍是合理任务动机，但应做尺度匹配的低对比子集、保持结构的颜色扰动测试，不能用看起来都是绿色推导因果结论。

### PR尾端到底是什么

本地 `ultralytics/utils/metrics.py:compute_ap()` 显式在最大已达到Recall处补Precision=0，再延伸到Recall=1。因此图尾贴零包含**未达到召回区间的补零约定**，不是模型真的在Recall=.95时测得Precision=0。此前陡降也有真实部分：阈值降低后错误候选增长、剩余小果依然漏检。本次没有改评估代码或美化曲线。

正确目标是改善实际召回覆盖、固定精度下的Recall和AP，而不是把尾部绘成高位。当前本地42在最低阈值也只匹配约90.09%的全部实例，极小果仅53.59%；这正是候选路由需要检验的地方。

## 3. 从桌面仓库和论文取什么、不取什么

仓库索引与重点源文件记录在 `reports/sage_v5_20260904/sources/`。模块集合包含期刊、arXiv和第三方复现，不是每个文件都是顶会作者的官方代码；例如其中CARAFE明确写了unofficial。研究搜索客户端未安装，因此本次使用网页检索核对CVF/ECVA/作者仓库，原始检索结果已保存。

| 来源与本地关键实现 | 学到的思路 | 迁移取舍 |
| --- | --- | --- |
| PIDNet，`github/PIDNet/models/model_utils.py:PagFM/LightBag` | 细节、上下文有不同职责，融合需要选择而不是直接淹没细节 | 保留语义筛选+小增益残差；不照搬整套三主干[^1] |
| Gated-SCNN，`github/GSCNN/network/gscnn.py` | 高层语义筛选浅层形状噪声 | 沿用42的窄细节估计；不搬源码里的CPU Canny往返、全图重型ASPP[^2] |
| QueryDet，`github/QueryDet-PyTorch` | 小目标候选需要利用高分辨率信息，但全图高分辨率检测很贵 | 新增C2→P3候选路由，不加密集P2检测头，也不引入稀疏卷积依赖；不是QueryDet复现[^3] |
| FastInst，`github/FastInst/fastinst/modeling/pixel_decoder/fastinst_encoder.py` | 像素解码器可以简化，计算预算不应全耗在融合上 | 将重空间混合放在低分辨率；不迁入Transformer query decoder[^4] |
| YOLACT，`github/yolact/yolact.py`及`data/config.py` | 共享原型与实例系数解耦 | 保持原型×系数范式，重排原型计算次序；不声称该范式是原创[^5] |
| WTConv论文；本项目`citrus_far.py:C3k2_WT` | 用频带处理扩大上下文表达 | 历史G10使其值得独立试验；56只改P5，不全网替换。项目实现是简化单层版本，不是官方全实现[^6] |
| Plug-play的RCM/CGRSeg | 轴向上下文指导空间重建 | 与已有语义筛选重叠，本轮不叠加RCM[^7] |
| Plug-play的PKIBlock/PKINet | 多尺度局部上下文与长程上下文分工 | 遥感框检测成功不等于本任务nano实例分割成功；不在高分辨率层并联五个核[^8] |
| Plug-play的HaarDownsampling、CARAFE | 分频/自适应重采样 | Haar后通道压缩并非整体无损；CARAFE的unfold可能放大中间激活，本轮不用[^9] |
| SFM、SCSegamba本地仓库 | 频率保护、结构信息建模 | SFM重采样需要检查坐标对齐与开销；裂缝连续性不等于果实例分离，本轮不引Mamba依赖[^10] |

新模块是本项目对这些问题分工的独立适配，不是复制多个作者模块后改名；也不是自动获得新颖性或涨点证明。可引用的创新对象是**具体信息路由与算力分配方式及其经消融验证的作用**。

## 4. SAGE V5的结构与消融

共保留三种尺度的原生检测/实例系数塔（P3、P4、C5）。主路保留SAGE42的非对称颈部；C2高分辨率细节与P3语义得到同一份16通道细节估计。

候选支路：把stride4的16通道按2×2像素重排为stride8的64通道，1×1投影后以可学习小增益残差注入P3，服务分类、框和实例系数。**像素重排本身不丢值，但随后投影不保证无损**，也不能把它叫成物理抗混叠定理。

掩膜支路：将原型的第二个64通道3×3卷积移到P3执行，之后nearest上采样，P2只保留1×1原型投影及细节注入。输出仍为32个stride4原型，不降低mask_ratio、不缩小训练输入。它改变了计算顺序，不只是调小通道数；主要风险是高分辨率空间混合能力下降，所以50必须先单独验证。

语义门在[0,1]内，小增益tanh初值对应0.01；避免双零初始化把新支路梯度同时封死。这只是控制启发的增益调度/保守残差，不是真正时间PID或闭环稳定性证明。42→43负向证据使回投影不再成为默认部分。

| YAML编号 | 相对改变 | 需要回答的问题 |
| --- | --- | --- |
| 30 | 原生对照 | 同预算真实基线 |
| 42 | 已有非对称+语义细节 | 已有强参照，不只与弱基线比 |
| 50 | 42 + 延后升采样原型 | 单独省计算会损失多少精度？ |
| 51 | 42 + 候选细节路由 | 直接改善候选是否能提高小果召回？ |
| 52 | 50 + 候选细节路由 | 两项结构改动能否互补？ |
| 53 | 52 + boundary=.1 | 可见凹陷边界约束的独立作用 |
| 54 | 52 + neighbor=.1 | 邻实例侵入惩罚会不会压低小果召回？ |
| 55 | 52 + 两项几何=.1 | 完整2×2监督消融，不预定为冠军 |
| 56 | 52仅P5 C3k2→C3k2_WT | G10历史信号能否迁移到本次固定协议？ |

50–55的主体主干仍是YOLO11，不能包装成“全新主干”。56确实改了P5特征提取，但也不是整个主干换掉。当前证据不支持为了“大改”而把浅层/P4全部替换；新结构的主体创新在跨尺度信息去向与掩膜解码预算。

### 控制文档中没有采纳的断言

文档里的具体mAP、延迟、参数预算及100%兼容勾选不是可验证的运行记录。本次全部重新测量。GAP不是时间积分器；深层非线性网络不自动满足LTI可观测性条件；tanh或凸组合不能直接推出整个网络Lyapunov稳定。

文档给定一阶对象P(s)=K0/(τs+1)、PID C(s)=(Kd s²+Kp s+Ki)/s，则标准负反馈特征多项式为 `(τ+K0*Kd)s² + (1+K0*Kp)s + K0*Ki`，并非文中套用的三阶式。未经建立对应模型与假设，不能把其Routh条件当作本网络的稳定性定理。

## 5. 计算与工程边界

同一本地Torch/THOP环境、nc=1，Ultralytics get_flops估计：

| 模型 | 参数/M | GFLOPs估计@640 |
| --- | ---: | ---: |
| SAGE30 | 2.843 | 10.356 |
| SAGE42 | 2.319 | 10.042 |
| SAGE50 | 2.303 | 7.781 |
| SAGE51 | 2.323 | 10.097 |
| SAGE52–55 | 2.307 | 7.836 |
| SAGE56 | 2.216 | 7.763 |

52对30参数−18.85%、GFLOPs约−24.33%。get_flops采用stride输入外推，且THOP可能漏计函数式运算，因此不是完整算子审计；不同版本的THOP/GPU日志数字不能直接混用。需要以目标GPU相同batch、精度、分辨率下实测延迟为准。

标准结构候选不使用Mamba、可变形卷积、grid_sample、CARAFE/unfold或额外注意力矩阵；保留原主干中已有C2PSA。56沿用现有纯PyTorch小波实现，没有新增安装包。新头按原类路径导出/注册，普通 `YOLO(yaml)` 即可训练，不依赖启动器临时注册模块。

批量程序沿用V4R的前台、同进程、串行、Ctrl+C即终止队列的方式。按种子固定随机化队列顺序；保存args、源代码摘要、加载样本清单和初始化相等参数比例；不要求用户确认数据指纹，不重命名数据集。保留官方best.pt，并另外保存按Mask AP50–95选择的best_mask.pt。

下一步和全部固定超参数见 [训练指南](SAGE_V5_TRAINING.md)。完整训练、三种子稳定性、真实GPU时延都还没有完成；任何“涨10点”“已经最优”“控制理论已证明稳定”的说法现在都不成立。

## 参考来源

[^1]: Xu et al. PIDNet. CVPR 2023. [论文](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html)，[官方代码](https://github.com/XuJiacong/PIDNet)。本项目不复制其MIT源码。
[^2]: Takikawa et al. Gated-SCNN. ICCV 2019. [论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Takikawa_Gated-SCNN_Gated_Shape_CNNs_for_Semantic_Segmentation_ICCV_2019_paper.pdf)，[官方代码](https://github.com/nv-tlabs/GSCNN)。
[^3]: Yang et al. QueryDet. CVPR 2022. [论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)，[官方代码](https://github.com/ChenhongyiYang/QueryDet-PyTorch)。
[^4]: He et al. FastInst. CVPR 2023. [论文](https://openaccess.thecvf.com/content/CVPR2023/papers/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.pdf)，[代码](https://github.com/junjiehe96/FastInst)。
[^5]: Bolya et al. YOLACT. ICCV 2019. [论文](https://openaccess.thecvf.com/content_ICCV_2019/papers/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.pdf)，[官方代码](https://github.com/dbolya/yolact)。
[^6]: Finder et al. Wavelet Convolutions for Large Receptive Fields. ECCV 2024. [论文索引](https://eccv.ecva.net/virtual/2024/poster/1059)，[官方代码](https://github.com/BGU-CS-VIL/WTConv)。本项目旧WTConv为单层简化实现。
[^7]: Ni et al. Context-Guided Spatial Feature Reconstruction for Efficient Semantic Segmentation. ECCV 2024. [论文](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/06867.pdf)。
[^8]: Cai et al. Poly Kernel Inception Network for Remote Sensing Detection. CVPR 2024. [论文](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)，[代码](https://github.com/PKINet/PKINet)。
[^9]: 本地Plug-play `HaarDownsampling.py`指向[HWD作者仓库](https://github.com/apple1986/HWD)；`CARAFE.py`自称非官方实现并使用nn.Unfold。这里的成本取舍来自源码检查，不借用论文的FPS作本项目速度。
[^10]: [SFM论文](https://arxiv.org/abs/2507.11893)、[SCSegamba论文](https://arxiv.org/abs/2503.01113)及桌面同名仓库README/核心实现。两者原任务不是本项目可见果实例分割。
