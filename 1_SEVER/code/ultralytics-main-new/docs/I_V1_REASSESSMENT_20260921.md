# I V1 重审：先分清漏在哪里，再决定改哪里

日期：2026-09-21。状态：证据复核与消融编排修订；**没有生成新的训练精度结果，也未重写十臂数值架构**。

## 1. 本次实际看了什么

- 最新 I1 实现 `ultralytics/nn/modules/citrus_i_v1.py`、V12 父类、十臂配置/生成器、训练入口与测试。
- 复用并核对 20260920 已有全历史索引（284 CSV 条目、270 个去重 CSV），不是声称今天逐行重读了全部历史代码。
- 重点复查 `I_V1_REVIEW_20260920/V12_metrics.csv`、`PR_evidence.json`、三审综合的划分/协议限制。
- 桌面作者仓库：QueryDet `models/querydet/qinfer.py`，RefineMask `refine_mask_head.py`，
  TOOD `tood_head.py`，PIDNet `model_utils.py` 的候选查询、语义/实例融合、任务对齐、边界融合实现。
- 桌面 Plug-play：PKIBlock/CAA 实现、MSCA.py 的来源标记；后者实际是 FcaNet 的频率通道注意力，
  不能因为文件名 MSCA 就当作 SegNeXt 的同名缩写模块。
- 《论文创新指南2026》共 51 页；抽读 1–4、41–43、51 页的范式、组合与条带卷积建议。
  这是二手设计启发，不是验证柑橘效果的科学证据，也不是“组合即可投稿”的保证。

历史峰值如旧 G10 mask AP50=83.824%、旧 78 配方=78.635%，不能相减就宣布 +5.189pp 架构收益。
不同划分、AMP、初始化、验证器与模型选择方式都必须排除。当前 baseline 随机初始化/AMP1，
I1 默认官方预训练/AMP0，并且输入、增强与 mask 栅格不同，不能直接作结构消融。

## 2. 最有价值的近期结果（仅旧验证集内部比较）

下面三行均采用同批 fine 切片预算、common-raster 评估；AP/R90 单位为 %。
tiny 是审计工作点下匹配数，不是 AP；背景 FP 也不是专门的“叶片 FP”。

| 模型 | mask AP50 | mask AP50-95 | R@P≥90% | tiny匹配/161 | 背景FP | 端到端ms/图 |
|---|---:|---:|---:|---:|---:|---:|
| V12_01 细节路径 | 89.351 | 77.478 | 81.125 | 98 | 385 | 296.5 |
| V12_03 分类识别路由 | 89.274 | 77.682 | 79.981 | 90 | 339 | 312.2 |
| V12_04 共享识别路由 | 89.566 | 78.309 | 81.506 | 97 | 436 | 338.7 |

结论：03 降误报但没有同时赢得 tiny/R90；04 AP 最高但比 01 多 51 个背景 FP、慢约 14.2%。
01 必须保留为轻量候选，不能只盯03/04。02 persistent 相比01的 fine AP 77.478→76.841，
tiny 98→89，同样不支持“多一个高分辨率路径一定更好”。这些都是单 seed 的探索性现象。

同一个01，global→fine：tiny 37→98/161（22.98%→60.87%），延迟78.7→296.5ms（约3.77倍）。
**分辨率供给/切片是当前最明确的 tiny 正向信号；增加解码模块是否划算仍需验证。**

旧 V12 的服务器成员与正式 grouped_dedup 不对应，且出现按组跨 split。
历史 audit 依据同名映射，服务器内容哈希未完成核对；不能把这些值当正式独立测试结论。
不能用已训练过正式 test 成员的旧权重“补测”来恢复独立性。正式结论需从正确初始化在正式划分重跑。

## 3. I1 当前具体不足

1. **干预位置偏后。** 新增的是原型解码分支，候选头仍为 P3/P4/P5。它可以改善已有候选的 mask IoU，
   也会通过共享训练梯度间接影响检测，但没有直接增加 tiny 候选机制，不能承诺找回完全没有候选的果实。
2. **潜在基底差异不等于语义误差。** 两套32通道原型联合学习，却没有保证逐通道语义对齐。
   `abs(proto_s-proto_d)` 可以是可学习线索，但不能直接解释成“叶果混淆残差”或闭环误差。
3. **识别修正和第二原型并未真正共享校正状态。** semantic proto 读原始 `x[1]`，识别修正在
   `Detect.forward → forward_head` 内另做。不能把“同步”写成已经证实的一致语义约束。
4. **零起始有优化代价。** mix=0 时新增语义/门控分支第一步梯度为0，先学习增益才打开分支。
   这是恒等启动设计而非死代码；新增真实 SGD 两步测试验证其能打开，但不代表真实训练必然打开充分。
5. **有界增益不是有界系统。** strip attention 没有限幅，DW反馈无范数约束；tanh 也允许负值外插。
   它既不是凸组合保证，也不是自动控制稳定性证明。已修正文档和注释，未偷偷加sigmoid改变算法。
6. **消融链缺口。** 旧 priority=00/01/04/05 无法区分“多一条原型”“上下文”“门控”各自作用；
   plain 是同宽、不是同参数量。新 mechanism 套件补齐 00/01/02/03/04。
7. I04 基于分类路由而不是04共享路由；不能说把历史所有最优部分已经组合起来。
   共享路由+双原型值得后续单独对照，但不应在门控是否有效尚不清楚时直接宣布它是最终模型。

## 4. PR 问题应如何处理

当前 `ultralytics/utils/metrics.py::compute_ap` 明确在最大实测 recall 后加 precision=0 的哨兵并延伸到1。
因此末端水平贴零部分是绘图/积分端点；横轴是 recall，不是 confidence。
最大召回之前的精度下降则可能是真实低置信度误报，不能全部归咎绘图。

不要改积分或裁去尾部来“涨点”。固定mask栅格、置信度下限、NMS、max_det、切片预算后，同时检查：

- 原始经验PR的最后一个真实点、Rmax、R@P≥0.9；
- 每个tiny GT在阈值前是否存在 box IoU≥0.5 候选；
- 有框却 mask IoU<0.5、排序低被过滤、NMS/跨切片融合丢失、完全无候选分别计数；
- 固定小样本难例盲标叶/枝/其它背景/漏标疑似，避免把背景桶都说成颜色混淆。

不要仅根据PR图推导“所有缺陷都是tiny”。遮挡形成深凹可见mask与接触果实的分离同样重要。

## 5. 文献与本地代码带来的取舍（不是直接搬模块）

| 来源 | 可借鉴的机制 | 对当前I1的用途 | 明确不做什么 |
|---|---|---|---|
| QueryDet，CVPR2022 | 低成本查询高分辨率位置 | 若主要是无候选，研究有限预算的细节候选补充 | 不直接搬 spconv/Detectron2；查询漏检需独立高召回兜底，未验证前不取代均匀切片 |
| RefineMask，CVPR2021 | 实例预测与语义、细粒度特征逐阶段融合 | 若有框但mask差，研究实例级、可监督的局部细化，而非解释未对齐基底差为误差 | 不在全图反复运行大宽度精修头 |
| TOOD，ICCV2021 | 分类与定位对齐 | 先测框/掩膜/分数错配，再决定是否做mask质量排序辅助 | 当前已有TAL；不能把现成TAL再次写成新创新，也不贸然同时换匹配器和损失 |
| PIDNet，CVPR2023 | 边界信息调节细节与上下文融合 | 借鉴抑制过度平滑的分工思想、保持局部凹边界 | 不把静态残差网络等同严格PID；不加入控制器/积分状态或额外多任务头 |
| PKINet/条带上下文 | 低成本扩大上下文 | 保留I03/I05作机制候选，不作为主贡献前提 | 不因为顶会标签就扩大高分辨率通道或叠更多注意力 |

论文与作者仓库：

- [QueryDet论文](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)，[作者代码](https://github.com/ChenhongyiYang/QueryDet-PyTorch)。
- [RefineMask论文](https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html)，[作者代码](https://github.com/zhanggang001/RefineMask)。
- [TOOD论文](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html)，[作者代码](https://github.com/fcjian/TOOD)。
- [PIDNet论文](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html)，[作者代码](https://github.com/XuJiacong/PIDNet)。

## 6. 建议的收敛方案与实施边界

**已实施**：baseline环境修复；I1错误说明纠正；mechanism/feedback套件；真实梯度打开测试。
十个YAML、模型前向公式、权重路径、损失/优化器、训练AMP和已完成结果均未重写。

本次本地验证：I1 44项（十臂构建/前后向、GFLOPs、融合、读写、重放、两步增益打开等）、
前台入口37项、baseline31项全部通过；服务器CUDA训练与准确率提升未验证。

**下一步验证顺序**：

1. 正式划分核对后，先做baseline 3轮smoke；不能把“78复现配方”当成刻意选弱对照。
2. 分清两种对照：同初始化/AMP/预算的结构消融；官方独立实现的系统基线。二者分表。
   如果保留scratch/AMP1基线，方法也必须增加同设定配对，不能仅报预训练/AMP0的胜出结果。
3. I1先跑mechanism五臂筛选，而非优先运行所有上下文变体；V12_01仍作为外部简洁锚点。
4. 若I04不优于I03，不继续增加门控；若plain也无收益，不把双原型设为论文核心。
5. 候选缺失占主要部分→把计算移到小目标候选与分辨率；已有候选mask差占主要部分→实例级精修；
   叶片误报确经标注确认且主要在低分段→考虑质量排序/训练集难负例，而不是硬编码绿色或圆形筛选。
6. 暂不同时改优化器、损失和架构。保留现有tiny/边界/邻近项，后续分别关掉做必要性对照；
   原有面积归一化不等于完全忽略tiny，进一步重加权可能放大标注/栅格噪声。
7. 以同预算 mask AP、R90、tiny召回、端到端延迟筛选非支配候选；最终基线/方法各3seed，再一次测试集评估。

前台入口仍是 `RUN_CITRUS_I_V1.py`，可设 `SUITE="mechanism"`。先EPOCHS=3独立目录，再50轮筛选，
再300轮新目录。默认DATA的旧路径字符串不会证明它是正式划分；务必指向实际grouped_dedup成员。
本次源文件hash发生变化，若已有I1项目，使用新PROJECT，不覆盖旧权重/协议记录。

这里建议的最终研究方向是**分辨率预算分配 + 候选/掩膜错误分工 + 遮挡边界保真**，
不是给当前模型再叠几个顶会块。候选缺失与掩膜失败的占比尚未量化，不能凭空预定谁是最终胜者。

## 7. TypeSafe技能的实际使用和限制

已阅读TypeSafe官方API、Choice和citation-check文档。将可判断的主张与对应事实分离，
设 supported / contradicted / insufficient_evidence 三类；算术与代码检查仍在本地完成。
获得用户对具体摘要外发的明确授权后，通过官方API完成一次7题核验，请求model=`jev-latest`。

产物在 `E:/mastercode/_work/20260921_baseline_i1/`：`typesafe_request.json`、`typesafe_response.json`。
密钥通过无回显输入读取，只保留内存；未写入脚本/产物。原密钥已在聊天出现，建议撤销轮换。

| 主张 | Jev返回 | 本地处置 |
|---|---|---|
| traceback是依赖故障 | supported | 以实际调用链与兼容测试修复 |
| 切片改善该验证集tiny检出但更慢 | supported | 保留该有限范围事实 |
| 分类路由已经解决颜色混淆 | insufficient_evidence | 不作此声称，补叶片难例标注 |
| tanh门控已证明闭环稳定 | insufficient_evidence | 删除有界系统暗示，需数学条件而非模型投票 |
| 当前baseline/I1能隔离架构效应 | contradicted | 分协议，不直接算架构增益 |
| I1已证明找回无候选果实 | insufficient_evidence | 候选级诊断优先 |
| PR贴零证明置信度85%以上全错 | insufficient_evidence | 以源码纠正坐标含义和哨兵点；不服从含混的语义判断 |

以上只是对所提供摘要的模型判断，不能独立验证数据真实性、替代实验或保证涨点。
API返回的概率/置信度原样存档，不当统计显著性或研究结论的置信区间。
参考：[TypeSafe API](https://docs.typesafe.ai/api)，[证据核验模式](https://docs.typesafe.ai/cookbooks/citation_check)。
