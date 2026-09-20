# Reviewer 2

本报告于 2026-09-20 独立完成并冻结。预先指定的侧重点是新颖性和机制证据强度。审阅仅使用 REVIEW_PACKET.md 所列材料，未读取其他审阅、综合意见或后续方案。以下文件名均指该证据包中的文件，运行名省略共同后缀 `_seed42` 时含义不变。

## 审阅范围与总体判断

输入是实现、实验记录和诊断结果，不是完整论文。现有材料支持“任务特定识别修正具有可继续检验的价值”，还不能建立“已经验证的轻量高精度方法，同时改善极小果实召回与叶片混淆”的完整结论。限制来自具体证据链，尤其是正式划分对应关系、仅单 seed 验证集筛选、机制对照不完整和计算预算差异，不是应用研究本身价值不足。

代码提出的研究问题是清楚的。先把低分辨率语义投影到窄通道，再用局部低频差异与细节残差修正分类分支，尝试避免识别增强直接干扰几何预测。这可以形成应用论文的技术假设。但实现了这种路径，不等于证明这种路径解释了改进；当前共享分支对照在切片评估中反而获得更高 AP。

目标读者包括果园视觉、农作业机器人感知以及资源受限实例分割研究者。极小果实遗漏、背景误报和推理成本之间的关系对这些读者有实际意义。当前证据不涉及机器人执行成功率，因此不评价套袋控制或部署完成度。

## 主要优点

- `citrus_e_v12.py` 将新增贡献限制为识别修正，`route=0/1/2` 分别对应重放、分类分支和共享分支。`citrus_e_v12_suite.py` 还预置了同参数量的求和证据对照，具有可检验性。
- 实现明确区分有界增益与有界特征，说明不存在循环状态、时间积分或稳定性保证，并承认共享骨干与分配器仍存在梯度耦合。这些限定避免了把前馈残差误写成控制理论证明。
- 六个优先实验均有 300 epoch 记录。`checks.json` 的参数差异仅涉及模型和输出路径，数据加载列表在六臂中一致，AMP 均关闭。六臂内部比较的基础比跨历史协议直接排序更充分。
- 同时提供普通全图、粗切片、细切片结果、极小目标匹配、背景误报、固定 precision 下的 recall 和延迟。报告未只保留最高 AP，这允许审查准确率与成本之间的真实取舍。
- `initialization_transfer.json` 显示 V12_02、03、04 均有 2,090,928 个参数与初始化源精确相等，03 与 04 的总参数数同为 2,283,950。这为分析分支路由差异提供了有用控制，但不等同于所有新参数初始状态或历史实现已被验证一致。

## Major Concerns

### R2-M1

- **Severity** Major
- **Blocking** Yes
- **Axis** technical soundness / data-resource-quality
- **Issue key** formal_split_identity_and_group_independence
- **Claim pointer** REVIEW_PACKET.md 中可发表的未成熟柑橘实例分割贡献，以及以正式 cleaned grouped 数据评估其有效性的基础。
- **Evidence pointer** V12_03 的 `completed.json`、`train_original_sources.json`、`loaded_data_summary.json`、`paired_fine_eval/paired_metrics.json` 的 `records.image`；本地正式数据的 `audit/split_manifest.csv` 和 `audit/audit_report.json`。
- **Evidence status** located

数据身份不能由相同的 676/193 图像数量推断。V12_03 记录的原始训练实例数为 4,324，验证实例数为 1,049；本地正式审计为 4,120 和 1,181。按图像名逐一对应正式 split manifest，服务器训练来源中有 183 张属于本地正式验证或测试集，其中 126 张属于验证集、57 张属于测试集；paired 验证的 193 个图像名中，148 张不属于本地正式验证集，其中 123 张属于训练集、25 张属于测试集。所有这些名称均能在正式 manifest 中找到。

进一步按同名图像对应的 group_id 映射，V12_03 的训练和 paired 验证涉及 94 个共同捕获组，虽然没有直接重复图像名。此组重叠计算依赖服务器与本地同名图像身份一致，证据包未提供服务器逐图内容哈希以独立消除这一条件。因此，这足以证明当前记录与本地正式划分不一致，并提示严重的组独立性风险，不能据此武断宣称服务器图像内容已逐字节核实或所有提升都来自泄漏。

这一问题阻断正式泛化结论。六臂使用同一批数据可以维持内部探索性比较，但共同暴露于同组信息仍会改变小幅性能差的解释，也不能把已参与这些运行的本地测试图直接视为未见测试集。

**Resolution test** 提供服务器原始图像和标签的内容哈希及实际成员清单，与正式 grouped_dedup manifest 逐项对应。若确认沿用了另一划分，冻结方法选择，在清晰标记的正式 grouped 划分上从共同初始化重新训练主基线和最终方法，并将当前结果保留为探索结果。正式报告需解释此前接触本地测试图的情况及最终独立验证安排。

### R2-M2

- **Severity** Major
- **Blocking** Yes
- **Axis** originality / mechanism-evidence
- **Issue key** recognition_specific_discrepancy_not_discriminated
- **Claim pointer** `protocols/citrus_e_v12.yaml` 的 recognition 与 shared_control 因子，以及 `citrus_e_v12.py` 中识别专用语义差异修正和保护几何路径的设计动机。
- **Evidence pointer** `citrus_e_v12.py` 第 21 至 51 行、第 55 至 63 行和 `forward_head`；`citrus_e_v12_suite.py` 的 FACTORS；`V12_metrics.csv` 中 02、03、04 三行；协议第 29 行的 sum_control。
- **Evidence status** located

最接近新增贡献的对照是 02 对 03，而不是把 03 的绝对 AP 全部归因于识别修正。以共同 raster 的 mask AP50-95 百分数计，02→03 在 global、coarse、fine 下分别为 68.324→68.507、75.646→76.025、76.841→77.682。增幅值得继续研究，但不能单独区分差异表示、额外容量、分类分数调整和训练扰动。

对“只修正识别比共享修正更有利”的判断，已有证据并不一致。同参数量的 04 在 global、coarse、fine 下为 68.608、76.534、78.309，均高于 03 的相应 AP；03 的背景误报则更少。说明可观察到的是 AP、误报和延迟的取舍，尚非任务专用路径全面占优。设计中的 V12_08 求和对照不在已完成六臂中，因而差异表示的必要性尚未测到。

代码头部列出了 TOOD、PIDNet、FreqFusion 等启发来源，但共同证据包没有这些方法的技术内容或系统文献比较。只能确认作者已有归因意识，不能确认该组合或其关键操作此前不存在。此处既不宣称缺乏新颖性，也不认可“新的机制”已经证实。

**Resolution test** 用相同初始化来源、正式划分和推理预算比较 02、03、04 与 08，至少对最终主张所需对照重复多个 seed。报告等 precision 或等背景误报条件下的整体和 tiny recall，同时给出预测分数变化、框/掩膜 IoU 变化及错误转移。若分类专用路径仅降低误报或成本，应将贡献限定为这一明确取舍；若差异与求和无稳定差别，不把差异表示列为必要机制。论文需补充最接近工作的逐操作差异，而不是只列启发性名称。

### R2-M3

- **Severity** Major
- **Blocking** Yes
- **Axis** scientific importance / claim-moderation
- **Issue key** tiny_recall_and_leaf_specificity
- **Claim pointer** REVIEW_PACKET.md 明确要求重点评估 tiny fruit、foliage confusion 和 recall。
- **Evidence pointer** `PR_evidence.json` 中 V12_02、03、04 的 global/coarse/fine 项；`V12_metrics.csv` 的 tiny、background 和 r90 列；V12_03 `paired_fine_eval/paired_metrics.json` 的 limits。
- **Evidence status** located

新增识别分支尚未表现为稳定的极小果实召回改善。02→03 在相同固定诊断阈值下，global 的 tiny 匹配为 35→33，coarse 为 74→69，fine 为 89→90，分母均为 161。03 的背景误报确实由 144/334/379 降至 108/275/339，但这可能包括分数压低带来的召回交换，不能只强调误报下降。

“background”也不是“叶片误识别”的语义标签。paired 报告的 limits 明确说明固定阈值错误桶不能构成叶片混淆的语义证据。当前指标没有区分叶、枝、其他背景与漏标果实，因而还不能证明特定的果叶辨别机制。

细切片会大幅提高 tiny 匹配，但同一模型同时改变了观测预算。例如 03 从 global 的 33/161 到 fine 的 90/161，同时背景误报从 108 增至 339，延迟从约 76.4 ms 增至 312.2 ms。该变化支持增加局部尺度观测能补充遗漏的经验观察，不能算作新增识别分支的独立效应。03 在 fine 下约 96.57% 的低阈值最大 recall，对应 PR 末端 precision 约 3%，也不代表实用工作点已经达到该 recall。

**Resolution test** 固定视图预算，并在独立于最终评估集的校准集上选择阈值，报告 P≥0.90 下的 tiny recall 和全部实例 recall、漏检率及背景误报。抽取预先定义的错误样本，用人工复核区分果叶混淆与其他原因，并报告样本量和不确定性。若无稳定 tiny 增益，将当前贡献改写为背景误报控制或准确率与成本取舍，不声称已解决极小果实召回。

### R2-M4

- **Severity** Major
- **Blocking** Yes
- **Axis** technical soundness / statistical-rigor
- **Issue key** single_seed_validation_selection
- **Claim pointer** 从当前 V12 优先实验推导可发表方法优势或最优模型的准备程度。
- **Evidence pointer** `protocols/citrus_e_v12.yaml` 第 6、22、32 行；`V12_metrics.csv` 六行均为 seed42；`20260918_citrus_e_v12_batch.py` 第 305 至 306 行及 `finish_paired_evaluation`；`audit.json` 的 limitation。
- **Evidence status** located

材料自身把实验定位为单 seed 筛选，六臂都是 seed42，配对推理使用由验证指标选择的 `best_mask.pt`。当前小幅 global AP 差异没有训练重复不确定性，六个候选与多个推理模式的筛选又增加了选择偏差。因此能够选择后续验证对象，尚不能证明该对象可靠优于匹配基线。

协议还明确说明 00/01/02 是 V11 架构与损失重放，不是官方 YOLO 基线。共同证据包不能把这些重放对照直接解释为相对成熟轻量模型的优势，更不能仅以历史记录中的较低数字替代协议匹配。

**Resolution test** 先处理 R2-M1，再固定最终方法、checkpoint 规则、推理模式和工作阈值。在同一协议下为主基线和最终方法提供至少三个 seed 的均值与标准差，在独立测试集报告结果。实例嵌套在图像和捕获组中，评价不确定性宜按组或图像配对估计；它不能替代训练 seed 重复。加入足以说明实际竞争力的匹配官方实例分割基线，不要求以通用顶刊的广泛影响门槛替代应用价值。

### R2-M5

- **Severity** Major
- **Blocking** No
- **Axis** scientific importance / experimental-design
- **Issue key** lightweight_system_budget
- **Claim pointer** REVIEW_PACKET.md 的 light/accurate 定位与协议规定的 actual latency 选择条件。
- **Evidence pointer** 六臂 `initialization_transfer.json` 的 total_parameter_numel；`V12_metrics.csv` 的 global_ms/coarse_ms/fine_ms；paired JSON 的 protocol.view_budget；`loaded_data_summary.json` 的 gpu。
- **Evidence status** located

约 2.28 M 参数说明网络规模较小，但不自动证明所报告最高准确率对应低成本系统。03 的 paired coarse 使用 global+4 个粗视图，fine 使用通常 global+9 个细视图，记录的延迟约 216.5 和 312.2 ms；这些数值来自 RTX 3090 环境。05 虽降至约 2.06 M 参数，其 global/fine 延迟约 89.6/330.2 ms，反而高于 03 的 76.4/312.2 ms，说明参数压缩与当前实现的速度并不一致。

现有延迟足以提醒作者审查实际成本，但因测量边界和重复测量信息有限，不宜把这些差值直接认定为确定的结构速度优势。该问题不阻断一篇诚实报告离线视觉方法的论文，却显著限制轻量部署论断。

**Resolution test** 将单图单次网络推理与含切片、合并、掩膜解码的端到端推理分开，固定硬件、精度、batch、预热和计时边界，重复测量并报告变异。官方基线也接受相同视图预算，同时报告 Params、GFLOPs、显存与延迟。若强调端侧实时应用，再给出相应设备验证；否则限定为小参数模型和明确的推理预算。

## Minor Comments

### R2-m1

- **Severity** Minor
- **Axis** readability for nonspecialists / writing-clarity
- **Affected element** 汇总指标的含义、尺度和单位。
- **Evidence pointer** `V12_metrics.csv` 的 ap、global_ap、global_rmax 和 global_r90 列；批处理脚本第 305 至 306 行；paired 报告 limits。

训练记录中的最佳 mask AP 与 paired common-raster AP 属于不同评估口径，例如 03 的 72.126 与 68.507 不能当作同一口径的前后变化。汇总中 rmax 使用 0 至 1，r90 使用百分数，tiny 列实际是匹配数量而非 AP。给读者的结果表应明确 raster、集合、checkpoint、阈值和单位，并写成“极小实例匹配数/总数”。这是呈现层面的局部问题；若将不同口径数字混用于主结论，则会升级为实质有效性问题。

## 需解决的技术问题与不可评估边界

核心阻断项为 R2-M1 至 R2-M4。R2-M5 限制轻量部署定位，不能仅凭参数量忽略系统成本。已有 YAML hash 一致性值得保留，但 `completed.json` 的 git 为 unavailable，`audit.json` 也明确提示当前 YAML 不能证明历史服务器 Python 相同。应保存并核对实际训练实现快照，不把这一缺失直接断言为服务器执行错误。

完整论文、最接近工作的原始技术材料、最终独立测试集结果、真实叶片误报标注和多 seed 结果不在本次可评估范围。对这些项目，本报告不作不存在、失败或已验证的推断。

## 五轴评价

| 评价轴 | 当前判断 |
| --- | --- |
| Originality | 存在明确的任务专用组合假设和可定位实现。尚缺差异表示及路由必要性的完成对照，外部先前工作差异不可评估。 |
| Scientific importance | 极小果实遗漏和果叶混淆有实际价值。当前最直接证据是误报、召回与计算成本的权衡，尚不能声称三个目标同时解决。 |
| Interdisciplinary readership | 农业视觉与高效实例分割读者可能受益。无需以面向所有学科的影响力作为应用论文是否值得发表的前提。 |
| Technical soundness | 内部实验记录较完整，但正式划分对应、组独立性、单 seed 筛选与独立测试尚未构成闭合证据链。 |
| Readability for nonspecialists | 实现注释较克制。正式稿需用一个清楚的问题解释路径设计，并统一诊断指标，避免连续版本名替代科学论证。 |

内部十二轴覆盖已独立检查。新颖性、机制、设计、统计、复现、数据质量、写作、主张范围及因果归因适用。图表轴仅能评价当前结构化结果呈现，完整论文图表不可评估。临床有效性不适用，证据包未触发人或动物研究伦理问题，不能凭空生成此类疑虑。

## 有条件的准备程度建议与未支持主张

建议把 V12 定位为有价值的单 seed 探索阶段，并先完成正式数据身份核对和机制辨别，再决定论文的最终主线。现有结果不支持直接宣布发表准备完成，也不足以据此断言该应用研究不能发表。

当前不得由这些材料推出“识别专用路由全面优于共享路由”“语义差异证明能辨别果实与叶片”“极小果实召回已稳定提高”“接近 97% recall 的实用系统”“小参数必然更快”或“正式 grouped 数据上的独立泛化优势”。可以保留的结论是，V12 提供了一个有明确代码边界的识别修正假设，并在当前探索性数据和预算下显示了一部分背景误报改善，其增益来源、泛化及成本仍需验证。

本报告已冻结，不依据其他报告的内容修改。
