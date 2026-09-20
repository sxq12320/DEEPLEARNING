# Reviewer 1

技术可靠性与实验设计独立审阅，2026-09-20。报告完成后冻结。

## 审阅范围

仅依据 REVIEW_PACKET.md 列出的实验记录、实现、配置和本地正式划分审计。未读取其他审稿报告，未运行训练或修改网络。材料不是完整论文，因此以下是对当前证据能否支撑应用研究论文的判断，不是期刊录用决定，也不以 Nature 的广泛影响力要求否定农业视觉研究。

下文 `RUN` 指 `E:/mastercode/1_SEVER/results/E/E_V12/CITRUS_EV12_PRIORITY_300EP/`，`CODE` 指 `E:/mastercode/1_SEVER/code/ultralytics-main-new/`，`PACKET` 指 `CODE/docs/I_V1_REVIEW_20260920/`，`DATA` 指 `E:/mastercode/data/orange_yolo_grouped_dedup_20260820/`。JSON 指针均为实际键路径，不是虚构稿件行号。

## 总体评价

现有结果足以支持继续研究，尚不足以建立“轻量、高精度并改善极小果和叶片混淆”的完整论文结论。最优先的问题是数据划分。实际 V12 训练成员与本地 grouped_dedup 正式划分不一致，而且按正式分组清单映射，V12_03 的训练与验证集有 94 个组交叉。这是可核对的成员证据，不能仅靠运行目录中同样的 676/193 张计数排除。

技术上，V12 有可理解且可消融的假设。分类专用修正路径确实在代码中实现，03 与 04 的参数量和已迁移参数量相同，能进行有价值的路由比较。但是观察到的优势不是一致的。03 的背景错误更少，04 的同预算 AP 更高，极小果召回也没有稳定支持分类专用路由。当前最稳妥的结论是存在误检、召回、精度和推理成本之间的权衡。

## 读者与主要优点

农业机器人视觉、果园实例分割、小目标识别和资源受限视觉部署研究者会关心该工作，因为识别未成熟果实时，局部纹理与语义上下文的冲突有实际意义。

证据包有六个完成的 300 epoch 运行，提供完整曲线、同图配对推理、初始化迁移和实际加载数量。`checks.json.args_differences` 只列 model/name/save_dir，六组的 AMP 均为 false。`citrus_e_v12.py` 明确区分分类专用路由和共享路由，并说明共享骨干与 TAL 仍有梯度耦合。这种限定是恰当的。配对报告也主动声明统一栅格 AP 与官方验证器 AP 不同，错误桶不是叶片混淆的语义证明。这些做法有利于可信复现。

## Major Concerns

### R1-M1 实际划分不满足独立验证条件

- Severity 为 Major，Blocking 为 Yes。
- Axis 为 technical soundness，技术检查轴为 data-resource-quality / experimental-design。
- Claim pointer 为 `REVIEW_PACKET.md` 第一段所述可发表的轻量高精度 RGB 实例分割贡献，以及其要求核对数据等价性的共同标准。
- Evidence pointer 为 `RUN/V12_03_recognition_route_seed42/train_original_sources.json` 的 source，`paired_fine_eval/paired_metrics.json.records[].image`，`DATA/audit/split_manifest.csv` 的 image/new_split/group_id，`DATA/audit/audit_report.json.split.actual`，以及该运行 `loaded_data_summary.json`。
- Concern 为以图像文件名连接上述成员记录与正式分组表，676 张训练图中只有 493 张属于正式 train，126 张属于正式 val，57 张属于正式 test。193 张运行验证图中，123 张属于正式 train，45 张属于正式 val，25 张属于正式 test。所有图像名称均能匹配清单。按其 group_id，运行训练与验证共有 94 个组交叉。例如 G0014 的 IMG_0019.jpg 在运行验证集，IMG_0020.jpg 在运行训练集，而正式清单将二者同归 val。运行还记录训练原始实例 4324、验证实例 1049，正式审计分别为 4120、1181。差异不能仅解释为服务器路径改名。
- Why it matters 为当前验证不能提供按采集组隔离后的泛化证据，且这批权重已经训练过正式 test 中的 57 张图。直接拿现有权重在该 test 上补评估不能恢复独立性。上述结论依赖清单中稳定图像 ID 的映射，未获得服务器文件内容哈希，因此不进一步声称内容完全相同或每个分组一定构成近重复。
- Resolution test 为导出服务器原图及标签哈希，复核成员映射；确认后使用正式清单，从相同的非本任务预训练权重重新训练基线和最终方法，隔离 train/val/test，留存实际加载成员与内容哈希。旧结果保留为探索性证据，不能作为正式独立测试成绩。

### R1-M2 分类专用路由的核心优势尚未建立

- Severity 为 Major，Blocking 为 Yes，针对将分类专用语义修正列为核心有效机制的论文主张。
- Axis 为 technical soundness / originality，技术检查轴为 mechanism-evidence。
- Claim pointer 为 `CODE/protocols/citrus_e_v12.yaml.factors.recognition` 与 `shared_control`，以及 `citrus_e_v12.py.SegmentCitrusEV12` 的路由设计。
- Evidence pointer 为 `PACKET/V12_metrics.csv` 的 02/03/04 行，`CODE/citrus_e_v12_suite.py.FACTORS`，三个运行的初始化迁移记录。
- Concern 为 03 相对直接父对照 02，global AP 只由 68.3241 升至 68.5071，coarse 由 75.6461 升至 76.0248，fine 由 76.8412 升至 77.6823。03 的 tiny 匹配数在 global/coarse 分别为 33/69，低于 02 的 35/74；fine 则为 90 对 89。共享路由 04 在三种预算的 AP 为 68.6076/76.5335/78.3091，均高于 03，fine tiny 为 97，高于 03 的 90。03 的优势主要是背景错误更少和部分预算下更好的 P≥0.9 召回。尚未提供 08 的同参数 sum evidence 结果。
- Why it matters 为现有数据不能证明“分类专用路由优于共享修正”或“显式差异表示是收益来源”，也不能把 fine 相对 global 的大幅收益全部归于网络设计。代码中的零起始和有界增益是结构事实，既不证明训练后的语义机制，也不证明极小果召回改善。
- Resolution test 为在有效划分上预先规定主要指标，比较 02/03/04，并加入 08 的差异与求和对照；报告同预算、同校准条件下的配对差值。若 03 仅降低误检，则将贡献收窄为误检与召回的权衡，而不是声称全面优于共享路由。

### R1-M3 单次筛选不能提供稳定优势与最终性能估计

- Severity 为 Major，Blocking 为 Yes。
- Axis 为 technical soundness，技术检查轴为 statistical-rigor / experimental-design。
- Claim pointer 为 `REVIEW_PACKET.md` 的论文就绪性问题，及 `protocols/citrus_e_v12.yaml.formal` 所列最终三 seed 要求。
- Evidence pointer 为 `PACKET/V12_metrics.csv` 六行均为 seed42，`protocols/citrus_e_v12.yaml.seeds`，`RUN/*/args.yaml` 的 split=val，以及配对报告 limits 中的 exploratory validation ablation。
- Concern 为六组均只有一次训练，模型、epoch 和推理配置均依赖验证结果筛选；当前小幅 AP 差值没有独立测试或训练随机性区间。last20 是相关 epoch 的平均，不是独立重复，也不能代替 seed 间不确定性。
- Why it matters 为最高观察值和稳定可复现提升是不同命题。训练轨迹波动及多个候选的选择效应可能影响很小的结构差值，不能用一次运行确定优劣。
- Resolution test 为修正划分后，冻结模型与选择规则，至少对主基线和最终方法运行计划中的三个 seed，报告均值、标准差和每 seed 结果；独立测试仅用于最终评估。配对区间应按采集组重采样，不能把同组图或同图实例当作独立样本。模型选择阶段与最终估计阶段应分开报告。

### R1-M4 轻量与精度的论证需要完整推理系统的公平比较

- Severity 为 Major，Blocking 为 Yes，针对“轻量高精度”这一组合主张。
- Axis 为 scientific importance / technical soundness，技术检查轴为 experimental-design。
- Claim pointer 为 `REVIEW_PACKET.md` 第一段的 light/accurate，以及 `protocols/citrus_e_v12.yaml.formal` 中的 actual latency。
- Evidence pointer 为 `RUN/V12_03_recognition_route_seed42/initialization_transfer.json`，其 fine 配对报告 `protocol.view_budget`，`PACKET/V12_metrics.csv`，`protocols/citrus_e_v12.yaml.control`。
- Concern 为 03 有 2,283,950 个参数，但其 global AP 68.5071 与 fine AP 77.6823 对应不同系统预算。fine 记录为 global+9 fine，延迟中位数由约 76.39 ms 升至 312.15 ms，约为 4.09 倍。当前六组均为继承 V11 的内部对照，协议也明确写明不是官方 YOLO 基线。它们不能单独证明整套方法相对标准网络的参数、运算和延迟优势。05 的参数虽降至 2,059,502，但预训练相等参数也由 2,090,928 降至 1,832,496，其精度变化同时涉及结构和初始化覆盖。
- Why it matters 为参数量不能替代切片、多视图、掩膜解码与合并的实际成本；单图网络轻量不等于达到该 AP 的系统轻量。跨预算或跨验证器比较会夸大收益归因。
- Resolution test 为提供相同划分与训练配方的标准主基线，在 global、同切片预算和相同合并协议下分别比较；补充适当跨系列基线。统一硬件、精度、批量、热身和计时边界，报告 Params、GFLOPs、端到端中位数及尾部延迟。将单次前向和完整切片系统分成两条准确率与成本曲线。05 应明确标注预训练覆盖变化，必要时增加初始化匹配控制。

### R1-M5 背景错误减少还不能证明解决叶片混淆

- Severity 为 Major，Blocking 为 No。它影响叶片机制主张，不否定一般实例分割研究本身。
- Axis 为 scientific importance / technical soundness，技术检查轴为 claim-moderation / mechanism-evidence。
- Claim pointer 为 `REVIEW_PACKET.md` 的 foliage confusion 焦点与 `citrus_e_v12.py.EV12RecognitionCorrection` 对局部对比和语义上下文的设计动机。
- Evidence pointer 为 `PACKET/PR_evidence.json` 中 02/03 的 global errors25 和 tiny_matched，以及 03 fine 配对报告 limits。
- Concern 为 global 背景错误从 02 的 144 降至 03 的 108，同时 tiny 匹配从 35 降至 33。错误桶未给出叶片、枝条、漏标果或其他背景的语义组成，且使用固定置信度，可能混入分数校准和召回下降的影响。原始报告已经明确该桶不是 leaf confusion 的语义证明。
- Why it matters 为“少报了一些背景”与“学会区分叶片和果实”需要不同证据。不能将通用背景桶直接重新命名为叶片误检率。
- Resolution test 为盲法标记固定难例集合中的误检类型，并在匹配总体召回或匹配 precision 的工作点比较叶片误检；同时报告极小果漏检，避免通过抑制所有低分候选制造貌似的识别改善。不做此分析时，将文字限定为背景错误诊断。

## Minor Comments

### R1-m1 指标名称和量纲应自解释

- Severity 为 Minor，Axis 为 readability for nonspecialists。
- Claim pointer / Affected element 为 `PACKET/V12_metrics.csv` 的 ap/global_ap/fine_ap、global_rmax/global_r90 和 tiny 列。
- Evidence pointer 为 CSV 表头以及配对报告 protocol.mask_raster、protocol.mask_iou、limits。
- Issue 为表中 rmax 使用 0 至 1，r90/AP 使用百分数；tiny 是 161 个目标中的匹配数，不是 AP。训练验证器 AP 与 640 统一栅格 AP 也有差别，容易混读。
- Required correction 为统一比例单位，并为每列标注 checkpoint、split、mask raster、置信度、匹配 IoU、视图预算。tiny 应写作 matched/161 或 recall，而不是笼统“小目标精度”。Rmax 应保留最低置信度限定，不能把 PR 作图端点解释为真实 100% 召回。

### R1-m2 历史 Python 实现的溯源需要独立列明

- Severity 为 Minor，Axis 为 technical soundness，技术检查轴为 reproducibility。
- Claim pointer / Affected element 为使用当前源码解释这批历史运行。
- Evidence pointer 为 `completed.json.git` 的 unavailable、`PACKET/checks.json.yaml_hash_match`、`audit.json.limitation`，以及 `20260918_citrus_e_v12_batch.py` 中 implementation_sha256.json 的写入逻辑。
- Issue 为当前 YAML 哈希一致不等于历史 Python 文件一致。脚本设计了实现快照，但该审阅材料中的运行目录未包含项目级 `_protocol` 快照，不能据此断言历史运行失败，也不能把当前代码的每个细节视作已确认的历史行为。
- Required correction 为归档并引用服务器对应 implementation_sha256.json、依赖版本、命令及数据快照，将“当前代码可见”与“历史运行已核实”分别标记。如快照缺失，应在正式重跑时补齐。

## 五轴判断

| 评价轴 | 当前判断 |
|---|---|
| Originality | 有可检验的局部设计贡献，差异表示与任务路由的必要性尚未得到鉴别性证据。当前材料不足以裁定完整文献层面的首创性。 |
| Scientific importance | 未成熟果实的小目标和背景混淆问题有明确应用价值，数据独立性与可部署成本决定这种价值能否落到论文结果。 |
| Interdisciplinary readership | 农业视觉、机器人感知和高效实例分割读者相关；无需预设跨所有学科的普遍意义。 |
| Technical soundness | 内部对照和记录质量有优点，但实际分组交叉、单 seed 验证筛选以及预算差异阻断正式性能结论。 |
| Readability for nonspecialists | 源码对概念边界表述较克制；需要将结构、输入放大、后处理、评估栅格和工作点分开说明。完整论文可读性不可评估。 |

## 必须处理的技术问题与建议姿态

R1-M1 至 R1-M4 是当前建立主要论文结论前应处理的事项。最先处理实际数据成员和分组，随后进行同协议基线与预先规定的路由比较，再形成独立测试及不确定性报告。R1-M5 可以通过补充定向证据或收窄叶片识别措辞解决。

建议姿态为继续推进，但当前应定位为有价值的探索性筛选结果，暂不宣称已经得到稳定、轻量且全面改善极小果识别的最终方法。不支持将“单次最高 AP”“同一权重增加九个视图的收益”“背景桶下降”“零初始有界增益”分别替换成稳定优势、网络机制、叶片理解或闭环稳定性证明。

内部覆盖检查已考虑创新、机制、设计、统计、复现、数据质量、表格、表述及因果归因。临床有效性不适用；材料没有激活需要判断的临床或人体动物伦理主张，不据此虚构问题。外部果园泛化、完整文献比较及整篇稿件质量在本证据包内不可评估。
