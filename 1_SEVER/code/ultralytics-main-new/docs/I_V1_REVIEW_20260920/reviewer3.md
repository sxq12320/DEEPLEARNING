# Reviewer 3

独立审阅，2026-09-20。预指定重点为主张边界、任务价值和应用实例分割论文成熟度，兼顾五项共同标准。本报告完成后冻结，不因其他报告的意见重写。

## 审阅范围

输入为 `REVIEW_PACKET.md` 指定的实验记录、汇总及实现，并非完整论文。评估的问题是 V12 当前能否支撑一篇轻量高精度 RGB 未成熟柑橘实例分割论文，尤其是极小果、叶片混淆和召回。以下不预测接收结果，也不以 Nature 的广泛影响门槛否定应用研究。未阅读其他审阅或后续 I 系列设计，未改变网络、数据或结果。

下文 `RUNS` 指 `E:/mastercode/1_SEVER/results/E/E_V12/CITRUS_EV12_PRIORITY_300EP/`，`CODE` 指 `E:/mastercode/1_SEVER/code/ultralytics-main-new/`，`CLEAN` 指 `E:/mastercode/data/orange_yolo_grouped_dedup_20260820/`。短名 02、03、04、05 对应本次六个 seed42 运行中同编号模型。

## 总体判断

V12 是有任务依据、值得继续验证的应用研究候选，但当前证据不足以作为正式论文的中心结果。最重要的限制首先是实际数据成员与正式清洗划分不一致，其次是只有单 seed 验证集筛选。即便暂时只在本批内部比较，分类专用识别分支也没有显示出极小果召回的一致优势。较强的已观察事实是切片加合并显著提高极小果检出，同时增加推理时间和背景误检；03 相对 02 有背景误检减少与部分 AP 改善的迹象。

这并不说明课题缺乏发表价值。应用实例分割论文可以围绕一个明确的果园视觉瓶颈、可信的数据划分和可复现的精度效率收益成立，不需要证明普适视觉理论，也不必为第一篇论文增加控制、深度或实际套袋系统。当前应先建立这个较窄而可靠的证据链。

## 读者和主要优点

农业视觉、果实计数与测量、机器人感知前端、轻量实例分割研究者会关注本任务。可见果实实例掩膜比单纯框检测更适合后续 ROI 提取和形状量化，但本包并未证明下游操作收益，不能把潜在用途写成已经验证的效果。

已有工作值得保留的优点如下。

- 六个模型都完成 300 epoch，内部训练参数差异在 `checks.json.args_differences` 中仅涉及模型名和输出位置，AMP 均关闭。初始化转移有实测记录，03 和 04 的参数量及相等参数数量一致，具备有意义的配对比较基础。
- 结果不只包含 AP，也包含极小果实例数、固定阈值错误桶、R90、经验 PR 和实际延迟。这些证据足以暴露精度、召回和计算代价之间的取舍。
- 实现主动限制自己的解释。`citrus_e_v12.py` 明确说明不是 PID、无稳定性保证、不是完全梯度独立，零初始化约束的是增益而非整个特征幅值。这些边界应保留到论文。
- 03 的细切片结果相对自身 global，极小果匹配从 33/161 增至 90/161，common-raster mask AP 从 68.507% 增至 77.682%。这是可追查的任务相关改善，但不能直接归因于新增识别模块。

## Major Concerns

### R3-M1 正式数据协议尚未对应到实际运行

- Severity 为 Major，Blocking 为 Yes。
- Axis 为 technical soundness，内部分类为 data-resource-quality。
- Claim pointer 为在正式 grouped_dedup 数据上建立可推广、无组间泄漏的轻量高精度结论。
- Evidence pointer 为 `RUNS/V12_03_recognition_route_seed42/completed.json.data`、`loaded_data_summary.json`、`train_original_sources.json`、`paired_fine_eval/paired_metrics.json.records[].image`，以及 `CLEAN/audit/audit_report.json.split.actual`、`CLEAN/audit/split_manifest.csv`。
- Evidence status 为 located。

六个运行记录均有 676 个训练原图和 193 个验证图，但实例数为训练 4,324、验证 1,049。本地正式审计是训练 4,120、验证 1,181。03 的完成记录指向服务器 `orange_yolo`，路径名称本身不能证明使用旧数据，因此又核对了具体成员。将训练原图 basename 与正式 manifest 对齐，126 张归属于正式 val，57 张归属于正式 test；训练成员集合对称差为 366。由 paired records 提取的 193 张验证图中，123 张属于正式 train，验证成员集合对称差为 296。按这些同名图在正式 manifest 的 group_id 映射，有 94 个组同时出现在本次训练和验证集合。

这些是成员层面的不一致，不应以图像总数相同或 `rgb_paired_cache_dedup=true` 消除。组交叉结论依赖同名图仍代表同一原始图，因服务器图像和标注内容哈希尚未核对，不能进一步声称已经确认每个像素相同或推断指标被高估多少。但是，现有证据已经不能证明本次遵守了指定的正式划分。

Resolution test 为逐图对齐服务器实际训练、验证和测试的文件成员、图像与标注哈希及捕获组。若无法证明所用版本等同于清洗版本，正式比较必须在已审计版本重跑，并保证测试组不参与训练或选择。旧结果可以保留为内部筛选，不能作为正式泛化结果。

### R3-M2 目前只能筛选候选，尚不能确立最终性能优越性

- Severity 为 Major，Blocking 为 Yes。
- Axis 为 technical soundness，内部分类为 experimental-design。
- Claim pointer 为 V12 比标准轻量实例分割模型更准确，并且该优势可复现。
- Evidence pointer 为 `CODE/protocols/citrus_e_v12.yaml` 的 `seeds`、`control`、`formal`，`V12_metrics.csv` 的六行 seed42，`audit.json.history` 中 G00 条目的 `args.mask_ratio` 与 `best`，以及 `RUNS/*/args.yaml.split`。
- Evidence status 为 located。

协议自己将本轮定义为 single-seed screening，00/01/02 也是改进架构的 replay anchor，并非官方 YOLO。所有六个已完成运行均为 seed42，指标来自验证集。汇总中的历史 G00 mask AP 为 0.67031，其 mask_ratio 为 4；本批为 2，训练视图和评估栅格又需分别核对。不能用 72.126% 或 77.682% 直接减去该历史数字来声称网络改进幅度。

在同批中，03 与 02 的 global AP 差仅约 0.183 个百分点，fine AP 差约 0.841 个百分点；04 的 fine AP 又比 03 高约 0.627 个百分点。这些单次差值尚不能给出稳定排名。训练曲线平稳或末 20 epoch 均值不能替代不同随机种子的重复，也不能替代未参与方法选择的测试集。

Resolution test 为先冻结数据版本、最终候选、权重选择、主评价口径和推理预算，在同协议下完成官方主基线与最终方法的三个 seed，并在独立测试集给出均值、标准差及按捕获组重采样的不确定性。加入有代表性的跨系列实例分割比较即可，数量服从论文主张，不要求为了规模而堆模型。对架构贡献的比较应控制训练配方，对完整系统贡献则应明确把切片与合并计入方法和成本。

### R3-M3 极小果识别改进不是目前可确立的中心机制

- Severity 为 Major，Blocking 为 Yes，针对将新增分类路由写成极小果召回改进原因的中心主张。
- Axis 为 originality 与 technical soundness，内部分类为 mechanism-evidence。
- Claim pointer 为 03 的识别专用语义差异纠正能够解决极小果漏检，并优于共享纠正。
- Evidence pointer 为 `CODE/citrus_e_v12_suite.py.FACTORS`、`CODE/ultralytics/nn/modules/citrus_e_v12.py.SegmentCitrusEV12.forward_head`，`V12_metrics.csv` 中 02/03/04 的 tiny、AP、R90 字段，及 `CODE/protocols/citrus_e_v12.yaml.factors`。
- Evidence status 为 located。

03 相对直接父对照 02，在 global、coarse、fine 的极小果匹配分别为 33 对 35、69 对 74、90 对 89。专用路由没有在三种预算下一致增加 tiny 检出。04 共享路由在 fine 有 97/161，AP 为 78.309%，R90 为 81.506%，均高于 03 的 90/161、77.682%、79.981%。这些结果并不否定专用路由可能更少产生背景误检，却不支持它同时主导极小果召回和总体精度。

协议中同参数量的 sum evidence 对照 08 尚不在这六个完成结果内，因此不能把差异表示本身的价值从普通附加变换中分离。实现注释提及 TOOD、PIDNet 和 FreqFusion 只说明灵感来源，不能替代实际新颖性与因果贡献的证据。

Resolution test 为将可检验主张收敛为明确的一个效应。若核心是背景抑制，预设同召回或同精度比较；若核心是 tiny 召回，则在独立 tiny 子集和固定推理预算下验证。完成必要的父对照、共享路由及 sum evidence 对照，并按多个 seed 判断效应。若结果仍无一致改善，应删除“该模块解决极小果漏检”的中心主张，保留更窄的工程发现。

### R3-M4 轻量参数量尚不等于应用中的高效推理

- Severity 为 Major，Blocking 为 No。
- Axis 为 scientific importance 与 technical soundness，内部分类为 claim-moderation。
- Claim pointer 为最终系统具备轻量、高精度且适用于实际感知前端的效率优势。
- Evidence pointer 为 `RUNS/V12_03_recognition_route_seed42/initialization_transfer.json`，同目录 `paired_fine_eval/paired_metrics.json.protocol.view_budget`、`summary.*.median_ms`、`summary.*.errors25`，`V12_metrics.csv` 中 03/05，及二者初始化记录。
- Evidence status 为 located。

03 有 2,283,950 个参数，这支持描述模型规模较小。但最高 AP 对应正常图像的 global 加 9 个 fine 视图。它的记录延迟从 global 的 76.39 ms 增至 fine 的 312.15 ms，约 4.09 倍；背景错误从 108 增至 339。fine 下 R90 从 77.979% 到 79.981%，增益约 2.00 个百分点，远小于低阈值最大召回所呈现的幅度。

05 参数减少至 2,059,502，却在现有测量中 global 为 89.64 ms、fine 为 330.22 ms，均未比 03 更快。初始化相等参数数量还从 2,090,928 变为 1,832,496。因此参数压缩、实测速度和训练初始条件不能混为一个“更轻更快”的结论。

Resolution test 为给出完整系统的 AP、R90、极小果召回、延迟和显存取舍，定义硬件、计时范围、预热、batch 与后处理。主表在同视图预算或同实测延迟下比较，并给出单次 global 的独立结果。若最终采用多视图，宜主张可调的精度效率方案；不必预设实时要求，但应先说明应用能够承受的时间预算。

### R3-M5 背景错误桶不能证明解决叶片混淆或遮挡拓扑

- Severity 为 Major，Blocking 为 No，若将“解决叶片混淆”作为唯一核心贡献则需要上调为 blocking。
- Axis 为 scientific importance 与 technical soundness，内部分类为 claim-moderation。
- Claim pointer 为方法改善同色叶果混淆、条带遮挡和接触果分离。
- Evidence pointer 为 03 `paired_fine_eval/paired_metrics.json.limits`、`summary.*.errors25`、`topology_proxy25`、`boundary_iou25_n`，以及 `PR_evidence.json` 中 02/03 对应条目。
- Evidence status 为 located。

03 相对 02 的背景桶确实下降，global 为 108 对 144，fine 为 339 对 379。但背景桶并未区分叶、枝、其他背景、未标注果实和标注边缘误差。文件本身也明确指出这些错误桶不是叶片混淆的语义证明。匹配成功实例上的 boundary IoU 会随被匹配样本组成变化，不能单凭数值上涨证明所有果实边界都改善。fragmented/merged proxy 亦不是经过人工确认的 split/merge 错误真值。

Resolution test 为在与模型输出无关的图像或实例定义上建立叶片相似背景、条带遮挡、接触实例和尺度跨度子集，报告样本量、规则及盲审错误类型。为“抑制叶片误报”至少提供同精度或同召回下、人工确认的叶片 FP 对比。对边界和拓扑主张使用固定 GT 集合与明确匹配规则；若不做这些验证，就把陈述限制为固定阈值背景错误减少的探索性观察。

## Minor Comments

### R3-m1 指标名称和单位需在论文表头统一

- Severity 为 Minor。
- Axis 为 readability for nonspecialists，内部分类为 writing-clarity。
- Affected element 为 `V12_metrics.csv` 中 `ap`、`global_ap`、`global_rmax`、`global_r90` 与 `tiny` 列。
- Evidence pointer 为该 CSV 表头、03 `paired_fine_eval/paired_metrics.json.protocol` 和 `limits`，及 `PR_evidence.json` 中 03 fine/trustedmask 的 `last_real_precision`。

`ap` 是训练验证器的峰值口径，`global_ap` 等为 common-raster 口径；rmax 用小数而 R90 用百分数，tiny 是匹配数而非 AP。03 fine 的 rmax 为 96.568%，对应 PR 末端精度约 2.984%，不能当作可用工作点召回。

Required correction 为给每列标明评估集合、掩膜栅格、单位、阈值、匹配 IoU、权重选择和视图预算。将 tiny 明确写为面积小于 256 平方像素的 640 栅格实例匹配率，避免与 COCO APsmall 混称。R90 写清由验证 PR 提取，若用于部署，应冻结验证阈值后在测试集重新报告实际 precision 和 recall。

## 技术缺口及五轴评估

中心结论成立前需优先解决 R3-M1、R3-M2；若中心结论继续强调新增识别路由解决 tiny 漏检，还需解决 R3-M3。R3-M4 和 R3-M5 可通过实证补强或缩小主张解决。

| 共同评价轴 | 独立判断 |
|---|---|
| Originality | 存在任务驱动的路由设计与对照思路，但已有技术适配的独立贡献尚未由完成的实验隔离。完整文献定位未提供，不能认定首创或重复。 |
| Scientific importance | 极小果遗漏、背景误报和掩膜质量是实际问题。重要性取决于真实数据上的稳定收益及计算代价，无须以跨学科突破为前提。 |
| Interdisciplinary readership | 对农业视觉和轻量分割具有明确相关性，对机器人前端有潜在用途。下游系统性能不在证据范围内。 |
| Technical soundness | 同批内部有较好参数和初始化记录，但正式数据不对应、验证选择与最终评价未分离，当前不足以支撑正式优越性。 |
| Readability for nonspecialists | 代码中边界说明清楚。论文需把多代模块和多种指标收敛为任务、机制、对照、收益与代价的连贯链条；完整行文尚不可评估。 |

内部十二轴覆盖记录如下，不能把不可评估自动转成缺陷。novelty-significance、mechanism-evidence、experimental-design、statistical-rigor、reproducibility、data-resource-quality、claim-moderation、causal-vs-correlative 均适用。figures-and-tables 仅能评估所给汇总表，完整论文图表不可评估。writing-clarity 可评估术语与指标，全文可读性不可评估。clinical-validity 不适用。ethical-governance 的临床、人类受试和动物审批不适用，数据授权与人工合成图披露在本包中不可完整评估，不据此推定违规。

## 有条件的成熟度建议与不支持的主张

建议定位为“已完成候选筛选，尚需正式验证”的应用论文准备阶段。先修复数据证据链并冻结核心问题，优先验证最有根据的收益。当前尚无理由仅凭这个证据包放弃整个研究方向，也无理由认定当前版本已经达到投稿结果成熟度。

现阶段可使用的措辞是“在本次单 seed 验证中，切片合并提高极小实例检出，识别路由显示一定背景错误减少，但存在计算和阈值取舍”。不得升级为正式 grouped_dedup 上稳定领先、色彩不变性、普遍叶片抑制、拓扑问题已解决、控制论稳定性、边缘设备实时性或实际套袋成功率提升。对这些边界的限制来自当前材料，不是对应用研究价值的否定。

报告已冻结。仅依据指定证据包形成，未读取其他独立报告。
