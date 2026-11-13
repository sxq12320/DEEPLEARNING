# 三个前台批量运行入口

2026-11-05 更新。按用户要求，当前批量消融入口保留历史 78 配方的整图 P00，增加相同超参数的混合切片 P00；用已验证的 V12_04 最佳配方训练 P01，并让 P02–P10 沿用 V12 配方，检验各组成部分。两版 P00 在相同验证模式下比较训练切片的作用；P00 对 P01 回答完整方案能提高多少，所得差值包含初始化、AMP、训练视图和结构等因素。

## 直接运行

上传整个当前 ultralytics-main-new 文件夹。分别打开三个入口，修改顶部 DATA、DEVICE、PROJECT。V12两个入口的 PRETRAINED 默认指向入口同目录的 yolo11n-seg.pt，请一并上传。DATA 填你实际使用的数据 YAML；没有划分清单确认和GPU占用检查。

在VS Code选择正确Python环境后，点击“运行Python文件”；或者进入代码目录执行下面任意一个命令。三个文件是独立入口，不必同时启动。

~~~bash
python RUN_YOLO11_BASELINE78.py
python RUN_V12_BEST.py
python RUN_V12_ABLATION.py
~~~

| 入口 | 默认训练次数 | 默认工作 |
|---|---:|---|
| RUN_YOLO11_BASELINE78.py | 2 | YOLO11n全图训练、YOLO11n混合切片训练；每个权重分别做全图/切片验证 |
| RUN_V12_BEST.py | 1 | V12_04完整结构，显式预训练、混合切片训练，评估完整多视图系统 |
| RUN_V12_ABLATION.py | 12 | 两版 P00（整图、混合切片）均用历史78配方，P01 最佳V12配方，P02–P10 按V12配方训练 |

每个文件内部串行运行，前一个完成才启动下一个；异常或Ctrl+C会停止整个队列。相同入口、相同PROJECT再次运行，会跳过已完成训练，或者从可恢复的last.pt继续。只有best.pt而没有有效优化器状态的半途实验不会被当作可恢复实验。

三个默认PROJECT分开保存。消融中的完整V12会重新训练一次，作为该消融组自身的参照；与BEST入口是相同配方，不代表两种方法。不要在同一张卡上同时运行三个入口。

当前消融入口默认 `BASELINE_RECIPE="legacy78"`、`BASELINE_TRAIN_VIEWS="both"`，输出到 `/data/sxq/results/PAPER_V12_VS78_SLICE_20261105`。P00 两版均为随机初始化、AMP=True，分别保存为 `P00_yolo11n_control_seed42_amp1` 和 `P00_yolo11n_control_mixed_seed42_amp1`；P01 和所有删减模型使用预训练、混合切片、AMP=False，保存名带 `amp0`。新增 P00 共用原 P00 YAML，没有修改网络结构。其他训练参数见下表。

如当前只关心两端差值，将 `ONLY="P00_yolo11n_control,P01_v12_full"`，即可串行训练三次（两版 P00 加 P01）。保持默认空字符串会跑全部 12 组。只补 P00 对照可设置 `ONLY="P00_yolo11n_control"`。`BASELINE_TRAIN_VIEWS` 可选择 `whole`、`mixed`、`both`。需要恢复之前同配方比较时，在一个新项目设置 `BASELINE_RECIPE="matched"`（此时忽略该视图选项，P00 按 V12 混合训练一次）；续跑旧项目则按该项目最初保存的模式和队列设置，不向旧结果目录直接追加不同配置。

## 先确认，再长跑

- DRY_RUN=True：仅打印队列，不要求数据和GPU存在。
- SMOKE=True：独立的PROJECT_SMOKE目录，只训练1轮，训练后验证限前2张。正式运行前改回False；smoke不用于论文指标。
- 默认EPOCHS=300、SEEDS=[42]、cache=True。V12默认耐心值300，训练至300轮；78配方保留原patience=100，可能提前停止。
- BATCH=16、WORKERS=4。硬件允许时保持一致；如果改batch，请给比较双方一起改并记录。
- 正式三seed复跑：V12设置SEEDS=[42,43,44]且使用新PROJECT。当前消融脚本会在首个seed运行12组，其余seed只重复两版P00和P01，共18次。设置REPEAT_ABLATIONS=True且ONLY为空则跑36次。
- V12的PROJECT会冻结队列和参数。修改SEEDS、ONLY、训练预算或权重后请换PROJECT，避免混入已有结果。单纯再次运行无需改PROJECT。
- 需要三个种子的逐模块对照时，将消融入口的 `REPEAT_ABLATIONS=True`。例如在新项目设置 `SEEDS=[42,43,44]`，并在 `ONLY` 填入 P00、P01、P02、P03、P06 的完整名称，两版 P00 加其他四个模型将串行训练 18 次。若已经用 seed 42 完成这六组，可在新项目仅填写 `SEEDS=[43,44]` 补 12 次。
- 已有同一项目和种子的 P01、P02、P03、P06 后，打开 `VIS_E12_ABLATION.py`，设置数据和 P01 权重，将 `ALL_RETRAINED=True` 并运行，即可在全图上输出三组逐果实对照图、轮廓统计和实测 PR。保留默认设置，仅用一个 P01 权重，也可先做指定通路的冻结干预。

## 两套训练配方的区别

| 参数/方式 | 历史78基线及当前P00 | V12 BEST及P01–P10 |
|---|---|---|
| 模型初始化 | YAML随机初始化，不加载pt | 显式加载同一yolo11n-seg.pt；新层按声明初始化 |
| AMP | True | False |
| dropout | 0.1 | 0 |
| train mask_ratio | 4 | 2 |
| copy_paste | 0 | 0.3 |
| 优化器 | AdamW | AdamW |
| lr0 / lrf | 0.001 / 0.01 | 0.001 / 0.01 |
| weight_decay | 0.0005 | 0.0005 |
| 输入尺寸 / batch | 640 / 16 | 640 / 16 |
| cache / seed | True / 42 | True / 42 |
| 训练视图 | 整图版和混合切片版各训练一次；两版超参数相同 | 混合切片 |
| 早停耐心值 | 100，保留历史设置 | 300，默认跑满300轮 |
| 训练期间验证 | 原图、EV6，GT栅格input/2 | 原图、EV6，GT栅格input/2 |

78配方来自 results/A_baselines/old_data_runs/001_3_yolo11-seg_adamw/args.yaml，活动训练参数已逐项核对。历史 args.yaml 虽记录 `pretrained: true`，实际入口从 YAML 构建模型，未显式加载预训练文件；这里用 `pretrained=False` 明确恢复随机初始化行为。数据路径填服务器上你实际使用的清洗数据。评分沿用当前统一 EV6 和配对 public640，因此“78”表示历史配方的来源，新数据结果由实测决定。

V12完整结构对应 0_orange_yaml/E_V12_series/V12_04_shared_route_control.yaml。P01是其单类消融副本，增加明确的预训练层映射标识；不是新结构。历史多视图高分支持将它作为完整系统候选，尚不证明它在新数据上仍最强。

当前 P00/P01 比较两套完整方案。P01/P02–P10 仍可用于定位被删除组成部分的作用。若需要整体方法的同配方参考，使用 `BASELINE_RECIPE="matched"` 另跑 P00；两种配置均记录初始化、AMP、训练视图和验证模式。

## 切片具体做了什么

独立基线入口默认TRAIN_VIEWS="both"：

1. C00_yolo11n_78_seed42：原图训练，保留历史78训练参数。
2. C00_yolo11n_78_mixed_seed42：同一训练参数，仅改成原图/粗切片/细切片混合抽样。

消融入口也默认训练对应的两版 P00，使用同一实现和同一 78 配方。两版的学习率、优化器、AMP、dropout、mask_ratio、batch、seed、初始化和早停条件均一致。相同早停规则可能因训练表现不同而在不同轮数停止，需按实际完成轮数记录。

混合训练按原图抽样，每张源图每轮提供一个逻辑样本；全图、0.6边长粗视图、0.4边长细视图的抽样概率为0.5/0.25/0.25。不是把所有切片都当独立图像无限扩大epoch。沿用现有多边形裁剪与回退规则，派生视图存在输出目录的_prepared_multiscale中，不更改原始图像/标签。切片缓存会额外占用磁盘和RAM。

准备后打印TRAIN VIEW AUDIT，并保存training_view_audit.json，记录实际保留切片数、仅能保留全图的源图数。若多边形裁剪规则导致所有切片均不可用，会明确报错，不将纯全图训练误报为切片训练。这是检查实验是否实际包含切片，不是数据划分清单确认；不会自动修正或删除标签。

每个训练好的权重都在同一原图验证集上评估：

| 模式 | 含义 |
|---|---|
| global | 仅原图一次前向，不切片 |
| fixed_box | 原图+细切片，按框合并 |
| crossmask | 原图+细切片，按掩膜跨视图合并 |
| trustedmask | 原图+细切片，加入现有边界可靠性处理 |

通常是1张全图+9张细切片。切片是多次推理，不能把其延迟当成一次前向延迟。验证GT仍属于原图，不在切碎的验证标签上算分。训练中每轮只做常规EV6验证；较贵的多视图对照在训练结束后执行，并缓存候选预测以便重跑复用。

V12两个入口还输出已有后处理机制对照，不需要为每个后处理方式重训。默认不运行COCO第二评估器，不读取test来挑模型。此处多视图结果是明确标注的public640诊断口径，不冒充官方COCO AP。

## 11种结构、12组默认训练

YAML均在 0_orange_yaml/PAPER_ABLATION/，可以逐个定位。为减少固定顺序影响，队列按seed确定性打乱，编号不等于实际开跑顺序。

| 编号与模型 | 对照问题 |
|---|---|
| P00_yolo11n_control | 默认历史78配方的整图版与切片版各一次；matched模式下改用V12配方一次 |
| P01_v12_full | V12完整系统参考 |
| P02_no_persistent_detail | 移除持续细节分支及其连接，检验细节保留路径 |
| P03_no_recognition | 关闭专门的共享识别路径；基础检测分类仍保留 |
| P04_classification_only | 识别路径仅服务分类，比较完整共享路由 |
| P05_full_pan_neck | 恢复完整PAN回传结构，比较V12不对称颈部 |
| P06_bilinear_only | 保留相同高分辨率输出栅格，用双线性插值替代学习式细化 |
| P07_no_boundary_loss | 关闭边界辅助损失 |
| P08_no_neighbor_loss | 关闭邻近实例辅助约束 |
| P09_no_tiny_dice | 关闭小目标Dice辅助项 |
| P10_no_extra_supervision | 联合关闭小目标、边界、邻近辅助项及质量监督/校准 |

P10是联合控制，差值对应整套额外监督与校准。P06保持相同输出分辨率，检验可学习细化。P00/P01默认比较历史78配方和最佳V12完整系统，P01与P02–P10分别比较组成部分作用；单seed结果需后续重复验证。

## 结果去哪里看

- 基线项目：CONTROL78_RESULTS.csv，AP/Recall列名带_percent，为百分数。
- V12项目：paper_results.csv，diagnostic_mask_ap50/ap5095为0–1小数，乘100才是百分数。
- 两版 P00 完成后生成 `P00_SLICE_COMPARISON.csv`，按同一 seed、split 和验证模式比较整图训练与切片训练；四种验证模式各一行。`*_delta_pp` 为切片训练减去整图训练的百分点差，不能把训练切片与推理切片混为一谈。
- 当前legacy78模式还会生成 `P00_P01_COMPARISON.csv`，对每一版 P00 在同一seed和split上输出三行预设比较，两版齐备共六行。`baseline_train_views` 区分两版；`requested_full_system` 为该版P00的整图推理与P01完整切片系统，另两行分别是双方均整图推理、双方均十视图推理。含两端Mask AP50/AP50–95/Recall/R@P90的百分数和提升百分点（`*_delta_pp`）。两端结果未齐时不生成该配对，达到零提升或负提升时原样记录。
- `paper_protocol.json` 在legacy78模式逐模型保存训练参数；`paper_results.csv` 同时标明 `training_recipe` 和 `train_views`。所有模式均使用同一输入数据划分与评估函数。
- 每个模型：results.csv（逐轮原图验证）、weights/best_mask.pt（按原图Mask AP50-95选出的权重）、eval_val/paired_metrics.json（同权重的原图和切片结果）。
- V12每次首次训练记录initialization_transfer.json：训练更新前与预训练文件的参数逐项对照。预训练加载比例并非模型能力指标。
- 控制同一PROJECT重复写入的文件锁仅保护结果文件，不检查/抢占GPU；使用不同项目可以各自在不同GPU运行。

所有结果先作为实测结果记录；不把历史78、历史V12多视图高分当成本轮保证值。
