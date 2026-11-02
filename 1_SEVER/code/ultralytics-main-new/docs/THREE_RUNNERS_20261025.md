# 三个前台批量运行入口

日期：2026-10-25。目标：复跑历史78配方、复跑V12高分系统、完成同配方消融。没有新增一个未经验证的模型系列。

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
| RUN_V12_ABLATION.py | 11 | 同配方YOLO11、完整V12和9项消融 |

每个文件内部串行运行，前一个完成才启动下一个；异常或Ctrl+C会停止整个队列。相同入口、相同PROJECT再次运行，会跳过已完成训练，或者从可恢复的last.pt继续。只有best.pt而没有有效优化器状态的半途实验不会被当作可恢复实验。

三个默认PROJECT分开保存。消融中的完整V12会重新训练一次，作为该消融组自身的参照；与BEST入口是相同配方，不代表两种方法。不要在同一张卡上同时运行三个入口。

## 先确认，再长跑

- DRY_RUN=True：仅打印队列，不要求数据和GPU存在。
- SMOKE=True：独立的PROJECT_SMOKE目录，只训练1轮，训练后验证限前2张。正式运行前改回False；smoke不用于论文指标。
- 默认EPOCHS=300、SEEDS=[42]、cache=True。V12默认耐心值300，训练至300轮；78配方保留原patience=100，可能提前停止。
- BATCH=16、WORKERS=4。硬件允许时保持一致；如果改batch，请给比较双方一起改并记录。
- 正式三seed复跑：V12设置SEEDS=[42,43,44]且使用新PROJECT。消融脚本会在首个seed运行11组，其余seed只重复P00/P01，共15次；它不是33次全部消融重复。
- V12的PROJECT会冻结队列和参数。修改SEEDS、ONLY、训练预算或权重后请换PROJECT，避免混入已有结果。单纯再次运行无需改PROJECT。

## 两套训练配方的区别

| 参数/方式 | 历史78基线 | V12 BEST及全部消融 |
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
| 训练视图 | 两个独立运行：全图、混合切片 | 混合切片 |
| 训练期间验证 | 原图、EV6，GT栅格input/2 | 原图、EV6，GT栅格input/2 |

78配方来自 results/A_baselines/old_data_runs/001_3_yolo11-seg_adamw/args.yaml。名称“78”指历史参数来源，并非承诺新数据也得78，也不是恢复旧数据划分。为跨模型统一验证，本轮使用EV6的input/2验证栅格，不宣称逐字复现旧版验证器。

V12完整结构对应 0_orange_yaml/E_V12_series/V12_04_shared_route_control.yaml。P01是其单类消融副本，增加明确的预训练层映射标识；不是新结构。历史多视图高分支持将它作为完整系统候选，尚不证明它在新数据上仍最强。

**不能把V12 BEST与78基线的差值全部归因于结构。** 请用消融组内P01与P00回答同配方结构是否有效。不同初始化、AMP、数据视图、推理视图均须在论文中说明。

## 切片具体做了什么

基线默认TRAIN_VIEWS="both"：

1. C00_yolo11n_78_seed42：原图训练，保留历史78训练参数。
2. C00_yolo11n_78_mixed_seed42：同一训练参数，仅改成原图/粗切片/细切片混合抽样。

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

## 11项V12消融

YAML均在 0_orange_yaml/PAPER_ABLATION/，可以逐个定位。为减少固定顺序影响，队列按seed确定性打乱，编号不等于实际开跑顺序。

| 编号与模型 | 对照问题 |
|---|---|
| P00_yolo11n_control | 同预训练、AMP、混合训练、增强和评估的原生YOLO11n参考 |
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

P10是联合控制，不能将其差值解释为某个单项的贡献。P06控制输出分辨率，尽量区分“更多像素”和“学习式细化”的作用。P00/P01的同配方比较与P02–P10消融分别回答整体效果和组成部分作用；单seed结果需后续重复验证。

## 结果去哪里看

- 基线项目：CONTROL78_RESULTS.csv，AP/Recall列名带_percent，为百分数。
- V12项目：paper_results.csv，diagnostic_mask_ap50/ap5095为0–1小数，乘100才是百分数。
- 每个模型：results.csv（逐轮原图验证）、weights/best_mask.pt（按原图Mask AP50-95选出的权重）、eval_val/paired_metrics.json（同权重的原图和切片结果）。
- V12每次首次训练记录initialization_transfer.json：训练更新前与预训练文件的参数逐项对照。预训练加载比例并非模型能力指标。
- 控制同一PROJECT重复写入的文件锁仅保护结果文件，不检查/抢占GPU；使用不同项目可以各自在不同GPU运行。

所有结果先作为实测结果记录；不把历史78、历史V12多视图高分当成本轮保证值。
