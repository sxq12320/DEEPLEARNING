# J V5：保留有效细节，用任务监督补偿共享预测塔的容量损失

2026-10-15。状态：**新实验代码，不是训练结论**。真实构建/梯度/序列化与smoke结果以同次交付的测试日志为准；本文件不捏造AP、显著性或提速。

**最新优先级：先精读论文，再冻结方案。** 用户随后明确要求先泛读、再精读，以论文架构为最高优先级。因此本轮代码定位为受控候选，不是已确定的最终网络；暂不建议立即启动all×300。先阅读`_work/20261015_research_rebuild/README.md`及精读卡。精读新发现的OverLoCK宏观分支尚未迁入学生图，不把该论文的思想冒称已实现。

## 1. 为什么这次不是重启已经失败的替换

历史原始复核在 `E:/mastercode/_work/20261015_research_rebuild/history/README.md`：311份表按内容去重297份，206份paired文件共813个推理臂。

- V9窄Proto：训练AP50–95 71.607→71.070%，同预算fine 77.273→76.566%。因此**不将宽原型解码器直接换成窄DW解码**当默认轻量化手段。
- V11 RepMix/native融合组合出现下降；J V1自研紧凑主干也没有保持识别性能。因此不再仅靠替换整块主干或泛用注意力。
- V12 route2保留有效细节/语义结构，同权重global→fine trusted AP50–95 68.608→78.309%、tiny33→97/161；但推理93.9→338.7ms且背景误检增多。该结果是旧val上的推理证据，不是正式COCO AP，也不是蒸馏收益。
- 对已经实现但**未正式训练的J65**，现在加入没有老师、仅mask监督、mask+class监督三种同图对照。核心问题从“模块是否漂亮”转向“压缩预测塔丢掉的实例支持关系能否被固定老师追回”。

这是有反例约束的受控试验，不能保证大幅涨点；小参数也不保证训练更快。J65同时继承了J4的窄语义、投影融合及共享塔，其结构整体收益需要J82对J81检验；蒸馏贡献必须用**相同J65图**的J83/J84对J82判断，不能把J84与J81之差全归因于蒸馏。

## 2. 六个实验臂与训练监督

| 新编号 | 部署图来源 | 新增训练监督 | 回答的问题 |
|---|---|---|---|
| J80_yolo_control | J60：原YOLO11n-seg | 无KD | 相同输入/配方/EV6的基础对照 |
| J81_v12_teacher | J61：V12_04复现图 | 无KD | 有效细节语义结构锚点；seed42训练后冻结 |
| J82_compact_control | J65：窄语义+共享任务塔 | 无KD | 紧凑图本身保留多少能力 |
| J83_compact_mask_kd | 与J82完全相同 | mask KD权重.5 | 实例掩膜关系能否补偿压缩 |
| J84_compact_joint_kd | 与J82完全相同 | mask .5 + class .1 | 识别与边界监督是否互补 |
| J85_thin_joint_kd | J67：细节16→12通道 | 同J84 | 进一步压缩细节的代价；可选 |

老师只在训练时对**同一增广视图**做前向，冻结且不参与反传。不使用test，不生成新GT，不给未匹配区域伪正标签。它不是J3的原图高分辨率特权老师，也不额外跑10视图老师。通过GT实例支持、老师框/掩膜可靠性过滤、固定点网格监督，避免把全背景或不可靠果叶判断整张复制给学生。默认前10epoch不加KD，其后20epoch线性升权；每图最多32个实例，优先小果；点网格12×12，teacher_min_score=.25、teacher_min_iou=.5。细节以 `citrus_j5_distill.py` 为准。

老师由trainer持有；部署checkpoint不应含老师模块。没有老师字段的验证batch只走原监督损失；**验证仍是I/E/J使用的EV6Validator，mask_ratio=2，不引入baseline COCO验证器，不改变PR/AP定义**。学生推理图不增加老师，但训练会多一次老师前向、更占显存，且老师本身也需独立训练。

`j5_distillation.csv`逐epoch记录训练曝光数、考虑数、接受数、框/分数拒绝、采样掩膜拒绝、small/non-small接受率、KL均值与ramp。这里small是**当前增广输入bbox面积<32²**，不是公共640的tiny-mask面积<16²，也不是唯一实例数。可靠性门只是GT支持的筛选，不证明teacher完全正确；HINTED提出的hard-instance选择偏差在此仍可能存在。门拒绝的GT保留原监督，不灌入新伪标。

采样采用一次打包grid，避免按实例扩展原型后反传N份dense梯度；匹配按图向量化，不在每个果实上做CUDA标量同步。仍须服务器实测KD开/关的step时间与峰值显存，CPU通过不代表GPU速度通过。

## 3. 最简单的运行方法

复制整个 `1_SEVER/code` 到服务器，在原环境中打开主代码目录，选择已有Python解释器。编辑 `RUN_CITRUS_J5.py` 顶部：

```python
DATA = "/data/sxq/datasets/orange_yolo/data.yaml"  # 按本次用户指定，不会偷偷改成另一划分
DEVICE = "1"  # 物理卡1，进程中映射到逻辑cuda:0
PROJECT = "/data/sxq/results/J/v5/CITRUS_J5_ALL_300EP_20261015"
SUITE = "all"
EPOCHS = 300
SEEDS = [42]
TEACHER = "auto"
```

然后在VSCode点击“运行Python文件”三角形，或终端：

```bash
python RUN_CITRUS_J5.py --dry-run
python RUN_CITRUS_J5.py
```

dry-run会**真实构建选中网络**但不读取数据、训练或准备缓存。正常运行前先在**另一新PROJECT**将EPOCHS改1–3做smoke；50轮筛选同样用新PROJECT。默认all为6个模型×300epoch=1800epoch，并不快；正式3seed为18个运行×300epoch。`priority`/`screen`为J80/J81/J82/J84四组，**screen这个名字不自动把EPOCHS改为50**。

自动老师固定为本项目 `J81_v12_teacher_seed42/weights/best_mask.pt`。涉及KD时即使ONLY没有J81，也会把其seed42依赖排到前面；不会偷偷找历史权重。多seed学生共用同一个seed42老师，老师随机性没有通过学生3seed重复估计，需如实报告。显式指定其他TEACHER只适合有清楚训练集来源的权重，其路径/SHA会入记录；不能将训练过test的老师用于正式比较。

环境沿用当前Ultralytics工程即可；不需要安装Mamba、自定义CUDA或另一个评估框架。未安装工程才运行 `pip install -e .`，不建议为了本试验升级现有Torch/CUDA。默认AMP=False、cache=True、AdamW、imgsz640、batch16、workers4；源图平衡的global/coarse/fine混合为.5/.25/.25，与J4一致。缓存视图RAM会明显超过原图字节数，准备阶段会打印估计；加老师后若OOM，**统一改所有臂的BATCH并使用新PROJECT**，不要只改失败的一臂。

## 4. 中断、保护和溯源

- 全程前台串行，无nohup，无后台子训练进程，无GPU占用锁；Ctrl+C后不启动下一个模型。
- 物理GPU号在Torch导入前绑定，并向Ultralytics传相同物理号，避免其select_device将可见卡重设成物理0。
- 相同协议且有完整 `j5_completed.json` 和 `best_mask.pt` 才跳过。部分目录/协议改变直接报错，不覆盖结果；可设置ONLY跳过中断臂，或用新PROJECT重跑该臂。当前脚本**不自动续训半途checkpoint**。
- 同路径更改图像/标签/划分时必须选新PROJECT；没有要求用户进行数据确认/指纹许可。记录hash用于复现，不代替真实数据审计。
- 每个运行有 `j5_protocol.json`、`j5_sources.json`、初始化张量hash、运行环境、实际加载train/val列表、原始train成员映射、老师SHA、最佳Mask checkpoint选择记录。KD模块另记录其有效匹配/监督统计；项目根目录 `j5_training_summary.csv` 汇总同一个最佳Mask AP50–95 epoch的AP50/R/P，不拼接不同峰值。
- 不改旧runner、旧YAML、原始数据、已完成结果或正在跑的baseline。新YAML在 `0_orange_yaml/J_V5_series/`，已登记MODEL_INDEX。

## 5. 怎样判断有用，怎样避免夸大

先看同EV6最佳Mask行的AP50–95、AP50、Recall，不能各取峰值拼一行。小目标还要固定分数/IoU的tiny匹配、R@P≥.90及真实最大观察Recall。重画PR不等于解决漏检。统一部署图/输入预算比较Params/GFLOPs、GPU真实延迟和端到端时延；老师训练开销单列。

如果J83/J84不超过J82的多seed噪声，蒸馏不成立；若J85精度损失明显，保留16通道，不硬说极小模型成功。若J82压缩本身严重损害能力，即使KD补回一点也未达任务目标，应回到J81而非继续叠损失。当前orange_yolo是用户指定探索数据，历史已知划分审计问题不会因为换runner消失；论文正式性能结论仍需独立、可信的val/test和锁定协议。
