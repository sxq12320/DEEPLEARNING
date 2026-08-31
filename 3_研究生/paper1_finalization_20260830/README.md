# 柑橘幼果实例分割：论文定稿工作区（2026-08-30）

这个目录是第一篇论文的唯一“决策入口”，不替代代码、数据和原始结果，也不删除或搬移任何历史资产。它把目前可以写进论文的事实、尚未复现的旧结论、候选网络和下一步实验严格分开。

## 一句话结论

现有 clean/group-aware 结果只证明了 **轻量头（S04/B01）与有限的上下文-拓扑协同（B06）有弱正信号**；它们没有解决小目标召回、叶果同色、深凹可见掩膜和邻果粘连。旧数据上的 G10 不能直接与 clean baseline 比较。下一阶段应停止大批量模块排列组合，优先验证 D 系列的双分辨率形状-语义主干，再开发“保真—稀疏搜索—果叶判别—不确定边界精修”的统一结构。

## 阅读顺序

1. [证据与目录审计](01_证据与目录审计.md)：哪些数字可信、哪些不能比较、哪些报告含虚构结果。
2. [范围化系统文献综述](02_范围化系统文献综述.md)：检索方法、论文证据、开源代码与可迁移边界。
3. [最终候选架构与实验路线](03_最终候选架构与实验路线.md)：保留哪些模型、淘汰哪些方向、先跑什么。
4. [检索日志](sources/search_log.md)：数据库、检索式、下载记录与接口失败。
5. [论文证据表](sources/selected_papers.csv) 与 [代码仓库审计表](sources/repository_registry.csv)。
6. [Citrus SDR-Net 结构草图](figures/citrus_sdr_architecture_concept.png)：仅为待消融的设计假设，不含性能承诺。

已补充活动 fork 中的 `eval_citrus_challenges.py`：它输出 challenge-subset IoU50 recall、Boundary F1、split/merge 诊断和逐实例 CSV；标准 Mask AP 仍由 `eval_citrus_seg.py` 负责，两者不能混为一个指标。

验证状态：D 系列 13 个结构/反向测试与挑战评估器 7 个合成测试合计 20 passed；Ruff 通过。评估器还用 S00 `best.pt`、clean val 的 1 张图完成了端到端 CPU smoke，临时输出验证后已安全删除。

## 当前唯一有效路径

| 角色 | 路径 | 结论 |
|---|---|---|
| 全系列活动代码 | `E:/mastercode/1_SEVER/code/ultralytics-main-new` | 187 个系列 YAML、D 系列模块与批量脚本所在位置 |
| 旧根目录 fork | `E:/mastercode/ultralytics-main-new` | 仍保留，不删除；不用于全系列复现实验 |
| 正式数据候选 | `E:/mastercode/data/orange_yolo_grouped_dedup_20260820` | 965 图、5,890 实例，group-aware 防泄漏 |
| 原始结果索引 | `E:/mastercode/1_SEVER/results/RESULTS_INDEX.csv` | 111 个运行目录的事实入口 |
| 难度量化 | `E:/mastercode/1_SEVER/results/_analysis/_analysis_20260824_network_redesign/dataset_difficulty` | 形状、邻距、尺度、颜色与组合难例统计 |

`E:/mastercode/data/test` 当前不存在，与旧研究计划中“941 图、4,576 实例”的描述冲突。正式论文必须统一采用带版本号的数据集与固定清单，禁止在两套计数之间切换。

## 禁止事项

- 不把旧数据 G/F/N/A 与 clean S/B 放进同一张精度排名表。
- 不把未训练的 C/D/H/L/T 写成“有效”或“最优”。
- 不把 PR 曲线在 recall=1 处的 `(1, 0)` 哨兵端点解释成真实工作阈值。
- 不引用 `architecture_search_20260827` 中未经运行的三种子均值、延迟、挑战子集或 B09=0.6275 等数字。
- 不承诺从 AP50 0.78 固定提升到 0.88；只能把它写为实验目标。
- 不再一次设计十个高度相似的注意力/卷积/上采样组合。
