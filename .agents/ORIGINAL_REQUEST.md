# Original User Request

## 2026-08-27T07:12:05Z

你现在是本项目的“计算机视觉首席研究员、系统文献检索人员和可复现工程负责人”。

当前日期为 2026-08-27。工作目录为 E:\mastercode。

你的任务不是随便寻找几个注意力模块，也不是把多个论文模块堆进 YOLO。你需要对“未成熟柑橘轻量级高精度实例分割”开展一次系统、可复现、证据驱动的全方面检索，并在此基础上提出一个结构完整、轻量且有充分提升潜力的新网络架构。

一、开始工作前必须阅读
1. E:\mastercode\AGENTS.md (以 E:\mastercode\ultralytics-main-new 为正式活动代码)
2. E:\mastercode\3_研究生\柑橘套袋视觉_完整研究执行计划.md
3. E:\mastercode\1_SEVER\results\README.md
4. E:\mastercode\1_SEVER\results\RESULTS_INDEX.csv
5. E:\mastercode\1_SEVER\results\S_series\grouped_clean_300ep\20260827_S_RESULTS_TO_B_V2.md
6. E:\mastercode\1_SEVER\code\ultralytics-main-new\0_orange_yaml\MODEL_INDEX.csv
7. 当前 train_citrus_seg.py、eval_citrus_seg.py、ultralytics/nn/tasks.py 和已有 citrus 模块
8. 数据集目录 E:\mastercode\data\orange_yolo_grouped_dedup_20260820

二、研究边界
只关注 RGB 未成熟柑橘实例分割，轻量化/高精度/实际推理速度，微小目标、单图极端尺度跨度、同色伪装、条带遮挡深凹掩膜、簇生粘连拓扑冲突、PR曲线尾部塌陷。不得扩展到 RGB-D、amodal、OBB、机器人控制。禁止 Mamba、selective scan 或难以部署的自定义 CUDA 扩展。

三、禁止事项
不得虚构论文、DOI 或 GitHub。必须阅读方法、实验和消融。不得把不同协议结果直接比较。

四、完成本项目事实审计 (形成量化表)

五、系统检索范围 (Themes A ~ O)
覆盖 CVF/arXiv/IEEE/ACM/Springer/Scholar/GitHub。

六、检索规模
初筛 >=80 篇，精筛 >=40 篇，精读 >=28 篇，审查 >=10 个官方仓库，形成 15~25 篇核心证据链。

七、八、论文与代码证据矩阵构建

九、架构设计原则
提出 Candidate A (保守剪枝)、Candidate B (⭐推荐主方案 CitrusB-Seg)、Candidate C (激进探索双流)。

十、十一、十二、轻量化约束与实验设计
Params <= 2.85M, GFLOPs <= 10.0G, 实测 CPU 延迟 <= 150ms / GPU <= 8ms。

十三、交付文件
在 E:\mastercode\3_研究生\architecture_search_20260827\ 生成：
00_research_scope.md, 01_search_strategy.md, 02_search_log.csv, 03_paper_evidence_matrix.xlsx, 04_repository_evidence_matrix.xlsx, 05_current_task_diagnosis.md, 06_negative_results_and_risks.md, 07_architecture_candidates.md, 08_final_architecture_recommendation.md, 09_ablation_and_experiment_plan.md, 10_reproducibility_checklist.md, references.bib, architecture_overview.mmd。

十四、最终回答方式
必须明确系统回答 12 大核心问题，严格标注证据等级（已由本项目实验验证 / 由外部论文支持 / 合理但尚未验证的研究假设）。
