# Theme 9 — 微小目标的标签分配与训练策略（14 篇，全部经 arXiv API / Crossref 核验）

课题背景：YOLO11n-seg 柑橘幼果实例分割；痛点 = <16px 微小果在 TAL 下正样本极少/为 0，远处果为估计标注（噪声框）。

---

## A. 标签分配（Label Assignment）

### 1. TOOD: Task-aligned One-stage Object Detection（TAL 原文）
- 第一作者/年份/venue：Chengjian Feng, 2021, ICCV 2021 (Oral)
- ID：arXiv:2108.07755
- 核心机制：提出任务对齐度量 t = s^α × u^β（分类得分 s 与 IoU u 的加权乘积），按 t 取 top-k 候选点作正样本，并用 t 归一化后作为分类软标签（Task-Aligned Learning）。Ultralytics YOLOv8/11 的 TaskAlignedAssigner 即此实现（α=0.5, β=6, topk=10）。
- TAL 框架内可实现性：这就是基线本身；关键失效点在 u=IoU——<16px 目标与 stride-8 网格点候选框 IoU 天然接近 0，t≈0 导致 top-k 里也全是低质样本甚至被过滤。改造入口 = 替换 u 的度量或改候选集。

### 2. ATSS: Adaptive Training Sample Selection
- 第一作者/年份/venue：Shifeng Zhang, 2020, CVPR 2020
- ID：arXiv:1912.02424
- 核心机制：每 GT 在每层 FPN 取中心距离最近的 k 个候选，用这些候选 IoU 的 mean+std 作为该 GT 的自适应阈值，从而每个 GT 至少能分到统计意义上的正样本。首次系统论证 anchor-based 与 anchor-free 差距主要来自正负样本定义而非回归形式。
- TAL 框架内可实现性：其"每 GT 自适应阈值 + 保底样本"思想可直接嫁接进 TaskAlignedAssigner：对 align_metric 做 per-GT 的 mean+std 动态阈值，替代固定 top-k，小 GT 不会被大 GT 挤出。

### 3. YOLOX（SimOTA）
- 第一作者/年份/venue：Zheng Ge, 2021, arXiv preprint（Megvii 技术报告）
- ID：arXiv:2107.08430
- 核心机制：SimOTA 把分配视为简化最优传输：cost = 分类损失 + λ·回归损失，每 GT 的正样本数 k 由 top-q 候选 IoU 之和动态决定（dynamic-k）。小/难 GT 的 IoU 和小 → k 小但不为 0，保证每 GT 至少 1 个正样本。
- TAL 框架内可实现性：dynamic-k 与"每 GT 保底 ≥1"两条可直接移植到 TAL 的 top-k 选择步骤，改动约 20 行（select_topk_candidates 处），不动损失函数。

### 4. RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection（深挖）
- 第一作者/年份/venue：Chang Xu, 2022, ECCV 2022
- ID：arXiv:2208.08738
- 核心机制：指出 anchor-based(IoU) 与 anchor-free(center-in-box) 先验对 tiny object 都存在尺度偏差——<16px GT 内部可能一个网格中心都不落。将每个特征点的有效感受野建模为 2D 高斯，GT 框也建模为高斯，用二者 KL 散度导出 Receptive Field Distance (RFD)，再经 Hierarchical Label Assignment (HLA)：先按 RFD 排名取 top-k 作正样本，再对 0 正样本的 GT 补充次优样本（rank 衰减因子 β 降权），保证每个 tiny GT 都有正样本。RFD 对两个不重叠的框仍给出平滑连续的度量（IoU 此时恒 0），这是其对 <16px 目标有效的根本原因。
- TAL 框架内可实现性：最对症的改法——在 TaskAlignedAssigner 里把 overlaps 从 CIoU 换成 RFD（或与 IoU 加权混合），t = s^α × RFD^β；HLA 的补偿分配作为第二阶段兜底，均不改损失与推理结构。

### 5. DSLA: Dynamic Smooth Label Assignment for Efficient Anchor-free Object Detection
- 第一作者/年份/venue：Hu Su, 2022, Pattern Recognition
- ID：arXiv:2208.00817；DOI: 10.1016/j.patcog.2022.108868
- 核心机制：将 FCOS 的 centerness 推广为连续的"平滑分配"：分类监督标签由 IoU 连续加权（soft label），正负样本间不再有硬边界，动态地把模糊区域样本按质量给部分正监督。缓解硬阈值下小目标一刀切被判负的问题。
- TAL 框架内可实现性：TAL 已用 t 归一化做软标签，DSLA 的启发是进一步取消 top-k 硬截断、按连续权重给 <16px 目标周边点部分正监督；可在 assigner 输出的 target_scores 上实现，但与 TAL 现有归一化耦合较深，工程量中等。

### 6. A Normalized Gaussian Wasserstein Distance for Tiny Object Detection（NWD，进 assigner 的度量替换）
- 第一作者/年份/venue：Jinwang Wang, 2021, arXiv preprint（后续 NWD-RKA 版发表于 ISPRS J. P&RS 2022）
- ID：arXiv:2110.13389
- 核心机制：把框建模为 2D 高斯，用归一化 Wasserstein 距离替代 IoU；NWD 对微小目标的位置偏移不敏感（IoU 对 tiny 框几像素偏移即剧烈跳变），且对无重叠框仍可度量。论文明确将 NWD 同时用于 assigner、NMS 与回归损失三处，assigner 处收益最大（AI-TOD 上显著涨点）。
- TAL 框架内可实现性：与 RFLA 并列的首选——在 TaskAlignedAssigner.get_box_metrics 里把 bbox_iou 换成 NWD（约 15 行高斯化 + Wasserstein 公式），t = s^α × NWD^β，是社区已有大量 YOLO 复现的成熟改法。

### 7. Towards Large-Scale Small Object Detection: Survey and Benchmarks（综述）
- 第一作者/年份/venue：Gong Cheng, 2023, IEEE TPAMI
- ID：arXiv:2207.14096；DOI: 10.1109/TPAMI.2023.3290594
- 核心机制：小目标检测系统综述 + SODA-D/SODA-A 基准；将小目标失效归因为信息丢失（下采样）、标注/度量敏感（IoU 抖动）、正样本稀缺（分配偏差）三类，并横向比较各类 assigner/度量/数据增广在 tiny 场景的实证表现。
- TAL 框架内可实现性：不直接实现；用于论文 related work 的分类框架与选择依据（引证"正样本稀缺是 tiny 检测三大瓶颈之一"）。

## B. 辅助监督（Auxiliary Supervision）

### 8. Objects as Points（CenterNet，Gaussian heatmap 先验）
- 第一作者/年份/venue：Xingyi Zhou, 2019, arXiv preprint
- ID：arXiv:1904.07850
- 核心机制：目标表示为中心点高斯热图，heatmap 峰值处回归尺寸；高斯半径随目标尺寸缩放，中心邻域给衰减的软正监督——天然做到"每 GT 必有正监督点"，与 IoU 无关。
- TAL 框架内可实现性：作为辅助分支：在 P2/P3 上加一个轻量高斯热图 head（仅训练时存在，推理移除），为 <16px 果提供与分配无关的密集中心先验监督；不动 TAL 主 head，成本低、可消融。

### 9. QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection（P2/高分辨率特征 + 辅助监督）
- 第一作者/年份/venue：Chenhongyi Yang, 2022, CVPR 2022
- ID：arXiv:2103.09136
- 核心机制：小目标必须用高分辨率特征（P2）检测，但全图 P2 计算昂贵；先在低分辨率层预测"小目标存在"的粗查询热图（辅助监督信号），再仅在被查询的稀疏位置计算 P2 检测头。证明 P2 层 + 稀疏计算可在小目标 AP 上大幅收益而不拖垮速度。
- TAL 框架内可实现性：YOLO11 加 P2 输出层（yaml 改 head 即可，Ultralytics 官方支持 p2 模型配置），stride-4 网格使 <16px 果在该层等效为 4×4 网格目标，TAL 正样本数自然增加；QueryDet 的稀疏查询可作为速度补救的加分项。

### 10. DEIM: DETR with Improved Matching for Fast Convergence（dense supervision, 2024-2025）
- 第一作者/年份/venue：Shihua Huang, 2024（CVPR 2025）
- ID：arXiv:2412.04234
- 核心机制：诊断 DETR 一对一匹配导致正监督过稀疏，提出 Dense O2O——用 mosaic/mixup 人为增加每图目标数以致密化匹配监督，并用 Matchability-Aware Loss (MAL) 按匹配质量调制损失。核心论点可迁移：正监督密度本身是收敛与小目标性能的一等因素。
- TAL 框架内可实现性：思想层面迁移——通过增广提高每 batch 中 tiny GT 的正样本总量（配合 copy-paste 小果），以及按 align 质量调制损失权重（MAL 与 TAL 的 t 加权同源），无需引入 DETR 结构。

## C. 微小目标训练技巧实证

### 11. Dynamic Scale Training for Object Detection（Stitcher）
- 第一作者/年份/venue：Yukang Chen, 2020, arXiv preprint
- ID：arXiv:2004.12432
- 核心机制：发现小目标损失占比在多数迭代中极低（监督被中大目标垄断）；以"小目标损失占比"为反馈信号，动态决定下一 batch 是否用图像拼接（4 图缩小拼 1 图，即 mosaic 类操作）制造更多小目标。给出了 mosaic 对小目标"利"的机理解释（人为造小目标、提高其损失份额）；反面是持续缩小会伤害本已 tiny 的目标——对 <16px 原生小目标，mosaic 的缩小分支可能把它压到不可学，需限制缩放下界或后期关闭 mosaic（Ultralytics close_mosaic 参数的实证依据；mosaic 本身出处为 YOLOv4, arXiv:2004.10934）。
- TAL 框架内可实现性：纯数据管线改动：设 mosaic scale 下界（如 0.5–1.0 而非默认 0.5 起含缩小)、close_mosaic=10-15 epoch，零结构改动。

### 12. Augmentation for Small Object Detection（copy-paste 过采样）
- 第一作者/年份/venue：Mate Kisantal, 2019, arXiv preprint
- ID：arXiv:1902.07296
- 核心机制：量化了 COCO 上小目标匹配到的 anchor/正样本数远低于大目标这一"监督不均"，提出对含小目标图像过采样 + 将小目标在图内多次 copy-paste，直接增加小目标正样本绝对数量，小目标 AP 提升显著。
- TAL 框架内可实现性：数据侧最廉价的"增正样本"手段：对 <16px 柑橘幼果做图内 copy-paste（果实近圆形、背景为叶片，粘贴伪影小），Ultralytics 已内置 copy_paste 超参（分割任务可用 mask 级粘贴）。

### 13. Slicing Aided Hyper Inference and Fine-tuning for Small Object Detection（SAHI，高分辨率微调策略出处）
- 第一作者/年份/venue：Fatih Cagatay Akyon, 2022, IEEE ICIP 2022
- ID：arXiv:2202.06934
- 核心机制：切片辅助微调（把大图切 tile 后目标相对尺寸变大再训练）+ 切片推理再合并；证明"训练/推理分辨率提升使 tiny 目标进入检测器有效尺度区间"是与模型无关的通用增益。"先 640 常规训练、再更高输入分辨率微调"的两阶段做法在此与 SNIP 尺度归一化（Singh & Davis, CVPR 2018, arXiv:1711.08189）一脉相承。
- TAL 框架内可实现性：零代码改动：imgsz=640 训练 → imgsz=960/1280 低学习率微调若干 epoch；960 下 16px 果变为等效 24px，TAL 正样本数直接翻倍以上；推理端可选 SAHI 切片。

## D. 噪声标注下的检测训练

### 14a. Towards Noise-resistant Object Detection with Noisy Annotations（co-teaching 式）
- 第一作者/年份/venue：Junnan Li, 2020, arXiv preprint
- ID：arXiv:2003.01285
- 核心机制：把标注噪声解耦为标签噪声与框噪声，交替进行"标注纠正"与"模型训练"：用双模型(co-teaching 式)的一致性/分歧过滤错误类标，用分类置信度加权的框预测聚合来软纠正噪声框坐标。证明检测器可在训练中自我修正估计式标注。
- TAL 框架内可实现性：可简化落地为"训练中期用模型高置信预测框对'估计标注'的远处果做坐标 refine（离线一轮伪标签纠正）"，不改 TAL 本体。

### 14b. Robust Object Detection With Inaccurate Bounding Boxes（OA-MIL）
- 第一作者/年份/venue：Chengxin Liu, 2022, ECCV 2022
- ID：arXiv:2207.09697
- 核心机制：面向"框位置不准"（正是估计式标注的形态），把每个噪声 GT 视为一个 bag，从其邻域构造候选实例集合，用多示例学习 (Object-Aware MIL) 交替选择 bag 内最优实例作为监督目标并合成更准的训练框，无需干净标注。
- TAL 框架内可实现性：MIL 全套较重；可借其"以 GT 邻域候选中分类响应最高者修正回归目标"思想，在 assigner 后对噪声标记的远处果降低回归损失权重（loss re-weighting by t），或对该子集只监督分类不强监督框。

---

## 结论：不推翻 TAL 前提下，最可行的 2 个"让 <16px 果拿到正样本"的改法

1. **度量替换（RFLA/NWD 路线）**：在 TaskAlignedAssigner 中把 align_metric 里的 CIoU 换成 NWD 或 RFD（或 IoU 与 NWD 的加权混合），并加 SimOTA 式"每 GT 保底 ≥1 正样本 / RFLA-HLA 补偿分配"。原因：<16px 目标正样本为 0 的直接根因是 IoU 度量在 tiny 尺度退化为 0/剧烈抖动，换成高斯距离度量后 t 不再坍缩，top-k 才有意义；改动集中在一个函数，创新叙事清晰（NWD/RFD 进 TAL 在柑橘幼果分割任务上属新组合）。
2. **提高有效分辨率（P2 层 + 高分辨率微调）**：加 stride-4 的 P2 检测层（Ultralytics 原生支持 *-p2.yaml），并采用 640 训练 → 960 微调（SAHI/SNIP 依据）。原因：不改 assigner 一行代码，直接把 <16px 果映射为 P2 上 4×4+ 网格目标，TAL 候选点数与 IoU 同时变好，是与改法 1 正交、可叠加消融的第二支柱。

辅助监督（CenterNet 式高斯热图辅助分支）与噪声处理（OA-MIL 式对估计标注降权/伪标签纠正）作为第三、四消融项，分别对应"分配无关的兜底监督"与"远处果噪声"痛点。
