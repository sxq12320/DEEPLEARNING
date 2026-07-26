# 主题一：柑橘 / 果园水果 YOLO 改进检测与分割文献调研

**课题背景**：未成熟柑橘（绿色幼果，与叶片同色）实例分割，基线 Ultralytics YOLO11n-seg。
**核心痛点**：① 远处柑橘极小（大量实例 <32px 甚至 <16px）、模糊；② 欠曝发黑，几乎认不出；③ 果实密集相邻；④ 叶片/枝条条状遮挡；⑤ 果实与背景同色（绿-绿）。

**核验方式**：全部 DOI 经 Crossref `api.crossref.org/works/{DOI}` 逐条解析成功；arXiv 条目经 arXiv abs 页核实。Semantic Scholar API 本次全程 429/403（key 已失效），故改用 Crossref 作为权威核验源。**下表每一条都是可点开验证的真实文献，无编造。**

**入选 16 篇（核心）+ 5 篇（补充）**

---

## 一、总表

| # | 标题 | 第一作者 | 年份 | 期刊/会议 | DOI / arXiv |
|---|---|---|---|---|---|
| 1 | Fruits hidden by green: an improved YOLOV8n for detection of young citrus in lush citrus trees | Ang Gao | 2024 | Frontiers in Plant Science | 10.3389/fpls.2024.1375118 |
| 2 | Green Citrus Detection and Counting in Orchards Based on YOLOv5-CS and AI Edge System | Shilei Lyu | 2022 | Sensors | 10.3390/s22020576 |
| 3 | Data-driven Bayesian Gaussian mixture optimized anchor box model for accurate and efficient detection of green citrus | Yunfeng Zhang | 2024 | Computers and Electronics in Agriculture | 10.1016/j.compag.2024.109366 |
| 4 | SALC-Net: an efficient and accurate green citrus detection model for edge devices | Zhenlun Chen | 2025 | Measurement Science and Technology | 10.1088/1361-6501/ae1aa2 |
| 5 | AI-based framework for early detection and segmentation of green citrus fruits in orchards | Manal El Akrouchi | 2025 | Smart Agricultural Technology | 10.1016/j.atech.2025.100834 |
| 6 | Green Fruit Detection with a Small Dataset under a Similar Color Background Based on the Improved YOLOv5-AT | Xinglan Fu | 2024 | Foods | 10.3390/foods13071060 |
| 7 | Green fruit detection methods: Innovative application of camouflage object detection and multilevel feature mining | Yuting Zhai | 2024 | Computers and Electronics in Agriculture | 10.1016/j.compag.2024.109356 |
| 8 | Polar-Net: Green fruit instance segmentation in complex orchard environment | Weikuan Jia | 2022 | Frontiers in Plant Science | 10.3389/fpls.2022.1054007 |
| 9 | Precision citrus segmentation and stem picking point localization using improved YOLOv8n-seg algorithm | Han Li | 2025 | Frontiers in Plant Science | 10.3389/fpls.2025.1655093 |
| 10 | Comparing YOLOv11 and YOLOv8 for instance segmentation of occluded and non-occluded immature green fruits in complex orchard environment | Ranjan Sapkota | 2024 | arXiv preprint (cs.CV) | arXiv:2410.19869 |
| 11 | Comparing YOLOv8 and Mask R-CNN for instance segmentation in complex orchard environments | Ranjan Sapkota | 2024 | Artificial Intelligence in Agriculture | 10.1016/j.aiia.2024.07.001 |
| 12 | Immature Green Apple Detection and Sizing in Commercial Orchards Using YOLOv8 and Shape Fitting Techniques | Ranjan Sapkota | 2024 | IEEE Access | 10.1109/ACCESS.2024.3378261 |
| 13 | MAE-YOLOv8-based small object detection of green crisp plum in real complex orchard environments | Qin Liu | 2024 | Computers and Electronics in Agriculture | 10.1016/j.compag.2024.109458 |
| 14 | YOLO-MECD: Citrus Detection Algorithm Based on YOLOv11 | Yue Liao | 2025 | Agronomy | 10.3390/agronomy15030687 |
| 15 | Multi-scale feature adaptive fusion model for real-time detection in complex citrus orchard environments | Yunfeng Zhang | 2024 | Computers and Electronics in Agriculture | 10.1016/j.compag.2024.108836 |
| 16 | LAM-YOLO: Drones-based small object detection on lighting-occlusion attention mechanism YOLO | Yuchen Zheng | 2025 | Computer Vision and Image Understanding | 10.1016/j.cviu.2025.104489 |
| S1 | ACDNet: Adaptive Citrus Detection Network Based on Improved YOLOv8 for Robotic Harvesting | Zhiqin Wang | 2026 | Agriculture | 10.3390/agriculture16020148 |
| S2 | Fine-grained multi-scale feature fusion for robust citrus detection under complex orchard conditions | Yuxuan Wei | 2026 | Applied Soft Computing | 10.1016/j.asoc.2026.115506 |
| S3 | Ta-YOLO: overcoming target blocked challenges in greenhouse tomato detection and counting | Yun Zhao | 2025 | Frontiers in Plant Science | 10.3389/fpls.2025.1618214 |
| S4 | Nighttime Harvesting of OrBot (Orchard RoBot) | Jakob Waltman | 2024 | AgriEngineering | 10.3390/agriengineering6020072 |
| S5 | Positioning of mango picking point using an improved YOLOv8 architecture with object detection and instance segmentation | Hongwei Li | 2024 | Biosystems Engineering | 10.1016/j.biosystemseng.2024.09.015 |

---

## 二、逐篇详解

### 【A 组】绿色果实 / 同色背景 —— 直接对口"绿-绿同色"痛点

#### 1. Fruits hidden by green: an improved YOLOV8n for detection of young citrus in lush citrus trees
- **第一作者 / 年份 / 期刊**：Ang Gao / 2024 / Frontiers in Plant Science
- **DOI**：10.3389/fpls.2024.1375118
- **核心改进思想**：针对"幼果被绿叶淹没"提出 YCCB-YOLO（Young Citrus in Complex Background），在 YOLOv8n 基础上重构骨干与特征融合，强化同色背景下的低对比度纹理响应，报道精度约 91.5%+ 水平。
- **对痛点适用性**：与本课题目标物几乎完全一致（未成熟绿色柑橘 + 茂密同色背景），是最直接的对标基线与"前人已做"证据。
- **建议接入点**：**backbone + neck**——可作为本课题的直接对比方法（comparison baseline），并借鉴其同色背景下的特征增强模块设计思路。

#### 2. Green Citrus Detection and Counting in Orchards Based on YOLOv5-CS and AI Edge System
- **第一作者 / 年份 / 期刊**：Shilei Lyu / 2022 / Sensors
- **DOI**：10.3390/s22020576
- **核心改进思想**：提出轻量 YOLOv5-CS（Citrus Sort），引入注意力与改进 loss 提升绿色柑橘在自然环境下的召回，并部署到 AI 边缘设备做果实计数（mAP@.5 约 98.2%，recall 约 97.7%）。
- **对痛点适用性**：绿色柑橘的经典引用文献，但其数据以中近距离为主，对"远距离 <16px 极小果"覆盖不足——正是本课题的空白点。
- **建议接入点**：**训练策略 + 部署**——引用为绿柑橘任务的起点与边缘端算力约束依据；其计数流程可复用于产量估计的落地论证。

#### 3. Data-driven Bayesian Gaussian mixture optimized anchor box model for accurate and efficient detection of green citrus
- **第一作者 / 年份 / 期刊**：Yunfeng Zhang / 2024 / Computers and Electronics in Agriculture
- **DOI**：10.1016/j.compag.2024.109366
- **核心改进思想**：用贝叶斯高斯混合模型（BGMM）从数据分布出发重新优化 anchor box 先验，替代 k-means 聚类，使先验框更贴合绿色柑橘真实尺度分布。
- **对痛点适用性**：极小目标最怕的就是先验尺度失配；把尺度先验建模成混合分布，正对"大量 <32px 实例与近处大果并存"的长尾尺度问题。
- **建议接入点**：**head / 标签分配**——YOLO11 是 anchor-free，可将其思想迁移为**尺度感知的正样本分配**或多分布尺度先验的 TAL 权重调整。

#### 4. SALC-Net: an efficient and accurate green citrus detection model for edge devices
- **第一作者 / 年份 / 期刊**：Zhenlun Chen / 2025 / Measurement Science and Technology
- **DOI**：10.1088/1361-6501/ae1aa2
- **核心改进思想**：面向边缘设备的高效绿色柑橘检测网络，在保持轻量的前提下用注意力/结构优化提升绿果与叶片的判别性。
- **对痛点适用性**：证明"绿色柑橘 + 轻量化"这一组合在 2025 年仍是活跃发表方向；对本课题 n 级模型的算力预算设定有直接参考。
- **建议接入点**：**backbone 轻量化**——作为轻量化改进的近期对标，避免与其重复。

#### 5. AI-based framework for early detection and segmentation of green citrus fruits in orchards
- **第一作者 / 年份 / 期刊**：Manal El Akrouchi / 2025 / Smart Agricultural Technology
- **DOI**：10.1016/j.atech.2025.100834
- **核心改进思想**：构建"早期绿色柑橘检测 + 分割"的完整 AI 框架，覆盖果实早期物候阶段的识别与掩膜提取，服务早期产量预测。
- **对痛点适用性**：**与本课题任务形式最接近的一篇**（绿柑橘 + 分割 + 早期幼果），必须作为核心相关工作正面对比与区分创新点。
- **建议接入点**：**整体 pipeline**——用于界定本课题的差异化定位（例如：他做框架/产量，本课题做极小目标+暗光的算法机理）。

#### 6. Green Fruit Detection with a Small Dataset under a Similar Color Background Based on the Improved YOLOv5-AT
- **第一作者 / 年份 / 期刊**：Xinglan Fu / 2024 / Foods
- **DOI**：10.3390/foods13071060
- **核心改进思想**：面向"同色背景 + 小样本"双重困难，改进 YOLOv5 得到 YOLOv5-AT，通过注意力与迁移/增强策略在有限标注下提升绿果-绿叶的区分度。
- **对痛点适用性**：直击"果实与叶片同色"，且同时解决小数据集问题——硕士课题自建数据集规模有限时的实用参考。
- **建议接入点**：**训练策略（数据增强 + 迁移学习）+ attention**——可直接用于本课题数据量不足时的对照实验设计。

#### 7. Green fruit detection methods: Innovative application of camouflage object detection and multilevel feature mining
- **第一作者 / 年份 / 期刊**：Yuting Zhai / 2024 / Computers and Electronics in Agriculture
- **DOI**：10.1016/j.compag.2024.109356
- **核心改进思想**：把绿色果实检测**重新表述为伪装目标检测（COD）问题**，用多层级特征挖掘放大果实与同色背景之间的细微边界/纹理差异。
- **对痛点适用性**：这是"绿-绿同色"问题最有理论深度的建模视角；对模糊、低对比度的远处小果尤其契合，因为 COD 本质就是在极低前景-背景差异下做分割。
- **建议接入点**：**neck + loss（边界/纹理引导监督）**——强烈建议作为本课题算法创新的主线灵感之一：引入 COD 中的边界感知分支或纹理差异放大模块到 YOLO11n-seg 的 mask 分支。

---

### 【B 组】绿果/柑橘实例分割 —— 与 YOLO11n-seg 任务同构

#### 8. Polar-Net: Green fruit instance segmentation in complex orchard environment
- **第一作者 / 年份 / 期刊**：Weikuan Jia / 2022 / Frontiers in Plant Science
- **DOI**：10.3389/fpls.2022.1054007
- **核心改进思想**：用极坐标（polar）表示做绿果实例分割，以极坐标轮廓回归替代逐像素掩膜，提升复杂果园中密集绿果的实例分离能力。
- **对痛点适用性**：正面回应"果实密集相邻"——极坐标轮廓天然对相邻同类实例的粘连不敏感；对小目标掩膜的紧凑表达也有优势。
- **建议接入点**：**mask head**——可对比 YOLO11-seg 的 prototype-mask 方案，探索轮廓式掩膜头在密集小果上的收益。

#### 9. Precision citrus segmentation and stem picking point localization using improved YOLOv8n-seg algorithm
- **第一作者 / 年份 / 期刊**：Han Li / 2025 / Frontiers in Plant Science
- **DOI**：10.3389/fpls.2025.1655093
- **核心改进思想**：以 YOLOv8n-seg 为基线做柑橘精细分割，并串接果梗采摘点定位，形成"分割→采摘点"的完整视觉链路。
- **对痛点适用性**：**基线同族（n-seg 级别）+ 柑橘 + 分割**，是本课题最贴近的方法学参照；其果梗定位思路可作为本课题下游应用的自圆其说依据。
- **建议接入点**：**head + 后处理**——直接对标其分割精度；采摘点定位可作为本课题"部署到自动化采摘设备"的落地论证。

#### 10. Comparing YOLOv11 and YOLOv8 for instance segmentation of occluded and non-occluded immature green fruits in complex orchard environment
- **第一作者 / 年份**：Ranjan Sapkota / 2024（v3 修订至 2025-01）
- **arXiv**：arXiv:2410.19869（cs.CV，DOI: 10.48550/arXiv.2410.19869）
- **核心改进思想**：在"遮挡 / 非遮挡未成熟绿果"数据上系统对比 YOLO11-seg 与 YOLOv8-seg 各规格，结论为 YOLO11m-seg 在 box/mask 指标最优、YOLOv8n 速度最快。
- **对痛点适用性**：提供了 **YOLO11-seg 在未成熟绿果 + 遮挡场景的官方级基线数值**，可直接引用来论证"选 YOLO11n-seg 作基线"的合理性，以及 n 规格的精度缺口 = 你的改进空间。
- **建议接入点**：**基线选型论证 + 遮挡/非遮挡分组评测协议**——建议照搬其"遮挡/非遮挡分组报告"的评测方式，用于凸显你对条状遮挡的改进。

#### 11. Comparing YOLOv8 and Mask R-CNN for instance segmentation in complex orchard environments
- **第一作者 / 年份 / 期刊**：Ranjan Sapkota / 2024 / Artificial Intelligence in Agriculture
- **DOI**：10.1016/j.aiia.2024.07.001
- **核心改进思想**：一阶段 YOLOv8-seg 与两阶段 Mask R-CNN 在真实果园多条件下的实例分割系统对比，结论 YOLOv8 在精度与速度上综合更优。
- **对痛点适用性**：为"为什么不用 Mask R-CNN / 为什么选 YOLO 系"提供可引用的实验依据，是开题合理性论证的常用弹药。
- **建议接入点**：**相关工作 / 方法选型章节**（非算法接入）。

#### 12. Immature Green Apple Detection and Sizing in Commercial Orchards Using YOLOv8 and Shape Fitting Techniques
- **第一作者 / 年份 / 期刊**：Ranjan Sapkota / 2024 / IEEE Access
- **DOI**：10.1109/ACCESS.2024.3378261
- **核心改进思想**：YOLOv8 检测未成熟绿苹果后，用形状拟合（椭圆/圆拟合）从掩膜反推果径尺寸，做无接触果实尺寸测量。
- **对痛点适用性**：证明"未成熟绿果"在国际期刊上是成立且持续的选题；其形状拟合可在小目标掩膜粗糙时做几何正则化补偿——对 <16px 果实的掩膜质量提升有借鉴。
- **建议接入点**：**后处理 + 应用层**——可作为本课题"分割掩膜→果径/产量估计"的下游价值论证。

---

### 【C 组】小目标 / 尺度 / 多尺度融合 —— 对口"远处极小果"

#### 13. MAE-YOLOv8-based small object detection of green crisp plum in real complex orchard environments
- **第一作者 / 年份 / 期刊**：Qin Liu / 2024 / Computers and Electronics in Agriculture
- **DOI**：10.1016/j.compag.2024.109458
- **核心改进思想**：以 **YOLOv8s-p2 为基线**（即显式增加 P2 高分辨率小目标检测层），叠加 MAE 式表征增强，专攻复杂果园中绿色青脆李的小目标检测（精度约 92.3%）。
- **对痛点适用性**：**本组最关键的一篇**——它同时命中"绿色果实 + 真实果园 + 小目标"，且给出了 P2 层这一最直接的极小目标解法与自监督预训练的组合。
- **建议接入点**：**neck（增加 P2/P1 检测分支）+ 训练策略（MAE 自监督预训练）**——建议作为本课题小目标改进的首要参考与消融对照。

#### 14. YOLO-MECD: Citrus Detection Algorithm Based on YOLOv11
- **第一作者 / 年份 / 期刊**：Yue Liao / 2025 / Agronomy
- **DOI**：10.3390/agronomy15030687
- **核心改进思想**：基于 **YOLOv11** 的柑橘检测改进，同时纳入树上果与落地果，做检测与计数。
- **对痛点适用性**：与本课题基线版本（YOLO11）完全一致的极少数柑橘论文之一，是"YOLO11 + 柑橘"这一组合已被占用/尚有空间的直接证据——**本课题必须与之明确区分（它是检测，你是实例分割 + 极小/暗光）**。
- **建议接入点**：**backbone/neck 对标**——务必精读其改进模块，避免撞车。

#### 15. Multi-scale feature adaptive fusion model for real-time detection in complex citrus orchard environments
- **第一作者 / 年份 / 期刊**：Yunfeng Zhang / 2024 / Computers and Electronics in Agriculture
- **DOI**：10.1016/j.compag.2024.108836
- **核心改进思想**：面向复杂柑橘果园的多尺度特征自适应融合模型，让不同尺度分支按目标尺度分布动态加权融合，兼顾实时性。
- **对痛点适用性**：远近柑橘尺度跨度极大（近处大果 vs 远处 <16px），固定权重的 FPN/PAN 融合会被大目标主导；自适应融合正是缓解这一失衡的手段。
- **建议接入点**：**neck（自适应加权 FPN/PAN）**——可与 P2 分支组合，形成"高分辨率分支 + 自适应尺度加权"的联合改进。

#### 16. LAM-YOLO: Drones-based small object detection on lighting-occlusion attention mechanism YOLO
- **第一作者 / 年份 / 期刊**：Yuchen Zheng / 2025 / Computer Vision and Image Understanding
- **DOI**：10.1016/j.cviu.2025.104489
- **核心改进思想**：提出光照-遮挡注意力机制（lighting-occlusion attention），在无人机小目标场景下联合建模光照退化与遮挡，提升暗光/遮挡小目标召回。
- **对痛点适用性**：**唯一同时正面处理"小目标 + 光照退化 + 遮挡"三者的方法论文**，与本课题"远处小 + 发黑欠曝 + 条状遮挡"痛点三重命中；虽非农业域，但机制可跨域迁移。
- **建议接入点**：**backbone/neck 注意力模块 + loss**——建议作为暗光小目标改进的核心迁移来源（CV 顶刊方法迁移到农业场景，是硕士小论文最常见且可行的创新范式）。

---

### 【D 组】补充参考（遮挡 / 暗光 / 采摘落地）

#### S1. ACDNet: Adaptive Citrus Detection Network Based on Improved YOLOv8 for Robotic Harvesting
- Zhiqin Wang / 2026 / Agriculture / **10.3390/agriculture16020148**
- 自适应柑橘检测网络，宣称在 40%–60% 遮挡率场景下仍能检出 YOLOv8n 漏检的果实。
- **适用性**：直接对口"叶片枝条条状遮挡"；且是最新（2026）柑橘 YOLO 改进，说明方向仍未饱和。
- **接入点**：neck / head 的遮挡鲁棒性模块。

#### S2. Fine-grained multi-scale feature fusion for robust citrus detection under complex orchard conditions
- Yuxuan Wei / 2026 / Applied Soft Computing / **10.1016/j.asoc.2026.115506**
- 细粒度多尺度特征融合的柑橘检测，强调复杂果园条件下的鲁棒性。
- **适用性**：证明"多尺度融合 + 柑橘"在 2026 年仍能发 SCI 一区/二区；是尺度改进路线的最新对标。
- **接入点**：neck。

#### S3. Ta-YOLO: overcoming target blocked challenges in greenhouse tomato detection and counting
- Yun Zhao / 2025 / Frontiers in Plant Science / **10.3389/fpls.2025.1618214**
- 面向温室番茄"目标被遮挡"问题的 YOLO 改进框架，兼顾小番茄果实检测与计数。
- **适用性**：番茄与串番茄场景的密集相邻 + 遮挡处理，可横向迁移到密集柑橘。
- **接入点**：head + 后处理（NMS/计数）。

#### S4. Nighttime Harvesting of OrBot (Orchard RoBot)
- Jakob Waltman / 2024 / AgriEngineering / **10.3390/agriengineering6020072**
- 果园机器人夜间采摘的系统性研究（含夜间照明与视觉条件分析）。
- **适用性**：为"果园暗光/夜间作业"这一场景提供**工程可行性与必要性论证**（不是算法论文，但是暗光选题合理性的引用来源）。
- **接入点**：绪论/应用背景章节；主动补光 vs 算法增暗光的方案取舍论证。

#### S5. Positioning of mango picking point using an improved YOLOv8 architecture with object detection and instance segmentation
- Hongwei Li / 2024 / Biosystems Engineering / **10.1016/j.biosystemseng.2024.09.015**
- 检测 + 实例分割双任务的改进 YOLOv8 架构，用于芒果采摘点定位。
- **适用性**：实例分割服务于采摘点的完整技术路径范例，与串果/柑橘采摘的机械落地衔接。
- **接入点**：head（多任务）+ 应用层论证。

---

## 三、按痛点归类的接入建议速查

| 痛点 | 最相关文献 | 可迁移的技术接入点 |
|---|---|---|
| 远处极小果（<32px / <16px） | #13 MAE-YOLOv8（P2 层）、#15 多尺度自适应融合、#3 BGMM anchor 先验、#16 LAM-YOLO | **neck**：新增 P2/P1 高分辨率检测分支；自适应尺度加权融合。**head/标签分配**：尺度感知正样本分配（NWD/尺度先验混合分布）。 |
| 绿果与背景同色 | #7 伪装目标检测 COD、#6 YOLOv5-AT、#1 YCCB-YOLO、#5 绿柑橘分割框架 | **neck + loss**：边界感知分支、纹理差异放大、多层级特征挖掘；同色场景专用数据增强。 |
| 欠曝发黑 / 暗光 | #16 LAM-YOLO（光照-遮挡注意力）、S4 OrBot 夜间采摘 | **backbone 前端**：可学习的光照自适应/曝光校正模块（端到端联合训练，而非离线预增强）。**训练策略**：低照度域增强 + 域自适应。 |
| 果实密集相邻 | #8 Polar-Net（极坐标实例分割）、S3 Ta-YOLO | **mask head**：轮廓式/极坐标掩膜表达替代或补充 prototype mask；密集场景 NMS 与计数后处理。 |
| 枝叶条状遮挡 | S1 ACDNet（40–60% 遮挡）、#10 Sapkota 遮挡分组评测、S3 Ta-YOLO | **neck/head**：遮挡鲁棒特征聚合；**评测协议**：按遮挡率分组报告 mask AP，凸显改进。 |
| 基线选型 / 论证 | #10（YOLO11-seg vs YOLOv8-seg 绿果）、#11（YOLOv8 vs Mask R-CNN）、#9（YOLOv8n-seg 柑橘） | 相关工作与实验设计章节的直接引用。 |

---

## 四、选题层面的三点判断（基于上述文献）

1. **"柑橘 + YOLO 改进"已相当拥挤**：仅 2024–2026 就有 YOLO-MECD(#14)、ACDNet(S1)、Fine-grained fusion(S2)、SALC-Net(#4)、YOLOv8n-seg 柑橘分割(#9) 等多篇，且已有人做到 YOLOv11 基线。**单纯"换几个注意力模块 + 涨点"极难发表**。

2. **真正的空白在三者交集**：绿色未成熟柑橘 **且** 极小尺度（<16–32px）**且** 欠曝暗光 **且** 实例分割。检索中未发现同时满足这四点的论文——#5 做绿柑橘分割但不聚焦极小/暗光；#13 做绿果小目标但是检测不是分割、且是李子；#16 做暗光小目标但非农业。**这个交集就是本课题的创新缝隙**。

3. **最可行的创新范式**：把 #7 的伪装目标检测（COD）视角 + #16 的光照-遮挡注意力，迁移进 YOLO11n-seg 的 neck/mask head，配合 #13 的 P2 分支，形成"同色-极小-暗光"三位一体的联合改进，并按 #10 的遮挡分组协议做评测。这条路线每一环都有真实文献支撑，且组合方式尚无人发表。

---

*核验记录：本文件所有 DOI 于 2026-07-26 经 Crossref REST API 逐条 `GET /works/{DOI}` 返回 200 并核对标题/作者/年份/期刊一致；arXiv:2410.19869 经 arxiv.org/abs 页核实。Semantic Scholar API 本次不可用（key 403 / 匿名 429），故未采用。*
