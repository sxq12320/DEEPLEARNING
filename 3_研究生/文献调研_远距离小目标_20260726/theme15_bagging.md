# 主题 15：果实套袋自动化全链条文献调研

> 调研范围 2015–2026，共 **30 篇**，全部经 Crossref DOI 核验（2026-07-26）。
> 课题背景：柑橘套袋视觉 —— 论文①柑橘幼果实例分割（已完成）；论文②基于果实 ROI 的柑橘果梗点精准定位（下一篇）。

---

## A. 套袋农艺与必要性（背景弹药，7 篇）

| 标题 | 作者/年份 | 期刊/会议 | DOI/ID | 核心内容 | 对本课题价值 |
|---|---|---|---|---|---|
| Fruit Bagging Enhances Peel Color and Affects Fruit Quality of Citrus under Protected Screen-grown Grapefruit-like Hybrid '914' | Pareek et al., 2025 | HortScience | 10.21273/hortsci18223-24 | 在防护网棚柑橘杂交种上验证套袋显著改善果皮着色，并系统评估对可溶性固形物、酸度等品质指标的影响。 | 最新、最直接的"柑橘套袋提质"证据，引言第一句的首选引文。 |
| Effects of fruit bagging on the physiochemical changes of grapefruit (Citrus paradisi) | Jiang et al., 2022 | Food Quality and Safety | 10.1093/fqsafe/fyac049 | 追踪葡萄柚整个发育期套袋后的理化变化，量化果皮色泽、糖酸比与挥发物差异。 | 柑橘属套袋的生理机制引文，支撑"套袋是刚需农艺"论断。 |
| The Effectiveness of Fruit Bagging and Culling for Risk Mitigation of Fruit Flies Affecting Citrus in China: A Preliminary Report | (Anon.), 2019 | Florida Entomologist | 10.1653/024.102.0112 | 在中国柑橘园验证套袋+清园对实蝇（检疫性害虫）的风险削减效果。 | 中国柑橘场景 + 检疫出口视角，是"为什么中国柑橘必须套袋"的硬证据。 |
| Evaluation of Fruit Bagging as a Pest Management Option for Direct Pests of Apple | Frank, 2018 | Insects | 10.3390/insects9040178 | 系统评估套袋作为苹果直接害虫的物理防治手段，对比化学防治的成本与效果。 | 把套袋定位为"减农药物理防治"，支撑减药/绿色生产的立项理由。 |
| Pre-harvest bagging of grape clusters as a non-chemical physical control measure against certain pests and diseases of grapevines | Karajeh, 2017 | Organic Agriculture | 10.1007/s13165-017-0197-3 | 证明采前套袋可作为葡萄有机栽培中替代农药的非化学病虫害控制措施。 | "套袋=有机/零农残路径"的经典引文，跨作物佐证普适性。 |
| Fruit bagging reduces the postharvest decay and alters the diversity of fruit surface fungal community in 'Yali' pear | Gao et al., 2022 | BMC Microbiology | 10.1186/s12866-022-02653-4 | 揭示套袋通过改变果面真菌群落结构降低采后腐烂率。 | 从微生物组角度给出机制解释，可用于讨论套袋的"隐性收益"。 |
| Effect of Paper and Aluminum Bagging on Fruit Quality of Loquat (Eriobotrya japonica Lindl.) | Zhi et al., 2021 | Plants | 10.3390/plants10122704 | 对比纸袋与铝箔袋对枇杷品质的差异化影响，说明袋型选择的重要性。 | 说明"袋型/材质多样" → 机械化开袋机构必须适配多种袋，是 B 方向的需求来源。 |

---

## B. 套袋机械与套袋机器人（核心，7 篇）

| 标题 | 作者/年份 | 期刊/会议 | DOI/ID | 核心内容 | 对本课题价值 |
|---|---|---|---|---|---|
| **Vision localization algorithms for apple bagging robot** | Gao, Liu, Li, Yu, 2017 | 2017 29th Chinese Control And Decision Conference (CCDC) | 10.1109/ccdc.2017.7978080 | 目前检索到的**唯一**一篇明确面向"套袋机器人"的视觉定位算法工作，针对苹果幼果做识别与定位。 | ★最关键对标文献：证明"套袋+视觉"这条线存在但极其稀薄（2017 年、传统方法），本课题可直接宣称深度学习时代的空白。 |
| **Design and Simulation of End Effector for Young-Pear-Bagging Robot** | Teng, Chen, Wu, Shen, 2024 | Processes | 10.3390/pr12020259 | 面向梨幼果套袋机器人设计末端执行器，含开袋-套入-扎口的机构方案与仿真验证。 | ★最新的套袋机器人 end-effector 设计，是"机构已有、感知缺位"的直接证据。 |
| Rigid-flexible coupling contact action simulation study of the open mechanism on the ordinary multilayer fruit paper bag for fruit bagging | Xia, Zhen, Chen, Zeng, 2020 | Computers and Electronics in Agriculture | 10.1016/j.compag.2020.105414 | 用刚柔耦合接触仿真分析多层纸袋开袋机构的作用过程，给出开袋成功率的力学依据。 | 开袋机构（bag-opening mechanism）的权威力学建模，写机械部分时的核心引文。 |
| An Ordinary Multilayer Fruit Paper Bag Supplying Device for Fruit Bagging | Xia, Zhen, Chen, Zeng, 2019 | HortScience | 10.21273/hortsci14171-19 | 设计并试验了多层果袋的自动供袋装置，解决连续作业中的取袋-分离环节。 | 与上一篇同组，构成"供袋→开袋"完整机构链，可用于描述系统边界。 |
| Development of a Bag-Opening Device for Apple Young Fruits Bagging System | Zhao, Qian, Ma, Zhang, 2025 | Journal of Agricultural Engineering (India) | 10.52151/jae2025622.1936 | 面向苹果幼果套袋系统研制开袋装置，报告开袋成功率与作业节拍。 | 2025 年最新机构工作，说明该方向仍活跃且仍停留在纯机械层面。 |
| Design of a Portable Peach Young Fruit Bagging Machine Empowered by Creo 2.0: An Optimization Algorithm Approach | Sun, Jing, Guo, Qiu, 2021 | 2021 3rd Int. Conf. on Artificial Intelligence and Advanced Manufacture (AIAM) | 10.1145/3495018.3501168 | 便携式桃幼果套袋机的结构设计与参数优化。 | 佐证"手持/便携半自动"是当前主流形态，全自动+视觉尚属空白。 |
| Design of a new fruit tree bagging machine | Wang, Zhang, Pu, Zhang, Wang, 2018 | IOP Conf. Series: Materials Science and Engineering | 10.1088/1757-899x/452/4/042099 | 提出一种新型果树套袋机整机方案。 | 早期整机方案，用于综述套袋机械的发展脉络。 |

---

## C. 果梗/果柄检测与定位（论文二直接相关，9 篇）

| 标题 | 作者/年份 | 期刊/会议 | DOI/ID | 核心内容 | 对本课题价值 |
|---|---|---|---|---|---|
| **Precision citrus segmentation and stem picking point localization using improved YOLOv8n-seg algorithm** | Li, Yin, Zuo, Pan, Zhang, 2025 | Frontiers in Plant Science | 10.3389/fpls.2025.1655093 | 改进 YOLOv8n-seg 同时完成柑橘果实精细分割与果梗采摘点定位。 | ★★论文二的**最强直接竞品**，方法路线（分割→果梗点）与用户设想高度重合，必须精读并明确差异化。 |
| **Picking-Point Localization Algorithm for Citrus Fruits Based on Improved YOLOv8 Model** | Liang, Jiang, Liu, Wu, Zheng, 2025 | Agriculture | 10.3390/agriculture15030237 | 基于改进 YOLOv8 的柑橘采摘点定位算法，面向自然果园环境。 | ★★第二强竞品，同为柑橘+YOLOv8。需对比其是否用 ROI 先验（用户的差异点很可能在这里）。 |
| Research on the Location of Citrus Picking Point Based on Structured Light Camera | Xiaomei, Bowen, Jianfei, 2019 | 2019 IEEE 4th Int. Conf. on Image, Vision and Computing (ICIVC) | 10.1109/icivc47709.2019.8980938 | 用结构光相机获取柑橘三维信息以定位采摘点。 | 柑橘采摘点定位的早期基线，做方法演进综述时的起点。 |
| **Robust keypoint-based method for peduncle pose estimation in unstructured environments** | Shi, Zhang, Wu, 2025 | Computers and Electronics in Agriculture | 10.1016/j.compag.2025.110380 | 提出基于关键点的果梗姿态估计方法，强调非结构化环境下的鲁棒性。 | ★★"关键点范式做果梗"的标杆方法论，是论文二方法选型（关键点 vs 分割）的核心参考。 |
| **3D pose estimation of tomato peduncle nodes using deep keypoint detection and point cloud** | Ci, Wang, Rapado-Rincón, Burusa, Kootstra, 2024 | Biosystems Engineering | 10.1016/j.biosystemseng.2024.04.017 | 深度关键点检测 + 点云融合，实现番茄果梗节点的三维姿态估计。 | ★★2D 关键点升维到 3D 的完整技术路线模板，直接可迁移到柑橘果梗点。 |
| Tomato Pedicel Picking-Point Localization via Improved YOLOv8n-EED-Seg and RGB-D Fusion | Wu, Liu, Teng, 2026 | Agriculture | 10.3390/agriculture16111197 | 改进 YOLOv8n-seg 结合 RGB-D 融合定位番茄果柄采摘点。 | 2026 最新，展示"分割+RGB-D"这条路的当前上限，用于设定性能对标基线。 |
| LeafRemoval-YOLO-K: A hybrid visual recognition network for stem-petiole segmentation and cutting point localization in tomato plants | Zhang, Guo, Zhao, Li, Yuan, 2026 | Computers and Electronics in Agriculture | 10.1016/j.compag.2026.111485 | 混合网络同时做茎-叶柄分割与切割点定位。 | 多任务（分割+关键点）混合架构范例，若论文二走多任务头可作为架构依据。 |
| Study on the fusion of improved YOLOv8 and depth camera for bunch tomato stem picking point recognition and localization | Song, Wang, Ma, Shi, Wang, 2024 | Frontiers in Plant Science | 10.3389/fpls.2024.1447855 | 改进 YOLOv8 融合深度相机实现串番茄果梗采摘点识别定位。 | 与用户此前串番茄选题呼应；串果 → 柑橘的迁移逻辑可复用。 |
| A method for litchi picking points calculation in natural environment based on main fruit bearing branch detection | Zhong, Xiong, Zheng, Liu, Liao, Huo, Yang, 2021 | Computers and Electronics in Agriculture | 10.1016/j.compag.2021.106398 | 先检测主结果枝，再几何推算荔枝采摘点。 | "先检测上下文结构、再几何推点"的思路，与用户"基于果实 ROI 反推果梗"逻辑同源，是重要的对照与致敬对象。 |

---

## D. 果实姿态与抓取（套袋操作前提，6 篇）

| 标题 | 作者/年份 | 期刊/会议 | DOI/ID | 核心内容 | 对本课题价值 |
|---|---|---|---|---|---|
| **The YOLO-OBB-Based Approach for Citrus Fruit Stem Pose Estimation and Robot Picking** | Ye, Ma, Lv, Guo, Lai, Ou, Li, Wu, 2025 | Agriculture | 10.3390/agriculture15222330 | 用旋转框（OBB）检测柑橘果梗并估计其姿态，驱动机器人采摘。 | ★★★**最接近论文二的工作**（柑橘+果梗+姿态）。OBB 是一条与"关键点/分割"并列的第三条路，必须在相关工作中正面比较。 |
| A Monocular Pose Estimation Framework for Automatic Dragon Fruit Harvesting Using Navel and Stem Keypoints | Yang, Bai, Zhang, Wu, 2026 | Horticulturae | 10.3390/horticulturae12040505 | 仅用单目图像，通过"果脐+果梗"两个关键点解算火龙果 6-DoF 姿态。 | ★用**果实自身解剖关键点**恢复姿态——套袋需要知道果实朝向才能确定套入方向，这是最省成本的方案，强烈建议借鉴。 |
| Efficient and Robust Orientation Estimation of Strawberries for Fruit Picking Applications | Wagner, Kirk, Hanheide, Cielniak, 2021 | 2021 IEEE ICRA | 10.1109/icra48506.2021.9561848 | 高效鲁棒的草莓朝向估计方法，面向实时采摘。 | 果实朝向估计的 ICRA 级基线，方法简洁、易复现，适合做对比实验。 |
| Fruit Detection and Pose Estimation for Grape Cluster–Harvesting Robot Using Binocular Imagery Based on Deep Neural Networks | Yin, Wen, Ning, Ye, Dong, Luo, 2021 | Frontiers in Robotics and AI | 10.3389/frobt.2021.626989 | 双目 + 深度网络实现葡萄串检测与姿态估计。 | 双目路线的完整系统参考，若柑橘套袋用双目可直接对标。 |
| Peduncle collision-free grasping based on deep reinforcement learning for tomato harvesting robot | Li et al., 2024 | Computers and Electronics in Agriculture | 10.1016/j.compag.2023.108488 | 用深度强化学习规划避免碰撞果梗的抓取姿态。 | 套袋末端必须"绕过果梗、从下方套入"，这篇给出了避碰规划的范式。 |
| Apple stem/calyx real-time recognition using YOLO-v5 algorithm for fruit automatic loading system | Wang, Jin, Wang, Xu, 2022 | Postharvest Biology and Technology | 10.1016/j.postharvbio.2021.111808 | YOLOv5 实时识别苹果梗端/萼端，用于自动上料定向。 | 证明"梗-萼轴"是判定果实朝向的可靠视觉线索，柑橘同理可用。 |

---

## E. 竞争地图：柑橘 + 套袋 + 视觉/机器人（1 篇新增 + 4 篇交叉引用）

| 标题 | 作者/年份 | 期刊/会议 | DOI/ID | 核心内容 | 对本课题价值 |
|---|---|---|---|---|---|
| Vision localization algorithms for apple bagging robot *(见 B)* | Gao et al., 2017 | CCDC 2017 | 10.1109/ccdc.2017.7978080 | **全库唯一**"套袋机器人 + 视觉定位"论文，对象是苹果，方法为深度学习之前的传统视觉。 | 该赛道 9 年来无人跟进，柑橘套袋视觉= **完全空白**。 |
| Design and Simulation of End Effector for Young-Pear-Bagging Robot *(见 B)* | Teng et al., 2024 | Processes | 10.3390/pr12020259 | 唯一近年"套袋机器人"论文，只做机构与仿真，**无感知模块**。 | 证实机构侧已有人做、感知侧无人做，本课题正好补位。 |
| The YOLO-OBB-Based Approach for Citrus Fruit Stem Pose Estimation and Robot Picking *(见 D)* | Ye et al., 2025 | Agriculture | 10.3390/agriculture15222330 | 柑橘果梗姿态估计，但**面向采摘（成熟果）**而非套袋（幼果）。 | 论文二最强竞争者，但生育期不同（幼果 vs 成熟果）是天然差异化空间。 |
| Precision citrus segmentation and stem picking point localization (YOLOv8n-seg) *(见 C)* | Li et al., 2025 | Front. Plant Sci. | 10.3389/fpls.2025.1655093 | 柑橘分割 + 果梗采摘点，同样面向采摘。 | 同上，需在"幼果/小目标/密集遮挡"上建立区分度。 |
| Crop design for improved robotic harvesting: A case study of sweet pepper harvesting | van Herck, Kurtser, Wittemans, Edan, 2020 | Biosystems Engineering | 10.1016/j.biosystemseng.2020.01.021 | 论证"改造农艺以适配机器人"比"让机器人适应现有农艺"更高效。 | 农机-农艺结合的方法论引文，可支撑"套袋作业本身可标准化以降低视觉难度"的论述。 |

---

## 阅读优先级建议

1. **先读 D 表第 1 篇（YOLO-OBB 柑橘果梗姿态, 2025）+ C 表前两篇（两篇 2025 柑橘果梗点定位）**。这三篇构成论文二的直接竞争圈，必须在一周内读完并做逐项对比表（数据集/生育期/输出形式/精度指标），才能确定论文二的差异化落点。目前看最可行的差异化是：**幼果期 + 基于分割 ROI 的先验约束 + 套袋（而非采摘）任务定义**。

2. **其次读 C 表第 4、5 篇（关键点范式：Shi 2025 CEA / Ci 2024 BiosysEng）**。这两篇决定论文二的方法论选型——关键点回归 vs 语义分割后处理 vs 旋转框。Ci 2024 的"2D 关键点 + 点云升维"是最成熟的 3D 落地路线，建议作为论文二的技术骨架。

3. **B 表全部 7 篇集中扫读（半天即可）**。这批文献总量小、方法陈旧，正是本课题的"论证弹药库"：读完你就能在引言里理直气壮地写出"现有套袋机械研究集中于开袋/供袋机构的力学设计，感知环节几乎空白"，并有 7 篇引文支撑。

4. **A 表按需取用，不必精读**。用途仅限引言第一段的必要性论证：柑橘套袋提质（Pareek 2025、Jiang 2022）+ 检疫害虫防控（Florida Entomologist 2019）+ 减农药（Frank 2018）三条线各引 1–2 篇即可。

5. **D 表第 2 篇（火龙果单目双关键点姿态, 2026）值得单独深读**。它给出了"用果实解剖关键点直接解算 6-DoF"的低成本方案。套袋比采摘更依赖果实朝向（袋口必须对准果实并避开果梗），若把这一思路迁移为"柑橘果脐-果梗轴"，很可能是论文二之后第三篇论文的创新点。

---

## E 方向竞争密度判断

**结论：柑橘套袋视觉是一个近乎无人竞争的窄赛道，但其"上游"（柑橘果梗点定位）竞争已经很激烈。**

- **套袋自动化整体（B+E）：竞争密度极低。** 全库检索仅得 7 篇套袋机械论文，其中带视觉的**只有 1 篇（Gao 2017, CCDC 会议）**，且是深度学习普及前的传统方法；2024 年梨套袋机器人（Teng）只做机构仿真、完全没有感知模块。近 9 年"套袋 + 深度学习视觉"零产出。中文侧同样以专利为主（如 CN116897751B 苹果幼果套袋装置），少见高水平期刊论文。这意味着"柑橘套袋视觉"作为**课题包装/故事线**几乎没有撞车风险。

- **柑橘果梗点定位（C+D）：竞争密度中高，且 2025 年集中爆发。** 仅 2025 一年就出现 3 篇柑橘果梗/采摘点定位论文（Front. Plant Sci. 10.3389/fpls.2025.1655093、Agriculture 10.3390/agriculture15030237、Agriculture 10.3390/agriculture15222330），方法均为 YOLOv8 系改进。这说明**论文二如果只写"改进 YOLOv8 定位柑橘果梗点"，创新性会被这三篇严重稀释**。

- **战略建议：用低竞争的"套袋"包装高竞争的"果梗定位"。** 三个可立即操作的差异化杠杆：(1) **生育期**——竞品全部面向成熟果采摘，幼果期果梗更细、颜色与叶片近似、遮挡更重，是尚未被占领的难题；(2) **ROI 先验**——竞品多为端到端一步出点，用户论文①已有的高质量实例分割结果可作为显式 ROI 约束，构成"两篇论文串成一条链"的独特叙事；(3) **任务定义**——采摘要的是"剪切点"，套袋要的是"袋口对准点 + 果实朝向"，输出语义不同，评价指标也可以重新定义，天然避开与三篇竞品的正面指标比拼。
