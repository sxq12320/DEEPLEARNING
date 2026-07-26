# 主题6：伪装目标检测(COD)模块级机制调研 —— 可迁移进 YOLO11n-seg neck/head 的模块设计

课题背景：绿色未成熟柑橘与叶片同色（"绿绿伪装"），实例分割基线 YOLO11n-seg。
锚点文献：Zhai et al. 2024, Comput. Electron. Agric., doi:10.1016/j.compag.2024.109356（已确认将绿果检测重构为 COD 问题，本文档不重复计入篇数）。
核验方式：全部条目经 Crossref API（DOI）或 arXiv API（ID）逐一核验，核验日期 2026-07-26。

---

## A. COD 经典奠基（问题定义与两阶段范式）

### 1. SINet — "Camouflaged Object Detection"
- 第一作者/年份/venue：Deng-Ping Fan, 2020, CVPR 2020
- DOI: 10.1109/CVPR42600.2020.00285
- 核心机制：模仿捕食者"搜索→识别"两阶段：搜索模块(SM)用多尺度感受野块(RF, 非对称卷积+空洞卷积)扩大 center-surround 感受野粗定位；识别模块(IM)用部分解码器组件(PDC)只聚合高层特征做级联精化。提出 COD10K 数据集，奠定 COD 基准。
- 迁移建议：RF 块的"大感受野 center-surround 对比"可替换 YOLO11 neck 中 C3k2 的瓶颈卷积，增强远处低对比小绿果的粗定位能力。

### 2. SINet-v2 — "Concealed Object Detection"
- 第一作者/年份/venue：Deng-Ping Fan, 2022, IEEE TPAMI
- arXiv:2102.10274（期刊版 DOI: 10.1109/TPAMI.2021.3085766）
- 核心机制：纹理增强模块(TEM)在每级特征内用 4 条不同空洞率的分支模拟视皮层感受野捕捉细微纹理差异；邻域连接解码器(NCD)只连接相邻层级避免语义稀释；分组反转注意力(GRA)用预测图的反转图作引导、按通道分组迭代擦除已定位区域、逼出残余伪装线索。
- 迁移建议：GRA 的"反转注意力擦除-再挖掘"可做成 seg head 前的轻量精化块，专治绿果边缘与叶片粘连处的漏分割。

## B. 干扰挖掘 / 边界引导 / 上下文融合（CNN 时代模块库）

### 3. PFNet — "Camouflaged Object Segmentation with Distraction Mining"
- 第一作者/年份/venue：Haiyang Mei, 2021, CVPR 2021
- arXiv:2104.10475（DOI: 10.1109/CVPR46437.2021.00866）
- 核心机制：定位模块(PM)串联通道注意力+空间注意力模拟捕食者视野收缩；聚焦模块(FM)显式建模两类"干扰"：用上一级预测的正/反注意力分别乘当前特征得到假阳性(FD)与假阴性(FD)干扰流，各经上下文探索块(CE, 4 分支空洞卷积)后从主特征中做逐元素减(去假阳)与加(补假阴)。
- 迁移建议：FM 的"减假阳/加假阴"双流可直接接在 YOLO neck 的 P3 输出后，专门抑制被误检为果实的高亮叶片斑块（最典型的绿果假阳源）。

### 4. C2FNet — "Context-aware Cross-level Fusion Network for Camouflaged Object Detection"
- 第一作者/年份/venue：Yujia Sun, 2021, IJCAI 2021
- arXiv:2105.12555（DOI: 10.24963/ijcai.2021/142）
- 核心机制：注意力诱导跨级融合模块(ACFM)先把相邻两级特征拼接后送入多尺度通道注意力(MSCA, 全局+局部两条通道注意力并联)再加权融合，解决层级间语义-细节失配；双分支全局上下文模块(DGCM)用两条不同下采样率的分支提取多尺度全局上下文再残差融合。
- 迁移建议：MSCA 是即插即用的极轻量注意力，可直接嵌入 YOLO11 neck 的 concat 之后，代价几乎为零，适合 nano 预算下的第一个消融项。

### 5. BGNet — "Boundary-Guided Camouflaged Object Detection"
- 第一作者/年份/venue：Yujia Sun, 2022, IJCAI 2022
- arXiv:2207.00794（DOI: 10.24963/ijcai.2022/186）
- 核心机制：边缘感知模块(EAM)融合低层特征(细节)与最高层特征(语义)专门预测"物体相关"的边缘图（而非全图边缘），受显式边缘监督；边缘引导特征模块(EFM)把边缘图下采样后与各级特征做逐元素乘+通道注意力，将边界先验注入每一级；上下文聚合模块(CAM)多分支空洞卷积逐级解码。
- 迁移建议：为 YOLO11n-seg 增加一条受边缘 GT 监督的辅助边界分支（训练时有、推理可留可弃），把边界热图乘回 P3/P4 特征，是治"果-叶边界模糊"最直接的方案。

### 6. UGTR — "Uncertainty-Guided Transformer Reasoning for Camouflaged Object Detection"
- 第一作者/年份/venue：Fan Yang, 2021, ICCV 2021
- DOI: 10.1109/ICCV48922.2021.00411
- 核心机制：用概率表征模型(Bayesian 学习)显式估计每个像素预测的不确定性图，再把高不确定区域作为 Transformer 推理的重点：不确定性图调制注意力权重，使模型"带着疑问"在模糊区域（多为伪装边界）做上下文推理而非直接回归。
- 迁移建议：可把不确定性估计做成 seg head 的辅助输出，用其加权 loss（难区域加权），零推理开销地提升远处低对比果实的召回。

## C. 多级 Zoom / 迭代放大 / 高分辨率反馈（对"远处小目标"最相关）

### 7. ZoomNet — "Zoom In and Out: A Mixed-scale Triplet Network for Camouflaged Object Detection"
- 第一作者/年份/venue：Youwei Pang, 2022, CVPR 2022
- arXiv:2203.02688（DOI: 10.1109/CVPR52688.2022.00220）
- 核心机制：模仿人类观察伪装图时"放大缩小"的行为，将输入按 1.5×/1×/0.75× 三尺度送入共享骨干，尺度合并单元(SMU)对齐融合三路特征；分层混合尺度单元(HMU)在通道分组内做迭代跨组交互强化细微差异；另设不确定性感知损失(UAL)以预测图的模糊度自适应加权。
- 迁移建议：UAL 损失可零成本加进 YOLO11n-seg 训练；三尺度推理太贵，但可退化为"训练期多尺度一致性蒸馏"。

### 8. SegMaR — "Segment, Magnify and Reiterate: Detecting Camouflaged Objects the Hard Way"
- 第一作者/年份/venue：Qi Jia, 2022, CVPR 2022
- DOI: 10.1109/CVPR52688.2022.00467
- 核心机制：分割-放大-再迭代的多阶段流水线：第一阶段粗分割后，用融合固定注视先验(判别区域由采样式注意力生成的 discriminative mask)裁剪并放大目标区域，送回同一网络再分割，迭代多轮逐步逼近小而模糊的伪装目标。
- 迁移建议：对采摘机器人可实现为"两段式推理"：整图 nano 模型粗定位果串 ROI，再对 ROI 放大二次分割，远处小果串精度可大幅提升且总算力可控。

### 9. HitNet — "High-resolution Iterative Feedback Network for Camouflaged Object Detection"
- 第一作者/年份/venue：Xiaobin Hu, 2023, AAAI 2023
- arXiv:2203.11624（DOI: 10.1609/aaai.v37i1.25167）
- 核心机制：针对低分辨率特征丢失伪装细节的问题，设计高分辨率特征反馈回路：基于 Transformer 的迭代反馈单元把上一轮的高分辨率精化特征经反馈连接注入下一轮的低层编码特征，并用迭代反馈损失约束每轮输出，逐轮找回被下采样抹掉的细纹理。
- 迁移建议：其"高分辨率反馈"思想提示在 YOLO11n-seg 中保留/增强 P2 高分辨率支路并做一次自顶向下反馈融合，对远处小果的 mask 质量比堆注意力更有效。

## D. 频域分解辨伪装（FEDER 重点方向）

### 10. FEDER — "Camouflaged Object Detection with Feature Decomposition and Edge Reconstruction"
- 第一作者/年份/venue：Chunming He, 2023, CVPR 2023
- DOI: 10.1109/CVPR52729.2023.02111
- 核心机制：认为前景背景"看起来像"是因为共享的低频分量占主导，故用可学习小波(deep wavelet-like decomposition)把特征显式分解为高/低频子带，经频域注意力筛出最具判别力的分量再重组；同时用 ODE 启发的边缘重建模块从高频分量恢复完整物体边缘，辅助分割头输出。
- 迁移建议：绿果与绿叶色彩(低频)几乎同分布，但果面光滑/叶面纹理的高频统计不同——把可学习小波分解块塞进 neck 的 P3 层做高频增强，是本课题最对症的"辨伪装"机制。

### 11. FPNet — "Frequency Perception Network for Camouflaged Object Detection"
- 第一作者/年份/venue：Runmin Cong, 2023, ACM MM 2023
- arXiv:2308.08924（DOI: 10.1145/3581783.3612083）
- 核心机制：用八度卷积(octave convolution)实现端到端在线高/低频分解（无需离线 DCT/小波变换），高频支路捕捉边缘细节、低频支路捕捉整体轮廓，两阶段"频域感知粗定位→细节保持精定位"级联，第二阶段用频域特征对齐融合修正边界。
- 迁移建议：八度卷积本身省算力(低频支路降分辨率计算)，用它替换 YOLO11n neck 部分标准卷积可"减参数的同时加频域先验"，非常契合 nano 预算。

### 12. FGSA-Net — "Frequency-Guided Spatial Adaptation for Camouflaged Object Detection"
- 第一作者/年份/venue：Shizhou Zhang, 2024, IEEE Trans. Multimedia (2025 刊出)
- arXiv:2409.12421
- 核心机制：冻结预训练基础模型，仅训练频域引导的空间适配器(adapter)：将 adapter 中间特征经 FFT 分组为不同频带，按各频带能量动态调制空间注意力权重，使极少量可训练参数即可把通用特征改造为伪装敏感特征。
- 迁移建议：其"频带能量→空间注意力权重"的调制方式可简化为一个几千参数的小模块插在 seg head 前，属于典型的"小论文级"创新点改造模板。

## E. 轻量/高效 COD（能塞进 nano 检测器的）

### 13. DGNet — "Deep Gradient Learning for Efficient Camouflaged Object Detection"
- 第一作者/年份/venue：Ge-Peng Ji, 2023, Machine Intelligence Research
- arXiv:2205.12853（DOI: 10.1007/s11633-022-1365-9）
- 核心机制：用"物体级梯度图"取代边缘图做辅助监督（梯度图同时编码强度变化与位置，比细边缘线更易学）；网络解耦为上下文分支与纹理(梯度)分支，梯度诱导转换模块(GIT)按软分组方式将两分支特征交叉融合。DGNet-S 仅 8.3M 参数、80fps，是首个明确面向高效 COD 的工作。
- 迁移建议："梯度图辅助监督"对 YOLO11n-seg 是零推理成本的即得方案——Sobel/Scharr 算出果实区域梯度 GT，加一条训练期辅助头即可。

### 14. ERRNet — "Fast Camouflaged Object Detection via Edge-based Reversible Re-calibration Network"
- 第一作者/年份/venue：Ge-Peng Ji, 2022, Pattern Recognition
- arXiv:2111.03216（DOI: 10.1016/j.patcog.2022.108414）
- 核心机制：选择性边缘聚合模块(SEA)只聚合对伪装体有效的边缘先验；可逆再校准解码器(RRD)利用"先验-当前预测"之间的可逆变换在低分辨率下反复校准模糊区域与边界区域，达到 79.3 FPS 的实时速度。
- 迁移建议：RRD 的低分辨率校准思路证明"边界精化不必在高分辨率做"，可在 P4/P5 低分辨率特征上做便宜的 mask 校准再上采样，控住 nano 的时延。

## F. 农业伪装迁移案例（除 Zhai 2024 外）

### 15. SWNet — "SWNet: A Cross-Spectral Network for Camouflaged Weed Detection"
- 第一作者/年份/venue：Henry O. Velesaca, 2026, arXiv preprint
- arXiv:2604.16147
- 核心机制：将"绿色杂草藏在绿色作物中"显式定义为农业伪装检测问题，用跨光谱(可见光+近红外)双流网络，在特征层做跨光谱注意力融合，利用植物间光谱反射差异突破 RGB 域的"绿绿同色"。
- 迁移建议：佐证了"绿色目标藏于绿色背景=COD"这一问题重构在农业界正在扩散（杂草/果实两条线），也提示若允许换传感器，NIR 通道是绕开 RGB 伪装的硬件级方案；论文层面可引用其强化"绿果=伪装目标"的立论。

### 16. Zhai 2024（锚点，已核验存在）— "Green fruit detection methods: Innovative application of camouflage object detection and multilevel feature mining"
- Comput. Electron. Agric., DOI: 10.1016/j.compag.2024.109356
- 说明：目前 Crossref/arXiv 检索范围内，"camouflage + fruit"的直接结合仍只有该文（另有 2025 一篇入侵植被监测用 COD 技术：doi:10.5220/0013630200003967，及 RGB-D COD 农业相关基准 doi:10.1515/phys-2024-0060，均非果实检测），说明"柑橘版 COD 模块"仍是低竞争创新窗口。

---

## 调研结论（模块迁移优先级）

1. **首选机制一：频域分解辨伪装（FEDER 可学习小波 / FPNet 八度卷积）**。绿果与绿叶的差异不在颜色(低频)而在表面纹理统计(高频)，频域分解是机理上最对症的；且八度卷积天然省算力，与 nano 预算兼容。
2. **首选机制二：边界引导 + 假阳/假阴干扰挖掘（BGNet EAM/EFM + PFNet Focus 模块）**。果-叶粘连边界是实例分割 mask 质量的主要瓶颈，辅助边界监督分支（训练期）+ 反/正注意力双流精化（推理期轻量）组合成本低、消融故事完整。
3. 零成本加分项：DGNet 梯度辅助监督、ZoomNet 不确定性感知损失 UAL、UGTR 不确定性加权——均只改训练不改推理。
4. 系统级备选：SegMaR 两段式"粗定位→ROI 放大再分割"针对远处小果串，适合写进采摘系统章节而非模块创新章节。
