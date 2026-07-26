# 主题2：小目标/微小目标检测的网络结构改进（文献核实清单）

课题背景：YOLO11n-seg 柑橘幼果实例分割；痛点为远处果实极小（<32px，部分 <16px）、运动/失焦模糊、光照暗。
共 18 篇（用户点名方法全覆盖，故略超 12-15 篇上限）。核实方式：17 篇经 Semantic Scholar Graph API（arXiv ID 直查端点，2026-07-26），HWD 一篇经 Crossref API 核实 DOI。所有 DOI/arXiv ID 均可核验，无编造。

---

## A. 高分辨率检测层 P2 / 浅层特征利用

### 1. QueryDet: Cascaded Sparse Query for Accelerating High-Resolution Small Object Detection
- 第一作者：Chenhongyi Yang ｜ 年份：2022（arXiv 2021）｜ venue：CVPR 2022
- DOI：10.1109/CVPR52688.2022.01330 ｜ arXiv：2103.09136
- 核心思想：证明小目标检测必须利用高分辨率浅层特征（P2 级），并用"级联稀疏查询"先在低分辨率层粗定位小目标、再仅在高分辨率层的对应稀疏位置精细计算，把加 P2 层的计算代价降到可接受。
- 柑橘适用性：为"YOLO11n-seg 增加 P2 检测头解决 <32px 幼果"提供了最直接的理论依据与代价控制思路。
- 建议接入点：Head——在 P3/P4/P5 基础上增加 P2 (160×160) 检测/分割输出层；论文写作中引用其论证浅层特征对小目标的必要性。

### 2. Towards Large-Scale Small Object Detection: Survey and Benchmarks
- 第一作者：Gong Cheng ｜ 年份：2023（arXiv 2022）｜ venue：IEEE TPAMI
- DOI：10.1109/TPAMI.2023.3290594 ｜ arXiv：2207.14096
- 核心思想：小目标检测权威综述，系统归纳数据增强、多尺度特征融合、上下文建模、超分、专用度量与标签分配五大改进路线，并发布 SODA-D/SODA-A 大规模小目标基准。
- 柑橘适用性：可作为小论文 Related Work 的总纲引用，并支撑"<32px 定义小目标、<16px 定义极小目标"的表述。
- 建议接入点：论文引言/相关工作章节的框架性引用；实验分析按其分类学组织消融维度。

## B. SPD-Conv 无步长卷积下采样

### 3. No More Strided Convolutions or Pooling: A New CNN Building Block for Low-Resolution Images and Small Objects (SPD-Conv)
- 第一作者：Raja Sunkara ｜ 年份：2022 ｜ venue：ECML-PKDD 2022
- DOI：10.48550/arXiv.2208.03641 ｜ arXiv：2208.03641
- 核心思想：指出步长卷积/池化会不可逆地丢弃小目标的细粒度信息，提出 space-to-depth + 无步长卷积（SPD-Conv）替代下采样，将空间信息无损折叠进通道维。
- 柑橘适用性：直接针对"低分辨率+小目标+模糊"设定（论文实验即含低清图像），适合替换 YOLO11n 骨干中的 stride-2 Conv，保留远处幼果的像素级证据。
- 建议接入点：Backbone——替换 YOLO11n 各 stage 的 stride-2 下采样卷积（尤其浅层前两次下采样）。

## C. 特征金字塔改进

### 4. EfficientDet: Scalable and Efficient Object Detection (BiFPN)
- 第一作者：Mingxing Tan ｜ 年份：2020（arXiv 2019）｜ venue：CVPR 2020
- DOI：10.1109/CVPR42600.2020.01079 ｜ arXiv：1911.09070
- 核心思想：提出 BiFPN——带可学习权重的双向（自顶向下+自底向上）跨尺度特征融合，去除单输入节点、增加同尺度跳连，以极低代价反复融合多尺度特征。
- 柑橘适用性：加权融合可让网络自适应上调浅层（小目标）特征贡献，是 Neck 改进的经典 baseline 对照。
- 建议接入点：Neck——用加权 BiFPN 替换 YOLO11 的 PAN，或仅引入其可学习融合权重；消融实验的对照方法。

### 5. AFPN: Asymptotic Feature Pyramid Network for Object Detection
- 第一作者：Guoyu Yang ｜ 年份：2023 ｜ venue：IEEE SMC 2023
- DOI：10.1109/SMC53992.2023.10394415 ｜ arXiv：2306.15988
- 核心思想：渐近式融合——先融合相邻层级、逐步纳入更远层级，配合自适应空间加权抑制跨级语义鸿沟，缓解非相邻层直接融合导致的小目标信息被高层语义淹没。
- 柑橘适用性：P2 引入后层级跨度增大（P2-P5），AFPN 的渐近融合可缓解浅层细节与深层语义的冲突。
- 建议接入点：Neck——整体替换 PAN-FPN，与 P2 层配合使用。

### 6. Gold-YOLO: Efficient Object Detector via Gather-and-Distribute Mechanism
- 第一作者：Chengcheng Wang ｜ 年份：2023 ｜ venue：NeurIPS 2023
- DOI：10.48550/arXiv.2309.11331 ｜ arXiv：2309.11331
- 核心思想：提出 Gather-and-Distribute（GD）机制：先全局汇聚所有层级特征（卷积分支+注意力分支），再按需分发回各尺度，避免传统 FPN 逐层传递造成的信息损失。
- 柑橘适用性：GD 的全局汇聚使 P2/P3 小目标层可直接获取全图上下文（枝叶遮挡下的果串位置先验），无需逐层稀释。
- 建议接入点：Neck——用 Low-GD/High-GD 替换 PAN；轻量版分支适配 n 级模型。

### 7. Accurate Leukocyte Detection Based on Deformable-DETR and Multi-Level Feature Fusion...（HS-FPN 原始出处）
- 第一作者：Yifei Chen ｜ 年份：2024 ｜ venue：Computers in Biology and Medicine
- DOI：10.1016/j.compbiomed.2024.107917 ｜ arXiv：2401.00926
- 核心思想：提出 HS-FPN（High-level Screening-feature Fusion Pyramid）：用高层语义特征经 CA 通道注意力生成权重，"筛选"过滤低层特征后再融合，突出小而密目标、抑制背景冗余。
- 柑橘适用性：白细胞与幼果同为小尺度、低对比度、背景干扰强的目标，其高层筛选思想可抑制枝叶背景对浅层特征的噪声污染。
- 建议接入点：Neck——以 HS-FPN 替换/改造 FPN 的横向连接（高层筛选低层）。

### 8. ASF-YOLO: A Novel YOLO Model with Attentional Scale Sequence Fusion for Cell Instance Segmentation
- 第一作者：Ming Kang ｜ 年份：2024（arXiv 2023）｜ venue：Image and Vision Computing
- DOI：10.1016/j.imavis.2024.105057 ｜ arXiv：2312.06458
- 核心思想：面向实例分割提出 SSFF（尺度序列特征融合，将多尺度特征堆叠为 3D 序列做 3D 卷积）与 TFE（三特征编码器，放大并拼接大/中/小尺度细节），并加 CPAM 通道-位置注意力。
- 柑橘适用性：同为 YOLO 系实例分割且针对小而密目标（细胞），是与本课题任务形态最接近的可迁移方案，也是理想对照方法。
- 建议接入点：Neck——SSFF+TFE 模块嵌入 YOLO11n-seg 颈部；实验部分作为 SOTA 对照。

## D. 内容感知上采样

### 9. CARAFE: Content-Aware ReAssembly of FEatures
- 第一作者：Jiaqi Wang ｜ 年份：2019 ｜ venue：ICCV 2019
- DOI：10.1109/ICCV.2019.00310 ｜ arXiv：1905.02188
- 核心思想：内容感知上采样：由输入特征自身预测每个位置的重组核，在大感受野内按内容聚合信息，取代与内容无关的最近邻/双线性插值。
- 柑橘适用性：FPN 自顶向下路径的上采样质量直接决定小目标层特征保真度；模糊暗光下内容感知重组比固定插值更能保住幼果边缘。
- 建议接入点：Neck——替换 YOLO11 上采样算子（nn.Upsample）；分割头 proto 上采样同样适用。

### 10. Learning to Upsample by Learning to Sample (DySample)
- 第一作者：Wenze Liu ｜ 年份：2023 ｜ venue：ICCV 2023
- DOI：10.1109/ICCV51070.2023.00554 ｜ arXiv：2308.15085
- 核心思想：把上采样重构为"学习采样点位置"（point sampling），无需动态卷积核生成，参数与延迟远低于 CARAFE 而效果相当或更好。
- 柑橘适用性：对算力受限的采摘机器人端侧部署（YOLO11n 量级）比 CARAFE 更友好，几乎零开销换取小目标上采样增益。
- 建议接入点：Neck——同 CARAFE 位置，二选一（轻量优先 DySample），消融对比两者。

## E. 感受野与上下文

### 11. Receptive Field Block Net for Accurate and Fast Object Detection (RFB)
- 第一作者：Songtao Liu ｜ 年份：2018（arXiv 2017）｜ venue：ECCV 2018
- DOI：10.1007/978-3-030-01252-6_24 ｜ arXiv：1711.07767
- 核心思想：仿人类视觉感受野的偏心率结构，用多分支不同核尺寸+不同空洞率卷积构建 RFB 模块，在轻量骨干上扩大并丰富感受野。
- 柑橘适用性：轻量模型增强上下文的经典模块，可帮助利用果梗/叶片等周边线索判别模糊小果，且开销小、适配 n 级模型。
- 建议接入点：Backbone 末端（替换/并联 SPPF）或 Neck 小目标分支前插入 RFB。

### 12. Large Selective Kernel Network for Remote Sensing Object Detection (LSKNet)
- 第一作者：Yuxuan Li ｜ 年份：2023 ｜ venue：ICCV 2023
- DOI：10.1109/ICCV51070.2023.01540 ｜ arXiv：2303.09030
- 核心思想：针对遥感小目标提出大选择核机制：分解的大核卷积序列提供长程上下文，空间选择机制按目标类型动态调节感受野大小。
- 柑橘适用性：遥感小目标与远景果园图像高度类似（目标小、依赖场景上下文），动态感受野可按远近果自适应取上下文。
- 建议接入点：Backbone——LSK 模块替换 C3k2 中的瓶颈单元，或作为小目标层的上下文增强块。

### 13. UniRepLKNet: A Universal Perception Large-Kernel ConvNet...
- 第一作者：Xiaohan Ding ｜ 年份：2024（arXiv 2023）｜ venue：CVPR 2024
- DOI：10.1109/CVPR52733.2024.00527 ｜ arXiv：2311.15599
- 核心思想：给出大核 ConvNet 的系统设计准则（含 Dilated Reparam Block：大核+并联小空洞核重参数化），用少量大核层高效获得大有效感受野。
- 柑橘适用性：Dilated Reparam Block 可重参数化，推理时零额外开销地扩大感受野，契合暗光模糊下需要更大上下文而算力受限的场景。
- 建议接入点：Backbone——以 DRB 改造 C3k2 卷积；作为大核路线的设计准则引用。

### 14. Poly Kernel Inception Network for Remote Sensing Detection (PKINet, 含 CAA)
- 第一作者：Xinhao Cai ｜ 年份：2024 ｜ venue：CVPR 2024
- DOI：10.1109/CVPR52733.2024.02617 ｜ arXiv：2403.06258
- 核心思想：多尺度并行卷积核（Inception 式，不用空洞）捕获不同尺度纹理，并提出 CAA（Context Anchor Attention）用条带卷积捕获长程中心-周边上下文关系。
- 柑橘适用性：CAA 是即插即用注意力，可给小目标层引入长程上下文（沿枝条方向的条带卷积恰合果串生长结构）。
- 建议接入点：Neck/Backbone——CAA 模块插入 C3k2 后或与 SPPF 组合；多尺度核思想用于浅层特征提取。

## F. 小目标专用度量与标签分配

### 15. A Normalized Gaussian Wasserstein Distance for Tiny Object Detection (NWD)
- 第一作者：Jinwang Wang ｜ 年份：2021 ｜ venue：arXiv（预印本，被广泛引用）
- arXiv：2110.13389（无期刊 DOI，以 arXiv ID 为准）
- 核心思想：指出 IoU 对微小目标的位置偏移极度敏感（几像素偏移即 IoU 骤降），将框建模为二维高斯分布、用归一化 Wasserstein 距离替代 IoU 做度量/分配/NMS。
- 柑橘适用性：<16px 幼果在训练中常因 IoU 波动被判负样本，NWD 可稳定正样本供给——这是标签分配层面的痛点根治。
- 建议接入点：Loss/Assigner——TaskAlignedAssigner 的对齐度量与回归损失中以 NWD（或 IoU-NWD 加权）替换 CIoU。

### 16. RFLA: Gaussian Receptive Field based Label Assignment for Tiny Object Detection
- 第一作者：Chang Xu ｜ 年份：2022 ｜ venue：ECCV 2022
- DOI：10.48550/arXiv.2208.08738 ｜ arXiv：2208.08738
- 核心思想：指出基于 anchor IoU 或 center 先验的分配对微小目标存在尺度失衡，改用"特征点高斯感受野与 GT 的距离"（RFD）做层级化标签分配，保证极小目标也能分到足量正样本。
- 柑橘适用性：与 NWD 互补，从感受野匹配角度解决 <16px 幼果正样本稀缺，且与 anchor-free 的 YOLO11 思想兼容。
- 建议接入点：Assigner——以 RFD 先验改造 YOLO11 的正样本分配（与 NWD 二选一或对比消融）。

## G. 小目标数据增强

### 17. Augmentation for Small Object Detection (copy-paste)
- 第一作者：Mate Kisantal ｜ 年份：2019 ｜ venue：CS & IT / 预印本（小目标 copy-paste 原始出处）
- DOI：10.5121/csit.2019.91713 ｜ arXiv：1902.07296
- 核心思想：统计发现 COCO 中小目标出现图像少、被分配 anchor 少导致损失贡献不足，提出对含小目标图像过采样并将小目标掩膜多次复制-粘贴到同图不同位置以提升小目标 AP。
- 柑橘适用性：幼果数据集天然存在"远景小果样本少"的分布不均，分割任务自带掩膜、可零成本实施 copy-paste。
- 建议接入点：数据管线——训练时对小果实例（<32px）做过采样+掩膜复制粘贴，与 mosaic 叠加；属零推理开销改进。

## H. Haar 小波下采样

### 18. Haar Wavelet Downsampling: A Simple but Effective Downsampling Module for Semantic Segmentation (HWD)
- 第一作者：Guoping Xu ｜ 年份：2023 ｜ venue：Pattern Recognition
- DOI：10.1016/j.patcog.2023.109819（Crossref 核实；无 arXiv 版本）
- 核心思想：用无损的 Haar 小波变换替代步长卷积/池化做下采样：空间分辨率减半的同时把高频细节保留到通道维，降低下采样的信息熵损失。
- 柑橘适用性：与 SPD-Conv 同一动机但基于频域、可分离出高频子带，对保留模糊暗光小果的边缘高频信息更有针对性，二者可做对照消融。
- 建议接入点：Backbone——替换下采样层（与 SPD-Conv 互为对照）；分割任务出身，与 seg 头兼容性好。

---

## 组合建议（面向"算法创新小论文"）
- 推荐主线：P2 检测层（#1 依据）+ SPD-Conv 或 HWD 下采样（#3/#18 二选一）+ DySample 上采样（#10）+ NWD 度量（#15）——四处改动各针对一个痛点（分辨率/信息丢失/上采样保真/标签分配），故事完整。
- 上下文增强备选：CAA（#14）或 LSK（#12）插入小目标分支，针对"模糊、暗"补上下文线索。
- 对照与消融：BiFPN/AFPN/Gold-YOLO/ASF-YOLO（#4-#6、#8）作 Neck 路线对照；CARAFE vs DySample、SPD vs HWD、NWD vs RFLA 三组消融。
