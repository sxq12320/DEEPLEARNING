# 主题5：轻量骨干 / 注意力机制 / 优化器与训练策略 — 原始出处核验清单

课题背景：YOLO11n-seg 柑橘幼果实例分割（nano 级 ~2.8M 参数；远处果实小、模糊、暗）。

**核验方式**：Semantic Scholar API 本网络环境 403/429 不可用，改用 arXiv 官方 API（export.arxiv.org，直连绕代理）逐条批量核验 21 个 arXiv ID（标题/作者/年份全部返回一致），无 arXiv 的 2 篇经 Crossref（MLCA、LSKA 期刊版 DOI）与 DBLP（SimAM）核验。**23 条全部为真实出处，无编造**。核验原始数据见同目录 `raw/arxiv_verified.json`、`raw/crossref_results.json`。

---

## A. 轻量骨干（6 篇）

### 1. StarNet
- **标题**：Rewrite the Stars
- **第一作者**：Xu Ma｜**年份**：2024｜**venue**：CVPR 2024
- **ID**：arXiv:2403.19967
- **核心思想**：证明逐元素乘法（star operation）可在低维空间隐式映射到高维非线性特征空间，据此构建极简 4 层级骨干 StarNet，无需复杂设计即达高效精度。
- **适用性**：StarNet-S050/S1 量级与 YOLO11n 骨干参数量匹配，star block 可作 C3k2 内替换单元。
- **建议接入点**：用 StarBlock 替换 backbone 中 C3k2 的 Bottleneck，或整体替换 P2–P4 骨干段。

### 2. FasterNet / PConv
- **标题**：Run, Don't Walk: Chasing Higher FLOPS for Faster Neural Networks
- **第一作者**：Jierun Chen｜**年份**：2023｜**venue**：CVPR 2023
- **ID**：arXiv:2303.03667
- **核心思想**：提出部分卷积 PConv——只对 1/4 通道做常规卷积、其余通道恒等传递，减少冗余计算与访存，FLOPS（每秒浮点吞吐）更高而非仅 FLOPs 更低。
- **适用性**：PConv 是 nano 检测器最常用的降参单元，嵌入式部署友好（低访存）。
- **建议接入点**：C3k2 → C3k2-Faster（Bottleneck 内 3×3 卷积换 PConv+PWConv），也可用于 neck。

### 3. RepViT
- **标题**：RepViT: Revisiting Mobile CNN From ViT Perspective
- **第一作者**：Ao Wang｜**年份**：2024（arXiv 2023.07）｜**venue**：CVPR 2024
- **ID**：arXiv:2307.09283
- **核心思想**：以 ViT 的宏观设计（分离 token mixer/channel mixer、结构重参数化）改造 MobileNetV3，得到纯 CNN 移动骨干，移动端延迟-精度优于同期轻量 ViT。
- **适用性**：RepViTBlock 训练时多分支、推理时重参数化为单路，加参不加时延，适合边缘部署叙事。
- **建议接入点**：替换骨干浅层 stage（P2/P3），或以 RepViTBlock 改造 C3k2。

### 4. EfficientViT
- **标题**：EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention
- **第一作者**：Xinyu Liu｜**年份**：2023｜**venue**：CVPR 2023
- **ID**：arXiv:2305.07027
- **核心思想**：分析 ViT 速度瓶颈在访存与注意力头冗余，提出级联分组注意力（CGA）三明治块，构建高吞吐轻量 ViT 家族。
- **适用性**：为 nano 模型引入全局建模能力且吞吐可控；注意区分同名的 Cai et al. ICCV 2023 版本（arXiv:2205.14756），引用时勿混淆。
- **建议接入点**：仅替换骨干深层（P4/P5）stage，浅层保留 CNN，控制总参数。

### 5. GhostNetV2
- **标题**：GhostNetV2: Enhance Cheap Operation with Long-Range Attention
- **第一作者**：Yehui Tang｜**年份**：2022｜**venue**：NeurIPS 2022
- **ID**：arXiv:2211.12905
- **核心思想**：在 Ghost 廉价操作上叠加解耦全连接注意力（DFC attention），以水平+垂直两个方向的 FC 捕获长程依赖，硬件友好。
- **适用性**：Ghost 系为轻量农业检测论文最常用对照/组件，DFC 的长程性对远处小果的上下文聚合有益。
- **建议接入点**：C3k2 内 Bottleneck → GhostV2 bottleneck；或作为消融对照骨干。

### 6. LSNet
- **标题**：LSNet: See Large, Focus Small
- **第一作者**：Ao Wang｜**年份**：2025｜**venue**：CVPR 2025
- **ID**：arXiv:2503.23135
- **核心思想**：模仿人眼"大视野感知、小区域聚焦"，提出 LS(Large-Small)卷积：大核感知聚合上下文 + 小核聚焦动态加权，构建轻量骨干家族。
- **适用性**：2025 年最新轻量骨干，"大感受野找远处小目标 + 小核聚焦细节"与本课题远小目标痛点高度契合，新颖性强。
- **建议接入点**：LS Conv 替换骨干下采样后的主干卷积，或嵌入 C3k2 构成 C3k2-LS。

---

## B. 注意力机制（9 篇）

### 7. CBAM
- **标题**：CBAM: Convolutional Block Attention Module
- **第一作者**：Sanghyun Woo｜**年份**：2018｜**venue**：ECCV 2018
- **ID**：arXiv:1807.06521
- **核心思想**：串联通道注意力+空间注意力的即插即用模块，先"看什么"再"看哪里"。
- **适用性**：经典 baseline 注意力，论文中主要作消融对照，不宜作为创新点主体。
- **建议接入点**：neck 各尺度输出后插入，作对照组。

### 8. Coordinate Attention (CA)
- **标题**：Coordinate Attention for Efficient Mobile Network Design
- **第一作者**：Qibin Hou｜**年份**：2021｜**venue**：CVPR 2021
- **ID**：arXiv:2103.02907
- **核心思想**：把通道注意力分解为沿 H、W 两个方向的 1D 池化编码，在保留精确位置信息的同时捕获长程依赖，专为移动网络设计。
- **适用性**：位置敏感，对密集串番茄/柑橘的果实定位友好；农业检测论文高频组件（同为对照候选）。
- **建议接入点**：骨干末端或 neck 融合节点后。

### 9. SimAM（参数无关）
- **标题**：SimAM: A Simple, Parameter-Free Attention Module for Convolutional Neural Networks
- **第一作者**：Lingxiao Yang｜**年份**：2021｜**venue**：ICML 2021, PMLR vol.139, pp.11863–11874
- **ID**：⚠️ 无 DOI、无 arXiv（PMLR 不注册 DOI）。已经 DBLP 核验：`dblp.org/rec/conf/icml/YangZLX21`；官方页 `proceedings.mlr.press/v139/yang21o.html`。引用时用 PMLR 出处。
- **核心思想**：基于神经科学能量函数推导 3D 注意力权重，闭式解、零附加参数。
- **适用性**：零参数特性与 2.8M 预算完美契合——加注意力不加参，是 nano 模型最"划算"的选项。
- **建议接入点**：C3k2 Bottleneck 内部逐块插入（零参可全网插）。

### 10. EMA (Efficient Multi-scale Attention)
- **标题**：Efficient Multi-Scale Attention Module with Cross-Spatial Learning
- **第一作者**：Daliang Ouyang｜**年份**：2023｜**venue**：ICASSP 2023
- **ID**：arXiv:2305.13563；DOI: 10.1109/ICASSP49357.2023.10096516
- **核心思想**：通道分组重塑到 batch 维，1×1 与 3×3 双分支跨空间学习，多尺度聚合且不做通道降维。
- **适用性**：2023–2025 农业 YOLO 改进论文的高频注意力，多尺度特性利于远近果实尺度差异大的场景。
- **建议接入点**：neck 的 P3 小目标分支上插入，或与 C3k2 组合成 C3k2-EMA。

### 11. LSKA (Large Separable Kernel Attention)
- **标题**：Large Separable Kernel Attention: Rethinking the Large Kernel Attention Design in CNN
- **第一作者**：Kin Wai Lau｜**年份**：2024（arXiv 2023.09）｜**venue**：Expert Systems with Applications, vol.236
- **ID**：arXiv:2309.01439；DOI: 10.1016/j.eswa.2023.121352
- **核心思想**：把 VAN 的大核注意力分解为水平/垂直 1D 可分离深度卷积级联，大感受野的参数与计算量从平方降为线性。
- **适用性**：大感受野对远处模糊小果的上下文补偿直接有效，且降参符合轻量约束。
- **建议接入点**：**SPPF → SPPF-LSKA**（池化串联后接 LSKA，即 YOLOv8-AM 类做法），这是最顺手的 SPPF 改进点。

### 12. MLCA (Mixed Local-Channel Attention)
- **标题**：Mixed local channel attention for object detection
- **第一作者**：Dahang Wan｜**年份**：2023｜**venue**：Engineering Applications of Artificial Intelligence, vol.123
- **ID**：DOI: 10.1016/j.engappai.2023.106442（无 arXiv）
- **核心思想**：混合局部+全局、通道+空间四种信息的轻量注意力，先局部池化再双路融合，参数极少。
- **适用性**：专为检测任务设计、以 YOLO 系为实验载体，审稿人接受度高；参数开销近似可忽略。
- **建议接入点**：neck 各 concat 之后（P3/P4/P5 三处），或 C3k2 出口。

### 13. ELA (Efficient Local Attention)
- **标题**：ELA: Efficient Local Attention for Deep Convolutional Neural Networks
- **第一作者**：Wei Xu｜**年份**：2024｜**venue**：arXiv preprint（未见正式会议/期刊版，引用时标注 preprint）
- **ID**：arXiv:2403.01123
- **核心思想**：指出 CA 的 BN+降维损害泛化，改用 1D 条带卷积+GroupNorm 处理 H/W 方向位置信息，更轻更准。
- **适用性**：CA 的直接改进版，2024 后农业检测改进论文常见；preprint 身份作引用时需注意。
- **建议接入点**：与 CA/EMA 同位消融（骨干末端或 neck 节点）。

### 14. iRMB (EMO)
- **标题**：Rethinking Mobile Block for Efficient Attention-based Models
- **第一作者**：Jiangning Zhang｜**年份**：2023｜**venue**：ICCV 2023
- **ID**：arXiv:2301.01146
- **核心思想**：统一 MobileNetv2 倒残差与 Transformer 注意力，提出倒残差移动块 iRMB（DW 卷积+窗口 EW-MHSA 一体化），构建 1M–5M 级 EMO 骨干。
- **适用性**：1–5M 参数带正好覆盖 2.8M 预算；iRMB 兼顾局部细节与全局依赖，适合模糊小目标。
- **建议接入点**：C3k2 → C3k2-iRMB（深层 P4/P5，窗口注意力代价随分辨率降低）。

### 15. MSCA (SegNeXt)
- **标题**：SegNeXt: Rethinking Convolutional Attention Design for Semantic Segmentation
- **第一作者**：Meng-Hao Guo｜**年份**：2022｜**venue**：NeurIPS 2022
- **ID**：arXiv:2209.08575
- **核心思想**：提出多尺度卷积注意力 MSCA：深度条带卷积（7/11/21 多分支）聚合多尺度上下文再作注意力权重，卷积注意力在分割上胜过自注意力。
- **适用性**：源自分割任务的注意力，与 seg 头任务同源，条带大核对枝叶遮挡下的长条上下文有效。
- **建议接入点**：分割头 protos 分支前，或 neck P3 层用 MSCA 增强掩码特征。

---

## C. SPPF 改进（1 篇 + 关联）

### 16. FocalNets (Focal Modulation)
- **标题**：Focal Modulation Networks
- **第一作者**：Jianwei Yang｜**年份**：2022｜**venue**：NeurIPS 2022
- **ID**：arXiv:2203.11926
- **核心思想**：用焦点调制替换自注意力：分层门控聚合多级上下文 + 逐元素调制 query，无注意力也能全局建模。
- **适用性**：分层上下文聚合恰好是"暗、糊、小"目标需要的多粒度环境证据；可讲成 SPPF 的现代替代。
- **建议接入点**：**SPPF → Focal Modulation 模块**（多尺度池化换成分层焦点聚合）；与 #11 SPPF-LSKA 二选一消融。
- 关联：SPPF+LSKA 方案的出处即 #11 LSKA 论文（其 YOLO 应用变体见各 YOLOv8 改进文，机理引 LSKA 原文即可）。

---

## D. 优化器（3 篇）

### 17. AdamW
- **标题**：Decoupled Weight Decay Regularization
- **第一作者**：Ilya Loshchilov｜**年份**：2019（arXiv 2017.11）｜**venue**：ICLR 2019
- **ID**：arXiv:1711.05101
- **核心思想**：指出 Adam 中 L2 正则≠权重衰减，提出解耦权重衰减的 AdamW，恢复泛化能力。
- **适用性**：训练小节引用的标准出处；Ultralytics `optimizer=AdamW` 一行即用。
- **建议接入点**：训练配置节引用；与 SGD 作训练策略消融。

### 18. Lion
- **标题**：Symbolic Discovery of Optimization Algorithms
- **第一作者**：Xiangning Chen｜**年份**：2023｜**venue**：NeurIPS 2023
- **ID**：arXiv:2302.06675
- **核心思想**：程序搜索自动发现的优化器 Lion：只跟踪动量、用 sign 更新，比 AdamW 省内存且多任务上精度相当或更优。
- **适用性**：小 batch 农业训练场景可作 AdamW 对照；sign 更新对 nano 模型正则效应明显。
- **建议接入点**：优化器消融表（SGD/AdamW/Lion 三方对比）。

### 19. Muon
- **标题**：Muon is Scalable for LLM Training
- **第一作者**：Jingyuan Liu（Moonshot AI/Kimi 团队）｜**年份**：2025｜**venue**：arXiv 技术报告
- **ID**：arXiv:2502.16982
- **核心思想**：对矩阵参数用 Newton–Schulz 正交化动量更新（Muon，原为 K. Jordan 2024 博客提出、无正式论文），该报告加入权重衰减与更新尺度校准，证明其大规模可扩展、约 2 倍算力效率。
- **适用性**：⚠️ 原始 Muon（Jordan 2024）只有博客/GitHub，无 DOI/arXiv，正式引用建议用本报告；在 CV 检测器上属未充分验证的新优化器，只宜作探索性小消融，不宜作论文主创新。
- **建议接入点**：可选消融：骨干卷积核用 Muon、头部用 AdamW 的混合方案；结论谨慎表述。

---

## E. 训练策略（4 篇）

### 20. CWD（通道级知识蒸馏）
- **标题**：Channel-wise Knowledge Distillation for Dense Prediction
- **第一作者**：Changyong Shu｜**年份**：2021（arXiv 2020.11）｜**venue**：ICCV 2021
- **ID**：arXiv:2011.13256
- **核心思想**：对每通道激活做 softmax 归一化后最小化师生 KL 散度，让学生聚焦每通道最显著区域，专为密集预测设计。
- **适用性**：可用 YOLO11s/m-seg 当教师蒸馏 11n-seg，不增推理参数即提精度——轻量化论文的"免费午餐"，与分割任务契合。
- **建议接入点**：neck 输出特征图（P3/P4/P5）上加 CWD 损失，教师取 YOLO11m-seg。

### 21. MGD（掩码生成蒸馏）
- **标题**：Masked Generative Distillation
- **第一作者**：Zhendong Yang｜**年份**：2022｜**venue**：ECCV 2022
- **ID**：arXiv:2205.01529
- **核心思想**：随机遮蔽学生特征、迫使其通过生成模块恢复教师完整特征，把蒸馏从"模仿"变为"生成"任务，对检测/分割通用。
- **适用性**：掩码-恢复机制天然模拟"遮挡下补全果实特征"，与枝叶遮挡痛点可讲成一致动机；与 CWD 二选一。
- **建议接入点**：同 #20 位置；消融 CWD vs MGD。

### 22. YOLOv4（mosaic 出处）
- **标题**：YOLOv4: Optimal Speed and Accuracy of Object Detection
- **第一作者**：Alexey Bochkovskiy｜**年份**：2020｜**venue**：arXiv preprint（无正式会议版）
- **ID**：arXiv:2004.10934
- **核心思想**：系统整合 bag-of-freebies/specials，**首次提出 Mosaic 数据增强**（4 图拼接混合上下文、变相增大 batch）。
- **适用性**：论文写 mosaic 增强时的规范引用出处；4 图拼接还隐式生成更多小目标样本，利于远处小果。
- **建议接入点**：训练策略小节引用（Ultralytics 默认 mosaic=1.0 的来源依据）。

### 23. YOLOX（close_mosaic 出处）
- **标题**：YOLOX: Exceeding YOLO Series in 2021
- **第一作者**：Zheng Ge｜**年份**：2021｜**venue**：arXiv 技术报告（无正式会议版）
- **ID**：arXiv:2107.08430
- **核心思想**：anchor-free+解耦头+SimOTA；**首创"最后 15 epoch 关闭 mosaic/mixup"策略**，让模型末期在真实分布上收敛。
- **适用性**：Ultralytics `close_mosaic=10` 参数的思想出处；对小而糊的幼果，末期关增强能修复 mosaic 造成的尺度失真。
- **建议接入点**：训练配置节引用；可消融 close_mosaic=0/10/20。

---

## 核验状态总表

| # | 论文 | ID | 核验渠道 | 状态 |
|---|------|----|----------|------|
| 1 | StarNet | arXiv:2403.19967 | arXiv API | ✅ |
| 2 | FasterNet | arXiv:2303.03667 | arXiv API | ✅ |
| 3 | RepViT | arXiv:2307.09283 | arXiv API | ✅ |
| 4 | EfficientViT | arXiv:2305.07027 | arXiv API | ✅ |
| 5 | GhostNetV2 | arXiv:2211.12905 | arXiv API | ✅ |
| 6 | LSNet | arXiv:2503.23135 | arXiv API | ✅ |
| 7 | CBAM | arXiv:1807.06521 | arXiv API | ✅ |
| 8 | CA | arXiv:2103.02907 | arXiv API | ✅ |
| 9 | SimAM | PMLR v139（无DOI/arXiv） | DBLP | ✅（特殊标注） |
| 10 | EMA | arXiv:2305.13563 + IEEE DOI | arXiv API | ✅ |
| 11 | LSKA | arXiv:2309.01439 + 10.1016/j.eswa.2023.121352 | arXiv+Crossref | ✅ |
| 12 | MLCA | 10.1016/j.engappai.2023.106442 | Crossref | ✅ |
| 13 | ELA | arXiv:2403.01123（preprint） | arXiv API | ✅ |
| 14 | iRMB/EMO | arXiv:2301.01146 | arXiv API | ✅ |
| 15 | MSCA/SegNeXt | arXiv:2209.08575 | arXiv API | ✅ |
| 16 | FocalNets | arXiv:2203.11926 | arXiv API | ✅ |
| 17 | AdamW | arXiv:1711.05101 | arXiv API | ✅ |
| 18 | Lion | arXiv:2302.06675 | arXiv API | ✅ |
| 19 | Muon | arXiv:2502.16982（技术报告） | arXiv API | ✅（特殊标注） |
| 20 | CWD | arXiv:2011.13256 | arXiv API | ✅ |
| 21 | MGD | arXiv:2205.01529 | arXiv API | ✅ |
| 22 | YOLOv4 | arXiv:2004.10934 | arXiv API | ✅ |
| 23 | YOLOX | arXiv:2107.08430 | arXiv API | ✅ |

**组合建议（面向 2.8M 预算与"远小暗糊"痛点）**：骨干选 1 个（LSNet/StarNet 新颖性最高）+ 注意力选 1–2 个（SimAM 零参 + EMA/LSKA 之一）+ SPPF-LSKA 或 FocalModulation 二选一 + CWD 蒸馏 + AdamW/close_mosaic 训练配置，总增参可控制在 ±0.3M 内。
