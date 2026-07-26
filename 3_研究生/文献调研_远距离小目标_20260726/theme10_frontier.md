# Theme 10 · 2024–2026 顶会新范式核验与"移植进 YOLO11n-seg (nano, 2.8M)"评估

背景约束：柑橘幼果实例分割（小目标、绿果绿叶低对比、遮挡/退化田间图像）；Ultralytics fork；Windows 训练 + 端侧部署（ONNX 可导出性是硬约束）；参数预算 ~2.8M。
核验方式：arXiv API（id_list 元数据 + comment/journal_ref 字段）、Crossref（DOI）、OpenReview API v2（venue 状态）。全部 ID 已逐条核验，无编造。共 16 个条目 / 17 篇论文。

评级：★★★ 强烈建议移植 | ★★ 值得做但有代价 | ★ 思路可借鉴、不建议整体移植 | ✗ 不适合本课题

---

## A. 检测新范式（DETR 系）

### 1. D-FINE ★★★
- **D-FINE: Redefine Regression Task in DETRs as Fine-grained Distribution Refinement**，Yansong Peng 等，2024/2025，**ICLR 2025**，arXiv:2410.13842
- **范式**：把 bbox 回归重定义为对四边偏移概率分布的逐层"细粒度分布精化"（FDR），而非一次性回归定值；配套 GO-LSD（全局最优定位自蒸馏）把深层精化后的定位分布蒸馏给浅层，推理零开销。
- **移植评估**：YOLO11 的 DFL 头本就是"分布式回归"，与 FDR 同源——可把检测头改造成两级"粗分布→残差精化分布"级联，GO-LSD 则可直接做成 DFL bin 分布的层间/EMA 自蒸馏 loss，**推理端零参数零延迟，完美契合 nano+端侧**。预期对小果定位精度（高 IoU 段 AP）收益明显，是本清单里"代价最低、故事最新"的头部级新范式，强烈建议作为核心创新点之一。

### 2. DEIM ★★
- **DEIM: DETR with Improved Matching for Fast Convergence**，Shihua Huang 等，2024/2025，**CVPR 2025**，arXiv:2412.04234
- **范式**：Dense O2O——用拼接式增广人为增加一对一匹配的正样本密度，加速 DETR 收敛；MAL（Matchability-Aware Loss）按匹配质量调制分类损失。
- **移植评估**：Dense O2O 针对 DETR 一对一匹配稀疏问题，YOLO 本身就是一对多分配，**该主创新对 YOLO 无意义**；只有 MAL 可作为 VFL/QFL 的替代 loss 即插即用，属于增量改进而非范式移植。只建议顺手做 loss 消融，不足以撑论文创新点。

### 3. RT-DETRv2 / RT-DETRv3 ★★
- **RT-DETRv2: Improved Baseline with Bag-of-Freebies**，Wenyu Lv 等，2024，arXiv preprint，arXiv:2407.17140；**RT-DETRv3: Real-time End-to-End Object Detection with Hierarchical Dense Positive Supervision**，Shuo Wang 等，2024/2025，**WACV 2025**，arXiv:2409.08475
- **范式**：v2 是部署友好化（离散采样替代 grid_sample 等 bag-of-freebies）；v3 核心是"分层稠密正监督"——训练期外挂 CNN 一对多辅助头 + 多组自注意力解码器分支提供稠密监督，推理时全部摘除。
- **移植评估**：v3 的思想可反向移植：给 YOLO11n-seg 的 neck 训练期外挂一个轻量 DETR 式辅助解码头做稠密监督、推理摘除，**推理零开销**，但工程量中等且在 nano 容量下收益不确定（辅助监督对小模型可能过强）。可作为训练范式实验项，不宜当主创新。

### 4. DEYO ✗
- **DEYO: DETR with YOLO for End-to-End Object Detection**，Haodong Ouyang，2024，arXiv preprint，arXiv:2402.16370
- **范式**：两阶段训练——先用 YOLO 一对多监督预训练 backbone+neck，再接 DETR 解码头实现无 NMS 端到端。
- **移植评估**：方向是"YOLO→DETR"，与本课题"保持 nano YOLO 形态"相反；解码器带来的延迟与导出复杂度对端侧分割不可接受。结论：不移植，仅在综述里作为训练范式引用。

---

## B. 超图 / 高阶关联范式

### 5. Hyper-YOLO ★★★
- **Hyper-YOLO: When Visual Object Detection Meets Hypergraph Computation**，Yifan Feng 等，2024/2025，**IEEE TPAMI 2025**（DOI: 10.1109/TPAMI.2024.3524377），arXiv:2408.04804
- **范式**：首次把超图计算引入 YOLO——neck 中构建跨层级、跨位置的超图（HyperC2Net），用超图卷积建模特征点之间的高阶（非成对）语义关联，突破常规 FPN 的成对/局部聚合。
- **移植评估**：官方即基于 YOLOv8 且有 N 尺度版本，迁到 YOLO11n-seg 属于同框架平移，**可行性最高的一档**；距离阈值建超边在 640 输入下有一定算力开销，nano 上需控制通道数。对"成串/成簇小果 + 绿叶背景"这类需要跨区域关联的场景针对性强，TPAMI 背书 + 农业无人用过，性价比极高。

### 6. YOLOv13 (HyperACE) ★★★
- **YOLOv13: Real-Time Object Detection with Hypergraph-Enhanced Adaptive Visual Perception**，Mengqi Lei 等，2025，arXiv preprint，arXiv:2506.17733
- **范式**：HyperACE 把 Hyper-YOLO 的固定阈值超边升级为**可学习自适应超边参与度**，并用线性复杂度消息传递（C3AH）；FullPAD 隧道把高阶关联增强特征回灌 backbone-neck-head 全链路。
- **移植评估**：代码即 Ultralytics 体系、自带 13n nano 尺度、线性复杂度，**是全清单中"新范式×可移植×端侧"三者交集最大的一篇**；移植进 YOLO11n-seg 主要是把 HyperACE 块插入 neck + 适配 seg 头，工程量小。风险仅在 venue 是 arXiv（但其范式源头 Hyper-YOLO 是 TPAMI 2025，引用链完整），建议与 D-FINE 头改造组合成"高阶关联 neck + 分布精化头"的完整故事。

---

## C. 物理启发 / 新算子

### 7. vHeat ★★
- **vHeat: Building Vision Models upon Heat Conduction**，Zhaozhi Wang 等，2024，arXiv preprint（OpenReview 显示 ICLR 2025 撤回投稿），arXiv:2405.16555
- **范式**：把视觉特征传播建模为物理热传导，用 DCT/IDCT 频域求解热传导方程（HCO 算子），O(N^1.5) 拿到全局感受野，热扩散率可学习。
- **移植评估**：HCO 本质是"DCT→逐元素调制→IDCT"，可实现为矩阵乘（DCT 矩阵固定），**ONNX 导出可行但需手写算子替换**；作为 neck 深层的全局上下文块替换一个 C3k2，参数近零、FLOPs 低，对分散小目标的全局感受野有帮助。物理可解释性在农业期刊是好故事；venue 弱（撤稿）是引用短板，建议作次创新点而非主打。

### 8. QuadMamba ★
- **QuadMamba: Learning Quadtree-based Selective Scan for Visual State Space Model**，Fei Xie 等，2024，**NeurIPS 2024**，arXiv:2410.06806
- **范式**：四叉树自适应划分图像，对信息密集区域细粒度扫描、背景区域粗粒度扫描，解决视觉 Mamba 的 1D 扫描破坏 2D 局部性问题。
- **移植评估**："对目标密集区自适应加密计算"的思想很契合果簇场景，但 selective scan 依赖定制 CUDA kernel，**Windows 训练已痛苦、ONNX/端侧导出基本不可行**。结论：范式可引用、机制可启发（如四叉树式特征选择），整体移植不建议。

### 9. Mamba-YOLO ★
- **Mamba YOLO: A Simple Baseline for Object Detection with State Space Model**，Zeyu Wang 等，2024/2025，**AAAI 2025**（OpenReview 记录确认），arXiv:2406.05835
- **范式**：ODMamba backbone + RG Block，首个把 SSM 线性复杂度全局建模系统接进 YOLO 框架的基线。
- **移植评估**：与 QuadMamba 同样卡在部署：selective scan 无原生 ONNX 路径，nano 尺度下相对卷积的收益也未证明。作为"为何不用 Mamba"的对比论据价值大于移植价值。

### 10. TTT layers ✗
- **Learning to (Learn at Test Time): RNNs with Expressive Hidden States**，Yu Sun 等，2024，arXiv preprint（OpenReview 显示投 ICLR 2025），arXiv:2407.04620
- **范式**：把 RNN 隐状态本身做成一个小模型，测试时用自监督损失对其做梯度更新——"隐状态即学习器"，线性复杂度。
- **移植评估**：推理期需要梯度计算，与端侧确定性推理、ONNX 静态图完全冲突；检测/分割上也无成熟落地。结论：不可移植，仅作前沿综述引用。

### 11. Vision-LSTM (ViL) ★
- **Vision-LSTM: xLSTM as Generic Vision Backbone**，Benedikt Alkin 等，2024/2025，**ICLR 2025**（arXiv comment 确认），arXiv:2406.04303
- **范式**：mLSTM（矩阵记忆、可并行）块交替双向扫描 patch 序列，作为通用视觉 backbone。
- **移植评估**：递归矩阵记忆导出困难、nano 尺度无预训练权重、检测/分割迁移证据少。移植代价高收益不明，不建议；线性序列家族里若必须选一个，选 Vision-RWKV。

### 12. Vision-RWKV ★★
- **Vision-RWKV: Efficient and Scalable Visual Perception with RWKV-Like Architectures**，Yuchen Duan 等，2024/2025，**ICLR 2025 Spotlight**（OpenReview 确认），arXiv:2403.02308
- **范式**：RWKV 式线性注意力（Bi-WKV + Q-Shift）做视觉 backbone，全局感受野、线性复杂度，高分辨率稳定，可作 ViT 平替。
- **移植评估**：线性 RNN 家族中最接近可部署的一支（社区已有 ONNX 实践），可尝试在 neck 深层用单个 VRWKV 块替换注意力做全局建模；但在 2.8M 预算下相对 PSA/卷积的增益预计 <1 AP，工程量中等。定位：备选算子，不是首选。

### 13. Convolutional KAN ✗
- **Convolutional Kolmogorov-Arnold Networks**，Alexander D. Bodner 等，2024，arXiv preprint，arXiv:2406.13155
- **范式**：把 KAN 的可学习样条激活装进卷积核——每个核元素是一个可学习非线性函数，而非标量权重。
- **移植评估**：样条计算显存/延迟开销大、无成熟 ONNX 导出路径，且在检测/分割尺度上精度增益至今未被严肃基准证实（原文仅 MNIST 级验证）。尽管 2024-2025 农业改进论文扎堆用 KAN 卷积，**审稿风险与部署风险双高，明确不建议**。

---

## D. 轻量新算子 / 重参数化 / 训练范式

### 14. RepNeXt ★
- **RepNeXt: A Fast Multi-Scale CNN using Structural Reparameterization**，Mingshu Zhao 等，2024，arXiv preprint（tech report），arXiv:2406.16004
- **范式**：串行+并行多尺度分支训练期建模、推理期全部重参数化折叠进单一卷积，多尺度感受野零推理成本。
- **移植评估**：对 C3k2/DWConv 做多分支重参数化改造是完全成熟的路线，**移植代价最低、部署零开销**；但学术新颖性低（RepVGG 谱系延伸），只能当"白送的涨点技巧"写在消融里，不能当创新点。

### 15. StarNet (Rewrite the Stars) ★★
- **Rewrite the Stars**，Xu Ma 等，2024，**CVPR 2024**（arXiv comment 确认），arXiv:2403.19967
- **范式**：证明逐元素乘法（star operation）等价于把特征隐式映射到高维非线性空间，无需加宽网络即可获得高维表征——为极小模型提供了新的算子级设计原理。
- **移植评估**：StarBlock 替换 C3k2 是半天工作量，纯标准算子、ONNX 完全安全，nano 尺度收益有实证。缺点：2024 论文且已开始被农业改进文使用，新颖性窗口正在关闭；适合作为 backbone 侧的辅助改进点。

### 16. LSNet ★★★ / OverLoCK ★★
- **LSNet: See Large, Focus Small**，Ao Wang 等（YOLOv10 团队/清华）2025，**CVPR 2025**（arXiv comment 确认 camera-ready），arXiv:2503.23135；**OverLoCK: An Overview-first-Look-Closely-next ConvNet with Context-Mixing Dynamic Kernels**，Meng Lou & Yizhou Yu，2025，**CVPR 2025 Oral**（arXiv comment 确认），arXiv:2502.20087
- **范式**：LSNet 提出 LS 卷积——仿人眼"看大（大核感知上下文）聚小（小核动态聚合）"的异尺度感知-聚合算子，专为轻量网络设计；OverLoCK 提出自顶向下注意的 ContMix 动态卷积——先概览全局再由上下文调制细看局部。
- **移植评估**：LSNet 出自 YOLOv10 同组、天然面向轻量检测，LS 卷积用标准算子组合、可导出，替换 YOLO11n-seg backbone 浅层的 C3k2 属中低工程量，**是"新算子"路线里部署最稳、血统最正的选择**，对绿叶背景中低对比小果的"上下文引导局部聚焦"有直接对应性。OverLoCK 的 ContMix 思想更强但动态核参数开销对 2.8M 预算偏重，建议只借鉴其"深层语义引导浅层"的连接方式（与 GO-LSD/FullPAD 可互证）。

---

## 结论（按移植优先级）

1. **HyperACE/Hyper-YOLO 超图高阶关联 neck**（TPAMI 2025 + arXiv 2506.17733）：Ultralytics 原生、有 nano 尺度、线性复杂度，对果簇高阶关联场景针对性最强——主创新点首选。
2. **D-FINE 的 FDR/GO-LSD 分布精化+定位自蒸馏头**（ICLR 2025）：与 YOLO 的 DFL 同源，GO-LSD 推理零开销，直击小目标高 IoU 定位——头部/损失侧创新首选，与 1 组合成完整故事。
3. **LSNet LS 卷积**（CVPR 2025）：轻量专用新算子、标准算子可导出、YOLOv10 团队血统——backbone 侧替换 C3k2 的最稳选择；vHeat HCO 作为"物理可解释"的差异化备选。
4. 明确排除：Mamba 系（QuadMamba/Mamba-YOLO，端侧导出不可行）、TTT（推理需梯度）、Conv-KAN（部署与实证双弱）、DEYO（与保持 YOLO 形态矛盾）。
