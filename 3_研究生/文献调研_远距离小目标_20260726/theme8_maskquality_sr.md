# 主题8：小目标掩码质量与超分辨率辅助检测（15篇，均经 arXiv API / Crossref 核验）

课题背景：YOLO11n-seg 柑橘幼果实例分割；远处果实 <16-32px；prototype mask 由 P3(/8) 生成，160x160 proto 对极小目标掩码质量差；3072x3072 原图压缩至 640 训练，输入端即丢失 ~4.8x 线性分辨率。

---

## A. 掩码质量 / 边界细化（5篇）

### 1. Mask Scoring R-CNN
- 第一作者：Zhaojin Huang | 2019 | CVPR 2019 | arXiv:1903.00241
- 核心机制：指出分类置信度与掩码质量（mask IoU）不一致，增加 MaskIoU head 直接回归预测掩码与 GT 的 IoU，用其校准掩码打分。推理时按掩码质量而非分类分数排序，提升 AP。
- 适用性：640 下小果的分类分数常高而掩码烂，加一个轻量 mask-IoU 分支可作为"掩码质量感知 NMS/打分"的低成本创新点，但不直接提高掩码分辨率。

### 2. PointRend: Image Segmentation as Rendering
- 第一作者：Alexander Kirillov | 2020 | CVPR 2020 | arXiv:1912.08193
- 核心机制：把分割视为渲染问题，在不确定度最高的点上自适应细分采样，用点级 MLP 结合细粒度（高分辨率）特征与粗掩码特征迭代上采样，仅在边界附近做高分辨率计算。
- 适用性：思想可移植——对 YOLO 的粗 proto 掩码在边界不确定点用 P2/更高分辨率特征做点级细化，计算只花在边界上，适合 n 级轻量模型；但 <16px 果实"整个都是边界"，收益需实验验证。

### 3. RefineMask
- 第一作者：Gang Zhang | 2021 | CVPR 2021 | arXiv:2104.08569
- 核心机制：多阶段逐级放大掩码（14→28→56→112），每阶段融合实例特征与细粒度语义特征，并用 boundary-aware refinement（SFM 模块）逐步补回丢失的高频细节。
- 适用性：证明"逐级上采样+每级注入高分辨率特征"比一次性 /8 proto 上采样好得多，是改 YOLO seg head 输出分辨率路线的直接参照。

### 4. BPR: Look Closer to Segment Better（Boundary Patch Refinement）
- 第一作者：Chufeng Tang | 2021 | CVPR 2021 | arXiv:2104.05239
- 核心机制：模型无关的后处理框架：沿粗掩码边界裁出小 patch，从原始高分辨率图像取对应区域，用一个小型二值分割网络在 patch 尺度重新细化边界，再拼回。
- 适用性：天然利用 3072 原图信息（细化时从原图取 patch，绕过 640 压缩），可作为不改训练协议的推理端掩码增强，代价是额外一个细化网络的推理时间。

### 5. Mask Transfiner
- 第一作者：Lei Ke | 2022 | CVPR 2022 | arXiv:2111.13673
- 核心机制：将掩码误差定位到四叉树上的稀疏"易错点"（主要在边界/高频区），仅对这些稀疏点用 Transformer 序列建模并重预测标签，实现高分辨率掩码而计算量低。
- 适用性：稀疏点细化的思路与 QueryDet 的稀疏查询同构，适合"只在小果区域花高分辨率算力"的总体设计；直接嫁接到 YOLO proto 体系需要较大改动。

## B. Prototype 类分割范式与 proto 分辨率（3篇）

### 6. YOLACT: Real-time Instance Segmentation
- 第一作者：Daniel Bolya | 2019 | ICCV 2019 | arXiv:1904.02689
- 核心机制：首创 prototype + 系数线性组合范式（YOLO-seg 的直接源头）：protonet 从 P3 生成 k 个全图 prototype，每实例预测组合系数，crop+threshold 得掩码。论文明确指出 proto 从最深层生成会漏小物体，故选 P3，且"更大的 prototype 分辨率对小目标质量至关重要"。
- 适用性：这是 YOLO11n-seg 掩码机制的理论出处——其消融直接支撑"把 protonet 输入换成 P2 或对 proto 上采样"这一改进的合法性，必引。

### 7. YOLACT++
- 第一作者：Daniel Bolya | 2020 | IEEE TPAMI（2022 卷出版）| arXiv:1912.06218 | DOI: 10.1109/TPAMI.2020.3014297
- 核心机制：在 YOLACT 上加入可变形卷积、优化 anchor 设计，并提出 fast mask re-scoring 分支（用掩码本身预测 mask IoU 来重打分），兼顾速度与掩码质量。
- 适用性：mask re-scoring + 更好采样对小目标掩码的增益分析可直接借鉴；其对失败案例（小物体掩码泄漏/断裂）的讨论正是柑橘远景果的典型症状。

### 8. SparseInst: Sparse Instance Activation
- 第一作者：Tianheng Cheng | 2022 | CVPR 2022 | arXiv:2203.12827
- 核心机制：抛弃 anchor/proto-crop，用一组稀疏 instance activation maps 直接按实例聚合特征并一次性解码掩码，无 NMS，实时。
- 适用性：作为 proto 范式的对照系：说明实例感知的特征聚合可避免 proto 线性组合在小目标上的表达瓶颈，可用于论文中范式对比或借其 IAM 思想改 protonet。

## C. 超分辨率辅助检测（4篇）

### 9. Better to Follow, Follow to Be Better（特征级超分）
- 第一作者：Junhyug Noh | 2019 | ICCV 2019 | DOI: 10.1109/ICCV.2019.00982
- 核心机制：对小目标 RoI 做特征级超分辨率，关键贡献是用"感受野匹配的高分辨率目标特征"作为 SR 监督（用原图更大分辨率前向得到的特征做 target），避免特征 SR 的监督失配。
- 适用性：与本课题完美对口——3072 原图天然可提供"高分辨率目标特征"作监督信号，训练一个 proto/特征超分支路，推理仍 640 输入，零推理分辨率代价。

### 10. Extended Feature Pyramid Network (EFPN)
- 第一作者：Chunfang Deng | 2022 | IEEE Trans. Multimedia | arXiv:2003.07021 | DOI: 10.1109/TMM.2021.3074273
- 核心机制：在 FPN 上额外外推一层超高分辨率金字塔层 P2'，由 feature texture transfer (FTT) 模块对 P3 做特征超分并注入 P2 纹理，专门负责小目标。
- 适用性：给 YOLO11 加"虚拟 P2 层"的现成蓝图：不改输入尺寸、用特征 SR 造出 /4 尺度特征喂 protonet，是固定 640 协议下最贴合的结构改进之一。

### 11. EESRGAN: 端到端边缘增强 GAN 超分 + 检测器
- 第一作者：Jakaria Rabbi | 2020 | Remote Sensing 12(9):1432 | DOI: 10.3390/rs12091432
- 核心机制：边缘增强 ESRGAN 先对低分辨率遥感图做 4x 超分，再接检测器，检测损失反传进 SR 网络实现端到端联合优化，小目标（车辆、油罐）检测大幅提升。
- 适用性：证明"检测损失驱动的 image-level SR"对 <20px 目标有效，但推理需先 SR 再检测，对边缘部署的 n 级模型算力压力大，适合作对比路线而非主方案。

### 12. 农业案例：Laplacian 金字塔深度递归超分用于害虫监测
- 第一作者：Yi Yue | 2018 | Computers and Electronics in Agriculture 150:26-32 | DOI: 10.1016/j.compag.2018.04.004
- 核心机制：提出深度递归 + Laplacian 金字塔超分网络重建田间监控图像中的微小害虫细节，重建后再进行识别/检测，验证 SR 预处理提升农业小目标可辨识度。
- 适用性：农业领域 SR 辅助小目标的先例引用（COMPAG 主刊），为"柑橘园远景小果需要分辨率补偿"提供领域内动机支撑，方法本身已过时。

## D. 高分辨率输入策略（2篇）

### 13. SAHI: Slicing Aided Hyper Inference
- 第一作者：Fatih Cagatay Akyon | 2022 | IEEE ICIP 2022 | arXiv:2202.06934 | DOI: 10.1109/ICIP46576.2022.9897990
- 核心机制：推理时将高分辨率图切成重叠 slice 分别检测再合并（辅以切片微调），使小目标在网络输入中占据更大相对尺度；模型无关、零结构改动。
- 适用性：对 3072 原图是最直接的"零训练改动"上限基线：3072 切成 5x5 个 640 块即近乎无损输入；代价是推理次数 ~25x，论文中应作为精度上界/效率对比锚点。

### 14. QueryDet: Cascaded Sparse Query
- 第一作者：Chenhongyi Yang | 2022 | CVPR 2022 | arXiv:2103.09136
- 核心机制：先在低分辨率特征图粗定位小目标位置（query），再仅在高分辨率特征图（含 P2）的对应稀疏位置上计算检测头，用稀疏卷积把高分辨率检测的代价降到可用。
- 适用性：与本课题"只有远处果实需要高分辨率"高度匹配——同一思想可扩展为"稀疏高分辨率掩码解码"：仅对小框在 P2 尺度解码掩码，是最有论文创新潜力的路线之一。

## E. 分辨率-小目标性能实证（1篇）

### 15. The Effects of Super-Resolution on Object Detection Performance in Satellite Imagery
- 第一作者：Jacob Shermeyer | 2019 | CVPRW 2019 (EarthVision) | arXiv:1812.04098 | DOI: 10.1109/CVPRW.2019.00184
- 核心机制：系统量化 GSD（地面采样距离/等效分辨率）从 30cm 退化到 4.8m 时检测 mAP 的变化，并测试 SR 恢复的补偿效果：分辨率每损失一档小目标 mAP 显著下降，SR 在中低分辨率区间可挽回 13-36% 的性能。
- 适用性：为"3072→640 压缩是主要性能瓶颈"提供可引用的定量实证范式，且其实验设计（分辨率消融曲线）可直接复刻到柑橘数据集作为论文的 motivation 实验。

---

## 核验说明
- 条目 1-8、10、13-15 的 arXiv ID 已通过 arXiv API (export.arxiv.org) 批量返回标题核验一致。
- 条目 9、11、12 及各 DOI 通过 Crossref API 核验（标题、第一作者、年份一致）。
