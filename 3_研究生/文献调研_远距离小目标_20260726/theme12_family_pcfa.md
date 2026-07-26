# Theme 12：模型家族设计方法论 + PCFA 新颖性核查

调研日期：2026-07-26。全部条目经 arXiv API（export.arxiv.org）逐条核验，arXiv ID / 标题 / 摘要均为真实返回值，无编造。检索工具：arXiv API（id_list 精确核验 + search_query 全文检索）、Semantic Scholar API 交叉检索。

---

## 任务 A：模型变体家族 / 缩放策略设计方法论（8 篇，全部核验）

### A1. EfficientNet — 复合缩放（compound scaling）
- **arXiv:1905.11946** (v5)，Tan & Le, "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"，ICML 2019。
- 家族机制：单一复合系数 φ 按固定比例 (α·β²·γ²≈2) **等比联动缩放深度/宽度/分辨率**，从 B0 基线生成 B0–B7 家族。
- 可引用原则：**"缩放三维必须联动平衡而非单独放大"**——家族成员共享基础架构，只在预算轴上移动。
- 局限（对我们有利）：家族只有一根"资源预算"轴，所有成员面向同一任务同一场景。

### A2. EfficientNetV2 — 训练感知 NAS + 缩放联合
- **arXiv:2104.00298** (v3)，Tan & Le, "EfficientNetV2: Smaller Models and Faster Training"，ICML 2021。
- 家族机制：training-aware NAS 与 scaling 联合优化产生 S/M/L 家族；提出 progressive learning（训练中渐增分辨率+自适应正则）。
- 可引用原则：**家族设计的优化目标本身可以换轴**（V1 优化 FLOPs-精度 → V2 优化训练速度-参数效率），证明"同一家族方法论、不同优化轴"是被顶会接受的迭代范式——这正是 SXQNet V1→V10 按轴分化的方法论依据之一。

### A3. MobileNets — 宽度乘子 + 分辨率乘子
- **arXiv:1704.04861** (v1)，Howard et al., 2017。
- 家族机制：两个全局超参（width multiplier α ∈ {1, 0.75, 0.5, 0.25}，resolution multiplier ρ）让使用者"**按自身应用的约束选取合适尺寸的模型**"（摘要原文意涵）。
- 可引用原则：家族的存在理由是**部署约束多样性**——最早明确把"用户按场景选型"写进设计目标的工作之一。

### A4. YOLOv4（+ YOLOv5/v8 家族惯例的可引用锚点）
- **arXiv:2004.10934** (v1)，Bochkovskiy, Wang & Liao, 2020。
- 家族机制：本文本身是"通用 trick（WRC/CSP/CmBN/Mosaic…）× 特定 trick"的组合验证方法论；其 CSP 思想（部分通道跨阶段直通）是后续 YOLO n/s/m/l/x 宽深缩放家族的结构基础。
- 注意：**YOLOv5 与 YOLOv8 无同行评审论文**（Ultralytics 软件仓库，YOLOv5 可用 Zenodo DOI: 10.5281/zenodo.3908559 作软件引用）；论文中引 n/s/m/l/x 家族惯例时，规范做法是引 YOLOv4 (2004.10934) + RTMDet (2212.07784) 作为文献锚点，Ultralytics 作软件脚注。

### A5. RTMDet — 实时检测器家族的经验性设计研究
- **arXiv:2212.07784** (v2)，Lyu et al. (OpenMMLab), 2022。
- 家族机制：tiny/small/medium/large/extra-large 五档宽深缩放；**同一基本块延伸到多任务**（检测 / 实例分割 / 旋转框检测），摘要明言 "easily extensible for many object recognition tasks"。
- 可引用原则：这是检测领域**除速度-精度轴外最接近"任务轴家族"的先例**——但其任务变体只换头部（head），主干家族仍是单一宽深轴，不是按场景特性（暗光/纹理/小目标）分化的家族。

### A6. RegNet — 设计"网络群体"而非单个网络
- **arXiv:2003.13678** (v1)，Radosavovic et al., "Designing Network Design Spaces"，CVPR 2020。
- 家族机制：把设计对象从"网络实例"提升到"**参数化的网络群体（design space）**"，通过统计（error EDF）逐步收缩设计空间，得到量化线性宽度规则的 RegNet 族。
- 可引用原则：**"设计设计空间"是家族式方法论最强的顶会背书**——SXQNet 家族可表述为"在一个共享设计空间内沿多条场景轴采样的实例群"，直接对齐 RegNet 的范式论述。

### A7. Once-for-All (OFA) — 一次训练、按部署场景特化
- **arXiv:1908.09791** (v5)，Cai et al., ICLR 2020。
- 家族机制：训练一个支持多种深度/宽度/核大小/分辨率的超网，之后**为每个部署场景（不同硬件、不同延迟约束）免再训练地抽取特化子网**；progressive shrinking 训练。
- 可引用原则：**这是"按场景特化变体"最正式的先例**——但其"场景"= 硬件/延迟预算，仍是资源轴；不涉及按视觉条件（光照/纹理/目标尺度）特化。是回答关键问题①的最佳引用。

### A8. Slimmable Networks — 运行时可切换宽度（补充）
- **arXiv:1812.08928** (v1)，Yu et al., ICLR 2019。
- 家族机制：单网络多宽度可切换（switchable BN），运行时按设备资源即时调档。
- 可引用原则：家族可以"共享权重"而非独立训练——反衬 SXQNet 独立变体设计的另一端。（同方向另有 BigNAS，arXiv:2003.11142，单阶段超网直接切片部署，已核验，可作脚注。）

### 关键问题①的结论：有没有"按任务/场景轴设计变体家族"的先例？
- **严格意义上：没有直接先例。** 已核验的全部家族工作的分化轴只有三类：(a) 资源/预算轴（EfficientNet、MobileNet、YOLO 系、RTMDet、RegNet 的宽深分辨率缩放）；(b) 部署硬件轴（OFA、Slimmable、BigNAS——本质仍是资源轴的部署侧表述）；(c) 优化目标轴（EfficientNetV2 换成训练效率）。RTMDet 的多任务扩展只换头不换主干。
- **未发现任何工作按"成像条件/场景特性"（暗光、纹理退化、小目标密度、频域特性）在同一设计空间内分化出专用变体家族**——arXiv 检索 "scenario-specialized family / condition-specialized detector family" 均无农业或通用检测先例。
- **最佳引用组合**：RegNet (2003.13678) 立"设计空间→网络群体"的方法论 + OFA (1908.09791) 立"一个空间、按场景特化多变体"的合法性 + MobileNet (1704.04861) 立"用户按约束选型"的动机 + RTMDet (2212.07784) 立检测家族惯例。SXQNet 的贡献表述建议为："将家族分化轴从资源预算轴扩展到**场景-任务轴**（暗光/纹理/小目标/延迟），是对 RegNet-OFA 范式在农业视觉场景下的轴扩展"。

---

## 任务 B：PCFA（Partial-Channel Frequency Attention）新颖性核查（7 篇核验 + 3 组检索证据）

PCFA 机制备忘：仅对 1/4 通道做 FFT→可学习频带调制→iFFT，其余 3/4 通道恒等直通后 concat（FasterNet PConv 的 partial 思想 × 显式频域调制）。

### B1. Octave Convolution（最接近、必须重点区分）
- **arXiv:1904.05049** (v3)，Chen et al., "Drop an Octave"，ICCV 2019。
- 机制：把特征图**按频率分解为高/低两组通道**，低频组以低一倍空间分辨率存储和计算，组间有信息交换；目的是**降低空间冗余、省显存省算力**。
- **与 PCFA 的三点机制差异**（核心）：
  1. **"频率"的定义不同**：OctConv 的"低频"= 空间下采样的低分辨率特征（隐式、由分辨率定义），**全程没有任何 FFT/DCT 谱变换**；PCFA 在显式 Fourier 谱域内做可学习频带调制。
  2. **通道覆盖不同**：OctConv 全部通道都被卷积处理（分成两组各自处理+交叉）；PCFA 只处理 1/4 通道，其余 3/4 **恒等直通零计算**。
  3. **设计目的相反**：OctConv 的分组是为了**省计算**（低频组降采样）；PCFA 的 partial 是为了**以局部代价换取谱域增强**（表征收益），直通分支是 FasterNet 式的访存效率手段。

### B2. Octave-YOLO（2024，检索中新发现，务必在 related work 中引用并区分）
- **arXiv:2407.19746** (v1)，"Octave-YOLO: Cross frequency detection network with octave convolution"。
- 机制：提出 CFPNet（cross frequency **partial** network），把特征图分为低分辨率-低频与高分辨率分支，用于嵌入式高分辨率实时检测。
- 与 PCFA 区别：它是 OctConv 的分辨率式频率分解进 YOLO，同样**无显式 FFT**；其 "partial" 指分辨率分支划分而非"少数通道进谱域、多数直通"。这是文献中**唯一同时出现 partial + frequency 字样的检测工作**，机制上仍与 PCFA 正交，但撞名风险最高，related work 必须点名区分。

### B3. FcaNet — 多谱通道注意力
- **arXiv:2012.11879** (v4)，Qin et al., ICCV 2021。
- 机制：证明 GAP 是 DCT 最低频分量的特例，把通道分组、每组用**不同 DCT 频率基做池化**得到注意力权重。
- 与 PCFA 区别：DCT 基只用于**压缩出标量注意力权重**，随后仍是对全部通道做标量重加权；特征内容本身从未进入谱域被滤波。PCFA 是对特征内容做谱域调制，且仅作用于部分通道。

### B4. FasterNet / PConv — partial 思想的来源
- **arXiv:2303.03667** (v3)，Chen et al., CVPR 2023。
- 机制：PConv 只对 1/4 通道做 3×3 空间卷积、其余直通，动机是降低访存、提高 FLOPS 利用率。
- 与 PCFA 关系：PCFA 显式继承其 partial 通道划分，但把"被处理分支"从空间卷积**替换为 FFT 频带调制**。FasterNet 论文中无任何频域操作——两者的组合在该文无先例。

### B5. GFNet — 全通道全局谱滤波
- **arXiv:2107.00645** (v2)，Rao et al., NeurIPS 2021。
- 机制：2D FFT → 可学习全局滤波器逐元素相乘 → iFFT，替代自注意力；**作用于全部通道**。
- 与 PCFA 区别：谱域调制思想同源，但 GFNet 是全通道、token-mixer 定位；无 partial 直通设计。

### B6. SpectFormer — 层级混合而非通道级混合
- **arXiv:2304.06446** (v2)，Patro et al., 2023。
- 机制：ViT 中**前几层用谱层、后几层用注意力层**的层级（layer-wise）混合。
- 与 PCFA 区别：混合粒度是"层"，不是"同层内部分通道谱域+部分直通"；仍是全通道进谱域。

### B7. WTConv — 小波卷积（谱域大感受野，补充对照）
- **arXiv:2407.05848** (v2)，Finder et al., ECCV 2024。
- 机制：小波分解级联+小核卷积获得对数参数增长的大感受野，多频响应；全通道处理。
- 与 PCFA 区别：无 partial 直通；小波域而非 Fourier 域；目标是感受野而非注意力调制。

### 检索证据（新颖性负面结果，可写进 rebuttal 备用）
1. arXiv 全文检索 `"partial channel" AND "frequency"`（2026-07-26）：前 15 条**全部为无线通信论文**（MIMO/CSI/beamforming），无一为视觉架构。
2. arXiv 检索 `cs.CV AND "partial convolution" AND (fourier OR FFT OR wavelet)`：仅 1 条（M3S-Net, arXiv:2602.19832，光伏功率预测的多模态融合，非 partial-FFT 算子）。
3. arXiv 检索 `cs.CV AND ("channel split" OR "split channels") AND (fourier OR "frequency domain")`：仅 1 条（DuetFace, arXiv:2207.07340，ACM MM 2022——按 DCT 子带把**输入图像**的频率通道分给客户端/服务端做隐私推理，目的与层级均不同）。
4. arXiv 精确短语 `"partial channel frequency"`：**0 条**。"Partial-Channel Frequency Attention" 名称无占用。

### 关键问题②的结论：PCFA 能否声称原创组合？
- **可以，且证据链完整。** 未发现任何工作在同一算子内同时具备：(i) 仅少数通道（如 1/4）进入显式 FFT 谱域做可学习频带调制；(ii) 多数通道恒等直通以保持访存效率。最接近的四类工作各缺一环：OctConv/Octave-YOLO 有"频率×分组"但无谱变换且全通道参与计算；FcaNet 有 DCT 但只产生标量权重不滤内容；GFNet/SpectFormer/WTConv 有谱域内容调制但全通道无 partial；FasterNet 有 partial 但纯空间域。
- **声明措辞建议**："首个将 partial-channel 计算范式（FasterNet）与显式 Fourier 频带调制（GFNet 系）结合的注意力算子"——避免声称"首个频域注意力"或"首个通道分组频域方法"（会被 OctConv/FcaNet/DuetFace 反驳）。
- **必须引用并区分**：1904.05049（OctConv）、2407.19746（Octave-YOLO，撞名风险最高）、2012.11879（FcaNet）、2303.03667（FasterNet）、2107.00645（GFNet）。

---

## 全部核验清单（15 条，均有 arXiv ID）
| # | arXiv ID | 工作 | 用途 |
|---|---|---|---|
| 1 | 1905.11946 | EfficientNet (ICML'19) | A-复合缩放 |
| 2 | 2104.00298 | EfficientNetV2 (ICML'21) | A-换轴迭代 |
| 3 | 1704.04861 | MobileNets | A-按约束选型 |
| 4 | 2004.10934 | YOLOv4 | A-YOLO家族锚点 |
| 5 | 2212.07784 | RTMDet | A-检测家族+多任务 |
| 6 | 2003.13678 | RegNet (CVPR'20) | A-设计空间范式（最佳引用） |
| 7 | 1908.09791 | Once-for-All (ICLR'20) | A-场景特化先例（最佳引用） |
| 8 | 1812.08928 | Slimmable (ICLR'19) | A-共享权重家族 |
| 9 | 2003.11142 | BigNAS | A-脚注 |
| 10 | 1904.05049 | Octave Conv (ICCV'19) | B-最接近工作 |
| 11 | 2407.19746 | Octave-YOLO | B-撞名风险最高 |
| 12 | 2012.11879 | FcaNet (ICCV'21) | B-DCT注意力区分 |
| 13 | 2303.03667 | FasterNet (CVPR'23) | B-partial来源 |
| 14 | 2107.00645 | GFNet (NeurIPS'21) | B-全通道FFT区分 |
| 15 | 2304.06446 | SpectFormer / 2407.05848 WTConv / 2207.07340 DuetFace | B-补充对照 |
