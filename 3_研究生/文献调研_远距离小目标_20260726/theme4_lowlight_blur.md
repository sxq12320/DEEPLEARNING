# 主题四：暗光/低照度/模糊退化图像下的检测与增强（14 篇核实文献）

> 课题背景：YOLO11n-seg 柑橘幼果实例分割，远处果实**发黑（欠曝/阴影）+ 模糊（失焦/低分辨率）**导致漏检。
> 核实方式：Crossref REST API（DOI+作者逐条核验）+ Semantic Scholar Graph API（arXiv ID 批量核验）+ arXiv API（2 篇补验），检索日期 2026-07-26。所有 DOI/arXiv ID 均来自 API 返回原文，无编造。

---

## A. 低照度增强（4 篇）

### 1. Zero-DCE
- **标题**：Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement
- **第一作者**：Chunle Guo（郭春乐）| **年份**：2020 | **venue**：CVPR 2020
- **DOI**：10.1109/CVPR42600.2020.00185 | **arXiv**：2001.06826
- **核心思想**：把低照度增强建模为逐像素高阶亮度曲线估计，轻量 DCE-Net 以一组无参考损失（曝光、色彩恒常、空间一致、照度平滑）零配对数据训练，迭代曲线映射提亮。
- **适用性**：对"远处局部欠曝"这类非全局暗光同样有效，且极轻量，适合作为采摘机器人端侧预处理。
- **建议接入点**：推理前置增强或离线数据增广（对训练集做亮度曲线扰动，模拟远处欠曝样本）。

### 2. SCI（Self-Calibrated Illumination）
- **标题**：Toward Fast, Flexible, and Robust Low-Light Image Enhancement
- **第一作者**：Long Ma | **年份**：2022 | **venue**：CVPR 2022 (Oral)
- **DOI**：10.1109/CVPR52688.2022.00555 | **arXiv**：2204.10137
- **核心思想**：自校准照明学习框架：训练时级联权重共享的照明估计块并互相校准，推理时只保留单个块，速度极快、无监督。
- **适用性**：毫秒级推理开销，是"增强不拖累 YOLO11n 实时性"的最现实前置选项。
- **建议接入点**：作为可插拔前端与 YOLO11n-seg 串联做消融基线，对比"图像域增强 vs 特征域增强"。

### 3. Retinexformer
- **标题**：Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
- **第一作者**：Yuanhao Cai | **年份**：2023 | **venue**：ICCV 2023
- **DOI**：10.1109/ICCV51070.2023.01149 | **arXiv**：2303.06705
- **核心思想**：单阶段 Retinex 框架 ORF + 照度引导注意力（IG-MSA）Transformer，显式建模提亮过程放大的噪声/伪影并利用照度信息引导长程依赖修复。
- **适用性**：增强质量高但计算较重，更适合离线处理而非端侧实时。
- **建议接入点**：离线增强训练集暗部样本（生成"提亮版"配对数据做一致性训练/蒸馏），不进推理链路。

### 4. HVI 色彩空间 / CIDNet（重点：仓库已有 HVI 模块）
- **标题**：HVI: A New Color Space for Low-light Image Enhancement
- **第一作者**：Qingsen Yan | **年份**：2025 | **venue**:CVPR 2025
- **DOI**：10.1109/CVPR52734.2025.00533 | **arXiv**：2502.20272
- **核心思想**：提出 HVI 色彩空间——极化 HS 色彩图 + 可学习强度压缩，把暗区噪声敏感的亮度与色彩解耦；配套 CIDNet 在 HVI 空间双分支交互注意增强，解决 sRGB/HSV 增强常见的黑区噪声放大与色偏。
- **适用性**：直接针对"发黑区域噪声大、绿色幼果与背景色相接近"的痛点——在 HVI 空间强度与色相解耦后，暗部绿果/绿叶的色彩可分性更好。
- **建议接入点**：仓库已有 HVI 模块——(a) HVI 变换作为 YOLO11n-seg 输入分支（RGB+HVI 双流或替换输入空间）；(b) 在 HVI 强度通道上做暗部增强再逆变换；(c) 把 HVI 空间损失用于增强子网监督。这是最顺手的创新落点。

---

## B. 暗光检测端到端（检测损失驱动增强，4 篇）

### 5. IA-YOLO
- **标题**：Image-Adaptive YOLO for Object Detection in Adverse Weather Conditions
- **第一作者**：Wenyu Liu | **年份**：2022 | **venue**：AAAI 2022
- **DOI**：10.1609/aaai.v36i2.20072 | **arXiv**：2112.08088
- **核心思想**：可微图像处理管线 DIP（去雾/白平衡/gamma/对比度/锐化滤波），小 CNN（CNN-PP）从低分辨率输入预测滤波参数，与 YOLOv3 端到端联训，只用检测损失即可学出"利于检测"的自适应增强。
- **适用性**：范式完全可迁移：远处发黑果实需要的 gamma/对比度参数因图而异，检测损失驱动比固定增强更贴任务。
- **建议接入点**：在 YOLO11n-seg 前接 CNN-PP + 精简 DIP（只留 gamma/对比度/锐化），分割+检测损失联合驱动。

### 6. DENet
- **标题**：DENet: Detection-driven Enhancement Network for Object Detection Under Adverse Weather Conditions
- **第一作者**：Qingpao Qin | **年份**：2022（LNCS 卷 2023 刊出）| **venue**：ACCV 2022（LNCS 13847）
- **DOI**：10.1007/978-3-031-26313-2_30
- **核心思想**：检测驱动的轻量增强网络：拉普拉斯金字塔把图像分解为低频（全局照度/色调，用类 SE 全局调制）与高频（细节，逐层增强），与检测器联训，无需正常光参考图。
- **适用性**："低频管发黑、高频管模糊"的分频思路与柑橘远景退化的两种成因一一对应。
- **建议接入点**：作为前端增强子网与 YOLO11n-seg 联训的结构参考（低频调亮 + 高频补细节双支路）。

### 7. PE-YOLO
- **标题**：PE-YOLO: Pyramid Enhancement Network for Dark Object Detection
- **第一作者**：Xiangchen Yin | **年份**：2023 | **venue**：ICANN 2023（LNCS）
- **DOI**：10.1007/978-3-031-44195-0_14 | **arXiv**：2307.10953
- **核心思想**：金字塔增强网络 PEN：拉普拉斯金字塔逐层配"细节处理模块（上下文+边缘）"与"低频增强模块"，接入 YOLOv3 端到端只用普通检测损失训练，在 ExDark 上验证。
- **适用性**：与 DENet 同范式但更贴 YOLO 工程实现，是"暗果检测"最直接的可复现基线。
- **建议接入点**：复现为对比方法；其边缘增强分支可借鉴到分割掩码边界（幼果轮廓）优化。

### 8. FeatEnHancer
- **标题**：FeatEnHancer: Enhancing Hierarchical Features for Object Detection and Beyond Under Low-Light Vision
- **第一作者**：Khurram Azeem Hashmi | **年份**：2023 | **venue**：ICCV 2023
- **DOI**：10.1109/ICCV51070.2023.00619 | **arXiv**：2308.03594
- **核心思想**：不增强图像而增强"多尺度层级特征"：尺度内注意力加权 + 跨尺度融合，由下游任务损失（检测/分割）直接监督，即插即用，无需成对低光数据。
- **适用性**：对本课题命中率最高——远处暗果是"小目标+弱特征"问题，特征域增强绕开图像域增强引入噪声的风险，且论文本身覆盖实例分割任务。
- **建议接入点**：插在 YOLO11n-seg backbone 与 neck 之间，对 P3（小目标层）特征做层级增强；可与 HVI 输入侧改造正交叠加。

---

## C. 频域/小波方法（3 篇）

### 9. WTConv
- **标题**：Wavelet Convolutions for Large Receptive Fields
- **第一作者**：Shahaf E. Finder | **年份**：2024 | **venue**：ECCV 2024（LNCS）
- **DOI**：10.1007/978-3-031-72949-2_21 | **arXiv**：2407.05848
- **核心思想**：在级联 Haar 小波分解的各频带上做小核卷积再逆变换，感受野随分解层数指数增长而参数仅对数增长；响应更偏低频/形状，对图像退化更鲁棒。
- **适用性**：远处模糊果实高频细节已丢失，低频形状/斑块信息是仅存线索——WTConv 的大感受野低频偏好正对症。
- **建议接入点**：替换 YOLO11n backbone 中 C3k2/瓶颈里的 3×3 卷积（drop-in），参数量几乎不增。

### 10. FreqFusion
- **标题**：Frequency-Aware Feature Fusion for Dense Image Prediction
- **第一作者**：Linwei Chen | **年份**：2024 | **venue**：IEEE TPAMI 2024
- **DOI**：10.1109/TPAMI.2024.3449959 | **arXiv**：2408.12879
- **核心思想**：指出密集预测中特征融合的两大病灶——类内不一致与边界位移；用自适应低通滤波平滑类内、偏移生成器重采样对齐、自适应高通滤波锐化边界，重建融合特征。
- **适用性**：暗黑模糊果实的特征本就"类内不一致"（同一果实明暗两半），FreqFusion 正是修这个；对分割掩码边界质量有直接收益。
- **建议接入点**：替换 YOLO11n-seg neck（PAN/FPN）中的上采样+concat 融合节点。

### 11. HWD（Haar Wavelet Downsampling）
- **标题**：Haar wavelet downsampling: A simple but effective downsampling module for semantic segmentation
- **第一作者**：Guoping Xu | **年份**：2023 | **venue**：Pattern Recognition（Vol. 143）
- **DOI**：10.1016/j.patcog.2023.109819
- **核心思想**：用 Haar 小波变换做"无损"下采样：空间分辨率减半的同时把信息搬到通道维，替代 stride 卷积/池化，减少下采样信息熵损失（配套提出 FEI 度量）。
- **适用性**：远处小果实本来就只占几十像素，stride-2 下采样是漏检主因之一——HWD 保留暗弱小目标的残存信息。
- **建议接入点**：替换 backbone 前两个 stride-2 Conv；与 WTConv/FreqFusion 组成"小波三件套"消融矩阵。

---

## D. 模糊鲁棒检测（2 篇）

### 12. 运动模糊对在线检测的影响与补救
- **标题**：Improved Handling of Motion Blur in Online Object Detection
- **第一作者**：Mohamed Sayed（与 Gabriel Brostow）| **年份**：2021 | **venue**：CVPR 2021
- **DOI**：10.1109/CVPR46437.2021.00175 | **arXiv**：2011.14448
- **核心思想**：系统量化运动模糊使检测器性能显著退化的机理（特征漂移、置信度塌陷），并验证针对模糊的训练策略（模糊增广及相关简单改动）能大幅恢复精度。
- **适用性**：为"模糊导致 YOLO 学不出远处果实"提供机理引文；证明不改架构、只改训练数据分布也有可观收益。
- **建议接入点**：训练管线加入散焦/运动模糊核增广（模拟远景失焦），作为最低成本的鲁棒性基线。

### 13. DeblurGAN-v2
- **标题**：DeblurGAN-v2: Deblurring (Orders-of-Magnitude) Faster and Better
- **第一作者**：Orest Kupyn | **年份**：2019 | **venue**：ICCV 2019
- **DOI**：10.1109/ICCV.2019.00897 | **arXiv**：1908.03826
- **核心思想**：首次把 FPN 用于去模糊生成器，配相对论双尺度判别器；骨干可换（Inception-ResNet 求质量 / MobileNet 求速度，可实时），并以"去模糊后 YOLO 检测精度提升"作为任务导向评测。
- **适用性**：其"去模糊服务下游检测"的评测范式可直接借用；MobileNet 版可作端侧预处理候选。
- **建议接入点**：(a) 离线去模糊清洗训练集；(b) 用其模糊合成管线造"清晰-模糊"配对做一致性/蒸馏训练。

---

## E. 阴影/逆光/复杂光照农业检测（1 篇 + 备选）

### 14. RFA-YOLOv8（复杂果园光照自适应增强）
- **标题**：RFA-YOLOv8: A Robust Tea Bud Detection Model with Adaptive Illumination Enhancement for Complex Orchard Environments
- **第一作者**：Qiuyue Yang | **年份**：2025 | **venue**：Agriculture (MDPI) 15(18):1982
- **DOI**：10.3390/agriculture15181982
- **核心思想**：面向复杂果园光照（逆光/阴影/光斑）的茶芽检测：光照自适应增强前端 + 注意力改进的 YOLOv8，提升光照鲁棒性。
- **适用性**：与本课题同构（果园小目标 + 复杂光照 + YOLO 改进），证明该选题范式在农业期刊的可发表性；小目标茶芽≈远处幼果。
- **建议接入点**：作为近期农业域对比方法/相关工作引文，佐证"光照自适应增强 + YOLO"路线。

### 备选补充（已核实，可按需替换进正文）
- **夜间荔枝检测**：A visual detection method for nighttime litchi fruits and fruiting stems — Cuixiao Liang, 2020, Computers and Electronics in Agriculture, DOI: 10.1016/j.compag.2019.105192。夜间人工补光下荔枝果实与结果母枝检测的经典 COMPAG 引文，证明"暗光水果检测"在农业顶刊有先例。
- **竞品信号（预印本）**：CCDW-YOLO: A Frequency-Aware Network for Citrus Detection in Complex Orchard Environments — SSRN 预印本, DOI: 10.2139/ssrn.7068286。**注意**：频域方法 + 柑橘 + 复杂果园与本课题高度撞车，说明方向热但需加快差异化（幼果实例分割 + HVI 暗部增强是差异点）。

---

## 组合建议（面向"远处发黑+模糊柑橘幼果"的创新拼图）
1. **输入/增强侧**（治"发黑"）：HVI 空间改造（仓库已有模块，#4）+ 检测损失驱动的轻量自适应增强（#5/#6/#7 范式），避免独立增强网络的噪声放大。
2. **特征侧**（治"模糊+小目标"）：FeatEnHancer 式层级特征增强（#8）+ WTConv 大感受野（#9）+ HWD 保信息下采样（#11）+ FreqFusion 融合（#10）——四者均即插即用、参数增量小，适合 YOLO11n 轻量约束。
3. **训练侧**（零架构成本）：模糊/暗光增广（#12 机理引文，#1/#3 造增广，#13 造配对）。
4. 论文故事线：远景退化二因素（欠曝+模糊）→ 频域/色彩空间双路修复 → 农业场景验证（#14 及备选佐证先例与竞争）。
