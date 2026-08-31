# `Plug-play-modules-main` 柑橘任务适配审计

审计日期：2026-08-28  
审计对象：`C:\Users\33836\Desktop\github\Plug-play-modules-main`

## 1. 仓库可信度边界

- 本地目录有 107 个 Python 文件，但没有 `.git` 历史，也没有根目录许可证；无法从这个副本追溯每个文件的提交版本或法律授权。
- README 的定位是“即插即用/涨点”代码汇总，不是单篇论文的官方复现仓库；部分中文注释已乱码。
- 汇总代码与官方源码并不等价。例如汇总版 `FreqFusion.py` 强制导入 `mmcv.ops.carafe`，而当前官方源码已经增加了纯 PyTorch 备用实现。
- 因此，本目录只用作论文线索索引。进入模型的思想必须回溯论文和官方仓库，并设置单因素消融。

## 2. 逐项结论

| 线索 | 原任务与依据 | 与柑橘痛点的关系 | 工程/证据风险 | 决策 |
|---|---|---|---|---|
| FreqFusion | TPAMI 2024；语义、检测、实例、全景分割；官方仓库 `Linwei-Chen/FreqFusion` | 自适应低通抑制类内高频噪声，自适应高通恢复边界，正面对应叶片纹理干扰和边界位移 | 官方仓库无根许可证；纯 PyTorch CARAFE 备用实现含调试输出且 `unfold` 显存代价高；尚无柑橘/YOLO 证据 | **保留为后续独立颈部实验**。D 主干成立后再以单个 top-down 融合点做 D10/E01，不能混入 D01-D09 |
| HCF-Net/PPA | arXiv 2024 红外小目标语义分割；官方仓库 Apache-2.0 | 高低层上下文与小目标有概念相关性 | 单通道红外与 RGB 实例分割域差异大；完整模型是 U-Net；PPA 本身叠加局部全局注意力、ECA、空间注意力、3 个卷积及 dropout，不能回答是哪一项起作用 | **不进入 D 系列**；其 `Bag` 选择性融合思想已由更成熟的 PIDNet 证据覆盖 |
| ContrastDrivenFeatureAggregation | AAAI 2025 ConDSeg 医学语义分割 | 前景/背景对比在表面上类似果实/叶片混淆 | 依赖显式前景、背景特征构造和局部展开注意力；医学语义分割没有相邻实例拆分问题；与“减少颜色依赖”目标可能相反 | **排除** |
| PKIBlock/CAA | CVPR 2024 遥感旋转检测 | 多尺度大核上下文可能覆盖大尺度跨度 | 与已经训练的 LSKA/CAA 类路径高度重叠；既有柑橘结果没有显示大核注意力能稳定提升 | **排除当前系列** |
| HaarDownsampling | Pattern Recognition 2023 语义分割 | 下采样时显式保留高频 | 额外 `pytorch_wavelets` 依赖；C04/C06 已经给出同类独立实验位，不能重复包装成新贡献 | **由 C 系列结果决定，不重复加入 D** |
| DFE | CVPR 2022 MonoDTR 单目 3D 检测的深度感知特征增强 | 无 | 输出和使用显式深度分支；违反当前论文只做 RGB 实例分割的范围 | **排除** |
| CF_loss | 3D 视网膜血管拓扑损失 | 拓扑概念表面相关 | 代码维度、类别定义和硬编码 CUDA 都针对 3D 医学体数据；不可直接迁移到 2D 多实例可见掩膜 | **排除** |
| Feature Refinement / FSAS / FFT 类 | 图像恢复或 Transformer 恢复 | 与可见掩膜边界仅弱相关 | 任务目标、监督和算子成本均不匹配 | **排除** |
| CGAFusion / Multi-scale AwarenessFusion | 通用融合/注意力代码 | 泛化描述可对应多尺度 | 缺少针对小目标、遮挡拓扑或实例边界的直接证据；会回到“堆注意力”路径 | **排除** |

## 3. 官方源码核验

已下载并阅读：

- `C:\Users\33836\Desktop\github\FreqFusion`，提交 `3fb0c70637a3c194fb74294d3ce4681958b26241`。
- `C:\Users\33836\Desktop\github\HCFNet`，提交 `fda3279e2d9c2f2b31abe9c67d8487b226d2ae98`。

FreqFusion 的正式证据包括 Mask R-CNN R50 的 COCO Mask AP 从 34.7 提升到 36.0，但这只是跨任务先验，不是本数据集的预期涨点。HCF-Net 的证据来自 SIRST 单通道红外语义分割，不能直接外推到 RGB 柑橘实例分割。

## 4. 对 D 系列的实际影响

本次审计没有把任何汇总模块直接复制进网络。D01-D09 保持一个可解释的主干假设：持久高分辨率形状流、像素差分结构证据、深层语义门控和选择性注入。FreqFusion 被记录为主干验证后的颈部单因素候选；HCF/PPA 和其余模块不加入，避免再次形成无法归因的模块堆叠。

## 5. 原始来源

- FreqFusion paper: https://doi.org/10.1109/TPAMI.2024.3449959
- FreqFusion official code: https://github.com/Linwei-Chen/FreqFusion
- HCF-Net paper: https://arxiv.org/abs/2403.10778
- HCF-Net official code: https://github.com/zhengshuchen/HCFNet
- ConDSeg official code: https://github.com/Mengqi-Lei/ConDSeg
- PKINet official code: https://github.com/PKINet/PKINet

