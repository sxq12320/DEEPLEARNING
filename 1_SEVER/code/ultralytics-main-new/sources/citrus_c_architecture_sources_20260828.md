# C 系列架构证据来源（检索日期：2026-08-28）

本文件记录 C 系列实际采用的公开证据，以及从论文结论到本项目实现之间的边界。论文只支持设计动机，不等于保证在柑橘数据上提升；最终判断必须依赖受控消融。

## 直接采用的证据

### YOLACT：原型与实例系数解耦

- 论文：Bolya et al., *YOLACT: Real-Time Instance Segmentation*, ICCV 2019。
- 论文页：https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html
- 证据：单阶段实例分割可以用一组图像级 prototype 与每个实例的 mask coefficient 线性组合生成实例掩膜。
- 本项目适配：保留 Ultralytics 原有系数机制，把 32 个 prototype 显式拆成 24 个 P3 语义原型和 8 个 P2 细节原型；不是照搬一个新检测器。

### RefineMask：高分辨率细节与边界细化

- 论文：Zhang et al., *RefineMask: Towards High-Quality Instance Segmentation with Fine-Grained Features*, CVPR 2021。
- 论文页：https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html
- 证据：连续下采样会损失掩膜细节，高分辨率细粒度特征和边界监督有助于困难边界。
- 本项目适配：P2 只贡献少量可学习细节原型和拓扑状态，不引入多阶段 ROI mask refinement，控制推理成本。

### Lite-HRNet：轻量高分辨率表示

- 论文：Yu et al., *Lite-HRNet: A Lightweight High-Resolution Network*, CVPR 2021。
- PDF：https://openaccess.thecvf.com/content/CVPR2021/papers/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.pdf
- 证据：持续高分辨率表示适合密集预测，但完整双流高分辨率网络并非免费。
- 本项目适配：没有复制完整 HRNet；只保留 P2 到 prototype 路径，检测仍在 P3–P5，以避免 S 系列中高分辨率检测带来的速度代价。

### HWD：基于 Haar 小波的下采样

- 论文：*Wavelet integrated CNNs for noise-robust image classification*, Pattern Recognition 2024, DOI: 10.1016/j.patcog.2023.109819。
- 出版页：https://www.sciencedirect.com/science/article/pii/S0031320323005174
- 开源仓库：https://github.com/apple1986/HWD
- 证据：Haar 分解可在降采样时把不同频带显式保留到通道维。
- 本项目适配：只在 C04/C06/C08 的一次 P2→P3 下采样测试，C03 保持标准主干作为严格控制。旧实验中 HWD 没有证明普遍有效，因此它不是核心方法。

## 没有被当作既定事实的部分

- “四状态拓扑图一定提高柑橘 AP”是待检验假设，不是上述论文已经验证的结论。
- context/interior/boundary/separator 标签由现有实例 mask 形态学派生，不新增叶片语义标签。
- CARAFE、RepContext、HWD 只作为单因素消融；B 系列已经否定把所有模块直接叠加的做法。
- C 系列不使用 Mamba、MMCV、自定义 CUDA 或新的部署依赖。
