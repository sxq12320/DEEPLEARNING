# SAGE-v4 全历史复盘的补充检索

_2026-09-03；检索客户端不可用，使用网页检索和本地作者仓库交叉核对。只将论文与作者代码用于技术结论。_

## 🔗 已核对的来源

- PIDNet：细节、语义、边界分工；不是本项目已获得控制稳定性证明。作者代码：https://github.com/XuJiacong/PIDNet
- AFPN：渐进式多尺度融合，作者实现仍有多轮融合和卷积，不能直接当成轻量颈部。作者代码：https://github.com/gyyang23/AFPN
- RefineMask：逐阶段引入细节并细化掩膜，是两阶段实例分割方法，不是 YOLO 的现成替换块。论文：https://openaccess.thecvf.com/content/CVPR2021/html/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.html ，作者代码：https://github.com/zhanggang001/RefineMask
- FreqFusion：自适应低通、高通和重采样；与项目 G03 的均值池化高低频残差不是同一实现。论文：https://arxiv.org/abs/2408.12879 ，作者代码：https://github.com/Linwei-Chen/FreqFusion
- Gated-SCNN：使用高层语义指导独立形状流，验证于语义分割，不能自动获得实例身份。论文：https://arxiv.org/abs/1907.05740 ，作者代码：https://github.com/nv-tlabs/GSCNN
- DBPN：上下投影之间显式计算重投影误差，验证于超分辨率而非柑橘分割。论文：https://openaccess.thecvf.com/content_cvpr_2018/html/Haris_Deep_Back-Projection_Networks_CVPR_2018_paper.html
- SRFBN：通过循环隐藏状态反馈高层信息；延迟与训练成本需重新评估。论文：https://openaccess.thecvf.com/content_CVPR_2019/html/Li_Feedback_Network_for_Image_Super-Resolution_CVPR_2019_paper.html ，作者代码：https://github.com/Paper99/SRFBN_CVPR19
- ReZero：小初始残差思想的来源之一，不证明本项目非线性网络全局稳定。论文：https://proceedings.mlr.press/v161/bachlechner21a.html

## 📋 首次查询记录

查询为 PIDNet detail/context/boundary、AFPN progressive fusion、RefineMask boundary refinement、FreqFusion，以及 DBPN error feedback、Gated-SCNN shape stream。排除营销页面、二手解读和未能对应作者实现的结论；这些来源仅支持机制选择，未复现论文指标。

## 🔍 补充源码核对

- 已读作者 [AFPN 的 YOLO 实现](https://github.com/gyyang23/AFPN/blob/master/mmyolo/mmyolo/models/necks/yolov5_afpn.py)：先进行两尺度融合，再进入三尺度融合；包含多次 BasicBlock，且先压缩到输入通道的四分之一。不能只复制融合算子而遗漏通道预算，再宣称继承其轻量性。
- 已读作者 [DBPN 的 UpBlock/DownBlock](https://github.com/alterzero/DBPN-Pytorch/blob/master/base_networks.py)：存在明确的上下投影、回投误差、残差修正。这支持“如何构造可检查的误差路径”，不支持其迁移到柑橘分割必然有效。
- 一次错误的 AFPN 原始路径返回 404，GitHub 树 API 也遇到未认证访问限流；随后通过作者网页目录找到了实际文件。未执行下载代码，未安装额外依赖。
