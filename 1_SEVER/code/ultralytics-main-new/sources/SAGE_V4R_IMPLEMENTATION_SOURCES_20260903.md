# SAGE V4R 实现阶段来源记录

## 📚 核对范围

日期：2026-09-03。依据此前的全历史代码/结果复盘和作者实现阅读，本轮进一步核对实现动机。
本地 research-lookup 的专用 CLI/凭据不可用，使用网页检索和已有本地来源作为替代。
不把汇总模块库的宣传、文件名或论文原任务成绩当作柑橘实例分割增益证据。

| 原始来源 | 可支持的机制 | 本轮实际用法与限制 |
| --- | --- | --- |
| [Gated-SCNN 论文](https://arxiv.org/abs/1907.05740) / [作者仓库](https://github.com/nv-tlabs/GSCNN) | 高层语义对形状细节进行门控 | 独立实现 16 通道语义引导掩膜细节；没有复刻原语义分割网络 |
| [DBPN 作者 UpBlock/DownBlock](https://github.com/alterzero/DBPN-Pytorch/blob/master/base_networks.py) | 上下投影误差与残差修正 | 一次低分辨率回投影，不用其密集超分辨率主干；不承诺控制稳定性 |
| [FasterNet 作者仓库](https://github.com/JierunChen/FasterNet) | PConv 与延迟友好的设计动机 | 复用项目已有 C3k2_Faster，仅作为 P4 对照，不称完整 FasterNet |
| [MambaOut 作者仓库](https://github.com/yuweihao/MambaOut) | 不含 SSM 的门控卷积 | 复用已核对的 SAGEGatedStage，仅替换 P4，不安装 Mamba |
| [BMask R-CNN](https://arxiv.org/abs/2007.08921) | 实例边界监督 | 改为同一 mask logits 的可分辨边界重加权；阈值与系数是预设实验参数 |
| [RefineMask 作者仓库](https://github.com/zhanggang001/RefineMask) | 细粒度信息参与掩膜细化 | 不照搬 RoI 多阶段头，不添加重复高分辨率检测塔 |

本轮网页已成功核对 Gated-SCNN 的 arXiv 摘要和 FasterNet 的作者 README。
DBPN 的 CVF 页面再次返回 403，采用此前已读的作者 `base_networks.py`，没有假称此次成功读取该网页全文。
更早的源码核对、桌面仓库与引用记录保留在 `SAGE_V4_SOURCE_INDEX_20260903.md` 和
`SAGE_V4_REASSESSMENT_SEARCH_20260903.md`。

## ⚖️ 方法归属

新增 SAGEMaskCorrection、SegmentCitrusSAGEV4R 与几何重加权为本项目独立适配。
继承的 CSP/PConv、门控阶段和 Ultralytics 基础头保留原代码路径与许可证。
本轮没有用模块合集中的整段不明许可代码替换项目文件，也没有执行第三方仓库安装器。
研究假设是有针对性的归纳偏置可能有益；是否优于基线必须由预注册预算和同协议消融验证。
