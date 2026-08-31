# G_0839：Citrus Preserve--Search--Discriminate--Refine 系列

`G_0839` 是一个严格递进的结构消融系列，不是旧 `G_series` 的改名，也不是注意力模块合集。它只处理当前 RGB 未成熟柑橘实例分割的三个已量化难点：极小目标、绿色果实与叶片局部伪装、条带遮挡和相邻果实造成的凹边界与 split/merge 冲突。

## 六个模型

| 模型 | 相对前一模型的唯一主要变化 | 实际存在的辅助输出 | Params | GFLOPs@640 | 要回答的问题 |
|---|---|---|---:|---:|---|
| G00 | 标准主干/颈部 + 已验证的单层轻量检测头 | 无 | 2.716M | 9.62 | 轻量头控制组，不能冒充官方 YOLO11n-seg 基线 |
| G01 | 整体替换 C3k2 主干：无色差分 stem、深度可分残差语义流、持续 P2 流与三次双向交换 | 无 | 2.732M | 10.71 | 非 YOLO 双分辨率主干是否有效 |
| G02 | P3/P4 粗查询选择 P2 支持区域 | query | 2.741M | 10.95 | 极小果实候选搜索是否提高召回 |
| G03 | 增加果实内部—近邻环局部差分判别 | query + contrast | 2.742M | 10.97 | 是否降低绿色果实/叶片混淆 |
| G04 | 增加可见边界支持 | query + contrast + boundary | 2.742M | 11.00 | 深凹可见掩膜和条带遮挡边界是否改善 |
| G05 | 增加 context/interior/boundary/separator 四状态拓扑 | query + contrast + boundary + topology | 2.742M | 11.00 | 是否减少相邻果实的 split/merge 错误 |

G01--G05 的主干不含 C3k2/CSP：`CitrusDualResolutionBackbone` 用低通降采样和深度可分残差块形成语义 P3--P5，同时把 P2/4 一直保留到主干末端；每个 stage 只用语义门控形状更新，再把池化后的形状证据回送语义流。颈部暂时保留相同 YOLO PAN/C3k2，以便 G00→G01 只回答“更换主干”而不同时更换颈部。

G02--G05 共享一个 `CitrusSDRSupport`，后续阶段复用相同 P2 表征，避免每个问题各堆一个模块。检测仍位于 P3--P5；P2 只修正掩膜原型，因此不会引入容易放大叶片误检的密集 P2 检测头。

## 论文与开源实现依据

- QueryDet：低分辨率查询后再处理高分辨率小目标候选，代码 `ChenhongyiYang/QueryDet-PyTorch`。
- Lite-HRNet、Gated-SCNN、PIDNet 与 PiDiNet：持续高分辨率形状支路、语义门控和像素差分证据。
- Mask2Camouflage：目标内部与周围上下文互补判别。
- RefineMask：细粒度边界特征逐步改善实例掩膜。
- Panoptic-DeepLab/HoVer-Net 类思想：利用边界或实例间分隔状态降低合并错误。

本地证据、论文入口、仓库 commit 和许可证状态见 `E:/mastercode/3_研究生/paper1_finalization_20260830/sources/`。当前实现是根据机制重新实现，不复制许可证不明确或非商业仓库的源代码。

## 正确实验顺序

1. 六个 YAML 全部做构建、前向、反向和 1--3 epoch smoke。
2. 同一数据划分、初始化和超参数下跑 50 epoch G00--G05 筛选。
3. 只允许在 Mask mAP50-95、AP_small、低 solidity 子集、near-gap 子集和 split/merge 指标共同支持时保留阶段。
4. 300 epoch 只跑筛选胜出的 1--2 个结构；最终方法和官方 YOLO11n-seg 均跑三个种子。

正式 G_0839 协议锁定 `amp=false`，与已有 S/B grouped-clean 实验一致。AMP 不是网络结构因素；若要研究它，只允许用相同模型、数据、seed 和超参数做 AMP on/off 成对实验。

批量和单模型入口都会把同一个 gain 向量传给全部六个模型：`query=0.03, contrast=0.05, boundary=0.10, topology=0.05`。某个阶段没有对应输出时，该损失自然为零。这样每份 `args.yaml` 的超参数完全相同；表中列的是模型实际存在的输出，而不是不同的命令行超参数。

查询 top-k 当前仅让高分辨率**信息支持区域**稀疏；普通 PyTorch 仍执行轻量 P2 卷积，不能把它表述成已经获得稀疏算子加速。实际速度必须以目标服务器实测 latency 为准。
