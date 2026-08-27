# 柑橘实例分割开源代码证据索引（2026-08-24）

## 结论先行

本轮不是把 36 个仓库中的模块机械拼接到 YOLO，而是用官方实现核对论文机制、算子代价和训练方式，再把可迁移部分重写成与当前 Ultralytics 分支兼容的最小实现。下载位置统一为
`C:\Users\33836\Desktop\github`。`CitrusSwift-Seg` 没有引入 Mamba、Detectron2、MMDetection 或自定义 CUDA 依赖。

代码“存在”不等于适合本任务。下面的 `采用` 表示机制进入本轮设计；`基线/后续` 表示保留作比较或下一阶段；`拒绝直接移植` 表示论文有价值，但直接搬入当前 nano YOLO 的风险或代价过高。

## 仓库清单与迁移决策

| 类别 | 官方/作者仓库 | 本地目录 | 固定提交 | 根目录许可证 | 本轮结论 |
|---|---|---|---|---|---|
| 小目标稀疏计算 | [QueryDet](https://github.com/ChenhongyiYang/QueryDet-PyTorch) | `QueryDet-PyTorch` | `feebf21` | 有 | 采用“低分辨率查询指导高分辨率信息”的原则；当前先做训练期 P2 查询监督，不移植稀疏 CUDA 推理栈 |
| 小目标定位损失 | [NWD](https://github.com/jwwangchn/NWD) | `NWD` | `9775ac2` | 有 | 采用可关闭的、仅对小目标启用的 NWD/CIoU 混合；必须单独消融 |
| 小目标蒸馏 | [TinyKD](https://github.com/haotianll/TinyKD) | `TinyKD` | `a743d2d` | 有 | 后续：最终大模型教师确定后再蒸馏，不能在架构筛选前混入 |
| 高频/调制 | [SFM](https://github.com/Linwei-Chen/SFM) | `SFM` | `b08e63e` | 有 | 拒绝整网移植；保留“下采样会伤害小目标”的问题诊断，避免全图重型频域支路 |
| 高分辨率主干 | [Lite-HRNet](https://github.com/HRNet/Lite-HRNet) | `Lite-HRNet` | `7b9049d` | 有 | 拒绝密集双流整网；历史和实测表明全图 P2 推理过慢 |
| 大核上下文 | [LSKA](https://github.com/StevenLauHKHK/Large-Separable-Kernel-Attention) | `Large-Separable-Kernel-Attention` | `bb2a8d2` | 有 | 采用在 P5 的单点 LSKA；这是旧结果中最稳定的正向结构信号 |
| 状态空间 | [SCSegamba](https://github.com/Karl1109/SCSegamba) | `SCSegamba` | `cc74c46` | 有 | 拒绝：用户不安装 Mamba；CUDA/部署依赖和小数据微调风险均高 |
| 边界互学习 | [BMask R-CNN](https://github.com/hustvl/BMaskR-CNN) | `BMaskR-CNN` | `c74b0bd` | 有 | 采用边界辅助监督思想，不移植两阶段 ROI 头 |
| 边界细化 | [RefineMask](https://github.com/zhanggang001/RefineMask) | `RefineMask` | `633ed2b` | 有 | 采用“细粒度特征只服务边界”的原则；不在全图运行多阶段细化 |
| 边界局部细化 | [BPR](https://github.com/tinyalpha/BPR) | `BPR` | `9eafc3f` | 有 | 后续可做独立后处理上界；不进入第一轮实时网络 |
| 高速轮廓 | [E2EC](https://github.com/zhang-tao-whu/e2ec) | `e2ec` | `a149a93` | 有 | 后续比较/启发；当前 YOLO 多边形标签与原型掩膜路径更易稳定继承 |
| 边界损失 | [boundary-loss](https://github.com/LIVIAETS/boundary-loss) | `boundary-loss` | `171c32d` | 有 | 机制参考；当前实现使用形态学可见边界监督，避免额外距离变换依赖 |
| 掩膜质量校准 | [Mask Scoring R-CNN](https://github.com/zjhuang22/maskscoring_rcnn) | `maskscoring_rcnn` | `0e8fae6` | 有 | 下一优先级：若 PR 曲线证实高分低质量掩膜排序错误，再加入 mask-IoU score |
| 重叠分层 | [BCNet](https://github.com/lkeab/BCNet) | `BCNet` | `d6580e8` | 有 | 仅用于理解遮挡层次；不做 amodal 推断，不改变可见掩膜标签定义 |
| 实时原型掩膜 | [YOLACT](https://github.com/dbolya/yolact) | `yolact` | `902073d` | 有 | 支持保留 YOLO 的 prototype + coefficient 路线，而非换成 ROI 逐实例高成本细化 |
| 稀疏实例分割 | [SparseInst](https://github.com/hustvl/SparseInst) | `SparseInst` | `a899015` | 有 | 跨家族正式比较；其 instance activation 也支持稀疏查询方向 |
| 实时查询分割 | [FastInst](https://github.com/junjiehe96/FastInst) | `FastInst` | `4996a61` | 有 | 采用“GT mask-guided learning 可只在训练期存在”的原则；保留为跨家族比较 |
| 位置式实例分割 | [SOLO](https://github.com/WXinlong/SOLO) | `SOLO` | `f4cd03b` | 有 | 论文要求的 box-free/location-based 比较家族，非主消融基线 |
| 伪装实例分割 | [DCNet](https://github.com/USTCL/DCNet) | `DCNet` | `f3c9098` | 有 | 采用实例前景与邻近背景应显式区分的原则；不移植重型 Fourier/prototype 栈 |
| 伪装实例分割 | [OSFormer](https://github.com/PJLallen/OSFormer) | `OSFormer` | `1786333` | 根目录未见 | 跨任务证据；不直接移植 Transformer 解码器 |
| 伪装实例分割 | [Mask2Camouflage](https://github.com/underlmao/Mask2Camouflage) | `Mask2Camouflage` | `64cde06` | 根目录未见 | 采用 foreground/background refinement 问题定义；不直接移植 |
| 搜索-识别伪装 | [SINet](https://github.com/DengPingFan/SINet) | `SINet` | `6202fb1` | 根目录未见 | 采用“先找候选，再辨别”的抽象；当前以训练期 query + contrast 实现 |
| 频带/边缘伪装 | [FEDER](https://github.com/ChunmingHe/FEDER) | `FEDER` | `fac6b2a` | 有 | 采用边缘重建监督；拒绝全套可学习小波解码器 |
| 频域边缘伪装 | [EPFDNet](https://github.com/LitterMa-820/EPFDNet) | `EPFDNet` | `3845e5a` | 根目录未见 | 支持局部高频与边界协同；不做全图频域双支路 |
| 纹理/梯度伪装 | [DGNet](https://github.com/GewelsJI/DGNet) | `DGNet` | `1ecc47a` | 有 | 支持梯度线索；当前只在训练辅助分支使用高频残差 |
| 伪装结构搜索 | [CamoNAS](https://github.com/rendaweiSIMIT/CamoNAS) | `CamoNAS` | `e028e1b` | 根目录未见 | 研究参考；数据规模与算力不支持本轮 NAS |
| 伪装特征选择 | [FSEL](https://github.com/CSYSI/FSEL) | `FSEL` | `f3be464` | 有 | 研究参考；不叠加额外多尺度选择模块 |
| 实时分割三分支 | [PIDNet](https://github.com/XuJiacong/PIDNet) | `PIDNet` | `4c158cf` | 有 | 采用 detail/context/boundary 分工的原则；同时采纳其“FLOPs 不等于延迟”的警告 |
| 多模型分割库 | [OpenSeg](https://github.com/openseg-group/openseg.pytorch) | `openseg.pytorch` | `aefc755` | 有 | 参考语义分割基线和边界评估，不合入 YOLO 主线 |
| 结构重参数化 | [FastViT](https://github.com/apple/ml-fastvit) | `ml-fastvit` | `8af5928` | 有 | 采用训练/推理解耦与实测延迟原则；本轮提供可融合 P5 context 候选 |
| 结构重参数化 | [RepNeXt](https://github.com/suous/RepNeXt) | `RepNeXt` | `f515377` | 有 | 结构参考，不整块移植 |
| 移动端结构 | [RepViT](https://github.com/THU-MIG/RepViT) | `RepViT` | `298f420` | 有 | 后续部署参考；本轮不替换整个预训练主干 |
| 部分卷积 | [FasterNet](https://github.com/JierunChen/FasterNet) | `FasterNet` | `e8fba44` | 根目录未见 | 支持按真实吞吐筛算子；旧 Faster 模块无稳定精度证据，暂不进入完整模型 |
| 高效 ViT | [EfficientViT](https://github.com/mit-han-lab/efficientvit) | `efficientvit` | `de7d773` | 有 | 跨部署参考，不在小数据上重建主干 |
| 大颈小头 | [DAMO-YOLO](https://github.com/tinyvision/DAMO-YOLO) | `DAMO-YOLO` | `319572e` | 有 | 采用小预测头原则；本轮还压缩了冗余 P5 bottom-up 融合以控制延迟 |
| 工业重参数网络 | [YOLOv6](https://github.com/meituan/YOLOv6) | `YOLOv6` | `e86a483` | 有 | 支持 Rep-PAN/部署友好设计；不直接替换当前训练生态 |

## 未成功获得的代码

- FDCOD 论文页面给出的 `luckybird1994/FDCOD` 在 2026-08-24 克隆时不可公开访问/要求认证，因此不能声称已复用其代码。
- SharpContour 项目页仍显示代码 “Coming soon”，没有把非官方复现冒充官方实现。
- PointRend 的官方实现位于 Detectron2 `projects/PointRend`；考虑仓库体积和本轮不直接集成 Detectron2，只引用论文与官方项目路径，未重复下载整个 Detectron2。

## 实际进入 CitrusSwift-Seg 的证据映射

| 当前实现 | 证据来源 | 实现边界 |
|---|---|---|
| P5 单点 LSKA | LSKA 官方代码 + 本项目 F14 历史结果 | 只放在低分辨率 P5，避免全图注意力成本 |
| `SPPFRepContext` 备选 | RepVGG/FastViT 的结构重参数化原则 | 使用当前 Ultralytics 已有 `RepVGGDW`，融合前后做数值等价测试 |
| 训练期 P2/P3 tiny-query | QueryDet、FastInst | 训练期保留高分辨率监督，推理不建立密集 P2 检测头 |
| 可见边界辅助监督 | BMask R-CNN、RefineMask、PointRend、FEDER | 监督的是可见掩膜边界，不补全被遮挡区域 |
| fruit-vs-leaf 邻域对比监督 | SINet、DCNet、Mask2Camouflage | 由 GT 果实内部与紧邻外环构造；不是引入新的人工标签 |
| 轻量预测头 | DAMO-YOLO、YOLACT、FastInst | 原型掩膜路径不变；减少检测/系数头的重复空间卷积 |
| 非对称 PAN | 延迟实测 + QueryDet 的稀疏计算原则 | 保留 P3→P4 回流，删除收益最可疑的 P4→P5 回流 |
| 小目标 NWD/CIoU 混合 | NWD 官方代码/论文 | 默认关闭；仅小目标门控；作为损失消融而非默认真理 |

## 许可证与学术边界

本轮新增代码是针对当前 Ultralytics 分支的重新实现，没有将上述仓库的大段源码复制进项目。发表前仍需逐项核对最终引用、许可证和任何真正复用的代码片段。对根目录没有许可证的仓库，只能阅读验证思路，不能默认其代码可自由再分发。
