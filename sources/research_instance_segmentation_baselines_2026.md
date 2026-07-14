# 柑橘幼果实例分割 Baseline 检索审计

> 检索日期：2026-07-13
> 目标：选择不局限于 YOLO、可复现且适合轻量化论文的实例分割基线。

## 选择原则

1. 必须同时输出实例类别、检测框和 mask，不能用仅语义分割或需要人工提示的 SAM 替代。
2. 必须有公开训练代码和 COCO 预训练权重，并能迁移到单类自定义数据。
3. 轻量对比尽量在相同输入尺寸下报告 Params、GFLOPs 和实测延迟。
4. 主消融 baseline 只保留一个；其他模型用于横向比较，不为每个框架实现本文模块。
5. 不强迫所有模型使用完全相同的优化器，但应固定数据划分、输入尺寸、训练图像次数、预训练来源和评估脚本。

## 推荐模型

| 角色 | 模型 | 官方规模/特点 | 建议 |
|---|---|---|---|
| 主消融基线 | YOLO11n-Seg | 本地已有结果，2.835M、10.2 GFLOPs | 保留，最快完成方法开发 |
| 非 YOLO 轻量主对照 | RTMDet-Ins-tiny | 官方约 5.6M、11.8 GFLOPs，640 输入 | 必做；与 YOLO11n 规模接近 |
| 经典两阶段 | Mask R-CNN R50-FPN | 标准 proposal-based instance segmentation | 必做；回答与经典方法的差异 |
| 当前 Transformer | RF-DETR Seg Nano | ICLR 2026；官方约 33.6M，默认 312 分辨率 | 必做或强烈建议；属于现代强对照，不称轻量参数模型 |
| 无框位置分割 | SOLOv2-Light R18-FPN | 动态 mask kernel、Matrix NMS、无需检测框 | 期刊版建议加入 |
| 旧代 YOLO | YOLOv8n-Seg | 成熟、广泛使用 | 必做；排除版本代际因素 |
| 当前 YOLO 强对照 | YOLO26n-Seg | 当前 fork 可运行 | 必做；回应最新 YOLO 对比 |
| 动态 mask，可选 | CondInst R50-FPN | 实例条件动态卷积，无 ROI crop | SOLOv2 已加入时通常省略 |
| 稀疏实时，可选 | SparseInst R50-d-DCN | 稀疏实例激活图，实时路线 | SOLOv2 已加入时通常省略 |
| 精度上界，可选 | Mask2Former R50 | 通用 mask classification Transformer | 显存和时间充足时加入 |
| 经典语义分割 | U-Net + marker watershed | 二值前景后处理为独立实例 | 必做辅助基线 |
| 多尺度语义，可选 | DeepLabV3+ + marker watershed | 空洞卷积与边界解码 | 与 SegFormer-B0 二选一 |
| 轻量 Transformer 语义，可选 | SegFormer-B0 + marker watershed | 轻量多尺度 Transformer | 与 DeepLabV3+ 二选一 |

## 最小可投稿对比集

核心实例分割表推荐以下 6 个模型：

1. YOLOv8n-Seg
2. YOLO11n-Seg
3. YOLO26n-Seg
4. RTMDet-Ins-tiny
5. Mask R-CNN R50-FPN
6. RF-DETR Seg Nano

期刊版或算力允许时增加：

7. SOLOv2-Light R18-FPN

另设“语义转实例”辅助表：

1. U-Net + distance transform + marker-controlled watershed
2. DeepLabV3+ + watershed 或 SegFormer-B0 + watershed，二选一

这组覆盖旧代/当前 YOLO、一阶段非 YOLO、经典两阶段、当前 Transformer、无框位置分割和经典语义分割。加入 SOLOv2 后不再同时增加 CondInst 与 SparseInst，避免多个 2020-2022 年动态 mask 模型造成重复。

## SOLOv2 是否必须加入

SOLOv2 将实例分割转化为按位置预测类别和 mask，并通过动态 mask kernel 与 Matrix NMS 直接输出实例，不依赖候选框。它对本数据有两项诊断价值：

1. 可检验不依赖检测框的 mask 生成方式是否更适合深凹轮廓；
2. 可暴露密集同类果实中心过近、跨尺度网格分配时的定位冲突。

推荐使用 MMDetection 当前提供的 `solov2-light_r18_fpn_ms-3x_coco.py`，而不是第一代 SOLO。官方 COCO 结果中 Light R18 约为 29.7 Mask AP，但该数字使用官方输入和训练协议，不能直接与本项目结果比较。正式实验必须统一到 640 或最近支持尺寸，并重新测量 Params、GFLOPs、显存和延迟。

SOLOv2 不是主消融 baseline，原因是其位置网格与动态 mask 结构和当前 Ultralytics 改动路径差异过大。它的角色是证明本文方法相对另一种原生实例分割范式仍有竞争力。

## U-Net 类模型的公平使用

U-Net、DeepLabV3+ 和 SegFormer 默认输出的是前景/背景语义 mask，同一类别的接触果实会被合并，不能直接产生实例 ID。实验时采用统一流程：

1. 将所有实例标签合并为二值幼果前景，训练语义分割模型。
2. 输出前景概率图，并从距离变换或单独中心图中生成 marker。
3. 使用同一 marker-controlled watershed 实现分离接触果实。
4. 连通域转换为独立实例，置信度取区域内前景概率的均值或分位数。
5. 分水岭阈值只在 val 集确定，不允许按 test 图像调参。

语义表报告 Dice、mIoU、Boundary F1；实例表把后处理结果转换为 COCO 格式，报告相同的 Mask AP。模型名称必须写成 `U-Net + Watershed`，不能把后处理后的结果简称为 U-Net 实例分割。

仓库中的 `1.coding/2_Unet/` 是早期学习代码，当前网络输出通道固定为 21，数据与评估流程也不是柑橘实例分割协议，不应直接作为论文 baseline。正式实验优先使用 `segmentation_models_pytorch` 的 U-Net（固定 encoder 和 ImageNet 预训练）或 MMSegmentation 官方 U-Net 配置，另写统一的数据转换、watershed 和 COCO 评估适配层。

## 公平实验协议

- 所有模型使用同一 group-aware train/val/test 文件清单和同一类别定义。
- 主精度表使用 640x640；RF-DETR 应设置为支持的相同或最近分辨率，不能把默认 312 的速度与 640 模型直接比较。
- 使用官方 COCO 预训练；记录成功加载范围。
- 每个框架采用其稳定的官方优化器/学习率策略，但固定训练图像总次数和早停规则。
- 速度统一在同一 GPU、batch=1、相同精度模式、相同预热次数下测量。
- 主表报告 Mask mAP50-95、AP50、AP-small/medium/large、Params、GFLOPs、显存和延迟。
- 难例表报告遮挡、同色低对比、相邻粘连和极端尺度差异子集。

## 是否切换主 baseline

先用相同新划分对 YOLO11n-Seg 与 RTMDet-Ins-tiny 各做一次 50 epoch 筛选：

- 若 RTMDet-Ins-tiny 的 Mask AP 与 YOLO11n 相差不超过 1.5 点、训练稳定且难例表现更好，可以考虑改为主 baseline，减少论文的 YOLO 同质化。
- 若适配 MMDetection 明显拖慢进度，或 RTMDet 落后超过 1.5-2.0 点，则维持 YOLO11n-Seg 为主消融基线，把 RTMDet 作为最重要的非 YOLO 横向对照。

## 官方来源

- RTMDet paper: https://arxiv.org/abs/2212.07784
- MMDetection RTMDet model zoo:
  https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet
- MMDetection projects, including CondInst and SparseInst:
  https://github.com/open-mmlab/mmdetection/tree/main/projects
- Mask R-CNN: https://arxiv.org/abs/1703.06870
- SparseInst: https://arxiv.org/abs/2203.12827
- SOLOv2: https://arxiv.org/abs/2003.10152
- MMDetection SOLOv2 configs:
  https://github.com/open-mmlab/mmdetection/tree/main/configs/solov2
- CondInst: https://arxiv.org/abs/2003.05664
- Mask2Former: https://arxiv.org/abs/2112.01527
- RF-DETR Segmentation benchmarks:
  https://rfdetr.roboflow.com/develop/learn/benchmarks/
- RF-DETR paper: https://arxiv.org/abs/2511.09554
- U-Net: https://arxiv.org/abs/1505.04597
- DeepLabV3+: https://arxiv.org/abs/1802.02611
- SegFormer: https://arxiv.org/abs/2105.15203
- segmentation_models_pytorch:
  https://github.com/qubvel-org/segmentation_models.pytorch
- MMSegmentation U-Net:
  https://github.com/open-mmlab/mmsegmentation/blob/main/configs/unet/README.md
