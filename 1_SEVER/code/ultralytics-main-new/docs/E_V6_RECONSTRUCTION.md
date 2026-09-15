# E V6：掩膜分辨率—小目标回归—实例多样性实验

> 2026-09-12 更新：本文件保留 swe-2 初版设计背景。当前实现、十组实验、共同验证栅格和实际测试结果，以 [V5 复审与 V6 修订报告](E:/mastercode/1_SEVER/code/ultralytics-main-new/docs/E_V6_REVIEW_20260912/复审结论与运行说明.md) 为准。不能按下面旧版的验证口径直接比较不同 proto_stride 模型。

2026-09-11。已实现并完成本地构建/前向/反向/官方验证通路检查；**尚未进行 V6 的正式训练，不能承诺涨点**。

## 1. 先说结论

V5 的六个已收到臂中，V5_01_fine 的最佳 Mask AP50–95 为 68.125，对照为 67.653；末 20 轮均值仅高约 0.050 个百分点。缺乏重复种子，不能人为设定 ±0.3~0.5pt 为统计噪声带。mask_route 与 region 没有一致、可叠加的证据；细切片推理时 route 的极小果匹配 100/161，对照 93/161，但背景误检增加（364→437）。

历史线一致指向：**结构性损失不在大目标检出（≥1024 px² 桶已匹配 98%），而在小目标的严格 IoU 掩膜质量与候选召回**。AP50(84)→AP50-95(68) 约 16 个百分点的缺口是 V6 的主目标。

V6 只拆三个单变量因子，全部臂复用 V5_01 的多尺度输入配方，控制臂即 V5_01 架构复现：

| 因子 | 机制 | 证据依据 |
|---|---|---|
| F 细掩膜 | proto 从 160²→320²：P1 茎特征经已精化的 P2 detail 估计门控后残差注入原型；监督侧 mask_ratio 4→2 | RefineMask/MaskTransfiner 证明细粒度特征修边有效；53.3% 小目标 + 160² 栅格下 6px 目标只剩 1.5px 掩膜 |
| N 小目标回归 | nwd_ratio=0.5（尺寸门控 NWD/CIoU 混合），此前只在失败的 full 捆绑中出现、从未单测 | NWD 原文对极小目标 +6.7 AP；<8px 框的 IoU 对 1px 偏移近乎二值 |
| P 实例多样性 | copy_paste=0.3 flip 模式（同图翻转实例粘贴到 IoA<0.3 区域），从未单测 | Simple Copy-Paste 对小数据集/稀有实例增益显著 |

V6_01_mr2 单独隔离监督栅格因子（同架构、仅 mask_ratio=2，proto 仍为 160²），避免 F 与栅格混淆。

## 2. 核对范围与可比性

- 逐段阅读 E_V5 六个已完成 run 的 results.csv、args.yaml、completed.json、best_mask_selection.json、best_mask50_selection.json、paired_coarse/fine 评估、loaded_data_summary、initialization_transfer。V5_05/V5_06 没有上传运行，2×2×2 未闭合。
- V5 协议不完全对称：control/route/region 用 `_prepared_views`，fine/fine_route/full 用 `_prepared_multiscale`。不能把 V5_01 的增益全部归给"训练时见细视图"以外的成分，也不能说 T 因子在固定输入下无效。
- 官方指标（每轮整图验证）与配对诊断（长边 640 公共栅格）不混用；best AP50 与 best AP50-95 来自不同 epoch，不拼接。
- 本地核对数据集：训练 676 图/4,324 实例，验证 193/1,049，测试 96/524；源分辨率多为 3072²。训练集按源图面积中位约 21.6k px²——640 输入下中位实例只有约 103 px² 网格面积，"小目标"主要是**输入分辨率造成的**，不是标注本身微小。
- 旧难度审计（965 图版数据）：COCO-small 53.26%，min-side<16px 17.39%，solidity<0.85 17.61%，邻接间隙 ≤2px 30.95%，低 Lab 对比 11.46%。
- 严格细读 V5 实现：citrus_e_v5.py（mask_route/context/region 分支）、citrus_e_v5_loss.py（可见掩膜并集前景、5px 膨胀环、均衡 BCE + 分离原型对比）、batch runner（双数据路径、双 trainer）、eval_citrus_e_v5.py（四模式合并）。mask_route 的残差从 0.01 tanh 起步、region 为训练专用——V5 代码本身没有结构性错误，问题在收益不够。

## 3. V5 官方整图指标（百分数，单种子 42）

| run | 最佳 Mask AP50–95 | 同轮 Mask AP50 | best epoch | 末轮 AP50–95 |
|---|---:|---:|---:|---:|
| V5_00 control | 67.653 | 83.891 | 220 | 67.017 |
| V5_01 fine | **68.125** | 84.266 | 175 | 66.890 |
| V5_02 mask_route | 67.641 | 83.150 | 210 | 66.991 |
| V5_03 region | 67.911 | 83.125 | 254 | 67.212 |
| V5_04 fine_route | 67.754 | 83.726 | 241 | 66.402 |
| V5_07 full | 67.604 | 83.121 | 277 | 67.017 |

历史参考（各自协议，不是公平总榜）：E30 控制 68.165，V4R05 68.080，G10 67.681。旧数据/AMP=1 的 YOLO11n 约 62.1 不可作为当前结构涨点的直接对照；SAGE V4R 的同期官方对照为 66.719。

## 4. 配对诊断的共同图景（同权重、同验证集、公共 640 栅格）

- 以复审表的同权重对照为准：V5_00 的 coarse→fine trustedmask，公共栅格 AP50–95 为 71.082→72.424，极小果匹配 72/161→93/161；这是观测差异，不是理论上限。
- 切片候选带来 341–437 背景误检（整图 130）与 14–25 重复；合并代理 15–23。低置信度长尾有效（conf .001 最大召回 ~96%），即相当一部分 GT 只以低分候选存在。
- 分辨率与误检是同一条链的两端：细视图恢复小目标证据，同时暴露更多纹理背景。

## 5. 论文与本地代码的取舍

| 来源 | 借鉴 | 不照搬 |
|---|---|---|
| [RefineMask, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Zhang_RefineMask_Towards_High-Quality_Instance_Segmentation_With_Fine-Grained_Features_CVPR_2021_paper.pdf) | 细粒度特征逐级改善实例掩膜 | 其多级 RoI 细化；本实现只做一次原型级残差，无逐实例循环 |
| [Mask Transfiner, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/papers/Ke_Mask_Transfiner_for_High-Quality_Instance_Segmentation_CVPR_2022_paper.pdf) | "掩膜误差集中在下采样丢失信息的少数像素" | 四叉树/query 细化；本实现是密集原型残差，不是 quadtree |
| [NWD, TGRS 2022](https://arxiv.org/abs/2110.13389) | 小目标用高斯距离替代 IoU 做回归 | 已在代码库中实现为尺寸门控混合（loss.py:139-159），只改系数，不改 TAL 分配 |
| [Simple Copy-Paste, CVPR 2021](https://openaccess.thecvf.com/content/CVPR2021/papers/Ghiasi_Simple_Copy-Paste_Is_a_Strong_Data_Augmentation_Method_for_Instance_CVPR_2021_paper.pdf) | 实例级粘贴扩充组合 | 跨图粘贴依赖外部上下文建模；本实现用 ultralytics 内置 flip 模式（同图翻转、IoA<0.3 门控） |

SAHI/SNIPER/ReCo/ConDSeg 的取舍沿用 V5 文档记录，不在 V6 重新声明。

## 6. V6 实际改动

- `ultralytics/nn/modules/citrus_e_v6.py`：`EV6FineDetail`（P1 茎特征 → 16 通道 → 与 160² detail 估计逐像素门控）与 `SegmentCitrusEV6`（在 V5 forward 的 proto 注入之后追加 stride-2 残差，`fine_scale` 0.01 初始化，`tanh` 有界）。`fine_mask=False` 时图、参数、state-dict 键与 V5 控制完全一致。
- `ultralytics/models/yolo/segment/val.py`：postprocess 与 `_prepare_batch` 原先硬编码 `4×proto`/`s//4`；改为读头部 `proto_stride`（默认 4，fine=2）。对全部既有模型行为不变；不改则 320² proto 验证崩溃。
- `citrus_e_v6_suite.py` + `0_orange_yaml/E_V6_series/V6_00..07.yaml`：fine 臂头部输入为 `[16,19,10,2,0]`（第五输入 = stem stride-2 特征），其余臂 `[16,19,10,2]`。
- `20260911_citrus_e_v6_batch.py`：沿用 V5 顺序队列、seed 随机序、paired 评估、协议快照与源码哈希；新增 `RUN_OVERRIDES` 逐臂下发 mask_ratio/nwd_ratio/copy_paste。
- 初版复用 EV5SegmentationLoss；当前 V6 使用 EV6SegmentationLoss，把质量监督与主掩膜损失的栅格对齐。region 分支仍全部关闭。

## 7. 参数与成本（构建实测，nc=1）

| 臂 | 参数 | THOP GFLOPs@640 | proto | proto_stride |
|---|---:|---:|---|---|
| V6_00/01/03/04 | 2,226,247 | 10.045 | 160² | 4 |
| V6_02/05/06/07 | 2,227,304 (+0.05%) | 10.268 (+2.2%) | 320² | 2 |

细通路新增约 1,057 参数；CPU 单前向中位 105→143ms（interpolate 与逐实例 decode 不计入 THOP）。不声称部署加速；GPU 延迟须实测。

## 8. 协议与验收

- 数据/训练协议 = `protocols/citrus_e_v6.yaml`（继承 `citrus_paper1_formal_v2_ram.yaml`）：300ep、AdamW、batch16、imgsz640、AMP=False、seed42 筛选；八种臂同一 `_prepared_multiscale` 数据。
- 逐臂差异只有 `mask_ratio`/`nwd_ratio`/`copy_paste` 三个 hyp 与 fine_mask 结构开关；其余 loss 显式清零，与 V5 相同。
- 配对评估沿用 `eval_citrus_e_v5.py`（0.6/0.4 两档，四模式合并）。
- 当前应先比较共同验证栅格下的 V6 内部对照，再用公共长边 640 配对评估与 V5 比较；V6_01 vs V6_02 分离监督栅格与细节通路。任何单 seed 的差异都不能代替重复实验，不设置任意 0.3pt 显著性门槛。
- 已知风险：mask_ratio=2 使 GT 掩膜张量与逐实例 loss 内存 ×4；copy_paste flip 模式只复制同图实例，多样性增益可能弱于跨图版本；fine 通路对 3072² 原图裁剪视图的收益可能大于对整图的收益（细视图里目标本就更大），需用 paired 模式分桶复核。
- 不做：不再加 P2 稠密检测塔（SAGE72 已证负）；不合并 R/C 进 V6 基线；不改 TAL 分配器本身。
