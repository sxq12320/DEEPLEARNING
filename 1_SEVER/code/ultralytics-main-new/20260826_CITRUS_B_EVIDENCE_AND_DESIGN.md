# CitrusB v2：S 系列全量结果驱动的重设计

更新日期：2026-08-27。实现以 `0_orange_yaml/20260826_citrus_b/` 和自动化测试为准。

## 结论

S00-S09 完成后，原 CitrusB 方案必须收缩。旧 B02-B09 的长驻 stride-4 双流、反复 PagFM、跨层细节
注入和 mask-quality 分数乘法没有本地正证据，并把本机 CPU 推理延迟推到基线约 2 倍。新版删除这些
路径，只保留三个可由 S 结果解释的因素，并用完整因子组合识别交互：

1. 主干：`SPPFRepContext`。S01 提高 AP50、选定点 recall 和可支持召回上限，但单次训练尾段不稳定，
   因此仅作为召回因素，不直接宣称有效。
2. 颈部：只在 top-down P3 融合点使用 `CitrusScaleFusion`，其初始函数等于普通 Concat。S02 已排除
   LSKA 的独立贡献，因此 S09 的严格 mask AP 收益应拆分验证尺度融合与拓扑头，而不是整体照搬。
3. 头部：以 S04 的轻量预测头为共同底座。`SegmentCitrusBLite` 测试推理期 P2→prototype 拓扑细化；
   `SegmentCitrusLiteBQ` 只在训练期施加 boundary/query 监督，部署时完全走 S04 的 P3-P5 路径。

新版继续保留完整 PAN。S05 删除 bottom-up PAN 后，Mask AP50 相对 S00 下降 0.0140，已经构成明确
反证。新版不使用 LSKA；S02 和 S07 分别说明它独立及与非对称颈部组合时没有正收益。新版架构筛选也
不启用 contrast、exclusive、mask-quality、VFL、NWD；S08 表明全结构与全损失堆叠不会自动叠加，
未被单独验证的损失只能进入后续 50-epoch 损失筛选。

## S 系列数据结论

以下为每个 `results.csv` 中 Mask AP50-95 峰值所在 epoch，均来自同一 grouped-dedup 数据和训练协议。
单次运行差值小于 0.003 不视为确定证据。

| 模型 | Epochs | Mask AP50 | Mask AP50-95 | P | R | 相对 S00 ΔAP | 稳定尾段 AP | 决策 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| S00 baseline | 300 | 0.7859 | 0.6074 | 0.8663 | 0.7138 | +0.0000 | 0.5996 | 基线 |
| S01 RepContext | 217 | 0.7894 | 0.6124 | 0.8588 | 0.7265 | +0.0050 | 0.5976 | 保留为召回因素 |
| S02 LSKA | 300 | 0.7791 | 0.6074 | 0.8885 | 0.7020 | -0.0000 | 0.5979 | 淘汰 |
| S03 train aux | 273 | 0.7851 | 0.6115 | 0.8573 | 0.7163 | +0.0041 | 0.6040 | 只保留 B/Q 思路 |
| S04 lite head | 300 | 0.7899 | 0.6150 | 0.8974 | 0.7155 | +0.0076 | 0.6040 | 当前 Pareto 支点 |
| S05 FPN-only | 300 | 0.7719 | 0.6022 | 0.8917 | 0.6975 | -0.0052 | 0.5995 | 明确淘汰 |
| S06 asym PAN | 300 | 0.7835 | 0.6135 | 0.8504 | 0.7222 | +0.0061 | 0.5985 | 不进最终结构 |
| S07 LSKA + asym | 300 | 0.7762 | 0.6051 | 0.8589 | 0.7062 | -0.0023 | 0.5954 | 淘汰 |
| S08 full stack | 300 | 0.7872 | 0.6122 | 0.8819 | 0.7142 | +0.0048 | 0.6032 | 低于 S04/S09，不推广 |
| S09 dense topology | 256 | 0.7843 | 0.6162 | 0.9143 | 0.6868 | +0.0088 | 0.6068 | 严格 AP 对照，召回偏低 |

用当前代码重新验证 best.pt 后，S01 的 `mask_recall_ceiling=0.8874` 为五个关键模型最高；S04 在
R=0.80 时 precision=0.5628 为关键模型最高；S09 的选定点 precision 高但 recall 低。所有官方 PR 图
末端的 `(R=1,P=0)` 都包含 COCO AP 插值哨兵点，不是一个真实置信度阈值。真正需要改进的是低置信
候选的误检排序和约 0.85-0.89 的候选召回上限，不能通过删掉绘图端点解决。

完整原始统计位于：
`1_SEVER/results/CITRUS_SWIFT_ALL_300EP/CITRUS_SWIFT_SUMMARY.md` 和
`_pr_supported_diagnostic/pr_summary.json`。

## 新 B 系列

| ID | 受控因素 | Params | GFLOPs@640 | 本机 CPU median | 训练损失 |
|---|---|---:|---:|---:|---|
| B00 | 原始 YOLO11n-seg | 2.843M | 10.36 | 152.30 ms | 官方 |
| B01 | Lite | 2.747M | 9.44 | 139.49 ms | 官方 |
| B02 | RepContext + Lite | 2.763M | 9.45 | 141.06 ms | 官方 |
| B03 | ScaleFusion + Lite | 2.747M | 9.44 | 144.13 ms | 官方 |
| B04 | Topology refine + Lite | 2.762M | 9.81 | 165.93 ms | boundary 0.25 + query 0.05 |
| B05 | RepContext + ScaleFusion + Lite | 2.763M | 9.45 | 143.75 ms | 官方 |
| B06 | RepContext + topology refine + Lite | 2.778M | 9.82 | 171.24 ms | boundary + query |
| B07 | ScaleFusion + topology refine + Lite | 2.762M | 9.81 | 166.15 ms | boundary + query |
| B08 | 三因素 + 推理期 topology refine | 2.778M | 9.82 | 166.62 ms | boundary + query |
| B09 | RepContext + ScaleFusion + 仅训练 B/Q | **2.697M** | **9.45** | **147.43 ms** | boundary + query |

这些延迟仅用于同机筛选。B09 比 B00 少 5.1% 参数、少 8.7% GFLOPs，本机 CPU median 快约 3.2%；
B08 用于判断推理期拓扑细化带来的严格 AP 是否足以抵消约 19 ms 的额外延迟和潜在召回损失。B09
目前只是“推荐筛选候选”，不是已经证明优于基线的最终模型。

## 文献与代码边界

- [RepVGG](https://openaccess.thecvf.com/content/CVPR2021/html/Ding_RepVGG_Making_VGG-Style_ConvNets_Great_Again_CVPR_2021_paper.html)：
  支持训练多分支、部署融合的上下文卷积；实现直接复用 Ultralytics `RepVGGDW`。
- [QueryDet](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)：
  支持高分辨率微小候选监督，但新版不构建昂贵 P2 检测头。
- [Boundary-preserving Mask R-CNN](https://arxiv.org/abs/2007.08921)：支持边界特征辅助实例掩膜；B08
  测推理融合，B09 测仅训练监督。
- [Lite-HRNet](https://openaccess.thecvf.com/content/CVPR2021/html/Yu_Lite-HRNet_A_Lightweight_High-Resolution_Network_CVPR_2021_paper.html)、
  [PIDNet](https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html)
  仍是高分辨率/细节分支证据来源，但本地 S 结果不足以支持旧 B 的长驻双流，故已从主试验移除。
- Mask Scoring、VFL 和 NWD 仅保留为后续独立损失/排序筛选，不进入新版架构默认配置。特别是 S09
  已呈现高 precision、低 recall，未经消融就再乘 mask-quality 分数可能进一步压低低置信真阳性。

对应开源仓库已固定在 `C:\Users\33836\Desktop\github`。上述引用只说明设计来源；能否改善本柑橘
任务必须由 B 系列和挑战子集实验验证。

## 服务器执行顺序

```bash
cd /服务器路径/code/ultralytics-main-new
pip install -e .

# 1. 预演队列
python 20260826_citrus_b_batch.py --data /清洗数据/data.yaml --suite smoke --epochs 3 --dry-run

# 2. B02-B09 三 epoch 冒烟；B00/B01 已由同协议 S00/S04 覆盖
python 20260826_citrus_b_batch.py \
  --data /清洗数据/data.yaml --suite smoke --epochs 3 \
  --batch 4 --workers 4 --device 0 --project 1_results/CITRUS_B_V2_SMOKE

# 3. 冒烟全部正常后做含 B00/B01 同周期基线的 50 epoch 结构筛选
nohup python 20260826_citrus_b_batch.py \
  --data /清洗数据/data.yaml --suite screening --epochs 50 \
  --batch 16 --workers 4 --device 0 --project 1_results/CITRUS_B_V2_SCREEN_50EP \
  > citrus_b_v2_screen.log 2>&1 &

# 4. 汇总
python report_citrus_b_results.py --project 1_results/CITRUS_B_V2_SCREEN_50EP
```

50-epoch 新模型不能直接与 300-epoch S04 数值比较，所以该筛选会重跑 B00/B01 作为同周期参考。只有
同时超过 50-epoch B01、没有明显降低 Mask AP50/recall、且 GPU 延迟合格的 1-2 个结构，才进入
300 epoch；到 300 epoch 后再与 S04 的 0.6150 比较。最终基线和入选模型用 seed 42/43/44 重复。

正式论文评价还需在固定 test split 上补充 AP-small、Boundary F1、solidity/凸包缺损子集、邻近实例
gap 子集、split/merge 错误和图内尺度比子集。默认 `results.csv` 不包含这些指标，不能仅凭总 mAP 声称
解决了条带遮挡或粘连分离。
