# Citrus-Light v3：依据完整 G0830 结果的轻量结构重设计

日期：2026-08-31  
范围：RGB 未成熟柑橘实例分割；不引入 Mamba、RGB-D、OBB、机械臂控制或多任务位姿头。

## 结论先行

完整 G0830 结果不支持继续堆叠双流、频率和注意力模块，但它也没有证明所有新主干都无效：

- G02 的双流主干只取得很小的 Mask AP50-95 峰值优势，却损失 AP50、召回率和训练速度，证据不足。
- G03 的频率颈部相对 G02 明显掉点且更慢，应停止。
- G04 相对 G03 只替换深层 P4/P5 的 C3k2 为 RepMixer，Mask AP50-95 提升 0.884 pt，训练时间减少约 32%。因此 RepMixer 仍值得做一次干净隔离；失败的是 G03/G04 所在的双流+频率整体支架，而不能把 G04 的主干贡献一并否定。
- 所有模型的最好点集中在约 54--87 epoch，300 epoch 末值普遍下降。今后必须先做 50 epoch 筛选，再做 100 epoch 晋级，不能盲目把全部模型跑满 300 epoch。

Light v3 因此采用“两种深层主干 × 两种颈部”的受控结构矩阵，并把轻量头、质量排序和损失函数放到后续独立队列。当前只有构建/复杂度证据，没有服务器精度结果，不能提前声称涨点。

## 完整 G0830 数据审计

五个 `results.csv` 均有 300 行。以下“峰值”是各模型自身最高 Mask mAP50-95 所在 epoch；比较时必须同时看同期 AP50/P/R、最好 fitness 和末轮退化。

| 模型 | 峰值 epoch | Mask AP50-95 | Mask AP50 | Mask P | Mask R | 第 300 轮 | 训练时长 |
|---|---:|---:|---:|---:|---:|---:|---:|
| G00 official control | 87 | 0.67031 | 0.83250 | 0.91066 | 0.75790 | 0.65383 | 1.80 h |
| G01 T04 anchor | 54 | 0.66864 | 0.82541 | 0.92469 | 0.73159 | 0.65567 | 2.76 h |
| G02 bilateral backbone | 86 | **0.67241** | 0.82051 | 0.90839 | 0.74207 | 0.65115 | 4.10 h |
| G03 frequency neck | 56 | 0.65631 | 0.81385 | 0.90824 | 0.73602 | 0.62334 | 4.43 h |
| G04 deep RepMixer | 86 | 0.66515 | 0.82282 | 0.92309 | 0.72784 | 0.65541 | 3.00 h |

### 可以下的结论

1. G02 相对 G00 的峰值 Mask AP50-95 仅 +0.21 pt，但同期 AP50 -1.20 pt、Mask R -1.58 pt，训练时间超过 2.2 倍。这不构成“明显优于基线”的证据。
2. G03 相对 G02 为 -1.61 pt，且更慢，频率颈部在当前实现和协议下为负贡献。
3. G04 相对 G03 为 +0.884 pt，并把训练时长从 4.43 h 降到 3.00 h。这是本轮最清晰的结构信号，但需要在官方 PAN 上单独复验才能确认因果。
4. 最后一轮不是最佳模型。正式评估必须使用 `best.pt`；300 epoch 末值不能代替峰值，也不能用各模型不同 epoch 的峰值做显著性结论。

### 不能下的结论

- G00 的 `citrus_boundary/citrus_query=0`，G01--G04 为 `0.15/0.03`，所以 G00 与其余模型不是纯结构对照。
- 单 seed 的 0.2--0.9 pt 差异可能处于随机波动范围。最终方法和基线必须用相同 group-aware split 做 3 个 seed。
- 当前平铺的 G0830 结果目录没有权重与 PR 原始数组，不能从图片外观推断某个模块修复了召回尾部。

## Light v3 结构矩阵

所有结构消融统一保留官方 P2/P3、SPPF、C2PSA、P2--P5 四尺度预测与标准 Segment head。之前 Light 草稿把 C2PSA 替换为 Identity，造成了隐藏变量；v3 已恢复。

| 模型 | 深层 P4/P5 | 颈部 | 头部 | 用途 |
|---|---|---|---|---|
| Light00 | `CitrusLightStage`（PConv） | 官方 PAN | Segment | PConv 主干独立效应 |
| Light01 | 官方 C3k2 | `CitrusLightAFPN` | Segment | 新颈部独立效应 |
| Light02 | PConv | Light-AFPN | Segment | PConv × AFPN 交互 |
| Light05 | `CitrusRepMixerStage` | 官方 PAN | Segment | G04 主干的干净隔离 |
| Light06 | RepMixer | Light-AFPN | Segment | RepMixer × AFPN 交互 |
| Light03 | PConv | Light-AFPN | `SegmentCitrusLite` | 激进轻量部署候选 |
| Light04 | 与 Light03 相同 | 与 Light03 相同 | 轻量头 + mask 质量分支 | 排序校准独立检验 |
| Light07 | RepMixer | 官方 PAN | `SegmentCitrusLite` | 保守精度/速度候选 |

### 主干设计边界

- 不替换浅层 P2/P3，避免极小果实在首次几次降采样中丢失，也保持官方权重迁移。
- PConv 路线只在四分之一通道做 3×3 空间混合，其余通道保留旁路，目标是降低冗余空间计算和内存访问。
- RepMixer 路线来自 G04 的正向局部证据，但 CPU 测速并不快，所以只作为“精度保守”候选，不强行包装成最快方案。
- 两条路线均保留 SPPF+C2PSA，防止把上下文能力的删除误计为主干收益。

### 颈部设计边界

- Light-AFPN 删除 PAN 中重复的 `Concat + C3k2`，只做相邻尺度的渐进 gather/distribute。
- 每个融合节点从接近恒等的目标尺度开始，以小残差注入相邻尺度，避免连续固定平均衰减弱小目标响应。
- 没有再加入频率、LSKA、Mamba、CARAFE、DCN 或额外高分辨率常驻支路。

## PR 末端问题的处理边界

Ultralytics 的 AP 实现会在最大可达 recall 后追加 `precision=0`，再把曲线补到 recall=1。因此 PR 图右侧的垂直跳崖本身是积分哨兵，不是可以通过改画图代码“修好”的模型错误。真正的问题是最大可达 Mask recall 约 0.88，以及逼近该上限时出现的低置信度叶片/果实假阳性。

PR 队列固定 Light03 结构，仅改变一个训练因素：

| 运行 | 变化 |
|---|---|
| LightP00 | BCE 控制 |
| LightP01 | `citrus_vfl=0.25`，检验 IoU 感知排序 |
| LightP02 | `nwd_ratio=0.25`，检验超小框定位鲁棒性 |
| LightP03 | VFL + NWD 交互 |
| LightP04 | 显式 mask-IoU 质量分支 |

评价时使用 `analyze_citrus_pr.py` 输出 recall ceiling、有效区间 precision、最佳 F1 阈值和原始 TP/FP/FN；不能只比较曲线是否在 recall=1 贴零。

## 静态复杂度与本机 CPU 筛查

环境：CPU 单线程，320×320 输入，5 次 warmup，3 组×20 次前向取中位数；GFLOPs 按 640×640 静态图计算。该延迟只用于同机相对筛选，论文最终延迟必须在目标服务器空闲 GPU 上重测。

| 模型 | Params (M) | GFLOPs @640 | CPU latency @320 (ms) | 相对控制观察 |
|---|---:|---:|---:|---|
| YOLO11n control | 2.8768 | 10.529 | 50.634 | 控制 |
| Light00 | 2.6555 | 10.243 | 51.333 | 参数略降，CPU 未加速 |
| Light01 | 2.2640 | 9.655 | 48.286 | 参数 -21.3%，小幅加速 |
| Light02 | 2.0427 | 9.368 | 47.053 | 参数 -29.0%，GFLOPs -11.0% |
| **Light03** | **1.9543** | **8.458** | **40.821** | **参数 -32.1%，GFLOPs -19.7%，CPU 约 -19.4%** |
| Light04 | 1.9847 | 8.529 | 47.647 | 质量分支带来实际延迟成本 |
| Light05 | 2.6137 | 10.192 | 52.860 | RepMixer 并非 CPU 快速算子 |
| Light06 | 2.0009 | 9.318 | 49.003 | 参数低，但实际速度优势有限 |
| Light07 | 2.5253 | 9.281 | 48.329 | 保守候选，轻量幅度小于 Light03 |

这张表只证明工程复杂度，不证明精度。Light03 是速度候选，Light07 是受 G04 支持的精度保守候选；两者必须训练后再比较 Pareto 前沿。

## 文献与开源代码依据

本轮继承并核验的主要结构原则如下；完整检索记录位于 `3_研究生/paper1_finalization_20260830/sources/light_architecture_search_20260830.md`。

| 来源 | 采用的原则 | 没有照搬的部分 |
|---|---|---|
| [FasterNet, CVPR 2023](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html) / [official code](https://github.com/JierunChen/FasterNet) | 部分通道空间卷积、关注真实访存和延迟 | 未整骨干替换浅层特征 |
| [AFPN](https://arxiv.org/abs/2306.15988) / [official code](https://github.com/gyyang23/AFPN) | 相邻尺度渐进融合、缓解尺度冲突 | 未复制完整重型 AFPN |
| [Gold-YOLO, NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/hash/a0673542a242759ea637972f053b2e0b-Abstract-Conference.html) / [official code](https://github.com/huawei-noah/Efficient-Computing/tree/master/Detection/Gold-YOLO) | gather/distribute 拓扑 | 未引入大型全局融合块 |
| [DAMO-YOLO](https://arxiv.org/abs/2211.15444) / [official code](https://github.com/tinyvision/DAMO-YOLO) | 把容量放在有效融合，减少重复预测开销 | 未照搬重型 RepGFPN |
| [QueryDet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) | 高分辨率小目标计算应避免全图常驻浪费 | 本轮未加入稀疏 ROI 推理支路 |

## 验证状态

- 8/8 Light YAML 可通过标准 `YOLO(yaml, task="segment")` 入口构建。
- 8/8 可加载 `yolo11n-seg.pt` 的形状兼容权重并完成前向。
- LightStage、Light-AFPN、轻量头、质量分支、VFL/NWD 损失路径均完成反向传播检查。
- Light/G0830/协议/PR 诊断联合测试：`42 passed`。
- Light03：1.9543M 参数、8.458 GFLOPs，低于测试门槛 2.2M/9.5 GFLOPs。
- 本机没有生成伪造的训练精度；服务器必须先完成 3 epoch smoke 和 50 epoch 筛选。

## 推荐决策顺序

1. 运行 `--suite smoke --epochs 3`，确认 8 个模型在服务器环境无异常。
2. 运行 `--suite screen --epochs 50`，先判断 PConv、RepMixer、AFPN 的独立贡献。
3. 仅当 Light00/01/02/05/06 中至少一条结构路线接近或超过同协议基线，才运行 `--suite pareto --epochs 50`。
4. 仅当 Light03 精度有竞争力时运行 PR 队列，否则不要在失败结构上继续调损失。
5. 晋级模型跑 100 epoch；最终模型与 YOLO11n 同协议 3×300 epoch，并报告均值±标准差、APtiny/APsmall、solidity/凹度挑战子集、邻接 gap、split/merge error、Params、GFLOPs 和实测延迟。
