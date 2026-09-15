# E V4 系列：三波递进改进（控制论颈部 → 全维度 → 范式重构）

> **2026-09-09 代码复核说明：**保留本系列现有 26 个网络，不新增结构。已修复预训练层号映射、
> DySample 坐标轴、NWD 零距离梯度及 SMC/SMCAO 的计时/AMP/断点状态问题；优化 CARAFE、LUT、
> 上下文分支和辅助标签计算。详细证据见 `../../docs/E_V4_CODE_REVIEW.md`。
> 下文的研究动机、历史归因和涨点预期均属于**待检验假设**，不是本系列训练结论。
> 恒等初始化只保证新增残差模块本身，不保证 concat 被替换后整个网络与对照等价。
> 修复前后的同名实验请使用不同运行名；不能直接续训旧实验后当作完整的修复版实验。

> 创建日期：2026-09-09
> 模块实现：`ultralytics/nn/modules/citrus_e_v4.py`
> 分割头：沿用 `SegmentCitrusSAGEV5`（与 E9/E_V2/E_V3 协议一致，relay=true；E52/E53 除外）
> 基座：E04/E05 已验证的 hybrid 主干（SAGEV8PhaseStem + C3k2_Faster + C3k2_WT + SPPF_LSKA + C2PSA）
> 三波结构：第一波 E40–E49 控制论颈部校正；第二波 E50–E64 主干/头部/速度/损失/优化器全维度；第三波 E70–E77 伪装检测与尺度空间范式重构。

---

## 一、痛点依据（来自 results 全量分析）

| 痛点 | 量化证据 | 来源 |
|---|---|---|
| 召回是瓶颈 | P≈0.91–0.95 而 R≈0.74–0.77；R>0.80 后 P 快速下降 | 总表 20260902 |
| 小目标漏检 | 53.3% 实例为 COCO-small，17.4% 最短边 <16px；tiny recall 全局仅 0.16–0.21 | _analysis/dataset_difficulty |
| 粘连/凹形遮挡 | 31% 实例 2px 内有邻居，17.6% 强凹形掩膜；去 PAN 回传 -0.52pt | S05 消融 |
| 轻量-精度矛盾 | T05 以 -8.6% FLOPs 换 -0.5pt；G03 频域颈 -1.4pt | T/G0830 复盘 |

**失败教训（E V4 的设计约束）：**
1. E_V2 重参数化主干 + 全局 context hub 雪崩（-3~-5pt）→ E V4 不动主干、不设 hub，只在校正点做单变量替换。
2. 盲目堆叠（SXQNet -2.95pt、F53 -1.68pt、B08 -0.64pt）→ 每个变体相对 E40 只改一处；组合变体（E47/E48）用于检验协同/反噬。
3. Full 多损失策略 -2.3~-3.7pt → 所有模块不触碰训练目标，纯结构校正。
4. 所有校正模块**零初始化门控、恒等出发**（ReZero 式），从预训练兼容路径起步——继承 E V3 的成功哲学。

## 二、自动控制原理 → 网络结构映射

| 控制论概念 | 机理 | 模块 | 对应痛点 |
|---|---|---|---|
| PID 控制律 u=Kp·e+Ki·∫e+Kd·ė | 比例=语义参考校正；积分=多尺度低通上下文累积（消稳态误差）；微分=细流高通差分（超前边界变化） | `EV4PIDFusion` | 小果漏检（I）+ 粘连分离（D） |
| Luenberger 状态观测器 | x̂⁺=x̂+λ·L⊙(y−x̂)，深层语义作为观测修正浅层状态，L 有界可学 | `EV4ObserverGate` | P3 局部证据不足导致的漏检 |
| 相位超前补偿 (1+aTs)/(1+Ts) | 微分型高频补偿 + 滞后极点阻尼防过冲 | `EV4PhaseLead` | 边界锐利化（近零参数） |
| 积分环节 ∫e·dt | 多尺度全局上下文累积回注（P5 处） | `EV4IntegralContext` | 系统性小目标漏检 |
| 抗混叠滤波/传递函数 | 语义先低通再上采样（无频谱拷贝）+ 语义门控高频注入；单点使用（吸取 G03 全颈替换失败教训） | `EV4FilterFuse` | 上采样伪影 + 边界一致性 |
| 串级控制 | 外环（慢）粗校语义设定值 + 内环（快）局部精校剩余误差 | `EV4CascadeRefine` | 语义冲刷细节的稳定性 |

注：均为特征空间类比而非字面控制器；docstring 已如实声明。`ct_modules.py` 中已有 Kalman/ESO/IDAPBC（RGB-D 融合专用），E V4 为 RGB 单模态新设计，不重复。

## 三、消融矩阵（相对 E40 单变量）

| 模型 | 改动 | Params | GFLOPs@640（≈） | 检验假设 |
|---|---|---|---|---|
| E40_control | 对照：hybrid 主干 + 标准 concat 颈 | 2.480M | ~10.4 | 协议基准 |
| E41_pid_fusion | 两处 top-down concat → EV4PIDFusion | 2.557M | ~11.0 | PID 三支校正有效 |
| E42_observer | P3 后加 EV4ObserverGate(P5→P3) | 2.504M | ~10.5 | 观测器修正提升召回 |
| E43_phase_lead | P3 后加 EV4PhaseLead | 2.480M | ~10.4 | 近零成本边界增强 |
| E44_integral_context | SPPF-LSKA 后加 EV4IntegralContext | 2.677M | ~10.5 | 积分上下文补小果 |
| E45_filter_fuse | upsample+concat → EV4FilterFuse ×2 | 2.522M | ~10.5 | 抗混叠单点融合 |
| E46_cascade | P3 后加 EV4CascadeRefine(P4'→P3) | 2.489M | ~10.4 | 双环稳定校正 |
| E47_pid_observer | E41 + E42 | 2.582M | ~11.1 | 协同检验 |
| E48_full | E41 + E42 + E43 | 2.582M | ~11.2 | 堆叠是否反噬 |
| E49_lite | PID 融合 + 颈部 C3k2_Faster 单重复 | 2.516M | ~10.9 | 对标 T05 轻量冠军 |

（GFLOPs@640 由 256 输入实测 ×6.25 估算；全部 10 个模型已通过构建/前向/反向/FLOPs 冒烟验证，CPU torch 2.8。）

## 四、第二波：全维度改进（主干 / 头部 / 速度 / 损失 / 优化器）

第一波（E40–E49）只动颈部。第二波把改进空间打开到五个维度，全部沿用"单变量、证据驱动、恒等出发"约束；每个变体都标注了历史证据来源。

### 4.1 结构与速度维度（新 YAML，已通过冒烟验证）

| 模型 | 维度 | 改动 | Params | GFLOPs@640（≈） | 证据/假设 |
|---|---|---|---|---|---|
| E50_hwdown_backbone | 主干 | 3 个 stride 卷积 → HWDown（Haar 小波下采样） | 2.213M | ~9.7 | F04 下采样最优（+4.44%），保小目标边缘高频，且更轻更快 |
| E51_moce_deep | 主干 | 仅最深层 C3k2_WT → C3k2_MoCE 专家路由 | 2.436M | ~10.3 | F64 效率王（2.67M/0.67178）；只动深层避免整体替换失败 |
| E52_topo_head | 头部 | SegmentCitrusTopo 拓扑头（[P2,P3,P4,P5]） | 2.487M | ~10.6 | T04 冠军路线；**用基线协议训练以分离头结构 vs 辅助损失归因** |
| E53_quality_head | 头部 | E41 + SegmentCitrusEV3Quality 掩膜质量排序校准 | 2.566M | ~11.1 | 新增独立的 mask-IoU 质量监督；输入特征与目标 detach，不直接反传至主分割特征 |
| E54_detail32 | 头部 | SAGEV5 relay 细节通道 16→32 | 2.487M | ~10.5 | 小目标几何容量的单变量头部消融 |
| E55_dysample | 速度/颈部 | 上采样 → DySample | 2.504M | ~10.4 | F17 证明学习上采样值钱；DySample 是廉价的 CARAFE |
| E56_carafe | 颈部 | 上采样 → CARAFE | 2.620M | ~10.6 | F17 证据（+5.09%）直接迁移；与 E55 构成精度/延迟对 |
| E57_slim_proto | 速度/头部 | 掩膜原型通道 256→128 | 2.420M | ~8.0 | 原型分支跑在 P3 分辨率最吃延迟；mAP 若守住就是白捡 -23% 计算 |

### 4.2 损失函数维度（复用 E40/E52 YAML，只改训练参数——单变量消融）

Full 多损失策略曾 -2.3~-3.7pt（梯度冲突），但项目判定总表明确：**保持 Baseline 协议、对单个损失做单变量消融**是合规路径。以下损失全部已在 `ultralytics/utils/loss.py` 接线，零新代码：

| 实验 | YAML | 训练参数 | 机理 | 针对痛点 |
|---|---|---|---|---|
| E58_nwd | E40_control | `--nwd-ratio 0.5` | NWD/CIoU 混合；sigmoid 尺度权重偏向 <32px，非硬截断。IoU 对小框偏移敏感，并非数学上不连续 | 小目标回归 |
| E59_vfl | E40_control | `--citrus-vfl 0.5` | 分类分数与质量混合（Varifocal 思想） | P/R 失衡（低置信误检/漏检校准） |
| E60_boundary | E52_topo_head | `--citrus-boundary 0.25` | 拓扑头 boundary 辅助监督（BCE+Dice） | 粘连果实边界分离 |
| E61_query | E52_topo_head | `--citrus-query 0.10` | 拓扑头 query 辅助 focal 监督 | 凹形遮挡果实完整性 |
| E62_topo_full | E52_topo_head | `--citrus-boundary 0.25 --citrus-query 0.10` | T04 完整配方复现 | 与 E52/E60/E61 构成归因矩阵 |

归因逻辑：E52（无辅助损失）vs E60/E61（单损失）vs E62（双损失）→ 头结构贡献与损失贡献完全可分。

### 4.3 优化器维度（滑模控制调度器，trainer 已接线）

trainer 支持 `optimizer=SMC`（滑模面停滞检测 + 限时噪声逃逸）和 `optimizer=SMCAO`（V2.2 四机制逃逸），包装 AdamW 使用——**这本身就是自控原理（滑模控制）在优化器上的落地**。
使用 `train_citrus_yaml.py --optimizer SMC` 或 `--optimizer SMCAO` 显式开展优化器消融；其余固定参数不变，有效优化器写入该运行的协议 JSON 与 args.yaml。普通结构比较仍用 AdamW。控制理论名称不构成收敛性或逃离局部极值的证明。

| 实验 | YAML | 改动 | 假设 |
|---|---|---|---|
| E63_smc | E41_pid_fusion | optimizer=SMC（协议副本） | 小数据（941 张）易陷局部极小，滑模逃逸提升最终收敛点 |
| E64_smcao | E41_pid_fusion | optimizer=SMCAO（协议副本） | 抖动+负阻尼注入进一步逃离浅极小 |

### 4.4 执行优先级建议

1. **第一批（50ep 筛选）**：E41、E42、E44、E50、E51、E55、E57——覆盖颈/主干/速度，证据最强。
2. **第二批**：E43（近零参数）、E45、E46、E52、E53、E54、E56。
3. **损失/优化器维度**：等结构冠军产生后，在冠军结构上做 E58–E64（避免结构与损失混淆）。
4. E47/E48/E49 用于检验协同与堆叠反噬，最后跑。

## 五、第三波：范式重构（伪装目标检测 + 尺度空间，E70–E77）

### 5.1 为什么需要范式层创新

青果-绿叶混淆在文献中的直接对应是**伪装目标检测（Camouflaged Object Detection, COD）**；极端尺度差异对应**尺度空间/混合尺度**范式。2026-09-09 文献调研结论：
- 农业青果论文（GreenFruitDetector、YOLOv7-SAP、MNC-YOLO 等）全部停留在 SE/CBAM/ASPP/P2 头的模块堆叠——**范式层答案不在农业 YOLO 文献里**，必须跨域迁移 COD。
- COD 核心机制：SINet-V2（TPAMI 2022）组反向注意力、PraNet（MICCAI 2020）并行反向注意力、ZoomNet（CVPR 2022）0.5×/1×/1.5× 混合尺度、FEDER（CVPR 2023）可学习频带分解+边缘重建、PlantCamo（CAAI AIR 2025）植物伪装、RISNet 极端农业密集小目标。
- **证据边界**：不能从部分切片结果推导“输入层切片无用，必须改特征层”。切片可提高 tiny recall；整体 AP 还受误检、融合和评估口径影响。输入与特征层处理应分别对照。

### 5.2 单模态 RGB 的"不一样的地方"——三个可利用先验

1. **色彩恒常先验**：果与叶的混淆是色度问题，判别性颜色投影应显式学习（ISP 的 CCM+gamma 角色），而不是指望第一层卷积自己发现。
2. **频带分离先验**（FEDER）：伪装相似性在低频色度带，果实身份（油胞纹理、边缘曲率）在带通结构。
3. **形状二分的先验**：果实=径向对称团块（各向同性），枝叶=各向异性条带——形状是比颜色更稳定的判别线索。

### 5.3 消融矩阵（全部零初始化恒等出发；已过冒烟验证）

| 模型 | 范式 | 改动 | Params | GFLOPs@640（≈） | 文献依据 |
|---|---|---|---|---|---|
| E70_chroma_front | 色彩恒常 | 输入端可学习 ISP 前端（3×3 CCM + 分段线性色调曲线，恒等初始化） | 2.480M | ~10.4 | ISP 色彩管线 |
| E71_lap_front | 频带分离 | 输入端拉普拉斯金字塔带通注入（3→3） | 2.480M | ~10.4 | FEDER (CVPR23) |
| E72_reverse_refine | 反向注意力 | P5 投粗对象图→擦除高置信响应→精修残差 | 2.485M | ~10.4 | SINet-V2 / PraNet |
| E73_radial_vote | 形状投票 | 条带池化各向同性投票（min-over-orientations）注入 P3 | 2.488M | ~10.5 | Fast Radial Symmetry (TPAMI03) |
| E74_scale_space | 尺度空间 | P3 处稠密 σ 阶梯（3/5/9 模糊）重混合 | 2.496M | ~10.6 | SIFT (IJCV04) / ZoomNet (CVPR22) |
| E75_chroma_reverse | 组合 | E70 + E72（输入侧 + 特征侧伪装处理协同检验） | 2.485M | ~10.4 | — |
| E76_camouflage_full | 组合 | E70 + E72 + E73 伪装三件套（堆叠反噬检验） | 2.493M | ~10.5 | — |
| E77_lap_scale | 组合 | E71 + E74（输入 + 特征双层尺度频率处理） | 2.496M | ~10.6 | — |

### 5.4 推理范式候选（暂不写 YAML，先在 suite 层验证）

- **擦除重扫（erase-and-rescan）**：首轮检测→掩膜擦除已检区域→二次前向找漏检伪装果。COD 的迭代擦除思想在推理管线的落地，可复用 E9 slicing 基础设施实现。
- **Camouflageator 式对抗增广**（CVPR 2023 思路）：训练时把果实 copy-paste 到高混淆叶丛区域制造"更难伪装样本"——数据维度，注意 Copy-Paste 曾在 Full 策略中作为多变量之一失败，需单变量验证。

### 5.5 优先级更新

以下仅为待筛选方向，不能按论文名判断收益排序。E70 只有 33 个参数但包含像素级计算，并非零成本；E72 是未直接监督的语义门控，不是已校准对象概率；E73 只有水平/垂直条带统计，不是真正径向对称检测；E74 使用均值滤波，不是高斯尺度空间或完整 ZoomNet。

## 六、训练协议（沿用项目规范）

1. 先做 1–3 epoch smoke run（E40/E41 优先），再 50 epoch 筛选，达标者 300 epoch × seeds 42,43,44。
2. 与 E40 同周期重跑作对照；比较口径：Mask mAP50-95 / mAP50 / P / R + 尺度 AP + tiny recall。
3. 运行命令（主代码目录下）：
   ```bash
   python train_citrus_yaml.py --model 0_orange_yaml/E_V4_series/E41_pid_fusion.yaml --data /your/dataset/data.yaml --device 1 --name E41_pid_fusion_review1
   python eval_citrus_seg.py --weights <run>/weights/best.pt
   ```
4. 记录命令、Git 状态、划分版本、硬件、最终指标；不覆盖已完成运行。
