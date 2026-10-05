# 柑橘网络、数据集与 J 系列的严谨重审

日期：2026-10-01  
结论性质：历史证据复核、当前实现审计、数学性质核验与文献范围检索；不是新的训练结果。

## 一、结论先行

目前最需要纠正的不是“图画得不够复杂”，而是**输入分辨率、候选发现、掩膜表达、排序和评估协议没有被充分拆开归因**。J11 是一个可测试的轻量综合候选，不是已证实的最强模型。数学可以说明某种操作的性质与局限，不能证明它在柑橘数据上必然涨点。

当前更有依据的研究路线是：

1. 保住小果实及小可见片段的有效输入信息，并确认它们真正进入训练分配和推理候选。
2. 用语义与局部结构的联合证据区分果实和叶枝，而不是假设灰度、梯度或差分注意力天然等于果实特征。
3. 对已发现实例精化**可见掩膜**，同时约束叶枝越界与接触果实混合；不强行补圆、不补不可见部分。
4. 以统一指标和端到端延迟选择 Pareto 解，而不是只凭参数量、FLOPs 或历史最高 AP 选“冠军”。

本轮新建了只读审计脚本和报告；没有修改训练架构、批量队列、超参数、基线评估或数据标签，没有启动正式训练。

## 二、历史结果究竟支持什么

沿用并复核本日的全量实验索引：299 个 results.csv，按内容去重为 285 条历史；模型清单有 457 个 YAML 图。它们不是 285 次同协议独立实验。源码对应清单的文件名匹配也不能证明历史训练使用了当前版本源码。

- [全部历史索引](E:/mastercode/1_SEVER/code/ultralytics-main-new/docs/J_SERIES_20261001/evidence/history_unique_runs.csv)
- [同权重切片对照与汇总证据](E:/mastercode/_work/20261001_j/evidence/bounded_evidence.json)

以下均为历史原生指标；AP 列采用该运行 mask AP50–95 峰值所在行，R 也是该行的 R，**不是最大可达召回率**。跨表、跨划分不得直接作因果比较。

| 运行 | 已记录轮数 | Mask AP50 (%) | Mask AP50–95 (%) | Mask R (%) | 主要比较限制 |
|---|---:|---:|---:|---:|---|
| 001_3_yolo11-seg_adamw | 300 | 78.635 | 62.076 | 70.162 | AMP=1，旧划分，val 成员未确认 |
| G10_hybrid | 300 | 83.824 | 67.681 | 76.747 | AMP=False；旧划分；不构成只改架构的控制实验 |
| F14_sppf_lska | 300 | 82.868 | 67.599 | 73.022 | 与最早基线存在协议混杂 |
| S08_swift_full | 300 | 78.719 | 61.220 | 71.419 | 数据成员、初始化和协议须逐项对齐 |
| SAGE42_asym_semantic_detail | 300 | 82.311 | 67.147 | 75.786 | 历史验证成员不等于正式 grouped_dedup |
| I00_replay_recognition | 300 | 84.875 | 72.363 | 75.538 | 仅用于本组同协议内部比较 |
| I05_sync_caa | 300 | 84.725 | 72.477 | 76.962 | 相对 I00 AP +0.114 点，AP50 反而 -0.150 点 |
| I20_corrected_control | 50 | 83.684 | 69.887 | 74.071 | 仅 50 轮，不是完成 300 轮 |
| I23_p2_semantic | 50 | 83.639 | 69.799 | 77.312 | 相对 I20 R 增加，但 AP 并未同步增加 |

由这些结果可以合理说：

- G10、F14 和 SAGE42 的一些路径值得保留为重放锚点，但不能把全部提升归因于某一模块，更不能先把基线设成较弱的 AMP 条件再声称架构贡献。
- 同步双原型在 I00/I05 对照中没有“大突破”的证据。单 seed、百余分之一 AP 的差别不足以支持稳定优势。
- I23 的高分辨率候选路径有召回线索，但伴随 AP 不变或下降，提示需要同时查误报、排序、定位及掩膜质量。不能因此简单宣布 P2 有用或没用。
- 越深、越多模块、越多融合不保证更强。例如 G0830 的 G01 / G04 历史 AP 为 66.864 / 66.515，增加深层混合并未自动改善。

### 2.1 相对更强的证据：同一权重，改变输入视图与合并

以下来自保存的 paired_metrics.json，保留其原有“tiny”定义与评估口径，不能等同于新正式数据上的 COCO AP_S。

该 paired 脚本的 tiny 定义是共同 640 栅格上可见 mask 面积 <256，score≥0.25、mask IoU≥0.5 的匹配诊断；表中 AP 也是该脚本共同栅格的 AP，不是原图 COCOeval 或官方验证器的 AP。这里只用于同权重管线内部分析。

| 权重与推理方式 | tiny 匹配数 / 161 | tiny R (%) | Mask AP50 (%) | Mask AP50–95 (%) | 保存的中位耗时 (ms) |
|---|---:|---:|---:|---:|---:|
| I20：global | 25 | 15.53 | 82.91 | 66.72 | 137.70 |
| I20：fine + trustedmask | 84 | 52.17 | 88.40 | 75.69 | 417.48 |
| V12_04：global | 33 | 20.50 | 84.22 | 68.61 | 93.86 |
| V12_04：fine + trustedmask | 97 | 60.25 | 89.57 | 78.31 | 338.72 |

这说明：**保留输入尺度与使用合适的跨视图实例合并，比若干小模块的变化显示出更明显的收益。**但它识别的是“整个推理管线的变化”，其中同时改变了视图数、目标有效尺度、边缘截断和合并；不能独立归因为主干。

速度代价也必须承认：上述 fine 管线约为对应 global 的 3.03 倍和 3.61 倍。它是“精度方向有根据”，不是“已实现轻量高精度”。旧验证集 tiny=161 的统计不能搬到当前正式验证集。

## 三、当前正式数据集：重新只读核验后的发现

数据：E:/mastercode/data/orange_yolo_grouped_dedup_20260820。实际清单在 audit/split_manifest.csv；不是假定存在的 group_split_manifest.csv。

此次检查文件成员、清单组别、图像字节 SHA256、标签和静态 letterbox640 几何。没有利用 test 难例或 test 表现选架构；test 仅核对成员、数量及来源统计。

| 项目 | train | val | test |
|---|---:|---:|---:|
| 图像数 | 676 | 193 | 96 |
| 实例数 | 4,120 | 1,181 | 589 |
| 完全无果实标签的图像 | 1 | 0 | 0 |
| 连续多边形框短边 <8 输入像素 | 233 / 5.66% | 104 / 8.81% | 未作难度筛选 |
| 连续框短边 <16 输入像素 | 834 / 20.24% | 243 / 20.58% | 未作难度筛选 |
| 原始 P3/P4/P5 网格中心不在框内 | 54 | 21 | 未检查 |
| 当前 TAL 扩框后仍无候选中心 | 0 | 0 | 未检查 |
| 独立 GT 掩膜栅格化消失，ratio=2 | 0 | 0 | 未检查 |
| overlap GT 编码后实例消失，ratio=2 | 1 | 0 | 未检查 |
| 独立 GT 栅格化消失，ratio=4 | 2 | 2 | 未检查 |
| overlap GT 编码后实例消失，ratio=4 | 3 | 3 | 未检查 |

检查到：清单没有缺失/额外成员；group_id 跨 split 为 0；字节相同图像跨 split 为 0。**这些检查不等于新完成感知近重复或未知拍摄组的穷尽排查，也不证明跨果园泛化。**

清单 SHA256：
6309079758cb69404e6294e3ab8cf1274fcb8461bc90662db059e6b91982b17b

按当前文件顺序拼接“成员、图像哈希、标签哈希”的身份 SHA256：
d1e2f00fcc7cfca4908885f5705dc3be0704355e616449b604b37b53716181d1

### 3.1 不能把全部 tiny 都理解为完整的小圆果

检查六张按固定顺序选取的 val 小实例原图裁块，可以看到模糊的小果、叶片后只露出的薄片/弧形可见部分。这是定性例子，不代表六张足以估计全数据比例。

[原图 GT 诊断裁块](E:/mastercode/_work/20261001_j_strict_review_v3/val_tiny_gt_context.png)

因此需要区分：

- 远距离造成的完整小果；
- 较大果实被叶枝遮住后只剩很小的**可见区域**；
- 与邻近果实接触或重叠的实例。

强制圆形、凸形、单连通或补全监督，可能把真实遮挡叶片错当果实，或把两颗果实连起来。当前论文限定 visible instance segmentation，不应悄悄转成 amodal completion。不能据几张样例判定标注有错，更不应删掉困难小实例抬分。

既有难度统计中 solidity<0.85、间隙≤2 的比例属于凸包缺损和邻近性**代理指标**；不是人工遮挡率或真实 split/merge 错误率。既有颜色距离采用 OpenCV uint8 Lab 数值，不是经标定的物理 CIE ΔE，邻域也没有叶片类别标注。因此“果叶混淆已量化为某个比例”仍缺直接证据。

此前整数 raster bbox 的短边计数，与本轮连续 polygon bbox 不同：前者有顶点取整和 boundingRect 像素占位，阈值附近会变化。不能把两套数量差异解释成数据突然改变，更不能混用其分母。COCO AP_S 的面积来自原图评估标注；640 输入面积分箱需另命名。

### 3.2 两个容易忽略的数据问题

**拒识验证不足。**val/test 没有纯叶枝、完全无果实的图像。这不代表训练没有负像素，而是不能据此充分证明模型面对“无果图”时的整图误报控制。建议后续另采真实无果图作为部署压力测试，先固定评估用途，不能从 test 反向挑选。

**来源异质性。**train 包含 565 张 camera_2026、79 张 camera_2023、29 张视频图、3 张清单标为 ai_generated 的图。对应实例为 2368、1237、428、87。val/test 主要由两个 camera 来源组成。来源标签需核实，但这不是自动删除的理由；应登记合成来源并在需要时做明确的训练来源消融，保持验证成员不变。

### 3.3 当前分配机制否定了一个过于简单的解释

本地 TaskAlignedAssigner 对短边不足 16 像素的轴扩大**候选资格区域**，并不扩大回归 GT。静态 640 输入下，train/val 全部 GT 均有扩框后的候选中心。

所以不能继续说：“极小目标没有网格中心，所以一定没有正样本。”

真正需要测的是：经过实际裁剪、Mosaic、缩放之后，每个 GT 的最终正样本数、匹配冲突、分类×IoU 质量、目标分数总量。几何上有资格 ≠ 最终获得有效正样本。本轮只核对几何资格，未编造训练分配统计。

## 四、PR 末端归零：现象与真正瓶颈必须分开

PR 图的横轴是 Recall，不是 Confidence。最大召回率附近的补零，不表示“置信度达到 0.85 后精度突然变成 0”。

当前本地 compute_ap() 在实际 recall 序列后再加入 (R_max,0) 和 (1,0)；本轮核对的 [Ultralytics 官方 main 源码](https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/utils/metrics.py) 也有同样处理。因此本次**没有发现该端点是本地私改造成的 bug**。

在不可达的 R>R_max 区间补零，与“实际低置信度候选中误报变多”是两个现象：

- 前者是绘图/积分的表示约定；
- 后者需要查看实际 TP/FP 排序与未检出的 GT。

构造示例 r=[0.2,0.5,0.8]、p=[1,0.9,0.8]，当前端点插值 AP 为 0.736，仅在 R=1 补零的另一插值方式为 0.820。**8.4 点是玩具数组的差别，不是本数据 AP 的纠偏量，后一方式也不是 COCOeval。**不能通过更换插值让论文数字“变漂亮”。

正式跨框架比较应继续使用统一原图 GT 与 COCOeval；原生 quick AP、COCO AP、固定 score=0.25 的 R、PR 的 R_max、COCO AR100 分列报告。COCO 的 101 个召回阈值也不意味着它等于上述梯形插值。[COCO 官方评估实现](https://raw.githubusercontent.com/cocodataset/cocoapi/master/PythonAPI/pycocotools/cocoeval.py)

要解决的是可达召回率及高召回段的真实误报，而不是要求所有图都延伸到 R=1 且保持高 P。


## 五、数学上能严格说什么，不能说什么

以下推导区分“代数恒等式”“带条件的界”和“需实验验证的效果”。SymPy 1.14 核验了融合导数、平方掩膜位移关系与反例；未用公式冒充训练或泛化性能证明。

### 5.1 目标越小，同样位移的 IoU 损失越大

考虑两个边长 d 的方形掩膜，在 x、y 方向各偏移 δ，且 0≤δ≤d：

\[
I=(d-\delta)^2,\qquad
\operatorname{IoU}=\frac{(d-\delta)^2}{2d^2-(d-\delta)^2}.
\]

各方向只差 1 像素时，d=4、8、16 的 IoU 分别为 9/23≈0.391、49/79≈0.620、225/287≈0.784。若要求 IoU≥τ：

\[
\frac{\delta}{d}\le 1-\sqrt{\frac{2\tau}{1+\tau}}.
\]

τ=0.5 时，允许的相对位移仅约 0.1835。对于 4 像素见方的目标，双轴偏移 1 像素已经越过 AP50 的匹配阈值。

这是理想方形的敏感性示例，不能当成真实不规则遮挡轮廓的误差预测。它说明输入尺度、掩膜栅格与对齐值得优先处理。NWD 缓和的是框距离监督，不改变正式掩膜 IoU 的评价条件。[NWD 原始论文](https://arxiv.org/abs/2110.13389)

3072 像素的整图缩为 640 时，16 原图像素约剩 3.33 输入像素。后续上采样不能凭空重建已经缺失的观测证据；与此相比，保持原始有效尺度的切片有更直接的作用位置。

### 5.2 像素重排可逆，不等于整个降采样模块无损

令 U 为 pixel-unshuffle，它是坐标置换，存在逆 U⁻¹。若随后用矩阵 W 压缩通道：

\[
z=WUx.
\]

在被压缩的特征空间中，只要 W 有非零核，就存在不同输入映射到相同 z。例如 W=[1,1]，差向量 [1,-1] 被压为 0。JDown 的 4C→C_out 通道压缩不能普遍保证细节无损。

JStem 的 RGB、灰度与局部对比输入含确定性冗余，不能仅按其扩展到 20 个通道的环境维度就断言一定丢失原 RGB。正确结论是：**重排本身可逆，后续学习变换没有普遍的无损保证。**

灰度或梯度 g(X) 是 RGB 的确定性函数：

\[
I(Y;X,g(X))=I(Y;X).
\]

它不会凭空增加传感器信息，但可能提供更合适的归纳偏置或优化路径。不能说“灰度天然提供了 RGB 没有的信息”。局部去均值 C(X)=X-G*X 在归一化滤波核及一致边界条件下满足 C(X+b)=C(X)；这个性质针对加性常量，不等于对所有光照、颜色变化或叶脉噪声鲁棒。

### 5.3 凸组合有幅值性质，没有“不会淹没小目标”的保证

JFusion 的混合部分为

\[
F=\sum_j\alpha_jV_j,\quad
\alpha_j=\frac{e^{a_j}}{\sum_k e^{a_k}},\quad
\alpha_j\ge0,\quad \sum_j\alpha_j=1.
\]

各分量位于对应源值的凸包内；在把 V 视为独立变量时：

\[
\frac{\partial F}{\partial a_j}=\alpha_j(V_j-F).
\]

这些性质成立。但如果弱果实仅在一个源出现，初始均匀权重仍会削弱其响应；低分辨率池化还可能丢掉它。后续 refine 卷积可改变幅值。

对实际输入 x 的完整导数还包括

\[
\frac{dF}{dx}=\sum_j\alpha_j\frac{dV_j}{dx}
+\sum_jV_j\frac{d\alpha_j}{dx}.
\]

所以不能只用权重和为 1 推出整个模块 Lipschitz≤1、训练稳定或无信息损失。空间尺度加权已有 ASFF 等先例，创新必须落在任务条件与验证上。[ASFF](https://arxiv.org/abs/1911.09516)

### 5.4 差分消除常量，不会自动消除叶片

JContext 令两个分别归一化的注意力矩阵为 A₁、A₂，D=A₁−A₂：

\[
D\mathbf1=0.
\]

因此共同常量值被消除，且按行绝对和定义的矩阵范数满足 ||D||∞≤2。**常量的果实响应也会被消除，真实叶片背景又并非常量。**还没有任何证明说明其差分方向必然对应“果实减叶片”。

J 的残差修正系数 a=0.25 tanh(g) 有界，但 W_out 与特征幅值没有因此获得统一界。更不应把 a 的有界性写成控制稳定性或训练收敛证明。

如果另有迭代 h_(t+1)=h₀+aR(h_t)，只有在 R 的 Lipschitz 常数 L_R 已知且 |a|L_R<1 等条件成立时，才能引用压缩映射性质。当前前馈残差网络不满足“仅凭限幅就证明闭环稳定”的推理。

JContext 的零初始化 gain 使第一次反向只有 gain 得到来自分支输出的梯度，qk 的梯度为 0；本轮实测如此。gain 更新后分支可开始学习，这不是永久死亡，但要观察实际学习轨迹。

原始 Differential Transformer 与 LinearDiff 都比“两个注意力相减”拥有更完整的参数化或归一化设计。J 是独立简化，不是论文原模型的已验证复现。[ICLR 2025 原文](https://proceedings.iclr.cc/paper_files/paper/2025/hash/00b67df24009747e8bbed4c2c6f9c825-Abstract-Conference.html)、[NeurIPS 2025 原文](https://proceedings.neurips.cc/paper_files/paper/2025/hash/5820ad65b1c27411417ae8b59433e580-Abstract-Conference.html)

### 5.5 高分辨率掩膜表示解决不了无候选实例

J11 对候选的未裁剪、未 sigmoid 掩膜 logit：

\[
L=CP,\quad C\in\mathbb R^{n\times32},\
P\in\mathbb R^{32\times HW},\qquad \operatorname{rank}(L)\le32.
\]

提高原型空间分辨率不等于取消共享表示容量限制。实例裁剪和 sigmoid 会改变最终表示，因此这个秩界不能直接施加到最终二值掩膜。

更基本的是：没有该果实的候选或系数，就没有对应实例可供掩膜精化。J11 的 P1/P2 用于原型与 scene 分路，检测候选仍来自 P3/P4/P5（640 时共 8400 个位置）。高分辨率解码器不等于高分辨率候选头。[共享原型先例 YOLACT](https://openaccess.thecvf.com/content_ICCV_2019/html/Bolya_YOLACT_Real-Time_Instance_Segmentation_ICCV_2019_paper.html)

### 5.6 辅助损失与主任务：先检查梯度是否合作

设主目标梯度 g_m=∇L_m，辅助梯度 g_a=∇L_a，简单梯度步为 Δθ=−η(g_m+λg_a)。一阶展开：

\[
\Delta L_m\approx-\eta||g_m||^2-\eta\lambda\langle g_m,g_a\rangle.
\]

辅助方向与主方向相反时，增加权重可能抵消主任务下降，而不是稳定涨点。语义、边界和 tiny mask 损失的梯度需要在共享层测范数与余弦相似度。[GradNorm](https://proceedings.mlr.press/v80/chen18a.html)、[PCGrad](https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html)

可以作一个**条件性分析**：若在同一个固定光滑目标上采用普通梯度下降，L_m 的梯度为 β-Lipschitz，且 λ||g_a||≤γ||g_m||、0≤γ<1，则

\[
L_m(\theta-\eta(g_m+\lambda g_a))-L_m(\theta)
\le\left[-\eta(1-\gamma)+\frac{\beta\eta^2}{2}(1+\gamma)^2\right]||g_m||^2.
\]

这是由 Cauchy–Schwarz、三角不等式和下降引理得到的上界。若 η<2(1−γ)/[β(1+γ)²]，该上界为负。

**边界条件极其重要：**未知 β、随机批次变化、非光滑算子和 AdamW 动量/衰减都不能直接套用这一普通梯度步保证。它可以指导“限制辅助修正对主梯度的相对强度”的试验，不能据此宣称当前代码已实现全局稳定的 PID 优化器。当前 J 使用固定辅助增益，尚未实现这个方案。

此外，当前基础 segmentation BCE 已经按实例框区域裁剪并做面积归一化，不能说它完全按全图像素平均、必然只照顾大果。真正还可能不平衡的是正样本数、匹配质量以及每个 GT 的贡献。

J 辅助项先平均同 GT 的系数再计算损失，这不等于逐候选监督。构造 GT=1、两个 logit 为 +8/-8 时，平均 logit 的 BCE≈0.693，而候选 BCE 均值≈4.000。这个反例说明“先平均”可能掩盖候选间差异；**现有逐候选基础 BCE 仍保留，辅助 Dice 也不具有相同凸性结论**，不能把反例夸大成基础监督失效。

## 六、J11 实现的实际优点与风险

本轮深查 J 主干/融合/解码、J 损失、TAL、掩膜栅格化、AP 计算、训练协议，并对 G10、SAGE42、I20/I23 的当前 YAML/相关源码作对照。没有声称把 457 个历史网络所有源码逐行重新读完。

| 部分 | 值得保留的基础 | 风险与需补的证据 |
|---|---|---|
| JStage 主干 | 部分通道空间混合、明确窄通道预算 | 多个小 DW/PW/门控的访存与 kernel 成本；压窄后的语义能力需实测 |
| JFusion 颈部 | 投影后对齐、一次源融合、不反复 PAN 回路 | nearest/avgpool 不是语义对齐；softmax 竞争也可能压低弱源 |
| JContext | 有限记忆避免 N² 注意力矩阵 | stride16 上 4×4 池化，一格对应约 160×160 输入区域；细小响应可能被平均稀释 |
| P1/P2 原型解码 | 保留浅层细节，候选与掩膜任务分工 | 不新增 P2 检测候选；没有召回提升的直接保证 |
| Scene 分路 | 非硬裁剪，分类输入乘性因子约在 0.75–1.25 | 前景稀疏时 scene 像素 BCE 可能偏背景；共享主干和 TAL 仍造成任务耦合 |
| tiny/boundary 辅助 | 基于真实可见 GT，避免强行补圆 | 未匹配 GT 不获得此辅助；固定形态带宽对 tiny 可能过粗 |
| 普通算子路线 | 不要求 Mamba / spconv | 标准算子也不意味着必然低延迟 |

注意 J11 默认 tiny_gain=0、boundary_gain=0；J14/J15/J16 才是对应损失对照。不能把这些创新都算成 J11 已开启的效果。3 像素形态边界带是在掩膜网格上定义，stride2 时其尺度约为输入 6 像素，不能称为“3 个输入像素”。

本地成本表：J00 2.843M 参数、约 10.356 GFLOPs；J11 1.075M、约 5.216 GFLOPs（含注意力 matmul 估算）。但该次 CPU forward 中位数约 118.76ms / 123.08ms，J11 没有表现出等比例加速。CPU 数据不是 GPU 部署速度，重复测量与完整管线尚缺；结论只能是“静态预算更小，实际速度待测”。[FasterNet 对 FLOPs 与访存/延迟的分析](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html)

训练协议也必须分清：J 的 settings() 将预训练改为 scratch、mask_ratio=2、copy_paste=0.3，并采用源图均衡混合视图；J00 同样如此。它与现有预训练、mask_ratio=4、copy_paste=0 的正式基线协议不同。内部对照可以这样做，但不能直接引用历史 78/83 或另一协议 baseline 作为结构增益。

当前 source_digest() 没覆盖 TAL、metrics、data/utils、augment 等关键文件。本轮审计已另存这些文件 SHA256；建议后续实验快照把它们纳入，不改变用户的数据确认流程。


## 七、文献检索及代码学习：不是看到模块就安装

本轮范围检索整理了 **34 项论文主题**，包含 CVPR/ICCV/NeurIPS/ICLR 的已核验原始来源、直接绿色果实研究，以及 2026 年的新方法和预印本。逐项记录发表状态、作者代码链接、阅读深度、迁移原则与否定理由：

- [34 项文献取舍矩阵](literature_matrix.csv)
- [17 批原始检索/打开记录](literature_search/)
- [前一轮 J 文献与合集审阅](../J_SERIES_LITERATURE_20261001.md)

这是工程范围检索，不是穷尽式系统综述；没有虚构 PRISMA 命中数，不宣称 34 篇全部逐页精读。全文访问失败的条目只使用可核对的原始摘要或方法片段，并注明限制。原先推荐的 parallel-cli 在本环境不可用，未安装或登录，使用内置浏览工具继续检索。

### 7.1 最值得研究的六条路线

| 原始工作 | 能解释本任务哪个问题 | 不应该如何使用 |
|---|---|---|
| [SAHI](https://arxiv.org/abs/2202.06934) / [QueryDet](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html) | 输入尺度及高分辨率候选的计算预算 | 不把 dense gate 乘法叫稀疏加速；不让粗级硬筛选成为不可恢复漏检 |
| [FreqFusion](https://arxiv.org/html/2408.12879v1) / [DySample](https://openaccess.thecvf.com/content/ICCV2023/html/Liu_Learning_to_Upsample_by_Learning_to_Sample_ICCV_2023_paper.html) | 类内一致性、边界和跨尺度错位是不同问题 | 不只换融合权重后就声称完成频率融合；不一上来堆重动态卷积 |
| [DCNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Camouflaged_Instance_Segmentation_via_Explicit_De-Camouflaging_CVPR_2023_paper.pdf) | 目标参考与周围背景的实例级对比 | 不把零行和常量相消称为可靠的果叶语义分离 |
| [Mask Scoring R-CNN](https://openaccess.thecvf.com/content_CVPR_2019/html/Huang_Mask_Scoring_R-CNN_CVPR_2019_paper.html) | 已生成掩膜的质量与分类置信度不一致 | 不能只做分数重标定解决没有候选的果实 |
| [PointRend](https://openaccess.thecvf.com/content_CVPR_2020/html/Kirillov_PointRend_Image_Segmentation_As_Rendering_CVPR_2020_paper.html) / [FastInst](https://openaccess.thecvf.com/content/CVPR2023/html/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.html) | 高分辨率边界精化与轻量实例表示 | 不将查询式完整框架再与全部现有头堆叠 |
| [FasterNet](https://openaccess.thecvf.com/content/CVPR2023/html/Chen_Run_Dont_Walk_Chasing_Higher_FLOPS_for_Faster_Neural_Networks_CVPR_2023_paper.html) / [GradNorm](https://proceedings.mlr.press/v80/chen18a.html) | 实际速度及主/辅助梯度的代价 | FLOPs 少不等于快；损失数量多不等于学得好 |

直接任务先例也很重要：

- [2025 年绿色柑橘整树切片分割](https://doi.org/10.1016/j.atech.2025.100834) 已把切片用于该类任务。因此“首次给柑橘切片”不是可成立的创新点。
- [2026 年 YOLO11n-LMPS 柑橘分割](https://doi.org/10.1016/j.scienta.2026.114895) 是相关新工作，但其数据/指标协议与本数据不一致，不能把其数字拿来直接判定本模型输赢。
- [Mask Positioner](https://link.springer.com/article/10.1016/j.jksuci.2023.101598) 的层次细化和绿果边界值得借鉴；其流程仍依赖先有区域，且数据处理排除了无果图，不能当作全场景拒识充分的证据。

### 7.2 2026 新思路也必须筛掉不合适的部分

[SOPSeg](https://openaccess.thecvf.com/content/CVPR2026F/html/Wang_Prompt-driven_Small_Object_Instance_Segmentation_in_Earth_Observation_CVPRF_2026_paper.html) 是 CVPR Findings，不能标成主会。区域放大值得参考，但提示依赖不能变成推理时偷看 GT。

[SAMIX](https://openaccess.thecvf.com/content/CVPR2026/html/Hu_SAMIX_Reinforcing_SAM2_with_Semantic_Adapter_and_Reference_Selecting_Policy_CVPR_2026_paper.html)、SDDF 和多种无监督伪装方法提示语义参考的重要性；但本任务已有人工实例 GT，重 SAM/MLLM 或伪标签流程未必划算。最多考虑独立训练期蒸馏对照，不把大模型推理常驻轻量网络。

[ROI-Gated SAHI](https://arxiv.org/abs/2608.23923) 预印本还报告了某些静态内容门控劣化的结果；[ASAHI](https://arxiv.org/abs/2604.19233) 的自适应切片则来自其他图像域。不能只摘涨点数字，忽视被筛掉的弱目标及选择策略的额外开销。

[clDice](https://openaccess.thecvf.com/content/CVPR2021/html/Shit_clDice_-_A_Novel_Topology-Preserving_Loss_Function_for_Tubular_Structure_CVPR_2021_paper.html) 面向血管、道路等管状网络的连通拓扑。果实实例的“保持一个被遮挡实例”和“分开接触的两颗果实”不同，不能把连通损失直接套过来。

### 7.3 实际读到的外部代码

本轮追读的机制文件，而不是运行整个仓库：

- QueryDet：models/querydet/qinfer.py，真实构建 SparseConvTensor；阈值无候选时直接返回空。说明粗筛错误会中断后续发现，且原实现有 spconv 依赖。
- FreqFusion：FreqFusion.py 的 _forward() 和偏移路径；先压缩引导特征，再结合低/高通重组、局部相似度与采样。不是简单 concat 或源权重。
- DCNet：dcnet/modeling/ICS/reference_attention.py，top-k 参考位置和相关性投影；与 J 的无目标标识常量差分不同。
- LinearDiff：image_classification/models/vca_deit.py，核对双阶段、位置/归一化及可学习差分系数；J 不是逐算子复刻。
- 模块合集：新增读完 DySample.py，看到常规 grid_sample 路径；复用上一轮对 StarNet、CGAFusion、MSAA 的实际审阅。合集文件名不等于原作者正确实现或录用证明。

未安装、执行或改动这些外部仓库。本轮未新增下载仓库；既有代码已经足够检查上述计算机制，新的文献以来源链接和检索记录留存。未把第三方代码改名后包装成原创。

## 八、后续架构方向：让每条新路径都有可否证的作用

推荐保留“低成本语义发现 + 窄高分辨率证据 + 实例化可见掩膜”的骨架，但不要预先承诺整套综合网络必胜。

### 8.1 优先级 A：有效分辨率与候选覆盖协同

先用现有强锚点重放 global、coarse、fine 对照，固定权重、候选阈值与输出协议，查 tiny 的新增 TP 来自哪里。

若漏检主要在候选发现而非掩膜，才测试以下二选一：

- 窄 P2 候选支路：真正增加发现位置，但不得用 raw P2 边缘直接高置信分类；
- 有软兜底的局部高分辨率查询：既利用语义定位，也保留弱果实的探索机会。

不是两条都叠加。QueryDet 的计算局部化原则可参考，但需要标准算子版本与实际 GPU 延迟证据；全图 dense 运算乘以稀疏 mask 并不会自动减少计算。

若未来设计概率预算，可用 p_k=(1−ε)q_k+ε/K 保留探索。对一个只出现在指定区域的目标，M 次独立抽样至少访问一次的概率下界为 1−(1−ε/K)^M。**这只保证区域访问概率，不保证果实被正确识别；无放回或确定性 top-k 不适用该表达式。**当前没有实现此策略。

### 8.2 优先级 B：局部目标—背景对比，而非无条件高频增强

RGB 保留颜色/语义，细节支路可利用亮度、边缘及局部相对差异，但必须同时保留原信号，防止把弱果实相消。该支路不应再引入一套重主干。

先测现有融合的边界错位、GT 内/外特征相似度及低对比 FP，再决定是否只在 P2/P3 的一次融合上增加轻量对齐。FreqFusion 的“内部一致性—外部边界—位置错位”分解更值得借鉴，而不是盲目引入所有滤波生成器。

局部对比不等于圆形先验：叶脉也有高频；只对中心—周围作差也可能压掉低对比小果。需要目标语义、局部结构与上下文共同决定，不准使用推理 GT 选窗口。

### 8.3 优先级 C：保留真实可见拓扑的解码与监督

高分辨率解码优先服务已有候选，结合可见 GT 的局部边界，不做 amodal 圆形补全。

针对三种失败分别判断：

- 框已有但 mask 漏掉薄片：掩膜表示、栅格/插值和精化；
- 同一果实被错误切碎：实例内一致性；
- 两个接触果实合并：实例间判别和合并规则。

不要以全局二值前景平滑同时处理三者。先量化 solidity/凸包缺损、间隙、split/merge，再声称解决结构难点。

若好掩膜在排序中得分过低，单独做 mask quality 对照；若根本无候选，不先加评分头。辅助损失先测梯度占比/方向，保持 AdamW 对照；没有分配或梯度证据时不改优化器。

“人类先组织图像，再识别实例、反复修正”可以启发分路与反馈设计，但这里只是算法类比，不是已证实的神经科学机制，也不是原创性证明。

## 九、最低限度的验证计划：减少不可归因的大批盲跑

### 9.1 先不增加新架构，完成诊断

在同协议验证集导出实际错误链，保存原始候选和最终预测：

1. 分配：按 GT 统计最终正样本数、冲突和目标分数质量；必须使用真实训练增广批次。
2. 发现：低阈值下的候选覆盖；不能只看 loss 或箱型几何。
3. 筛选：score、top-k、NMS、maxDet、切片合并各自丢掉哪些 GT。
4. 掩膜：候选已有时的 best mask IoU 分布；IoU=0.5/0.75 的失配率。
5. 排序：候选分数与掩膜质量关系，固定 score 的 P/R 与 R_max 分开。
6. 任务难例：缩放后 tiny、低对比、深凹可见形状、近邻果实；交集子集也报样本数。

当前 train/val 单图 GT 最大为 46/45，低于 maxDet=100，但大量误报仍可能挤占候选预算。可以做 maxDet/阈值的诊断敏感性分析；正式指标仍锁定统一设置，不挑最高的一组写成主要结果。

这六项是下一步需要实现/执行的诊断计划，不是声称本轮已经取得真实模型错误分布。仅凭 results.csv 不足以将全部 FN 都归到颜色或极小目标。

### 9.2 已有 J 模型最有意义的消融顺序

| 比较 | 要回答的问题 |
|---|---|
| J00 vs J01 vs J02 | 同输入/优化协议下，官方/历史锚点/紧凑结构差别 |
| J04 vs J05 | P2 与 P1 掩膜分辨率是否值得其速度代价 |
| J05 vs J06 | 灰度/局部对比是否真正提供任务收益 |
| J05 vs J08 | 竞争式权重是否比均匀融合有效 |
| J09 vs J10 | 差分上下文是否胜过普通有限记忆上下文 |
| J11 vs J12 / J13 | 综合模型是否真的需要 scene / differential |
| J11 vs J14 / J15 / J16 | tiny 和边界辅助是否单独/联合有益 |

priority 队列适合初筛，但缺若干单因素臂；它不能代替完整贡献归因。上述只是建议实验矩阵，当前没有自动改用户队列或启动训练。

### 9.3 正式论文条件

- 架构消融固定输入管线、初始化、优化器/LR、AMP、mask_ratio、copy_paste、训练轮数、seed 和验证成员。
- 为创新输入管线做架构×输入的拆分：基线/方法都要有相同 global 与相同切片策略，避免把切片收益全部记给网络。
- AMP=0/1 是明确的数值训练因素；两组对称跑，正式评估精度一致。没有孤立对照，不宣称 AMP 单独造成所有历史涨点。
- 统一原图 mask COCO AP50–95/AP50/AR，另报固定操作点 P/R、tiny 输入分箱、可见形状子集与 split/merge。
- 速度计入全部视图、合并和掩膜重建；batch1 与吞吐分开，固定 GPU/环境/精度/输入。参数和 FLOPs 是辅助，不替代端到端延迟。
- 筛选一次；最后方法与主基线各三 seed，报告均值±标准差。置信区间按图或拍摄组重采样，不把同图果实当完全独立样本。
- 验证集选方法，最终 test 在协议冻结后使用。多轮从同一 val 峰值挑冠军会产生选择偏差；三 seed 也不能替代独立 test。
- 精度提升、tiny 召回与真实耗时必须一起看；若综合版不如简单版，应保留简单版，而不是继续增加创新名称。

现在还不能证明一套“完美网络”。比较可信的论文主张应是：**在有限计算预算下保持微小可见实例的发现能力，并用结构—语义联合证据减少同色背景误报，同时维持遮挡与接触果实的可见实例边界。**这是待验证的研究假设，不是已经证实的三大原创成果；能否投稿需同协议完整实验支撑。

## 十、本轮产物与验证边界

- [只读审计脚本](E:/mastercode/1_SEVER/code/ultralytics-main-new/scripts/audit_j_research_assumptions.py)
- [最终当前数据/数学审计 JSON](E:/mastercode/_work/20261001_j_strict_review_v3/audit.json)
- [逐实例监督几何 CSV](E:/mastercode/_work/20261001_j_strict_review_v3/instance_supervision_geometry.csv)
- [当前验证集 GT 裁块](E:/mastercode/_work/20261001_j_strict_review_v3/val_tiny_gt_context.png)
- [34 项文献表](literature_matrix.csv)
- [历史成本表](E:/mastercode/_work/20261001_j/profile/model_costs.csv)

审计脚本成功完成，Ruff F/E9 检查通过；现有数学恒等式/构造反例断言通过。数据只读，没有删除、修改或重新划分原始图像和标签。

本轮没有正式 J 训练、GPU 测速、各权重重新预测、全量近重复感知排查或所有历史源码版本追溯；这些缺失不能用语言或漂亮网络图补齐。基线与其他人的未完成改动保持原样。

复跑该只读审计时选择一个新的输出目录（脚本避免覆盖已有报告）：

~~~powershell
python scripts/audit_j_research_assumptions.py --dataset E:/mastercode/data/orange_yolo_grouped_dedup_20260820 --output E:/mastercode/_work/j_strict_review_new
~~~

**建议下一步：先在强锚点与轻量对照上做候选—掩膜错误分解及 GPU 小规模测速，再决定是改发现路径、融合对齐还是掩膜/评分。不要为了图更复杂而重构，也不要在没有结果时把 J11 宣传为最强。**
