# I_V4：分层发现—区域比较—边界保护的并行实例分割

> 路线说明：本文件及 I40–I47 仍使用 YOLO 系实例头。独立于 YOLO 的新架构方案见 [Citrus Native V1](CITRUS_NATIVE_V1_ARCHITECTURE_20260925.md)；目前是设计稿，不对应可运行 YAML。

## 1. 结论与边界

这次不再把“加灰度、加注意力、加一个检测尺度”作为主要改进。主假设是：**先用明确的可见前景监督建立场景分层，再利用前景/背景的相对特征帮助实例预测，同时保留未被分层筛掉的局部细节。**

已实现 8 个 YAML 和前台串行批量入口。I44 是待验证的主候选，不是已证实的最强模型。没有 I_V4 训练结果，不能承诺颠覆性成功、上涨若干点或达到期刊接收标准。本轮没有读取到 I_V3 的完成结果，因此 I33 双支路只作为 I46 的探索性对照。

采用 scientific-brainstorming 的“类比—反例—可检验假设”方法，将用户关于人类图像理解的描述转化为工程实验；不把拟人描述当作生理事实或性能证明。

## 2. 对“先分割，再理解”的辨证理解

值得借鉴的是职责分离、对象与上下文相互约束、保留细节并修正初始判断。不能照搬的有三点：

1. 人类视觉不是被证实严格遵守“颜色/数量分完→空间理解→形态理解”的串行程序。场景与对象处理可以并行，高层信息也可以反向影响局部解释。[R1–R3]
2. 单张 RGB 不能直接提供运动、深度或遮挡后的真实形状。本课题仍只预测**可见果实实例**，不增加动静分类、姿态头、深度或 amodal 补全。
3. 若先用硬阈值分出前景再处理对象，早期漏掉的小果实将没有补救机会。因此分层只影响残差修正，不删除候选、不切掉背景像素、不乘零屏蔽原始细节。

“反馈”也不自动有效。2026 年关于遮挡识别的研究讨论了循环连接的作用与适用条件，不能概括为循环一定优于前馈。[R3] 本方案最多进行两次窄特征状态更新，没有重复跑 backbone；没有 PID、积分状态、控制稳定性证明或自动纠错保证。

## 3. 本地证据：什么值得保留，什么不能继续默认有效

本轮直接复核 I_V2 五个运行的 `results.csv` 最后一行及 I20 的原始 PR 诊断，阅读 I_V1/V2/V3 设计与历史审计结论、V3 代码、继承的 V12/I_V2 头部和损失。没有声称本轮重新逐文件读完全部历史系列。

### 3.1 I_V2 的同批 50 epoch 最后一行（Mask）

| 运行 | Recall | AP50 | AP50–95 | 相对 I20 的 AP50–95（百分点） |
|---|---:|---:|---:|---:|
| I20 三尺度对照 | 0.74071 | 0.83340 | 0.69622 | 0 |
| I21 关闭 recognition | 0.75327 | 0.83062 | 0.69604 | −0.018 |
| I22 直接 P2 检测候选 | 0.74833 | 0.82761 | 0.69566 | −0.056 |
| I23 语义过滤 P2 | 0.75405 | 0.83281 | 0.69580 | −0.042 |
| I24 大小核 P2 | 0.74452 | 0.83244 | 0.69701 | +0.079 |

单 seed、50 epoch，且已有成员审计指出该批与正式 grouped_dedup 不同，不能据此下显著性或论文增益结论。但足以说明：没有证据支持把更密集的 P2 检测头当作大幅涨点的默认答案。

此前 I_V1 的 fine 评估中 I07 比 I01 的 AP50–95 仅高约 0.056 个百分点；双原型的额外复杂度尚未获得支持。保留可见细节、共享 mask 解码、质量预测及原有损失作实验锚点，不继续默认增加双原型。

### 3.2 PR 不是“置信度 0.85 时精度归零”

PR 横轴为 Recall。当前 `ultralytics/utils/metrics.py` 的 `compute_ap` 在实测最大召回与 Recall=1 处补零，图尾贴零包含积分/绘图约定。本轮不改 AP 算法、不修饰曲线。

直接复核 `I20_corrected_control_seed42/paired_fine_eval/observed_pr_diagnostics/summary.json`：

| 模式 | 实测最大 Recall | 最后一个实测点 Precision | R@P≥0.90 |
|---|---:|---:|---:|
| global | 0.90467 | 0.04362 | 0.75024 |
| trustedmask（fine 切片管线） | 0.97140 | 0.02018 | 0.78646 |

这里的真实问题是：切片提高候选覆盖，同时也释放很多误检；高精度工作点的召回改善远小于极低阈值的最大召回改善。最大召回不能单独作为成功标准。已有尺寸诊断还显示极小目标是明显弱项，但不能把所有错误归因于极小目标。

### 3.3 当前三个待验证机制

- **发现不足**：目标在下采样和候选分配之前缺少明确监督。即使某个 GT 没有匹配到正样本，也应该提供前景学习信号。
- **前景与背景混淆**：只保留高频会把叶脉一起放大，需要比较“果实区域特征”和“背景区域特征”；灰度并不天然能区分绿色果实与叶片。
- **实例边界与遮挡凹口冲突**：不能把轮廓强制变圆或填成凸形。保留标注中的叶枝遮挡缺口，也必须区分接触果实。

## 4. 检索与作者代码取舍

本轮检索了视觉分层/反馈、伪装目标、对象中心分组、快速实例分割、细节修复及近期双主干路线。重点阅读本地作者代码：

- `C:/Users/33836/Desktop/github/SparseInst/sparseinst/decoder.py`：IAM 归一化后聚合对象区域特征。
- `C:/Users/33836/Desktop/github/FastInst/fastinst/modeling/transformer_decoder/fastinst_decoder.py`：query/pixel 交互、逐层掩膜以及训练时额外引导分支。

| 方法 | 借鉴的思想 | 没有直接搬入的部分/风险 |
|---|---|---|
| SINet [R4] | 搜索与辨识分工 | 二值伪装前景不是实例分割；不能替代实例身份 |
| SparseInst [R5] | 通过区域汇聚得到判别性描述 | I_V4 只使用两个场景语义区域，不宣称实现 IAM 对象集合 |
| Mask2Former [R6] | 预测区域可用于引导后续表征 | 不引入完整 masked Transformer；避免硬掩膜永久排除小目标 |
| FastInst [R7] | 对象与像素信息交互 | 暂不引入额外训练前向、完整 query decoder 和新框架依赖 |
| PEM [R8] | 用紧凑区域表征降低冗余计算 | 两个均值描述不等价于 PEM 的 prototype cross-attention |
| PIDNet [R9] | 语义、细节、边界分工 | 不冠以 PID 控制器，不强制平滑果实轮廓 |
| PointRend [R10] | 将精修资源分配给不确定位置 | 本轮未实现自适应点采样，避免未验证的插值/采样开销 |
| Slot Attention [R11] | 对象中心分组的方向 | 真实果园小数据、遮挡和数量变化下的收敛需要独立验证，不直接取代全部候选 |
| BDNet 2026 [R12] | 颜色/结构分工，可作为双支路动机 | 本轮 CVF 页面访问失败，沿用已有 I_V3 论文审阅记录；不新增未核实的结论 |

本次未复制作者模块源码；采用普通 PyTorch 算子实现任务适配。参考、代码作者、许可证应在论文与发布仓库中明确标注。真正的新颖性仍需更完整的查重式文献比较和消融，不以改名代替创新。

## 5. 已实现的信息流

```text
RGB → 既有轻量混合编码器 → C3 / C4 / C5
          └→ 持续的窄 P2 细节 ───────────────────────┐
                                                   │
C3/C4/C5 各投影到16通道 → 对齐 P2 → 联合场景状态 Z    │
                                         │         │
                              前景概率 p、边界概率 b │
                                         │         │
                       前景/背景软汇聚 → 区域差异描述 │
                                         │         │
                             有界残差修正 Z（1或2步）│
                                         │         │
             分别降至 P3/P4/P5，再投影回检测通道      │
                    + 各尺度未经筛除的局部投影        │
                                         │         │
                    三尺度实例预测 + 共享掩膜解码 ←───┘
```

### 5.1 用并行场景解码替代串行 PAN

I41–I47 的 YAML 删除了旧 head 中的反复上采样、Concat、C3k2、再下采样链，直接将编码器特征送入 `SegmentCitrusIV4`。新颈部在该类内部以 `LayeredSceneNeck` 明确实现，并非仍保留原 PAN 再额外叠加模块。

投影后的窄状态在 stride=4 汇聚；送回各检测尺度时**先下采样窄状态，再扩大通道**。不在整张高分辨率特征上运行宽 Transformer、FFT、deformable 或额外 GPU 扩展。P3/P4/P5 都继续参与上下文和三尺度实例预测，没有丢掉深层语义。

主干主体仍保留现有混合编码器和持续细节支路，本轮主要重构的是颈部信息流与学习目标。I46 单独尝试 I33 的 RGB/非对称灰度结构支路，不能说全系列已经换成全新 backbone。

### 5.2 受监督分层，不只是无标签 attention

从实例标注生成两个辅助目标：可见果实区域并集，以及实例标签转变带。边界包括外轮廓与接触实例之间的界面；不是对隐藏果实的形状猜测。

辅助目标先在标注栅格计算，再通过 max-pool 生成窄状态分辨率目标，尽可能保留已经存在的极小占据单元。它无法恢复在原标签栅格化阶段已经消失的信息。普通实例 mask GT 完全不变。

辅助 BCE 按每张图的正/负像素分别归一化，不由背景像素数量主导；空图仍训练背景抑制。即使 TAL 没有分到正候选，分层分支仍能得到监督。这是对候选瓶颈的间接干预，**不保证为每个 GT 建立新候选**。

### 5.3 软区域比较与边界保护

令 p=sigmoid(f(Z))，用 p 和 1−p 分别对 Z 做归一化加权平均，得到前景/背景描述。二者的差值与局部状态一起产生修正 Δ：

`Z_next = Z + 0.5 * tanh(gain) * (1 - sigmoid(boundary_logit)) * Δ`

`gain` 从零初始化，限制的是修正系数而非整个特征幅值，也不构成稳定性证明。边界附近减少场景信息修正，未筛选细节仍送入 mask 解码器。两个区域均值只是语义参照，不是实例标签，实例仍由独立候选和掩膜系数区分。

I45 对窄状态重复同一个单元两次，并对两次分层预测取平均辅助损失。它是固定步数的共享状态更新，不是按困难程度动态停止，也不是实例输出回到输入的完整闭环。

## 6. 实验设计与计算预算

| YAML | 主要差异 | 参数/M | GFLOPs@640 |
|---|---|---:|---:|
| I40_replay | I20/I30 重放，仅显式 nc=1 | 2.284 | 10.601 |
| I41_parallel_neck | 只改并行颈部 | 2.126 | 9.791 |
| I42_discovery_aux | I41 + 前景辅助监督 | 2.126 | 9.792 |
| I43_region_feedback | I42 + 前景/背景比较修正 | 2.126 | 9.832 |
| I44_boundary_protected | I43 + 边界监督与保护 | 2.126 | 9.832 |
| I45_two_step | I44 窄状态更新两次 | 2.126 | 9.874 |
| I46_achromatic | I44 + I33 非对称结构支路 | 2.138 | 10.141 |
| I47_no_layer_supervision | I44 关闭显式分层损失 | 2.126 | 9.832 |

参数统计 nc=1；GFLOPs 来自现有 THOP 口径估计，部分函数式区域归约/池化开销可能未完整计入，不能当作真实 FPS。I44 对 I40 参数约下降 6.9%、报告 GFLOPs 约下降 7.3%。没有 GPU 延迟实测，不承诺训练一定更快。

I43→I44 联合引入边界监督和边界门控，只能判断该机制包是否有用；论文若单独声称边界门控贡献，需补“仅边界辅助损失、不启用门控”对照。

新颈部与预训练权重匹配比例不同，不能把共享同一个权重文件等同于初始化完全公平。批量程序保留 `initialization_transfer.json`；最终结构对照应增加相同 scratch 协议复核。

## 7. 固定训练条件与运行

代码位于 `E:/mastercode/1_SEVER/code/ultralytics-main-new/`。

- YAML：`0_orange_yaml/I_V4_series/`，已加入 `MODEL_INDEX.csv`，每层仍采用一行的常规写法。
- VS Code 入口：`RUN_CITRUS_I_V4.py`；底层：`20260925_citrus_i_v4_batch.py`。
- 模块：`ultralytics/nn/modules/citrus_i_v4.py`；辅助损失：`ultralytics/utils/citrus_i_v4_loss.py`。
- 完成 modules 导出、tasks 导入、parse_model 注册与自动选择损失；支持标准 `YOLO(yaml)` 构建。

保持 V2/V3 的输入与评估：来源均衡的 global/coarse/fine 混合，不增加先验裁图。固定 AdamW、lr0=0.001、lrf=0.01、weight_decay=0.0005、warmup=3、batch=16、imgsz=640、workers=4、cache=True、AMP=False、mask_ratio=2、copy_paste=0.3、cos_lr=False、dropout=0。其余增强和训练项继承 `protocols/citrus_paper1_formal_v2_ram.yaml`，系列覆盖项在 `citrus_i_v4_suite.py` 中显式声明。这里不同时试验新优化器。

所有组使用同一官方 `yolo11n-seg.pt` 初始化来源。**不能直接与 legacy78 的 scratch/AMP=True 基线做结构增益比较**。正式论文必须有匹配初始化、AMP、划分、输入切片与评估预算的基线。

服务器上传整个代码工作副本后，在 `RUN_CITRUS_I_V4.py` 修改 DATA 和 DEVICE。不会要求确认划分清单，按你的路径直接加载；请自行确保各组使用相同正式清洗划分。

先设置：

```python
SUITE = "smoke"
EPOCHS = 1
DRY_RUN = False
```

点击 VS Code 右上角三角形，或执行：

```bash
python RUN_CITRUS_I_V4.py
```

之后切换 `SUITE="priority"`、`EPOCHS=50`，运行 I40–I44。PROJECT 默认由 suite/epochs 生成，不覆盖 smoke。其他修改或复跑同一配置时另取新 PROJECT。只有前五组有希望才运行 mechanism/all；最终筛选模型和匹配基线各做 300 epoch、seeds 42/43/44。

如只检查构建，将 DRY_RUN=True；会打印 BUILD OK 并退出，**不会训练**。cache=True 会占用 RAM，不是速度保证。无 GPU 占用保护；DEVICE 仍被用来限定训练设备。

单个 YAML 标准 API 示例（仅说明入口，正式比较优先用固定协议批量程序）：

```python
from ultralytics import YOLO
model = YOLO("0_orange_yaml/I_V4_series/I44_boundary_protected.yaml")
# model.train(data=..., ...) 还需显式提供相同训练协议，默认参数不等于本系列配方。
```

## 8. 验证与判停标准

本地 Torch 2.8 CPU 完成：8/8 标准 YAML 构建、非方形前向、8/8 真实分割损失反向、I44 的 640 输入、保存/加载回环、空图分层监督、极小占据与接触边界目标、极端概率数值稳定性、批量 all dry-run。V4 专属测试 24 项通过；V3 回归 17 项通过。未在服务器旧 Torch/CUDA 上执行训练、AMP 或导出测试。首次必须先 smoke。

主要看：Mask AP50、AP50–95、固定 P≥0.90 下 Recall、极小尺寸分组 Recall、背景误检/图、边界误差、接触果实 split/merge 以及实测端到端延迟。不同切片模式分别报告，不能拿 fine AP 与 global 延迟拼一张表。

若只是最大召回变高、R@P≥0.90 不变或下降，说明误检问题未解决；若前景分层明显改善但实例召回不变，优先诊断候选/TAL/打分，而不是再加分层模块；若 AP50 提升但高 IoU AP 与边界下降，则检查融合对形状的破坏；若 I47 与 I44 相当，就没有证据支持“显式分层监督”这一主张。

把成功门槛预先定为：三个 seed 的匹配协议下，两项 Mask AP 与关键工作点 Recall 有一致改善，参数/计算与实测延迟在可接受范围；以配对图像分析确认收益不是只来自少数近重复难例。未通过就如实否定假设，而不是修改 PR 图。

## 9. 一手参考资料

- R1：Predictive processing of scenes and objects, Nature Reviews Psychology (2024). https://www.nature.com/articles/s44159-023-00254-0
- R2：On the Necessity of Recurrent Processing during Object Recognition: It Depends on the Need for Scene Segmentation (2021). https://pubmed.ncbi.nlm.nih.gov/34088797/
- R3：Recurrent connections facilitate occluded object recognition by explaining-away, Nature Communications (2026). https://www.nature.com/articles/s41467-026-68806-5
- R4：Camouflaged Object Detection, CVPR 2020. https://openaccess.thecvf.com/content_CVPR_2020/papers/Fan_Camouflaged_Object_Detection_CVPR_2020_paper.pdf
- R5：Sparse Instance Activation for Real-Time Instance Segmentation, CVPR 2022. https://arxiv.org/abs/2203.12827 ; https://github.com/hustvl/SparseInst
- R6：Masked-attention Mask Transformer for Universal Image Segmentation, CVPR 2022. https://arxiv.org/abs/2112.01527 ; https://github.com/facebookresearch/Mask2Former
- R7：FastInst, CVPR 2023. https://github.com/junjiehe96/FastInst
- R8：PEM: Prototype-based Efficient MaskFormer for Image Segmentation, CVPR 2024. https://arxiv.org/abs/2402.19422
- R9：PIDNet, CVPR 2023. https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html ; https://github.com/XuJiacong/PIDNet
- R10：PointRend, CVPR 2020. https://arxiv.org/abs/1912.08193
- R11：Object-Centric Learning with Slot Attention, NeurIPS 2020. https://arxiv.org/abs/2006.15055
- R12：BDNet, 既有 I_V3 审阅依据与访问限制见 `I_V3_EVIDENCE_AND_RUN_20260923.md`。
