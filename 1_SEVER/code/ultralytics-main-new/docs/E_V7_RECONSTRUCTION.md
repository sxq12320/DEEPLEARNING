# E V7 八组网络复审与运行说明

日期：2026-09-12。范围：用户确认的 swe-2 新增 E V7 八组；不另造系列，不改数据与已完成结果。
本版用 EV7R2 输出目录区分修订前后的实现，模型名仍为 V7_00–07。

## 结论与历史依据

这八组值得作为筛选实验，但还没有正式训练成绩，不能称为最优网络。
它们继承 V5/V6 的主干与不对称融合结构，新增工作集中在 P2 检测分支、掩膜细节采样、
融合后特征处理和几何监督；不是全新主干，也不是完整的四因素全析因实验。

历史索引见 [V6 复审说明](E_V6_REVIEW_20260912/复审结论与运行说明.md)。
下面均为当时协议下的单次实验，不与本版 input/2 验证 AP 直接混排行：

| 历史比较 | 观察 | 对 V7 的含义 |
|---|---|---|
| V5_00 → V5_01 | 同一选定 epoch 的 Mask AP50-95 67.653 → 68.125；末 20 轮均值差约 0.05 点 | 更细输入有候选价值，但峰值优势不足以证明稳定涨点 |
| 同一 V5_00 权重，coarse → fine 评估 | tiny 匹配 72 → 93/161，背景错误 307 → 364 | 提升可见尺度同时引入误检；tiny 为统一评估栅格面积 <256，不是 COCO AP_S |
| V5 路由 fine 与上述 fine control | tiny 匹配 100 vs 93，背景错误 437 vs 364 | 不能只优化召回；背景错误需逐例归因，不能直接认定全是树叶 |
| SAGE40 → SAGE42 | Mask AP50-95 66.863 → 67.502 | 语义引导细节值得保留作控制基础 |
| SAGE44 / 45 / 46 | Mask AP50-95 67.433 / 67.291 / 67.491 | 边界、邻域及组合以前测试过，本次是换上下文重测 |
| SAGE70 → SAGE72 P2 | Mask AP50-95 67.433 → 64.651 | P2 并非必胜；但不能跨架构一概否定 |

PR 横轴是召回率，不是置信度。未达到的召回区间可能由绘图插值/补零形成尾部归零。
应分开报告最大可达召回、P≥0.90 时召回、tiny 漏检、误检与掩膜质量，不能以画圆滑曲线为目标。
AP50 与 AP50-95 的差也不能单独定位到小目标、颜色或边界原因。

## V6 全量结果与本次重定基线（2026-09-13）

V6 十组 300ep 已跑完（`results/E/E_V6/CITRUS_EV6R_ALL_300EP`，seed42，同一 input/2 协议）。
配对评估全局协议核心读数（paired_fine_eval/paired_metrics.json）：

| 臂 | 验证 mAP50-95 | 全局 mAP50-95 | tiny召回25 | 边界IoU25 | 背景错 | 合并 | 中位ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| V6_00 control | .6999 | .6154 | .161 | .523 | 136 | 16 | 111.9 |
| V6_01 mr2 | .7102 | .6044 | .161 | .518 | 121 | 15 | 94.3 |
| V6_02 fine | .7142 | .6747 | .193 | .625 | 158 | 13 | 114.4 |
| V6_03 nwd | .7030 | .6160 | .168 | .528 | 113 | 16 | 91.8 |
| V6_04 cp | .7052 | .6189 | .161 | .532 | 122 | 11 | 114.6 |
| V6_05 fine+nwd | .7117 | .6763 | .217 | .616 | 167 | 14 | 137.8 |
| V6_06 fine+cp | .7146 | .6800 | .224 | .624 | 184 | 12 | 128.6 |
| V6_07 full | .7143 | .6759 | .193 | .623 | 142 | 16 | 128.1 |
| V6_08 phase | .7107 | .6719 | .211 | .621 | 119 | 14 | 85.7 |
| V6_09 phase+nwd | .7092 | .6705 | .186 | .615 | 150 | 15 | 92.0 |

复核结论（取代前稿过强判断；完整数字见 [V6→V7 复核](E_V7_REVIEW_20260913/复核与运行.md)）：

1. fine/phase 在配对 640 栅格的严格 IoU 指标改善较明确，但 epoch AP 与配对 AP 不同口径。
   phase 相对 mr2 的 epoch 峰值仅 +0.048 点，末20轮 -0.134 点；不能将配对 +6.755 点全归因于特征学习。
2. phase 与 dense 是取舍：phase 的全局 AP50-95 低 0.274 点，细切片 trustedmask 高 1.310 点，
   但该切片协议的 tiny 匹配少 8 个。没有重复 seed，不能称“差异在噪声带”或“等价”。
3. phase 报告的全局延迟为85.7ms，较dense114.4ms低，但无同场交错重测；
   训练每轮中位18.9秒 vs19.4秒，收益远小于推理记录的比例，不能据此宣称普遍加速。
4. NWD 混合结果不支持默认叠加；copy-paste 保留单独对照，而不是所有模型默认启用。
5. rmax 是 conf≥0.001、max_det=300、特定NMS和IoU下的观测上限，不是所有置信度下的理论上限。
   现有小框候选扩张也不能证明后续匹配竞争没有问题。

所有 V7 臂保留已经接入的 phase-fine（mask_ratio=2）；V7_00 与 V6_08 图结构一致。
本轮新增 EV7SegmentationLoss 修正 P2 后置引起的 TAL 尺度元数据错误：检测顺序不变，
分配器以 [4,8,16,32] 识别最细两级。只影响 V7，不改 V6 或共享 TAL 历史行为。

现有质量打分仍保留，不增加重复质量模块。几何监督的最小面积仍为256输入像素，
改成mask_ratio=2不等于放宽该面积门槛；栅格变化对内部占比的影响需测量，不能声称必然覆盖更多tiny。

### 误差分解驱动的臂位分配

按 conf .25 配对评估把误差拆开（十臂一致结构）：总 GT≈1049，未匹配 204-231（~20%），
**其中约 57-62% 是 tiny 实例**；mask_or_localization 仅 3-16 是 conf .25 / IoU .5 的未匹配预测诊断，不能证明已匹配掩膜在严格 IoU 下准确；
背景误检 113-184 是第二大预测误差；dup/merge 仅 15-29。

因此臂位按误差族排序分配，而非按模块堆叠：

| 臂 | 因子 | 针对误差族 |
|---|---|---|
| V7_00_control | — | phase-fine 锚点（≈V6_08，逐位相等锁定） |
| V7_01_p2 | 检测尺度 | 漏检主族（60% tiny）——召回杠杆 1 |
| V7_02_deform | 细节采样 | strict-IoU 形状/深凹轮廓 |
| V7_03_pmce | 频分通道增强 | 背景误检（第二大族） |
| V7_04_boundary | 几何监督 | 边界+邻接（320² 栅格上更公平） |
| V7_05_cp | copy_paste=0.3 | 漏检——召回杠杆 2，零推理成本（V6 唯一拉高 tiny 至 .224） |
| V7_06_p2_pmce | S+C | 两大主误差族组合；假设 PMCE 压制 P2 新引入的叶 FP |
| V7_07_full | S+D+C+G | 全结构因子（不含 cp，保持架构归因干净） |

其余配对延后：若某单因子胜出，再补定向组合。deform+bnd 这类"同族形状机制"配对
让位给双召回杠杆布局——漏检是最大误差族，值得占两个臂位。

## 八组与资源成本

输入 640、nc=1、本机 THOP 构建测量；不是 GPU 延迟。
几何表示 boundary=0.5 与 neighbor=0.25 成对开启。所有臂含 phase-fine 基座（较旧control增加3,361参数、约0.174 GFLOPs）。

| 模型 | P2 | DCN | PMCE | 几何 | 参数量 | THOP GFLOPs |
|---|---|---|---|---|---:|---:|
| V7_00_control | — | — | — | — | 2,229,608 | 10.219 |
| V7_01_p2 | 是 | — | — | — | 2,248,818 | 11.225 |
| V7_02_deform | — | 是 | — | — | 2,235,844 | 10.418 |
| V7_03_pmce | — | — | 是 | — | 2,265,410 | 10.397 |
| V7_04_boundary | — | — | — | 是 | 2,229,608 | 10.219 |
| V7_05_cp | — | — | — | — | 2,229,608 | 10.219 |
| V7_06_p2_pmce | 是 | — | 是 | — | 2,286,837 | 11.507 |
| V7_07_full | 是 | 是 | 是 | 是 | 2,293,073 | 11.706 |

DCN 两组（02、07）另有约 0.118 GFLOPs 卷积乘加未被 THOP 捕获，采样/插值也未完整计入。
P2 检测候选从 8,400 增至 34,000，会增加分配、头部激活和后处理成本。
CPU 顺序计时存在漂移，不据此声称 PMCE 加速。正式速度须同卡、同 batch/精度、预热后测量，
另报切片生成、切片前向和合并的整图端到端延迟。

## 已实施的修正

1. P2 特征由 n-scale 64 通道收窄到 32，新增检测塔隐藏宽度降到 16；保留 P3/P4/P5 塔索引。
   原始 YAML 和模块备份：E:/mastercode/_work/citrus_ev7_review_20260912/originals/。
   降计算量已经测量，精度影响尚未验证。
2. P2 预训练映射为 {19: -1, 20: 17, 22: 19, 26: 23}，新增 P2 层不得误载旧层。
   区分 citrus_ev7_p2 与 citrus_ev7_control，防止 control→P2 因同 family 跳过必要映射。
   覆盖官方权重→各模型、control→P2、同图保存重载；不承诺任意历史模型之间迁移等价。
3. 提取并复用 V6 精细原型解码函数，修复 V7 组合 phase decoder 时通道不匹配。
   V6 结果落地后，phase-fine 经实测被采纳为全系列基座（见上文重定基线节）。
4. PMCE 低通池化忽略 padding 零值，避免常量特征在图像边缘产生虚假高频。
   这是 box-filter 低频/残差高频分解，不是作者原版 FEDER、HWD 或 FreqFusion 的完整复现。
5. DCN 按需导入，并在切片缓存前检查选定设备的算子前向/反向；不静默降级成普通卷积。
   这是依赖检查，不是用户已取消的单 GPU 占用/锁保护。
6. 公共 FLOPs 统计输入由 torch.empty 改成 torch.zeros，包括全尺寸回退。
   本机 PyTorch 2.8 确定性模式给 empty 填 NaN，传播成 DCN 非法坐标并触发原生崩溃。
   同配置 8 线程改用有限输入后构建成功；最初线程数猜测不成立。没有关闭确定性或改变训练输入。
7. 全部 V7 用 EV6TrainingTrainer/EV6Validator：验证 GT 从多边形统一栅格化到 input/2，
   原型 logits 在阈值化前对齐，训练 mask_ratio 随 phase-fine 基座固定 2。
   旧 input/4 epoch AP 不直接和新值比较；应对历史权重运行同一独立评估。
8. 八份 YAML 登记到 0_orange_yaml/MODEL_INDEX.csv；输出采用 EV7R2 新目录避免混用。

## 算法取舍与限制

P2 针对检测候选空间，但不自动提高原型掩膜分辨率；正样本收益需要尺寸分桶统计证明。
PMCE 假设纹理与上下文互补，但颜色混淆不能单凭背景混淆矩阵证明。
需分别复核叶片假阳性、条带遮挡和接触果实，报告可见 mask 的 solidity/凸包缺损、
实例间隙与 split/merge。切勿把遮挡造成的深凹可见掩膜强制补圆。

几何监督沿用 SAGEV4R，边界约 4 输入像素、邻域约 8 像素，并设面积与内部占比门槛。
很多极小实例不参与该损失。因此它主要验证较大可见边界/邻接分离假设，不是极小目标专用损失，
没有拓扑正确性保证。无额外推理参数，不等于无训练成本。

DCN 偏移初始化为 0、调制为 0.5、残差增益为 tanh(0.01)，不是完整普通卷积等价初始化。
收益必须和采样延迟、数值稳定性及部署支持一同考核。PMCE 作用于融合后的 P3/P4 和可选 P2，
原始 P5 不变。

这次没有引入新优化器。固定 AdamW 为了分开结构与训练策略效应，
不把 AMP=False 收益算作架构创新。八组不能单独估计 boundary 与 neighbor，也未覆盖全部交互作用。
胜出后再拆分，不预先宣称三个独立创新均有效。
控制原理可以启发残差纠偏，但这里没有闭环动态系统稳定性证明；以前几个 PID/observer 命名实验
失败也不意味着控制理论思路已经用尽。

## 运行与公平比较

打开根目录 RUN_CITRUS_E_V7.py，修改 DATA 为自己的服务器 data.yaml，DEVICE 为目标 GPU。
保持 SUITE="all"、EPOCHS=300、DRY_RUN=False，在 VS Code 选择服务器环境后点击三角形。
同一个 Python 前台串行跑八组；Ctrl+C 中止队列。没有 nohup、占用拒绝或跨进程 GPU 锁。

固定：cache=True、amp=False、seed=42、batch=16、workers=4、imgsz=640、
AdamW、lr0=0.001、weight_decay=0.0005。其余公共设置见 protocols/citrus_e_v7.yaml 及继承协议。
三尺度训练采样 global/coarse/fine=0.5/0.25/0.25，不启用学习型引导或 NWD；
仅 V7_05_cp 臂启用 copy_paste=0.3（数据级召回杠杆），其余臂保持 0。
最终以每次保存的 args.yaml、采样记录与 checkpoint 为准。

YAML 可用官方 YOLO(path) 构建与加载。复现本系列的采样和统一验证仍需传
trainer=EV6TrainingTrainer；仅使用官方默认 trainer 不等于本实验协议。

先比较 00 与单项 01–04，再看组合 05–07。同一 checkpoint 同时报 Mask AP50/AP50-95、
tiny 匹配/召回、P≥0.90 召回、背景误检、split/merge 和端到端速度。
boundary_iou25_mean 是匹配实例上的双侧边界带诊断，不是官方 Boundary IoU/Boundary AP。
最终方法与控制至少三个 seed；未知重复方差时，不能设“超过 0.5 点就显著”等武断阈值。

## 验证与证据

单元测试覆盖八组官方入口、方形/矩形前向、空/非空反向、预训练映射和 checkpoint；
另测 phase 组合、PMCE padding、确定性 FLOPs 有限输入和算子预检。
当前版验证见 [20260913验证记录](E_V7_REVIEW_20260913/验证记录.md)。
昨日旧版记录保留在 [20260912验证记录](E_V7_VALIDATION_20260912.md)，不能当作当前版测试成绩。
失败与修复日志保存在 E:/mastercode/_work/citrus_ev7_review_20260912/，未覆盖失败证据。

## 一手技术依据

- [PyTorch 2.8 torch.empty](https://docs.pytorch.org/docs/2.8/generated/torch.empty.html)：确定性与未初始化内存填充同时启用时，浮点 empty 内容为 NaN；本次修复依据。
- [Torchvision deform_conv2d](https://docs.pytorch.org/vision/main/generated/torchvision.ops.deform_conv2d.html)：mask 对应调制可变形卷积，需要匹配的编译后端。

旧设计中的跨数据集论文高分、未经本轮复核的 GPU 毫秒值，以及通用“6px 果实 IoU 上限”
均不作为本任务证据。本次是八组实现的正确性和可比较性复审，不是新一轮完整文献综述。

## 第二轮文献与代码调研（历史笔记，非本轮逐项核验）

调研报告全文存 `E:\mastercode\_work\lit_survey_r2\`（report_A 果梗+农业域、report_B 方法+工具），
仓库索引与收录理由见 `Desktop\github\调研索引_ROUND2.md`。以下只记录对 V8 候选池有约束力的结论。

### 已被代码级核实的新机制

- **STAL 候选选择补丁**（上游 `ultralytics/utils/tal.py` `select_candidates_in_gts`）：GT 的 w/h 小于
  stride 时在候选阶段被抬升到 stride_val，但不保证最终分配正样本非零。
  **更正：本 fork 的 TAL 已含此段**（`ultralytics/utils/tal.py:305-311`，所有 V5–V7 训练一直生效），
  并不证明tiny候选一定充分；P2后置的尺度错误本次另行修复，匹配竞争仍待诊断。
- **Matrix-NMS**（`mmdet/models/layers/matrix_nms.py`）：按 mask IoU（非 box IoU）做高斯衰减抑制。
  邻接果 box 重叠高而 mask 重叠低，对应 31% 接触实例的后处理误删；纯推理侧，可在既有 checkpoint
  上无训练验证。上游 ultralytics 主线无 mask-NMS，属可声明的空缺点。
- **MaskIoU 重打分**（`mmdet/models/roi_heads/mask_heads/maskiou_head.py`）：回归预测掩膜与 GT 的
  IoU 并乘入分数重排，针对 AP 高阈值段的分类分-掩膜质量错位。
- **DCT 掩膜表示**（`DCT-Mask/projects/DCT_Mask/dct_mask/mask_encoding.py`）：zigzag DCT 系数向量回归、
  iDCT 解码到 128²，掩膜分辨率与 proto stride 解耦。注意是频域**表示**，与已试的频域滤波融合不同。
- **YOLO26 内置蒸馏**（上游 `nn/distill_model.py`）：FeatureHook + 双层 1×1 projector，自动探测
  Detect 头输入层索引；为 V8 蒸馏选项提供同生态实现模板。
- **YOLO26 Pose26/RLELoss**（`nn/modules/head.py`、`utils/loss.py`）：归一化流 σ 关键点输出——
  论文二果梗点“可定位概率/拒识”的同生态先例；StarBL-YOLO 仓库核实其 KeypointLoss 为 masked L1
  （OKS 被注释），单点采摘任务 OKS→L1 的农业先例成立。

### 被否或降权的方向（有实证依据）

- **零样本 SAM 教师**：同色/伪装/植物场景 auto 模式 mask precision 实测 0.29–0.63
  （SAMCOD、Leaf-Only-SAM、梨树标注评测、MinneApple SAM3 对照等多源）。可行形态仅限
  “GT/检测框 prompt + 掩膜质量门控”的掩膜精修与扩标，不做 auto 伪标签。
- **纯半监督**：~1k 图且无额外未标注数据时 SSOD 增益证据弱（Practical Insights into SSOD 复测）；
  仅当获得未标注野外图时 DenseTeacher 式 dense 伪标签才有意义。
- **RF-DETR/SAM 系部署**：参数量超预算一个数量级，只作对照或教师，不进主链路。

### V8 候选池（按成本与证据强度重排）

| 层级 | 候选 | 成本 | 依据 |
|---|---|---|---|
| T0 先行验证 | Matrix-NMS 后处理；copy_paste+小目标定向过采样 | 分别增加后处理/训练成本 | NMS可用冻结权重测；copy-paste需重训；STAL并非候选充分性的证明 |
| T1 轻量结构 | 质量打分已存在；DCT掩膜头；易错点精修 | 极小增量 | 直击 AP50→95 与深凹轮廓 |
| T2 训练策略 | MGD/FreeKD 或 DistillModel 式特征蒸馏；box-supervised 扩标（BoxInstSeg） | 零推理、训练成本中 | 965 图约束下的标准出路 |
| 论文二锚点 | Pose `kpt_shape=[1,3]` + L1 + RLE σ 拒识 + 掩膜条件化（MaskPose 先例） | — | 柑橘采摘点关键点范式经核实为空白 |

### 新增警示（写论文时引用）

随机 copy-paste 存在负结果先例（Dvornik ECCV'18），粘贴边缘 artifact 可成捷径特征（DeePaste）；
DINOv3 冻结特征对 <16px 级小目标受 patch 离散化限制（蓝莓评测）；YOLO 系在 IoU 阈值上升时掉分
48-50pt 的边界质量诊断（MinneApple SAM3 对照）与本任务 AP50→95 缺口同型。
