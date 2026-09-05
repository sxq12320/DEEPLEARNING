# SAGE 结果审计与 SAGE-v4 设计

## 结论先行

SAGE-v3 不是失败系列，但它提供的是“小幅、可定位的信号”，不是已经证明的大幅涨点。四个模型都实际训练了 300 epoch，目录名中的 `50EP` 已过时。SAGE21 的渐进融合取得本组最高 Mask AP50-95 峰值 66.935、同轮 AP50 83.033；相对历史 G00 的 67.031 仍低 0.096 个百分点。尚无多 seed 结果判断这个小差值是否超出随机波动，两个结果也没有可验证的相同 split manifest，因此不能宣称胜负。

更重要的新发现是：验证可视化同时包含 `IMG_x.jpg` 与 `IMG_x.npy`，混淆矩阵标注数为 2,098，恰好是本地验证集 1,049 个实例的两倍。本地同名划分为 193 张验证图；此前服务器日志报告过 386 张，但尚未拿到本次服务器的完整加载清单，不能单凭路径证明两端成员相同。源码把磁盘 `.npy` 缓存列入 `IMG_FORMATS`，目录枚举会把源图和缓存各算一个样本；训练 85 batch/epoch 也与 676 张源训练图被翻倍、batch=16 相吻合。如果训练目录全部存在配对缓存，去重后每轮约从 85 减到 43 batch；不能据此保证总耗时减半，也不能用它解释所有异常停顿。既往 SAGE 与修复后的结果不可直接比较。

代码现在只在 RGB 目录自动发现时排除“与栅格源图同路径干的 `.npy` 缓存项”，不删除任何文件；NPY-only 数据、显式清单中的 NPY 和四通道数据不受影响。正式新实验必须重新跑一个 SAGE30 control。

这是数据加载器问题，不等于清洗失败，也不等于已经证明 train/val 泄漏。完全均匀地重复验证样本可能几乎不改变 AP，
但会翻倍计数与耗时；训练重复还会改变同一 epoch 的更新步数。因此不能把过去的所有增益或 PR 末端形状都归因于它。

## SAGE-v3 数值

| 模型 | 核心变量 | Mask AP50-95 峰值 | 峰值 epoch | 同轮 AP50 | 官方 best.pt AP50-95 | 尾 20 轮均值 | epoch 中位秒 |
|---|---|---:|---:|---:|---:|---:|---:|
| SAGE20 | P4/P5 轴向形状主干 | 66.621 | 166 | 81.776 | 66.621 | 65.181 | 28.02 |
| SAGE21 | 渐进创新融合 | **66.935** | 77 | **83.033** | **66.935** | 65.220 | 30.05 |
| SAGE22 | SAGE21 + 旧共享拓扑损失 | 66.593 | 77 | 82.680 | 66.274 | 65.568 | 34.57 |
| SAGE23 | 形状主干 + 融合 + 旧辅助损失 | 66.717 | 158 | 82.315 | 66.517 | **65.800** | 36.35 |

“官方 best.pt”按 Box AP50-95 与 Mask AP50-95 的和选择。SAGE22/23 的掩膜峰值不一定在该 checkpoint，因此新脚本同时保存 `best_mask.pt`。完整机器可读统计见 `results/SAGE/_analysis_20260903/all_results_inventory.json`，自动表见同目录 `SAGE_METRICS.md`。

## 历史结果不是同一张排行榜

全目录共读取 144 份含 Mask 指标的 CSV，其中包括整理目录内的复制件，不等于 144 次独立实验。

| 历史信号 | Mask AP50-95 峰值 | 当前可取结论 |
|---|---:|---|
| T04 LSKA topology | 67.367 | 精度候选，但只看到 195 行且辅助损失不同；不能宣称 300 轮公平击败基线 |
| G00 official | 67.031 | 保留为历史参考，修复缓存枚举后重跑同预算控制 |
| T02 G10 retest | 67.115 | 与 G00 仅差 0.084 点，不支持旧 G10 的“大幅领先”能直接复现 |
| T05 lite head | 66.526 | 轻量有价值，但当前数据上并未同时增精度 |
| G03 frequency neck | 65.631 | 不支持继续加频率颈部 |
| G04 deep RepMixer | 66.515 | 不能以更换主干算子本身作为涨点证据 |
| S09 vs S00（另一组划分） | 61.616 vs 60.740 | 组内有正向信号，不能与本次 66--67 的组混算 |
| B06 vs B00（另一组划分） | 61.862 vs 61.396 | 上下文/拓扑轻量化曾有效，但应重新隔离监督与结构因素 |

## 混淆矩阵中的取舍

这些图是 box 混淆矩阵，默认置信度 0.25，不能当作 mask split/merge 统计。按图中数字转录：

| 模型 | TP | FP（背景→果实） | FN（果实→背景） | 相对 SAGE20 |
|---|---:|---:|---:|---|
| SAGE20 | 1,626 | **257** | 472 | 高精度、漏检多 |
| SAGE21 | **1,706** | 553 | **392** | 多找回 80 个，但新增 296 个误检 |
| SAGE22 | 1,650 | 371 | 448 | 较 SAGE21 少 182 FP，也丢 56 TP |
| SAGE23 | 1,664 | 335 | 434 | 较 SAGE21 少 218 FP，也丢 42 TP |

SAGE21 这组组合表现出更高召回，但 SAGE20→21 同时改变主干与颈部，不能仅凭这组对比把收益归给某一个模块。
更清楚的是 SAGE21→22：添加旧共享监督后，误检减少但漏检增加、Mask AP 峰值下降。SAGE22→23 也有值得保留的正信号：
在有辅助监督时加入形状主干，TP 增 14、FP 减 36、Mask AP 峰值提高 0.124 点、尾20均值提高 0.232 点；代价是速度下降。
这提示主干与监督可能存在交互，而非“所有形状模块都无用”。这些都是单 seed、各自选优 checkpoint 的探索证据，尚不是稳定因果结论。
绿色叶片伪装场景需要同时处理候选召回、背景误检和实例边界监督，新版本用独立消融验证它们能否兼顾。

## PR 末端不是绘图故障

Ultralytics 在 `compute_ap()` 中显式给 PR 序列追加精度为 0 的哨兵点，并将最后横坐标补到 recall=1。当模型的最大可达召回小于 1 时，曲线会从实际最大召回掉到 0，再贴着横轴走到 1。未达到的召回区间没有 AP 面积；真正需要改善的是跳崖前的最大召回和高召回区精度，而不是删除哨兵点美化图片。当前四条曲线约在 recall 0.87--0.90 后归零，说明仍有一批在本次最低验证阈值下也无法正确匹配的 GT；只能借助逐尺度 AP、逐图错误和 split/merge 指标判断它们是否为超小果、遮挡果或标注/匹配误差。

## 从模块库取舍

桌面库盘点到 107 个 Python 文件；文件名宣称 CVPR 16、ICCV 5、ECCV 5、AAAI 4、ICLR 3、NeurIPS 3、arXiv 33。文件名不是论文真实性或有效性的证明。阅读候选实现和原论文后，本轮取舍如下：

| 思想 | 取舍 | 在 SAGE-v4 中的处理 |
|---|---|---|
| PIDNet 的 detail/context/boundary 分工与 PagFM 相邻尺度相似性[^pid] | 保留思想，不照搬语义分割三分支 | 语义融合、P2 掩膜细节、几何监督分开；不宣称真正 PID 控制器 |
| ReZero 的小幅残差初始化[^rezero] | 保留 | 新分支在最终投影后用 0.01 channel gain 注入，减轻预训练特征扰动；并非原论文的精确零初始化 |
| BMask R-CNN 的 mask/boundary 相互学习[^bmask] | 部分保留 | 边界只作为独立辅助目标，实例掩膜仍由标准 YOLO loss 负责 |
| MambaOut / 语言模型 Gated CNN[^mambaout][^gate] | 仅作 P4 主干替换消融 | 普通 NCHW 1×1、DW 3×3、门控乘法；无 Mamba、无 NHWC 反复 permute、无全局 attention |
| ConDSeg 的对比驱动动态聚合[^condseg] | 拒绝直接迁移 | 原实现两次 `unfold/fold` 和位置动态核，不放入高分辨率热路径；不移植其算子 |
| 频域、可变形采样、CARAFE、多注意力堆叠 | 本轮拒绝 | 既往频率颈 G03 为负，Light 已暴露实际延迟风险；无新证据前不再堆叠 |

SAGE-v4 的“预测—观测—修正”只保证每个已对齐张量的更新是两个输入之间的凸组合；学习投影和整个网络并没有建立状态空间模型，也没有 Lyapunov 稳定性证明。控制理论在这里提供的是可检验的结构约束，不是包装词。

## SAGE-v4 因果消融

| YAML | 变量 | Params（nc=1） | GFLOPs@640 | 预期回答的问题 |
|---|---|---:|---:|---|
| SAGE30 | 修复数据枚举后的官方控制 | 2.843M | 10.356 | 新协议基线是多少 |
| SAGE31 | 仅有界渐进融合，且只修正 P3 | 2.858M | 10.435 | 能否保留 SAGE21 召回信号而少扰动 P4/P5 |
| SAGE32 | 仅 P2→stride-4 prototype 细节 | 2.845M | 10.448 | 小目标/凹边界是否真正需要直接细节路径 |
| SAGE33 | SAGE31 + SAGE32 | 2.860M | 10.528 | 语义候选与掩膜细节是否互补 |
| SAGE34 | SAGE33 + 独立三通道几何监督 | 2.860M | 10.528 | 无冲突的前景/边界/分隔监督是否改善 AP75/错误类型 |
| SAGE35 | SAGE33 + P4 Gated CNN 主干替换 | **2.824M** | 10.412 | 真正替换 C3k2 是否比保留预训练 CSP 更好 |

SAGE34 不再使用旧四类 softmax 同时派生 query/boundary：前景、边界、相邻实例分隔是可重叠的三个二元目标；分隔监督限制在边界/空隙，避免把相邻果实深层内部标成 separator；BCE/Dice 按图、按通道归一化，空正样本不会使 loss 随像素数爆炸。它仍只是局部几何代理，不保证解决 concavity、split 或 merge。

旧代码中 `citrus_query` 是所有非背景类别相对背景的 log-odds，但 query target 只标记小果中心。
因此同一个果实内部非中心像素，一边被拓扑目标要求为“非背景”，另一边被 query 目标要求为负；两个监督方向不一致。
旧 query target 的面积阈值注释按 P2 stride=4 编写，实际 SAGE21--23 的 logits 位于 P3 stride=8，也会改变“小目标”的定义。
这些是代码可验证的问题，不是凭 PR 图片推测；旧模型实现仍保留以复现实验，新损失独立实现。

## 实验决策

先做 1--3 epoch 冒烟，再跑 SAGE30--34 的 50 epoch 单 seed 筛选；SAGE35 是失去部分预训练迁移的高风险主干消融，资源允许再跑。只有 SAGE33/34 在相同数据清单上超过 SAGE30，且速度可接受，才进入 300 epoch。300 epoch 阶段只跑 SAGE30、胜出的一个方法和现有 T04 候选，三 seed 报均值±标准差。现有拼图不是全验证集逐实例数值预测，本报告尚不能给出 AP_small、Boundary F1、solidity 分层和 split/merge 结论；下一轮必须用 `best_mask.pt` 导出预测后补齐。

筛选门槛是工程决策而非统计显著性：固定预算下 Mask AP50-95 不低于对照，且中位训练 step 时间不超过对照 1.15 倍，再考虑投入长训。即便通过，也不代表最终会提升，更不保证“涨 10 个点”。

## 已完成的工程验证

本机为 CPU（PyTorch 2.8.0+cpu），没有服务器 GPU。旧 SAGE 回归测试 50 项、新 YAML/损失/前台入口等 31 项通过。
SAGE34 和 SAGE35 还使用复制出的 4 张训练图、4 张验证图完成 1 epoch 的标准 YOLO API 烟测（batch=2，imgsz=256，workers=0）；
这是管线测试，参数偏离正式协议，不得作为效果实验。原数据集没有被改写。

本机 NumPy 1.21.6 与已安装 Matplotlib（要求 NumPy≥1.23）不兼容，正式绘图路径的联调在验证绘图时失败。
已为批量入口增加绘图依赖预检，避免到第一轮结束才报错；没有擅自升级整个环境，也没有在正式协议中关闭 plots。
本地烟测使用 plots=False。这项环境限制不代表服务器也有问题，服务器应先执行
`python -c "import numpy, matplotlib.pyplot; print(numpy.__version__)"` 检查所选解释器。

另用 640 输入验证了新批量 main 依次完成 SAGE30、SAGE34 各 1 epoch，保存 completed 标记和两个 best checkpoint，
第二次调用正确跳过两项完成实验。此集成测试仍使用同一 4+4 图 fixture，且仅在测试调用中关闭 plots，不能作为正式效果结果。

独立串行微基准：CPU 2 线程、batch=1、256×256，预热 5 次后取 30 次 forward+loss+backward 中位数。
不包括读取图片、优化器更新、验证、绘图；每模型只有一次测量序列，不能当论文速度结果：

| 模型 | 毫秒/步 | 相对 SAGE30 |
|---|---:|---:|
| SAGE21 | 97.11 | 1.130× |
| SAGE23 | 104.11 | 1.211× |
| SAGE30 | 85.97 | 1.000× |
| SAGE31 | 86.75 | 1.009× |
| SAGE32 | 84.80 | 0.986× |
| SAGE33 | 89.07 | 1.036× |
| SAGE34 | 90.73 | 1.055× |
| SAGE35 | 87.05 | 1.013× |

这些测量只表明本机合成测试未出现 Light 式极端变慢，不能保证 CUDA 上相同排序。服务器可用
`scripts/benchmark_sage_v4.py --device cuda:0 --batch 16 --imgsz 640 --steps 30 --output benchmark.json`
独立复测；同时记录正式训练的 batch 秒数和每轮 validation 秒数。

## 固定超参数与运行

唯一超参数来源为 `protocols/citrus_paper1_formal_v1.yaml`：AdamW、lr0=0.001、lrf=0.01、momentum=0.937、
weight_decay=0.0005、batch=16、imgsz=640、workers=4、AMP=False、deterministic=True、dropout=0、mask_ratio=4。
warmup=3、mosaic=1、close_mosaic=10、copy_paste=0，其余增强、损失增益和评估设置也从该文件读取。
SAGE34 唯一方法损失为 YAML 中 structure_gain=0.1；旧 citrus_query/topology/boundary 等全部显式设零。
不为获得速度而偷偷开 AMP，不使用 optimizer=auto，也不自动缩小 batch。出现资源问题应停止并用明确的新协议重做对照。

VS Code 打开根目录 `RUN_CITRUS_FOREGROUND.py`，选择服务器环境，改 DATA/DEVICE/PROJECT。默认 DRY_RUN=True，
只构建不训练；通过后按前台使用文档先 smoke，再 screen。无需 nohup 或 `&`。先前后台任务不会被新入口自动杀掉，
必须由用户确认停止。前台运行并不会天然减少 GPU/CPU 竞争；入口的串行队列、同用户设备锁和占用检查只能降低误启动风险。

## 资料与代码

[^pid]: Xu et al., PIDNet, CVPR 2023: https://openaccess.thecvf.com/content/CVPR2023/html/Xu_PIDNet_A_Real-Time_Semantic_Segmentation_Network_Inspired_by_PID_Controllers_CVPR_2023_paper.html ; official code: https://github.com/XuJiacong/PIDNet
[^bmask]: Cheng et al., Boundary-preserving Mask R-CNN, ECCV 2020: https://arxiv.org/abs/2007.08921 ; official code: https://github.com/hustvl/BMaskR-CNN
[^rezero]: Bachlechner et al., ReZero, UAI 2021: https://proceedings.mlr.press/v161/bachlechner21a.html ; official code: https://github.com/majumderb/rezero
[^gate]: Dauphin et al., Language Modeling with Gated Convolutional Networks, ICML 2017: https://proceedings.mlr.press/v70/dauphin17a.html
[^mambaout]: Yu & Wang, MambaOut, CVPR 2025: https://openaccess.thecvf.com/content/CVPR2025/html/Yu_MambaOut_Do_We_Really_Need_Mamba_for_Vision_CVPR_2025_paper.html ; official code: https://github.com/yuweihao/MambaOut
[^condseg]: Lei et al., ConDSeg, AAAI 2025: https://arxiv.org/abs/2412.08345 ; official code: https://github.com/Mengqi-Lei/ConDSeg
