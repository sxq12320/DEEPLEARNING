# Baseline 重审与 J_V4：证据保留、配方分轨和可验证的压缩

日期：2026-10-09。状态：源码与CPU契约验证；未完成新模型正式训练、CUDA短训练或服务器测速。

## 1. 先说结论

已有七种实例分割基线覆盖了单阶段原型、两阶段ROI、位置式与Transformer查询方法，问题不只是“数量不够”，还包括初始化、配方强度及比较边界。原来的78版固定scratch配方仍保留；它适合受控对照，但不能代表所有方法最合适的迁移学习效果。因此新增**独立的官方COCO预训练/原生优化轨道**，不悄悄改旧实验，也不混进同一表。

历史结果最强的机制证据是分辨率分配及跨视图选择，不是所有新骨干均有效。J3只改监督、部署图仍为YOLO11n。J4保留实测V12窄细节与stride2原型，依次检验颈部返程删除、投影式融合、深层压缩、跨尺度预测权重共享，并配实际有限视图推理。**J4尚无涨点成绩，不称“完美模型”。**

## 2. 基线两条轨道

| 事项 | 保留的78版 | 新paper transfer轨道 |
|---|---|---|
| 入口 | RUN_CITRUS_BASELINES_AMP.py / RUN_BASELINE_OTHER_MODELS.py | RUN_PAPER_BASELINES.py |
| 初始化 | 显式scratch | 对应官方COCO权重，不失败回退scratch |
| YOLO优化 | AdamW，lr0=.001 | 官方optimizer=auto；记录解析后的实际优化器/LR |
| RTMDet | 通用scratch配方 | 官方AdamW/paramwise/调度；按单卡实际batch缩放LR和eta_min |
| Mask R-CNN / SOLOv2 | 通用scratch配方 | 官方SGD与各自调度；按参考batch缩放LR |
| RF-DETR | lr与encoder lr均.001，scratch | pinned1.4.0.post0：lr=.0001、encoder=.00015、wd=.0001、layer/component decay=.8/.7 |
| AMP | 1/0配对 | 1/0配对，配方和实际初始state保持一致 |
| 验证 | common原图COCO（新受控入口） | 完全相同的common原图COCO，不换成低分辨率终测 |
| 输出 | 原输出协议 | 独立project、transfer_任务名前缀、独立protocol ID |

不是原样复刻COCO论文：柑橘固定640输入（RF624）、单类、固定epoch预算、无删除小果的裁剪/过滤、common验证都属于明确的适配。保留原始网络、损失和分配方法。RF原生AMP是BF16，YOLO/MM通常FP16，必须注明，而不是把AMP=1当成同一种精度。

默认batch下，单卡RTM参考256→8，base LR .004→.000125，eta_min .0002→.00000625；Mask参考16→2，.02→.0025；SOLO参考16→2，.01→.00125。它们是预登记的线性缩放起点，不保证是小数据最优LR。若要调优，给每种模型相同搜索预算、仅用val选择；不能只给自研模型调参。

官方依据：[MMDet3.3.0配置](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs)、[RF-DETR固定版本配置](https://github.com/roboflow/rf-detr/blob/1.4.0.post0/rfdetr/config.py)、[RF-DETR训练说明](https://rfdetr.roboflow.com/latest/learn/train/)。最新RF说明已经换Lightning，不将其新API塞进旧post0。

### 修复与速度边界

- 修复官方300epoch调度缩为1epoch时`begin=end=1`导致MMEngine不能启动；有效区间改为0→1，保留原官方调度类型。
- RF先加载COCO的91类头，训练时才适配单类；初始化hash改到首次训练batch、首次forward/update之前。
- YOLO载入.pt后若仍传`pretrained=False`会丢掉权重；新轨道使用对应.pt及`pretrained=True`。
- 初始化记录同时保存实际适配后state SHA和原始checkpoint文件SHA。原始COCO head不同于单类适配后的head，两者不能用同一个hash代替。
- 保留此前bounded mask解码、GT/IoU缓存、CPU预取与推理专用插值优化。每epoch quick640，epoch1/每10/最后原图验证；best仅用原图AP，最终统一FP32原图COCO。
- 这些属于工程与配方修正。本次没有GPU测速，**不能声称又缩短了多少分钟**；少做原图验证的频率与加速每次验证是两件事。

在RUN_PAPER_BASELINES.py设置自己的DATA、DEVICE、PYTHONS、PROJECT，然后VSCode右上角Run。先只选一个模型，EPOCHS=1–3在新project短测；再50筛选，最后另开300epoch正式project。Ctrl+C会停止当前队列，不开启下一模型。没有单GPU占用保护，不会关闭他人进程。

## 3. 全量证据，而不是最新系列崇拜

本轮读取311份CSV，内容去重后297条不同轨迹；1,437份JSON和813条配对视图记录。可读args主要seed42，跨系列初始化、AMP、增强、评价不一致，因此不能按最大AP直接排模块。

同权重V12_04、193图/1049实例、公共原图640栅格：

| 方法 | Mask AP50–95/% | AP50/% | tiny命中/161 | 历史中位ms |
|---|---:|---:|---:|---:|
| global | 68.608 | 84.218 | 33 | 93.860 |
| fine+box融合 | 60.274 | 69.458 | 98 | 341.867 |
| fine+crossmask | 69.477 | 84.897 | 96 | 338.779 |
| fine+trustedmask | 78.309 | 89.566 | 97 | 338.716 |

trustedmask不是像素加权融合，也不是闭环控制：crossmask与trustedmask都用同一IoU/IoS片段规则，后者主要将贴内部裁剪边的候选得分乘.5。此差异同时影响选择顺序与评分排序，不能归因为新增轮廓模块。

I05对I00单视图AP50–95只高.114个百分点，AP50反低.150；J自研骨干没有可靠增益。选择性放大的另一个探索协议：等预算约2.508视图时，简单self选择AP75.209、随机72.965；GT oracle约2.808视图保留全切片约95.78%的增量收益，但oracle不能部署。不同协议不混表。

以上都不是正式test。旧成员/单seed只能选方向；分组去重数据需冻结并三seed重跑。当前本机未找到正式数据目录，运行器不会用旧数据静默替代，由用户DATA指定。原标注不改，不删除小目标或缺标签样本。

## 4. J4架构与八组阶梯

| YAML ID | 相对上一级的变化 | 参数/M | 单视图GFLOPs@640 |
|---|---|---:|---:|
| J60_yolo_control | 原YOLO11n受控视图训练参考 | 2.843 | 10.356 |
| J61_v12_replay | V12_04结构/辅助目标重放，scratch | 2.284 | 10.601 |
| J62_no_pan_return | 删除P3下采样返回N4的路径，直接预测P3/N4/C5 | 2.160 | 10.203 |
| J63_projected_direct | 两次源分辨率投影+有界残差融合，代替concat/C3k2重建 | 2.268 | 10.591 |
| J64_narrow_semantics | J63深语义宽256→192 | 1.839 | 10.246 |
| J65_shared_towers | J64窄box/cls/coeff塔，各任务跨尺度共享空间权重 | 1.440 | 9.312 |
| J66_without_auxiliary | J65关tiny/boundary/neighbor/quality目标及质量降分 | 1.440 | 9.312 |
| J67_thin_detail | J65常驻细节宽16→12 | 1.428 | 9.152 |

这些是当前源码CPU构建/THOP，不是训练后速度。J63并不比J62更小，刻意保留为负成本对照；“sum比concat便宜”不是无条件结论。J65对J60参数约减49.33%、计算估计约减10.08%；对V12分别减36.93%、12.16%。保留stride2原型仍使高分辨率mask开销明显，不承诺训练更快。

整体路径：

```text
RGB -> C2 -> C3 -> C4' -> C5 (192宽)
        |      |      |       |
        D0 -> D1 -> D2        semantic context
                      |       |
          C4'+project(C5) -> N4
                  C3+project(N4) -> P3
       P3 / N4 / C5 -> 窄共享空间权重预测塔 -> boxes/classes/coefficients
       C2/stem/D2/P3 -> stride2原型 -> instance masks
```

没有P2候选头，仍8400候选@640；细节流是stride4、原型stride2，两者不能混称小目标检测头。主干保留C3k2浅层和V12深层局部卷积，压缩深层而非未经验证地推倒全部骨干。没有RGB-D、灰度第二重主干、Mamba/Triton/custom CUDA、循环控制器或强制圆形。共享空间权重按三个任务分别建；尺度投影BN和最后输出都不共享，以保留不同尺度统计与分类先验。

J60与J61的差异含结构和辅助目标，不能当纯结构消融；J66是目标+质量排序联合关闭，也不能当独立quality-loss证明。J62→J65逐步成本/结构对照更容易定位失败。J3伪标签单列监督研究，不直接与V12辅助损失混用：其正样本重索引与自定义几何目标当前没有兼容证明。

### 共享参数的实现修复与验证

共享权重不是简单把同一卷积放三遍。为避免THOP累计计数，各尺度使用不同Conv对象但绑定同一Parameter；进一步修复Ultralytics优化器按别名重复注册参数、EMA按state_dict别名重复更新。前者会造成重复优化，后者使三别名的衰减由d变为d³。新增按参数身份/精确存储别名去重，对普通非共享模型保持原行为。共享BN或最终预测层均未采用。

CPU组合契约测试75项通过、3项真实后端测试因环境缺失跳过；另跑完整baseline_amp得到460通过/45跳过，两组有交集不相加。实际J65检查通过checkpoint/strict load/fuse、1/3/3/10次真实前向以及动态ONNX/ORT CPU原始头输出一致性。仅直接ONNX导出，不代表官方Exporter、带NMS导出或TensorRT已验证；CUDA、AMP/DDP、真实训练和服务器时间均未验证。

## 5. 数学：能证明的是算子与代理，不能证明AP必涨

### 5.1 有效尺寸

全图长边缩放L，原尺寸W，目标跨度w，则\(w_g=Lw/W\)。先取比例f的完整裁剪再缩放，\(w_f\simeq w_g/f\)。f=.4时线性约2.5倍。只说明更多原始观测进入网络；不保证模糊果实可识别，也不允许把640图上采样当原像素裁剪。

对固定训练权重，若\(Y\to T(I)\to F_\theta(T(I))\)为Markov链，数据处理不等式给出
\(\mathcal I(Y;F_\theta(T(I)))\leq\mathcal I(Y;T(I))\)。因此更复杂的后续算子没有凭空增加输入观测信息的保证；这不是任何具体8/16px硬下界。

### 5.2 残差不把浅层直接乘到零

沿用显式融合
\[N=\phi_d(C)+\alpha\,U(\phi_s(S)),\quad \alpha=0.5\tanh(a),\quad |\alpha|<0.5.\]
只限制语义增量的系数，不限制卷积范数、总输出幅度或证明稳定性。与\(C\sigma(g(S))\)相比，它保留一条加性局部路径；但投影\(\phi_d\)仍可丢信息，语义也可能错误，所以实际召回必须测。

### 5.3 跨尺度参数复用

L=3尺度、Q=3任务、宽h=48的3×3中间卷积，独立权重为\(QL9h^2\)，共享后为\(Q9h^2\)，节省\(Q(L-1)9h^2=124,416\)个参数。各尺度投影/输出保留独立，实际总参数节约还来自塔宽和算子变更。

共享权重**本身不省FLOPs**：该卷积仍在每层计算，成本为
\[2Q9h^2\sum_{\ell=1}^{L}H_\ell W_\ell.\]
本模型FLOPs降低还依赖窄投影、删除重复重建等，不将“共享参数”误说成“只计算一次”。共享Parameter但不同module实例，避免THOP对复用module累计计数重复汇总；训练参数用去重后的model.parameters()计。

### 5.4 有限视图覆盖代理

令全图可用候选点i权重\(w_i\ge0\)，窗口集合S，则
\[f(S)=\sum_i w_i\mathbf1\{\exists j\in S: i\in W_j\}.\]
边际收益是尚未覆盖的点权重之和，非负且随S增大不增，所以这是单调次模覆盖代理。标准基数约束下贪心有经典近似界；当前九窗口甚至可以枚举。但该代理不是AP：未见候选、叶片误检、截断与评分排序不受该定理保证。[原始理论](https://link.springer.com/article/10.1007/BF01588971)

部署预算K=2时最多1+2=3次完整前向，而all9为10次。若单视图成本F，则总成本近似\((1+K)F+F_{select}+F_{merge}\)。J65约27.936G卷积估计（不含选择/融合），仍明显高于单视图YOLO约10.356G；相对全切片降成本，不是相对单YOLO免费提高召回。

## 6. 文献与源码取舍

本次认真查看官方作者代码并下载DEIMv2、EdgeCrafter到桌面github，记录commit及非商业许可。另读ESOD、QueryDet、RefineMask、RF-DETR、PIDNet、LSNet等。报告见`_work/20261009_paper_draft/resolution_design_literature.md`。

- [ESOD/TIP2024](https://arxiv.org/abs/2407.16424)：高分辨率共享前缀+特征切片，提示区域算力分配；当前版本仍采用原像素裁剪，不冒称复现ESOD。
- [QueryDet/CVPR2022](https://arxiv.org/abs/2103.09136)：粗到细查询和稀疏头；不创造缩小后已不可观测的输入细节。
- [RefineMask/CVPR2021](https://arxiv.org/abs/2104.08569)：已有实例的细边界更新；不替代小果发现。
- [RF-DETR/ICLR2026](https://arxiv.org/abs/2511.09554)：强查询式基线，作为原模型比较，不整网塞进nano成本。
- [EdgeCrafter/TMLR2026](https://arxiv.org/abs/2603.18739)：adapter简化，但后续仍有FPN/PAN，不能借它证明删PAN无损。
- [DEIMv2作者代码](https://github.com/Intellindust-AI-Lab/DEIMv2)：单尺度轻编码和共享bbox头参照，任务/结构不同；本次没有复制其模型实现。

J4是证据驱动的压缩实验，不是已经证明新颖的算法。收益路由若要成为核心方法，下一步应该用train上“fine相对global找回的真实已标实例”监督净增量恢复，显式惩罚新FP，而不是再训练一个密度热图就声称自适应涨点。该学习路由尚未实现，不与当前coverage推理混称。

## 7. 运行与判定

1. 上传当前`1_SEVER/code`，在项目环境使用RUN_CITRUS_J4.py设置DATA/DEVICE/PROJECT；不会要求确认数据指纹或抢占保护。
2. EPOCHS=1–3、ONLY只选J60或J65，新PROJECT做CUDA smoke；cache固定True。
3. 50epoch priority只跑J60/J61/J64/J65；eight-arm screen用于定位neck/head/损失失败，不默认一次耗尽所有300epoch预算。
4. 最后选择真正有益模型，300epoch、三seed、相同正式划分/初始化/AMP/增强/视图预算；J4训练验证仍EV6。
5. 冻结每个模型权重，用eval_citrus_j4_budget.py额外比较global、coverage K2、random K2、all；这是公共640栅格诊断，不替换正式评估。

```bash
python RUN_CITRUS_J4.py --dry-run
python RUN_CITRUS_J4.py
python RUN_PAPER_BASELINES.py --dry-run
python RUN_PAPER_BASELINES.py
python eval_citrus_j4_budget.py --weights /path/best.pt --data /path/data.yaml \
  --output /path/new_budget_diagnostic --device 0 --policy coverage --budget 2
```

eval中device是当前进程的逻辑索引；RUN入口先物理卡绑定→逻辑0。附加推理只执行选中窗口，记录实际视图与P95时间，含磁盘读取/裁剪/回映射/合并，不通过预先算全切片再虚计有限成本。随机控制固定相同K，需重复不同seed；coverage对没有global候选的小果仍可能漏检。

日志中的源码、权重和实际输入哈希仅用于追溯，**不是需要点击确认的数据门槛**。训练记录使用用户DATA；同路径若修改了标签或图片，必须换NEW PROJECT，不能让旧完成标记代表新数据已跑过。上传副本没有.git时也保存本地代码列表/摘要；有.git时另记录commit和工作区状态。

PR末端不设“必须平滑或不归零”的目标。区分插值补零、低阈值海量FP、实测最大召回不足；报告原始TP/FP/FN与R@固定P、尺寸漏检、背景错例。颜色混淆仍需人工确认叶片误检子集；不能把所有background桶都当叶片。

继续保留条件：同预算AP/Recall稳定、small/tiny改善且背景误检/接触split-merge不显著恶化，实测延迟/峰值显存接受。若共享塔损害识别，则保留J64/更简J62；若直接投影更重又不涨点，删除该路线，不为“创新数量”保留。不存在数学保证一区录用或10点必涨。
