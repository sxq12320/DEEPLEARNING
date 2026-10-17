# J_V6 实现与验收边界

2026-10-17。状态：**10个YAML、正式网络/损失/数据/EV6适配、前台批运行已实现；尚无正式300epoch结果。**

当前实现为**R2正确性修订**，详见[J_V6_R2_FIXES_20261017.md](J_V6_R2_FIXES_20261017.md)。ARIS第一轮方案评审为6.00/10、REVISE，并非科学有效性通过。此前无标记的enabled-J6权重语义不兼容，不能直接续训、验证、预测或通过`.load()`迁移；从当前YAML在新PROJECT训练，不覆盖旧结果。

运行先看 [J_V6_RUN_20261017.md](J_V6_RUN_20261017.md)。入口为主代码根目录 `RUN_CITRUS_J6.py`，模型在 `0_orange_yaml/J_V6_series/`，已登记 `MODEL_INDEX.csv`。不修改已有系列、数据标签、划分或baseline程序。

## 实际结构，不夸大为全新基础理论

- **全局模型**：仍是YOLO11n-seg图，保留P3/P4/P5及原box/cls/DFL/proto监督。不是声称重新发明骨干。
- **观测/计算结构变化**：从同一原图提供全局640与高分1280；高分只经过窄共享前缀，保留stride2/4特征。新增独立发现入口不依赖global框。默认最多16个新候选，每图最多32个局部实例。
- **实例头**：在同一世界坐标ROI中读取细节、语义、本实例和有效邻实例logit，一次有符号残差；全局候选只投影残差，不把原mask重新压成16²，保证零修正的精确退化。
- **集合算法**：质量重排＋有效共同域的对称身份关系；关系稀疏边最多128，局部候选最多32，有限greedy/add/remove/one-swap，不用指数穷举部署，也无最优/AP保证。最终保留原max_det=300合同。
- **未改动**：不添加Mamba、自定义CUDA扩展、额外图像模态、PID优化器、amodal圆形/凸性补全；不通过改PR端点或EV6积分制造涨点。

所有新增臂使用同样宽度及参数容量，只切换信息/监督/推理机制；J96关闭stride2细节的读取，而不是删除产生D4所需的前缀。J95只扰乱邻身份，保持几何可用域，避免把“邻信息不同”混同于“可用信息数量不同”。

## 与前期构想的两项明确区别

1. **gain为在线、in-sample的新实例有限模型代理**：R2的source1训练/推理均从零粗logit开始，按其实际assigned scout cell读取gain，监督detached `(Y-0.5)^2-(Y-p_local)^2`。未assigned的source1及无对应cell的背景ROI不监督gain。它不是相对global mask的增益、cross-fitted信息价值或AP保证，只改固定预算候选排序，不减少高分前缀计算。J91/J98和J99/J94分别隔离gain因素，失败可直接不用。
2. **共同域是当前显式logit的有效预测域**：共享前缀只读取一幅完整高图，不是多个独立crop图像。全图输入覆盖、ROI mask有效域、果实可见掩膜是三个不同概念。当前关系比较使用有效预测域交集；不把ROI bbox声称为真实输入视图边界或amodal轮廓。共同前景不足时unknown退避，特别小的mask在128²关系采样格上可能没有充分证据；不能因此声称已经解决tiny重复。

上述版本是可实验的机制实现，而非三项原创已经成立。若后续证明收益仅来自额外原像素、mask质量重排或普通一元修边，就据此收缩论文贡献。

## 监督与优化

全局原损失5项不变，新增4项独立记录：

| 项 | 实际定义 | 外层权重 |
|---|---|---:|
| scout_loss | 关键点存在focal＋偏移SmoothL1＋log宽高SmoothL1 | 0.25 |
| local_loss | per-ROI BCE＋0.25×mask质量BCE | 0.5 |
| identity_loss | 有效共同域且已知GT身份的pair BCE | 0.1 |
| gain_loss | 带符号、对齐动作的detached Brier风险差SmoothL1 | 0.1 |

J97仅在局部mask项中提高小实例权重（上限2），与J91配对，不把已存在的面积归一化重新包装为新损失。

训练每图最多14个GT提供global/new两种局部候选，其余GT仍由原全局损失监督；新增4个当前scout低GT框重叠负候选，遵循普通完整标注监督假设，不声称抗漏标。身份负例仅由两个已知且不同的GT产生，未匹配实例不强标“不同身份”。重复正例使用不同域大小与小范围训练抖动，不只学习bbox IoU=1的捷径。

GT mask在原栅格或ROI采样后消失时，新增local/quality/identity/gain监督设为unknown；保留其scout存在/box监督。原全局损失保持原样，不声称已修复所有tiny栅格问题。scout单槽位置冲突尝试周围8格；存在性cell由可见点分配，无界offset统一表达GT框中心相对该cell的偏移，而不是把可见点误当框中心。仍无法分配者明确计数；对应source1局部/质量/身份/gain监督为unknown。

R2发现候选采用存在性阈值＋稳定cell topK，不再使用会使相邻正格互相压制的3×3峰值门。宽高的训练/解码下限统一为normalized `1e-5`，不再强制至少一cell。该下限仍是截断，不是任意小框无损保证；取消峰值抑制也可能增加重复候选并挤占预算，须正式实验评估。

AMP=False、AdamW、lr0=.001、其他共同配方沿J5/J2冻结；初始化scratch。对比必须用本次同配方J90，不能直接拿旧pretrained或AMP不同的78/83成绩当因果对照。

## 特别修复的工程风险

- GT `[0,1]`世界采样坐标显式转换为grid_sample的`[-1,1]`，有解析数值oracle，不只检查shape。
- 邻系数在本果ROI的域外旁瓣不作为有效证据，返回rival_known；错误邻对照保持可用域不变。
- CPU原全局候选数与局部子集数不同，不改变mask裁边round/float策略；4和64个候选均验证零修正恒等。
- 只有成功optimizer step才提交aux/gain/identity readiness；AMP跳步丢弃pending，不把计算loss当成学会。
- 通用EMA只更新浮点，J6精确同步整数/布尔状态，防止训练后验证/保存模型永远base-only。
- 默认预测输出有界显示mask，不为8000×10000大图的300候选隐式分配几十GB；显式retina才恢复原图栅格。EV6评价仍原input/2。
- 统一前台串行，不需要nohup，不启动其他GPU；中断/异常不继续下一模型。partial结果不覆盖，续训必须显式且含optimizer状态。
- R2修复可见点/框中心、sub-cell宽高、相邻正格、source1输入及gain采样位置的合同不一致，增加旧权重语义保护；这是正确性修复，不申报为独立方法创新。

## 数量与成本

未融合参数：J90为2,842,803；新增臂均为2,847,668，增加4,865（约0.171%）。THOP的640全局/1280高分**forward估算**约10.356→10.675 GFLOPs。

该数值**不包括**显式ROI解码、mask组合/采样/投影、NMS和集合选择，不是整系统GFLOPs，更不能证明GPU更快。本设计不比原始全局YOLO少算，而是尝试以少量增量代替多次完整切片编码。需要服务器同硬件、同像素预算的端到端median/P95、显存和AP/R曲线验收。

## 验证状态

本机Python3.9.13、Torch2.8.0+cpu，无CUDA。已验证10 YAML构建/三尺度/矩形前向、真实目标loss反向、数据及EV6合同、EMA/checkpoint恢复、官方YOLO三入口与调度。

首版已使用原数据**副本**执行J90与J94各1epoch前台串行训练；该旧权重属于修复前语义，不能用作R2权重。R2另有208项回归全部通过（4条THOP依赖弃用警告），并在独立工作目录完成真实副本smoke：`E:/mastercode/_work/20261017_aris_j94/real_smoke_20261017_131410/verification.json`。J90/J94各一轮训练、保存重载、独立val和predict均通过；不将首版测试冒充修复后验收。两张train/两张val的一轮训练仅为工程检查，不是精度实验。

本机Matplotlib与NumPy版本不兼容，真实smoke显式`plots=False`，未修改用户环境；正式PLOTS默认True且提前检查绘图依赖。尚未验证CUDA实测速度/显存、服务器旧Torch环境、正式300epoch精度、三seed稳定性或论文原创性。

## 需要的同权重后验对照

`eval_citrus_j6.py`支持`full,quality_only,global,no_gain,no_neighbor`。其中full与quality_only保留相同局部观测、候选与质量重排，只关闭pair选择，可防止把质量重排的收益都归为关系机制。不会强行把未训练关系标记为ready。

```bash
python eval_citrus_j6.py --weights /path/to/J94_full_seed42/weights/best_mask.pt --data /path/to/data.yaml --device 1 --project /path/to/NEW_J6_ABLATION --modes full,quality_only,global
```

每种mode单独保存EV6结果、干预元数据与同权重汇总CSV。这是推理干预，不替代重新训练的结构消融和最终三seed实验。
