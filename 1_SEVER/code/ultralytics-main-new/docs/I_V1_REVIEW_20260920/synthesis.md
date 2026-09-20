# Cross-review synthesis（审后综合；不向审阅人展示）

2026-09-20。输入为 `REVIEW_PACKET.md` 指定的同一不可变证据包，以及已冻结的
`reviewer1.md`、`reviewer2.md`、`reviewer3.md` 三份互盲报告。
本综合不修改任何已冻结报告，不引入证据包之外的新事实；重复的关切按底层问题合并到综合键。
判定对象是当前成果能否支撑一篇"轻量高精度 RGB 未成熟柑橘实例分割"应用论文，
重点为极小果、叶片混淆与召回。不以 Nature 的广泛影响门槛要求应用研究。

## Review setup

- **Input scope**：V12 六个 seed42 完成运行 + 实现 + 配置 + 本地正式划分审计；非完整论文。
- **Assessment boundary**：仅评估证据链能否支撑正式论文主张；不预测编辑决定。
- **Shared manuscript claim summary**：轻量高精度 RGB 实例分割，声称改善极小果召回与叶果混淆。
- **Visible evidence base**：`V12_metrics.csv`、`checks.json`、`PR_evidence.json`、`audit.json`、
  六运行的 completed/loaded/initialization/paired 记录、`citrus_e_v12.py` 及父链实现、
  grouped_dedup 正式 manifest 与 audit。
- **Missing materials affecting confidence**：服务器逐图内容哈希、历史 Python 实现快照
  （`completed.json.git` = unavailable）、V12_08 sum-evidence 结果、同协议官方基线、
  三 seed 重复、独立测试、错误类型人工标注。

## Consensus strengths（≥2 报告独立提出）

- 六臂均完成 300 epoch，内部参数差异仅限 model/name/save_dir，AMP 均关闭；同批比较基础成立。
- `initialization_transfer.json` 实测初始化迁移；V12_03 与 V12_04 参数量完全相同（2,283,950），
  构成有意义的路由配对对照。
- 实现对自身边界克制（非 PID、无稳定性保证、承认共享骨干与 TAL 的梯度耦合、
  零起始约束的是增益而非特征幅值），审稿口径一致认可应保留。
- 诊断维度超出单一 AP：tiny 匹配数/161、固定阈值背景错误桶、R90、经验 PR、实测延迟，
  使精度-召回-成本取舍可见。
- 可复现的最强正向事实：切片+合并显著提高极小实例检出
  （03：global 33/161 → fine 90/161），识别路由伴随背景错误减少（global 144→108）。

## Consensus blocking concerns（≥2 报告独立提出）

### S-B1 实际数据成员与正式划分不一致（R1-M1 / R2-M1 / R3-M1，三报告一致 Blocking）

三报告独立核对：运行记录 676/193 张但实例数为 train 4,324 / val 1,049，
正式审计为 4,120 / 1,181。按图像名映射正式 manifest，V12_03 训练集中 126 张属正式 val、
57 张属正式 test；paired 验证 193 张中 123 张属正式 train、25 张属正式 test；
按 group_id 计 94 个组跨 split。组交叉结论依赖同名图对应同一原图（服务器内容哈希未核对），
但已足以证明未遵守正式划分。**后果：当前 val 结果不可作为泛化证据；
这批权重已在正式 test 的 57 张图上训练过，不能在其上补评估恢复独立性。**

### S-B2 单 seed 验证集筛选不构成稳定优势（R1-M3 / R2-M4 / R3-M2，三报告一致 Blocking）

六臂全为 seed42，`best_mask.pt` 由验证指标选择；无独立测试、无训练随机性区间、
存在多臂多推理模式的选择偏差。last-20 均值与曲线平稳不能替代 seed 间不确定性。
02→03 global AP 差仅 0.183pp、fine 差 0.841pp，处于无法排除噪声的量级。

### S-B3 核心机制未获鉴别性证据（R1-M2 / R2-M2 / R3-M3，三报告一致 Blocking）

分类专用路由未一致改善极小果召回（tiny 匹配 33 vs 35、69 vs 74、90 vs 89，/161）；
同参数量的共享路由 04 在三种预算 AP 均高于 03（fine 78.309 vs 77.682，fine tiny 97 vs 90）。
差异表示（discrepancy）的必要性未测——设计的 V12_08 sum 对照不在完成六臂内。
可观察的事实是误报、召回、AP、延迟之间的**取舍**，而非机制性全面优势。

### S-B4 轻量-高精度组合主张未建立（R1-M4 Blocking / R2-M5 / R3-M4 Major）

最高 AP 对应 fine 预算（global+9 视图）：03 延迟 76.4→312.2 ms（≈4.09×）、
背景错误 108→339；05 参数更少（2.06M）却更慢（89.6/330.2 ms），参数≠速度。
无同协议官方 YOLO 基线。R1 对"轻量+高精度"联合主张判 blocking；
R2/R3 判 Major 非阻断但显著限制部署论断。综合口径：**在单次前向与完整切片系统
分成两条成本-精度曲线之前，轻量主张不成立。**

## Other consensus major concerns（Major 非阻断）

### S-M5 背景错误桶 ≠ 叶片混淆语义证据（R1-M5 / R2-M3 / R3-M5）

背景桶未区分叶、枝、其他背景、漏标果、标注边缘误差；原始报告亦声明其非语义证明。
需在固定难例子集上人工标注错误类型，在同 precision 或同 recall 工作点比较叶片 FP；
否则措辞限为"固定阈值背景错误减少"。

## Where emphasis differs across reviewers

- **R1（技术可靠性/实验设计）**：最重的成员取证——逐图 membership 算术
  （493/126/57 训练侧、123/45/25 验证侧）与历史实现快照缺失列为独立问题。
- **R2（新颖性/机制）**：强调新颖性定位不完整——TOOD/PIDNet/FreqFusion 仅灵感来源名，
  需补最接近工作的逐操作差异；差异 vs 求和的必要性对照缺失是机制判定的硬伤。
- **R3（主张边界/应用成熟度）**：主张收窄路线——可保留措辞为"切片合并提高极小实例检出、
  识别路由显示背景错误减少、存在计算与阈值取舍"，其余均须降级。
- **分歧点**：blocking 校准——R1 标 4 项 blocking、R2 标 4 项、R3 标 3 项；
  差异在轻量-效率主张是否阻断（仅当论文主张"轻量+高精度"联合时阻断）。
  三报告对 B1/B2/B3 的 blocking 判定完全一致，属真实共识而非措辞相似。

## Minor revision checklist（合并且非阻断）

- 统一指标口径与单位：训练验证器峰值 ap 与 common-raster global_ap 分列；
  rmax（0–1）与 r90/AP（%）统一；tiny 写为"极小实例匹配数/161"而非 AP；
  每列标注 checkpoint、split、mask raster、置信度、匹配 IoU、视图预算。
  （R1-m1 / R2-m1 / R3-m1 三报告一致）
- 溯源补齐：归档服务器 `implementation_sha256.json`、依赖版本、命令、数据快照；
  `completed.json.git` 为 unavailable 时不得把当前源码视同已核实历史行为。

## Broad-interest / significance readout

应用意义成立：未成熟果实的极小目标漏检与绿色背景混淆是果园机器人感知的真实瓶颈，
农业视觉与轻量实例分割读者群明确。当前可成立的贡献雏形是**任务特定的识别-细节分流
加切片-合并的精度-成本权衡方法**，而非"已解决小目标+叶果混淆的完整方案"。

## Most important issues to resolve before a strong case is established（优先级排序）

1. **数据身份核验与重跑**：导出服务器逐图/逐标签哈希对照 grouped_dedup manifest；
   确认后在正式划分上、从同一非本任务预训练权重，三 seed（42/43/44）重跑
   官方主基线（YOLO11n-seg）与最终方法，测试集仅末次使用、按采集组重采样估计区间。
2. **冻结主指标与工作点**：预先声明 mask AP50-95（固定视图预算）、P≥0.9 处 recall、
   tiny 匹配率与端到端延迟为主指标；模型选择（val）与最终估计（test）分离报告。
3. **补齐机制对照**：sum vs discrepancy（V12_08 或 I 系等价臂）、recognition on/off、
   共享 vs 分类路由；效应未稳定则不把该机制列为中心贡献。
4. **错误类型人工标注**：固定难例子集盲标叶/枝/其他背景/漏标/边界，报告样本量与不确定性。
5. **系统级成本报告**：单次前向与完整切片系统分列两条 AP-成本曲线；
   同硬件同计时边界报告 Params、GFLOPs、中位及尾部延迟、显存。

## Risk / unsupported claims（当前材料不得声称）

- grouped_dedup 正式划分上的独立泛化优势（数据成员不对应，S-B1）。
- 识别专用路由全面优于共享路由（04 在全部预算 AP 更高，S-B3）。
- 极小果召回已稳定改善（三种预算不一致，且切片收益不可归因于识别分支）。
- "解决叶片混淆/同色识别"（背景桶无语义构成，S-M5）。
- 接近 97% 的实用召回（fine rmax 96.6% 对应 PR 末端 precision ≈3%，非工作点）。
- 小参数必然更快（05 反例；切片系统成本未计入）。
- 差异表示为必要机制（sum 对照缺失）、颜色不变性、控制论稳定性、边缘实时性、
  下游套袋成功率。
- I_V1 任何精度或效率主张：已实现并通过 42 项契约测试与父模型逐参数重放，
  但**零训练证据**；其筛选必须运行在核验后的正式划分上，否则证据将再次作废。

## Readiness verdict

**NOT READY（未达论文产出水平），但路径明确且不需要推倒方向。**
当前材料的合法定位：有价值的单 seed 探索性筛选结果 + 一条可检验的机制假设
（识别/细节分流 + 双原型同步）。论文成立的窄路径已列明：先修复数据证据链（S-B1），
再用冻结协议完成基线×最终方法三 seed（S-B2），并把中心主张收窄到有对照支撑的效应
（S-B3 的胜者决定是"识别路由控误报"还是"双原型解码"还是"切片系统的精度-成本方案"）。
在 S-B1/S-B2 关闭前，任何结构新颖性主张都不具备进入论文的证据资格。
