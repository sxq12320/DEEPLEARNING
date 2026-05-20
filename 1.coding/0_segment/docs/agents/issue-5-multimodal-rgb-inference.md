# Issue #5：多模态训练 + RGB 单模态推理

**标签**: `AI可接手`

---

## What to build

利用 Depth 在训练时提供额外几何信息，通过知识蒸馏将 Depth 分支的知识迁移到 RGB 分支，实现推理时不依赖 Depth，只用 RGB 就能达到接近双模态的效果。

## 具体操作

### 1. 修改 TSDualBackbone 支持深度开关
在 `models/backbones.py` 的 `TSDualBackbone` 中增加 `use_depth` 参数：
- `forward(rgb, depth, use_depth=True)` — 双分支正常前向
- `forward(rgb, depth, use_depth=False)` — 只走 RGB 分支，跳过 depth_stem/stage/exchange/fusion

### 2. 新增知识蒸馏损失
在 `engine/losses.py` 中新增两个蒸馏损失：
- **结构蒸馏** `StructureDistillationLoss`：冻结 Depth 分支特征，用 MSE 约束 RGB 分支特征向 Depth 分支对齐。`loss_struct = MSE(rgb_feat, depth_feat.detach())`，对 P2/P3/P4 三层分别计算取均值
- **响应蒸馏** `ResponseDistillationLoss`：用 KL divergence 约束 RGB 分支和 Depth 分支输出的 mask 概率分布一致。`loss_resp = KL(sigmoid(rgb_logits), sigmoid(depth_logits).detach())`

### 3. 实现两阶段训练策略
修改 `train.py` 支持三阶段训练：
- **Phase 0（可选）**: 纯 RGB baseline，不涉及 depth
- **Phase 1**: 双分支联合训练（RGB+Depth → SegDetLoss），收敛后保存 `phase1.pt`
- **Phase 2**: 加载 `phase1.pt`，冻结 Depth 分支，用 `SegDetLoss + α·StructureDistillationLoss + β·ResponseDistillationLoss` 微调 RGB 分支。`α=0.1, β=0.05` 默认
- 两阶段通过 `--phase 1|2` 命令行参数控制

### 4. 推理
推理时调用 `model(rgb, depth=None, use_depth=False)`，只走 RGB 分支，不依赖 depth 输入。

### 5. 对比实验

| 实验 | 训练 | 推理 | 预期 |
|---|---|---|---|
| RGB-only baseline | 只用 RGB | RGB | IoU 基准 |
| RGB+Depth 联合 | 双分支 | RGB+Depth | IoU 上限 |
| RGB+Depth 蒸馏 | Phase1→2 | 仅 RGB | 接近联合 |

## Acceptance criteria

- [ ] 蒸馏后 RGB-only 推理 IoU >= 联合推理 IoU 的 95%
- [ ] RGB-only 推理速度与单分支一致（不增加额外计算）
- [ ] 蒸馏后 RGB-only IoU > 未蒸馏纯 RGB 训练的 IoU（证明 Depth 知识成功迁移）
- [ ] `python train.py --model-type ts_dual --phase 2 --resume phase1.pt` 可正常训练
- [ ] `python infer.py --model best.pt --image test.jpg` 可纯 RGB 推理

## Blocked by

None - 可立即开始
