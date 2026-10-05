# J 系列验证记录

**2026-10-02 更新：** J 已按用户要求改回 I 系列 `EV6Validator` 原生每轮验证；当前验证报告见 [J_NATIVE_VALIDATION_REPAIR_20261002.md](../J_NATIVE_VALIDATION_REPAIR_20261002.md)。以下公共 COCO smoke 记录保留为历史验证，不代表当前训练方式。

日期：2026-10-01。验证对象是当前主副本，不是服务器的其他 Ultralytics 安装。

## 已执行且通过

| 验证 | 实际范围 / 结果 |
|---|---|
| YAML 入口 | 20 个 `YOLO(yaml, task="segment")` 均成功构建 |
| 前向 / 反向 | 全 20 组、96×128 非正方形输入；训练输出、真实 criterion、有限损失及梯度、推理输出契约 |
| 新头部风险 | 空 GT / non-overlap mask / 可见凹口 / 32 原型 / checkpoint 独立重载 |
| 历史锚点 | J01 的 backbone/head/scales/scale 与现有 I20 YAML 逐项一致；nc 固定为 1，不声称重放历史权重 |
| 数学数值测试 | 相位重排逆与范数、融合凸组合、差分常值抵消、key 去均值 softmax 不变反例 |
| 符号检查 | SymPy 对融合导数、Dice 导数、softmax 平移不变性化简残差均为 0 |
| 官方代价入口 | RGB 输入通道与 stem 内部 20 通道区分，get_flops 能运行；另计 JContext 矩阵乘法 |
| 入口全量 dry-run | all 队列 20 组全部构建；未读或改正式数据，未启动正式训练 |
| 联合回归（含测速新增项） | **92 passed / 8 skipped / 5 warnings** |
| 隔离端到端 | 4 张训练 / 2 张验证，J00 与 J16 各 1 epoch 前台串行；公共原图 COCO 评价、EMA 最佳权重选择、重载、真实 PR 统计完成 |
| 最后改动复核 | `smoke_full_v4` 对 J16 再跑 1 epoch，通过；CPU_THREADS=2 的回调生效并写进 runtime.json；源码摘要为 64 位 SHA256 |
| 已完成跳过 | 用相同参数和代码再启动 `smoke_full_v4`，打印 SKIP，不重训、不覆盖已有权重 |
| 测速入口 | J00/J11/J17 的96输入、batch2 CPU前向、融合前后输出一致性、损失反向和优化器步骤通过；没有GPU速度结论 |
| 图件 | SVG 为合法 XML；公式与结构以代码和数学文档为准，AI PNG 仅为版式预览 |

J 系列模型自身现有 33 项测试，测速新增4项；联合回归还覆盖 I_V4 原生模块和公共 baseline 验证适配器。联合检查不等于把所有历史代码逐行审计过。

## 命令和原始记录

在主目录执行，依赖使用本机隔离 CPU 栈：Python3.9.13、Torch2.8.0+cpu、NumPy1.26.4。

```powershell
$env:PYTHONPATH='E:\mastercode\_work\baseline_common_validation_20260930\cpu_deps;E:\mastercode\1_SEVER\code\ultralytics-main-new'
& E:/AppInstallion/0_4_annaconda/python.exe -m pytest tests/test_citrus_j.py tests/test_citrus_j_benchmark.py tests/test_citrus_i_v4_native.py baseline_amp/test_common_validation.py baseline_amp/test_inference_runtime.py --junitxml=E:/mastercode/_work/20261001_j/j_regression_lightweight.xml -q
& E:/AppInstallion/0_4_annaconda/python.exe scripts/smoke_citrus_j.py --data E:/mastercode/_work/citrus_native_smoke_dataset/data.yaml --project E:/mastercode/_work/20261001_j/smoke_full_v4
```

端到端完整输出：`E:/mastercode/_work/20261001_j/smoke_full_v2/`（两组串行）和 `smoke_full_v4/`（最终 J16）。`J_SMOKE_NOT_RESEARCH` 协议有意使用128输入、batch2、workers0、零 warmup、关闭增强，不混入任何科研排名。小样本 AP=0 是预期的短跑输出，不是待发表的结果。

## 跳过和警告

8 个跳过属于公共 baseline 的可选环境测试：6 个缺少 mmengine，1 个需要服务器 CUDA，1 个需要固定版本官方源码；**J 自身测试未跳过**。5 个警告含4个 THOP distutils 弃用警告，以及旧 SciPy 对 NumPy1.26.4 的版本范围提醒。它们未使当前测试失败，也不代表服务器全环境已验证。

RAM cache 按用户要求为 True；框架的不可逐位复现提示已保留。最初 smoke 的自动加 `-2` 输出目录风险已修复：只允许复用当前 evaluator 自己新建的目录，任何预先存在的半途/冲突运行仍拒绝覆盖。

## 未验证，不可提前声称

- 服务器 Torch1.13.1/CUDA、实际 GPU 吞吐、显存与整轮训练时间。
- 20 组正式50轮/300轮的召回和 mAP；没有 J 系列正式精度结果。
- 优于历史 G10 或当前最好方法；跨数据成员/训练协议旧峰值不能用来直接归因。
- 三 seed 稳定收益、独立测试集及 split/merge / tiny / low-contrast 子集提升。
- 论文新意充分性和期刊录用；数学性质不是精度增长或闭环稳定性证明。

下一步是同配方筛选与真实 GPU 测速，而不是继续叠加未经验证的模块。
