# E V7 修订版验证记录

日期：2026-09-12。对应 [复审与运行说明](E_V7_RECONSTRUCTION.md)。

## 已通过

| 检查 | 结果 |
|---|---|
| V6/V7 相关单元回归 | 92 passed，4 条 THOP distutils 弃用警告，44.13 秒 |
| 正式批量脚本 all --dry-run | 八组构建、前向、参数量与 GFLOPs 输出通过，退出码 0 |
| 八组官方 YOLO YAML 入口 | 构建、空/非空目标反向、矩形前向、预训练与保存重载通过 |
| 共享 fine decoder | V7 与 phase decoder 组合，含/不含 P2 均通过 |
| PMCE 常量特征 | 图像边缘未出现由 padding 引入的虚假高频 |
| 确定性 FLOPs | 小尺寸及全尺寸回退均使用有限输入，回归通过 |
| DCN 后端预检查 | 本机 CPU 前向/反向通过，随机数状态保持 |
| 控制组和全组合真实短训练 | 两组前台顺序完成各 1 epoch；各自保存 best、best_mask、best_mask50，重载和独立验证通过 |
| Ruff / 相关 tracked diff 空白检查 | 通过 |

本机环境：Windows，Python 3.9.13，PyTorch 2.8.0+cpu。
真实短训练只用专门的 4 张训练图、2 张验证图和 128 输入，batch=2、workers=0、cache=True、amp=False。
这不是 300 轮实验，也不是服务器性能测试。短训练 AP 为 0 不用来判断结构优劣。
本机没有验证服务器 CUDA 训练、DCN GPU 延迟或最终收敛效果。

## 崩溃归因记录

初次顺序短训练在第二组 DCN 模型的 FLOPs 统计阶段发生 Windows access violation，退出码 -1073741819。
最初怀疑 CPU 线程数；固定为 2 后仍复现，该假设被否定。
定位到 get_flops 的 torch.empty 在确定性模式下产生 NaN；更换为 torch.zeros 后，
同配置 8 线程模型构建和原始顺序短训练均通过。正式代码没有关闭确定性或更改线程上限。

新增的两个 FLOPs 回归测试第一次使用裸 Conv2d 作测试容器，因其 stride 为 tuple 而走了全尺寸回退，
与测试预期不符；改为 Sequential 容器后覆盖了预期的两个路径。未放宽有限输入断言。

## 可追溯文件

完整记录在 E:/mastercode/_work/citrus_ev7_review_20260912/：

- originals/：修改前模块、生成器、suite 和八份 YAML。
- audit.log、audit_cpu2.log：未修复的失败尝试，保留原始证据。
- finite_profile_probe.log：修复后确定性 8 线程构建成功。
- audit_finite_profile.log：两组真实短训练与独立验证成功。
- verification_finite_profile/audit.json：八组参数/THOP/初始化匹配及短训练结果。
- runner_dry_run.log：生产批量入口八组检查。
- pytest_final2.log：最终 92 项回归结果。

训练输入栅格与本系列验证栅格有意分离，不能用默认官方 trainer 替代 EV6TrainingTrainer 后
仍声称复现本版统一验证协议。历史 input/4 的结果需同口径重新评估后再横向比较。
