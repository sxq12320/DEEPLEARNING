# J 系列恢复 I 系列原生验证方式

日期：2026-10-02。用户要求 J 的训练验证继续沿用 I 系列，不接入跨框架 baseline 验证器。

## 当前行为

- `RUN_CITRUS_J.py` 与底层 `run()` 均默认 `COMMON_VALIDATION=False`；兼容参数仍在，但 True 会在创建输出、导入模型或训练前明确报错，不再切换到 baseline。
- 来源均衡训练直接使用 **同一个** `citrus_e_v6_training.EV6TrainingTrainer` / `EV6Validator`，与 I_V2/I_V3/I_V4 的历史原生验证一致。
- 非切片调试模式仅更换训练数据集构建方式，验证仍使用 EV6Validator；不混用标准 prototype/4 与 EV6 input/2 指标栅格。
- 每轮原图验证，GT 与预测 mask logit 使用 input/2 栅格；conf=0.001、NMS IoU=0.7、max_det=300、FP32。不生成原图 polygon COCO 标注，也不走 quick640/COCO 双阶段评估。
- 原生 `results.csv` 保留 Box/Mask P、R、mAP50、mAP50-95。P/R 保持原生最大平均 F1 工作点统计，不改成 baseline 的固定 score=0.25 P/R。
- `plots=True`，恢复原有 Box/Mask PR、P/R/F1、混淆矩阵和结果图；未修改 AP 插值、PR 曲线末尾或混淆矩阵算法。
- `weights/best.pt` 保持原生框+掩膜 fitness 选优。按 I 系列保存方式另存 `best_mask.pt` 和 `best_mask50.pt`，并记录各自轮次；它们不覆盖 best.pt。
- 额外 `DIAGNOSE_PR=False`，默认没有额外 observed-PR 验证。手动开启时仅附加原生 input/2 匹配点 CSV，不替代曲线或选优，并沿用 max_det=300。
- J 的源码摘要不再包含 baseline 实现；本次没有修改 baseline 运行和评估代码，也没有更改 J 模型/YAML、数据路径、训练输入、AMP、cache、batch、优化器、学习率或模型消融因子。

## 运行与查看

仍使用 `RUN_CITRUS_J.py`，修改 DATA、DEVICE、SUITE、EPOCHS 等原有设置后，在 VSCode 选原训练解释器并点击 Run Python File，或运行：

```bash
python RUN_CITRUS_J.py
```

新默认输出目录有 `_NATIVEVAL` 后缀，避免混入旧验证实验。当前协议：`citrus_j_rgb_scratch_sourcebalanced_native_val_v2_20261002`。

查看每组直接位于运行目录中的：

- `results.csv`：每轮原生验证结果。
- `MaskPR_curve.png` / `BoxPR_curve.png`、`confusion_matrix.png` 等：原有图形输出。
- `weights/best.pt`：原生最佳权重；`best_mask.pt` / `best_mask50.pt` 为独立掩膜选优快照。
- `args.yaml` / `j_protocol.json` / `runtime.json`：有效配置和运行环境。

不再查找 `validation_common/epochs.csv` 或 `best_unified.pt`。已完成结果与半途目录不删除、不覆盖。正在服务器运行的进程不会因本地文件更新自动切换验证器；需要下一次启动才使用更新代码，不能将旧训练/选优过程伪装为新协议。

**比较限制：** 恢复的是 I/J 内部训练验证口径，不能因此把 baseline 原图 COCO AP 与 I/J input/2 栅格 AP 直接相减声称涨点。正式跨框架比较需另行对冻结权重执行同一最终评价协议，这次没有把那套评估插入 J 训练。

## 已执行验证

本机隔离 CPU 栈：Windows、Python 3.9.13、PyTorch 2.8.0+cpu、NumPy 1.26.4。

1. **102 passed，5 warnings**：J 模型全量契约、J 验证路由、测速、EV6 共同栅格和 I_V4 原生模型相关回归。新增 9 项 J 路由/默认值/日志/选优测试；模拟 baseline 导入被拒绝时，默认 J 队列仍可执行与跳过已完成任务。
2. **实际短训练通过**：隔离小样本 4 张训练/2 张验证，J00 与 J16 各 1 epoch 前台串行，128 输入、batch=2、workers=0。标准每轮验证、最终 best.pt 重载验证、日志和图形输出完成；四个 checkpoint 文件均生成。额外 COCO/observed-PR 流程没有启动。
3. Ruff 对本次 3 个 Python 文件检查通过；Python 3.8 语法解析通过；本次文件范围 diff 空白检查通过。

命令：

```powershell
$env:PYTHONPATH='E:\mastercode\_work\baseline_common_validation_20260930\cpu_deps;E:\mastercode\1_SEVER\code\ultralytics-main-new'
& E:/AppInstallion/0_4_annaconda/python.exe -m pytest tests/test_citrus_j_validation.py tests/test_citrus_j.py tests/test_citrus_j_benchmark.py tests/test_citrus_e_v6.py tests/test_citrus_i_v4_native.py -q --junitxml=E:/mastercode/_work/20261002_j_nativeval/regression.xml
& E:/AppInstallion/0_4_annaconda/python.exe scripts/smoke_citrus_j.py --data E:/mastercode/_work/citrus_native_smoke_dataset/data.yaml --project E:/mastercode/_work/20261002_j_nativeval/smoke --only J00_yolo11n_control,J16_joint_mask_loss
```

原始证据：`E:/mastercode/_work/20261002_j_nativeval/regression.xml`、`smoke.log`、`smoke/`。5 个警告来自 THOP distutils 弃用和本机旧 SciPy 的 NumPy 范围提醒，不是测试失败。

未验证：服务器 Python 3.8 / Torch1.13.1 CUDA 运行、整轮 GPU 速度、正式训练精度收益。短训练 AP=0 只用于验证工程链路，不用于模型排名；恢复原生验证不等于解决真实小目标漏检或保证 PR 曲线尾端变高。
