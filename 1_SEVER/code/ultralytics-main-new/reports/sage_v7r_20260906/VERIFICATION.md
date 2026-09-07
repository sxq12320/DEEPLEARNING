# V7R 实现验收（2026-09-06）

环境：Windows、Python3.9.13、Torch2.8.0 CPU、AMD Ryzen5 5600H。无服务器GPU验证或正式精度结果。

## 已执行

- `pytest tests/test_citrus_sage_v7.py tests/test_citrus_sage_v7r.py tests/test_citrus_sage_v7_batch.py tests/test_citrus_foreground.py -q --disable-warnings --maxfail=1 --junitxml=reports/sage_v7r_20260906/tests.xml`
  ：66 passed，4 warnings，31.07秒；覆盖70–78公开YAML构建、正常/空GT损失和反向、矩形输入、导出输出形式、预训练加载、检查点重载、Conv-BN fuse、多类别、特征路由隔离、队列中断和真实批量回调。
- 构建阶段发现字符串 `class` 与当前 YAML 参数解释器冲突，已改用 `cls_only`，没有扩大修改全局解析规则。
- `scripts/smoke_sage_v7.py` 对75/76/77/78分别完成一轮真实训练/验证及检查点保存。
  指定既有4训练图/4验证图fixture；CPU、batch2、256、workers0、关闭mosaic/plots。
  四项均输出 `SMOKE PASSED`，日志见 [smoke.log](smoke.log)，配置见
  [SMOKE_NOT_FORMAL.json](smoke_new_four/SMOKE_NOT_FORMAL.json)。
  仅功能验收，不把这些小样本一轮指标解释为精度结果；没有修改正式训练协议或数据。
- 批量入口 `--suite refusion --dry-run` 六个模型均 BUILD OK。
- 新增实现/测试/测速脚本/启动器/suite 的 Ruff 检查通过。
- 对照上次交付SHA256：旧70–74 YAML、旧V7模块、旧RUN_SAGE_V7.py及固定超参协议均未改变。
  注册文件、suite、批量来源记录新增V7R入口；没有修改已完成的results文件。

## 同次交错测速

CPU、FP32、batch1、640；每阶段3次预热+20次计时，每轮随机模型顺序。
训练步包括零梯度、前向、标准损失、反向和简单单参数组AdamW更新。
不含数据加载、真实增强、NMS、验证和回调；不是正式优化器分组，也不是GPU FPS。
测速脚本 [benchmark_sage_v7r.py](../../scripts/benchmark_sage_v7r.py)，
原始数据 [cpu_interleaved640.json](cpu_interleaved640.json)。

| 模型 | 参数量 | GFLOPs640 | 前向ms | 训练步ms |
| --- | ---: | ---: | ---: | ---: |
| 70旧结构 | 2,323,380 | 10.097 | 96.97 | 433.53 |
| 72旧V7候选 | 1,797,333 | 8.713 | 93.67 | 446.76 |
| 75细节旁路 | 1,796,228 | 8.697 | 92.70 | 434.46 |
| 76共享上下文 | 1,799,845 | 8.744 | 100.31 | 453.34 |
| 77任务路由 | 1,799,845 | 8.744 | 100.91 | 453.01 |
| 78单尺度路由 | 1,799,845 | 8.746 | 99.67 | 449.01 |

77相对72参数只增加2,512，GFLOPs增加约0.031；本次训练步中位时间约增加1.4%，前向约增加7.7%。
相对70参数减少约22.5%、GFLOPs减少约13.4%，但CPU训练步反而约慢4.5%。
因此不能声称“低GFLOPs保证更快”；本地未出现数量级变慢，但仍须用户同GPU实测。
旧报告训练微测不含优化器更新，与本报告不同，不能跨报告直接算速度提升。
GFLOPs由当前get_flops估算，部分函数式池化/插值/逐元素操作可能未覆盖。

## 实验后才能回答

1. 直接细节旁路是否确实改善极小目标的框/掩膜身份，而非增加叶片误报。
2. 分类专用上下文是否优于共享上下文（77对76，同参数），多尺度是否优于单尺度（77对78，同参数）。
3. Mask AP50–95、固定Precision下极小Recall和同GPU耗时能否同时达到要求。
4. 当前主干仍沿用YOLO11，不把这一系列宣传为新的主干网络或已验证控制理论创新。

新增正式入口：[RUN_SAGE_V7R.py](../../RUN_SAGE_V7R.py)。默认新project，300轮，前台串行，AMP=False。
