# 其他基线模型：统一 Python 入口，保留官方结构与损失

本次按确认的范围实施：**统一调用，不改成 YOLO 头，不重写为 YOLO 损失。**
这不是把四个网络计算图完整移植成 Ultralytics 模型，也不是去掉 MMDetection/RF-DETR 依赖。
现有官方训练后端、模型结构、损失、匹配和原生后处理继续保留；已有柑橘训练配方不变。
不会覆盖已完成的旧实验，不修改 I/J 系列的 EV6 验证器。

## 1. 一次运行其他四种模型

文件：`RUN_BASELINE_OTHER_MODELS.py`，位于主代码目录，与原来的 `RUN_CITRUS_BASELINES_AMP.py` 同级。

默认前台顺序运行以下四种模型，各有 AMP0/AMP1，共 8 个任务；不重跑 YOLO。

| 模型标识 | 训练后端 | 现有训练尺寸 / batch | AMP1 |
|---|---|---|---|
| `rtmdet_ins_tiny` | MMDetection 3.3.0 | 640 / 8 | FP16 |
| `mask_rcnn_r50` | MMDetection 3.3.0 | 640 / 2 | FP16 |
| `solov2_light_r18` | MMDetection 3.3.0 | 640 / 2 | FP16 |
| `rfdetr_seg_nano` | RF-DETR 1.4.0.post0 | 624 / 2，梯度累积8 | BF16 |

1. 上传更新后的 `code/ultralytics-main-new`，不能只上传新入口而遗漏 `baseline_amp/`。
2. 在入口的 `SETTINGS` 中设置 `DATA`、`PROJECT`、`DEVICE`、`PYTHONS`。DATA 用你真实的清洗数据路径；不要求改目录名或手动确认指纹。
3. 两个依赖环境继续使用原来的 `citrus_mmdet` 与 `citrus_baseline`，没有新依赖；未安装者使用现有 `INSTALL_CITRUS_BASELINE_ENVS.py`。
4. **先把 EPOCHS 改成 1–3，PROJECT 指向全新的 smoke 目录**，跑通后再改为 300 和另一个新目录。
5. 在 VS Code 选择 Python，然后点右上角“运行 Python 文件”；或者在服务器运行：

```bash
cd /data/sxq/code/ultralytics-main-new
python RUN_BASELINE_OTHER_MODELS.py
```

查看计划，不创建目录、不加载模型、不下载、不训练：

```bash
python RUN_BASELINE_OTHER_MODELS.py --dry-run
```

不编辑脚本也能做3轮试运行，正式训练时另换一个新目录：

```bash
python RUN_BASELINE_OTHER_MODELS.py --epochs 3 --project /data/sxq/results/BASELINES/OTHER_SMOKE_3EP_NEW
```

只跑 RTMDet 一对 AMP 对照：

```bash
python RUN_BASELINE_OTHER_MODELS.py --models rtmdet_ins_tiny --project /data/sxq/results/BASELINES/RTM_UNIFIED_NEW
```

只跑 MMDetection 三种模型，或只做环境检查：

```bash
python RUN_BASELINE_OTHER_MODELS.py --suite mmdet --dry-run
python RUN_BASELINE_OTHER_MODELS.py --preflight-only
```

`MODELS=None` 表示全部 SUITE 成员，也可以改成模型标识列表。保留 suite 原排序及成对 AMP 顺序。
在训练终端 Ctrl+C 会结束当前任务并停止后续队列；没有 nohup、占卡保护或自动降 batch。
PYTHONS 内的两个解释器分别启动隔离子进程；同一时刻只运行一个训练/评估任务。

## 2. 单模型也有统一 train / val / profile 调用

自定义 Python 文件也可以直接运行，不读取 VS Code/debugger 的命令行参数：

```python
from baseline_amp import OfficialBaseline

if __name__ == "__main__":
    model = OfficialBaseline("rtmdet_ins_tiny")
    model.train(
        data="/data/sxq/datasets/你的清洗数据/data.yaml",
        project="/data/sxq/results/BASELINES/RTM_SMOKE_NEW",
        epochs=3,
        device=1,
        workers=4,
        amp_modes=(1, 0),
    )
```

如果环境路径不同，传入 `pythons={"mmdet": "/实际环境/bin/python", "modern": "/另一环境/bin/python"}`。
每个模型默认 batch 取原配置；指定 `batch` 时同时作用于 AMP0/AMP1。不接收 `cache=True`、YOLO box/cls loss 系数等无对应含义的参数。

对**已有模型运行目录**重评或测速，不是对整个批量 PROJECT：

```python
from baseline_amp import OfficialBaseline

if __name__ == "__main__":
    model = OfficialBaseline("rtmdet_ins_tiny")
    run = "/data/sxq/results/BASELINES/你的项目/rtmdet_ins_tiny_amp0_seed42_300ep"
    model.profile(run, device=1, images=5)  # 最大的5张图，同预测策略，不计算子集AP。
    # model.val(run, device=1)              # 整个保存的验证集，写新runtime_evaluate目录。
```

以上方法使用原 `RUN_BASELINE_RECOVERY.py` 同一执行路径，读取 job.json 和保存的权重记录。
不会改变该实验的数据/验证协议，不冒用 best 来源，不覆盖原指标；旧 native 协议不自动升级成 commonval。
原入口及恢复脚本继续可用。权重仍是原模型格式，**不能交给 `YOLO("native_weights.pth")` 加载**。

## 3. 为什么不承诺换入口就像 YOLO 一样快

原入口与新入口共享已修复的验证实现：有界输入预取、GT/IoU缓存、分块原图掩膜解码、显式推理作用域内的插值优化。
本次新增的是可复用 Python API、模型选择及独立非 YOLO 批量入口，没有再次改变验证算法。

所有新任务继续：每轮 quick640 监控，首轮/每10轮/末轮原图 COCO；仅原图 Mask AP50–95 选模；最终 FP32 同代码重评。
它不同于 I/J 的 EV6 网格指标，不能直接混用数值。不会为提高速度把原图掩膜换成160/320网格掩膜，或减少候选上限。

换 Python 外壳不会消除原图大掩膜生成、传输与编码工作量。因此**本轮未测得任何 GPU 提速倍数**；不能说已经接近 YOLO。
先对已有 RTMDet 权重运行 `profile()`，检查输出里的 predict、encode、evaluation、input_wait 分项，再判断下一处真实瓶颈。
详见 [同口径验证加速说明](BASELINE_FAST_VALIDATION_20261007.md) 与 [重新审查说明](BASELINE_RECHECK_20261008.md)。

加固后本地回归 **459 passed / 22 skipped**；新旧入口 dry-run、模型筛选及 Ruff 静态检查通过。
记录见 `baseline_amp/VERIFICATION_BATCH_HARDENING_20261008.json`；初版API验证另见 `VERIFICATION_UNIFIED_ENTRY_20261008.json`。
这些是 CPU 接口/回归测试，不代替服务器上四种原生后端的 CUDA smoke run。

## 4. 批量控制器加固（2026-10-08）

新入口默认输出目录为 `BASELINES_OTHER_SAFE_20261008`，不会复用初版 `BASELINES_OTHER_UNIFIED_20261008` 的结果。

- 启动前校验 epochs/workers/seeds/batch；不把浮点、字符串或布尔值偷偷转换为整数。seed必须在32位非负范围内。
- `START_FROM` / `SKIP_RUNS` 在打印计划时即校验；拼写错误不会先启动GPU检查、转换数据再报错。
- 正式启动前检查整条请求队列的旧运行目录、job记录、权重文件与完成来源信息。后面的坏目录会提前发现，不会等前面的模型跑几天才报错。
- worker退出码为0仍不代表完成：训练必须留下可用的trained记录/权重，验证必须留下指标及完成记录，检查通过才计入完成。
- 已有训练权重但验证中断的任务只补验证；半途训练仍需明确恢复或跳过，不自动覆盖、不假装完成。
- 同一PROJECT只允许一个本控制器写入，防止重复点三角形破坏共享数据/结果。这是**结果目录写入锁，不是GPU锁**；不同PROJECT不受它限制。
- `batch_protocol.json`、`job.json`、`trained.json`、`complete.json`、状态及汇总使用同目录临时文件+原子替换，避免中断留下半截JSON/CSV。AMP配对校验失败时不覆盖原汇总。
- 子进程输出有界读取；Ctrl+C、SIGTERM、非零退出、日志/终端写入异常都停止本次拥有的进程树并中止队列，不扫描或杀掉其他训练。

每个PROJECT新增 `batch_status.json`，记录当前job/stage、已完成、已跳过和错误。状态为 `failed` / `interrupted` 时查看其中的job、log和error。
`preflight_passed` 仅表示检查通过，不表示训练成功；`completed` 仅覆盖本次选定队列，user_skipped不算成功。
非正常断电或Linux `kill -9` 无法保证执行Python清理，状态也可能停留在running；不能据此判定进程仍活着或实验已完成。

同一个PROJECT的保护依赖文件系统支持本地文件锁/原子替换，建议使用服务器本地SSD。
新控制器的源码记录发生变化，旧PROJECT可能提示实现不同；不要删掉旧协议标记绕过，正式复跑使用新目录，已有权重可用原恢复/重评工具单独处理。
