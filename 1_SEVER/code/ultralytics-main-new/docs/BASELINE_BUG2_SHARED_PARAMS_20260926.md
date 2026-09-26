# 2026-09-26：RTMDet 优化器共享参数报错修复

附件 bug2 中失败任务为 `rtmdet_ins_tiny_amp0_seed42_300ep`。它在 `Runner.train()` 创建 AdamW 时停止，尚未完成第一轮，异常为 `ValueError: some parameters appear in more than one parameter group`。

## 根因及修改

RTMDet 的多个特征尺度共享卷积权重。MMEngine 对模块逐层建立参数组时，同一 tensor 可以沿多个模块路径被发现。本地 worker 为统一 AdamW 超参重建了 `optim_wrapper`，遗漏官方原有的 `paramwise_cfg.bypass_duplicate=True`，于是同一个 tensor 进入多个参数组，PyTorch 拒绝构建优化器。

官方依据：[MMDetection v3.3.0 RTMDet 配置](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/rtmdet_l_8xb32-300e_coco.py) 同时使用 `share_conv=True` 与 `bypass_duplicate=True`；[MMEngine v0.10.7 构造器](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/optim/optimizer/default_constructor.py) 实现重复参数跳过逻辑。

现已在 AMP0/AMP1 的同一优化器配置函数中恢复去重，保留 AdamW lr=0.001、betas=(0.937,0.999)、weight_decay=0.0005 以及 bias/norm 不衰减规则。训练开始时写入 `optimizer_parameters.json`，确认无重复、无漏掉的可训练参数；预检会在 CPU 上构建三个 MMDetection 模型及优化器，随后继续现有 CUDA 算子检查。无需因本错误重装 Conda/CUDA。

## 本次运行队列

`non_yolo` 包含 RTMDet-Ins-tiny、Mask R-CNN R50-FPN、SOLOv2-Light R18-FPN、RF-DETR Seg Nano，四组各跑 AMP0 和 AMP1，共8次。YOLO 的三组完全不进入此队列。原 `all` 套件保留，但默认入口现已切换为 `non_yolo`。

上传本地 `baseline_amp/` 整个文件夹到服务器同名目录。服务器现有 `RUN_CITRUS_BASELINES_AMP.py` 可保留其 DATA、DEVICE、PYTHONS，以下命令通过新 batch 的参数覆盖 SUITE/PROJECT：

```bash
cd /data/sxq/code/ultralytics-main-new
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py --suite non_yolo --project /data/sxq/results/BASELINES/BASELINES_NONYOLO_FIXED_PAIR_20260926 --dry-run
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py --suite non_yolo --project /data/sxq/results/BASELINES/BASELINES_NONYOLO_FIXED_PAIR_20260926 --preflight-only
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py --suite non_yolo --project /data/sxq/results/BASELINES/BASELINES_NONYOLO_FIXED_PAIR_20260926
```

前两条分别应显示8项非YOLO任务，以及 `OPTIMIZER PREFLIGHT OK` / `PREFLIGHT OK`。若路径已经用于另一份配置，请换一个未用的新目录。此次修改了实现 hash 和队列，所以不要将新命令的 PROJECT 指向旧 `.../baseline`。旧 YOLO 结果和失败日志无需删除。

若使用 VS Code 三角形：同步新版根入口后，把服务器自己的 DATA/PYTHONS/DEVICE 填回设置区，确认 `SUITE="non_yolo"`、`AMP_MODES=[1,0]`，再运行。仍是前台串行；Ctrl+C 停止当前任务及后续队列。

## 验证范围

在独立临时依赖目录加载 MMEngine 0.10.7：用三尺度共享卷积复现原异常，修复后完成实际优化器更新；核对每个权重只登记一次，所有可训练参数均在优化器中，norm/bias 衰减规则保留。另检查非YOLO队列为4组×2 AMP，命令行覆盖不改写服务器设置。

本机没有服务器完整的 Linux CUDA/MMCV 环境，因此没有声称已在服务器完成三个模型的全量训练；新增预检会在服务器使用真实模型检查优化器。
