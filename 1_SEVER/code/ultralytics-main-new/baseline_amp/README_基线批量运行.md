# 柑橘实例分割 baseline：当前唯一批量入口

更新：2026-10-11。主代码只保留 **`RUN_PAPER_BASELINES.py`** 作为 baseline 批量训练入口。
旧的 `RUN_CITRUS_BASELINES_AMP.py`、`RUN_BASELINE_OTHER_MODELS.py` 已从主代码移除，
清理前副本位于 `E:/mastercode/_work/20261011_baseline_entry_cleanup_backup/`。
已有结果、权重、数据及旧协议恢复能力均保留。

当前入口使用官方 COCO 预训练权重和各框架原生优化配方，**不是历史“78版”从零训练配方**。
不能把它与 scratch 的 I/J 实验直接作同初始化架构消融。旧评测结果仍按原协议记录，不自动升级或混表。
详细配方见 [Baseline 重审与 J4 说明](../docs/BASELINE_TRANSFER_AND_J4_20261009.md)。

## 1. 运行哪些模型

默认 `SUITE="all"`，7 个原版分割模型，各跑 AMP=1、AMP=0；单 seed 共 14 个前台顺序任务，每个训练 300 轮。

| 模型标识 | 后端 | 输入 / batch | AMP=1 |
|---|---|---|---|
| `yolo11n_seg` | 官方 Ultralytics | 640 / 16 | FP16 |
| `yolov8n_seg` | 官方 Ultralytics | 640 / 16 | FP16 |
| `yolo26n_seg` | 官方 Ultralytics | 640 / 16 | FP16 |
| `rtmdet_ins_tiny` | MMDetection | 640 / 8 | FP16 |
| `mask_rcnn_r50` | MMDetection | 640 / 2 | FP16 |
| `solov2_light_r18` | MMDetection | 640 / 2 | FP16 |
| `rfdetr_seg_nano` | RF-DETR | 624 / 2，梯度累积 8 | BF16 |

不替换原模型的结构、损失或匹配方法。AMP=0 为 FP32；同一模型成对任务保持初始化、seed、batch 和配方一致。
跨框架的 batch、增强、EMA、624/640 输入和 AMP 类型差异仍须披露，不称为完全相同超参数。

## 2. 环境安装

已安装并通过检查的两个 baseline 环境直接复用，不重复安装，也不修改你的 `sxq` 环境。
尚未安装时，在 Linux x86_64 / NVIDIA 服务器运行：

```bash
cd /data/sxq/code/ultralytics-main-new
python INSTALL_CITRUS_BASELINE_ENVS.py
```

安装器创建两个隔离环境，使用 Python 3.10 和预编译 CUDA 11.8 wheels，不需要 mamba。

| 环境 | 核心版本 | 执行模型 |
|---|---|---|
| `~/.conda/envs/citrus_baseline` | Torch 2.5.1 / torchvision 0.20.1 / Ultralytics 8.4.60 / RF-DETR 1.4.0.post0 | YOLO / RF-DETR |
| `~/.conda/envs/citrus_mmdet` | Torch 2.1.0 / torchvision 0.16.0 / MMCV 2.1.0 / MMEngine 0.10.7 / MMDetection 3.3.0 | RTMDet / Mask R-CNN / SOLOv2 |

默认解释器路径与安装器一致；实际环境在别处时修改 `PYTHONS`。
不要在这两个官方 baseline 环境执行改进分支的 `pip install -e .`。
需要兼容 NVIDIA 驱动及 RF-DETR AMP1 所需的 BF16 支持；首次运行需要联网下载官方初始权重。
下载失败不会悄悄回退成随机初始化。

安装命令预览：`python INSTALL_CITRUS_BASELINE_ENVS.py --dry-run`。
仅修复已有 baseline 环境时，明确选择 `--env modern --repair` 或 `--env mmdet --repair`；
先停止使用该环境的任务。旧版 RF/MMDet 故障修复脚本保留，它们不是第二套训练入口。

## 3. 修改设置，直接点 VSCode 三角形

打开主代码目录的 `RUN_PAPER_BASELINES.py`，修改 `SETTINGS`：

```python
"DATA": "/你的服务器/清洗数据集/data.yaml",
"PROJECT": "/data/sxq/results/BASELINES/PAPER_BASELINES_TRANSFER_300EP_20261011",
"DEVICE": 1,
"SUITE": "all",  # non_yolo 只跑后四种；mmdet 只跑三种 MMDetection 模型。
"MODELS": None,   # 首次短测可改为 ["rtmdet_ins_tiny"]。
"EPOCHS": 300,    # 最新默认直接300轮；可选短测另设新 PROJECT。
"SEEDS": [42],
"AMP_MODES": [1, 0],
"WORKERS": 4,
```

保留 `RECIPE_POLICY="official_transfer"` 和共同 `VALIDATION`，不要按模型单独更改评测条件。
DATA 用你实际的清洗数据路径，不要求重命名目录或手动确认指纹。
数据 YAML 的 `path/train/val/test` 也应能在服务器解析；不能保留 Windows 盘符。
RF-DETR 会构建 test loader，因此需要真实独立 test；它不用于本批训练选模。

查看队列，不下载、不加载模型、不创建结果目录：

```bash
python RUN_PAPER_BASELINES.py --dry-run
```

确认 DATA 和环境路径后，可先检查环境，再直接启动默认 300 轮队列：

```bash
python RUN_PAPER_BASELINES.py --preflight-only
python RUN_PAPER_BASELINES.py
```

也可以在 VSCode 选择 Python，点击右上角“运行 Python 文件”。
父程序依次调用两个隔离环境，不需要 nohup。子进程内 `cuda:0` 是所选物理卡的逻辑重编号。
没有 GPU 占用保护或自动降 batch；也不会关闭他人的进程。前台执行本身不保证训练更快。

默认结果目录为 `PAPER_BASELINES_TRANSFER_300EP_20261011`，与之前 50 轮结果分开。
若需要独立的 1–3 轮短测或 50 轮筛选，用 `--epochs` 并指定新 PROJECT，不复用正式目录。
要另开一组 300 轮实验，例如：

```bash
python RUN_PAPER_BASELINES.py --epochs 300 \
  --project /data/sxq/results/BASELINES/PAPER_TRANSFER_300EP_NEW
```

只跑非 YOLO 的四种模型仍使用同一个入口：

```bash
python RUN_PAPER_BASELINES.py --suite non_yolo --epochs 300 \
  --project /data/sxq/results/BASELINES/PAPER_NON_YOLO_300EP_NEW
```

正式主基线和最终方法各用三个 seed；本入口可设置 `SEEDS=[42,43,44]` 或使用 `--seeds 42 43 44`。
新增模型选择、初始化或训练预算时均使用新 PROJECT，不覆盖已完成实验。

## 4. 验证与公平性边界

每轮 quick640 监控；首轮、每 10 轮、末轮做原图 COCO 验证。
仅原图 Mask AP50–95 选择最佳权重，最终共同评估为 FP32，不用 test 选模。
记录 Mask AP50–95、AP50、APS、AR100，以及固定 score=0.25 / mask IoU=0.5 的 P/R。
COCO `maxDets=100`，各网络保留原生后处理；不把原图 mask 换成低分辨率网格来伪装提速。

原图掩膜分块解码、CPU 预取、GT/IoU 缓存和显式推理作用域内的插值优化保留。
仍需实际测量服务器验证耗时，不能仅因为统一 Python 入口就声称与 YOLO 一样快。
I/J 创新系列仍使用其 EV6 验证器，没有修改；论文对比需额外共同评估，不能直接混用两种指标数值。

历史修复原理可查 [验证速度说明](../docs/BASELINE_FAST_VALIDATION_20261007.md) 和
[2026-10-08 加固说明](../docs/BASELINE_UNIFIED_ENTRY_20261008.md)；这些是历史记录，其中旧入口命令已退役，
当前运行一律以本页和 `RUN_PAPER_BASELINES.py` 为准。

## 5. 日志、停止与恢复

训练日志前台显示并保存至 `PROJECT/logs/`；每个任务以 `transfer_` 开头。
结果汇总为 `baseline_summary.csv`，成对 AMP 差值为 `amp_paired_deltas.json`，
队列进度和错误记录在 `batch_status.json`。

训练终端 Ctrl+C 停止当前任务和整个队列；看日志的 tail 终端 Ctrl+C 只退出看日志。
相同项目/协议下，有效已完成任务自动跳过；已有完成训练记录但评估中断的任务只补评估。
半途训练不会自动覆盖，可明确 `SKIP_RUNS` / `--skip-runs`，或另开目录重新训练。
`START_FROM` / `--start-from` 是跳过前面的任务，不是断点续训。

`RUN_BASELINE_RECOVERY.py` 保留，用于已有权重测速、复评以及受支持的 MMDet 恢复。
底层旧配方兼容仍保留，防止历史权重无法恢复；不会把旧配方指标改成新配方指标。
源码或协议变化时不要删除旧记录绕过校验；新训练使用新 PROJECT。

只刷新当前项目汇总：

```bash
python RUN_PAPER_BASELINES.py --summarize-only
```

本次清理仅整合入口和运行说明；随后按用户要求将默认预算改为 300 轮并使用独立结果目录。
模型、优化器、AMP 两模式、验证实现和数据路径未改变。
入口测试和队列预览不代替服务器 CUDA 短测。
