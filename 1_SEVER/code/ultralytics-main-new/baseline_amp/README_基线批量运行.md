# 柑橘实例分割基线：7 个模型 × 2 种 AMP

入口：主代码目录的 `RUN_CITRUS_BASELINES_AMP.py`。默认 300 epoch、seed=42，共 14 次训练，前台串行执行。
不需要 nohup；在 VSCode 选择 Python 后，点击“运行 Python 文件”即可。它逐个调用两个环境里的 Python，
终端始终显示训练，Ctrl+C 停止当前任务并终止后续队列。它不会占卡检查、抢占锁、自动选卡，也不会停止别人的进程。

## 1. 为什么选这 7 个

| 模型 | 作用 | 实现来源 | 本套输入 / batch | AMP=1 的实际类型 |
|---|---|---|---|---|
| YOLO11n-seg | 主消融基线、重新验证 AMP 影响 | 官方 Ultralytics 8.4.60 | 640 / 16 | FP16 混合精度 |
| YOLOv8n-seg | 果园文献中常见的 YOLO 分割路线 | 同上 | 640 / 16 | FP16 混合精度 |
| YOLO26n-seg | 更新的实时模型对照；不是旧论文使用的模型 | 同上 | 640 / 16 | FP16 混合精度 |
| RTMDet-Ins-tiny | 非 YOLO 的轻量一阶段实例分割 | MMDetection 3.3.0 | 640 / 8 | FP16 混合精度 |
| Mask R-CNN R50-FPN | 经典两阶段强对照 | MMDetection 3.3.0 | 640 / 2 | FP16 混合精度 |
| SOLOv2-Light R18-FPN | 无框、位置式实例分割；补充不同技术路线 | MMDetection 3.3.0 | 640 / 2 | FP16 混合精度 |
| RF-DETR Seg Nano | Transformer 分割对照，真实 Nano，不是 Preview | RF-DETR 1.4.0 | **624 / 2，累积 8 次** | **BF16 混合精度** |

所有 AMP=0 使用 FP32，并关闭 TF32；每个模型的 AMP=0/1 使用相同初始化、seed、batch、优化器、数据、输入和增强。
相邻 AMP 对交替先后顺序，减少固定先后顺序与机器负载的混淆。单 seed 只能初筛，不据此声称统计显著。

选择依据不是“照搬一篇文章的所有模型”：绿色果实研究使用过 Mask R-CNN、SOLOv2、YOLACT、Cascade R-CNN；
果园模型比较研究使用过 YOLOv8 和 Mask R-CNN。本套选择其中代表性技术路线，再补充当前轻量/Transformer 对照。
轻量型号与统一柑橘训练协议是本项目的选择，不冒充原论文逐项复现。

- [Accurate segmentation of green fruit based on optimized mask RCNN application in complex orchard](https://pmc.ncbi.nlm.nih.gov/articles/PMC9399748/)：Algorithm comparison / Table 4 可核对比较方法。
- [Comparing YOLOv8 and Mask RCNN for object segmentation in complex orchard environments](https://arxiv.org/abs/2312.07935)：幼果与绿叶背景的相关场景。
- [RTMDet 论文](https://arxiv.org/abs/2212.07784)、[官方配置及模型库](https://github.com/open-mmlab/mmdetection/tree/v3.3.0/configs/rtmdet)。
- [SOLOv2 论文](https://arxiv.org/abs/2003.10152)、[官方 Light R18 模型库](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/solov2/metafile.yml)。
- [RF-DETR 1.4.0 官方型号配置](https://github.com/roboflow/rf-detr/blob/1.4.0/rfdetr/config.py)。

不把纯检测器或纯语义分割网络的指标冒充实例 Mask AP。本次不改动你的创新模型，不增加自定义注意力/损失到基线。

## 2. 最简单的安装：在服务器新建两个环境

上传整个 `code` 后，在服务器主代码目录执行（现有 base 环境的 Python 即可启动安装器）：

```bash
cd /data/sxq/code/ultralytics-main-new
python INSTALL_CITRUS_BASELINE_ENVS.py
```

安装器为 **Linux x86_64 + NVIDIA** 设计，使用 Conda Python 3.10 和预编译 wheel。
它不会修改你的 `sxq` 环境，不安装 mamba，不要求编译 CUDA 扩展。首次下载模型权重还需要联网。

| 环境 | 固定核心版本 | 执行模型 |
|---|---|---|
| `~/.conda/envs/citrus_baseline` | PyTorch 2.5.1 / torchvision 0.20.1 / CUDA 11.8 wheel / Ultralytics 8.4.60 / RF-DETR 1.4.0 | YOLO ×3、RF-DETR |
| `~/.conda/envs/citrus_mmdet` | PyTorch 2.1.0 / torchvision 0.16.0 / CUDA 11.8 wheel / MMCV 2.1.0 / MMEngine 0.10.7 / MMDetection 3.3.0 | RTMDet、Mask R-CNN、SOLOv2 |

两套环境用于隔离 MMCV 的编译版本与较新的 Transformer 依赖，避免升级旧环境后历史实验不能运行。
安装器打印实际 Python 路径；本套运行入口的 Linux 默认路径正好与它一致。
需要已有、兼容 CUDA 11.8 wheel 的 NVIDIA 驱动；用 `nvidia-smi` 检查。RF-DETR 原生 AMP 还要求 GPU 支持 BF16，
预检会检查。不支持时停止，不暗中替换为另一种 AMP。新架构 GPU 的支持需另行验证，不能仅凭显存大小判断。

只想先安装 YOLO/RF 环境：

```bash
python INSTALL_CITRUS_BASELINE_ENVS.py --env modern
```

之后运行入口的 `SUITE` 先设成 `"yolo"`。要跑全套，再执行安装器 `--env mmdet` 并使用新的 PROJECT。

先看安装命令、不执行：`python INSTALL_CITRUS_BASELINE_ENVS.py --dry-run`。
如果本安装器中途因网络失败，修复网络后可用 `--env modern --repair` 或 `--env mmdet --repair` 继续安装对应新环境。
不要把这些环境替换成正在训练的旧环境。不要在新基线环境里执行本项目的 `pip install -e .`，
否则会把官方 Ultralytics 替换成你的改进分支；预检会拒绝这种混用。

MMCV 安装使用 `--only-binary=mmcv`：找不到匹配 wheel 时直接报错，不偷偷开始漫长编译。
已核对官方索引包含 Python 3.10 / Linux / Torch 2.1 / CUDA 11.8 / MMCV 2.1.0 wheel。
参见 [MMCV 安装说明](https://mmcv.readthedocs.io/en/latest/get_started/installation.html)。

## 3. 编辑入口，再点三角形

打开 `RUN_CITRUS_BASELINES_AMP.py`，主要修改：

```python
"DATA": "/你的服务器/清洗数据集/data.yaml",
"PROJECT": "/data/sxq/results/BASELINES/BASELINES_AMP_SMOKE_3EP",
"DEVICE": 1,
"SUITE": "all",
"EPOCHS": 3,
"SEEDS": [42],
```

DATA 必须指向你真实使用的清洗数据集，不要求你改目录名，也没有人工指纹确认步骤。
入口默认 grouped_dedup 的路径只是示例；服务器目录名不同就修改 DATA。
数据 YAML 内的 `path/train/val/test` 也要能在服务器解析，不能仍写 Windows 盘符。
历史 `orange_yolo` 目录名不能证明里面是旧还是新划分；请以实际图像、标签和划分为准。

先打印 14 个任务，不加载模型、不下载、不训练：

```bash
python RUN_CITRUS_BASELINES_AMP.py --dry-run
```

然后做环境预检：

```bash
python RUN_CITRUS_BASELINES_AMP.py --preflight-only
```

接着点击 VSCode 的“运行 Python 文件”，或执行：

```bash
python RUN_CITRUS_BASELINES_AMP.py
```

**先让 14 个任务各跑 1–3 epoch。** 确认无 NaN、无 OOM、类别正确、生成共同评估结果后，
把 EPOCHS 改成 300，PROJECT 改成一个新的 `BASELINES_AMP_300EP` 目录，再点击三角形正式跑。
这样不必先用 300 轮才发现后半段跨框架环境有问题。

所有子进程只看到 DEVICE 指定的物理卡，其内部显示 `cuda:0` 是正常的重编号，并不意味着偷偷跑物理 0 卡。
YOLO 入口保留已绑定的 CUDA 可见设备，不让默认 `device=0` 改写物理卡选择。
前台运行本身不使训练更快，也不保证其他用户不能使用同一张卡。

## 4. 固定的训练协议与比较边界

共同项：同一个原始划分、COCO 预训练初始化、原图输入而非切片、相同训练轮数（默认 300）、seed=42、
不使用测试集选模、每轮验证、AMP 成对运行。模型使用自己的官方损失，只有分类数改为数据集类别数。
全套不是一次“只改架构”的严格同超参消融；不同家族采用不同优化器、增强和缩放方法，记录在每次运行的有效配置中。

| 家族 | 优化与调度 | 输入增强/其他 |
|---|---|---|
| YOLO | AdamW, lr0=.001, lrf=.01, wd=.0005, nbs=64, warmup=3；线性衰减 | 640 letterbox；cache=True；mosaic=1、最后10轮关闭；copy_paste=0；mask_ratio=4；dropout=0；其余显式参数见 registry.py |
| RTMDet-Ins | 官方 AdamW/参数组；LR 按实际 batch / 官方 base_batch_size 缩放；cosine 到初始 LR 的5% | 固定640等比例缩放+水平翻转；保留官方损失与 EMA；移除 pipeline switch、裁剪和小框过滤；单卡 SyncBN→BN |
| Mask R-CNN / SOLOv2 | 官方 SGD/参数组；LR 按 batch / 官方 base_batch_size 缩放；在2/3、8/9训练进度各衰减0.1 | 固定640等比例缩放+水平翻转；保留官方损失；不人为过滤小果实或空背景图 |
| RF-DETR Seg Nano | AdamW, lr=1e-4, encoder lr=1.5e-4, wd=1e-4；80%轮数处降 LR；EMA开启；early_stopping=False | 624方形输入，multiscale关闭；保留官方翻转/随机裁剪；batch2×grad_accum8；预训练位置编码保持26，官方前向动态插值 |

这些适配都是本套明确的柑橘预算协议，不宣称等于原文完整 COCO 训练配方。
RF-DETR 原生 Nano 是312，本套选624接近其他模型的640，**不是完全同分辨率**；不能把624的耗时宣称成官方312测速。
MMDetection、RF-DETR 没有与 Ultralytics `cache=True` 等价的统一 API，不能塞一个无效参数假装开启缓存。
派生图像优先硬链接，不重复解码保存；源数据保持不变。需要快速本地磁盘，避免网络盘 IO 成为瓶颈。

**为什么不全部强制 batch=16、同一优化器？** Mask R-CNN/SOLOv2 显存需求不同，全部强行统一可能无法启动；
不同架构原生优化方案也不同。最关键的 AMP 因果比较在“同一模型、同一 seed”内部，除 AMP 外保持完全相同的训练配方。
如果 OOM，请先停止，在入口 BATCHES 中修改该模型，让两种 AMP 一起改变，并换 PROJECT 重新跑这一对；
不允许只给 AMP=0 降 batch、AMP=1 保持大 batch。

YOLO/各框架训练时的原生验证精度与最佳检查点选择规则仍沿用框架实现；本套另外对最终选择的权重进行统一 FP32 验证。
因此这是“原生 AMP 训练工作流”的成对比较，不等于证明每个梯度误差或每个 AMP 算子单独造成了性能变化。

## 5. 数据适配与评估

`PROJECT/_prepared/` 保存派生的 YOLO、COCO 和 RF-DETR 数据视图以及 manifest，**不清洗、不重新划分、不删除小目标**。
原始标签缺失/非法时停止并指出文件，不自动修复；合法空标签保留为背景图。
RF-DETR 1.4 任务头的类别编号需要0..N-1；它的派生视图单独重编号，评估导出时再映射回共同 COCO 的1..N。
不要混用这些不同框架的 JSON。边界坐标按原始图像宽高缩放，不把 `x=1` 擅自裁到 `width-1`。
如果某实例栅格化后为0面积，程序明确报错，不静默删除。

RF-DETR 1.4 即使 `run_test=False` 仍会构建 test loader，所以 data.yaml 必须包含**真实独立的 test**。
本套不对它运行预测/评分，不用它选超参，也不拿 val 冒充 test。正式论文最后固定模型后另做一次 test 评估。

训练结束后各模型都在原始验证图上按同一 COCO mask 标准重评：

- Mask AP50–95、AP50、APS、AR100；`AP` 与 `AR` 不是同一个指标。
- 固定 `score=0.25, mask IoU=0.5` 的 Precision/Recall/F1。
- 预测保留阈值0.001；COCO AP使用 maxDets=100；各网络原生 NMS/查询数保留，所以不是统一检测器后处理。
- 原图掩膜坐标必须精确匹配，不用错误尺寸的 mask 硬算指标。
- 大中小目标分桶按原图掩膜面积，而非640缩放后面积。
- 全部统一评估使用 FP32；官方训练日志另行保留，不能把其中“最佳F1处的 Recall”与本套固定阈值 Recall混为一谈。

若你的最终方法包含切片、专门增强和不同损失，和本套整图基线的差值是**整个方法的收益**，
不是纯网络结构收益。论文还需增加 YOLO11n 在相同切片/增强配方下的对照，不能把输入收益全部归到主干上。

## 6. 日志、停止、中断和结果

目录示例：

```text
BASELINES_AMP_300EP/
  batch_protocol.json
  _prepared/                 数据格式适配及划分记录
  _pretrained/               成对实验共用的官方初始权重
  jobs/                      每次启动的完整配置
  logs/                      终端同内容的持久日志
  yolo11n_seg_amp0_seed42_300ep/
    train/                   原生训练产物和 best/last 权重
    amp_actual.json          实际 AMP 类型，非只有配置声明
    initialization.json      初始权重来源及校验值
    dataset.json             原始数据路径、图像/实例数
    val_predictions.coco.json
    val_common.json          统一评估指标
    complete.json            完整训练且评估成功才生成
  yolo11n_seg_amp1_seed42_300ep/
  ...
  baseline_summary.csv
  amp_paired_deltas.json
```

在训练的 VSCode 终端直接查看；另开终端也可以：

```bash
tail -f /data/sxq/results/BASELINES/BASELINES_AMP_300EP/logs/yolo11n_seg_amp0_seed42_300ep.log
```

在训练终端按 Ctrl+C 关闭当前任务和队列；在 `tail` 终端按 Ctrl+C 只退出看日志。
训练完成的任务再次启动自动跳过；训练已完成但共同评估中断的任务，会只重做评估。
训练半途退出的任务不会自动假装完成，也不会覆盖。可以将其完整名称加入 `SKIP_RUNS`，先跑后续；
或者使用新的 PROJECT 单独重新运行相应 suite。此版本**不承诺跨三个训练框架的一键断点续训**。
`START_FROM` 接受 `--dry-run` 打印的完整任务名，明确跳过它之前的任务。

相同 PROJECT 不允许改变 batch、seed、epoch、环境路径或实现代码；防止中途换协议后仍混在同一个结果表里。
成对汇总检查初始权重及数据记录一致，差值统一定义为 `AMP1 - AMP0`，单位是百分点。

只重新生成表格：

```bash
python RUN_CITRUS_BASELINES_AMP.py --summarize-only
```

## 7. 本地验证边界

已执行：18项 CPU 合约/配置测试（包括三个官方 MMDetection 配置、前台子进程日志及失败退出）、语法检查、关键静态检查、14任务队列 dry-run。
核对了 RF-DETR 1.4 源码的 BF16、类别编号、best-mask 检查点选择、test loader 和位置编码插值。
**尚未在你的服务器 GPU 上执行这14次训练，也未声称完整 Conda 安装已验证成功。**
服务器上应按第3节先预检、再1–3轮 smoke，再300轮；耗时、GFLOPs和部署延迟仍需真实测量，不能由参数量推断。

实验设计技能在本套中的落实：模型×seed为配对单元、AMP为实验因素；固定配方，显式保存偏差，单seed筛选后再三seed。
不会为了让某个模型胜出而单独搜索它的 AMP 或学习率，然后宣称结构创新显著。
