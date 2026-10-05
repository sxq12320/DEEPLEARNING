# 基线验证速度修复与服务器操作（2026-09-30）

## 结论与证据

这次不是“RTMDet 天生比 YOLO 慢一点”。回传 `bug2` 的第 37–48 轮表明，
训练与保存平均 85.4 秒，验证平均 2,046.7 秒（34.1 分钟），约占整轮的 96%。
193 张验证图片通常约 10–11 秒/张。COCO 最终计算只有数秒，不能把 34 分钟归咎于 COCOeval。
这里的训练耗时按上一轮验证结束到本轮 checkpoint 日志计算，包含保存等开销。

| 轮数 | 训练与保存/秒 | 验证/秒 |
|---|---:|---:|
| 37 | 91 | 2107 |
| 38 | 95 | 2010 |
| 39 | 83 | 2005 |
| 40 | 80 | 2003 |
| 41 | 83 | 2005 |
| 42 | 77 | 2007 |
| 43 | 90 | 2005 |
| 44 | 80 | 2012 |
| 45 | 79 | 2008 |
| 46 | 93 | 2032 |
| 47 | 83 | 2057 |
| 48 | 91 | 2209 |

### 已确认的代码链与尚未实测的部分

1. 本套 MMDetection 配置设 `randomness.deterministic=True`。
   [MMEngine 0.10.7 源码](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/runner/utils.py)
   因此调用 `torch.use_deterministic_algorithms(True)`。
2. [PyTorch 2.1 的 interpolate 源码](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/nn/functional.py)
   遇到 CUDA、bilinear、确定性开启时，**即使没有梯度**也调用注释标明的 `slow decomp`。
   之前 OOM 的回传调用栈也经过这个 decomposition/_unsafe_index 分支。
3. [PyTorch CUDA 双线性算子源码](https://github.com/pytorch/pytorch/blob/v2.1.0/aten/src/ATen/native/cuda/UpSampleBilinear2d.cu)
   中的非确定性限制针对 backward。验证只有 forward，可直接调用普通 PyTorch 使用的 ATen forward。
4. 原来防 OOM 的掩膜分块，解决了大块分配，却没有避开上述慢算子。
   原图可达数百万像素，300 个候选反复调用它，开销非常大。

以上是源码与日志共同支持的主要嫌疑链，**不是已经测量出的瓶颈占比或加速倍数**。
本地只有 CPU，服务器 GPU 上的提速、显存峰值与最终 AP 仍需下面的速度检查验证。

## 为什么 YOLO 平时验证快

[官方 YOLO 8.4.60 分割验证器](https://github.com/ultralytics/ultralytics/blob/v8.4.60/ultralytics/models/yolo/segment/val.py)
默认 `save_json=False/save_txt=False` 时使用 `process_mask`，在原型掩膜网格上匹配 GT/预测。
640 输入通常对应约 160×160 的原型网格；指标匹配主要在 GPU 上做矩阵计算，
不是每张图、每个候选都恢复到 3072×4096 再转 CPU/RLE。
若导出 JSON，它也有更高分辨率恢复与编码开销，不能简单说“YOLO 的 COCO 原图评估永远很快”。

MMDetection 的官方 COCO 路径则恢复原图掩膜并编码 RLE，工作量本来更大。
不能拿不同网格的 AP、不同后处理流程的验证耗时当成严格同口径比较。
本套最后的 `val_common.json` 仍在原图 FP32 + COCO mask AP maxDets=100 上统一评估。

## 本次实际代码修改

| 范围 | 修改 | 保持不变 |
|---|---|---|
| YOLO/MMDet/RF-DETR worker | `inference_runtime.py` 仅在无梯度、CUDA、bilinear、非抗混叠、严格确定性开启的条件下转普通 ATen forward | 有梯度的训练 interpolate、全局确定性/TF32/AMP 标志、采样尺寸/align_corners/scale_factor |
| RTMDet / SOLOv2 | 保留原来的两级插值、阈值、NMS 与 CPU bool 分块；上述原生算子修复覆盖其最终恢复 | 原图坐标、候选数量、评分/排序 |
| Mask R-CNN | 保留官方 ROI/mask paste 分块；共用推理优化、计时和进程释放修复 | 官方模型与后处理路径 |
| MMDet val/test 输入 | CocoMetric 已有 ann_file，停止重复解码原图 GT 位图；保留 bbox 和元信息，启用 pin_memory | 真值 JSON、COCO 标注、图像输入640、batch=1、每轮验证/选 best 的频率 |
| YOLO 最后共同评估 | 按原图像素预算分块，输出直接转 CPU uint8；不再把所有原图掩膜 `torch.cat` 回 GPU | 官方 process_mask_native，每个预测仍保留 |
| 所有共同评估 | 复用 RLE 计算 mask bbox，去除 `np.nonzero` 巨型坐标数组及多余全掩膜复制 | 掩膜和 bbox 几何定义、置信度、COCO 参数 |
| 全部框架批量运行 | 训练 worker 完全退出，批量父进程才启动评估 worker | 前台顺序运行、VSCode 三角形入口、AMP 配对、固定超参数 |
| 可观测性 | 每轮 MMDet 写 `validation_timing.json`，共同评估写 `evaluation_timing.json` | 不把小样本测速写成完整 AP 或 complete.json |

原来“训练 worker 内调用 subprocess 后等待评估”的做法**并不能保证释放训练 CUDA context/cache**。
本次改为两个顺序且不重叠的 worker 进程，批量父进程只用标准库，不持有 GPU 模型。

注意：原生 CUDA 和分解实现的最后几个 FP32 位可能不同，极接近阈值的像素也可能受影响。
这不是改分辨率、阈值或训练精度，但不能承诺 GPU 输出逐位相同或 AP 绝对一字不差。
服务器预检会对小张量校验误差，并记录到日志；正式指标请用修复后的共同评估再核验。
有梯度的训练分支不切换到原生非确定性 backward。

## 上传什么

复制更新的整个 `baseline_amp/` 目录和两个根目录文件：

- `RUN_CITRUS_BASELINES_AMP.py`：开始新的非 YOLO 批量实验。
- `RUN_BASELINE_RECOVERY.py`：已有实验测速/续训/复评，不需要从零重跑。

不要盲目覆盖服务器上自己填写的 DATA、DEVICE、PYTHONS。
无需重新安装环境，也不要为了这个修复升级 torch/mmcv。程序仍要求现有固定版本。
先停止旧训练实例再做续训；正在运行的 Python **不会自动加载刚上传的新源码**。

## 先测已有权重，不重新训练

编辑 `RUN_BASELINE_RECOVERY.py`：

```python
"RUN_DIR": "/data/sxq/results/BASELINES/baseline_2/rtmdet_ins_tiny_amp1_seed42_300ep",
"DEVICE": 1,
"MODE": "profile",
"PROFILE_IMAGES": 5,
"CHECKPOINT": "",
```

RUN_DIR 只是目录示例，必须换成**自己实际有 job.json 的模型目录**，不是批量项目父目录。
不清楚名称时执行：

```bash
find /data/sxq/results/BASELINES -maxdepth 3 -name job.json
```

PYTHONS.mmdet 填 `/home/amax/.conda/envs/citrus_mmdet/bin/python`，
modern 填 `/home/amax/.conda/envs/citrus_baseline/bin/python`（以实际环境为准）。
CHECKPOINT 留空会读 trained.json 的已选权重；尚未训练完则读 MMEngine 的 `train/last_checkpoint`。
非 MMDet 未完成运行可显式填已有 checkpoint 来测速，不能借此声明完成训练。

运行：

```bash
cd /data/sxq/code/ultralytics-main-new
/home/amax/sxq/bin/python RUN_BASELINE_RECOVERY.py
```

也可以在 VSCode 打开此文件点击右上角三角形。没有 nohup、后台启动或单卡占用保护。

结果存入该模型目录下**新建**的 `runtime_profile_日期时间/output/evaluation_timing.json`。
只测验证集原图面积最大的 5 张照片，包含 GPU 同步、CPU 传输和 RLE 编码。
第一张含模型热身，不是推理延迟基准；这个结果也不是全验证集平均速度或 AP。
关注 `predict_seconds` 与 `encode_seconds`，以及 `native_bilinear_calls`。
若仍慢，把该 JSON 和 recovery.log 回传，才能继续区分 GPU 恢复与 CPU/RLE/I/O 瓶颈。
只运行这些图片，不会训练，不会覆盖旧结果或生成 complete.json。

## 继续已跑到几十轮的 RTMDet

速度检查成功后，把同一文件的 `MODE` 改成 `"resume"`，CHECKPOINT 留空，再运行。
读取原 `effective_config.py`，仅调整无用的验证 GT 解码和输入 pin_memory；
MMEngine 恢复 last_checkpoint 的 epoch、优化器、学习率调度、AMP scaler 状态。
原 initialization.json、dataset.json 和 effective_config.py 不被改写；
修复后的配置另存 resumed_effective_config.py，runtime_repair.json 记录修改。
数据路径保持已保存实验中的路径，不重新生成或改变划分。
中断在一轮中间时从最近已保存轮恢复，不承诺恢复到尚未保存的 batch。

这个入口只续当前模型，不自动启动整套旧队列，也不修改旧 batch_protocol.json 的代码指纹。
不同运行时版本的结果应在论文中注明；修复后的 best 若有边界像素微小差异，需复核最终共同指标。

## 已完成模型重新评估

同一入口改 `MODE="evaluate"`。输出保存到新 `runtime_evaluate_日期时间/output/`，
原 val_common.json 与 complete.json 不覆盖。此模式完整验证全部 val 图，不是5张测速。
支持 YOLO、RTMDet、Mask R-CNN、SOLOv2、RF-DETR，不重新训练。

## 未来新的批量任务

仍打开 `RUN_CITRUS_BASELINES_AMP.py`，填写自己的 DATA、DEVICE、PYTHONS，直接运行。
SUITE 默认 non_yolo，AMP_MODES 保持 `[1, 0]`，新输出默认
`BASELINES_NONYOLO_SPEEDFIX_20260930`，不复用旧 batch_protocol 的指纹。
不改优化器/学习率/batch/seed/输入尺寸来伪装提速；也没有偷偷设成每10轮才验证。

## 本地验证与边界

当前本地 Python 3.9 / Torch 2.8 CPU，仅利用隔离的 MMDetection 3.3.0 源码、MMEngine 配置测试。
最终相关回归测试 **103 项通过、1 项 CUDA 专项跳过**；Ruff、git diff --check、批量 dry-run 与入口 help 检查通过。
记录位于 `baseline_amp/VERIFICATION_SPEED_20260930.json`。
已经验证原生插值参数映射、训练梯度路由不变、CPU 掩膜/官方后处理一致、RLE bbox、
三种 MMDet 配置及旧错误回归、分进程阶段和 profile 不写 complete 等契约。
CUDA 专项测试在本地跳过；没有运行服务器完整训练，也没有声称已经实测加速多少倍。

服务器新批量预检会用严格确定性开启的实际大掩膜解码，再做小张量原生/分解误差检查。
要验证本次修复，请先用现有权重测5张，再续训或开新任务，不必先投入一整轮300 epoch。
