# baseline bug2(1)：图像与实例掩膜对齐修复

## 根因和范围

用户日志：RTMDet-Ins-tiny AMP1 已进入训练，在 `RTMDetInsHead.loss_mask_by_feat` 的
`torch.cat(pos_gt_masks, 0)` 报 `Expected size 640 but got size 480`。
此前优化器重复参数问题已通过；这不是同一个错误，也不是 meshgrid 的警告造成的。

本项目把官方训练管线改成了 `Resize(640, keep_ratio=True)`，移除了官方固定尺寸 Pad。
等比缩放后横图和竖图的宽高不一致，图像在批次预处理时自动补齐，但官方
`DetDataPreprocessor` 默认 `pad_mask=False`，导致标注掩膜仍为各自尺寸。
RTMDet 继承的默认 padding divisor 还是1，也没有保证任意图像形状满足多尺度主干的32倍数约束。
这是适配代码遗漏，不能靠重装环境、关闭 AMP、把 batch 改为1或删除小目标规避。

依据：[官方 RTMDet-Ins 训练 Pad 配置](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/configs/rtmdet/rtmdet-ins_l_8xb32-300e_coco.py)、
[官方预处理实现](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/data_preprocessors/data_preprocessor.py)。

## 修复

所有三个 MMDetection 实例分割基线统一使用：

```python
pad_mask=True
mask_pad_value=0
pad_size_divisor=32
batch_augments=None
```

保留原始归一化 mean/std、颜色通道顺序、等比缩放、batch、AMP配对、优化器和种子。
补齐是右侧/底部背景 padding，不是拉伸、裁剪或标签修复。图像按批次最大宽高补齐，
不要求每个批次均为640正方形。训练掩膜与模型输入对齐；验证掩膜仍在原图尺寸，避免评估坐标错位。
上次 `bypass_duplicate=True` 修复同时保留。没有修改原始数据或已有结果。

## 已验证与未验证

本地独立依赖目录：`E:/mastercode/_work/baseline_bug2_20260926/deps`。
只在此目录安装 MMEngine0.10.7、MMDetection3.3.0、MMCV-lite2.1.0及辅助依赖用于CPU检查；
没有替换主环境，也不要把 MMCV-lite 安装到服务器训练环境。

本地43项测试全部通过，包括：

- 三个官方模型配置在AMP0/1下逐项加载、生成和核对；此前缺包跳过的3项本次也执行通过。
- 真实 MMEngine 共享参数优化器旧错误复现，修复后唯一性检查和参数更新。
- 从官方wheel源码隔离加载未经改写的预处理/BitmapMasks类，复现未pad掩膜的cat错误。
- 正确padding后640×480/480×640、127×95/95×127批次可拼接；空标注、单像素前景保留。
- 训练掩膜像素/框不变，验证掩膜不被改为训练画布尺寸。
- 已有基线数据转换、配对协议、结果汇总、环境检查回归。

CPU源码隔离检查不包含编译算子，也不能证明CUDA全模型训练成功。本地无服务器对应的CUDA环境，
没有声称已完成真实服务器训练。

服务器的批量预检查新增：三个MMDetection模型各自合成横竖图批次，实际AMP0和AMP1损失、反向、
优化器更新，再检查FP32预测掩膜原图尺寸。每项成功才输出对应OK；任意失败停止队列。
合成检查为128级尺寸，不能替代640真实数据的显存/吞吐验证。检查AMP临时模型的GradScaler以1起步，
避免正常的初始溢出回退被误判；正式训练仍用原配置的dynamic默认尺度。模型在独立预检查进程销毁，
正式训练重新设置seed并初始化，不污染随机初始化配对。逐任务轻量复核不重复全部6次合成训练检查。

## 服务器操作

1. 上传最新的整个 `baseline_amp` 文件夹到 `/data/sxq/code/ultralytics-main-new/baseline_amp`，
   尤其不要漏掉新增的 `mmdet_smoke.py`。无需重装环境。
2. 可保留服务器 `RUN_CITRUS_BASELINES_AMP.py` 的 DATA、PYTHONS、DEVICE 设置。
3. 换新输出目录。原 `baseline_2` 保留；不能把改变实现后的实验混写进去。

直接运行（自动先检查、再训练，仅非YOLO的4模型×2AMP）：

```bash
cd /data/sxq/code/ultralytics-main-new
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py \
  --suite non_yolo \
  --project /data/sxq/results/BASELINES/BASELINES_NONYOLO_MASKFIX_20260926
```

若只想先检查，在上面命令最后加 `--preflight-only`。成功后去掉该参数运行，项目参数未变可复用。
VSCode三角形方式：将原入口中的 `SUITE="non_yolo"`、`PROJECT` 改为上述新目录，保留自己的其余设置后点击运行。

新日志应出现 `MASK PADDING PREFLIGHT OK`、各模型AMP0/1的 `TRAIN STEP PREFLIGHT OK` 和
`PREDICTION PREFLIGHT OK`；三个模型通过后才开始长训练。
首轮实际训练仍建议观察loss、首轮验证及保存结果；预检查不能担保长期硬件、内存或磁盘永不出错。
