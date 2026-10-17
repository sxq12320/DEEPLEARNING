# Baseline 验证提速：保持评价口径（2026-10-07）

## 结论

已改进 baseline 的验证执行链，不改网络、数据划分、训练超参数、AMP 配对或指标定义。
I/J 等创新系列的 EV6Validator **未改动**。
这不是宣称服务器已经达到 YOLO 的验证速度；本地没有 CUDA，实际加速需测已有权重。

## 为什么不能直接换成 YOLO 的默认指标

YOLO 分割验证默认不导出 JSON 时，通常在原型掩膜网格上匹配；当前 baseline 统一口径则为
COCO mask AP，原图阶段还需恢复每个实例的原图掩膜。二者工作量和指标计算并不相同。
借鉴输入预取、少复制、少重复计算的工程方法，不把降低评价分辨率当成等价加速。

原回传日志曾出现 193 图验证约 34 分钟，COCO 汇总仅数秒。
已有 9 月 30 日修复针对 PyTorch 2.1 严格确定性下 CUDA bilinear 的慢分解，
该修复继续保留。不能据此断言本次服务器仍是同一瓶颈；请检查下面的新分段计时。

## 修改

| 环节 | 新实现 | 不变的内容 |
|---|---|---|
| 三个框架输入 | 有界 CPU 线程提前读图和预处理；按原顺序交给主线程推理。上限4张，RF-DETR上限2张；WORKERS=0串行 | batch=1、颜色通道、resize/归一化顺序、模型前向与FP32 |
| MMDet入口 | 官方单图pipeline + test_step，输入预取真正生效 | 官方单图后处理；并非把图片列表传给inference_detector假装批推理 |
| 大掩膜回传 | YOLO/RTMDet/SOLOv2直接阻塞复制到最终CPU切片，删除CPU临时副本 | 两级插值、阈值、分块预算、像素值与候选 |
| YOLO原图掩膜 | 全部非空时不做复制整个数组的布尔索引 | 空掩膜仍按原规则过滤 |
| RLE与候选 | 编码前做同样的稳定每类别top-100；uint8/Fortran布局一次转换 | score>=.001、同分顺序、坐标、RLE和bbox |
| 共同评价 | 缓存quick/original两份GT的RLE；AP与固定P/R复用IoU。每轮复制GT元数据以隔离COCOeval突变 | 官方COCOeval、固定P/R的原greedy匹配与crowd行为 |
| 测速与日志 | common任务的profile也走同一predictor/cap；记录predict/encode/COCO/input_wait | profile不计算子集AP，不改best、val_common或complete |

仍为每轮全验证集 quick640；原图第1、10、20…及最后一轮验证并选模。
原图 Mask AP50-95 严格更高才保存，平分保留早期权重；频率、COCO maxDets=100、
P/R阈值 score=.25 / mask IoU=.5 均不变。RF-DETR regular/EMA原有监测安排也未偷偷减少。
没有缓存整套图像到RAM，也没有把整幅原图掩膜固定在pinned memory中。

## 你在服务器上怎么用

1. 上传更新后的整个 `baseline_amp/` 文件夹。新实验再上传 `RUN_CITRUS_BASELINES_AMP.py`；
   已有权重测速/续训使用 `RUN_BASELINE_RECOVERY.py`。保留你自己填写的 DATA、DEVICE、PYTHONS。
   无需重装或升级 torch/mmcv。
2. **先测速，不从零训练**：打开 `RUN_BASELINE_RECOVERY.py`，RUN_DIR填具体模型目录
   （含job.json，不是整个批量项目），MODE="profile"，PROFILE_IMAGES=5，CHECKPOINT留空，
   DEVICE/PYTHONS填原环境。VSCode点击右上角三角形，或执行 `python RUN_BASELINE_RECOVERY.py`。
3. 输出在模型目录的新 `runtime_profile_时间/output/evaluation_timing.json`。
   common任务 `timing` 为阶段汇总，`per_image_timing` 为逐图记录。
   重点看predict_seconds、encode_seconds、input_wait_seconds；profile不运行COCO评价。
   最大5张不是全验证集平均速度；第一张含热身，不是部署延迟基准。
4. 已有未完成MMDet任务可停掉旧进程后，把MODE改成"resume"，沿用优化器/调度/已保存epoch。
   **正在运行的Python不会自动加载新文件**。旧原生验证任务续训仍保持旧评价协议，不强制转为common。
5. 新批量实验仍运行 `python RUN_CITRUS_BASELINES_AMP.py` 或VSCode三角形。
   默认新输出名 `BASELINES_CONTROLLED_FASTVAL_20261007`，保护旧结果；已跑过YOLO可选SUITE="non_yolo"。
   不复用旧代码指纹锁定的PROJECT。不要删旧结果来绕开指纹检查。

训练中的完整分段计时保存在 `validation_common/epoch_XXXX_*.json`；epochs.csv保留quick/original总耗时列。
input preparation可以与推理重叠，prepare_seconds不可再次加进总耗时。

## 验证边界

新增测试以冻结的旧实现为参考，覆盖非零AP、同分、阈值、crowd、空mask、多类别、
超100候选、缓存隔离、同步/预取输入一致、按序/有界/线程清理、profile结果保护。
本地CPU数值一致不等于已经验证服务器CUDA速度；CUDA测试须在原服务器环境执行。
最终测试记录见 `baseline_amp/VERIFICATION_FASTVAL_20261007.json`。

## 官方源码依据

- [YOLO 8.4.60分割验证器](https://github.com/ultralytics/ultralytics/blob/v8.4.60/ultralytics/models/yolo/segment/val.py)
- [MMDetection 3.3.0官方单图推理](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/apis/inference.py)
- [COCOeval官方实现](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- [PyTorch 2.1 interpolate慢分解路径](https://github.com/pytorch/pytorch/blob/v2.1.0/torch/nn/functional.py)
