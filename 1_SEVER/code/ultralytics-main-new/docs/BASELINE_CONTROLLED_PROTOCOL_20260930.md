# 全部 baseline 的统一验证与选模协议

更新日期：2026-09-30。适用入口：`RUN_CITRUS_BASELINES_AMP.py`，不是归档实验或独立工作台。

## 结论与范围

七种 baseline 接入同一套掩膜评估实现、验证频率、原图选模规则和最终重评规则。
训练仍调用锁定版本的官方 trainer，只替换验证入口；不向基线添加创新模块、改变匹配/损失公式。
不把 YOLO 原生日志的 AP 与 MMDetection 原生 COCO AP 混在一个排名中。

新训练协议：`legacy78_scratch_fixedbudget_commonval_v1_20260930`。
评估协议：`common_mask_coco_v1_20260930`。
默认新目录：`BASELINES_CONTROLLED_20260930`。不能沿用旧结果目录、覆盖旧权重，或给旧运行补标签声称它已受控。

按实验设计技能锁定因变量、候选选取频率、AMP 成对因素和跨框架差异；验证图像不是训练重复实验单位。
筛选 seed=42，正式比较使用相同三个 seed=42/43/44，各模型 AMP0/AMP1 配对。不能把单 seed 的差值宣称统计显著。

## 严格统一的部分

| 项目 | 全部模型执行规则 |
|---|---|
| 数据 | 同一个 DATA 所指的 train/val/test 成员与标签；默认 grouped_dedup，不重新划分、不筛除小实例 |
| 初始化 | scratch；记录真实初始张量校验值，AMP 对必须相同 |
| 训练预算 | 默认固定 300 epoch；全部关闭原生早停，避免不同停止/挑峰规则 |
| AMP | 同模型同 seed 分别 AMP0/AMP1；其余配方固定，显式核对实际 AMP |
| 每轮快验证 | 全部 val 图像，统一缓存的 640×640 等比 letterbox 网格；只做监测 |
| 原图验证 | epoch 1、10、20……及最后一轮；1–3 轮 smoke 时也保证首末原图验证 |
| AP 实现 | 同一个 `pycocotools==2.0.11` COCOeval，mask IoU 0.50:0.05:0.95，101 个 recall 位置 |
| AP 候选 | score≥0.001；当前单类任务最多 100 个候选/图；COCO maxDets=[1,10,100] |
| 固定 P/R | score≥0.25，mask IoU≥0.5，同样的候选上限；不是各自最佳 F1 阈值处的 P/R |
| 验证精度 | 全部 FP32、关闭 TF32/autocast，确定性前向；验证后恢复随机状态和模块 train/eval 状态 |
| 最佳权重 | 仅按定期原图 Mask AP50–95 严格增大选择；相同分数保留较早轮次 |
| 最终评估 | 训练进程退出后，独立进程用选定 FP32 权重、同一预测适配器、同一原图评估器重评 |

快验证的 GT polygon/图像一起变换，原图标注、所有图像 ID 与实例 ID 保留。
极小掩膜在 640 网格上可能栅格化成零面积，记录在 `_prepared/validation_quick640/view.json`，不删除实例。
因此 **快验证 AP、640 网格 APS 都不能代替论文的原图 AP/APS**。
原图验证始终使用原始像素坐标与原图掩膜面积分桶。

没有美化 PR 曲线、修改 IoU、填平末端或移除难例。PR 末端的插值/未达到召回区域不能靠更换绘图规则证明识别改善。

## 不能伪称完全相同的训练部分

| 家族 | 实际输入 / batch | AMP1 类型 | 选模权重 |
|---|---|---|---|
| YOLOv8n / YOLO11n / YOLO26n-seg | 640 / 16 | FP16 | 官方 EMA |
| RTMDet-Ins-tiny | 640 / 8 | FP16 | 官方 EMA |
| Mask R-CNN R50-FPN / SOLOv2-Light R18-FPN | 640 / 2 | FP16 | 普通权重（原配置无 EMA） |
| RF-DETR Seg Nano | 624 / 2，累积 8 次 | BF16 | 官方 EMA，仅这一条权重轨迹有资格选模 |

固定共同基础项：AdamW、基础 LR=.001、WD=.0005、warmup=3、scratch、整图输入、同一 seed 集合。
保留旧 legacy78 配方的 YOLO 学习率/增强/损失参数，但 `patience` 由 100 改成 0。
MMDetection 删除 EarlyStoppingHook，RF-DETR `early_stopping=False`。
同时修正旧配置递归改 `max_per_img` 误把 Mask R-CNN 的训练/RPN proposals 上限也降成300的问题：
新协议保留官方训练/RPN上限1000，只将最终实例输出上限设成100。
这既避免无效恢复大量原图掩膜，也不把内部候选提案错误当成最终 COCO maxDets。

仍存在模型特定差异：

- RF-DETR 的 patch12/window 整除要求使本配置采用 624；它不是 640 模型，更不是原生 312 Nano。
- YOLO/MMDetection 原图预处理等比；RF-DETR 沿用官方推理的正方形拉伸。共同输出在原图坐标上评估。
- YOLO 有 nbs 累积与 bias/momentum warmup、mosaic；MMDetection 没有对应的原生同义机制。
- RF-DETR 保留自身增强、AdamW betas、阶梯调度；各框架有效 batch/更新次数并不相同。
- NMS、SOLO matrix NMS、无 NMS query 选择、掩膜解码属于各模型官方方法，不强行改成同一种检测器。
- 三个环境的 Torch/AMP dtype、模型损失、EMA 细节不同，不能把全部差值都归因于架构。

因此该批属于“共同数据、预算、评估与选模下的跨框架方法比较”，不是“所有训练变量相同的纯架构消融”。
严格架构消融应在同一 YOLO trainer 中对基线/改进模型采用相同输入、增强、优化器、损失、AMP、EMA 和初始化。
不得只用 AMP1 的较低基线对照 AMP0 的改进模型；两种 AMP 应分别成表或配对报告。

## 运行

上传整个 `code`，保留已安装的官方 baseline 两个隔离环境，不在这些环境执行研究包 `pip install -e .`。
在入口顶部仅修改 `DATA`、`DEVICE`、`PYTHONS`、必要的 `PROJECT`。

首次先将 `EPOCHS=3`，PROJECT 命名为新的 smoke 目录，`SUITE="all"`：

```bash
cd /data/sxq/code/ultralytics-main-new
python RUN_CITRUS_BASELINES_AMP.py --dry-run
python RUN_CITRUS_BASELINES_AMP.py --preflight-only
python RUN_CITRUS_BASELINES_AMP.py
```

也可在 VSCode 直接点击该文件的运行三角形。全队列 7 模型×2 AMP=14 次，前台串行，无占卡保护。
确认 14 个 smoke 均生成 `complete.json`，再改 `EPOCHS=300`、换新 PROJECT 开正式训练。
正式论文将 SEEDS 改为 `[42,43,44]`，全队列共 42 次。Ctrl+C 终止当前任务并停止后续队列。

需要暂时跳过 YOLO：`--suite non_yolo`；但旧 YOLO 的峰值/旧选模不自动变为同协议结果。
旧结果保留为历史参考，若纳入这张受控表仍需按新协议重新训练。
旧中断的 MMDetection 可继续使用 `RUN_BASELINE_RECOVERY.py` 恢复原协议，不会被偷偷转换到新协议。

## 看哪些结果

每个任务：

```text
job.json                         固定训练/验证协议
effective_train.json 或
effective_config.py              官方训练配置及适配内容
initialization.json              真实初始状态校验
amp_actual.json                  AMP 是否实际启用及 dtype
validation_common/
  epochs.csv                     quick 与 original 分列，未做原图验证的行留空
  epoch_0001_ema.json             逐轮指标、scope、时间和选模状态
  best.json                      原图选模分数、轮次、权重类型和权重路径
train/…best_unified.pt/.pth       实际被验证并选中的 FP32 快照；仅推理权重
train/…last.pt 或 epoch_*.pth     官方续训权重，不能拿 best_unified 当完整优化器续训文件
val_common.json                  最终同协议原图指标
complete.json                    训练且最终评估成功才生成
```

全项目 `baseline_summary.csv` 用最终 `val_common.json` 排名；`amp_paired_deltas.json` 检查初始化/数据/协议一致后计算成对差值。
YOLO 原生 CSV 的 mask 字段此时是 quick640 监测；bbox AP/val loss 未计算，明确留 NaN，不伪造数值。
RF 原生日志可包含 regular/EMA 快验证与原图记录，不用原生 best_total 排名，使用 best_unified。
MMDetection 保存 best 时直接复制已经参与验证的权重，避免 EMAHook 再次交换而存成普通权重。
最终加载后 AP 与选模分数容差 1e-5 不一致就停止，不静默标记完成。

## 速度与验证边界

若训练 300 轮，原图选择验证次数由每轮 300 次变为 **31 次**，减少约 89.7%的原图验证次数；
另有每轮 quick640、训练结束后的原图重评，以及 RF regular quick 监测。
这是次数变化，不是已经测得总耗时降低89.7%，也不承诺任何固定加速倍数。
原图 GT 不重复解码到 MMDetection val loader；YOLO/RF-DETR 大掩膜按实例分块再转 CPU。
文件预测专用 pipeline 去掉 LoadAnnotations：inference_detector 的输入没有 GT instances，
直接复用带标注加载器的 val/test pipeline 会报 KeyError；Resize/PackDetInputs 保留，GT由共同COCO评估器读取。
既有 RTMDet/SOLO bounded-mask、无梯度 deterministic bilinear 修复保留，训练梯度路径未改。

本地验证详见 `baseline_amp/VERIFICATION_COMMON_20260930.json`。
CPU 契约测试不能替代服务器 CUDA 混合精度与真实数据 smoke；正式训练前必须检查原图选模、类别、NaN/OOM 和速度。
本次没有运行服务器训练、没有修改原始数据或已完成结果、没有把旧验证集峰值当最终 test AP。

实现参考：

- [COCOeval 官方匹配和统计](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py)
- [Ultralytics 8.4.60 官方分割预测](https://github.com/ultralytics/ultralytics/blob/v8.4.60/ultralytics/models/yolo/segment/predict.py)
- [MMEngine 0.10.7 EMAHook](https://github.com/open-mmlab/mmengine/blob/v0.10.7/mmengine/hooks/ema_hook.py)
- [RF-DETR 1.4.0.post0 官方训练入口](https://github.com/roboflow/rf-detr/blob/1.4.0.post0/rfdetr/main.py)
