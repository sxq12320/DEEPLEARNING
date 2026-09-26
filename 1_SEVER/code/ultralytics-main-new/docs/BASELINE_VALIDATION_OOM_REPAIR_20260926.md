# bug2(2)：第一轮验证时原图掩膜恢复显存溢出

## 日志证据

2026-09-26 12:23—12:24，RTMDet-Ins-tiny AMP1完成第1轮85个训练批次并保存checkpoint，
随后验证在`rtmdet_ins_head.py`第500行的原图尺寸双线性插值处OOM。
GPU总显存23.69 GiB、进程占用22.56 GiB、剩余1.10 GiB，下一次分配需要10.55 GiB。
这不是训练batch本身溢出，不是此前的掩膜尺寸不一致，也不能从逻辑cuda:0推断跑错物理卡。

模型640输入与验证内存并不矛盾：输出需要恢复到原图，每个实例都产生一张原图大小的掩膜。
低分阈值保留较多候选时，完整FP32掩膜及插值临时量开销可远高于训练特征。
原配置RTMDet验证batch=5又会叠加多图输出。cache=True通常使用系统内存，不增加GPU显存。

## 已修改

- 新增`baseline_amp/mmdet_memory.py`，在当前worker进程内适配固定MMDetection3.3.0，
  不写入site-packages，不修改网络权重或训练损失。
- RTMDet保留官方框缩放、score factor、尺寸过滤、NMS、候选顺序和上限；
  两阶段双线性插值按实例分块执行，完成阈值化后将布尔掩膜存CPU，避免整批FP32大图常驻显存。
- 分块上限8个，并按中间／原图展开面积进一步缩小；大图通常逐实例恢复。
- 同时检查到SOLOv2的类似插值路径，保留原有动态卷积、maskness、Matrix NMS和先裁剪再恢复顺序，
  仅对最终掩膜恢复采用分块方式。Mask R-CNN原本有分块粘贴逻辑，本次不替换该实现。
- 三个MMDetection模型验证、测试batch=1；训练batch、AMP0/1、seed、阈值、候选数、原图评估不变。
- 适配同时安装于预检查、训练中验证、最终共同评估，防止训练结束后的独立评估再次OOM。
- 增加3072×4096原图、16个掩膜的服务器CUDA解码检查，输出额外显存峰值。
  它不是整数据集最坏情况保证，但覆盖了先前128级训练检查遗漏的原图恢复阶段。
- 每轮记录`amp_runtime.json`：AMP检查迭代数、loss scale下降次数与当前scale。
  不修改正式训练的GradScaler策略。

官方实现依据：
[RTMDet-Ins](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/rtmdet_ins_head.py)、
[SOLOv2](https://github.com/open-mmlab/mmdetection/blob/v3.3.0/mmdet/models/dense_heads/solov2_head.py)。

## 测试与边界

本地70项测试通过：包含此前43项回归，以及分块与整块掩膜比较、分块上界检查、
官方RTMDet后处理对照、官方SOLOv2与真实Matrix NMS对照（正常／无候选／无有效掩膜／NMS后为空）。
测试覆盖不同实例数、阈值、非整数尺度和rescale开关；掩膜及其关联字段在CPU测试中完全一致。
RTMDet的NMS使用两条路径共享的受控oracle，测试选择顺序和字段处理，不冒充CUDA NMS验证。
本地没有相同CUDA环境，服务器实际峰值与长时间训练尚未验证。最终布尔掩膜仍需要系统内存，
CPU保存和传输也有开销，不能宣称该修复必然加速。极高分辨率图像或RAM不足仍需按日志诊断。

日志中的grad_norm早期出现inf、后段恢复有限值，和本次验证OOM是两件事。
AMP动态缩放可能触发溢出回退；日志平均窗口中的inf不能直接解释为每一步都溢出。
也不能因为loss有限就断言所有更新成功。新记录用于进一步判断回退是否持续。

## 服务器如何使用

上传最新整个`baseline_amp`文件夹，特别包含`mmdet_memory.py`与`mmdet_smoke.py`。
保留服务器原入口中的DATA、PYTHONS和DEVICE设置。无需重装环境，旧结果和首轮checkpoint保留。
由于实现已变化，使用新PROJECT；本次没有擅自实现跨版本断点续训。

```bash
cd /data/sxq/code/ultralytics-main-new
/home/amax/sxq/bin/python RUN_CITRUS_BASELINES_AMP.py \
  --suite non_yolo \
  --project /data/sxq/results/BASELINES/BASELINES_NONYOLO_MEMFIX_20260926
```

仍为前台串行4种非YOLO模型×2种AMP，共8次，最多300轮。
VSCode三角形运行：把原入口中的SUITE设为non_yolo，PROJECT改为上述新目录，保留自己的数据和环境路径。
先观察`LARGE MASK DECODE PREFLIGHT OK`，然后确认第一轮验证完成且打印COCO指标。
不要因显示训练epoch就认定整条训练—验证—保存—最终评估流程已经通过。
