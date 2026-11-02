# J7R：精度优先的重构、依据与运行

> **晚间V6回传更新：当前优先队列已经改为J116/J117/J118，新增原生full-PAN三臂，旧六臂结构保留。最新运行说明和结果表见[J7R_V6_REFRESH_20261019.md](J7R_V6_REFRESH_20261019.md)。下面六臂预算和审查是早版记录，不代表当前priority含义。**

2026-10-19。状态：6个模型/批量入口已实现，200项针对性回归通过，真实图片副本一轮检查通过；**没有正式训练AP，不能称已涨点或最强网络。** 原J100–109与所有历史结果保留，不混成同一轮实验。

## 1. 本轮怎么改了方向

用户要求精度与召回优先，计算量/参数尽可能接近YOLO11n-seg；压缩放到后面。

此前J7主变化是24ch公共hub、低网格深层通道混合及共享窄塔。这些省计算的假设尚未正式训练，不能称已经失败；但不足以直接回答小果是否被有效识别。本轮主候选不采用这些压缩，保留成熟结构的语义主路、PAN/P4空间返回、宽Proto和独立三尺度塔。

新增的是**真实像素的读入路径**：高分支不自己判“有几个果”、不独立出框、不乘另一套质量分。它把真实高分证据送给同一个实例网络。主候选J112只读C3一处；mask-only/joint是对照，不预称模块越多越好。

## 2. 从结果和数据得到什么

- 当前本地965图/5897GT；640投影面积<16²有1337（22.67%），val183/1049。817图原始3072²，压到640确有观测损失。
- 色彩均值接近、条带叶枝导致的深凹可见轮廓、近邻实例是相交但不同的问题。灰度/边缘不自动可分；不使用强圆形/凸包补全。
- 历史I22/I23的P2候选使某些工作点R增加而P下降，AP未净增；历史压Proto/删P4及紧凑骨干有反例。本轮不先砍主语义再指望细节救回来。
- 本轮真实成熟J90/J94全val193图/1049GT审计：扩域后无几何候选均0，最终未分到正点仅4/7（0.381%/0.667%）。这不足以解释整体召回缺口，因此**不把重写TAL或新增loss当默认主方案**。
- 一个近重复标注位于val/IMG_0627第2/6行，列为待人工复核，没有删改标签。它不是整个曲线问题的解释。
- 框的非负LTRB乐观IoU上界<.5仅1个GT；这个框界不是Mask AP界，不支持“大量漏检由解码几何硬限制”。
- 当前正式grouped_dedup目录本地缺失，未静默换回旧数据；服务器DATA按实际路径填写，不设置fingerprint/确认门禁。

原始CSV、实际checkpoint超参、逐GT/全val表和数学边界：
[全验证集诊断](../../../../_work/20261019_j7_taskfit/assignment/ANCHOR_GEOMETRY_AND_FULL_VAL.md)，
[数据几何](../../../../_work/20261019_j7_taskfit/dataset/TASK_FIT_DIAGNOSIS.md)，
[历史代码/结果复核](../../../../_work/20261019_j7_taskfit/history_taskfit.md)。

## 3. 结构落点

```text
同一个原始 RGB 图
├─ global640 → 原stem/C2/C3 → C3' → C4/C5 → 成熟PAN → 同一组box/cls/mask系数
│                              ↑                              ↓
└─ real-high1280 → 窄phase证据编码 → D80                 可见实例掩膜
                                      可选D320仅进最终Proto（对照臂）
```

上述数字是训练/profile方图示例，不是硬编码验证尺寸。新增高分路径在C3形成之后、C4/C5之前读入，不使用未来C4/C5倒流再偷偷重算主干。

输入先显式打包：G为RGB3、H为配准2倍RGB，X=[G,pixel_unshuffle(H,2)]为15通道。传感器还是单RGB，dataset通道不变。模型从显式张量分出两路，没有待消费的隐藏high缓存。正常raw模式缺高分就报错，只有J111明确把640放大作信息对照。

高分编码依次为相位重排、1×1投影、深度卷积；主要宽度16。读入是普通concat残差，只有残差gain零起点，输出投影不同时置零。**像素重排/普通残差不是原创；投影有损，不声称无损抗混叠。**

真正J113只改变最终Proto，固定权重测试确认box/cls/coeff不变；联合训练梯度仍共享。J115走旧P2→C4路径，会影响实例语义，绝不是mask-only。J112与J115使用D80/D160和不同后续变换，比较的是整条路径，不纯隔离“早”的作用。

## 4. 配置与实测预算

|配置|目的|参数量M|GFLOPs@640|正式AP|
|---|---|---:|---:|---|
|J110_paired_global_control|同配对数据的成熟global控制|2.260953|10.504653|未测|
|J111_c3_upsampled_control|同构伪高分，排除只因多参数|2.268889|10.858138|未测|
|J112_raw_c3_evidence|真实高分→C3，主候选|2.268889|10.858138|未测|
|J113_raw_final_proto|真实高分仅进最终Proto|2.265721|11.239680|未测|
|J114_raw_c3_and_proto|C3+最终Proto，位置对照|2.270985|11.307059|未测|
|J115_raw_inherited_ev11|同高分编码器走既有EV11路径|2.265193|10.876774|未测|

同口径官方YOLO11n-seg（nc1）：2.842803M，10.355763 GFLOPs。
J112参数约少20.19%、计算约多4.85%，是近似预算而非严格等算力；参数下降来自继承的成熟结构，不是本轮主要目标。

THOP采用完整15×640×640输入、包含高分encoder、未融合、2FLOPs/MAC；不含原图resize/传输/NMS/系数×Proto合成，部分函数式操作未充分计数。**FLOPs低不保证GPU快**，高分图缓存、打包复制、训练显存和端到端时延仍须实测。J114多约9.19%仅作可选位置对照，不默认最强。

## 5. 固定超参与验证

|项目|所有J7R臂共同设置|
|---|---|
|初始化|scratch，pretrained=False；不按旧layer编号盲迁移|
|AMP/cache|False / True|
|输入|训练H1280单次增强→AREA G640；外部mask_ratio2|
|优化器|AdamW；lr0=.001，lrf=.01，momentum=.937，weight_decay=.0005|
|warmup|3轮，warmup_momentum=.8，warmup_bias_lr=.1|
|训练|300轮，batch16，workers4，seed42，nbs64，patience301|
|增强|CP=.3，mosaic1，close_mosaic10；hsv .015/.7/.4；translate .1；scale .5；fliplr .5|
|其他|dropout0，cos_lr=False，multi_scale0，rect=False；无新loss/optimizer/assignment|
|验证|仍为I/J的EV6，mask画布为实际输入/2；conf=.001，NMS IoU=.7，max_det300，half=False|
|终评|选best_mask.pt；保留原best.pt/last.pt；PR来自同一选定权重|

验证不强制320掩膜。原EV6的rect padding保留：名义imgsz640的方图可形成672画布、high1344、mask336、9261候选；一般矩形按实际长宽推导。640方图是8400候选。不能将这些区别写成新增分辨率涨点。

首次train/val输入、GT与运行环境落档。workers0在训练开始重置Python/NumPy/CPU Torch增强RNG，不触发CUDA，不冒充sampler/worker恢复；workers>0沿用原生worker初始化。修正后的六臂真实smoke首批图序/global/GT摘要一致，**不证明全流bitexact**，RAM缓存仍有异步因素。续训不能声称逐位还原连续随机流。

每次保存完整j7r_protocol.json、初始状态、runtime、实际有效args和已加载文件列表；代码hash是记录/防覆盖，不是数据确认门禁。

## 6. 如何运行

整体上传更新后的code/ultralytics-main-new，不能只复制YAML或一个脚本。沿用原来能训练的环境；不需安装Mamba、DCN或新自定义CUDA算子，不要为此升级Torch。必要时在该目录 `python -m pip install -e . --no-deps` 仅重新绑定当前源码，不自动改依赖。

打开 `RUN_CITRUS_J7R.py`，编辑：

```python
DATA = "/你实际的数据集/data.yaml"
DEVICE = "1"          # 物理GPU1；绑定后日志logical0是正常映射
SUITE = "legacy"      # 本页旧六臂；最新priority改为J116/J117/J118
EPOCHS = 300
PROJECT = "/data/sxq/results/J/v7r/一个全新目录"
```

先在新PROJECT把EPOCHS改为1–3、PLOTS=True验证服务器环境，再另选正式PROJECT跑300。VSCode选择正确Python，点右上角三角形；或：

```bash
python RUN_CITRUS_J7R.py --dry-run
python RUN_CITRUS_J7R.py
```

dry-run只构建/profile，不加载数据，也不证明绘图通过。最新SUITE="all"跑九组、"legacy"跑本页六组；ONLY="J112"只跑一组。无nohup、无占用/单GPU保护、无数据确认；Ctrl+C停整队列，不启动下一模型。已完成且记录一致才跳过，半成品不覆盖，换新PROJECT或ONLY选其它模型。

也可用官方API；新模型按协议自动选正确adapter：

```python
from citrus_j7r_batch import bind_device
device = bind_device("1")  # 必须在import torch/ultralytics之前
from ultralytics import YOLO
from citrus_j2_batch import settings

model = YOLO("0_orange_yaml/J_V7R_series/J112_raw_c3_evidence.yaml", task="segment")
model.train(data="/你的数据集/data.yaml", device=device,
            project="/你的新结果目录", name="J112_seed42", **settings(300, 42))
```

直接3通道tensor送模型会报错，这是防止伪造高分；原始RGB图片路径的 `model.predict()` 已接入专用预处理。编译、TTA、导出未认证，首版明确拒绝。plots=True预检失败会在训练前报清楚，不悄悄关闭PR图。

## 7. PR问题应如何判定

PR横轴是Recall，不是confidence。Ultralytics在最大可达召回之后补零，不等于“置信度到85%精度突然归零”。不能修绘图/抬尾巴冒充漏检解决。
本轮保留原EV6和PR算法，额外保存 `selected_mask_validation/mask_pr_audit.json` 与 `raw_mask_matches.npz`，分开真实Rmax、R@P90/P95和低分尾部；相同置信度成组处理。它们是验证描述，不是测试阈值保证；仍需把剩余GT按score、预测框覆盖和框内mask质量拆解。
曲线不在Recall1保持高P本身不是bug。模型选择看真实AP、固定P下R和失败归因，而不是尾部是否好看。

## 8. 文献、数学与可否证性

本轮方法/消融复读8篇及作者代码范围见工作目录literature_taskfit.md，不声称这次读完所有612篇。重点是：
[ESOD原文](https://arxiv.org/abs/2407.16424)及[作者代码](https://github.com/alibaba/esod)的共享高分浅算/后续识别；
[OSFormer](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136780019.pdf)的语义与细mask职责；
[DCNet](https://openaccess.thecvf.com/content/CVPR2023/papers/Luo_Camouflaged_Instance_Segmentation_via_Explicit_De-Camouflaging_CVPR_2023_paper.pdf)的伪装背景关系；
[SuperYOLO](https://arxiv.org/abs/2209.13351)的分辨率/训练辅助取舍。没有复制完整作者网络，项目自己的EV11/12/J6也明确作为prior。

能论证的是观测多对一限制、相位重排可逆但降维有损、框裁mask的必要上界与计算复杂度；**不能由这些公式证明AP会增加**。若raw不胜pseudo，否定额外信息有效；若旧EV11等效，删除特殊接口；若只mask-only有效，收缩为轮廓收益，不宣称找回新果。正式基线/最终模型共同正式成员与recipe、3seed后才作论文结论，test不选结构。

research-refine五轮审查最终REVISE：工程可试，但原创机制与正式收益未成立。审查促成删门控、修正mask-only因果图与空间/RNG合同；没有为追评分快速堆出“三个大创新”。

## 9. 验证范围与下一步

本地Python3.9/Torch2.8 CPU：200项相关回归通过；6个YAML构建/前反向、fuse/EMA/权重重载、官方分派、输入/GT几何、PR审计、队列中断/防覆盖通过；六臂各两张train/两张val拷贝、一轮真实训练/终评通过。
发现一份旧benchmark测试夹具仍使用已废弃的单一prediction_counts，已更新为当前decoded_after_nms与returned_nonempty_instances双计数；未改benchmark生产算法。
本地smoke因NumPy/Matplotlib不兼容显式plots=False；未认证实际科研绘图、Torch1.13/CUDA或GPU速度。1epoch AP0没有研究意义。

本段是早版六臂工程记录。当前请按晚间更新文档上传完整code，核实DATA路径，用新PROJECT跑服务器1–3轮并生成图，然后优先J116–J118三组；旧六臂仍可选，不建议把未训练的全部旧系列同时重跑。
