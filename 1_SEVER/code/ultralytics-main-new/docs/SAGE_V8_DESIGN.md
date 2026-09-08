# SAGE V8：先保住预测能力，再重分配小目标预算

日期：2026-09-07。状态：代码与本地功能验证完成；没有V8正式精度结果，不承诺涨点。

## 一、V7R告诉了我们什么

本轮六个模型均完成300轮，训练参数一致、加载193张验证图/1049实例，676张训练图/4324实例。
修改前对照服务器 `_protocol/implementation_sha256.json`，记录的本地源码、协议和预训练文件全部一致。
按最高 **Mask mAP50–95** 选同一CSV行，不能把不同轮次的P/R/AP峰值拼起来：

| 模型 | Mask AP50–95 | 同轮Mask AP50 | 相对70的严格AP变化/百分点 |
|---|---:|---:|---:|
| SAGE70 relay control | 67.433 | 83.085 | 0 |
| SAGE72 P2 compact candidates | 64.651 | 81.922 | −2.782 |
| SAGE75 direct detail | 65.117 | 82.178 | −2.316 |
| SAGE76 shared context | 65.356 | 81.983 | −2.077 |
| SAGE77 classification-only context | 64.706 | 81.656 | −2.727 |
| SAGE78 single-scale context | 65.188 | 81.338 | −2.245 |

75比72提高0.466点、76比75提高0.239点，但都没有追回70。77比76下降0.650点，
因此**本轮不支持“只把上下文送分类分支更好”**；77也没有超过参数相同的单尺度78。
这些都是单seed结果，小幅差异还不能证明可重复收益。

源代码中，70保留原来的每尺度box/cls/mask-coefficient独立塔；72–78删除全部cv2/cv3/cv4，
改成32通道共享空间stem和线性输出，还增加P2。后者同时改变了容量、初始化和候选数。
所以不能只用70对72断言P2无用，也不能断言注意力是唯一原因。V8针对这个混杂重新做消融。

## 二、PR跳崖：绘图事实和真实漏检是两件事

当前 `ultralytics/utils/metrics.py::compute_ap` 在最后一个实际recall位置补precision=0，
再补recall=1；`ap_per_class` 用这一包络插值绘图。
这与核对的 [Ultralytics v8.4.60源码](https://github.com/ultralytics/ultralytics/blob/v8.4.60/ultralytics/utils/metrics.py) 一致。
因此最右侧横向贴零并不表示模型在recall=1时真的输出了precision=0，它根本没有达到该召回率。
**V8没有修改AP计算、NMS阈值、max_det或旧PR图片。**

另做了不补虚拟尾巴的实测PR诊断。相同本地CPU/FP32/batch1条件下：

| best_mask模型 | 实际最大Mask R | 最后一个实际P | 低阈值预测总数 |
|---|---:|---:|---:|
| 70 | 89.32% | 7.47% | 12545 |
| 76 | 87.51% | 8.89% | 10330 |

虚拟置零可以解释最后那根垂线，但不能解释掉真实尾部低精度：低置信预测中确实有大量误检。
目标应当是提高有效召回和排序质量，而不是把虚拟垂线“修漂亮”。独立诊断图及全部点保存在
`reports/sage_v8_20260907/pr70/`、`pr76/`，原结果不动。

## 三、任务痛点的量化与边界

193张本地验证图与服务器文件名集合相同，尚无服务器像素哈希，不能宣称字节级复现。
本地best_mask重验AP与服务器CSV不完全相同，因此以下子集只在相同本地条件下比较。
tiny定义为640输入下stride4栅格掩膜面积<256，**不是COCO原图AP_small**。

70在conf=.001时共漏112个Mask实例，其中80个为tiny，占漏检的71.4%；尺度问题不是仅凭视觉推测。
另有56个实例栅格面积≤64，其中1个掩膜栅格面积为0。V8未删除这些目标；
零面积监督与输入缩放损失不能靠新模块保证恢复，后续需要独立的掩膜分辨率/输入尺度实验，不能和本轮偷偷混改。

| 子集/固定conf=.25的Mask R | 70 | 72 | 76 |
|---|---:|---:|---:|
| tiny：153实例 | 22.88% | 15.69% | 15.69% |
| small：279实例，面积256–1024 | 74.19% | 67.74% | 70.61% |
| larger：617实例，面积≥1024 | 97.57% | 96.76% | 96.11% |
| 低颜色对比：359实例 | 79.67% | 77.72% | 78.83% |

对tiny再做“原始框→分数→NMS→同一实例掩膜”的失败分解：70有45个没有IoU≥.5原始框，
71个存在合格框但分数<.25，35个成功；76对应23、103、23。
**76把部分几何候选做出来了，却没有把这些候选可靠地判为果实。**
分解采用box-first一对一匹配，子集表采用mask匹配，所以两者成功数不必完全一样；这不是因果证明。

颜色不能先入为主：按尺度分层，70的低对比tiny召回27.40%，较高对比tiny为18.99%；
该颜色距离代理没有证明“越绿越难”是主导因果。它受阴影、纹理和背景环选取影响。
叶片仍是合理的困难背景，但当前最强证据是**极小尺度 + 候选评分不足**。
凹陷可见掩膜、邻接果实的拆分/合并仍需保留诊断，不能把遮挡处强制补成凸包。

76的上传混淆矩阵显示TP822、FP155、FN227；该图是框匹配统计，只有fruit/background两类，
不能由它直接判断155个误检都是叶子，更不能当作Mask IoU诊断。

## 四、历史经验：保留和停止什么

本次扫描174个CSV，去重160个，读取271个历史YAML，并按保存文件名建立174条源码候选映射。
完整表见 `reports/sage_v8_20260907/history.csv` 和 `history_source_mapping.json`。
文件名候选不是源码版本证明；不同数据、AMP、cache、训练长度的结果不做混合排行榜。
重点复核G10和V4R–V7R的结构实现，不能宣称所有历史实现都逐行读完。

| 历史证据 | 本次取舍 |
|---|---|
| G10混合Faster/WT主干、CARAFE/BiFPN、P2原型头；历史严格AP67.681，T复测只到240轮且条件不同 | 保留高分辨率证据通路的动机；不宣称所有模块各自涨点，不直接拼入整套 |
| V4R 30→42：66.719→67.502 | 保留非对称颈部与语义细节估计；是组合结果，不把每部分都算独立创新 |
| V5 late-proto两次明显掉点；relay控制较好 | 保留原型解码空间卷积，不再为了FLOPs把它全部移到低分辨率 |
| V6 60=67.503，61单向颈部=65.711，63高分支=64.331，64选择交换=63.766；62仅118轮 | 不继续无证据地简化融合或重置整套主干；62不能当完整300轮失败 |
| 本轮70强于全部压缩头；76比77好 | 恢复任务容量；不继续堆分类专用上下文门控 |

旧 `citrus_far.py` 有“步长卷积直接丢3/4像素”等过强文字；卷积会聚合邻域，不是简单抽掉3/4输入。
像素重排本身可逆，不代表后续通道投影也无损，更不能恢复输入缩放时已消失的信息。V8不沿用这些过度表述。

## 五、V8实际结构与五个实验

| 编号/YAML stem | 唯一比较关系 | 实际改动 |
|---|---|---|
| SAGE80_relay_control | 与70配置等价 | 同一主干/非对称颈部/relay/原型/独立任务头，当前协议对照 |
| SAGE81_decoupled_p2 | 对80 | 增加P2独立box、cls、mask-coefficient塔；原P3/P4/P5不压缩 |
| SAGE82_scale_budget | 对81 | 去掉P5预测塔，候选尺度变P2/P3/P4；C5上下文与颈部仍保留 |
| SAGE83_phase_scale | 对82 | 输入RGB主干stem增加亮度相位差分旁路，组合主方案 |
| SAGE84_phase_control | 对80 | 只增加同一亮度旁路，验证旁路在不加P2时是否有效 |

结构连接：RGB → 双路stem（仅83/84）→ C2/C3/C4/C5 → 原非对称语义颈部。
C2与P3形成语义细节估计D；D一支回传P3，一支修正stride4 Proto，一支形成P2候选。
82/83保留P3/P4独立任务塔，用P2替换P5的预测预算。C5不是删除，而是只承担深层上下文。
独立box/cls/系数塔分别学习，最终所有尺度共用同一个实例原型空间和标准NMS。

亮度旁路：Y=0.299R+0.587G+0.114B；将每个2×2块四个相位重排为4通道；
减去块内均值，再经4→8的1×1与8→stem宽度的3×3卷积，乘tanh(可学习增益)后加到原RGBstem。
它减少该旁路对绝对亮度的依赖，而不是宣称整网颜色不变。叶子边缘也会被激活，因此需要83/82和84/80验证。
没有整幅图FFT、动态grid_sample、unfold、额外CPU Canny、Mamba或自定义CUDA。
这只是主干入口的信息通路改造，**不是整套C3k2主干替换**；中深层保留是历史证据下的选择。

这里的有界残差增益可看作受控细节注入，但没有时间闭环、积分器、PID控制或稳定性证明，不能包装成控制理论已验证创新。

## 六、论文与代码依据：借鉴什么，不移植什么

| 依据 | 核查的代码/思想 | V8取舍 |
|---|---|---|
| [RTMDet](https://arxiv.org/abs/2212.07784)，[官方head](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py) | `cls_convs/reg_convs` 独立，跨尺度可共享卷积而BN分开 | 提醒不能将跨尺度共享混同于跨任务挤到一个小stem；V8保留任务独立，不声称复现RTMDet |
| [SPD-Conv，ECML PKDD 2022](https://arxiv.org/abs/2208.03641)，[官方仓库](https://github.com/TrustAIoT/SPD-Conv) | 空间到通道重排，原项目SPDConv也已核查 | 只借相位保留，在亮度旁路使用；不替换所有下采样 |
| [PiDiNet，ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Su_Pixel_Difference_Networks_for_Efficient_Edge_Detection_ICCV_2021_paper.html)，[代码](https://github.com/hellozhuo/pidinet) | 桌面PiDiNet/models/ops.py的差分卷积 | 借相对像素差分，独立实现四相位去均值；不是原PDC算子，更不把边缘检测成绩当柑橘AP证据 |
| [Gated-SCNN](https://github.com/NVIDIA/GSCNN) | 桌面network/gscnn.py的shape/regular双流及Canny/ASPP融合 | 借形状证据补充语义的动机；不搬CPU Canny与昂贵全分辨率流 |
| [SAHI，ICIP 2022](https://arxiv.org/abs/2202.06934)，[代码](https://github.com/Small-Object-Detection/SAHI) | 切片推理/微调让小物体占据更多输入像素 | 后续输入分辨率上限诊断；本轮不混入，因它增加整图推理成本且需一致实例合并评估 |

也核查桌面Plug-play集合的FreqFusion实现：依赖MMCV CARAFE与grid_sample局部重采样。
目前不采用，避免把新依赖和潜在速度问题带进当前消融。并非断言该论文无效。
NWD/RFLA等标签分配方案可作为后续独立算法实验，本轮不同时换损失/匹配器以免再次无法归因。

## 七、开销与已验证范围

| 模型 | Params/M | GFLOPs@640 | 本地CPU前向/ms | 本地CPU训练步/ms |
|---|---:|---:|---:|---:|
| 80 | 2.323 | 10.097 | 97.2 | 435.9 |
| 81 | 2.332 | 10.544 | 111.2 | 505.3 |
| 82 | 2.035 | 10.305 | 108.2 | 496.6 |
| 83 | 2.036 | 10.567 | 114.2 | 503.5 |
| 84 | 2.325 | 10.359 | 104.4 | 459.4 |

有效基准为 `benchmark_cpu_final.json`：CPU FP32，640、batch1、2线程，交错随机顺序，3步预热+15步采样。
训练步含合成12实例、loss/backward/简单AdamW；不是正式参数分组，也不含真实loader/验证/NMS。
GFLOPs为现有THOP估算，部分函数操作未计入。83比80参数少约12.4%，但GFLOPs多4.7%，CPU训练步慢15.5%。
所以它是精度优先的预算重分配假设，**不是已实现提速**。服务器GPU延迟/显存必须实测。
初次 `benchmark_cpu.json` 与smoke重叠且发现stem第一参数影响FLOPs统计，属于已废弃调试记录，不引用。

最终19项V8构建/真实loss/空标签/反传/矩形输入/预训练键/保存重载/融合/奇数原图predict测试通过；
此前与V7/V7R相关回归共61项通过（包含当时18项V8测试，不能重复相加）。
另外6项V8批量队列/回调测试通过，四个新模型的一轮真实train/val/save smoke通过，最终日志见验证目录。
smoke是4图、CPU、256、batch2、workers0、plots=False，不能作为涨点证据。
本机Matplotlib与NumPy版本不兼容，因此正式绘图未在此环境验证；独立PR诊断使用不依赖Matplotlib的SVG。
未擅自升级环境。服务器正式runner会在训练前检查绘图库，失败时明确停止。

## 八、如何运行

上传更新后的整个 `code/ultralytics-main-new`，不要只复制YAML：自定义类、注册、协议和批量入口都必须一致。
打开 `RUN_SAGE_V8.py`，选择服务器训练环境，核对顶部 `DATA`、`DEVICE`、`PROJECT`，点击VS Code右上角三角形。
默认300轮、5个模型、seed42，当前Python前台串行；Ctrl+C停止整个队列，不会启动下一个。
不想先跑全部可把 `SUITE="priority"`，只跑80/82/83，但完整消融还需81/84。
`DRY_RUN=False` 才训练；True只构建后退出。

```bash
cd /data/sxq/code/ultralytics-main-new
python RUN_SAGE_V8.py
```

固定超参来自 `protocols/citrus_paper1_formal_v2_ram.yaml`：cache=True、AMP=False、AdamW、lr0=.001、
lrf=.01、batch16、imgsz640、workers4、mask_ratio4、dropout0、相同数据和预训练权重。
其余增广/损失权重也不变；没有因新架构悄悄换AMP、学习率或目标过滤。
RAM不足时底层可能拒绝缓存，应看启动日志；True并不保证磁盘/CPU瓶颈全部消失。

单模型仍支持官方API：

```python
from ultralytics import YOLO
from citrus_protocol import fixed_train_args, load_protocol

args = fixed_train_args()
args.update(load_protocol()["fixed_validation"])
model = YOLO("0_orange_yaml/SAGE_V8_series/SAGE83_phase_scale.yaml").load("yolo11n-seg.pt")
model.train(data="/data/sxq/datasets/orange_yolo/data.yaml", epochs=300, device=1,
            seed=42, project="/data/sxq/results/SAGE/V8_SINGLE_NEW", name="SAGE83_seed42", **args)
```

以上单模型例子只有官方best.pt；若要与本报告一致的mask-only checkpoint，优先用RUN文件的 `ONLY="SAGE83_phase_scale"`。
模型82/83的检测步长为4/8/16，但输入仍必须按32对齐，因为C5保留；注册代码已分别处理输入padding与候选步长。

已完成项目可跳过；中断项目不会覆盖或假装完成。若要跳过半途模型，设ONLY为剩余模型，并使用新的PROJECT。
源代码/协议变化也要新PROJECT，不混到旧实验目录。

## 九、什么结果才算成功

先看82对80、83对82、84对80、81对80；同时看严格Mask AP、AP50、tiny召回、低对比分层、split/merge和GPU速度。
若只是P2多框而置信质量仍差，停止继续堆P2，再做目标尺度/评分监督的独立算法实验。
若83没有超过82且84没有超过80，亮度旁路不保留；不能因为它“有论文思路”就坚持。
验证集筛选后，基线与最终方案用42/43/44三seed，并在未参与选择的测试集报告均值±标准差。
当前不能宣称涨10点、最优模型已找到，或PR真实漏检问题已被解决。
