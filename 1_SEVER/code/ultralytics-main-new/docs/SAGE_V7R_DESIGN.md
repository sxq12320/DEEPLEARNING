# V7R：细节直达与任务选择性语义融合

2026-09-07运行更新：按用户要求，所有活动入口默认 `cache=True`（RAM），固定协议切换为v2；
见 [RAM缓存说明](CACHE_RAM_20260907.md)。下述2026-09-06结果仍属于旧设置。

2026-09-06。V7 的追加重构实验，不覆盖 SAGE70–74 的 YAML 或已完成结果。
状态：已实现、待正式精度验证；不宣称最优架构、首创机制或保证涨点。

## 一、为什么这次改变信息去向，而不是再换一个大模块

依据既有全量结果盘点和关键源码复核，详见
[V6/V7 证据](../reports/sage_v7_20260906/RESULTS_AND_DECISIONS.md) 与
[历史对应索引](../reports/sage_v7_20260906/history_mapping/)。本轮复核这些证据和当前 V4R/V5/V7 实现，
不是声称重新逐行读完两个仓库的所有文件。没有新增正式 V7 训练结果可供选优。

- 同协议 V6 完成组，严格 Mask AP50–95：60=67.503%、61=65.711%、63=64.331%、64=63.766%。
  62 在此前审计时只有 118 轮，不能作为完成的 300 轮结果比较。
- 60 的本地固定阈值诊断：153 个极小实例中，85 个有 IoU≥0.5 原始候选但得分低，
  42 个缺少合格原始框，3 个匹配框的同一实例掩膜失败，23 个成功。
  这是 conf=0.25 的分阶段诊断，不是 AP，也没有证明训练分配、颜色或下采样是唯一原因。
- 60 本地严格 AP 与服务器 CSV 仍有评估差异；本轮不替换服务器指标、不混排不同条件的历史成绩。
- V5 延后 Proto 上采样的两组独立比较均损害严格 AP，因此保留 stride4 原型空间处理。

旧 72 的 refiner 在 stride4 执行 `D = S + sigmoid(gate)*(L-S)`：L 是浅层细节，S 是上采样语义。
同一 D 随后用于 P2 分类、框、掩膜系数、Proto 细节补偿以及 P3 relay。
**这不是已证实的代码错误，但存在可检验的表示冲突：语义融合可能有助识别，却不一定有助极小目标定位。**
本轮把“是否融合”和“融合给谁”拆开，不再把所有目标交给同一混合张量。

## 二、这篇三分支论文究竟能借鉴什么

Yawen Bai 等，*A Triple-Branch Architecture With Multiscale Attention for Spatiotemporal Remote Sensing Fusion*，
IEEE TGRS，2026，DOI [10.1109/TGRS.2026.3660753](https://doi.org/10.1109/TGRS.2026.3660753)。
依据用户提供的 17 页全文；页码指 PDF 页码。

| 论文事实与定位 | 迁移判断 |
| --- | --- |
| 第3–5页：目标时刻粗图 + 前后两个时刻粗/细图，共五幅图；两个空间流与一个时间差分流 | 本任务只有单幅 RGB，不能照搬三路输入，更不能把普通三分支命名为时序模型 |
| 第5–7页：STAM 按**通道**分四组，一组保留原分辨率，其余按2/4/8缩放；DWConv、SE、聚合与调制 | 借鉴多尺度上下文；不是把图像分四个区域，也不是检测小目标的直接证据 |
| 第5页：四个64通道 ASPP 分支，膨胀率6/12/18/24；再与 STAM 拼成320通道 | 不复制到高分辨率柑橘预测头；很大感受野/稀疏采样未必符合极小果需求 |
| 第8页：按时间差异进行局部加权融合 | 借鉴选择性融合原则；本轮门控由任务学习，不把无时序特征差异冒充时间置信度 |
| 第9、16页：MSE、MS-SSIM、相邻像素梯度 MSE | 是影像重建损失，不直接替换实例分类/定位/掩膜目标；叶片边缘增强也可能增加误报 |

第14页表 IV：CIA 中，TDNet 保留而 STAM/SE 都移除时 PSNR=32.73644；加入 STAM、SE 仍关闭时
为33.80351；完整模型为34.18191 dB。注意第一个对照同时去掉 STAM 和 SE，不能称为
“仅去除 STAM”的独立实验。这里支持的是该任务的影像重建效果，不是柑橘 AP 或极小目标召回。
本文没有实例级 AP_small、漏检或粘连分离实验。当前未找到可确认的该论文作者官方代码仓库，
因此本实现不标注为其代码复现。

值得注意：论文参考文献 [39] 指向 ICCV 2023 SAFM。已核对模块库与 SAFMN 作者实现，
二者都有通道分组、分辨率缩放、DWConv、拼接聚合、空间调制这一操作链。
这条已有方法脉络必须如实引用，不能把“多尺度分组”本身写成我们的首创。

## 三、两个桌面代码目录的具体取舍

| 实际核读代码 | 保留的思想 / 不采用的部分 |
| --- | --- |
| `Plug-play-modules-main/3. Block（功能模块）/(ICCV 2023) SAFM.py`；作者 `basicsr/archs/safmn_arch.py` | 分组在不同尺度计算上下文；不搬整套超分重建网络 |
| `github/PIDNet/models/model_utils.py` 的 PagFM、Light_Bag、Bag | 细节与上下文应选择性交互，而非无条件平均；不复制边缘监督分支或声称实现 PID 控制 |
| `github/FreqFusion/FreqFusion.py` 的高低通融合与采样 | 低分辨率语义可能污染细边界这一问题值得重视；暂不使用 CARAFE、unfold、grid_sample 组合 |
| `github/PKINet/mmrotate/models/backbones/pkinet.py` 的 InceptionBottleneck | 认可密集局部核与上下文互补；保留旧74作为可选对照，不把五大核+CAA再堆到新路径 |
| 模块库 `(arXiv 2024) MSAA.py` | 三路拼接加3/5/7卷积及通道/空间注意力不是当前优先选项；其文件注明预印本，不能笼统声称全库均为顶会 |

FreqFusion 的未装 mmcv 时后备实现含 `unfold` 及空间扩展，存在中间激活/内存访问成本；
这只是具体代码风险，不是已经在用户 GPU 上证明它慢。本轮不新增第三方运行依赖、不安装 Mamba。

已核验的原始来源：

- [SAFM，ICCV 2023](https://openaccess.thecvf.com/content/ICCV2023/html/Sun_Spatially-Adaptive_Feature_Modulation_for_Efficient_Image_Super-Resolution_ICCV_2023_paper.html)，[作者代码](https://github.com/sunny2109/SAFMN/blob/main/basicsr/archs/safmn_arch.py)。
- [PIDNet 作者仓库](https://github.com/XuJiacong/PIDNet)，[作者论文](https://arxiv.org/abs/2206.02066)。
- [FreqFusion 作者仓库与论文入口](https://github.com/Linwei-Chen/FreqFusion)。
- [PKINet，CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)。

## 四、实现结构与公式

保留60/72的 YOLO11 C3k2 主干、非对称颈部、stride4 Proto、P3 relay、32通道窄预测头和 P2/P3/P4/P5 候选。
这次**没有更换主干**：已有 V6 结果不支持再同时更换主干和融合方式。实质改动是细节分支到不同预测任务的连线。

1. 从 C2 提取16通道细节 L，取消旧语义凸组合。这是高分辨率特征旁路，**不是原始像素无损保留**。
2. L 直接供 P2 窄 stem、Proto 细节补偿和既有 P3 relay；P2 的框与掩膜系数保留这条细节路径。
3. 另从 relay 之前的 P3 提取32通道上下文 C。分为四个8通道组，在 P3 内按1/2/4/8缩放，
   各用3×3 DWConv，再恢复 P3 尺寸、1×1聚合。多尺度处理不作用于 L。
4. 将上下文恢复到 P2 后，通过门控残差送到指定任务。

设 X 为 P2 窄 stem 输出，U 为上下文上采样结果：

`C' = C * (1 + 0.5*tanh(A(group_multiscale(C))))`

`g = sigmoid(Conv1x1([X, U, abs(X-U)]))`

`X_cls = X + tanh(alpha) * g * U`，alpha 初始化0.1；门参数零初始化，所以初始 g=0.5。

主候选77：P2分类用 X_cls；P2框与掩膜系数用 X。76则三者全部用 X_cls。
其余尺度和 Proto 在76/77中完全一致。没有推理阈值补丁、分数直接加常数或 GT 参与推理。
对上下文分组采用平均池化而非 SAFM 的最大池化；调制带恒等通路且有界，不用原文未约束乘积，
也不增加 SE/FFN/ASPP。这些是任务适配，不等价原 STAM/SAFM。

控制理论联系只限于“保留直通、限制校正增益”的工程类比：abs(X-U)是学习空间内的差异，
不是已校准的不确定度；本结构没有积分器、误差时间导数或迭代反馈，没有稳定性证明。
类别专用语义路径仍会通过训练梯度影响共享主干，不能声称彻底消除任务冲突。

## 五、实验矩阵：默认只跑六个，不把旧系列全部重跑

所有 YAML 位于 `0_orange_yaml/SAGE_V7_series/`。

| 编号 | 文件名 | 回答的问题 |
| --- | --- | --- |
| 70 | SAGE70_relay_control | 旧60结构锚点，未改变 |
| 72 | SAGE72_p2_candidates | 旧V7高分辨率候选锚点，未改变 |
| 75 | SAGE75_detail_bypass | 相对72：取消语义混合后，直接细节是否更适合候选？这是refiner整套变化，不是纯零参数对照 |
| 76 | SAGE76_shared_context | 相对75：低分辨率多尺度语义经残差进入P2三项预测，是否有用？ |
| 77 | SAGE77_task_routed_context | 相对76：相同参数，仅将语义注入限制到P2分类，几何是否更可靠？主研究候选，不是已知最佳 |
| 78 | SAGE78_single_scale_route | 相对77：四组都在P3原尺度，参数形状/数量相同；隔离多尺度缩放贡献 |

77若不胜76，则“按任务路由优于共享融合”不成立；77若不胜78，则不要坚持多尺度。
75若胜复杂模型，则优先选择75。若新方案不胜70/72，则拒绝这次结构假设，而不是靠额外改AMP制造提升。
77的直接几何支路也可能缺少必要语义而降低定位质量，76就是为检验这一风险设置的对照。

## 六、训练：VS Code 右上角运行

上传更新后的整个 `code/ultralytics-main-new`，不是只上传一个 RUN 文件。
打开 **RUN_SAGE_V7R.py**，按服务器实际情况编辑：

```python
DATA = "/data/sxq/datasets/orange_yolo/data.yaml"
DEVICE = "1"
SUITE = "refusion"  # 70、72、75–78；refusion_new仅75–78
EPOCHS = 300
DRY_RUN = False
```

选择原训练环境，点击运行即可。当前进程前台、模型串行；Ctrl+C中断整队，不继续下一模型。
终端等价命令：`python RUN_SAGE_V7R.py`。默认新输出目录：
`/data/sxq/results/SAGE/CITRUS_SAGE_V7R_REFUSION_300EP`。
不要沿用已有V7输出目录：源码签名变化会被既有保护拒绝。旧 RUN_SAGE_V7.py 的 structure 队列仍为70–74。
批量顺序在seed内打乱后串行执行，不是并行训练。前台/后台本身不会改变GPU吞吐或禁止其他用户占卡。

超参现来自 [固定协议v2](../protocols/citrus_paper1_formal_v2_ram.yaml)：cache=True，AMP=False，batch16，640，workers4，
AdamW，lr0=0.001，lrf=0.01，weight_decay=0.0005，dropout=0，seed42，同一 yolo11n-seg.pt，
其余增强、mask_ratio、损失权重不变。本轮不新增损失。模型配置中也不写数据路径。

单模型仍支持标准 API（当前修改版 Ultralytics，不能用服务器另一份未经注册的 pip 包）：

```python
from ultralytics import YOLO
from citrus_protocol import fixed_train_args, load_protocol

if __name__ == "__main__":
    model = YOLO("0_orange_yaml/SAGE_V7_series/SAGE77_task_routed_context.yaml").load("yolo11n-seg.pt")
    args = fixed_train_args()
    args.update(load_protocol()["fixed_validation"])
    model.train(data="/data/sxq/datasets/orange_yolo/data.yaml", epochs=300, device=1, seed=42,
                project="/data/sxq/results/SAGE/V7R_SINGLE_NEW", name="SAGE77_seed42", **args)
```

单模型示例使用官方train入口；如需批量入口的严格Mask最佳检查点、初始化报告和防覆盖检查，优先使用RUN。

## 七、验收边界

本地测得：75为1.796M/8.697GFLOPs；76、77为1.800M/8.744GFLOPs；78为1.800M/8.746GFLOPs，均以nc=1、640计。
77相对72只增加2,512参数；同次CPU交错测速的训练步中位时间约增加1.4%，未出现数量级变慢。
相对70虽降低参数与GFLOPs，训练步仍约慢4.5%；这些不是用户GPU实测，不承诺提速。

验证记录见 [V7R 验证报告](../reports/sage_v7r_20260906/VERIFICATION.md)。
无用户GPU上的实测速度、没有正式V7R精度结果。GFLOPs并不覆盖全部函数式算子开销。
P2仍有34000候选，相比三尺度8400有额外匹配/分类/NMS成本，本轮没有消除这项成本。
正式GPU实验先观察前5–10轮稳态每轮时间；若比同卡72明显变慢，应检查训练/验证/数据加载各阶段，
不能只凭参数少就继续长跑。

效果判定优先：Mask AP50–95、AP50、极小实例Recall@固定Precision、低/高阈值召回、误报/图，
再看深凹及接触果的split/merge和实测延迟。最终方法与基线做三个seed。
PR图超出实测召回范围的补零尾段不能通过美化绘图来“解决”；这里不改变指标或PR绘制方式。
