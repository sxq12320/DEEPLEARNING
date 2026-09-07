# SAGE V7：面向极小果候选的计算预算重分配

更新入口：2026-09-06追加了[细节直达与任务语义路由V7R](SAGE_V7R_DESIGN.md)，新增75–78；本文保留为原70–74设计记录。

2026-09-06。状态：可训练实验系列，不是已证明涨点的新方法。用户给定活动工程为 `E:/mastercode/1_SEVER/code/ultralytics-main-new`，不与根目录旧工程混合。

## 1. V6结果迫使我们撤回什么

详见 `../reports/sage_v7_20260906/RESULTS_AND_DECISIONS.md`。本地不是五组全完成：62仅118轮且没有完成标记，另四组300轮。60/61/63/64严格Mask AP分别为67.503/65.711/64.331/63.766%。在已有结果下，61整套单程融合及63/64轻量持续细节组合没有达到目标。62不能用118轮直接判输300轮，但前118轮也弱于其他组，且初始化覆盖不同。

因此，不把替换C3k2本身当作创新价值，不把“保持高分辨率”视为必然涨点；恢复60的主干与非对称颈部作为可靠锚点。新方法不是再次全换主干，而是改变候选网格、预测任务的信息共享和计算预算。这是依据负结果调整研究方向。

历史已重读168份CSV（154份不同内容），配置/对应索引见 `../reports/sage_v7_20260906/history_mapping/`。历史G10的旧配方正信号、额外训练配方负收益、LSKA位置依赖、V4R语义细节弱正收益，以及V5延后上采样损害严格AP的结论，见既有重审报告。不同数据/AMP/初始化记录不混排。全量覆盖指CSV和YAML盘点、关键实现复核，不是逐行读完全部历史依赖；文件名匹配也不证明历史源码版本。

## 2. 当前痛点：先让极小果成为可靠候选

SAGE60统一本地验证：153个极小实例，mask Recall@.001=43.14%，Recall@.25=15.03%；较大组617个对应98.70%/96.60%。极小分组按640输入stride4栅格面积<256，不是COCO AP_small。

新诊断在NMS前保存原始框/分数，NMS后执行框的一对一匹配，再检查同一预测ID的掩膜。固定conf=.25的极小分解：

| 去向 | 个数 |
| --- | ---: |
| 存在IoU≥.5原始框，但得分不足 | 85 |
| 原始框没有IoU≥.5候选 | 42 |
| 框匹配成功但同一预测掩膜IoU不足 | 3 |
| 框与掩膜成功 | 23 |

该诊断支持先改候选表示/定位和得分学习，而非继续堆掩膜损失或首先修改NMS。它不是证明所有漏检来自stride8，也没有测完真实训练TAL正样本质量。阈值0.25仅为固定探针，后续需补验证集选择的Recall@Precision=.90/.95。低色差只是外观代理，尚不能声称网络过度依赖颜色。

第二痛点仍是叶枝遮挡深凹可见掩膜、相邻果实例身份冲突。SAGE64虽然极小低阈值召回49.67%，整体严格AP和分裂/合并代理却更差，因此不能只追求宽松召回。保留原生高分辨率Proto，不做凸包填充、强制圆化和全局连通监督。

诊断采用CPU FP32 batch1，与服务器不能混排。尤其60本地严格AP约69.36%而服务器67.503%，差异尚未归因，不能将本地值当成新提升。数据文件名已核对，跨机器图像字节一致性尚未证明。

## 3. 五个模型与可解释的比较

全部YAML在 `0_orange_yaml/SAGE_V7_series/`，全部标准 `YOLO(yaml)` 可构建。

| 模型 | 结构变化 | 参数M | GFLOPs640 | 对照关系 |
| --- | --- | ---: | ---: | --- |
| SAGE70_relay_control | 与V6的60相同，复现锚点 | 2.323 | 10.097 | 旧结构 |
| SAGE71_compact_candidates | 三尺度各自窄空间stem，框/分类/系数独立1×1输出 | 1.784 | 8.040 | 71–70：预测头整套预算变化 |
| SAGE72_p2_candidates | 71基础上，语义细节直接生成stride4候选 | 1.797 | 8.713 | 72–71：加P2直接检测/系数 |
| SAGE73_p2_no_relay | 72基础上，取消细节压回P3的relay | 1.793 | 8.658 | 73–72：已有P2时relay是否仍必要 |
| SAGE74_p2_local_context | 72基础上，仅P2空间stem换局部多核混合 | 1.790 | 8.350 | 74–72：单处模块适配 |

保留YOLO11主干、60的非对称颈部、原型宽度64与stride4空间卷积。71–74把每尺度重复的预测空间处理改为32通道共享任务stem，分类/定位/掩膜系数仍有独立投影。共享不等于任务天然对齐，可能带来任务冲突，71是必要对照。

72–74的P2来自已有语义指导细节估计，不复制一整套高分辨率PAN。细节同时被实例候选、系数和原型使用，不再只在掩膜端补细节。原型仍从P3生成stride4，绝不随新增P2误变成昂贵stride2。输出候选顺序为P2/P3/P4/P5，对应stride4/8/16/32；旧71为8/16/32。

当前TAL按头部stride确定小框扩展，增加P2后其最小/第二stride随之变化。这是新增网格的实现后果，不能宣称只改变像素采样而其他赋值行为绝对不变。分类负样本数也会变化。优化器、损失权重、AMP和增强保持固定，不同时插入NWD/Focal等额外变量。

## 4. 桌面代码与论文的具体取舍

这是针对性检索与代码筛选，不是系统综述。搜索日期2026-09-06，概念为高分辨率小目标候选、低成本预测头、多尺度局部上下文。第三方模块库包括预印本和改写实现，并不意味着所有模块都来自顶会、都可直接复现原论文。

| 来源与已读代码 | 借鉴或暂缓 |
| --- | --- |
| QueryDet：`github/QueryDet-PyTorch/models/querydet/detector.py`、`det_head.py` | 学习把高分辨率计算用于小目标候选；本次不用其Detectron2/spconv稀疏查询实现，不冒充QueryDet复现 |
| TOOD：`github/TOOD/mmdet/models/dense_heads/tood_head.py` | 任务交互与独立预测的分工值得借鉴；新头没有复制其动态任务分解和可变形对齐，不能宣称等价T-Head |
| RTMDet作者实现 `mmdet/models/dense_heads/rtmdet_head.py` | 参考头部计算/共享设计的工程原则；本次不是官方RTMDet头，也不移植其训练配方 |
| PKINet作者 `github/PKINet/mmrotate/models/backbones/pkinet.py` 与模块库 `3. Block（功能模块）/(CVPR 2024) PKIBlock.py` | 74仅保留局部3×3及在其上5×5深度卷积上下文、1×1混合，不复制五大核+CAA+FFN完整块 |
| 模块库RCM/RCA | 行列池化可能抹平稀疏小果；与已有语义门重叠，暂缓。目录会议信息本轮未独立核验 |
| 模块库RFAConv | 核面积倍数通道展开/重排可能加大P2内存访问，暂缓；这是代码开销风险，不是实际GPU测慢结论 |
| 模块库FreqFusion | 滤波/采样思想相关，但CARAFE/重采样依赖与高分辨率开销需独立基准，不与本轮候选变量混入 |
| 历史SegmentP2DetectBoundary | 过去已经有P2头，不能把P2本身宣称新颖；本次复用语义细节、取消重复预测空间塔，且保留原型分支作为区别 |

74的实现也不是完整PKI：它改变了卷积类型和上下文范围，两者尚未拆开消融。仅“把两个核拼起来”不构成已证明论文创新。真正待验证的贡献是针对极小实例失效路径的计算重分配、候选与掩膜共享有效细节，以及成本受控的局部上下文。

核验的主要来源：

- [QueryDet, CVPR 2022](https://openaccess.thecvf.com/content/CVPR2022/html/Yang_QueryDet_Cascaded_Sparse_Query_for_Accelerating_High-Resolution_Small_Object_Detection_CVPR_2022_paper.html)；[作者代码](https://github.com/Small-Object-Detection/QueryDet)。
- [TOOD, ICCV 2021](https://openaccess.thecvf.com/content/ICCV2021/html/Feng_TOOD_Task-Aligned_One-Stage_Object_Detection_ICCV_2021_paper.html)；[作者代码](https://github.com/fcjian/TOOD)。
- [RTMDet](https://arxiv.org/abs/2212.07784)；[作者实现](https://github.com/open-mmlab/mmdetection/blob/main/mmdet/models/dense_heads/rtmdet_head.py)。
- [PKINet, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/html/Cai_Poly_Kernel_Inception_Network_for_Remote_Sensing_Detection_CVPR_2024_paper.html)；[作者代码](https://github.com/PKINet/PKINet)。

不额外加入“PID/闭环”命名。当前只是单幅图像推断与残差路由，没有控制稳定性证明；不能为了三条创新硬凑控制理论。

## 5. 效率、初始化和风险

72相对70约少22.6%参数、13.7%GFLOPs，74约少23.0%参数、17.3%GFLOPs；这是算量，不是速度承诺。P2候选由8400增至34000，TAL、分类loss、NMS和显存访问仍有成本。CPU初次逐模型测速波动较大，因此另做交错顺序微测；最终数值见报告，不据单次顺序声称GPU提速。

所有模型从同一yolo11n-seg.pt开始。71–74保留backbone/neck/proto原键，新的预测器用独立名称避免P2意外加载旧P3同形权重。新头随机初始化、旧头有预训练覆盖，这仍影响71–70解释；每次训练保存真实初始化相等比例。不得只凭“Transferred多少项”声称完全等价初始化。

默认不训练新主干、不改数据、不删小标签、不做数据清洗、不更换AMP。若新P2在验证Precision固定条件下没有召回改善，或降低严格Mask AP，则该假设被否决；不得用AP50小幅上升掩盖形状质量下降。

## 6. 验证与下一步

本地官方API构建、前向、反向、空GT、矩形输入、权重重载、fuse、GFLOPs及批量安全测试已通过。四个新模型各完成4训练图/4验证图fixture的一轮训练，仅为功能验证。没有V7正式精度结果，也未在用户服务器GPU/Torch1.13验证。

完成同协议实验后，对70/72/74优先比较严格Mask AP、AP50、低/高阈值极小召回、Recall@固定Precision、深凹与邻果split/merge，以及真实同卡延迟。最后只给胜出方法和基线做三seed；不把单次最优峰值当作可靠论文结论。
