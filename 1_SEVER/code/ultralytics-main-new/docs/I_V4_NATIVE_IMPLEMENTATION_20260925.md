# I_V4 独立实例分割分支（I48–I53）

## 定位

本分支仍放在 `0_orange_yaml/I_V4_series/`，但不是 I40–I47 的另一个 YOLO 注意力组合。它只复用 Ultralytics 的数据读取、优化器、checkpoint 和评估入口；神经网络本体、对象匹配和损失独立实现。没有 YOLO C3k2/CSP 主干、PAN 颈部、Detect/Segment 头、TAL 分配、DFL 或框回归。框只在推理时从最终掩膜导出；为了复用现有验证器，每个对象掩膜被包装成一张独立 prototype，这个**接口适配**不是模型里的 YOLO 原型头。当前通用验证/预测器仍调用 NMS 去重，因此不能宣称“完全 NMS-free”。

YAML 中的 `nn.Identity` 只是适配 Ultralytics 要求非空 backbone 列表；真正完整的主干、场景分支与实例解码都封装在紧接着的 `CitrusNativeSegment`，不是空主干。

## 结构与实验问题

```text
RGB → 新建 s2/s4/s8/s16 深度残差编码器
       │                ├→ s4 场景前景/可见实例边界
       │                └→ s4 候选证据 → 64 个图像驱动对象槽＋位置编码
       └→ s2 细节          + 16 个可学习后备对象槽
                                  ↓
                    对象读取低分辨率语义上下文
                                  ↓
                   可选：对象→场景有界回写→对象重读
                                  ↓
                  对象×s4 像素嵌入（含位置编码）→ 独立可见实例掩膜
                                  ↓
                 存在分数、掩膜、从掩膜导出框/果实数量
```

| 组别 | 增量假设 | 输出槽位 |
|---|---|---:|
| I48 | 独立 learned-query 分割基架，架构对照 | 64 |
| I49 | s4 可见候选证据选点，不靠全局 learned-query 搜索 | 64 |
| I50 | 前景并集与可见实例交界监督，复杂叶片作为未标注负区域 | 64 |
| I51 | 16 个 learned fallback 槽，允许找回候选漏掉的果实 | 80 |
| I52 | 对象读取→有界写回→重读，初始错误不硬删除像素 | 80 |
| I53 | I52 加灰度 Sobel 细节旁路，测试颜色无关的边缘线索 | 80 |

每张图历史标注最多 46 个果实，因此 64/80 槽位有容量余量；其他数据集若实例数超过槽位，损失会显式报错。Hungarian 一对一匹配不把 80 个槽位都当果实。图像驱动查询与像素掩膜均加入明确的 Fourier 位置编码，避免相似外观的多个果实只因特征相近而难以分离。训练目标为存在/无对象、可见掩膜 BCE+Dice、场景并集/边界、每实例至少一个候选证据点与低权重场景一致性。对小掩膜给予有上限的权重，匹配 GT 下采样采用 max-pooling 保留极小目标。没有叶片类别标签，所以不声称直接监督了叶片分类。没有 amodal 推断或果实圆形补全。

本结构借鉴 [SparseInst](https://openaccess.thecvf.com/content/CVPR2022/html/Cheng_Sparse_Instance_Activation_for_Real-Time_Instance_Segmentation_CVPR_2022_paper.html) / [FastInst](https://openaccess.thecvf.com/content/CVPR2023/html/He_FastInst_A_Simple_Query-Based_Model_for_Real-Time_Instance_Segmentation_CVPR_2023_paper.html) 的对象集合和候选思想，但把候选放在 s4、保留 s2 细节；借鉴[场景/对象双向组织综述](https://www.nature.com/articles/s44159-023-00254-0)的启发，但不声称是神经科学模型。Sobel 是经典梯度算子，不属于原创模块。I53 是否比 I52 更好只能由实验判断。

## 运行与固定设置

上传整个 `ultralytics-main-new` 后，编辑根目录 `RUN_CITRUS_I_V4_NATIVE.py` 的 `DATA`、`DEVICE`、`PROJECT`，在 VS Code 点击“运行 Python 文件”即可按 I48→I53 顺序前台训练，Ctrl+C 会停止队列。`DRY_RUN=True` 仅构建六个模型。`ONLY` 可选单个或若干组。脚本不启用 GPU 占用守卫或数据指纹闸门，也不会后台启动。结果已有 `native_completed.json` 时跳过；有部分目录但未完成时停止，避免覆盖。仅作本地测试时可设置 `SOURCE_BALANCED=False`，正式比较需保持 `True`。

使用清洗分组数据的显式 `data.yaml` 路径；正式入口沿用先前 I_V4 的原图/粗切片/细切片源均衡输入、AdamW `lr0=0.001`、`imgsz=640`、`batch=16`、`mask_ratio=2`、`copy_paste=0.3`、`amp=False` 等超参数。不同之处：**I48–I53 从头训练**，不能直接加载 YOLO11n-seg 的权重，因此与预训练 I40–I47 比较时存在初始化混杂。想归因结构，必须补同超参/输入/种子的 YOLO11n-seg scratch 对照，或者为独立主干另建经验证的同等级预训练。建议先 50 epoch 筛选，观察收敛与单卡速度，再对入围组做 300 epoch 和 3 seed。

默认 `EPOCHS=50`；若完成筛选后改成 300，**同时换新的 `PROJECT` 目录**，不要覆盖 50 轮结果。现有 1 epoch 烟测使用 128 输入、4 张训练/2 张验证的隔离副本，仅验证程序链路；其 0 AP 不代表模型最终性能。

## 已验证与未验证

- 六组官方 YAML 均可构建、训练前向/反向、输出验证器兼容的预测；空图损失也可反向。
- I53 使用源均衡切片、定制训练器与验证器完成隔离副本 1 epoch；best.pt 可载入并验证。
- I52：231,894 参数；本机 `torch.profiler` 计数约 3.52 GFLOPs@640，CPU 单次前向约 80–100 ms（单次测量波动较大）。部分算子可能不计入 FLOPs；CPU 速度不能代表服务器 GPU。`scripts/profile_citrus_i_v4_native.py` 可同设备重测。
- **尚未**在正式清洗数据上跑 50/300 epoch，没有 mAP、Recall、PR 尾部或论文增益结论。高召回端 precision 下降可能是低分数背景误报，也可能只是 PR 绘图补零；必须保存实测 PR 点、最大可达 R、固定 P≥0.90 的 R 和分尺度/混淆 FP，而非仅看曲线终点。
- 当前掩膜为 s4 全图解码，s2 只作细节旁路；PointRend 式稀疏局部精修、质量分数校准均**尚未实现**。若小目标 mask AP 仍低，这些才是后续可检验分支，不能在当前结果中宣称已完成。

## 本地数据副作用

第一次尝试在本地正式数据目录直接烟测时，Ultralytics 输出了历史 JPEG“restored and saved”提示。核对源码与文件时间后确认：提示是旧 `val/labels.cache` 中保存的扫描消息再次打印，不能据此认定这次改写了源图；源图修改时间未变化。这次只在正式数据的 `train/` 下新增一个 `labels.cache`。随后测试改用 `_work/citrus_native_smoke_dataset` 隔离副本。正式实验仍须记录服务器端划分成员及版本；不要把本地烟测的 cache、checkpoint 或 AP=0 当作正式实验。
