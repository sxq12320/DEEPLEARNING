# 主题7：Mamba/SSM 与线性注意力在视觉检测中的应用（16 篇，均经 arXiv API / Crossref 核验）

课题背景：YOLO11n-seg 柑橘幼果实例分割，~2.8M 参数 nano 约束；需在 P5（~20x20 特征图）低代价全局建模，为远处小暗果提供全图上下文。三个评估维度：①20x20 小图上的实际延迟；②纯 PyTorch（无 CUDA 扩展）可实现性 → Ultralytics fork 集成；③ONNX 端侧导出。

---

## A. 视觉 Mamba 骨干

### 1. VMamba: Visual State Space Model
- 第一作者：Yue Liu | 2024 | NeurIPS 2024 | arXiv:2401.10166
- 核心机制：提出 SS2D（2D Selective Scan），将图像沿四个方向展开为序列做 selective scan 再融合，使 S6 具备 2D 感受野；线性复杂度实现全局有效感受野。
- 集成判断：官方实现依赖 selective_scan CUDA kernel，纯 PyTorch 退化实现慢 5-10 倍；且 selective scan 的数据依赖递归在 ONNX 导出时需展开为循环，端侧部署阻力大。不建议直接塞进 nano 模型。

### 2. Vision Mamba (Vim): Efficient Visual Representation Learning with Bidirectional State Space Model
- 第一作者：Lianghui Zhu | 2024 | ICML 2024 | arXiv:2401.09417
- 核心机制：ViT 式 plain 结构 + 双向（前向/后向）Mamba 扫描，宣称高分辨率下比 DeiT 省 86.8% 显存；证明无注意力也能做全局建模。
- 集成判断：优势体现在长序列（高分辨率）场景；P5 仅 400 token，序列太短，Mamba 相对注意力的复杂度优势完全无法兑现，反而引入递归延迟。

### 3. EfficientVMamba: Atrous Selective Scan for Light Weight Visual Mamba
- 第一作者：Xiaohuan Pei | 2024 | AAAI 2025 | arXiv:2403.09977
- 核心机制：空洞采样跳跃式选择扫描（ES2D）降低扫描 token 数；关键设计：浅层（大分辨率）用 SSM、深层用卷积——与"深层小图上 SSM 不划算"的直觉一致。
- 集成判断：其"深层改用卷积"的结论反向说明：在 P5 这种小特征图上作者自己都放弃了 SSM。对本课题是重要反证文献。

### 4. MobileMamba: Lightweight Multi-Receptive Visual Mamba Network
- 第一作者：Haoyang He | 2024 | CVPR 2025 | arXiv:2411.15941
- 核心机制：指出现有 Mamba 模型 GPU 实际吞吐远低于同 FLOPs 的 CNN/ViT；提出多感受野三分支（局部卷积+全局 WTE-Mamba 小波增强），只对部分通道做 Mamba，速度提升约 21x（vs LocalVim）。
- 集成判断：移动端 Mamba 的最强基线，但仍依赖 selective scan 算子；其"只切一小部分通道给全局分支"的思路可借鉴到任何 P5 全局模块设计。

### 5. PlainMamba: Improving Non-Hierarchical Mamba in Visual Recognition
- 第一作者：Chenhongyi Yang | 2024 | BMVC 2024 | arXiv:2403.17695
- 核心机制：非层级 plain 结构，连续 2D 扫描（蛇形路径保持空间邻接）+ 方向感知更新，去掉 CLS token，易于多尺度特征提取。
- 集成判断：结构简单利于改造，但检测实验用的是完整骨干替换而非即插模块；nano 场景下同样受 selective scan 算子拖累。

### 6. MambaVision: A Hybrid Mamba-Transformer Vision Backbone
- 第一作者：Ali Hatamizadeh | 2024 | CVPR 2025 | arXiv:2407.08083
- 核心机制：重设计 Mamba block（去因果卷积、加对称非 SSM 分支），并系统消融发现：**在最后几个 stage（小分辨率）放自注意力而非 Mamba 效果最好**——因为小图上注意力已不贵且全局捕捉更直接。
- 集成判断：对本课题的关键结论性证据：P5 级小特征图上，注意力（含线性变体）优于 SSM。NVIDIA 官方实现，工程质量高。

## B. Mamba × YOLO 检测器

### 7. Mamba YOLO: A Simple Baseline for Object Detection with State Space Model
- 第一作者：Zeyu Wang | 2024 | AAAI 2025 | arXiv:2406.05835
- 核心机制：ODMamba 骨干 + RG Block（残差门控 MLP 补偿 SSM 局部建模不足），SS2D 做全局依赖；Mamba-YOLO-T 约 5.8M 参数，MS COCO 上超 YOLOv8n 约 +3.4 AP。
- 集成判断：证明 SSM 进 YOLO 可行，但 T 版参数已是 YOLO11n 两倍且依赖 selective_scan CUDA 包（Ultralytics 官方 fork 无法直接 pip 装）；ONNX 导出社区 issue 多、无官方支持。

### 8. IM-YOLO: an improved Mamba-YOLO-based model for small object detection in aerial images
- 第一作者：Long | 2025 | Journal of Electronic Imaging | DOI: 10.1117/1.JEI.34.5.053029
- 核心机制：在 Mamba-YOLO 上针对航拍小目标改进（多尺度融合+扫描策略调整），验证 SSM 全局上下文对小目标检出的增益。
- 集成判断：佐证"全局上下文帮助小目标"这一动机本身成立，但其代价结构（继承 Mamba-YOLO 的 CUDA 依赖）在 nano 端侧不成立。

### 9. MiM-ISTD: Mamba-in-Mamba for Efficient Infrared Small Target Detection
- 第一作者：Tianxiang Chen | 2024 | IEEE TGRS | arXiv:2403.02148 / DOI: 10.1109/TGRS.2024.3485721
- 核心机制：外层 Mamba 处理 patch 级"visual sentence"、内层 Mamba 处理 patch 内像素，层级式捕捉全局+局部，用于红外弱小目标（与暗小幼果场景形态相似）。
- 集成判断：任务形态最接近"暗背景小目标需全局佐证"，但它是专用分割网络而非即插模块；可借鉴其"小目标依赖全局统计对比"的论证写入引言。

## C. 线性注意力

### 10. FLatten Transformer: Vision Transformer using Focused Linear Attention
- 第一作者：Dongchen Han | 2023 | ICCV 2023 | arXiv:2308.00442
- 核心机制：聚焦函数（focused function）锐化线性注意力过度平滑的权重分布 + DWConv 秩恢复，弥补线性注意力表达力短板，O(N) 复杂度。
- 集成判断：纯矩阵乘 + DWConv，纯 PyTorch 十几行实现、ONNX 友好；是 P5 全局建模的首选机制族。已被多篇农业 YOLO 改进论文当即插模块使用。

### 11. Agent Attention: On the Integration of Softmax and Linear Attention
- 第一作者：Dongchen Han | 2023 | ECCV 2024 | arXiv:2312.08874
- 核心机制：引入少量 agent token 作中介：agent 先聚合全局信息（softmax），再广播回全部 query，等价于一种广义线性注意力，兼得表达力与线性复杂度。
- 集成判断：pooling + 两次小 softmax 注意力，全部为标准算子，ONNX 导出零障碍；在 20x20 图上 agent 数可设 9-16，计算量可忽略。与 FLatten 并列首选。

### 12. PolaFormer: Polarity-aware Linear Attention for Vision Transformers
- 第一作者：Weikang Meng | 2025 | ICLR 2025 | arXiv:2501.15061
- 核心机制：指出既有线性注意力丢失 q-k 负值交互；按极性分解（同号/异号分量分开计算）+ 可学习幂函数降低注意力熵，恢复表达力。
- 集成判断：机制新（2025），同样纯矩阵乘可 ONNX；作为 FLatten/Agent 的更新替代做消融对比，是小论文"新颖性"好素材。

### 13. EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention
- 第一作者：Xinyu Liu | 2023 | CVPR 2023 | arXiv:2305.07027
- 核心机制：分析出 ViT 速度瓶颈在访存而非 FLOPs；级联分组注意力（每组喂不同特征切片、逐组级联）减少注意力头冗余、省访存。
- 集成判断："访存主导延迟"的分析框架直接适用于 nano 模型选型——20x20 图上普通 softmax 注意力访存本就不大，这弱化了 Mamba 的必要性。

### 14. SwiftFormer: Efficient Additive Attention for Transformer-based Real-time Mobile Vision Applications
- 第一作者：Abdelrahman Shaker | 2023 | ICCV 2023 | arXiv:2303.15446
- 核心机制：加性注意力——去掉 k-v 交互，仅用 query 与全局 query 向量的逐元素运算替代点积，复杂度 O(N·d)，专为移动端实时设计。
- 集成判断：三个线性方案中算量最低、结构最简（几个 Linear + 逐元素乘），iPhone 上已验证实时；ONNX/NCNN 均友好，nano P5 兜底方案。

## D. 机制对比与农业应用

### 15. Demystify Mamba in Vision: A Linear Attention Perspective (MLLA)
- 第一作者：Dongchen Han | 2024 | NeurIPS 2024 | arXiv:2405.16605
- 核心机制：理论证明 Mamba 的 selective SSM 数学上是带遗忘门+特殊块设计的线性注意力变体；把 Mamba 的两个有效成分（遗忘门→用位置编码替代、块设计）移植回线性注意力得 MLLA，**精度和推理速度双超视觉 Mamba**（可并行、无递归）。
- 集成判断：本主题的裁决性文献：既然 Mamba≈线性注意力+遗忘门，而遗忘门在非自回归视觉任务可用位置编码替代，则 nano 检测器直接用线性注意力即可拿到 Mamba 的收益而不付递归/CUDA 代价。

### 16. DDS-Mamba-YOLO: Improved Adaptive State-Space Modeling for Tomato Fruit and Leaf Disease Detection
- 第一作者：Zhu | 2026 | Smart Agricultural Technology | DOI: 10.1016/j.atech.2026.102424
- 核心机制：在番茄果实/叶片病害检测中引入自适应状态空间模块改进 Mamba-YOLO，利用全局依赖处理遮挡与密集小病斑。
- 集成判断：说明"SSM 进农业 YOLO"已有人做（2026 已见刊），单纯 Mamba×农业检测的组合新颖性窗口正在关闭；差异化应落在 nano 约束 + 部署可行性 + 分割任务上。
- （同类可引：MCS-YOLO 轻量除草检测, Yan, Agriculture 2026, DOI: 10.3390/agriculture16050539）

---

## 结论（对应三个评估点）

1. **20x20 小特征图上的延迟**：P5 仅 400 token，注意力的 O(N²) 在此规模下绝对量极小（400²·d 级矩阵乘，GPU/NPU 一次 GEMM）；Mamba 的 selective scan 是数据依赖递归，无法整图并行，在短序列上实际延迟反而更高（MobileMamba、MLLA 均给出实测）。MambaVision 消融直接表明深层小图放注意力优于 SSM。**线性注意力（或小图上直接 softmax 注意力）胜。**
2. **纯 PyTorch / Ultralytics fork 可行性**：FLatten/Agent/PolaFormer/SwiftFormer 全部由标准 Linear/DWConv/矩阵乘构成，可作为单文件模块注册进 ultralytics.nn.modules；Mamba 系依赖 mamba_ssm / selective_scan CUDA 编译（Windows 编译尤其痛苦），纯 PyTorch 等价实现慢数倍。**线性注意力胜。**
3. **ONNX 端侧导出**：线性注意力全为 ONNX 原生算子；selective scan 需自定义算子或 Loop 展开，Mamba-YOLO 的 ONNX 导出至今无官方支持。**线性注意力胜。**

选型建议：P5 全局建模模块首选 focused/agent 类线性注意力（含 2025 的 PolaFormer 作新颖性增量），以 MLLA 的"Mamba≈线性注意力"理论作机制论证；Mamba 系文献用作动机与对比基线，不进最终模型。
