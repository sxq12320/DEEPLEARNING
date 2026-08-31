# 00. 研究边界与任务规范书 (Research Scope, Specifications & Mathematical Metrics)

**项目名称**：果园自然光照下 RGB 未成熟柑橘轻量级高精度实例分割 (Lightweight High-Precision Instance Segmentation for Immature Citrus in Orchard Environments)  
**课题定位**：硕士学位论文第一阶段核心感知模块（为后续柑橘自主套袋机器人提供高保真果实几何掩膜与着生点 ROI 先验）  
**基准日期**：2026-08-27  

---

## 1. 核心研究任务与应用定位

本研究聚焦于**自然果园非结构化环境下单目 RGB 图像的未成熟柑橘（幼果）实例分割**。

### 1.1 应用场景与上下游对接
- **上游输入**：果园移动机器人搭载的单目 RGB 工业/车载相机采集的自然场景图像（包含直射强光、背光、树冠阴影、风吹动态模糊等复杂干扰）。
- **算法输出**：图像中每个未成熟柑橘个体的精细像素级二值掩膜（Polygon / Binary Mask）、类别置信度（单一前景类 `orange_immature`）以及轴对齐边界框（Bounding Box）。
- **下游协同**：输出的高保真果实实例分割结果直接作为论文第二阶段“柑橘果梗着生点（剪切点/套袋口）微距精确定位”的几何掩膜与区域先验（ROI Prior）。

---

## 2. 严格排除与研究边界 (Strict Research Boundaries)

为保证研究工作的纯粹性、学术深度以及嵌入式硬件落地可行性，严格设定以下研究边界：

| 维度 / 类别 | 允许与要求范围 (Included) | 严格禁止与排除范围 (Excluded) | 设定理由与学术规范 |
| :--- | :--- | :--- | :--- |
| **感知模态** | 单目 RGB 图像 (3 通道) | ❌ 禁止扩展至 RGB-D、LiDAR 点云、多光谱/高光谱、双目立体视差图 | 降低果园机器人硬件成本与标定复杂度，保持论文第一阶段核心纯粹性 |
| **任务类型** | 轴对齐框目标检测 + 2D 像素级实例分割 (Instance Segmentation) | ❌ 禁止扩展至 Amodal 补全、旋转目标检测 (OBB)、机械臂末端控制、抓取姿态估计、3D 关键点 | 严格遵守硕士论文两阶段研究计划，避免任务蔓延导致核心分割退化 |
| **类别空间** | 单一前景类别：未成熟柑橘 (`orange_immature`, ID: 0) | ❌ 禁止混入成熟柑橘、枝叶背景多分类、果园病虫害多任务 | 聚焦幼果期套袋刚需，避免多类别混淆干扰几何与拓扑研究 |
| **算子与架构** | 标准 PyTorch 算子、结构重参数化卷积、标准注意力机制 | ❌ 严禁引入 Mamba (SSM)、Selective Scan、Deformable Conv v3/v4 自定义 CUDA 扩展 | 保障 ONNX/TensorRT 跨平台无缝导出与工业级嵌入式稳定部署 |
| **数据与协议** | 分组去重防泄露划分 (Group-aware Split)，单图 640×640 Letterbox | ❌ 严禁混用旧版连拍泄露划分，严禁跨协议直接数值对比 | 确保实验结论的真实性与可复现性 |

---

## 3. 硬件约束与轻量化预算红线 (Strict Hardware Constraints)

所有模型在标准 $640 \times 640$ 输入尺寸下，必须严格满足以下硬性指标边界：

| 评估指标 | 符号与单位 | 严格约束阈值 | 对标 YOLO11n-seg 基线 | 约束目的与部署场景 |
| :--- | :--- | :--- | :--- | :--- |
| **模型参数量** | $\text{Params}\ (\text{M})$ | $\le \mathbf{2.85\ M}$ | 基线为 2.835M ($\le 1.01\times$) | 适配板载嵌入式芯片 (如 Jetson Orin Nano 4GB/8GB) 的紧凑显存 |
| **计算复杂度** | $\text{GFLOPs}\ (\text{G})$ | $\le \mathbf{10.0\ G}$ | 基线为 10.2G ($\le 0.98\times$) | 控制每秒浮点运算量，降低移动机器人功耗与发热 |
| **CPU 推理延迟** | $t_{\text{CPU}}\ (\text{ms})$ | $\le \mathbf{150\ ms}$ | 基线为 152.3ms ($>6.7\text{ FPS}$) | 保证在工控机无 GPU 环境下的准实时作业能力 (Batch=1) |
| **GPU 推理延迟** | $t_{\text{GPU}}\ (\text{ms})$ | $\le \mathbf{8.0\ ms}$ | 基线为 6.2ms ($>125\text{ FPS}$) | 满足机器人快速行进过程中的高帧率实时感知需求 (Batch=1, FP16) |
| **预训练权重匹配率** | $\text{Weight Matching}\ (\%)$ | $\ge \mathbf{95.0\%}$ | 基线为 100% | 确保可充分继承 COCO 预训练特征，杜绝小样本冷启动训练崩溃 |

---

## 4. 评价指标的严密数学定义 (Mathematical Metric Definitions)

为杜绝评价口径歧义，本研究所有评估指标均严格遵循以下数学公式与定义。

### 4.1 混淆矩阵与基础指标 (Precision & Recall)
设 $TP$（True Positive）、$FP$（False Positive）、$FN$（False Negative）分别代表在给定 IoU 阈值 $\theta$ 下的真阳性、假阳性与假阴性实例数量：

1. **精确率 (Precision)**：
   $$\text{Precision}(\theta) = \frac{TP(\theta)}{TP(\theta) + FP(\theta)}$$

2. **召回率 (Recall)**：
   $$\text{Recall}(\theta) = \frac{TP(\theta)}{TP(\theta) + FN(\theta)}$$

3. **PR 曲线与单阈值平均精度 (Average Precision, AP)**：
   根据 101 点插值法计算 PR 曲线下的积分面积：
   $$\text{AP}(\theta) = \int_{0}^{1} p_{\text{interp}}(r; \theta) \, dr = \frac{1}{101} \sum_{k=0}^{100} \max_{\tilde{r} \ge \frac{k}{100}} p(\tilde{r}; \theta)$$

---

### 4.2 实例分割核心指标 (Mask mAP50, Mask mAP50-95)
设预测掩膜为 $M_p \in \{0, 1\}^{H \times W}$，真实标注掩膜为 $M_g \in \{0, 1\}^{H \times W}$，两者之间的像素级交并比定义为：
$$\text{IoU}_{\text{mask}}(M_p, M_g) = \frac{|M_p \cap M_g|}{|M_p \cup M_g|} = \frac{\sum_{i,j} M_p(i,j) \cdot M_g(i,j)}{\sum_{i,j} \left( M_p(i,j) + M_g(i,j) - M_p(i,j) \cdot M_g(i,j) \right)}$$

1. **$\text{Mask mAP}_{50}$**：
   在掩膜交并比阈值 $\theta = 0.50$ 时的平均精度：
   $$\text{Mask mAP}_{50} = \text{AP}(\theta = 0.50)$$

2. **$\text{Mask mAP}_{50\text{-}95}$ (COCO Primary Metric)**：
   在 10 个等间隔 IoU 阈值（$\theta \in \{0.50, 0.55, 0.60, \dots, 0.95\}$）上的算术平均值：
   $$\text{Mask mAP}_{50\text{-}95} = \frac{1}{10} \sum_{k=0}^{9} \text{AP}(\theta = 0.50 + 0.05k)$$

---

### 4.3 尺度特定指标 (Scale-Specific Mask AP)
根据 COCO 标准，按照真实多边形掩膜的绝对像素面积 $A = \sum_{i,j} M_g(i,j)$ 对实例进行尺度分级：
- **$\text{AP}_s$ (Small Objects)**：评估面积 $A < 32^2 = 1,024\text{ px}^2$ 的微小果实；
- **$\text{AP}_m$ (Medium Objects)**：评估面积 $32^2 \le A \le 96^2 = 9,216\text{ px}^2$ 的中等尺度果实；
- **$\text{AP}_l$ (Large Objects)**：评估面积 $A > 96^2 = 9,216\text{ px}^2$ 的近景大果实。

$$\text{AP}_{\text{scale}} = \frac{1}{10} \sum_{k=0}^{9} \text{AP}_{\text{scale}}(\theta = 0.50 + 0.05k), \quad \text{scale} \in \{s, m, l\}$$

---

### 4.4 边界感知指标 (Boundary IoU & Boundary F1)
为精确度量条带遮挡导致的深凹非凸边缘及近邻果实分界，引入 Boundary 评测协议：

1. **边界膨胀区域定义**：
   设 $d$ 为边界距离容限（本课题取 $d = 2\%\times \min(H, W)$ 或固定 $d = 4\text{ px}$），形态学膨胀操作记为 $\oplus$，则掩膜边缘过渡带定义为：
   $$M_{\text{bound}} = (M \oplus B_d) \cap (M^c \oplus B_d)$$
   其中 $B_d$ 为半径为 $d$ 的圆形结构元，$M^c$ 为掩膜补集。

2. **边界交并比 ($\text{Boundary IoU}$)**：
   $$\text{Boundary IoU}(M_p, M_g) = \frac{|(M_p \cap M_{p,\text{bound}}) \cap (M_g \cap M_{g,\text{bound}})|}{|(M_p \cap M_{p,\text{bound}}) \cup (M_g \cap M_{g,\text{bound}})|}$$

3. **边界精确度、召回率与 $\text{Boundary } F_1$**：
   设 $C_p$ 与 $C_g$ 分别为预测与真实轮廓点集，匹配距离容限为 $\epsilon$：
   $$P_B = \frac{1}{|C_p|} \sum_{p \in C_p} \mathbb{I}\left( \min_{g \in C_g} \|p - g\|_2 \le \epsilon \right), \quad R_B = \frac{1}{|C_g|} \sum_{g \in C_g} \mathbb{I}\left( \min_{p \in C_p} \|g - p\|_2 \le \epsilon \right)$$
   $$\text{Boundary } F_1 = \frac{2 \cdot P_B \cdot R_B}{P_B + R_B}$$

---

### 4.5 语义级辅助分割指标 (Dice & mIoU for Semantic Auxiliary Baselines)
针对 U-Net / SegFormer + Watershed 等语义到实例辅助基线模型，额外报告语义分割层面的重叠度：

1. **Dice 系数 (Sørensen–Dice Coefficient)**：
   $$\text{Dice}(M_p, M_g) = \frac{2 |M_p \cap M_g|}{|M_p| + |M_g|} = \frac{2 \sum_{i,j} M_p(i,j) M_g(i,j)}{\sum_{i,j} M_p(i,j) + \sum_{i,j} M_g(i,j)}$$

2. **平均交并比 (mIoU)**：
   $$\text{mIoU} = \frac{1}{2} \left[ \frac{|M_p \cap M_g|}{|M_p \cup M_g|} + \frac{|M_p^c \cap M_g^c|}{|M_p^c \cup M_g^c|} \right]$$

---

## 5. 实验学科纪律与可复现性原则

1. **三随机种子重复验证**：初筛实验单次运行，正式推荐方法与核心基线必须在 3 个独立随机种子（Seed: 42, 43, 44）下重复运行 300 epoch，并报告 $\text{Mean} \pm \text{Std}$。
2. **严禁硬编码与虚构数据**：所有模型参数量、FLOPs、延迟与精度指标必须经由官方驱动代码与评测脚本直接输出生成，坚决杜绝人工伪造或选择性报告。
3. **环境统一**：所有模型必须在相同软硬件基准（Python 3.10+, PyTorch 2.x, CUDA 12.x, 相同 GPU/CPU 型号）下进行横向对齐对比。
