# 柑橘远距离小目标改进方案（F 系列）— 文献支撑 + 消融实验设计

> 针对无锡柑橘幼果数据集（965 图 / 5,897 实例，group-aware split 676/193/96）的 YOLO11n-seg 系统性改进。
> 共 **54 个模型 yaml（F01-F56）+ 11 种损失变体 + GA-TAL 标签分配 + 2 个新优化器 + 数据配方与切片推理 ≈ 70+ 处改进**，
> 其中 **16 项原创**：模块级 DFEM/LIAM/CSFG/TDAM/LCE/MWCA + XX-Former 范式 FarFormer/LumiFormer
> + 架构级 HR-Stream/FreqDetail-PAN/Shallow-Heavy/CitrusFormer-Net + 损失级 NWD-Wise/FFL
> + 训练级 GA-TAL + 部署级 CitrusFar-Edge 系列。
> **全部改进由 §10 的数据集量化体检结果驱动**（每处改进对应一条统计证据）。
> 全部 yaml 已通过 build / 640 前向 / 反向传播冒烟（见 `_verify_report.csv`），
> 单元测试 38 项全过（`tests/test_citrus_far.py`）。
>
> 生成器：`_generate_far_yamls.py`（改模板重新运行即可批量再生）
> 验证器：`../../verify_far_yamls.py`（服务器上大实验前先跑）

---

## 1. 痛点 → 数据集证据 → 技术路线

| # | 痛点（用户描述 + 标签统计） | 数据集证据 | 对应改进组 |
|---|---|---|---|
| P1 | **远处柑橘极小**，"很小基本认不出来" | 34.9-40.5% 实例至少一边 <32px；11.3-12.8% <16px（P3/8 上不足 2 格） | A 组 P2 层、B 组无损下采样、F 组上采样、J 组架构 |
| P2 | **模糊**（失焦/分辨率不足 → 高频衰减） | 远景实例边缘梯度弱 | C 组小波块、G 组 DFEM、J 组 FreqDetail-PAN |
| P3 | **发黑**（欠曝/阴影 → 低亮度低对比） | 2026 批 825 张"深绿哑光、与叶片同色" | G 组 DFEM/LIAM/HVI、D 组注意力 |
| P4 | **估计标注**（远处果实框凭感觉画 → 低质量标签） | 用户自述 + audit 脚本发现 483 张数量不一致 | 损失组 WIoU v3 / NWD / NWD-Wise |
| P5 | **密集相邻 + 大尺度跨度** | 47-58.5% 图有近粘连；单图最大/最小面积比中位数 ~6 | BiFPN 加权融合、CSFG、Shallow-Heavy |
| P6 | **端侧部署**（单片机/嵌入式） | 采摘装置算力约束 | K 组 Edge 系列 + INT8/蒸馏路线 |

**创新缝隙（文献调研结论）**：2024-2026 "柑橘+YOLO 改进"已拥挤（YOLO-MECD、ACDNet、SALC-Net 等），
但**未发现**同时命中"绿色未成熟柑橘 + 极小尺度(<16-32px) + 欠曝暗光 + 实例分割"四点交集的论文——这正是本方案主攻方向。
**撞车预警**：SSRN 预印本 CCDW-YOLO（doi:10.2139/ssrn.7068286，频域+柑橘检测）与频域路线部分重叠，
差异化点 = **幼果实例分割 + 暗区补偿 + 尺度自适应损失 + 端侧轻量化**，建议加快进度。

---

## 2. 改进总表（43 个 yaml，每处含文献）

> 参数量/GFLOPs 为 n 尺度、640 输入实测（`_verify_report.csv`）。基线 YOLO11n-seg = 2.84M / 10.2G / mask mAP50-95 0.642。

### A 组 — P2 高分辨率检测层（治 P1）

| yaml | 改进 | Params | GFLOPs | 文献 |
|---|---|---:|---:|---|
| F01 | +P2/4 检测层，mask 原型改由 P2 生成（160×160） | 2.92M | 24.4 | QueryDet, CVPR 2022, doi:10.1109/CVPR52688.2022.01330；MAE-YOLOv8 绿果 p2, doi:10.1016/j.compag.2024.109458 |
| F02 | P2 + SPD 无损下采样 | 4.36M | 27.9 | + SPD-Conv, arXiv:2208.03641 |

### B 组 — 信息保持型下采样（治 P1）

| yaml | 改进 | Params | GFLOPs | 文献 |
|---|---|---:|---:|---|
| F03 | 骨干 3 处步长卷积 → SPD-Conv | 4.28M | 13.9 | Sunkara & Luo, ECML-PKDD 2022, arXiv:2208.03641 |
| F04 | 骨干 3 处 → Haar 小波下采样 HWDown（**更轻**：2.58M） | 2.58M | 9.7 | Xu et al., Pattern Recognition 2023, doi:10.1016/j.patcog.2023.109819 |

### C 组 — C3k2 块变体（治 P2/轻量化）

| yaml | 改进 | Params | GFLOPs | 文献 |
|---|---|---:|---:|---|
| F05 | 骨干 C3k2 → C3k2_Faster（PConv 部分卷积） | 2.71M | 10.1 | FasterNet, CVPR 2023, arXiv:2303.03667 |
| F06 | 骨干 C3k2 → C3k2_WT（小波卷积，抗模糊） | 2.73M | 10.2 | WTConv, ECCV 2024, doi:10.1007/978-3-031-72949-2_21 |
| F07 | 颈部 C3k2 → C3k2_DWR（多膨胀率上下文） | 2.84M | 10.4 | DWRSeg 2022, arXiv:2212.01173 |

### D 组 — 注意力横评（统一插在颈部 P3 = 小目标路径 + mask 原型来源，治 P1/P3）

| yaml | 注意力 | Params | 文献 |
|---|---|---:|---|
| F08 | EMA 高效多尺度注意力 | 2.84M | Ouyang et al., ICASSP 2023, arXiv:2305.13563 |
| F09 | SimAM 无参能量注意力（**+0 参数**） | 2.84M | Yang et al., ICML 2021, PMLR v139（无 DOI，按 PMLR 格式引用） |
| F10 | CBAM（经典对照） | 2.85M | Woo et al., ECCV 2018, arXiv:1807.06521 |
| F11 | Coordinate Attention | 2.84M | Hou et al., CVPR 2021, arXiv:2103.02907 |
| F12 | ELA 高效局部注意力 | 2.84M | Xu & Wan 2024, arXiv:2403.01123 |
| F13 | CAA 上下文锚点注意力（大条带核） | 2.85M | PKINet, CVPR 2024, doi:10.1109/CVPR52733.2024.02617 |

### E 组 — SPPF / 全局上下文（治 P2/P3“远处暗点是不是果”）

| yaml | 改进 | Params | 文献 |
|---|---|---:|---|
| F14 | SPPF → SPPF_LSKA（大核分离注意力） | 3.12M | LSKA, ESWA 2024, doi:10.1016/j.eswa.2023.121352（注意：常见错误 DOI 122116 是错的） |
| F15 | SPPF → RFB（离心感受野） | 3.03M | RFB, ECCV 2018, doi:10.1007/978-3-030-01252-6_24 |

### F 组 — 颈部融合与上采样（治 P1/P5）

| yaml | 改进 | Params | 文献 |
|---|---|---:|---|
| F16 | 4 处 Concat → BiFPNConcat 加权融合 | 2.84M | EfficientDet, CVPR 2020, doi:10.1109/CVPR42600.2020.01079 |
| F17 | 上采样 → CARAFE 内容感知重组 | 2.98M | CARAFE, ICCV 2019, doi:10.1109/ICCV.2019.00310 |
| F18 | 上采样 → DySample 动态点采样（更轻） | 2.87M | DySample, ICCV 2023, doi:10.1109/ICCV51070.2023.00554 |

### G 组 — 原创模块（治 P2/P3，本课题核心创新候选）

| yaml | 改进 | Params | 说明 |
|---|---|---:|---|
| F19 | **DFEM 双域频率增强**（原创）：rFFT2 分频带可学习增益（补模糊高频）+ 暗区响应补偿（补欠曝）+ 残差融合，init 恒等 | 2.87M | 文献基础：FreqFusion (TPAMI 2024, doi:10.1109/TPAMI.2024.3449959) + PE-YOLO (ICANN 2023, arXiv:2307.10953)；联合调制方式原创 |
| F20 | DFEM 双位置（P2+P3）——位置敏感性消融 | 2.88M | 同上 |
| F21 | **LIAM 亮度不变注意力**（原创）：IN 亮度对齐（可学习门控）+ SimAM 能量注意力级联 | 2.84M | 文献基础：IBN-Net (ECCV 2018, arXiv:1807.09441) + SimAM (ICML 2021)；门控级联原创 |
| F22 | **CSFG 跨级小目标引导**（原创）：P2 细节 SPD 无损对齐 + 高通提取 + P3 内容门控注入——P2 头的轻量替代 | 3.03M | 文献基础：Gold-YOLO GD (NeurIPS 2023, arXiv:2309.11331) + ASF-YOLO (doi:10.1016/j.imavis.2024.105057)；组合原创 |
| F23 | HVIEnhance（图像域，仓库已有）+ DFEM（特征域）双重暗光增强 | 2.87M | HVI/CIDNet, CVPR 2025, doi:10.1109/CVPR52734.2025.00533 |

### H 组 — 两两交互（消融矩阵的交互效应行）

| yaml | 组合 | Params |
|---|---|---:|
| F24 | SPD × DySample | 4.31M |
| F25 | SPD × EMA | 4.28M |
| F26 | DFEM × LIAM（双原创协同） | 2.87M |
| F27 | BiFPN × DySample | 2.87M |
| F28 | DFEM × SPD | 4.31M |

### I 组 — CitrusFar-Seg 组合与 leave-one-out

| yaml | 组合 | Params | GFLOPs |
|---|---|---:|---:|
| F30 | **Lite** = SPD + DySample + EMA | 4.31M | 14.0 |
| F31 | **Full** = DFEM + SPD + SPPF_LSKA + BiFPN + DySample + LIAM（配 `--iou-type NWDWise`） | 4.61M | 14.4 |
| F32-F37 | Full 的 6 个 leave-one-out（**同拓扑占位，逐层索引对齐** → 干净消融） | 3.17-4.61M | — |
| F38 | Full + P2 层（性能上限探索） | 4.69M | 28.4 |

### J 组 — 架构级原创（大改网络拓扑，论文核心创新候选）

| yaml | 架构创新 | Params | GFLOPs | 说明 |
|---|---|---:|---:|---|
| F40 | **HR-Stream 双流高分辨率辅助流**：P2 细节流与主干并行，三路 BiFPN 融合进 P3 | 2.93M | 11.7 | HRNet 保持高分辨率思想 (TPAMI 2020, arXiv:1908.07919) 的 nano 级实现，拓扑原创 |
| F41 | **FreqDetail-PAN 细节-语义双路颈部**：整个 neck 重设计——语义自顶向下 + P2→DFEM→SPD 细节直达通路 + C3k2_WT 小波融合 | 3.02M | 12.8 | 颈部拓扑原创 |
| F42 | **Shallow-Heavy 骨干**：P2/P3 深度×2、P5 通道 -25%——把算力搬到小目标所在分辨率，**参数反降 21%** | **2.23M** | 10.2 | 重分配方案原创；依据 QueryDet + EfficientNet 复合缩放 (ICML 2019, arXiv:1905.11946) |
| F43 | **CitrusFar-Seg-V2** = F42 骨干 + F41 颈部（+NWDWise 损失）——**比基线轻 15% 的完整原创方法** | **2.40M** | 12.6 | 论文主打候选 |

### K 组 — 部署导向轻量化（单片机/嵌入式；只用 conv/pool/concat/slice 算子）

| yaml | 设计 | Params | GFLOPs | 说明 |
|---|---|---:|---:|---|
| F44 | **CitrusFar-Edge** = Shallow-Heavy + 全网 C3k2_Faster + HWDown + BiFPN + SPPF_LSKA + CSFG | **2.15M** | 11.5 | 无 FFT/grid_sample/unfold/IN → ONNX→NCNN/RKNN/TFLite-INT8 直转 |
| F45 | **Edge-Nano** 极限压缩（P4/P5 通道 384/512、去 C2PSA 保量化友好） | **1.42M** | 10.2 | INT8 后权重 <1.5MB，可上 RV1106/K230 级端侧；320 输入时 ~2.6G |

### L 组 — XX-Former 范式原创模块（MetaFormer 结构；论文主打创新，按"两三篇融合出全新模块"方法论设计）

范式：`x + TokenMixer(Norm(x)); x + FFN(Norm(x))`（MetaFormer, Yu et al., CVPR 2022, arXiv:2111.11418）。
主创新点在 Token Mixer，次创新点在 FFN，两处均为多论文机制的有机融合而非搬运。

| yaml | 模块 | Params | GFLOPs | Token Mixer（主创新） | FFN（次创新） | 融合来源 |
|---|---|---:|---:|---|---|---|
| F46 | **FarFormer**（远场感知 Former，×2 替换 P5 端 C2PSA） | 3.70M | 10.7 | **LGFM**：α·LRSA（QKV 池化 8×8 的低分辨率全局注意力，近线性代价全图上下文）+ (1-α)·Haar 高频子带分支，α 可学习通道门控 | **MSDFFFN**：5×5/7×7 通道拆分深度卷积 + 洗牌 | LRFormer (TPAMI 2025, IEEE doc 11029508) + WTConv (ECCV 2024) + SRConvNet DML (IJCV 2025, doi:10.1007/s11263-024-02147-y) |
| F47 | **LumiFormer**（亮度感知 Former，颈部 P3） | 2.86M | 10.6 | 频域通道注意力（rFFT 幅谱去直流→"有结构"通道加权）→ 暗区空间调制（亮度图→暗区门控放大）串联 | **EDFFN**：末端可学习频带筛选（LayerScale 残差，init 恒等） | HS-FPN HFP (AAAI 2025, arXiv:2412.10116) + CIDNet/PE-YOLO 暗区思想 + EVSSM EDFFN (CVPR 2025, arXiv:2405.14343) + CaiT LayerScale (arXiv:2103.17239) |
| F48 | **CitrusFormer-Net**（完整架构）= Shallow-Heavy 骨干 + FarFormer@P5 + LumiFormer@P3 + 全 BiFPN | **2.74M** | **10.6** | 三个原创组件各治一痛点：重分配治"小"、FarFormer 治"远+模糊"、LumiFormer 治"暗"；配 `--iou-type NWDWise` 构成完整方法。**比基线轻 3.5%、FLOPs 持平** | | 论文主打候选（与 F43 互为架构对照）|

> F46 曾被 ultralytics `get_flops` 误报 37.5G——其 stride-trick 外推对含固定代价模块（LRSA 8×8 注意力）的网络会严重高估；
> `verify_far_yamls.py` 已改为 thop@640 直测（×2 = Ultralytics MACs→FLOPs 口径），全表数字为真实值。

---

## 3. 损失函数改进（10 种，代码在 `ultralytics/utils/iou_ext.py` + `loss.py`）

通过 `train_citrus_seg.py` 旗标启用，**默认值 = 原协议逐字节一致**（CIoU 走 stock 代码路径，单元测试证明数值相同）：

| # | 损失 | 启用方式 | 针对痛点 | 文献 |
|---|---|---|---|---|
| L1 | EIoU | `--iou-type EIoU` | 宽高直接回归更快收敛 | Zhang et al. 2022, doi:10.1016/j.neucom.2022.07.042 |
| L2 | SIoU | `--iou-type SIoU` | 角度感知回归 | Gevorgyan 2022, arXiv:2205.12740 |
| L3 | MPDIoU | `--iou-type MPDIoU` | 角点距离，简洁高效 | Ma et al. 2023, arXiv:2307.07662 |
| L4 | Shape-IoU | `--iou-type ShapeIoU` | 形状/尺度权重——柑橘近圆形先验 | Zhang & Zhang 2023, arXiv:2312.17663 |
| L5 | **WIoU v3** | `--iou-type WIoU` | **动态非单调聚焦，削减低质量标注的有害梯度——直接对 P4"估计标注"** | Tong et al. 2023, arXiv:2301.10051 |
| L6 | Inner-IoU | `--inner-ratio 0.75`（可叠加任意 iou-type） | 辅助尺度框加速小目标收敛 | Zhang et al. 2023, arXiv:2311.02877 |
| L7 | Focaler | `--iou-type FocalerCIoU / FocalerWIoU` | 线性区间重映射聚焦难样本 | Zhang & Zhang 2024, arXiv:2401.10525 |
| L8 | NWD 混合 | `--nwd-ratio 0.4`（可叠加） | 高斯 Wasserstein 距离对 <16px 目标的像素偏移不敏感 | Wang et al. 2021, arXiv:2110.13389 |
| L9 | **NWD-Wise**（原创） | `--iou-type NWDWise` | **按目标尺度 sigmoid 自适应混合：极小目标（<~4 特征格）NWD 主导（容忍标注偏移），中大目标 WIoU 主导（抑制离群梯度）——P1+P4 联合求解** | 组合原创，基于 L5+L8 |
| L10 | Slide Loss | `--slide` | 以 batch 均值 IoU 为界指数加权难正样本（远处模糊果常年低 IoU） | YOLO-FaceV2, arXiv:2208.02019 |

**引用链（写论文立论用，theme3 调研给出）**：Focal (doi:10.1109/TPAMI.2018.2858826) → GHM 指出梯度离群点多为错标样本 (doi:10.1609/aaai.v33i01.33018577) → Learning From Noisy Anchors (CVPR 2020, doi:10.1109/cvpr42600.2020.01060) → WIoU → 本课题"估计标注"场景。
注意转述分寸：WIoU 论文主张的是"机制天然抑制异常梯度、对不精确标注鲁棒"，**并未**专做人为噪声标注实验。

## 4. 优化器与训练策略

| # | 改进 | 启用方式 | 文献 |
|---|---|---|---|
| O1 | **Lion**（新增，`ultralytics/optim/lion.py`）：sign 动量更新，显存省半；**lr 需为 AdamW 的 1/3-1/10** | `--optimizer Lion --lr0 0.002` | Chen et al., NeurIPS 2023, arXiv:2302.06675 |
| O2 | PIDAO / SMCAO / MuSGD（fork 已有，纳入消融） | `--optimizer PIDAO` 等 | Muon 正式引用：arXiv:2502.16982 |
| T1 | copy_paste 增强（分割任务免费的小目标扩增） | `copy_paste=0.3`（需协议变更批准） | Kisantal et al. 2019, arXiv:1902.07296 |
| T2 | close_mosaic 末期关马赛克（小目标训练稳定收尾） | Ultralytics 内建（默认 10） | YOLOX, arXiv:2107.08430 |
| T3 | CWD 通道蒸馏：YOLO11s-seg 教师 → F44/F45 学生，**不加推理参数补精度** | 后续实现 | Shu et al., ICCV 2021, arXiv:2011.13256 |

---

## 5. 消融实验设计（五阶段）

**协议纪律**：全部用 `train_citrus_seg.py` 固定协议（AdamW/lr0=0.01/seed42/640/300ep/patience100，架构外零变量）；
**先修复 burst 泄漏做 group-aware split**（README §3 警告），再跑正式矩阵；每阶段冠军进入下一阶段。

### Phase 1 — 单模块粗筛（50 epoch，一次一因子）
A-G 组共 23 个单模块 yaml + 基线，各跑 50ep。出各组冠军（预算不足优先：F04、F06、F09、F13、F14、F18、F19、F21、F22）。
**判据**：mask mAP50-95(small) 提升 ≥0.5pt 且延迟增幅 <15% 者晋级。

### Phase 2 — 组内冠军全程验证（300 epoch）
各组冠军 + F01(P2) + F42(Shallow-Heavy) + F46(FarFormer) + F47(LumiFormer) 跑满 300ep，报告全表指标（见 §6 协议）。

### Phase 3 — 组合与 leave-one-out（论文表 3/表 4）
- F30/F31/F43/F44/F48 全程训练；
- F31 的 LOO：F32-F37 六行（同拓扑占位保证干净）；
- F43 的组件消融：F42（只有骨干）、F41（只有颈部）、F43（全量）三行阶梯；
- **F48 CitrusFormer-Net 的组件消融**：F42（只有重分配骨干）、F46（只有 FarFormer）、
  F47（只有 LumiFormer）、F48（全量）四行阶梯——每个原创 Former 的独立贡献与协同；
- 交互效应：F24-F28 五行（验证 1+1>2 还是相互抵消）。

### Phase 4 — 损失矩阵（对 Phase 3 冠军架构）
| 行 | 命令后缀 |
|---|---|
| CIoU 基线 | （无旗标） |
| WIoU v3 | `--iou-type WIoU` |
| NWD 混合 | `--nwd-ratio 0.4` |
| **NWDWise（原创）** | `--iou-type NWDWise` |
| NWDWise + Slide | `--iou-type NWDWise --slide` |
| Inner-WIoU | `--iou-type WIoU --inner-ratio 0.75` |
| FocalerWIoU | `--iou-type FocalerWIoU` |
优化器支线：AdamW（基线）vs `--optimizer Lion --lr0 0.002` vs `--optimizer SMCAO`（各 1 seed 粗筛）。

### Phase 5 — 轻量化与部署（论文表 5 + 落地）
1. F44/F45 全程训练；不达标则 T3 蒸馏补精度（教师 = Phase 3 冠军的 s 尺度版）。
2. 导出链路：`yolo export format=onnx opset=12` → NCNN/RKNN INT8 PTQ →（掉点 >2pt 则 QAT）。
3. 端侧报告：参数量 / 模型体积(INT8) / 目标板实测延迟 / 320-416-640 三档输入的精度-速度曲线。

### 收尾纪律
- 冠军与基线**3 seeds 重复**报 mean±std（AGENTS.md 要求）；
- 每完成一组立即写入 `results_summary.csv`，不混口径。

## 6. 评测协议（比总 mAP 更能体现改进点）

1. 常规：mask mAP50-95 / mAP50 / P / R + Params / GFLOPs / 实测延迟；
2. **按尺度分组**：AP-small(<32²)/medium/large——远小目标改进的主证据；
3. **难例子集**（README §6 已定义）：small / dense / adjacent-pair / concave-occlusion / scale-span / truncated / cross-batch 分组报告；
4. **遮挡分组协议**：occluded vs non-occluded 两组分别报 mask AP（照搬 Sapkota et al., arXiv:2410.19869 的协议，便于国际对比）；
5. 可视化：`vis_pred_vs_gt.py` 远景裁剪放大对比图（论文图 5 素材）。

---

## 7. 与现有工作的差异化（答审稿人）

| 潜在撞车 | 差异点 |
|---|---|
| YOLO-MECD（YOLOv11 柑橘检测, doi:10.3390/agronomy15030687） | 它是检测，本课题是**实例分割**且聚焦**极小+暗光**幼果 |
| CCDW-YOLO（频域柑橘, SSRN 预印本） | 本课题 DFEM = 频域**+暗区响应**双域联合，且有 NWDWise 损失与端侧轻量化整条线 |
| MAE-YOLOv8（绿果 p2, doi:10.1016/j.compag.2024.109458） | 它是李子检测；本课题以 CSFG/HR-Stream 做 **P2 的轻量替代**并给出 LOO 证据 |
| 伪装目标检测视角（Zhai 2024, doi:10.1016/j.compag.2024.109356） | 可引用其"绿-绿同色=伪装目标"立论，本课题从频率/亮度域给出不同解法 |
| LAM-YOLO（光照-遮挡注意力, doi:10.1016/j.cviu.2025.104489） | 无人机域；本课题 LIAM 用 IN 亮度对齐 + 无参注意力，机制不同且更轻 |

## 8. 文件与代码变更清单

**新增**（全部为加法，不影响现有 E 系列协议）：
- `ultralytics/nn/modules/citrus_far.py` — 18 个新模块（含 DFEM/LIAM/CSFG 原创）
- `ultralytics/utils/iou_ext.py` — 扩展 IoU 族 + NWD + WIoU + Focaler
- `ultralytics/optim/lion.py` — Lion 优化器
- `0_orange_yaml/1_far_small/` — 43 个 yaml + 生成器 + 验证报表 + 本文档
- `verify_far_yamls.py`、`tests/test_citrus_far.py`

**修改**（均为注册/开关，默认行为不变）：
- `ultralytics/nn/modules/__init__.py`、`ultralytics/nn/tasks.py` — 模块注册（AGENTS.md 四步流程）
- `ultralytics/utils/loss.py` — BboxLoss 分发 + Slide Loss（默认 CIoU 走 stock 路径，数值一致有测试）
- `ultralytics/cfg/default.yaml` — 新增 iou_type / inner_ratio / nwd_ratio / use_slide 四键
- `ultralytics/engine/trainer.py`、`ultralytics/optim/__init__.py` — Lion 注册
- `train_citrus_seg.py` — 新增可选旗标（默认值下与原脚本行为完全一致）

**未动**：`DATA`/`PROJECT` 路径、`200_orange_wuxi_seg.yaml`、数据集、已有 runs、E 系列协议。

## 9. 参考文献（均经 Crossref / arXiv / Semantic Scholar API 核验，2026-07-26）

<details><summary>展开完整列表（52 条）</summary>

**柑橘/绿果**：Gao 2024 (10.3389/fpls.2024.1375118)；Lyu 2022 (10.3390/s22020576)；Zhang 2024 (10.1016/j.compag.2024.109366)；Chen 2025 SALC-Net (10.1088/1361-6501/ae1aa2)；El Akrouchi 2025 (10.1016/j.atech.2025.100834)；Fu 2024 (10.3390/foods13071060)；Zhai 2024 伪装视角 (10.1016/j.compag.2024.109356)；Jia 2022 Polar-Net (10.3389/fpls.2022.1054007)；Li 2025 (10.3389/fpls.2025.1655093)；Sapkota 2024 (arXiv:2410.19869; 10.1016/j.aiia.2024.07.001; 10.1109/ACCESS.2024.3378261)；Liu 2024 MAE-YOLOv8 (10.1016/j.compag.2024.109458)；Liao 2025 YOLO-MECD (10.3390/agronomy15030687)；Zhang 2024 多尺度自适应 (10.1016/j.compag.2024.108836)；Zheng 2025 LAM-YOLO (10.1016/j.cviu.2025.104489)；Wang 2026 ACDNet (10.3390/agriculture16020148)；Wei 2026 (10.1016/j.asoc.2026.115506)；Liang 2020 夜间荔枝 (10.1016/j.compag.2019.105192)。

**小目标**：QueryDet (10.1109/CVPR52688.2022.01330)；小目标综述 TPAMI 2023 (10.1109/TPAMI.2023.3290594)；SPD-Conv (arXiv:2208.03641)；BiFPN (10.1109/CVPR42600.2020.01079)；AFPN (10.1109/SMC53992.2023.10394415)；Gold-YOLO (arXiv:2309.11331)；HS-FPN (10.1016/j.compbiomed.2024.107917)；ASF-YOLO (10.1016/j.imavis.2024.105057)；CARAFE (10.1109/ICCV.2019.00310)；DySample (10.1109/ICCV51070.2023.00554)；RFB (10.1007/978-3-030-01252-6_24)；LSKNet (10.1109/ICCV51070.2023.01540)；UniRepLKNet (10.1109/CVPR52733.2024.00527)；PKINet/CAA (10.1109/CVPR52733.2024.02617)；NWD (arXiv:2110.13389)；RFLA (arXiv:2208.08738)；copy-paste (arXiv:1902.07296)；HWD (10.1016/j.patcog.2023.109819)。

**损失**：DIoU/CIoU (10.1609/AAAI.V34I07.6999)；EIoU (10.1016/j.neucom.2022.07.042)；SIoU (arXiv:2205.12740)；WIoU (arXiv:2301.10051)；Inner-IoU (arXiv:2311.02877)；Shape-IoU (arXiv:2312.17663)；MPDIoU (arXiv:2307.07662)；Focaler-IoU (arXiv:2401.10525)；PIoU v2 (10.1016/j.neunet.2023.11.041)；Focal (10.1109/TPAMI.2018.2858826)；VariFocal (10.1109/CVPR46437.2021.00841)；Slide (arXiv:2208.02019)；GHM (10.1609/aaai.v33i01.33018577)；Boundary Loss (10.1016/J.MEDIA.2020.101851)；Noisy Anchors (10.1109/cvpr42600.2020.01060)。

**暗光/频域**：Zero-DCE (10.1109/CVPR42600.2020.00185)；SCI (10.1109/CVPR52688.2022.00555)；Retinexformer (10.1109/ICCV51070.2023.01149)；HVI/CIDNet (10.1109/CVPR52734.2025.00533)；IA-YOLO (10.1609/aaai.v36i2.20072)；DENet (10.1007/978-3-031-26313-2_30)；PE-YOLO (arXiv:2307.10953)；FeatEnHancer (10.1109/ICCV51070.2023.00619)；WTConv (10.1007/978-3-031-72949-2_21)；FreqFusion (10.1109/TPAMI.2024.3449959)；DeblurGAN-v2 (10.1109/ICCV.2019.00897)；RFA-YOLOv8 (10.3390/agriculture15181982)。

**骨干/注意力/优化器**：StarNet (arXiv:2403.19967)；FasterNet (arXiv:2303.03667)；RepViT (arXiv:2307.09283)；EfficientViT (arXiv:2305.07027)；GhostNetV2 (arXiv:2211.12905)；LSNet (arXiv:2503.23135)；CBAM (arXiv:1807.06521)；CoordAtt (arXiv:2103.02907)；SimAM (ICML 2021, PMLR v139)；EMA (arXiv:2305.13563)；LSKA (10.1016/j.eswa.2023.121352)；MLCA (10.1016/j.engappai.2023.106442)；ELA (arXiv:2403.01123)；HRNet (arXiv:1908.07919)；EfficientNet (arXiv:1905.11946)；IBN-Net (arXiv:1807.09441)；AdamW (arXiv:1711.05101)；Lion (arXiv:2302.06675)；Muon (arXiv:2502.16982)；CWD (arXiv:2011.13256)；MGD (arXiv:2205.01529)；YOLOv4 (arXiv:2004.10934)；YOLOX (arXiv:2107.08430)。

</details>

原始核验档案（含逐篇适用性分析）：`E:\mastercode\3_研究生\文献调研_远距离小目标_20260726\theme1-9*.md`（已归档，共 9 主题 130+ 篇核验文献）。

---

# 第二轮扩展（数据驱动 + 频域专线，2026-07-26 晚）

## 10. 数据集量化体检（`analyze_citrus_dataset.py` → `_dataset_analysis.md` / `_dataset_stats.csv`）

对 965 图 / 5,897 实例逐实例测量（尺寸 / HSV-V 亮度 / Laplacian 模糊度 / LAB-a* 绿度对比 / 背景差）：

| 发现 | 数据 | 驱动的改进 |
|---|---|---|
| **47.9% 实例 <32px**（19.4% <16px），比 README 旧估计更严重 | 尺寸分箱 | P2/CSFG/HR-Stream/Shallow-Heavy/GA-TAL |
| 小果**显著更暗**：V=103 vs 大果 132，且比背景暗 -9~-12（大果比背景亮 +6） | 亮度列 | LCE / LumiFormer / DFEM 暗区支 / M-dark 配方 |
| 小果**模糊度差 20 倍**：Lap.var 146 vs 2948 | 模糊列 | MWCA / C3k2_WT / FFL / FarFormer 高频支 |
| 小果**更贴近背景色**：\|Δa*\|=2.2 vs 2.9——伪装效应在小果最强 | 对比列 | TDAM（COD 纹理差分） |
| **<32px 小果原生短边中位数 93px**：3072→640 压缩毁掉 79% 线性分辨率，信息本存在 | 分辨率账 | 切片推理 / 960 微调 / P2 路线 |

## 11. 第二轮新增（F49-F56 + 训练侧）

### N 组 — 数据驱动原创（全部端侧友好算子）

| yaml | 创新 | Params | GFLOPs | 融合来源 |
|---|---|---:|---:|---|
| F49 | **TDAM 纹理差异放大**@P2P3（原创）：多尺度 center-surround 差分 + 内容门控——绿绿伪装的 COD 解法 | 2.91M | 11.8 | SINet 感受野对比 + PFNet distraction + Zhai2024 立论 |
| F50 | **LCE 暗区门控曲线增强**前端（原创）：Zero-DCE 曲线 LE(x)=x+A·x·(1-x) × 暗区门控，A init=0 恒等起步 | 2.85M | 13.2 | Zero-DCE (CVPR 2020) + PE-YOLO；比 HVI 更端侧友好 |
| F51 | LCE + TDAM 联合（治暗×治伪装） | 2.91M | 14.6 | — |
| F52 | **CitrusFar-Edge-V2** = F44 + LCE + TDAM（部署主推升级，全端侧算子） | **2.17M** | 15.1 | — |
| F53 | **CitrusFormer-Net-Plus** = F48 + LCE + TDAM（精度主打；配 NWDWise+GA-TAL+FFL 为完整方法） | 2.76M | 14.1 | — |
| F54 | FarFormer-FLA：token mixer 换 focused 线性注意力（theme7 裁决：P5 上线性注意力优于 Mamba，MLLA 理论背书） | 3.71M | 11.0 | FLatten (ICCV 2023) + MLLA (NeurIPS 2024) |

### O 组 — 频域专线原创

| yaml | 创新 | Params | GFLOPs |
|---|---|---:|---:|
| F55 | **MWCA 多级小波跨频带注意力**（原创）：2 级 Haar→7 子带 + 跨频带注意力选频带 + 高频显著图门控低频；无 FFT、端侧可转 | 2.94M | 10.6 |
| F56 | **CitrusFreq-Seg 频域主线** = HWDown + C3k2_WT + MWCA（配 `--freq-loss 0.1`）——"分解-卷积-注意力-监督"全频域链路，**2.56M 比基线轻 10%** | **2.56M** | **9.7** |

频域家族至此覆盖 4 个层面：**下采样**(HWDown) / **卷积块**(C3k2_WT) / **注意力**(MWCA·DFEM·EDFFN·HFBranch·FreqChannelAttn) / **损失**(FFL)——审稿人问"频域创新在哪"有完整回答。

### 训练侧原创（代码已接入，默认关闭=原版行为）

| 改进 | 启用 | 机理与依据 |
|---|---|---|
| **GA-TAL 高斯度量标签分配**（原创组合） | `--tal-metric NWD`（或 Mix）`--tal-min-pos` | <16px 果在 IoU 度量下 t=s^α·u^β 坍缩；NWD 度量修复 topk 排序质量，min_pos 保底每 GT ≥1 正样本（RFLA 补偿思想）。注：本 fork 上游已含 stride_val 虚框补偿（测试 `test_fork_virtual_box_compensation_exists` 记录），GA-TAL 是其上的度量升级+兜底 |
| **FFL 频域掩码对齐损失**（原创迁移） | `--freq-loss 0.1` | Focal Frequency Loss (ICCV 2021, arXiv:2012.12821) 迁移到实例分割 proto-mask：谱误差自聚焦 → 高频差异（小果边界糊）权重最大 |
| M 系列数据配方 | `--aug-preset dark/smallobj/dark_smallobj` | dark: hsv_v 0.6（数据: 小果更暗）；smallobj: copy_paste 0.3 + scale 0.7 (Kisantal 2019) |
| 切片推理工具 | `python predict_citrus_sliced.py --weights ... --tiles 3` | SAHI (ICIP 2022) 思想：精度上界基线 + 量化"640 压缩损失了多少检出"；输出 full vs sliced 对比图 |

### 第二轮消融设计追加

- **Phase 1 追加行**：F49/F50/F54/F55 单模块 50ep 粗筛；
- **Phase 3 追加阶梯**：F53 = F48+LCE+TDAM 三行增量（F48 → +LCE → +LCE+TDAM）；F56 频域 LOO（去 MWCA / 去 WT / 去 HWD / 去 FFL）；
- **Phase 4 追加行**：`--tal-metric NWD`、`--tal-metric NWD --tal-min-pos`、`--freq-loss 0.1`、`--aug-preset dark_smallobj`（各自独立 + 与 NWDWise 组合）；
- **上界参照**：切片推理在 test 集跑一次，报告 sliced vs 640 的检出差 → 论文谈"分辨率损失"的实证。

## 12. 第三轮：顶会新范式（P 组，2024-2026 顶会/顶刊直接移植）

**P5-mixer 五路横评**（同一槽位替换 C2PSA，论文表格干净）：

| 槽位方案 | yaml | Params | GFLOPs | 范式 | 出处 |
|---|---|---:|---:|---|---|
| C2PSA（基线） | 001_3 | 2.84M | 10.2 | 部分自注意力 | YOLO11 |
| FarFormer | F46 | 3.70M | 10.7 | LRSA+小波双分支 Former（原创） | 本课题 |
| FarFormer-FLA | F54 | 3.71M | 11.0 | focused 线性注意力 mixer | FLatten ICCV 2023 |
| **HCO 热传导算子** | F57 | 3.42M | 10.8 | **物理范式**：热方程频域指数核，k=可学习传播距离，O(N log N)，k 可视化=可解释性卖点 | vHeat 2024, arXiv:2405.16555 |
| **HyperACE-lite 超图** | F58 | **2.73M** | 10.2 | **超图范式**：8 条自适应软超边做多对多高阶关联——"果串群体证据互相佐证"（数据: 47-58.5% 图密集相邻） | Hyper-YOLO TPAMI 2025, arXiv:2408.04804；YOLOv13, arXiv:2506.17733 |

**SAM2 标签精修数据引擎**（`refine_labels_sam.py`，基础模型辅助路线）：
用现有粗框做 SAM2 的 box prompt，只精修原生短边 <96px 的估计标注小果；IoU 安全阈值防翻车；
输出 `labels_samrefined/` + 逐实例报告。带来一个**论文中少见的数据侧消融行**："原标注 vs SAM 精修标注"，
直接量化标注质量变量（SAM ICCV 2023, arXiv:2304.02643; SAM2, arXiv:2408.00714）。

**F59 C3k2_LS**（2.64M/10.0G，比基线轻）：LSNet "看大聚小" 动态卷积 bottleneck（CVPR 2025,
arXiv:2503.23135，theme10 裁决"backbone 侧最稳的新算子"，fork 内置官方实现）——加入 C 组横评
（F05 PConv vs F06 WTConv vs F59 LSConv）。

**theme10/11 顶会核验结论**（档案已归档 3_研究生）：
- 超图范式（F58 已实现）被评为"新范式×可移植×端侧交集最大"，主创新首选，Hyper-YOLO TPAMI DOI: 10.1109/TPAMI.2024.3524377；
- **D-FINE FDR+GO-LSD（ICLR 2025, arXiv:2410.13842）**：DFL 同源的分布精化+定位自蒸馏，推理零开销
  ——列为下一步头部级改造项（工程量中等，建议 Phase 3 后实施，与超图 neck 组成"高阶关联 neck + 分布精化头"完整故事）；
- vHeat 注意：ICLR 2025 撤稿，只能以 arXiv preprint 引用（F57 文档已按此处理）；
- 已明确否决（有可引用理由）：Mamba 系（部署）、TTT（不成熟）、Conv-KAN（实证不足）、DEYO；
- SAM 精修路线定位：精修噪声伪标签有顶会先例（SAM_WSSS/SemiRES ICML 2024/S⁴M/SAMST），农业"基础模型标注→轻量 YOLO"已有 SDM-D（arXiv:2411.16196），
  但**精修已有人工估计标注 + 绿色幼果**无先例；三个坑的对策已内置 `refine_labels_sam.py`
  （box+质心双 prompt / 最大连通域 / IoU 闸门回退），论文级创新点 = **分尺度精修可信度判别器**（GMM 动态阈值，文献缺口）。

**第三轮消融追加**：P 组五路 50ep 横评 → 冠军 mixer 替换进 F53/F48 再跑增量；SAM 精修标注 vs 原标注各训一次 F53（其余全同）；C 组三路块横评补 F59。

### 第二轮参考文献（theme6-9，均已核验，档案在 3_研究生）

COD：SINet (CVPR 2020)、PFNet (CVPR 2021)、FEDER (CVPR 2023)、BGNet、ZoomNet、HitNet 等 15 篇；
Mamba/线性注意力：VMamba、Vision Mamba、MambaVision (CVPR 2025)、Mamba-YOLO、FLatten (ICCV 2023, arXiv:2308.00442)、Agent Attention、**MLLA (NeurIPS 2024, arXiv:2405.16605，Mamba≈线性注意力的理论裁决)** 等 16 篇；
掩码质量/SR：PointRend、Mask Transfiner、RefineMask、BPR、YOLACT++、EFPN、Noh ICCV19、**SAHI (ICIP 2022, doi:10.1109/ICIP46576.2022.9897990)** 等 15 篇；
标签分配：TOOD/TAL、ATSS、SimOTA、**RFLA (ECCV 2022, arXiv:2208.08738)**、DSLA、OA-MIL、Stitcher、**Focal Frequency Loss (ICCV 2021, arXiv:2012.12821)** 等 15 篇。
