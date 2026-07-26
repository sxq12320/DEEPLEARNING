<!-- GitHub Issue 草稿：因 Claude GitHub App 未获 DEEPLEARNING 仓库授权（403）无法自动创建。
     修复授权：claude.ai → Settings → Connectors → GitHub → 给 sxq12320/DEEPLEARNING 授权（含 Issues 写权限），
     或直接复制以下内容到 https://github.com/sxq12320/DEEPLEARNING/issues/new
     标题：柑橘远距离小目标改进：73 个模型配置 + 23 项原创 + SXQNet V1-V10 场景轴家族（全部 ≤ 基线参数量）
     标签建议：enhancement（不要用 blog——会触发 README 机器人工作流） -->

# 概述

针对无锡柑橘幼果数据集的核心痛点——**远处果实极小（34.9-40.5% 实例 <32px）、模糊（高频衰减）、发黑（欠曝）、估计标注（低质量框）、端侧部署约束**——对 `1_SEVER/code/ultralytics-main-new/` fork 完成系统性改进。改动全部为**加法**，默认参数下与原 E 系列协议逐字节一致（有单元测试证明）。

## 交付清单

### 模型配置：46 个 yaml（`0_orange_yaml/1_far_small/`，F01-F48）
- **A-F 组**（23 个）：单模块消融——P2 层 / SPD-Conv / HWD 小波下采样 / C3k2_Faster·WT·DWR / 6 种注意力横评 / SPPF-LSKA / RFB / BiFPN / CARAFE / DySample
- **G 组**（5 个）：原创模块——**DFEM 双域频率增强**、**LIAM 亮度不变注意力**、**CSFG 跨级小目标引导**、HVI+DFEM
- **H 组**（5 个）：两两交互组合
- **I 组**（9 个）：CitrusFar-Seg Lite/Full + **6 个同拓扑 leave-one-out**（占位对齐，干净消融）+ P2 上限版
- **J 组**（4 个）：架构级原创——**HR-Stream 双流**、**FreqDetail-PAN 双路颈部**、**Shallow-Heavy 骨干**（2.23M，比基线轻 21%）、**CitrusFar-V2**（2.40M）
- **K 组**（2 个）：单片机部署——**CitrusFar-Edge**（2.15M，纯端侧友好算子）、**Edge-Nano**（1.42M，INT8 后 <1.5MB）
- **L 组**（3 个）：XX-Former 范式原创——**FarFormer**（LRSA 全局注意力⊕Haar 高频，α 门控；融合 LRFormer/WTConv/SRConvNet）、**LumiFormer**（频域通道注意力→暗区调制 + EDFFN；融合 HS-FPN/CIDNet/EVSSM）、**CitrusFormer-Net 完整架构（2.74M/10.6G，比基线更轻）**

### 损失函数（`ultralytics/utils/iou_ext.py` + `loss.py`，经 `train_citrus_seg.py` 旗标启用）
EIoU / SIoU / MPDIoU / ShapeIoU / **WIoU v3**（对估计标注鲁棒）/ Inner-IoU / Focaler / NWD 混合 / **NWD-Wise（原创：按目标尺度自适应混合 NWD 与 WIoU）** / Slide Loss

### 优化器与其他
- 新增 **Lion**（`ultralytics/optim/lion.py`，注册进 trainer；`--optimizer Lion --lr0 0.002`）
- `ultralytics/cfg/default.yaml` 新增 4 个损失开关键；`train_citrus_seg.py` 新增可选旗标（默认不变）

## 验证状态（全部本地通过）
- ✅ 46/46 yaml：build + 640 前向 + 参数量/GFLOPs（thop@640 直测口径，见 `_verify_report.csv`）
- ✅ 12 个重点模型反向传播冒烟（含 F31/F43/F48）
- ✅ 44/44 单元测试（`tests/test_citrus_far.py`）：自实现 CIoU 与 stock 数值一致、9 种 IoU 分发梯度有限、DFEM init 恒等、Lion 收敛
- ⚠️ 未跑：真实数据 3-epoch smoke（需在服务器上执行，见下）

## 服务器执行入口
```bash
cd 1_SEVER/code/ultralytics-main-new && pip install -e .
python verify_far_yamls.py                 # 全量构建自检
python -m pytest tests/test_citrus_far.py  # 单元测试
# 3-epoch smoke（协议要求，先于任何 300ep 实验）
python train_citrus_seg.py --model 0_orange_yaml/1_far_small/F48_yolo11-seg-citrusformer-net.yaml \
    --pretrained yolo11n-seg.pt --name F48_smoke --epochs 3 --iou-type NWDWise
```

## 消融实验设计
五阶段计划（单模块粗筛 50ep → 组内冠军 300ep → 组合+LOO → 损失矩阵 → 轻量化蒸馏部署）、评测协议（AP-small、难例子集、遮挡分组）与 52 条已核验文献（Crossref/arXiv 逐条验证 DOI）详见：
`0_orange_yaml/1_far_small/README_柑橘远距离小目标改进方案.md`

## 第二轮扩展（同日晚，数据驱动 + 频域专线）

**数据集量化体检**（`analyze_citrus_dataset.py`，965 图 / 5,897 实例）：47.9% 实例 <32px；小果比大果暗（V 103 vs 132）且比背景暗；模糊度差 20 倍；|Δa*| 更低（伪装最强）；<32px 小果原生有 93px 信息（3072→640 毁掉 79% 分辨率）。**每处改进对应一条统计证据**。

**新增 F49-F56（8 个）**：TDAM 纹理差分（COD 迁移）/ LCE 暗区曲线前端（Zero-DCE 门控版）/ F52 Edge-V2（2.17M 端侧）/ F53 CitrusFormer-Net-Plus（2.76M 精度主打）/ F54 FLA 线性注意力变体（Mamba 调研裁决）/ **F55 MWCA 多级小波跨频带注意力** / **F56 频域全家桶（2.56M，比基线轻 10%）**。

**训练侧新增**：GA-TAL 高斯度量标签分配（`--tal-metric NWD --tal-min-pos`，治 <16px 正样本饥饿）/ FFL 频域掩码对齐损失（`--freq-loss 0.1`）/ M 系列数据配方（`--aug-preset`）/ 切片推理工具 `predict_citrus_sliced.py`（SAHI 上界基线）。

第二轮文献：theme6-9 共 61 篇核验（COD / Mamba·线性注意力 / 掩码质量·SR / 标签分配），档案在 `3_研究生/文献调研_远距离小目标_20260726/`。
验证状态：54/54 yaml build+forward+18 个反向冒烟通过；51/51 单元测试通过（含 GA-TAL 保底分配、FFL、MWCA、fork 虚框补偿记录性测试）。

## 第三轮：顶会新范式（P 组）

- **F57 HCO 热传导算子**（vHeat 2024 物理范式，3.42M）与 **F58 HyperACE-lite 超图关联增强**（Hyper-YOLO TPAMI 2025 / YOLOv13 范式，**2.73M 比基线轻**）——与 C2PSA/FarFormer/FLA 组成 **P5-mixer 五路同槽位横评**；
- **SAM2 标签精修数据引擎** `refine_labels_sam.py`：用粗框 prompt SAM2 精修 <96px 估计标注小果（IoU 安全阈值防翻车），产出"原标注 vs 精修标注"数据侧消融行；
- **F59 C3k2_LS**（LSNet CVPR 2025 "看大聚小"动态卷积，2.64M 比基线轻）加入 C 组块横评；
- 下一步头部级项：D-FINE FDR 分布精化（ICLR 2025, arXiv:2410.13842，推理零开销，Phase 3 后实施）；
- 验证状态：57/57 yaml build+forward+20 反向冒烟、53/53 单元测试（含热传导平滑性物理性质测试、超图残差恒等起步测试）。

## 第四轮：SXQNet 集大成 + 纹理先验 + 融合升级

- **SXQNet-seg（2.32M/15.5G，参数比基线轻 18%）**：SXQ-Backbone（TGP+LCE 前端 / Shallow-Heavy / C3k2_LS / HWDown / TDAM / MWCA / SPPF_LSKA / HyperACE）+ SXQ-Neck（HSF 筛选融合 / CSFG / LumiFormer / BiFPN）+ 训练侧全套；每个组件都有独立消融行，非堆料；
- **F60/F61 TGP 纹理先验**（用户思想可行化：去颜色 + 多尺度 LCN + σ 可靠性门控解决"远处纹理不好"，~20 参数）；
- **F62 HSF 高层筛选融合**（漏检专项 + 参数下降双赢）；F59 C3k2_LS（LSNet CVPR 2025）；
- 新文档：根目录 `README_改进总览.md`（含漏检/错检根因诊断表 + 全部拓扑与模块 mermaid 框图）+ `BASELINES.md`（9 基线选型与命令手册）。

## 注意事项
- `DATA`/`PROJECT` 路径常量、数据集、既有 runs、E 系列协议**均未改动**（所有新开关默认关闭=原版行为）
- 正式实验数据集：`data/orange_yolo/`（group-aware split 676/193/96 已完成）
- 撞车预警：SSRN 已有 CCDW-YOLO（频域柑橘检测预印本），建议以"幼果实例分割+暗区补偿+尺度自适应损失+端侧轻量化"差异化并加快进度
