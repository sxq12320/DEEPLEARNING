# APEX-8 Whole-Image (640x640) Green Citrus Instance Segmentation Architecture Design & Evidence Re-Audit

> 本文档为整图（`640×640` 不切片）绿柑橘实例分割课题的终版结构设计与可行性评估文档，已整合：**历史 370 份训练 CSV 实证复核**、**`G10 / SAGE42 / V12_06` 涨点核心机制吸收**、**Argon × DeepSeek-V4.1-Flash 联合审计 7 项精修**，以及本地实测的静态参数/GFLOPs/预训练迁移/第 1 步非零梯度验证结果。

---

## 1. PR 曲线末端垂直置零的底层机理与双轨解决方案

### 1.1 代码与数学根因（非训练崩溃，系评估边界插值与召回天花板耦合）
1. **代码级哨兵补零（`ultralytics/utils/metrics.py` L769–770, L843–846）**：
   - `compute_ap(recall, precision)` 在经验曲线两端强制拼接端点：
     ```python
     mrec = np.concatenate(([0.0], recall, [recall[-1] if len(recall) else 1.0], [1.0]))
     mpre = np.concatenate(([1.0], precision, [0.0], [0.0]))
     ```
   - 横轴是 **Recall**（而非置信度）。当模型在 `conf=0.001, max_det=300` 下的经验最大召回率 $R_{\max} < 1.0$ 时，坐标在 $R_{\max}$ 处从 $(R_{\max}, P_{\text{end}})$ 垂直落到 $(R_{\max}, 0.0)$ 再水平延伸至 $(1.0, 0.0)$。
2. **物理与算法层面的真实失配（为何全图 $R_{\max}$ 长期卡在 `0.872–0.893`）**：
   - 历史审计表明：`ES00` 基线 `R_max = 0.8723`（915/1049 命中），`V12_06` `R_max = 0.8932`（937/1049 命中，112 个漏检中仅 59 个短边 $<8\text{px}$，至少 **53 个漏检果短边 $\ge 8\text{px}$**）；20% 实例 $<16\text{px}$、72% 灰度 $\text{CNR} < 1$、27% 因枝条切割呈现凹形（solidity $<0.9$）。
3. **APEX 双轨方案**：
   - **向右推高 $R_{\max}$**：`M1` 绿拮抗双域对比主干 + `M2` `C2PhaseBridge` 无损四相位微果证据注入 + `M3` 激活 `forward_head` 专属 `TDC` 语义差异校正（复现 `ES02_semantic` `+3.47% Recall` 机制）与 `64ch` 全秩条带边界原型解码。
   - **出版级标准化 PR 可视化**：[`citrus_apex_pr.py`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/citrus_apex_pr.py) 在保留标准 COCO AP 口径公平性的前提下，明确标出经验支持域 $[0, R_{\max}]$、$R@P90$ 与未达区间 $(R_{\max}, 1.0]$。

---

## 2. 历史实证对比（G10 / SAGE42 / V12_06 / GPT Astra RE07）与架构吸纳决策

| 模型 | 参数量 (M) | Conv/Linear GFLOPs@640 | 历史实测全图 AP50 / AP50-95 | 吸收与纠偏结论 |
| :--- | :---: | :---: | :---: | :--- |
| `ES00 / RE00 (YOLO11n-seg)` | 2.843 | 9.360 | 78.79–81.15% / 62.99–63.33% | 官方同口径对照基准 |
| `G10` (`10_yolo11n-seg-hybrid...`) | 3.214 | 13.513 | **83.824% / 67.681%** | **吸纳其 4 大精度核心**：`64ch` `P2` 细节、`k=7/11/21` 三级 LSKA、`BiFPN` 归一化权重、边界对原型残差调制；**替换掉** `CARAFE` 与 `160×160` 稠密 `3×3`（原单头 `p2_detail` 耗算力 9.55G） |
| `SAGE42` | 2.319 | 9.042 | 82.982% / 67.502% | **吸纳其低算力核心**：`P3` 语义一致门控 + 1 步 `DBPN` 下采样重建误差反投影校正 |
| `V12_06` | 2.284 | 9.592 | 81.941% / 65.332% | **吸纳其有效入口**：`tanh((x-\mu)/\sqrt{\sigma^2+0.05^2})` 局部对比，并保持 `proto_mid=64` 彻底破除 `V12` 原型 16 维秩瓶颈 |
| `ES02_semantic` | 2.857 | — | 80.156% / 64.013% (`+3.47% Recall`) | **吸纳其召回核心**：跨层语义差异校正只作用于 `forward_head` 塔输入，**绝不扰动** `self.proto(P3)` |
| `RE07 (GPT Astra)` | 2.331 | 9.030 | (未训；同系列 `ES02` 仅 ~64.0%) | 三重短板：`nn.Identity` 删层凑因子、192 参数灰度单尺度卷积、完全切断 `C2` 精细原型通路 |

---

## 3. APEX 三大核心模块与 Argon × DeepSeek 7 项精修汇总

详细实现见 [`ultralytics/nn/modules/citrus_apex.py`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/ultralytics/nn/modules/citrus_apex.py)：

1. **Module 1 — `MS-CSFA`（多尺度色度-空间-频率协同主干增强）**
   - `CitrusChromaStripStem`：保留原生首层 `3×3` 权重 100% 迁移，并行构造 $[Z_Y^{(3)}, Z_Y^{(7)}, Z_{C_{\text{opp}}}^{(5)}]$（`V12_06` 平滑 `tanh` 归一化 + $G - 0.5(R+B)$ 绿拮抗色度），经各向同性与**对称双向正交条带 `0.5*(H→V + V→H)`** 深度卷积 + SE 门控，以 `0.05` 暖启动注入。
   - `SPPF_WaveLSKA`：在原生 `SPPF` 后并行引入正交 Haar 2D 小波四子带分支与 `G10` 同款 **`k=7, 11, 21` 三级条带大核注意力**。

2. **Module 2 — `BiFreq-PAN`（预训练保护型双向频率对齐金字塔）**
   - `CitrusFreqAlignedFusion`：主路保持 **100% 恒等直通**（吸取 `G03` 颈部改动掉点教训，零偏移保护下游预训练 `C3k2`）；支路融合 `Softplus BiFPN` 归一化权重、上采样低通抗混叠、$(3\times 3 + 5\times 5)$ 拉普拉斯高频残差与 `PagFM` 像素一致性门控。
   - `C2PhaseBridge`：通过先 `PixelUnshuffle(2)` 后做零均值四相位差分解 `_subpixel_phase_decompose` 完整保留亚像素边缘并平滑直流叶噪，采用 **`P3 + C2-Phase` 联合门控**防止微果在 `P3` 被冲刷时误关 `C2` 信号（救回 `42/153` 无框漏检）。

3. **Module 3 — `BSD-Head`（`SegmentCitrusBSD`：64 通道全秩条带边界反投影头 + 三尺度 `cls_head` 专属 `TDC`）**
   - **`C2-P3-P4` 三尺度 `cls_head` 专属 `TDC`（避开 `V12_04` 跨塔掉点雷区）**：依据 `V12_03 (route=1) > V12_04 (route=2)`（跨三塔掉 `0.99% AP50`）与 `ES02_semantic (+3.47% Recall)` 实证，默认 `use_tdc=True` 将 `tdc_blocks[0]` 升级为融合 `C2` 亚像素四相位、`P3` 与 `P4` 的三尺度差异校准（覆盖无 `M2` 的 `APEX06` 至 `APEX08` 全系），并**严格只作用于 `cls_head` 分类塔**（救回 `85/153` 低分抑制果）；`box_head`、`mask_head`、`self.proto(P3)` 与 `self.p3_prior(P3)` 全程读取原始未扰动几何特征 `x`。
   - **`64ch` 全秩 `FBP-Proto`（避开 `SAGE43` 噪声放大与全家桶互扰雷区）**：用 `3×3 + 5×5` 深度可分离多尺度卷积保留 `G10` 级 64 通道高分辨细节容量；对 `DBPN` 误差 `e_bp` 加 `3×3` 低通滤除叶脉高频噪声并配 `0.02` 保守初值 + `0.2*tanh` 限幅；全模块采用 `0.2*tanh / 0.5*tanh` 有界残差与零中心双向门控 `(1 + 0.5*tanh(boundary))` **仅调制 `proto_detail`**，保持纯原生 `v8SegmentationLoss` 与预训练原型纯净。

---

## 4. 八大网络架构最终实测静态档案与避雷消融顺序

以下全部数据均在最终代码与 YAML 修改后，经 [`tests/test_citrus_apex.py`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/tests/test_citrus_apex.py)（**10/10 通过**，含纯原生 loss、第 1 步全内部分支含三尺度 `tdc_blocks.0.fine_proj/fine_unshuffle` 与联合门控 `p3_proj` 严格非零梯度、`P3==0` 时 `C2` 脉冲救活测试及 `model.fuse()` 等价性）与 [`RUN_CITRUS_APEX.py`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/RUN_CITRUS_APEX.py) `--check-only` 实测写入 [`MODEL_PROFILE_SUMMARY.csv`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/docs/APEX_8_NETWORK_DESIGN_20261113/MODEL_PROFILE_SUMMARY.csv)（本机无数据集训练，故不编造任何新模型 AP）；**八大历史无效/掉点雷区排查、PR 曲线 $R_{\max}\approx 0.90$ 双 AI 辩论纠偏与止损红线**见 [`AI_DECISION_EMPOWERMENT_REPORT.md`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/docs/APEX_8_NETWORK_DESIGN_20261113/AI_DECISION_EMPOWERMENT_REPORT.md)：

| 编号 | 架构配置 (`0_orange_yaml/APEX_series/`) | 消融/路线定位 | Params (M) | Conv/Linear GFLOPs@640 | 预训练元素迁移率 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| `APEX00` | `APEX00_baseline.yaml` | 基准对照 (`M1=0, M2=0, M3=0`) | 2.843 | 9.360 | 98.4% (510/561) |
| **`APEX06`** | `APEX06_m3_bsd_only.yaml` | **消融第 1 步 (`+M3 Only`)**：100% 原生主干/颈部 + `64ch BSD & Tri-Scale TDC` | **2.893** | **10.881** | **96.7%** (510/726) |
| **`APEX07`** | `APEX07_m2_m3_bifreq_bsd.yaml` | **消融第 2 步 (`+M2 +M3`)**：原生主干 + `BiFreq-PAN` + `M3` | **2.956** | **11.335** | **94.5%** (510/917) |
| **`APEX03`** | `APEX03_full_pan_bsd.yaml` | **消融第 3 步 / 主力全 PAN 旗舰 (`+M1 +M2 +M3`)** | **3.034** | **11.603** | **92.0%** (510/992) |
| **`APEX01`** | `APEX01_bsd_flagship.yaml` | **轻量非对称主力旗舰 (`+M1 + Asym-M2 + M3`)** | **2.493** | **11.173** | **90.9%** (450/891) |
| **`APEX02`** | `APEX02_g10_ultra_light.yaml` | **轻量超级 G10**：保留 G10 `FasterNet+WTConv` + 三大模块（较 G10 `-15.8%` GFLOPs） | **2.912** | **11.384** | **88.1%** (421/931) |
| **`APEX04`** | `APEX04_g10_asym_bsd.yaml` | **敏捷轻量 G10**：`FasterNet+WTConv` + 非对称 `M2` + `M3` | **2.371** | **10.955** | **86.0%** (361/830) |
| **`APEX05`** | `APEX05_pds_fullrank_bsd.yaml` | **整图重生版 V12**：`M1` + 双流 `PDS` + 5 输入全秩 `M3` | **2.535** | **11.519** | **89.4%** (450/958) |
| **`APEX08`** | `APEX08_ultimate_max_capacity.yaml` | **理论上限旗舰**：`M1` + `PDS` + 全 PAN `M2` + 5 输入 `M3` | **3.075** | **11.949** | **90.8%** (510/1059) |
| `REF_G10` | `10_yolo11n-seg-hybrid...yaml` | 历史全图最高参照 (`83.824% / 67.681%`) | 3.214 | 13.513 | 80.0% (421/554) |
| `REF_SAGE42` | `SAGE42_asym_semantic_detail.yaml` | 历史低算力高掩膜标杆 (`82.982% / 67.502%`) | 2.319 | 9.042 | 97.9% (450/523) |

> 完整联合 AI 审计细节、API 实测与超参建议见 [`AI_DECISION_EMPOWERMENT_REPORT.md`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/docs/APEX_8_NETWORK_DESIGN_20261113/AI_DECISION_EMPOWERMENT_REPORT.md)。
