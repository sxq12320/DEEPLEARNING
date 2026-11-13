# APEX-8 联合 AI 架构再审计与实验决策赋能报告（Argon + DeepSeek-V4.1-Flash）

> 本文档汇总本课题 **370 份历史训练 CSV 实证规律**、**Argon 架构与梯度流审计**以及 **DeepSeek-V4.1-Flash (`deepseek-v4.1-flash-1017`)** 的实时技术复核意见，并对 `intern-decision-4b` 接口适配机制与全图（$640\times 640$ 不切片）优先级训练决策给出可执行规范。

---

## 一、历史 370 份全图实验铁证复盘（可行性与涨点根因判定）

对 `docs/EVIDENCE_RESET_20261112/latest_all_all.csv` 与 `docs/SAGE_V4_RESULTS_AND_DESIGN_20260903.md` 中的同协议全标签整图实验进行交叉核查，锁定了以下 4 条决定全图涨点与掉点边界的实证规律：

| 历史对照实验 | 全图 Mask AP50 | 全图 Mask AP50-95 | Recall / $R_{\max}$ | 机制归因（代码级可验证） |
| :--- | :---: | :---: | :---: | :--- |
| `ES00_baseline` (原生 `YOLO11n-seg`) | 78.79% | 62.99% | 69.27% / 87.23% | 基准对照 |
| `ES02_semantic` (仅在 `forward_head` 内作 `[P3,P4]` 跨层语义校正，`proto` 读原 `P3`) | **80.16%** | **64.01% (+1.03)** | **72.74% (+3.47)** / **88.47%** | 全图单一改动召回率提升最强因子，不污染 `proto(P3)` |
| `V12_06` vs `V12_03` (入口新增 `tanh` 局部对比残差 `Stem`) | 81.94% | **65.33% (+1.40 vs V12_03)** | 72.16% / **89.32%** | `tanh((x-\mu)/\sqrt{\sigma^2+0.05^2})` 平滑压制反光/阴影极值，明显拉升 AP50-95 与 $R_{\max}$ |
| `G10` (`64ch P2` + 边界乘性残差 + `LSKA 7/11/21` + `BiFPN`) | **83.82%** | **67.68%** | — | `64ch P2` 细节流与 `(1+\sigma(boundary))` 对**原型残差**调制极其有效，但稠密 `3x3` 带来 `13.513G` 算力开销 |
| `SAGE42` (`P3` 语义门控 + 1 步 `DBPN` 反投影细节校正) | 82.98% | 67.50% | — | 以 `9.042G` 低算力逼近 `G10`，证明**语义门控 + 反投影**是轻量高效的核心 |
| 负向教训 1：`G03` (重型频域颈部替换) | — | 65.63% (-1.40 vs G00 67.03%) | — | 在 `C3k2` 之前大幅改写特征幅值会破坏预训练特征分布 |
| 负向教训 2：`ES03` (`ES01` 无门控细节 + `ES02` 简单相加) | 79.81% | 63.63% (-0.38 vs ES02) | 71.50% | 未经 `P3` 语义一致门控的 `C2` 噪声直接污染原型解码 |

---

## 二、Argon × DeepSeek-V4.1-Flash 联合再审计与 7 处关键工程精修

在结合上述 4 条铁证对初版 APEX 实现逐行复查并咨询 `deepseek-v4.1-flash-1017` 后，双方达成一致并完成了以下两轮、共 **7 处零浪费外科手术级修复**（已全部写入 [`citrus_apex.py`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/ultralytics/nn/modules/citrus_apex.py) 与 `APEX01–08` YAML）：

### 第一轮：消除隐藏 AP 陷阱 & 吸收 `G10` 核心优势
1. **默认回归 100% 纯净原生 `v8SegmentationLoss`**：`use_quality_calib=False` 消除正样本区间才监督的 `q_head` 在推理期导致的 Mask-IoU 置信度反转；避免辅助 loss 与主分割 loss 抢占梯度。
2. **`0.05` 暖启动门控取代死零初始化**：所有新增残差分支门控（`stem_gain, sppf_gain, sem_gain, det_gain, bridge_gain, proto_gain, tdc_gain`）采用 `0.05` 初始值，配合 `pytest` 强制检验第 1 步内部所有卷积层梯度严格非零。
3. **吸收 `G10` 四大精度驱动器并压缩算力**：将 `SegmentCitrusBSD` 细节宽度从 32 升至 **`64ch`**（`3x3 + 5x5` 多尺度深度可分离替代 `G10` 稠密 `3x3`）、在 `SPPF_WaveLSKA` 吸收 `G10` 三级 `k=7,11,21` 条带卷积、在 `CitrusFreqAlignedFusion` 吸收 `BiFPN` 归一化权重。

### 第二轮：Argon + DeepSeek-V4.1-Flash 联合数学复核精修
4. **对称双向条带卷积 & `V12_06` 平滑 `tanh` 局部对比（M1）**：
   - `CitrusChromaStripStem._local_zscore` 严格采用经 `V12_06` 验证的 `tanh((t - \mu) / \sqrt{\sigma^2 + 0.05^2})` 抑制蜡质高光与深冠阴影；
   - 条带分支改用 `0.5 * (H->V + V->H)` 双向对称响应，消除横竖枝条方向顺序偏置。
5. **颈部 `1.0x` 恒等主干保护 + `Softplus` `BiFPN`（M2）**：
   - 吸取 `G03` 掉点教训，`CitrusFreqAlignedFusion` 主路保持 **100% 恒等直通 `1.0 * sem / 1.0 * det`**，`F.softplus(self.bifpn_w)` 仅作用于 `0.2 * tanh(gain)` 的高频/低通残差，杜绝 ReLU 零梯度死区并保护下游预训练 `C3k2` 分布。
6. **修复预训练 `self.proto` 污染风险（M3）**：
   - 放弃对预训练 `proto` 直接相乘 `(1 + 0.5*tanh(boundary))`，严格对齐 `G10/SAGE42` 范式，将零中心双向边界门控 `[0.5, 1.5]` 仅施加在新增的 `proto_detail` 残差上：`proto = proto + proto_gain.tanh() * (proto_detail * (1.0 + 0.5 * tanh(boundary)))`。
7. **激活 `forward_head` 专属 `TDC` 跨层语义校正并严格隔离 `proto(P3)`（M3 & YAML）**：
   - 在排查中发现 `APEX01–08` YAML 原第 6 位参数 `False` 意外关闭了 `use_tdc`。现已统一开启 `use_tdc=True`，但严格在 `SegmentCitrusBSD.forward_head` 内部对检测/分类/系数塔执行 `[P3, P4]` 语义差异校正（0.5 `tanh` 限幅残差），而 `self.proto(det_raw[0])` 与 `self.p3_prior(det_raw[0])` 始终读取**零扰动的原始 `P3`**，完整复现 `ES02_semantic` `+3.47% Recall` 机制。

---

## 三、API 协议实测说明与 `--ai-advise` 自动化赋能

- **实测接口状态（`https://discovery-api.intern-ai.org.cn/v1`）**：
  - `deepseek-v4.1-flash-1017`：支持 `/v1/chat/completions`（含 `reasoning_content` + `content`，注意需预留足够 `max_tokens` 与 `timeout` 供思维链完成）。
  - `intern-decision-4b`：网关返回 `404: intern-decision-4b does not support openai_chat protocol`（`/v1/models` 列表内当前活跃的书生对话模型为 `intern-s2`）。
- **自动化集成**：[`RUN_CITRUS_APEX.py`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/RUN_CITRUS_APEX.py) 已内置 `--ai-advise`（通过环境变量 `CITRUS_AI_API_KEY` 传入，**代码仓库零明文私钥**）：先尝试 `intern-decision-4b`，若遇 OpenAI Chat 协议 404 则自动平滑回退到 `intern-s2`，并与 `deepseek-v4.1-flash-1017` 共同读取已完成的 `results.csv` 与 `MODEL_PROFILE_SUMMARY.csv` 输出实验决策建议。

---

## 四、DeepSeek + Argon 联合实验决策与超参规范

### 1. 推荐训练推进顺序（兼顾算力预算与三模块消融闭环）
1. **第一梯队（4 组标准 3-Module 因果消融链，必跑）**：
   - `APEX00_baseline`（YOLO11n-seg 基准，9.360G）
   - `APEX06_m3_bsd_only`（仅加 `M3`，预训练迁移 **96.7%**，验证 `64ch BSD + forward_head TDC` 纯净增益）
   - `APEX07_m2_m3_bifreq_bsd`（`+M2 +M3`，预训练迁移 **94.8%**，验证 `BiFreq-PAN + C2PhaseBridge` 增量）
   - `APEX03_full_pan_bsd`（`+M1 +M2 +M3` 完整 4 节点 PAN 主候选，**3.023M / 11.429G**，预训练 **92.4%**）
2. **第二梯队（轻量化与 `G10` 降本超越对比）**：
   - `APEX01_bsd_flagship`（`M1 + 非对称 M2 + M3`，**2.483M / 11.000G**，轻量主力旗舰）
   - `APEX02_g10_ultra_light`（吸收 `G10 FasterNet+WTConv` 主干 + `M1/M2/M3`，**2.901M / 11.211G**，比原版 `G10` 降算力 `-17.0%`）
   - `APEX04_g10_asym_bsd`（敏捷轻量版 `G10`，**2.361M / 10.781G**）
3. **第三梯队（双流重生 `V12` 与理论容量上限）**：
   - `APEX05_pds_fullrank_bsd`（保留双流 `PDS` 并破除 16ch 秩瓶颈的整图重生 `V12`）
   - `APEX08_ultimate_max_capacity`（`M1 + 双流 PDS + 全 PAN M2 + 5输入 M3`，**3.065M / 11.775G**）

### 2. 整图 `640×640` 统一训练与防掉点超参决策
- **推荐训练预算**：`epochs=250~300`, `imgsz=640`, `batch=16`, `mask_ratio=2`（与 `EVIDENCE_RESET` 统一全标签高精度掩膜对齐）, `amp=False`, `close_mosaic=15`。
- **绿柑橘色彩增强防护（DeepSeek 重点提示）**：切勿使用过高饱和度/色相扰动破坏绿色拮抗色度特征，建议 `hsv_h=0.015`, `hsv_s=0.3`（低于默认 0.7）, `hsv_v=0.4`；消融阶段先设 `copy_paste=0.0`（排除贴图伪边界干扰），确认结构最优后再单独测试 `copy_paste=0.2~0.3`。
- **评估与 PR 诊断**：全标签 `val` 统一使用 `conf=0.001, iou=0.7, max_det=300`，训练结束后同时以全标签与 `min8` 复核，并用 `citrus_apex_pr.py` 输出实测 `R_max` 与 `R@P90`。


---

## 五、八大历史无效/掉点雷区全量普查与 APEX 避雷止损规范（三模型联合审定）

在对 `E_V4–V12`、`SAGE_V4–V7`、`I_V1–V4`、`J_V6/J_V7R`、`G00–G13` 等历史负向消融做地毯式排查后，提炼出本课题已实测的 **8 大“无效/掉点雷区”** 及其共性规律，并已全部在当前 `citrus_apex.py` 与训练策略中完成代码级封堵：

### 5.1 八大历史负向雷区明细与 APEX 代码防线
| 雷区 | 历史负向实测对比与具体掉点 | 失效物理/数学根因 | APEX 落地的防雷机制 |
| :--- | :--- | :--- | :--- |
| **1. `TDC` 跨三塔污染定位** | `V12_03(route=1, 仅cls)` 80.19% AP50 → `V12_04(route=2, 共享三塔)` **79.20% (-0.99% AP50 / -0.95% R_max)** | 高层 P4/P5 语义平滑破坏 `box_head` 回归与 `mask_head` 空间对齐 | `SegmentCitrusBSD.forward_head` 中 `x_cls = self._refine_detect_feats(x)` **严格仅喂给 `cls_head`**，`box_head / mask_head / proto / p3_prior` 全用纯净几何 `x` |
| **2. `DBPN` 反投影后期放大噪声** | `SAGE42` (无反投影，尾20轮 66.483%) → `SAGE43` (1步反投影，峰值 `-0.135%`，尾20轮 **66.001% / `-0.481%`**) | 训练后期反投影残差放大绿叶高频叶脉伪影，稀释有效信号 | 对 `e_bp = s_p3 - down_recon` 施以 `3×3` 均值低通去高频叶噪，并将 `bp_scale` 降至 `0.02` 保守初始化 + `0.2*tanh` 严密限幅 |
| **3. 压缩原型通道或延后上采样** | `SAGE42→50`(延后上采样): **`-1.655%`**；`V10_01→02`: 细切片 **`-0.791%`**；`V9_01→03`: `-0.537%` | 压缩空间通道或把 `3×3` 移到 stride-8 损失高频边界秩容量 | 全程保持 `proto_mid=64`（`G10` 级满秩容量），严格在 `160×160` stride-4 上融合 `C2` 与生成 32 维满秩原型 |
| **4. 主干内插持续双流 (`PDS`) 互扰** | `V12_04 → P02`(去掉PDS): 全图 **`+0.560%`**；`SAGE60→63`(主干加16ch PDS): **`-3.172%`**；`J90→J94`: **`-1.059%`** | 多处插入持续双流与原主干特征分布打架，挤占预训练表示空间 | 主力消融链 `APEX00→06→07→03` 及 `01/02/04` **默认零 PDS**；`APEX05/08` 单列为全秩控制对照 |
| **5. 硬替换 `C3k2` 主干或 `Concat→Add`** | `SAGE60→61`(相加颈部): **`-1.792%`**；`SAGE60→62`(换主干): **`-3.737%`**；`G00→G03`(频域颈): **`-1.400%`**；`V11_02→03`: **`-0.880%`** | 破坏 `yolo11n-seg.pt` 预训练分布，小数据集从头学或通道裁剪容量崩塌 | `APEX03/06/07` 保留 100% 原生 `C3k2` + `Concat` 通道排布（预训练迁移 **92.4%–96.7%**）；`M2` 颈部保持 `1.0x` 恒等主路 |
| **6. 密集 `P2` 检测头 (`34000` 候选)** | `I20–I24 / SAGE72`：全图 AP 差值仅 `0.07%` 量级，正负样本失衡 | $160\times 160$ 网格密集分类/框回归产生海量极低分绿叶背景假阳性 | 仅保留 `P3/P4/P5` 8400 锚点；`C2` 仅经轻量 `C2PhaseBridge` 与 `64ch` 原型头输出精准边界 |
| **7. 叠加多余几何/对比/边缘损失** | `G10 standard(67.68%) → full loss(64.03%, -3.65%)`；`V12_04→P08`(关邻域loss **`+0.695%`**)；`V11_06→08`(ring loss **`-0.535%`**) | 多组边界/邻域/对比辅助梯度与主分割 TAL 分配、凹陷遮挡真值矛盾 | **默认 100% 纯净原生 `v8SegmentationLoss`**，`use_quality_calib=False`，零外部几何损失干扰 |
| **8. “全家桶”无边界残差堆叠** | `V9_05(71.92%) → V9_08(70.95%, -0.97%)`；`V10_05(72.12%) → V10_08(71.11%, -1.01%)` | 多模块大跨度残差同时注入叠加方差，后期协同负效应 | 全系统统一采用 `0.2*tanh` / `0.5*tanh` 有界残差包络，坚持**先单步后组合 3 模块清晰消融** |

### 5.2 DeepSeek-V4.1-Flash + Intern-S2 联合隐患复核与服务器上卡止损线
1. **双重模型共识隐患提示**：
   - `direction_label_interactions.csv` 显示核心增益高度集中在全图困难实例（`<8–16px` 微小果与重度枝叶遮挡果），而 `min8` 仅大果上差值趋近于零，因此**必须严格以全标签（`all/all`）+ 遮挡分层结果作为主判据**，警惕仅看 `R_max` 却伴随低质量框/掩码的“假召回涨点”。
   - `cls_head` 与 `box_head` 分流后，需盯紧 `AP75` 与 `FP/img` 是否协同同步上升；三模块有界残差需通过 `APEX06 → APEX07 → APEX03` 阶梯消融证实无负向互扰。
2. **服务器训练量化止损线**：
   - **主精度线**：前 10~30 轮验证集 `Mask AP50-95` 若显著落后基准且趋势发散、或后段尾 20 轮均值较 `APEX00` 低于 `-0.3%`，立即停止当前分支并核查梯度/饱和率；
   - **质量一致性线**：若 `R_max` 或 `Recall` 提升的同时 `Mask AP50-95`（或高阈值精度 `P@R`）反跌超 `0.5%` 或背景 FP 恶化超 `5%`，判定为低质虚假召回，回滚对应开关（优先检查 `use_backproj` 与 `use_tdc` 单开消融）；
   - **`min8` 底线护栏**：`min8` 子集评估 `Mask AP50-95` 不得比基准掉点超 `0.2%`（确保微小果召回增强不以牺牲中大果精度为代价）。

---

## 六、PR 曲线 $R_{\max}\approx 0.90$ 末端置零悬崖诊断与双 AI 辩论纠偏重设计

针对用户指出的 **PR 曲线在 `0.90` 出头直接归零并与横坐标重合** 现象，Argon 与 `deepseek-v4.1-flash-1017`、`intern-s2` 展开了多轮源码级核查与交叉辩论（且由 Argon 依托本仓库实测历史反驳并否决了外部 AI 的 2 处错误建议）：

### 6.1 两重根因诊断（图表表象 vs 模型物理失配）
1. **评估插值机制表象（`ultralytics/utils/metrics.py` L769–770）**：`compute_ap` 在经验最大召回点处强制追加哨兵点 `(recall[-1], 0.0)` 与 `(1.0, 0.0)`，横轴实为 `Recall` 而非置信度。因此任何 $R_{\max} < 1.0$ 都会在绘图末端呈现垂直落崖并在 $[R_{\max}, 1.0]$ 与横轴重合。
2. **极小果实物理漏检壁垒（`SAGE_V7_DESIGN.md` 对 153 个 $<16\times 16$ 极小果拆解）**：
   - **85 / 153（占比 55.6%）已有 `IoU >= 0.5` 原始框，但 `cls_head` 分类分数过低**而被 `conf` 阈值或绿叶假阳性挤出 `max_det=300`。源码排查确认：`SegmentCitrusBSD`（`M3`）虽已传入 `C2`（$160\times 160$），但原 `tdc_blocks[0]` 只读 `P3` 与 `P4`（stride 16 下 $<16\text{px}$ 果不足 1 像素），**分类塔 `cls_head[0]` 完全未获得 `C2` 高分辨率对比度校准**（尤其在无 `M2` 桥接的 `APEX06` 中完全缺失）。
   - **42 / 153（占比 27.5%）在 `P3` 上根本没有产生 `IoU >= 0.5` 原始框**。源码排查确认：原 `C2PhaseBridge` 门控仅看降采样冲刷后的 `P3`（`gate = ctx_gate(p3).sigmoid()`），当微果在 `P3` 上已被淹没时，门控直接把 `C2` 独有的微果信号一起关死！

### 6.2 Argon 依托仓库实证否决 DeepSeek 误判并确立最终无损重构
1. **否决改 `tal.py` (NWD) 与 `ops.py` (`crop_mask` 扩框)**：
   - DeepSeek 曾建议直接修改全局 `tal.py`（给 $<32\text{px}$ 加 NWD 软标签）和 `ops.py::crop_mask`（用 `floor/ceil` 强制扩大极小框）。
   - Argon 核查 [`E_V12_RECONSTRUCTION.md`](file:///C:/Users/777416/Downloads/code_apex/code/ultralytics-main-new/docs/E_V12_RECONSTRUCTION.md) 发现：`V11_05`（单改 tiny NWD/TAL）实测验证 AP 反而掉点 **`-0.525%`**（`71.960 → 71.435`），且去掉 NWD 反而升 `fine AP`、降背景误检；同时 `floor/ceil` 会令 `1.2×1.2` 极小框面积暴涨 `2.78` 倍混入绿叶且破坏与 `APEX00` 的同协议公平对比。辩论后 `deepseek-v4.1-flash-1017` 与 `intern-s2` 均认可撤回该改动，**100% 保持原生 `tal.py` 与 `ops.py` 不变**。
2. **否决 `PixelUnshuffle` 前置均值池化，改为无损后分解 `_subpixel_phase_decompose`（`SPD-Conv + PiDiNet` 思想）**：
   - 前置 `3×3` 均值池化会在 `160×160` 阶段把 $2\times 2$ 块内四个亚像素相位提前混合抹平；改为先执行无损 `F.pixel_unshuffle(z, 2)` 拆分为四相位 `[B, w, 4, H, W]`，提取零均值四相位差 `p_diff = phases - p_mean`（严格保留亚像素高频几何），仅对块内直流分量 `p_mean` 在 $80\times 80$ 上做 `3×3` 抑噪后再重组。
3. **落地的 `M2 + M3` 精准协同修补（已通过 10/10 `pytest` 与 `--check-only` 全检）**：
   - **`M2` `C2PhaseBridge` 联合门控**：输入改为 `cat(p3_proj(p3), _subpixel_phase_decompose(c2))` 共同开门，即使微果在 `P3` 被冲刷也能由 `C2` 亚像素脉冲独立打开门控，直击 `42/153` 无框漏检。
   - **`M3` `Tri-Scale C2-P3-P4 TDC`（严格只送 `cls_head`）**：将 `SegmentCitrusBSD.tdc_blocks[0]` 升级为融合 `[P3, P4_ref, C2_subpixel_phase]` 的三尺度差异校准（含 `APEX06` 在内的 `APEX01–08` 全系受益），并严格隔离在 `forward_head` 的 `cls_head` 通路，确保 `box_head / mask_head / proto` 零扰动，直击 `85/153` 低分抑制痛点。
