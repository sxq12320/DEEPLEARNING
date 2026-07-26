# Theme 13：大模型时代架构思想 → 轻量 CNN（YOLO11n-seg 柑橘分割）文献核验

核验日期：2026-07-26。所有 arXiv ID 均已通过 arXiv API（export.arxiv.org）逐条实测验证，非 arXiv 条目通过 CVF Open Access / ICLR proceedings 官方页面核验。

---

## 任务 A："换个方向的残差连接"归属核验

### 结论：用户记忆基本准确，指的是月之暗面 **Attention Residuals (AttnRes)**

**Kimi Team (Moonshot AI / 月之暗面), "Attention Residuals", arXiv:2603.15031 (2026-03-16)**
- 作者：Kimi Team — Guangyu Chen, Yu Zhang, **Jianlin Su (苏剑林)**, Weixin Xu 等（已由 arXiv API 验证；第一作者 Guangyu Chen 为 17 岁高中生，曾获马斯克转发点赞，国内报道广泛）
- 代码：github.com/MoonshotAI/Attention-Residuals
- **一句话机制**：把标准残差连接"沿深度方向的固定单位权重累加"替换为"沿深度方向的可学习注意力"——即注意力从 token 方向（时间）转到 **layer 方向（深度）**，模型按内容自主决定从之前哪些层的输出中检索、以多大权重融合，缓解 PreNorm 深层稀释。这正是"换了个方向的残差连接/注意力"这一记忆的出处。<2% 额外计算，已集成进 Kimi Linear 48B-A3B 验证。

### 候选逐一核验（全部真实存在，ID 无误）

| 工作 | 准确引用 | 机构/Venue（已核验） | 机制一句话 | 是否用户所指 |
|---|---|---|---|---|
| **Attention Residuals** | **arXiv:2603.15031** (2026-03) | **Moonshot AI (Kimi Team)**；技术报告 | 深度方向注意力替代固定残差累加 | **是（最可能）** |
| Hyper-Connections | arXiv:2409.19606 (2024-09) | **ByteDance Seed**（Defa Zhu, Yutao Zeng 等）；**ICLR 2025 poster**（proceedings.iclr.cc 已核验，用户记的机构与 venue 均正确） | 把单条残差流扩成 n 条可学习加权的并行流（宽度方向扩展残差） | 否，但为同一谱系开山作 |
| mHC | arXiv:2512.24880 (2025-12-31) | **DeepSeek**（Zhenda Xie 等） | 把 HC 的可学习残差矩阵约束到双随机流形上恢复恒等映射性质，稳定超大规模训练 | 否（用户可能与 DeepSeek 混淆过） |
| Value Residual Learning (ResFormer) | arXiv:2410.17897 (2024-10, v5) | Zhanchao Zhou 等（西湖大学系）；venue 未在本次核验中确认，建议引 arXiv | 值向量 V 的跨层残差：深层 attention 的 value 与首层 value 做残差 | 否 |
| LAuReL | arXiv:2411.07501 (2024-11, v4) | **Google**（Gaurav Menghani, Ravi Kumar, Sanjiv Kumar） | 残差 `x+f(x)` 推广为 `αf(x)+g(x, 历史激活)` 的可学习低秩版本 | 否 |
| DyT (Transformers without Normalization) | arXiv:2503.10622；**CVPR 2025**（CVF Open Access 已核验） | Meta FAIR + NYU + MIT（Zhu, Chen, Kaiming He, LeCun, Zhuang Liu） | 用逐元素 tanh(αx) 替换 LayerNorm——是归一化改造，**不是残差改造** | 否 |
| Muon | arXiv:2502.16982 (2025-02) "Muon is Scalable for LLM Training" | **Moonshot AI + UCLA**（Jingyuan Liu, Jianlin Su 等） | 优化器（矩阵正交化动量），与残差连接无关 | 否 |
| MoBA | arXiv:2502.13189 (2025-02) | Moonshot AI + 清华 | 块稀疏注意力，无残差改造 | 否 |
| Kimi K2 | arXiv:2507.20534 (2025-07) | Moonshot AI | MuonClip + 超稀疏 MoE，残差为标准形式 | 否 |
| Kimi Linear | arXiv:2510.26692 (2025-10) | Moonshot AI | KDA 混合线性注意力；本身无残差改造（但后来 AttnRes 集成于其上） | 否 |
| Kimi k1.5 | arXiv:2501.12599 (2025-01) | Moonshot AI | RL scaling，无残差改造 | 否 |

**谱系链（写进论文 related work 很顺）**：ResNet 残差 → Hyper-Connections (ByteDance, ICLR 2025) 把残差扩成多流 → mHC (DeepSeek, 2025-12) 加流形约束 → **AttnRes (Moonshot, 2026-03) 把深度方向累加换成深度方向注意力**。迁移到 YOLO11n-seg 的落点：把 backbone/neck 的 C3k2 内部或跨 stage 的 concat/add 换成轻量可学习多流连接（HC 的静态版本足够，nano 级慎用动态版）。

---

## 任务 B：MoE → CNN 血脉核验（8 篇核心 + 2 篇近作）

### 主链（全部 arXiv API 实测验证）

1. **Shazeer et al., "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer", ICLR 2017, arXiv:1701.06538**（Google Brain，含 Hinton、Quoc Le）。稀疏门控 top-k 硬路由 MoE 奠基。
2. **CondConv — Yang, Bender, Le, Ngiam, NeurIPS 2019, arXiv:1904.04971**（Google Brain）。关键转折：把 MoE 从"路由到不同专家分支"改为"**对 n 个卷积核做逐样本加权求和**"——软路由在**权重空间**融合，推理仍是一次卷积，天然可导出。
3. **Dynamic Convolution — Chen et al., CVPR 2020, arXiv:1912.03458**（Microsoft）。CondConv 同思路 + 温度退火 softmax 注意力聚合 K 个核，明确面向小模型（MobileNet 级）涨点。
4. **ODConv — Li, Zhou, Yao, ICLR 2022, arXiv:2209.07947**（Intel）。动态卷积推广到核数/空间/输入通道/输出通道四个维度的注意力，是"每维一组专家"的极致软路由。
5. **V-MoE — Riquelme et al., NeurIPS 2021, arXiv:2106.05974**（Google Brain）。稀疏 top-k MoE 进 ViT（15B），证明视觉稀疏路由可行，但为服务器级。
6. **AdaMV-MoE — T. Chen et al., ICCV 2023, pp. 17346-17357**（CVF Open Access 核验；**无 arXiv 预印本**，引 ICCV 版）。多任务视觉 MoE，按任务自适应决定激活专家数，覆盖检测与实例分割。
7. **Soft MoE — Puigcerver et al., ICLR 2024, arXiv:2308.00951**（Google DeepMind）。"From Sparse to Soft Mixtures of Experts"：全可微软槽位路由，**论文明确讨论了硬 top-k 路由的训练不稳/丢 token 问题**——是"软路由 vs top-k 硬路由"论述的最佳引用。
8. **Mr. DETR++ — Zhang, Zhong, Han, arXiv:2412.10028 (v4)**（CVPR 2025 系检测 Transformer + MoE 多路训练近作）。

### 近作 / 新颖性对照（重要）

9. **YOLO-Master — Lin et al., arXiv:2512.23273 (2025-12)**。**目前与我们最接近的先例**：在 YOLO backbone/neck 里放 ES-MoE 模块（Transformer 专家），动态路由 + softmax 门控，训练软 top-k / 推理硬 top-k 切换。差异点：(a) 它是通用 RTOD、专家是 Transformer 块而非 DW 卷积；(b) 不按**成像条件**（光照/逆光/遮挡/雨雾）语义化路由；(c) 非 nano 级农业边缘分割。
10. 农业轻量 YOLO 近作（GAE-YOLO、YOLO-PLNet、Edge-YOLOv11 等 2025-2026 Frontiers/ScienceDirect 一批）均为静态结构裁剪/注意力改造，**未发现任何"按成像条件路由多个 DW 卷积专家"的农业边缘检测/分割先例**。

### 软路由 vs top-k 硬路由的部署可导出性（写作要点）

- 硬 top-k（Shazeer/V-MoE/YOLO-Master 推理态）：数据相关控制流，ONNX/TensorRT 导出需自定义算子或 gather 技巧，batch 内负载不均。
- 软路由权重空间融合（CondConv/DynamicConv/ODConv）：门控输出只做核加权求和，导出后就是"conv + 少量逐元素乘加"，**边缘端零障碍**——这是我们 4 专家 DW 卷积应选软路由（或训练软/推理蒸馏为软）的文献依据；Soft MoE (arXiv:2308.00951) 提供 Transformer 侧的同方向论证。

### 我们 MoCE（4 个 DW 卷积专家按成像条件路由）新颖性判断

- 组件均有出处：专家=DW 动态卷积（CondConv/DynamicConv 谱系）、条件路由（Shazeer 谱系）、YOLO+MoE（YOLO-Master 已做）。
- **组合缝隙真实存在**：nano 级实例分割 + 农业成像条件语义先验作为路由信号 + 全软路由保导出性，三者交集未检索到先例。可声称"组合创新 + 面向边缘可导出的条件语义路由设计"，**不可**声称首创 MoE-in-YOLO（须引 YOLO-Master 并对比）。

---

## 引用一致性备注
- 所有 arXiv ID 已于 2026-07-26 经 arXiv API 逐条返回验证（标题/作者/日期与上表一致）。
- AdaMV-MoE 无 arXiv 号，务必引 ICCV 2023 正式版，不要编造 arXiv ID。
- Value Residual Learning 的正式 venue 本次未确认，投稿前建议引 arXiv 版或再查 ACL Anthology。
