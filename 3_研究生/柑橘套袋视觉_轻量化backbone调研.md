# 柑橘套袋视觉 —— 轻量化 backbone 调研（论文②效率轴）

> **目标**：把 YOLO11n-seg 的 backbone 换成更轻的架构，降参数/FLOPs/延迟，用于套袋机器人边缘部署。
> **检索**：Semantic Scholar API，2026-07-12，均为真实命中（附 S2 链接）。
> **关联**：[`柑橘套袋视觉_完整研究执行计划.md`]（论文②）、[`柑橘套袋视觉_相关论文调研.md`]（论文①频域）、[`改进方向_基于001基线.md`]

---

## 0. 一句话 + 一个必须记住的权衡

- **你的 001 基线已是 nano、且召回偏低（漏检 34%）**。换更轻的 backbone → **精度大概率还会降**。
- 所以轻量化的正确目标是 **"少掉参数/FLOPs/延迟的同时尽量守住 mask mAP"**，追求**精度-成本曲线的拐点**，而不是一味求小。
- **建议**：论文②做"backbone 消融找拐点"，并**与论文①的精度模块（P2/频域）解耦**——否则轻 backbone 会把召回打穿。

---

## 1. 候选轻量 backbone（按适配你的场景排序，★=推荐先试）

| Backbone | 原文 | 出处 | 引用 | 特点 / 为什么适合 | fork 是否已有 |
|---|---|---|---|---|---|
| ★★★ **StarNet** | *Rewrite the Stars* (Xu Ma et al.) | **CVPR 2024** | **484** | star operation（逐元素乘）以极低算力换高表达；2024-25 农业 YOLO 大热 | ✅ **`starnet_depth.py` 已实现**（可复用） |
| ★★★ **FasterNet** | *Run, Don't Walk: Chasing Higher FLOPS* (Jierun Chen et al.) | **CVPR 2023** | **1878** | PConv 部分卷积，真快（高 FLOPS 利用率）；YOLO 落地案例最多、最稳 | ❌（模块小，易加） |
| ★★ **GhostNet / V2** | *GhostNet: More Features From Cheap Operations* (Kai Han et al.) | CVPR 2019 / NeurIPS 2022 | **4136** / 574 | cheap operations 生成冗余特征；农业 YOLO 轻量化的"标配"，参考最多 | ❌ |
| ★★ **MobileNetV4** | *MobileNetV4: Universal Models for the Mobile Ecosystem* (Qin et al.) | **ECCV 2024** | 618 | 最新移动端通用架构（UIB 模块）；对边缘硬件友好 | ✅ **`mobilenetv4_rgb.py` 已实现** |
| ★ MobileNetV3 | (经典) | — | — | 成熟稳，SE+hswish | ✅ `mobilenetv3_rgb.py` |
| ★ ShuffleNetV2 | (经典) | — | — | channel shuffle，极低算力 | ✅ `shufflenetv2_depth.py` |
| ○ **EfficientViT** | *EfficientViT: Memory Efficient ViT with Cascaded Group Attention* (Xinyu Liu et al.) | CVPR 2023 | 793 | CNN-Transformer 混合，注意力强但更复杂；想要 attention 再考虑 | ❌ |

**链接**：
- StarNet: https://www.semanticscholar.org/paper/c9b7d078835ec4ec583e79df394480efe5bcf76b
- FasterNet: https://www.semanticscholar.org/paper/a3aa1323a7f08c40207eaa359041e5bd72b25b27
- GhostNet: https://www.semanticscholar.org/paper/a4cc0701170331a1fd0e58bad962bd7f39f5efc9 ｜ GhostNetV2: https://www.semanticscholar.org/paper/3e420beb7f5d1bc370470b31908dd766ba35eedd
- MobileNetV4: https://www.semanticscholar.org/paper/3891db6b0adc2de204d89dceced1a739674340d6
- EfficientViT: https://www.semanticscholar.org/paper/9a83aeadc8db65fb6da39ec977360541cddaff5c

---

## 2. 应用佐证：这些 backbone 在 YOLO / 农业 / 分割里的落地（可对标）

**轻量 YOLO-seg（和你任务最近，YOLO11/12-seg 实例分割）**：
- ★ **GS-YOLO-Seg: A Lightweight Instance Segmentation Method ... Based on Improved YOLO11-Seg** (Sustainability 2025, 7) — https://www.semanticscholar.org/paper/c2c97e738058fa6d47a2b6ab5d5c1d787b2df9dc　**直接同框架，必读**
- PS-YOLO-seg: Lightweight ... YOLOv12-seg (J. Imaging 2025, 8) — https://www.semanticscholar.org/paper/5ab940b578a0de4d4d344236d79406125e632eee
- BHI-YOLO: Lightweight Instance Segmentation for Strawberry Diseases (Applied Sciences 2024, 11) — https://www.semanticscholar.org/paper/ff7f9b029b37a418be8984adaf3639bd7c4a6350
- YOLO-AppleSeg: Lightweight Apple Fruit Instance Segmentation (CVDL 2024, 4) — https://www.semanticscholar.org/paper/547c25efe4bc148046dc0983752aa6a47c93ed82

**FasterNet / StarNet 进 YOLO（证明可行）**：
- ★ A lightweight YOLOv8 integrating **FasterNet** for real-time underwater object detection (J. Real-Time Image Proc. 2024, **91**) — https://www.semanticscholar.org/paper/8194583828a336081e3109db1448428c65cfb491
- Lightweight Helmet-Wearing Detection Based on **StarNet-YOLOv10** (Processes 2025, 2) — https://www.semanticscholar.org/paper/56f11b24afe333461d070f947c32ba5d498cf3a8
- Lightweight Transformer and **Faster Convolution** for Efficient Strawberry Detection (Applied Sciences 2025) — https://www.semanticscholar.org/paper/e2025da6329d608206f77103ecfb2897b52e0234

**农业边缘端轻量 YOLO（应用背景/对标）**：
- A Lightweight YOLO-Based Architecture for Apple Detection on Embedded Systems (Agriculture 2025, 6) — https://www.semanticscholar.org/paper/9f8445de7a6e8ed68cb0cef74836b7d54c9b7984
- GAE-YOLO: lightweight multimodal detection for tomato smart agriculture with edge computing (Front. Plant Sci. 2025, 2) — https://www.semanticscholar.org/paper/6f43f6e92acefc3bab1235fdd7364391d2745bb4
- EdgeFormer-YOLO: Lightweight Multi-Attention for Real-Time Red-Fruit Detection in Complex Orchard (Mathematics 2025, 2) — https://www.semanticscholar.org/paper/10b3a2e9e7c2361b9949aed0dbb99cb2649dd31c
- Edge-YOLOv11: Lightweight UAV Fruit Detection in Dense Canopies (Smart Agric. Technol. 2026) — https://www.semanticscholar.org/paper/9b6392a2e9c1f0f64027217da5e93253ba21bf1c

**轻量 backbone 设计参考**：
- DecoupleNet: A Lightweight Backbone Network ... (IEEE TGRS 2024, 43) — https://www.semanticscholar.org/paper/f2145bfba485d4c4f523cf97d0043cec22087a34

---

## 3. 你 fork 里已有、可复用的 backbone（省大量代码）

`ultralytics/nn/modules/` 下（原为 RGB-D 双流苹果研究实现，均为**单流可复用**的 backbone 实现）：
- `starnet_depth.py` → **StarNet** ✅
- `mobilenetv4_rgb.py` → **MobileNetV4** ✅
- `mobilenetv3_rgb.py` → **MobileNetV3** ✅
- `shufflenetv2_depth.py` → **ShuffleNetV2** ✅

> 这意味着 **StarNet / MobileNetV4 你几乎不用重写**，只需把它们**作为单流 backbone 接进 YOLO11-seg 的 YAML**（改 `from` 索引、按 4 文件机制确认注册），再让 P3/P4/P5 出口对上 neck。FasterNet/GhostNet 需新写但模块很小。

---

## 4. 推荐落地路线

1. **先做 backbone 消融**（论文②主表）：`YOLO11n-seg(基线) vs +StarNet vs +FasterNet vs +GhostNet vs +MobileNetV4`，统一用 `eval_citrus_seg.py` 报 **Params / GFLOPs / FPS / mask mAP50-95**，画**精度-成本曲线**，取拐点。
2. **先试 StarNet（fork 已有）和 FasterNet（YOLO 落地最稳）** 两个，其余作对比。
3. **实现方式**（在 fork 内）：新建 `citrus_yaml/E-lite_*.yaml`，把 backbone 段换成对应轻量块 → 按 tasks.py 的 `base_modules`/`elif` 分支确认注册 → `YOLO(yaml).load('yolo11n-seg.pt')` 迁移可匹配层。
4. **守住召回**：轻 backbone 后若召回明显掉，配合论文①的 **P2 小目标头 / 频域先验** 补回——或严格把效率轴与精度轴分开写。
5. 参考 **GS-YOLO-Seg**（改进 YOLO11-seg 的轻量实例分割）的做法与指标口径。

---

## 5. 备注

- StarNet/FasterNet/GhostNet 都是**通用轻量 backbone**，单换 backbone**不构成论文级创新**——论文②的创新点应在"**面向套袋边缘部署的轻量化设计 + 精度保持策略**"，backbone 只是其中一环（配合剪枝/蒸馏/重参数化等）。
- 若追求"最新"，StarNet(CVPR24)、MobileNetV4(ECCV24) 是 2024 梯队；FasterNet(CVPR23)、EfficientViT(CVPR23) 是 2023 梯队但落地更成熟。
