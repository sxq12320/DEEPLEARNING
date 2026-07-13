# 柑橘套袋视觉 —— 轻量化 backbone 调研与接入记录

> **定位**：论文②（轻量化边缘端实时推理轴）的 backbone 选型与集成记录。
> **日期**：2026-07-12 ｜ 基座：YOLO11n-seg（001 基线）｜ 数据：`data/test`（941 张，单类）
> **关联**：[`柑橘套袋视觉_完整研究执行计划.md`]、[`柑橘套袋视觉_难点与创新点分析.md`]、`ultralytics-main-new/1_results/001_yolo11n_seg_AdamW/改进方向_基于001基线.md`

---

## 0. 一句话结论

> **StarNet 真的更轻（-21% 参数 / -20% FLOPs），已接入并验证可训；MobileNetV4-conv_m 反而更重（+28% 参数）**——因为 yolo11n 本身已极小，塞进中号移动 backbone 会超。**轻量化轴优先用 StarNet**；MobileNetV4 作为"换 backbone 精度对比"更合适。

---

## 1. 接入结果（已验证：build ✓ / 前向 ✓ / 反向 ✓）

两个模型 YAML 已放入 `ultralytics-main-new/0_orange_yaml/`，命名延续 `NNN_` 规范：

| 编号 | 文件 | backbone | 参数量 | GFLOPs@640 | 相对基线 |
|---|---|---|---|---|---|
| 001 | `001_yolo11-seg.yaml` | 原版 YOLO11n | 2.877M | 10.53 | 基线 |
| **002** | **`002_yolo11-seg-starnet.yaml`** | **StarNet** (CVPR'24) | **2.267M** | **8.45** | **-21% / -20%** ✅ |
| 003 | `003_yolo11-seg-mobilenetv4.yaml` | MobileNetV4-conv_m (ECCV'24) | 3.680M | 11.81 | +28% / +12% ❌ |

**接入方式**：只换 backbone，保留 YOLO11 的 neck（PAN）+ Segment 头。backbone 模块本已在 fork 里实现且注册（`starnet_depth.py` / `mobilenetv4_rgb.py` + `tasks.py` parse_model 分支），但**此前从未在任何 YAML 里接过**——本次是首次接线并逐项验证（实例化、640 前向、反向梯度流均通过）。

**分辨率/通道流**：
- StarNet：`Conv(P1/2) → StarNetStem(4×→P3/8, 32ch) → StarNetStage(P4/16, 64ch) → StarNetStage(P5/32, 128ch)`
- MobileNetV4：`Stem(P1/2) → Stage32(P2/4) → Stage64(P3/8) → Stage96(P4/16) → Stage128(P5/32)`

⚠️ **MobileNetV4 为何更重**：fork 只注册了 `conv_m`/`conv_l` 两档（都偏大），没有更小的 `conv_s`。若一定要用更轻的 MobileNetV4，需在 `mobilenetv4_rgb.py` 里补一个 conv_s 配置，或收窄 neck。

---

## 2. 如何训练（与 001 对齐、干净对比）

001 基线是从 YAML **从头训练**（非预训练）。002/003 同样从头训练——backbone 自定义部分本就没有预训练权重，三者都 scratch，**对比公平**。

复制 `train_orange_wuxi_yolo11n_seg.py`，只改 model 路径即可（其余超参锁定与 001 一致：AdamW / 300ep / imgsz640 / batch4 / seed42）：

```python
yolo = YOLO(r"E:\mastercode\ultralytics-main-new\0_orange_yaml\002_yolo11-seg-starnet.yaml")
yolo.train(data=r"E:\mastercode\ultralytics-main-new\200orange_wuxi_seg.yaml",
           project=r"...\1_results", name="002_yolo11n_seg_starnet",
           optimizer="AdamW", epochs=300, imgsz=640, batch=4, seed=42, amp=0, device=0)
```

训完用 `vis_pred_vs_gt.py` 复查召回，用 `eval_citrus_seg.py` 统一评测，指标与 001 并列。

---

## 3. 轻量化 backbone 调研（landscape，供论文②对比与选型）

### 3.1 通用轻量 backbone（可换进 YOLO）
- **StarNet**（Rewrite the Stars, CVPR 2024）——星型操作 `(W₁x)⊙(W₂x)` 隐式高维映射，极省算力。**本次已接入，真更轻**。
- **FasterNet**（PConv 部分卷积, CVPR 2023）——农业/水下 YOLO 常用轻量替换，如 [轻量 YOLOv8+FasterNet 水下实时检测](https://www.semanticscholar.org/paper/8194583828a336081e3109db1448428c65cfb491)（J. Real-Time Image Proc., 91 引）。
- **MobileNetV4**（ECCV 2024，UIB 通用倒残差）——[MobileNetV4+注意力智能感知](https://www.semanticscholar.org/paper/ce0c1adf7b9e4cb08feaebc880fb9e79d72b70a5)。本次 conv_m 偏重。
- **GhostNet / EfficientViT / MobileNetV3** ——经典轻量族，YOLO 农业改进里高频出现。
- [DecoupleNet](https://www.semanticscholar.org/paper/f2145bfba485d4c4f523cf97d0043cec22087a34)（TGRS 2024，43 引）——特征解耦轻量 backbone。

### 3.2 农业/果实 YOLO 轻量化（应用对标，论文②对比方法）
- [StarNet-YOLOv10 安全帽检测](https://www.semanticscholar.org/paper/56f11b24afe333461d070f947c32ba5d498cf3a8)（Processes 2025）——StarNet 接 YOLO 的现成范例。
- [GAE-YOLO 番茄边缘计算多模态](https://www.semanticscholar.org/paper/6f43f6e92acefc3bab1235fdd7364391d2745bb4)（Frontiers in Plant Science 2025）。
- [EdgeFormer-YOLO 果园红果实时检测](https://www.semanticscholar.org/paper/10b3a2e9e7c2361b9949aed0dbb99cb2649dd31c)（Mathematics 2025）。
- [Edge-YOLOv11 密集树冠无人机果实检测](https://www.semanticscholar.org/paper/9b6392a2e9c1f0f64027217da5e93253ba21bf1c)（Smart Agricultural Technology 2026）。
- [轻量 YOLO 苹果嵌入式检测](https://www.semanticscholar.org/paper/9f8445de7a6e8ed68cb0cef74836b7d54c9b7984)（Agriculture 2025）。

### 3.3 轻量实例分割（与本课题任务最贴）
- [GS-YOLO-Seg 改进 YOLO11-seg](https://www.semanticscholar.org/paper/c2c97e738058fa6d47a2b6ab5d5c1d787b2df9dc)（Sustainability 2025）——直接改 YOLO11-seg 轻量化，最贴你的基座。
- [PS-YOLO-seg 改进 YOLOv12-seg](https://www.semanticscholar.org/paper/5ab940b578a0de4d4d344236d79406125e632eee)（Journal of Imaging 2025）。
- [YOLO-AppleSeg 轻量苹果实例分割](https://www.semanticscholar.org/paper/547c25efe4bc148046dc0983752aa6a47c93ed82)（CVDL 2024）。
- [BHI-YOLO 草莓病害实例分割](https://www.semanticscholar.org/paper/ff7f9b029b37a418be8984adaf3639bd7c4a6350)（Applied Sciences 2024，11 引）。

---

## 4. 重要权衡与建议

1. **轻量化 ≠ 免费**：001 基线召回已偏低（漏检 34%）。换更轻的 backbone**大概率精度再降**——论文②的正确叙事是"**在可接受精度损失下大幅降本/提速**"，要报 精度-参数-FLOPs-FPS 的**权衡曲线**，而非单点。
2. **两轴分工**：精度轴（论文①，频域/P2/注意力，见难点分析）先把精度做上去；轻量轴（论文②，StarNet 等）在此之上做效率。别把两件事混在一次实验里。
3. **推荐路径**：
   - 论文② 主力：**002 StarNet**（真轻量）+ 可加重参/蒸馏找回精度。
   - 003 MobileNetV4：留作"换 backbone 精度对比"或"更重但更准"的一档，不作轻量主打；若要轻量版需补 conv_s。
4. **下一步**：先训 **002 StarNet**，与 001 并列比 精度/参数/FLOPs/FPS；再决定是否补蒸馏或收窄 neck。

---

## 附：本次改动清单
- 新增 `0_orange_yaml/002_yolo11-seg-starnet.yaml`
- 新增 `0_orange_yaml/003_yolo11-seg-mobilenetv4.yaml`
- 均已验证 build/前向/反向通过（参数与 FLOPs 见 §1 表）
