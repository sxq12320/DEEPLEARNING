# 柑橘套袋视觉研究仓库

> 硕士课题「柑橘套袋视觉」的代码与文献仓库，支撑两篇关联小论文：
>
> 1. **论文一（当前优先）**：RGB 幼果实例分割 — 轻量化 + 高精度
> 2. **论文二**：基于论文一 ROI 的柑橘果梗点精准定位
>
> 研究计划唯一来源：[`3_研究生/柑橘套袋视觉_完整研究执行计划.md`](3_研究生/柑橘套袋视觉_完整研究执行计划.md)（版本 2026-07-13）
> 仓库规则：[`AGENTS.md`](AGENTS.md)

---

## 1. 顶层目录

| 目录 | 作用 |
|---|---|
| [`1_SEVER/code/ultralytics-main-new/`](1_SEVER/code/ultralytics-main-new/) | **活跃代码库**，定制版 Ultralytics fork，柑橘训练/评估/审查脚本均在此 |
| [`1_SEVER/code/baseline_choice/`](1_SEVER/code/baseline_choice/) | 跨框架 baseline 启动包（YOLO / mmdet / torchvision Mask R-CNN / RF-DETR / U-Net） |
| [`3_研究生/`](3_研究生/) | 研究计划、历史资料、对话记忆 |
| [`4_baseline_choice/`](4_baseline_choice/) | 独立 baseline 对比工程（含 vendored `detectron2-main/`、`UNet_server_package/`、统一 `baselines.yaml`） |
| `2_catoon/` | 早期学习代码（LeNet），与主线无关 |

> ⚠️ `2_catoon/` 与其它非主线目录是独立的遗留/侧项目，柑橘实验期间不重构。

---

## 2. 活跃代码库

路径：[`1_SEVER/code/ultralytics-main-new/`](1_SEVER/code/ultralytics-main-new/)

### 2.1 关键入口

| 文件 | 作用 |
|---|---|
| [`train_citrus_seg.py`](1_SEVER/code/ultralytics-main-new/train_citrus_seg.py) | 训练驱动，固定协议（AdamW / lr0=0.01 / dropout=0.0 / seed=42 / amp=0 / deterministic / imgsz=640 / batch=4 / epochs=300 / patience=100） |
| [`eval_citrus_seg.py`](1_SEVER/code/ultralytics-main-new/eval_citrus_seg.py) | 评估驱动，统一写入 `1_results/ORANGE_WUXI_SEG/results_summary.csv` |
| [`audit_yolo_seg_labels.py`](1_SEVER/code/ultralytics-main-new/audit_yolo_seg_labels.py) | 漏标审查，输出原图/GT/预测三栏对比，已发现 483 张数量不一致 |
| [`compute_difficulty_attrs.py`](1_SEVER/code/ultralytics-main-new/compute_difficulty_attrs.py) | 数据难度属性脚本（solidity / 凸包缺口 / 相邻距离 / scale-span / truncated） |
| [`convert_orange_wuxi_to_yolo.py`](1_SEVER/code/ultralytics-main-new/convert_orange_wuxi_to_yolo.py) | 数据集转换为 YOLO 格式 |
| [`vis_pred_vs_gt.py`](1_SEVER/code/ultralytics-main-new/vis_pred_vs_gt.py) | 预测与 GT 可视化对比 |
| [`200_orange_wuxi_seg.yaml`](1_SEVER/code/ultralytics-main-new/200_orange_wuxi_seg.yaml) | 当前数据集 YAML |

### 2.2 柑橘模型 YAML

位于 [`0_orange_yaml/`](1_SEVER/code/ultralytics-main-new/0_orange_yaml/)：

| YAML | 说明 |
|---|---|
| `001_1_yolov8-seg.yaml` ~ `001_5_yolo26-seg.yaml` | 旧代/当前/最新 YOLO-Seg 跨代对比 |
| `002_yolo11-seg-starnet-official-s1.yaml` / `s2.yaml` | **官方结构版 StarNet**（替代旧版 `002_yolo11-seg-starnet.yaml`） |
| `003_yolo11-seg-mobilenetv4.yaml` | MobileNetV4 负结果 |
| `004_yolo11-seg-mano.yaml` | MANO 实验 |

### 2.3 自定义模块

位于 [`ultralytics/nn/modules/`](1_SEVER/code/ultralytics-main-new/ultralytics/nn/modules/)：

- `starnet.py` — 官方 7×7 depthwise StarNet block（已注册到 `__init__.py` 与 `tasks.py`）
- `mobilenetv4_rgb.py` / `mobilenetv3_rgb.py` / `shufflenetv2_depth.py`
- `mano.py` / `lscon.py` / `custom_blocks.py`
- `scale_aware_fusion.py` / `rgbd_fusion_neck.py` / `smc_scheduler.py` / `smcao_v22_scheduler.py`

> 新增 YOLO 模块的注册流程见 [`AGENTS.md`](AGENTS.md)「Coding and Model Integration」节。

### 2.4 训练 / 评估命令

```powershell
cd E:\mastercode\ultralytics-main-new
pip install -e .

# E0 主 baseline
python train_citrus_seg.py --model yolo11n-seg.pt --name E0_yolo11n_seg_baseline_941

# 改进架构（从 YAML 加载并迁移匹配的 COCO 权重）
python train_citrus_seg.py --model citrus_yaml/E1_yolo11n_seg_p2.yaml \
    --pretrained yolo11n-seg.pt --name E1_p2_head

# 3-epoch smoke
python train_citrus_seg.py --model yolo11n-seg.pt --name E0_smoke --epochs 3

# 评估（追加一行到 results_summary.csv）
python eval_citrus_seg.py --weights 1_results\ORANGE_WUXI_SEG\<run>\weights\best.pt
```

> 大型实验前必须先做 1-3 epoch smoke 和 `pytest tests`。

---

## 3. 数据集

| 项 | 值 |
|---|---|
| 路径（Windows 开发机） | `E:/mastercode/data/test/` |
| 图像数 | 941 张 RGB |
| 实例数 | 4,576 个幼果 |
| 当前 split（**预实验，有泄漏**） | train 659 / val 188 / test 94 |
| 批次 | 2023 批 116 张（亮果、密集、连拍多）；2026 批 825 张（深绿哑光、与叶片同色） |

### 关键标签统计（写入论文动机）

- 每图平均约 4.8–5.2 实例，单图最多 35 个
- 640 输入下：34.9%–40.5% 实例至少一边 <32 px；11.3%–12.8% <16 px
- 47.0%–58.5% 图像存在相邻/近粘连实例；38.7%–46.8% 存在外接框重叠
- 24.0% 实例 `solidity < 0.90`；7.5% `< 0.80`
- 单图最大/最小面积比中位数约 6.0，33.6% 图像 >10，13.2% >25，最大约 228

### 数据泄漏警告

现有划分把同一 burst 序列（如 `IMG20231120161107_BURST*`）拆到 train/val/test 不同集合。因此 001-003 结果只能作为预实验，正式实验前必须完成 group-aware split（见 [`4_baseline_choice/scripts/build_grouped_citrus_cv.py`](4_baseline_choice/scripts/build_grouped_citrus_cv.py)）。

---

## 4. 预实验结果（不可与新结果混表）

| 编号 | 模型 | Params | GFLOPs | 推理 | Mask mAP50-95 | 备注 |
|---|---|---:|---:|---:|---:|---|
| 001 | YOLO11n-Seg | 2.835M | 10.2 | 6.6 ms | **0.642** | 当前精度基线 |
| 002 | StarNet-YOLO11n-Seg | 2.261M | 8.4 | 5.6 ms | 0.612 | 过度压缩负对照 |
| 003 | MobileNetV4-YOLO11n-Seg | 3.675M | 11.7 | 12.3 ms | 0.606 | 已放弃 |

001-003 使用 `lr0=0.001、dropout=0.1、从 YAML 训练`；新 `train_citrus_seg.py` 使用 `lr0=0.01、dropout=0.0、COCO 预训练`。**两套口径不可放入同一张表**。

---

## 5. 跨家族 baseline 矩阵

配置中心：[`4_baseline_choice/configs/baselines.yaml`](4_baseline_choice/configs/baselines.yaml)

| 角色 | 模型 | 必做 |
|---|---|---|
| 旧代轻量 | YOLOv8n-Seg | ✅ 核心 |
| 主消融基线 | YOLO11n-Seg | ✅ 核心（primary） |
| 当前强对照 | YOLO26n-Seg | ✅ 核心 |
| 非 YOLO 轻量一阶段 | RTMDet-Ins-tiny | ✅ 核心 |
| 经典两阶段 | Mask R-CNN R50-FPN | ✅ 核心 |
| 当前 Transformer | RF-DETR Seg Nano | ✅ 核心 |
| 无框位置分割 | SOLOv2-Light R18-FPN | 期刊版 |
| 经典语义转实例 | U-Net + Watershed | ✅ 辅助 |
| 语义补充（二选一） | DeepLabV3+ 或 SegFormer-B0 + Watershed | 可选 |
| 精度上界 | YOLO11s-Seg | 可选 |
| 过度压缩负对照 | 全 StarNet backbone | 已完成 |

执行入口：
- [`4_baseline_choice/run_yolo_baselines.py`](4_baseline_choice/run_yolo_baselines.py)
- [`4_baseline_choice/run_mmdet.py`](4_baseline_choice/run_mmdet.py)
- [`4_baseline_choice/run_maskrcnn.py`](4_baseline_choice/run_maskrcnn.py)
- [`4_baseline_choice/run_rfdetr.py`](4_baseline_choice/run_rfdetr.py)
- [`4_baseline_choice/run_unet.py`](4_baseline_choice/run_unet.py)

---

## 6. 论文方法规划（CitrusTopo-Seg）

**推荐题目**：面向遮挡-接触拓扑冲突的轻量化柑橘幼果实例分割方法

### 创新点

1. **BPSC**（Boundary-Preserving Stage-wise Compression）— 分阶段保边轻量化
   - 保留 P2/P3 浅层细节，仅在 P4/P5 或 neck 深层使用 StarBlock/深度可分离
   - 目标：Params / GFLOPs 至少降 10%，实测延迟不劣于 YOLO11n-Seg
2. **COB Loss**（Concave Occlusion Boundary Loss）— 凹陷遮挡边界损失
   - 在凸包缺口对应的深凹边界上计算漏分/外溢，仅训练期开销
3. **AIE Loss**（Adjacent Instance Exclusivity Loss）— 相邻实例排他损失
   - 在接触走廊惩罚 cross-instance leakage 与桥接

### 新增可解释指标

- `Concave-BF1` — 凹陷遮挡边界 F1
- `Gap-Preservation` — 相邻间隙被正确保留的实例对比例
- `Split/Merge Error` — 错分/合并/跨实例泄漏比例

### 难例子集

`small` / `dense` / `adjacent-pair` / `concave-occlusion` / `scale-span` / `truncated` / `cross-batch`

### 实验矩阵

完整 E0–E4 与 B0–B6 矩阵见研究计划 [§7](3_研究生/柑橘套袋视觉_完整研究执行计划.md)。

---

## 7. 立即执行清单

1. 修复数据划分泄漏，生成带版本号的新 dataset YAML
2. 将同一 group-aware 划分导出为 YOLO 与 COCO JSON，供 Ultralytics / MMDetection / RF-DETR / U-Net 共用
3. 统一 `train_citrus_seg.py` 与正式实验协议（已基本完成）
4. 在新划分上完成 YOLO11n-Seg 与 RTMDet-Ins-tiny 50 epoch 筛选
5. 完成 6 个核心 baseline 配置；期刊版补充 SOLOv2-Light R18-FPN
6. 编写统一数据难度脚本，输出 solidity / 凸包缺口 / 相邻距离 / scale-span / truncated
7. 人工复核低 solidity 样本，冻结 `concave-occlusion` 测试子集
8. 实现 BPSC → COB Loss → AIE Loss，每步先 build / forward / backward / FLOPs / 3-epoch smoke
9. 每完成一组实验立即更新统一结果表，不手工复制不同口径的数字

---

## 8. 纪律

- **不提交**：数据集、权重、`runs/`、大尺寸结果图、归档、视频
- 实验名保持编号、**永不覆盖**已完成的 run
- 每次论文实验记录：精确命令、Git 状态、数据 split 版本、硬件、最终指标
- 不还原无关的用户变更；提交信息用 `citrus: ...` 等作用域前缀
- 未做 Jetson/边缘设备测试时，只能称 "lightweight" / "efficient"，不得声称已完成边缘部署
