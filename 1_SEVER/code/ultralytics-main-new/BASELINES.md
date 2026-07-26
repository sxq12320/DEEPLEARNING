# 基线选型与执行手册（柑橘幼果实例分割）

> 解决"4_baseline_choice 太乱不知道怎么用"：本页是唯一入口。选型逻辑 → 统一口径 → 逐条命令 → 汇总表。
> 数据集统一用 `E:/mastercode/data/orange_yolo/`（group-aware split 676/193/96，burst 泄漏已修复）。

## 1. 选型逻辑（每个基线回答一个审稿人问题）

| # | 基线 | 回答的问题 | 必要性 | 框架 |
|---|---|---|---|---|
| B1 | **YOLO11n-seg** | 主消融基线（所有 F 系列的对照锚点） | ★核心 | 本 fork |
| B2 | YOLOv8n-seg | "比上一代强吗" | ★核心 | 本 fork |
| B3 | YOLO26n-seg | "比最新一代强吗" | ★核心 | 本 fork |
| B4 | RTMDet-Ins-tiny | "非 YOLO 的轻量一阶段呢" | ★核心 | mmdet |
| B5 | Mask R-CNN R50-FPN | "经典两阶段上限" | ★核心 | mmdet/detectron2 |
| B6 | RF-DETR Seg Nano | "Transformer 检测器呢" | ★核心 | rfdetr |
| B7 | U-Net + 分水岭 | "语义分割转实例的朴素方案" | ★辅助 | smp |
| B8 | SOLOv2-Light R18 | "无框位置式分割呢" | 期刊版补充 | mmdet |
| B9 | YOLO11s-seg | 精度上界参照（也是蒸馏教师） | 可选 | 本 fork |

已否决（有可引用理由，见文献档案）：Mamba 系（端侧部署困难）、CondInst/SparseInst（被 B8 替代）。

## 2. 统一口径（违反即不可比，写论文前自查）

1. **同一数据划分**：orange_yolo 的 group split；跨框架需 COCO JSON——用 `E:\mastercode\4_baseline_choice\scripts\build_grouped_citrus_cv.py` 导出，**禁止各框架自行重划**。
2. **同一输入**：640×640；**同一评测**：mask mAP50-95 / mAP50 / P / R / AP-small(<32²) + Params/GFLOPs/实测延迟。
3. 训练预算对齐：YOLO 系 300ep(patience100)；mmdet 系用其标准 schedule 并在表格中注明（不同框架学习率策略不可强行统一，注明即可，这是学界惯例）。
4. 筛选跑 1 seed；**最终表格：B1 与最终方法跑 3 seeds 报 mean±std**（AGENTS.md 纪律）。
5. 结果只进 `1_results/ORANGE_WUXI_SEG/results_summary.csv` 一张表，标注 split 版本，**永不混口径**。

## 3. YOLO 系（本 fork 内直接跑，现在就能执行）

```bash
cd 1_SEVER/code/ultralytics-main-new && pip install -e .
# B1 主基线（3 seeds 之第一枚；--seed 需改 train_citrus_seg.py 的 SEED 或复制脚本，默认 42）
python train_citrus_seg.py --model yolo11n-seg.pt  --name B1_yolo11n_seg_s42
# B2 / B3
python train_citrus_seg.py --model yolov8n-seg.pt  --name B2_yolov8n_seg
python train_citrus_seg.py --model yolo26n-seg.pt  --name B3_yolo26n_seg
# B9 上界/蒸馏教师
python train_citrus_seg.py --model yolo11s-seg.pt  --name B9_yolo11s_seg
# 评测统一入口（追加进 results_summary.csv）
python eval_citrus_seg.py --weights 1_results/ORANGE_WUXI_SEG/B1_yolo11n_seg_s42/weights/best.pt
```
注意：`train_citrus_seg.py` 的 `DATA` 常量当前指向 `data/test/orange_wuxi_seg.yaml`（旧预实验路径）；
正式跑 B 系列前把它切到 `data/orange_yolo/data.yaml`（改一行常量或加 `--data` 旗标，**改前确认服务器路径**）。

## 4. 跨框架（在 `E:\mastercode\4_baseline_choice\` 内，配置中心 `configs/baselines.yaml`）

| 基线 | 入口 | 前置 |
|---|---|---|
| B4 RTMDet-Ins-tiny | `python run_mmdet.py --model rtmdet-ins_tiny` | mmdet 环境 + COCO json |
| B5 Mask R-CNN | `python run_maskrcnn.py`（或 detectron2-main） | 同上 |
| B6 RF-DETR | `python run_rfdetr.py` | rfdetr 包 + COCO json |
| B7 U-Net+分水岭 | `python run_unet.py`（smp 实现；实例掩码合并为前景训练，分水岭切分，Dice/mIoU + 实例 mAP 都要报） | COCO json |
| B8 SOLOv2-Light | `python run_mmdet.py --model solov2_light_r18` | mmdet 环境 |

执行顺序建议：**先 B1-B3+B9（本机/服务器即可）→ 50ep 筛选 B4 对比 B1（AGENTS.md 要求的主基线复核）→ 再补 B5-B8**。

## 5. 汇总表模板（论文表 1 直接用）

| Model | Params(M) | GFLOPs | Latency(ms) | mAP50-95(mask) | mAP50 | AP-small | P | R |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| YOLOv8n-seg | | | | | | | | |
| YOLO11n-seg (B1) | 2.84 | 10.2 | 6.6 | 0.642* | | | | |
| YOLO26n-seg | | | | | | | | |
| RTMDet-Ins-tiny | | | | | | | | |
| Mask R-CNN R50 | | | | | | | | |
| RF-DETR Seg Nano | | | | | | | | |
| U-Net+WS | | | | | | | | |
| **CitrusFormer-Plus (F53, ours)** | 2.76 | 14.1 | | | | | | |
| **CitrusFar-Edge-V2 (F52, ours)** | 2.17 | 15.1 | | | | | | |

*旧口径预实验数值，正式表须在新 split 重跑。
