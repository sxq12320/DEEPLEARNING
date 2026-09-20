# 柑橘幼果实例分割研究仓库

> 更新时间：2026-09-20。这里是项目唯一入口；服务器训练只复制"正式主线"中的代码和指定数据集。

## 正式主线

| 优先级 | 位置 | 用途 | 当前状态 |
|---:|---|---|---|
| 1 | [`1_SEVER/code/ultralytics-main-new/`](1_SEVER/code/ultralytics-main-new/) | **当前唯一活跃代码库**（服务器回传副本）；E/I 系列模型、训练、评估和批量实验 | 正式使用 |
| 2 | [`data/orange_yolo_grouped_dedup_20260820/`](data/orange_yolo_grouped_dedup_20260820/) | **当前唯一正式数据集**；965 图、5,890 实例、严格 group-aware split | 正式使用 |
| 3 | [`3_研究生/柑橘套袋视觉_完整研究执行计划.md`](3_研究生/柑橘套袋视觉_完整研究执行计划.md) | 研究问题、实验纪律与论文路线 | 研究依据 |
| 4 | [`4_baseline_choice/`](4_baseline_choice/) | 跨家族基线：YOLO、RTMDet-Ins、Mask R-CNN、RF-DETR、U-Net 等 | 对比实验 |

根目录的 `ultralytics-main-new/` 是 2026-08 下旬的旧开发副本，已不再是主线；
`1_SEVER/code/ultralytics-main-new/` 保存 Linux 服务器 `/data/sxq/` 的回传代码，所有新模型在这里开发并回传。
它是镜像：可以改逻辑，**不得改 `SERVER_*` 常量和 `/data/sxq/...` 路径**。

## 系列演进与当前状态

| 阶段 | 系列 | 结论 |
|---|---|---|
| 早期 | B/C/D/F/G/H/L/N/S/SXQ/T、G_0830/0839、Light、ORCHID | 历史消融库；多数指标为旧数据旧协议峰值，跨协议不可比 |
| 中期 | SAGE V2–V8、E V1–V8 | 切片/混合输入、控制论式零初始化有界修正方向确立 |
| 近期 | E V9–V12 | 持久 P2 细节通路 + 识别/定位任务分流。V12 六臂完成 300ep：04 共享路由 fine 栅格 AP50-95≈78.3%（内部口径最高），03 识别路由背景误报最少；**三审一致判定未达投稿成熟度**——val 成员与 grouped_dedup 不一致、单 seed、缺同协议官方基线 |
| 当前 | **I_V1** | 同步双原型掩膜解码：P4 语义原型 + 已验证细节原型 + 逐位置门控仲裁，全部零初始化。10 臂（2 个 V12 精确重放锚点 + 8 个隔离臂），42 项契约测试通过，待服务器筛选 |

审查证据：`1_SEVER/code/ultralytics-main-new/docs/I_V1_REVIEW_20260920/`；
I_V1 设计与停止判据：`docs/I_V1_DESIGN_20260920.md`。

## 训练入口（服务器 VS Code 前台模式）

```bash
cd ultralytics-main-new
pip install -e .
pytest -q tests/test_citrus_i_v1.py          # 契约测试：构建/反传/fuse/父模型重放
python 20260920_citrus_i_v1_batch.py --data <data.yaml> --suite all --dry-run
```

批量训练统一走 `RUN_CITRUS_<系列>.py` 前台入口（如 `RUN_CITRUS_I_V1.py`），改顶部
`DATA`/`DEVICE`/`SUITE`/`EPOCHS` 后点 VS Code ▶；`SUITE="priority"` 先跑 4 臂。
系列清单见 `citrus_foreground.py` 的 `RUNNERS` 与 `FOREGROUND_TRAINING_README.md`。

## 数据目录

| 位置 | 定义 | 是否用于正式训练 |
|---|---|---|
| `data/orange_yolo_grouped_dedup_20260820/` | 严格分组去重版；当前标准（676/193/96 = train/val/test） | **是** |
| `data/orange_wuxi/` | LabelMe 原始数据和原始素材 | 否；作为源数据保留 |
| `data/orange_yolo/` | 旧 YOLO 转换版（123/303 组跨 split 泄漏） | 否；历史复核 |
| `data/_backups/` | 清洗前备份 | 否 |
| `data/*.zip` | 传服务器/离线归档包 | 否 |

注意：V11/V12 服务器跑的 val 成员（193 图/1,049 实例）与 grouped_dedup（193 图/1,181 实例）不一致，
其内部排序可用于筛选，**正式论文结论必须在 grouped_dedup 上三 seed 重跑**。

## 历史与旁支

| 位置 | 说明 | 整理原则 |
|---|---|---|
| [`1_SEVER/results/`](1_SEVER/results/) | 服务器全部实验结果（按系列/协议分目录） | 不混入新表；不覆盖已完成 run |
| `1_SEVER/archive/` | 旧代码快照 | 只读追溯，不上传 |
| [`3_研究生/`](3_研究生/) | 研究计划、文献、定稿（`paper1_finalization_20260830/`） | 保留 |
| `2_catoon/`、`5_novels/` | 与论文主线无关的个人项目 | 独立旁支 |

## 实验纪律

1. 当前主消融基线是 YOLO11n-seg；同协议比较 mask mAP50-95、mAP50、P、R、AP by scale、Params、GFLOPs、实测延迟和难例子集。
2. 正式数据 split、输入尺寸、预训练、优化器、学习率、batch、AMP、seed 和评估 split 必须完全一致；协议见 `protocols/citrus_paper1_formal_v2_ram.yaml`。
3. 筛选实验可单次运行；最终基线和最终方法使用 3 个 seed（42/43/44），报告均值±标准差。
4. 已完成 run 永不覆盖；每次记录命令、Git 状态、数据版本和硬件。
5. 数据集、权重、结果图、压缩包和密钥不得提交到 Git。
