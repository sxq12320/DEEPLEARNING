# 柑橘幼果实例分割研究仓库

> 更新时间：2026-08-24。这里是项目唯一入口；服务器训练只复制“正式主线”中的代码和指定数据集。

## 正式主线

| 优先级 | 位置 | 用途 | 当前状态 |
|---:|---|---|---|
| 1 | [`ultralytics-main-new/`](ultralytics-main-new/) | **当前唯一活跃代码库**；CitrusTopo-Seg、训练、评估和批量实验 | 正式使用 |
| 2 | [`data/orange_yolo_grouped_dedup_20260820/`](data/orange_yolo_grouped_dedup_20260820/) | **当前唯一正式数据集**；965 图、5,890 实例、严格 group-aware split | 正式使用 |
| 3 | [`3_研究生/柑橘套袋视觉_完整研究执行计划.md`](3_研究生/柑橘套袋视觉_完整研究执行计划.md) | 研究问题、实验纪律与论文路线 | 研究依据 |
| 4 | [`4_baseline_choice/`](4_baseline_choice/) | 跨家族基线：YOLO、RTMDet-Ins、Mask R-CNN、RF-DETR、U-Net 等 | 对比实验 |

本机根目录的 `ultralytics-main-new/` 是开发源；`1_SEVER/code/ultralytics-main-new/` 是整理后的服务器上传副本。
旧服务器代码已归档到 `1_SEVER/archive/code_pre20260824/`，只用于历史追溯。

## 当前方法与训练入口

| 文件 | 作用 |
|---|---|
| [`ultralytics-main-new/20260824_CITRUS_TOPO_REDESIGN_REPORT.md`](ultralytics-main-new/20260824_CITRUS_TOPO_REDESIGN_REPORT.md) | 本轮网络设计、证据、模块和实验矩阵 |
| [`ultralytics-main-new/20260824_CITRUS_TOPO_SERVER.md`](ultralytics-main-new/20260824_CITRUS_TOPO_SERVER.md) | 服务器环境与批量训练说明 |
| [`ultralytics-main-new/20260824_citrus_topo_batch.py`](ultralytics-main-new/20260824_citrus_topo_batch.py) | 10 个模型一键筛选、断点续跑与正式训练 |
| [`ultralytics-main-new/0_orange_yaml/20260824_citrus_topo/`](ultralytics-main-new/0_orange_yaml/20260824_citrus_topo/) | 10 个 CitrusTopo-Seg 网络 YAML |
| [`ultralytics-main-new/20260824_citrus_topo_report.py`](ultralytics-main-new/20260824_citrus_topo_report.py) | 批量汇总实验结果 |
| [`ultralytics-main-new/audit_citrus_difficulty.py`](ultralytics-main-new/audit_citrus_difficulty.py) | 超小果、凹遮挡、邻接/粘连和尺度跨度审计 |

服务器侧的最短流程：

```bash
cd ultralytics-main-new
conda env create -f environment_citrus_topo.yml
conda activate citrus-topo
pip install -e .
pytest -q tests/test_citrus_topo.py
python 20260824_citrus_topo_batch.py --help
```

具体参数以服务器说明为准。先执行构建测试和 1--3 epoch smoke，再批量筛选；不要直接启动 10 个 300 epoch 长跑。

## 数据目录

| 位置 | 定义 | 是否用于正式训练 |
|---|---|---|
| `data/orange_yolo_grouped_dedup_20260820/` | 严格分组去重版；当前标准 | **是** |
| `data/orange_wuxi/` | LabelMe 原始数据和原始素材 | 否；作为源数据保留 |
| `data/orange_yolo/` | 旧 YOLO 转换版，旧 split | 否；历史复核 |
| `data/_backups/` | 清洗前备份 | 否；待确认后精简 |
| `data/*.zip` | 传服务器/离线归档包 | 否；只在传输或灾备时使用 |

数据版本选择证据见 [`data/DATASET_SELECTION_REPORT_20260824.md`](data/DATASET_SELECTION_REPORT_20260824.md)。更换为正式数据集后，旧数据集上的所有模型不能与新结果直接比较；基线和候选方法必须在同一 split、初始化和训练协议下重跑。

## 历史与旁支

| 位置 | 说明 | 整理原则 |
|---|---|---|
| [`1_SEVER/results/`](1_SEVER/results/) | 服务器旧实验、权重、曲线和统计证据 | 不混入新表；精简前保留 `best/last`、CSV、参数和图 |
| [`1_SEVER/code/`](1_SEVER/code/) | 服务器上传目录，包含最新训练副本和跨框架辅助代码 | 上传使用 |
| `1_SEVER/archive/code_pre20260824/` | 旧代码快照与过去 10 组/混合网络 | 只读追溯，不上传 |
| [`3_研究生/`](3_研究生/) | 研究计划、文献和项目管理 | 保留 |
| [`2_catoon/`](2_catoon/) | 早期学习/课程代码，与论文主线无关 | 独立旁支，不擅自删除 |
| [`5_novels/`](5_novels/) | 个人写作，与论文主线无关 | 独立旁支，不擅自删除 |

## 实验纪律

1. 当前主消融基线是 YOLO11n-seg；同协议比较 mask mAP50-95、mask mAP50、P、R、AP by scale、参数量、GFLOPs、实测延迟和难例子集。
2. 正式数据 split、输入尺寸、预训练、优化器、学习率、batch、AMP、seed 和评估 split 必须完全一致。
3. 筛选实验可单次运行；最终基线和最终方法使用 3 个 seed，报告均值和标准差。
4. 已完成 run 永不覆盖；每次记录命令、Git 状态、数据版本和硬件。
5. 数据集、权重、结果图、压缩包和密钥不得继续提交到 Git。

## 目录整理记录

本次只读审计、重复项证据和待确认清理项见
[`3_研究生/00_项目管理/目录整理审计_20260824.md`](3_研究生/00_项目管理/目录整理审计_20260824.md)。
