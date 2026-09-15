# Citrus research

- 目标：两篇衔接论文——轻量高精度未成熟柑橘实例分割；基于果实实例/ROI 的果梗点精确定位。
- 研究依据：`3_研究生/柑橘套袋视觉_完整研究执行计划.md`。论文一限定 RGB 未成熟果实实例分割；除非另有任务，不加入 RGB-D、amodal、OBB、控制或多任务姿态头。
- 难点：条带状叶枝遮挡造成深凹可见掩膜；保持遮挡果实完整与分离接触果实的拓扑冲突；单图极端尺度跨度。用 solidity/凸包缺损、实例间隙、split/merge 错误、单图尺度比量化后再声称解决。

# Workspace

- 主代码：`ultralytics-main-new/`；模型 YAML：`0_orange_yaml/`；驱动：`train_citrus_seg.py`、`eval_citrus_seg.py`；结果：`1_results/`（后三者相对主代码目录）。**最新主工作副本是 `1_SEVER/code/ultralytics-main-new/`**（整体上传服务器跑实验，0_orange_yaml 按字母系列目录组织并登记 MODEL_INDEX.csv；根目录副本约停留在 2026-08 下旬）。课题相关：`1_SEVER/`（实验汇总+审计）、`4_baseline_choice/`（全基线工作台）、`_work/`（回传产物）、`3_研究生/`（计划与定稿）；`2_catoon/`、`5_novels/` 等为无关个人项目，不做关联重构。
- 主线现状：SAGE→E 系列（切片混合输入+控制论零初始化），最优 mask mAP50-95≈0.682（E30）vs 新协议锚点 G00 0.67031（`3_研究生/paper1_finalization_20260830/`）。历史 orange_yolo 划分有 123/303 组跨 split 泄漏（`1_SEVER/review_20260908/`），多数旧指标仅为验证集峰值、跨协议不可比；正式结论须在 grouped_dedup 上三 seed 复跑。
- 数据：仓库根 `data/`。`orange_wuxi/` 存原始标注 JSON；`orange_yolo/` 为早期划分，不用于正式实验；正式数据是 `orange_yolo_grouped_dedup_20260820/`（按组划分+去重：train/val/test = 676/193/96 张，共 965 图、5,890 实例，单类 `orange_immature`，划分审计见其 `audit/` 与 `group_split_manifest.csv`）。train/eval 脚本默认指向它，勿改回旧划分。

# Experiments

- 主消融基线：YOLO11n-seg。切换前先做 YOLO11n 与 RTMDet-Ins-tiny 的 50 epoch 筛选。
- 最少跨系列比较：YOLOv8n-seg、YOLO11n-seg、YOLO26n-seg、RTMDet-Ins-tiny、Mask R-CNN R50-FPN、RF-DETR Seg Nano。
- 期刊加强比较加入无框位置式 SOLOv2-Light R18-FPN，替代可选 CondInst/SparseInst 名额，不替代主消融基线。
- 辅助基线：U-Net + marker-controlled watershed。训练合并实例为前景；验证集调优距离变换分水岭拆分；报告语义 Dice/mIoU 及实例 Mask AP。U-Net 单独不是实例分割。
- 可选一个语义比较：DeepLabV3+ 或 SegFormer-B0 + 同样分水岭。使用成熟 segmentation_models_pytorch/MMSegmentation；`1.coding/2_Unet/` 已删除，不用于正式论文。
- `001`–`003` 为初步运行；仅在划分、初始化、优化器、学习率、dropout、图像尺寸、seed、评估集合完全一致时与新实验可比。正式实验前解决脚本协议冲突。
- 最终报告 mask mAP50-95/mAP50、precision/recall、尺度 AP、Params、GFLOPs、实测延迟、难例子集表现；语义模型另报 Dice、mIoU、Boundary F1。
- 筛选运行一次；主基线与最终方法各用三个 seed，报告均值±标准差。300 epoch 前做针对性检查及 1–3 epoch smoke run。
- 每次正式实验记录命令、Git 状态、划分版本、硬件、最终指标；编号命名，不覆盖已完成运行。

# Development

- Python 四空格、120 列，遵循 Ruff/isort/YAPF 和 Google docstrings；优先连贯的任务方法，避免堆叠已有模块。
- 新 YOLO 模块：实现于 `ultralytics/nn/modules/` → `__init__.py` 导出 → `ultralytics/nn/tasks.py` 导入 → `parse_model()` 注册通道/重复行为 → 最小 YAML 验证构建、前向、反向、FLOPs。
- 在主代码目录执行：`pip install -e .`；训练 `python train_citrus_seg.py --model yolo11n-seg.pt --name E0_baseline`（输出 `1_results/ORANGE_WUXI_SEG/<name>/`，imgsz 锁定 640）；评估 `python eval_citrus_seg.py --weights 1_results/ORANGE_WUXI_SEG/E0_baseline/weights/best.pt`，每个 split 追加一行到 `1_results/ORANGE_GROUPED_DEDUP/results_summary.csv`。运行名已存在时换新编号。
- 按改动选择 `pytest tests` 中的相关测试；上面是开发命令参考，不要求每次任务都安装、训练或运行完整测试。
- 不提交数据集、权重、runs、大结果图、压缩包或视频；不撤销无关改动。提交信息简短且限定范围，例如 `citrus: add cross-family baseline configs`。
