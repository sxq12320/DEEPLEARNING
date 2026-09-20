# 柑橘实例分割代码入口

这是服务器上传版本的导航页。当前实际代码库是本目录，不是外层其他 Ultralytics 副本。

## 目录结构

```text
ultralytics-main-new/
├── 0_orange_yaml/       # 423 个模型 YAML，按系列存放（MODEL_INDEX.csv 登记）
├── 1_results/           # 本地审计、诊断和兼容性输出
├── docs/                # 各系列设计/重建文档与 I_V1 三审报告
├── protocols/           # 固定训练超参数，正式实验唯一来源
├── scripts/             # 各系列 YAML 生成器（拒绝覆盖已有 YAML）
├── tests/               # 模型契约与回归测试（test_citrus_*.py）
├── ultralytics/         # 修改后的框架源码
├── citrus_*_suite.py    # 各系列 NAMES/FACTORS/SUITES 定义
├── citrus_foreground.py # RUNNERS 注册表 + 前台串行训练
├── RUN_CITRUS_*.py      # 各系列 VS Code ▶ 前台入口
├── *_batch.py           # 各系列顺序批量训练脚本
├── train_citrus_yaml.py # 通用单模型训练入口
└── yolo11n-seg.pt       # YOLO11n-seg 初始化权重
```

数据集、训练结果和权重不要放进 `0_orange_yaml/`。模型 YAML 只描述网络结构，数据地址始终由训练命令的 `--data` 提供。

## 当前优先级

1. **`I_V1_series/`（当前）**：同步双原型掩膜解码。I00/I01 是 V12_03/04 精确重放锚点；
   I04 是主假设（逐位置门控）；其余臂隔离上下文块、差异反馈、初始化偏向、语义替换、无分类路由。
   设计与停止判据 `docs/I_V1_DESIGN_20260920.md`；前置审查 `docs/I_V1_REVIEW_20260920/`。
2. **`E_V12_series/`**：V12 六臂已完成 300ep 筛选；三审判定未达投稿成熟度
   （val 成员与 grouped_dedup 不一致、单 seed、缺同协议基线）。排序证据保留。
3. **`A_baselines/current/001_yolo11-seg.yaml`**：正式 YOLO11n-seg 主消融基线。
4. **E_V9–V11**：持久 P2 细节、识别路由的前身；重建文档 `docs/E_V9..V12_RECONSTRUCTION.md`。
5. **SAGE/E_V1–V8/ORCHID/Light/G/B/C/D/F/H/L/N/S/SXQ/T**：历史消融库，保留复现，不建议全量重跑。

所有模型路径、头部和状态均登记在 `0_orange_yaml/MODEL_INDEX.csv`。

## 标准单模型用法

```python
from ultralytics import YOLO

model = YOLO("0_orange_yaml/I_V1_series/I04_sync_msca.yaml", task="segment")
model.load("yolo11n-seg.pt")
model.train(data="/data/sxq/datasets/orange_yolo/data.yaml", epochs=300, imgsz=640)
```

## 批量入口（推荐）

```bash
python 20260920_citrus_i_v1_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite all \
  --dry-run
```

或直接改 `RUN_CITRUS_I_V1.py` 顶部配置后点 VS Code ▶（SUITE="priority" 先跑 I00/I01/I04/I05）。
完整系列清单见 `FOREGROUND_TRAINING_README.md` 与 `citrus_foreground.py`。
正式实验参数来自 `protocols/citrus_paper1_formal_v2_ram.yaml`，不得在不同模型之间静默改变
AMP、优化器、学习率、dropout、增强或图像尺寸。

## 正式数据提醒

正式数据集是 `orange_yolo_grouped_dedup_20260820`（676/193/96，5,890 实例）。
V11/V12 服务器运行的 val 成员（193 图/1,049 实例）与 grouped_dedup（1,181 实例）不一致：
其结果仅作内部排序证据；论文结论必须在 grouped_dedup 上以三 seed（42/43/44）复跑。
