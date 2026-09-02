# 柑橘实例分割代码入口

这是服务器上传版本的导航页。当前实际代码库是本目录，不是外层其他 Ultralytics 副本。

## 目录结构

```text
ultralytics-main-new/
├── 0_orange_yaml/       # 234 个模型 YAML，按系列存放
├── 1_results/           # 本地审计、诊断和兼容性输出
├── figures/             # 网络结构图和复杂度/延迟表
├── protocols/           # 固定训练超参数，正式实验唯一来源
├── sources/             # 论文、代码仓库和设计证据记录
├── tests/               # 模型集成与回归测试
├── ultralytics/         # 修改后的框架源码
├── *_batch.py           # 各系列顺序批量训练入口
├── train_citrus_yaml.py # 通用单模型训练入口
└── yolo11n-seg.pt       # YOLO11n-seg 初始化权重
```

数据集、训练结果和权重不要放进 `0_orange_yaml/`。模型 YAML 只描述网络结构，数据地址始终由训练命令的 `--data` 提供。

## 当前优先级

1. `SAGE_series/`：当前 SAGE-v3 结构系列；SAGE20 单独改 P4/P5 形状主干，SAGE21--22 单独验证创新融合与拓扑监督，SAGE23 是联合核心，SAGE24--26 分别验证颜色统计、PR 排序和遮挡拓扑损失。SAGE00--17 保留用于复现。
2. `ORCHID_series/`：候选区域条件化、检测/掩膜分流，并包含单画布颈部对照。
3. `A_baselines/current/001_yolo11-seg.yaml`：正式 YOLO11n-seg 控制。
4. `Light_series/`：已完成的轻量主干/AFPN探索；根据当前观察不再作为首选路线。
5. `G_0830_series/`、`G_0839_series/`：结构重构研究，但速度较慢。
6. `T_series/`：跨历史系列的统一复核组，已有结果不能替代完整基线。
7. B/C/D/F/G/H/L/N/S/SXQ：历史实验或消融库；保留用于复现，不建议无选择地全部重跑。

所有模型路径、头部和状态均登记在 `0_orange_yaml/MODEL_INDEX.csv`。

SAGE 在长训练前必须运行 `benchmark_citrus_sage.py`。SAGE-v3 服务器命令见
`20260902_CITRUS_SAGE_V3_SERVER.md`，结构、证据与算子预算见 `20260902_CITRUS_SAGE_V3_DESIGN.md`。

## 标准单模型用法

```python
from ultralytics import YOLO

model = YOLO("0_orange_yaml/SAGE_series/SAGE23_joint_core_v3.yaml", task="segment")
model.load("yolo11n-seg.pt")
model.train(data="/data/sxq/datasets/orange_yolo/data.yaml", epochs=300, imgsz=640)
```

ORCHID 入口已经逐 YAML 完成构建、官方权重加载、真实前向、关键损失反传和 GFLOPs 验证。

## ORCHID 批量入口

```bash
python 20260901_citrus_orchid_batch.py \
  --data /data/sxq/datasets/orange_yolo/data.yaml \
  --suite all \
  --dry-run
```

完整命令见 `20260901_CITRUS_ORCHID_SERVER.md`。正式实验参数来自
`protocols/citrus_paper1_formal_v1.yaml`，不得在不同模型之间静默改变 AMP、优化器、学习率、dropout、增强或图像尺寸。

## YAML 兼容结论

- 索引：234/234，无缺项、无失效路径、无重复索引行。
- 历史 203 个 YAML 的兼容结论见旧报告；新增 ORCHID 7/7 已独立完成构建、真实前向和权重加载验证。
- Light 索引中 4 个后续 YAML 已补录，路径一致性测试通过。

详见 `0_orange_yaml/_archive_metadata/YAML_COMPATIBILITY_20260830.md`。
