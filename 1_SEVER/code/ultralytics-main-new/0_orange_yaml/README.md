# 柑橘模型 YAML 总目录

本目录只保存模型结构。所有正式模型必须能够通过标准 Ultralytics 入口构建：

```python
from ultralytics import YOLO

model = YOLO("0_orange_yaml/Light_series/Light02_joint_core.yaml", task="segment")
model.load("yolo11n-seg.pt")
```

数据集路径不写进模型 YAML；训练时通过 `data=...` 或批量脚本的 `--data` 指定。

## 系列索引

| 目录 | YAML数 | 定位 | 建议 |
|---|---:|---|---|
| `A_baselines/` | 30 | 跨家族与早期基线；分 current/legacy | 新实验优先使用 current |
| `B_series/` | 10 | 清洗数据上的轻量/拓扑筛选 | 历史消融 |
| `C_series/` | 9 | 双原型与结构组合 | 历史消融 |
| `D_series/` | 9 | 形状、边缘和语义流 | 历史结构研究 |
| `F_series/` | 63 | 大规模单模块与组合筛选库 | 只用于复现或证据回查 |
| `G_series/` | 10 | 旧协议组合模型 | G10是历史候选，不可跨协议比较 |
| `G_0830_series/` | 5 | T结果驱动的主干/颈部重构 | 当前结构研究 |
| `G_0839_series/` | 6 | 双分辨率搜索—判别—精修 | 当前结构研究，计算较慢 |
| `H_series/` | 6 | AAFM/SAVSS/P2探索 | 历史探索 |
| `L_series/` | 10 | LSKA、尺度融合与拓扑 | 有历史正向信号 |
| `Light_series/` | 5 | 轻量非CSP主干＋自适应渐进颈部 | 当前轻量化主线 |
| `N_series/` | 10 | 旧协议证据组合 | 历史组合实验 |
| `S_series/` | 10 | Citrus Swift结构消融 | 已完成一轮统一数据实验 |
| `SXQ_series/` | 10 | 早期SXQNet全家桶 | 负结果/复现库 |
| `T_series/` | 10 | 历史代表模型统一复核 | 结果需结合完成轮数解读 |
| `_archive_metadata/` | 0 | 兼容性、重复关系和目录元数据 | 不参与训练 |

合计：203个模型 YAML。

## 命名和存放规则

1. 新系列必须建立 `<Series>_series/`，不得把模型 YAML 放在本目录根部。
2. YAML 文件名必须包含稳定模型编号，训练名称使用同一编号。
3. 已产生结果的 YAML 不改名、不移动，保证结果与结构可追溯。
4. `MODEL_INDEX.csv` 必须与真实文件一一对应。
5. 新模块必须完成实现、导出、tasks导入、`parse_model()`注册、构建、前向、反向和复杂度测试。

## 兼容性状态

2026-08-30 全量审计结果：

- 203/203 标准构建通过；
- 203/203 eval前向通过；
- 203/203 官方 `yolo11n-seg.pt` 加载后前向通过；
- `MODEL_INDEX.csv` 无缺项、无失效路径、无重复索引。

`A_baselines/current` 和 `A_baselines/legacy` 中有12组内容完全相同的文件。这些是为了保留旧实验路径而有意保留的兼容副本，不要删除或用于双重计数。

详细报告：`_archive_metadata/YAML_COMPATIBILITY_20260830.md`。

