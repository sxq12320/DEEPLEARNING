# 柑橘模型 YAML 总目录

本目录只保存模型结构。所有正式模型必须能够通过标准 Ultralytics 入口构建：

```python
from ultralytics import YOLO

model = YOLO("0_orange_yaml/I_V1_series/I04_sync_msca.yaml", task="segment")
model.load("yolo11n-seg.pt")
```

数据集路径不写进模型 YAML；训练时通过 `data=...` 或批量脚本的 `--data` 指定。

## 系列索引（2026-09-20）

| 目录 | YAML数 | 定位 | 建议 |
|---|---:|---|---|
| `I_V1_series/` | 10 | **当前主线**：同步双原型掩膜解码；I00/I01=V12_03/04 精确重放锚点 | 服务器筛选中 |
| `E_V12_series/` | 10 | 识别/定位任务分流；六臂完成 300ep，三审判定未达投稿成熟度 | 排序证据保留 |
| `E_V11_series/` | 12 | 持久 P2 细节通路、上下文聚合 | 前代方向 |
| `E_V10_series/` | 10 | 细节传输、对比茎、均衡原型 | 前代方向 |
| `E_V9_series/` | 10 | 掩膜解码细化、细节中继 | 前代方向 |
| `E_V2..V8_series/` | 96 | 切片/混合输入、细节与语义通路逐步分离（V4 含子目录 34 个） | 历史消融 |
| `E_series/` | 9 | E 系列初代 | 历史消融 |
| `SAGE_V5..V8_series/` | 20 | 控制论式零初始化修正演进 | 历史消融 |
| `SAGE_series/` | 43 | SAGE V2–V4 全家 | 历史消融 |
| `ORCHID_series/` | 7 | 候选区域条件化、检测/掩膜分流 | 历史消融 |
| `A_baselines/` | 30 | 跨家族与早期基线；分 current/legacy 子目录 | 新实验优先 current |
| `B_series/` | 10 | 清洗数据上的轻量/拓扑筛选 | 历史消融 |
| `C_series/` | 9 | 双原型与结构组合 | 历史消融 |
| `D_series/` | 9 | 形状、边缘和语义流 | 历史结构研究 |
| `F_series/` | 63 | 大规模单模块与组合筛选库 | 复现/证据回查 |
| `G_series/` | 10 | 旧协议组合模型 | 不可跨协议比较 |
| `G_0830_series/` | 5 | T 结果驱动的主干/颈部重构 | 历史结构研究 |
| `G_0839_series/` | 6 | 双分辨率搜索—判别—精修 | 历史结构研究 |
| `H_series/` | 6 | AAFM/SAVSS/P2 探索 | 历史探索 |
| `L_series/` | 10 | LSKA、尺度融合与拓扑 | 有历史正向信号 |
| `Light_series/` | 8 | 轻量非 CSP 主干＋自适应渐进颈部 | 已证伪主线 |
| `N_series/` | 10 | 旧协议证据组合 | 历史组合实验 |
| `S_series/` | 10 | Citrus Swift 结构消融 | 历史消融 |
| `SXQ_series/` | 10 | 早期 SXQNet 全家桶 | 负结果/复现库 |
| `T_series/` | 10 | 历史代表模型统一复核 | 结果需结合完成轮数解读 |
| `_archive_metadata/` | 0 | 兼容性、重复关系和目录元数据 | 不参与训练 |

合计：423 个模型 YAML（含 `A_baselines` 与 `E_V4` 子目录）。

## 命名和存放规则

1. 新系列必须建立 `<Series>_series/`，不得把模型 YAML 放在本目录根部。
2. YAML 文件名必须包含稳定模型编号，训练名称使用同一编号。
3. 已产生结果的 YAML 不改名、不移动，保证结果与结构可追溯。
4. `MODEL_INDEX.csv` 必须与真实文件一一对应。
5. 新模块必须完成实现、导出、tasks 导入、`parse_model()` 注册、构建、前向、反向和复杂度测试。
6. 新系列 YAML 由 `scripts/generate_citrus_*_yaml.py` 生成，生成器**拒绝覆盖**已有文件；
   配套 `citrus_*_suite.py` 记录 NAMES/FACTORS/SUITES。

## 兼容性状态

- 2026-08-30 全量审计：203/203 当时 YAML 构建、eval 前向、官方权重加载通过
  （`_archive_metadata/YAML_COMPATIBILITY_20260830.md`）。
- E V9–V12、I_V1 各有独立契约测试（`tests/test_citrus_e_v*.py`、`tests/test_citrus_i_v1.py`），
  覆盖构建、反传、fuse、官方权重映射、父模型逐参数重放和 checkpoint 往返。
- `A_baselines/current` 和 `A_baselines/legacy` 中有 12 组内容完全相同的文件：
  为保留旧实验路径的有意兼容副本，不要删除或用于双重计数。
