# G_0830：Structure-Preserving Citrus 系列

本目录包含 5 个可直接由 Ultralytics 公共 YAML 入口构建的模型。它们不是注意力模块排列组合，而是一条逐级消融路线：官方控制 → 已完成 T04 结构锚点 → 双向高分辨率主干 → 频率对齐颈部 → 深层 C3k2 替换。

| YAML | 唯一结构变化 | Params | GFLOPs@640 |
|---|---|---:|---:|
| `00_g00_official_control.yaml` | 完整官方 YOLO11n-seg 控制 | 2.877M | 10.529 |
| `01_g01_t04_anchor.yaml` | residual LSKA + topology prototype head | 2.965M | 10.957 |
| `02_g02_bilateral_backbone.yaml` | 窄 P2 形状流贯穿 P3/P4/P5 | 3.003M | 11.596 |
| `03_g03_frequency_neck.yaml` | G02 + 四次频率对齐 PAN 融合 | 3.038M | 11.756 |
| `04_g04_deep_repmixer.yaml` | G03 + P4/P5 非 CSP RepMixer | 2.774M | 11.418 |

G02/G03 保持标准 YOLO11 语义层为 YAML 顶层节点，并用 `pretrained_layer_map` 完整迁移官方 561 个权重项。新增主干/颈部残差门均从 0 初始化，避免随机模块一开始破坏预训练特征。G04 为参数更少但预训练覆盖更低的风险消融，不能预设为赢家。

检测始终只在 P3/P4/P5；P2 只支持 query/boundary 辅助和 mask prototype，不增加密集 P2 检测头，也不声称 sparse-kernel 加速。

批量入口：

```bash
python 20260830_citrus_g0830_batch.py --data /your/dataset/data.yaml --suite smoke --epochs 3 --device 0
python 20260830_citrus_g0830_batch.py --data /your/dataset/data.yaml --suite structure --epochs 50 --device 0
python 20260830_citrus_g0830_batch.py --data /your/dataset/data.yaml --suite all --epochs 300 --device 0
```

`--suite loss` 固定 G03 架构，分别检查 NWD、VFL、二者组合和 T04 强辅助权重。它们是 loss 消融，不是新网络。

验证命令：

```bash
python -m pytest -q tests/test_citrus_g0830.py
```
